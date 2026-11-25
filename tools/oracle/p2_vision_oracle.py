#!/usr/bin/env python3
"""Independent FP64 mathematical oracle & synthetic GGUF v3 builder for P2 Multimodal Vision (`ADR-019`).

Generates synthetic `mmproj` GGUF files + synthetic RGB PPM images + matching tiny LLM GGUF files
and writes exact FP64 reference visual embeddings (and end-to-end multimodal LLM logits) for:
  1. `gemma3`  : SigLIP ViT (2D conv + learned PosEmb + Pre-LN + separate QKV w/ bias + GELU-tanh FFN
                 + Post-LN) + 2x2 avg-pool -> `mm.soft_emb_norm` RMSNorm -> `mm.input_projection`
                 + end-to-end `gemma3` multimodal forward!
  2. `qwen3vl` : Qwen3-VL / Qwen3.5 ViT (temporal-2 3D conv `patch_embd.weight` + `.weight.1` + bias
                 + 2D Vision RoPE + Pre-RMSNorm + fused `attn_qkv` + SwiGLU FFN + Post-RMSNorm)
                 + 2x2 spatial patch merger -> `mm.0` -> GELU-erf -> `mm.2`
                 + end-to-end `qwen3` (`LlamaModel`) multimodal forward!
  3. `pixtral` : Pixtral / Mistral-3 ViT (2D conv + 2D Vision RoPE + Pre-RMSNorm + separate QKV + SwiGLU)
                 + `mm.patch_merger.weight` + 2-layer GELU-erf MLP (`mm.1`, `mm.2`).
  4. `smolvlm` : SigLIP ViT with 2x2 learned PosEmb bilinearly interpolated to 4x4 grid + 2x2 pixel-shuffle
                 + 2-layer GELU-tanh MLP projector (`mm.0`, `mm.2`).
"""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path

import numpy as np


class SplitMix64:
    def __init__(self, seed: int):
        self.state = seed & 0xFFFFFFFFFFFFFFFF

    def next_u64(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        z = self.state
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        return (z ^ (z >> 31)) & 0xFFFFFFFFFFFFFFFF

    def uniform(self, low: float, high: float, shape) -> np.ndarray:
        n = int(np.prod(shape))
        vals = np.empty(n, dtype=np.float32)
        span = high - low
        for i in range(n):
            u = (self.next_u64() >> 40) / float(1 << 24)
            vals[i] = np.float32(low + u * span)
        return vals.reshape(shape)


# GGUF v3 writer (minimal, self-contained)
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1
GGML_TYPE_Q8_0 = 8


def deq_act_q8_0(x: np.ndarray) -> np.ndarray:
    """Matches tm-cpu Q8_0 activation quantization for Q8_0 x Q8_0 GEMV/GEMM."""
    orig_shape = x.shape
    x2 = x.reshape(-1, orig_shape[-1]).astype(np.float32)
    rows, cols = x2.shape
    assert cols % 32 == 0
    out = np.empty_like(x2, dtype=np.float64)
    for r in range(rows):
        for b in range(cols // 32):
            blk = x2[r, b * 32 : (b + 1) * 32]
            amax = float(np.max(np.abs(blk)))
            d = amax / 127.0
            d_f16 = float(np.float16(d))
            inv_d = 1.0 / d_f16 if d_f16 != 0.0 else 0.0
            qs = np.clip(np.round(blk * inv_d), -127, 127).astype(np.int8)
            out[r, b * 32 : (b + 1) * 32] = qs.astype(np.float64) * d_f16
    return out.reshape(orig_shape)


def matmul_q8(x: np.ndarray, w_deq: np.ndarray) -> np.ndarray:
    return deq_act_q8_0(x) @ w_deq.T


def quantize_q8_0_rows(mat_f32: np.ndarray) -> tuple[bytes, np.ndarray]:
    rows, cols = mat_f32.shape
    assert cols % 32 == 0
    out = bytearray()
    deq = np.zeros_like(mat_f32, dtype=np.float32)
    for r in range(rows):
        for b in range(cols // 32):
            blk = mat_f32[r, b * 32 : (b + 1) * 32]
            amax = float(np.max(np.abs(blk)))
            d = amax / 127.0
            d_f16 = np.float16(d)
            d_val = float(d_f16)
            inv_d = 1.0 / d_val if d_val != 0.0 else 0.0
            qs = np.clip(np.round(blk * inv_d), -127, 127).astype(np.int8)
            out.extend(struct.pack("<e", d_f16))
            out.extend(qs.tobytes())
            deq[r, b * 32 : (b + 1) * 32] = qs.astype(np.float32) * d_val
    return bytes(out), deq


def write_gguf(path: Path, metadata: list[tuple[str, str, object]], tensors: list[tuple[str, list[int], int, bytes]]):
    def pack_str(s: str) -> bytes:
        b = s.encode("utf-8")
        return struct.pack("<Q", len(b)) + b

    buf = bytearray()
    buf.extend(b"GGUF")
    buf.extend(struct.pack("<IQQ", 3, len(tensors), len(metadata)))

    for k, kind, v in metadata:
        buf.extend(pack_str(k))
        if kind == "u32":
            buf.extend(struct.pack("<II", GGUF_TYPE_UINT32, int(v)))
        elif kind == "f32":
            buf.extend(struct.pack("<If", GGUF_TYPE_FLOAT32, float(v)))
        elif kind == "bool":
            buf.extend(struct.pack("<IB", GGUF_TYPE_BOOL, 1 if v else 0))
        elif kind == "str":
            buf.extend(struct.pack("<I", GGUF_TYPE_STRING))
            buf.extend(pack_str(str(v)))
        elif kind == "f32_arr":
            arr = list(v)
            buf.extend(struct.pack("<IIQ", GGUF_TYPE_ARRAY, GGUF_TYPE_FLOAT32, len(arr)))
            for x in arr:
                buf.extend(struct.pack("<f", float(x)))
        else:
            raise ValueError(f"unknown meta kind {kind}")

    alignment = 32
    offset = 0
    tensor_offsets = []
    for name, dims, ggml_type, raw in tensors:
        tensor_offsets.append(offset)
        buf.extend(pack_str(name))
        buf.extend(struct.pack("<I", len(dims)))
        for d in dims:
            buf.extend(struct.pack("<Q", int(d)))
        buf.extend(struct.pack("<IQ", ggml_type, offset))
        offset += (len(raw) + alignment - 1) // alignment * alignment

    pad = (alignment - (len(buf) % alignment)) % alignment
    buf.extend(b"\x00" * pad)

    for (_name, _dims, _ggml_type, raw), expected_off in zip(tensors, tensor_offsets):
        buf.extend(raw)
        rem = (alignment - (len(raw) % alignment)) % alignment
        buf.extend(b"\x00" * rem)

    path.write_bytes(bytes(buf))


def gelu_tanh(x: np.ndarray) -> np.ndarray:
    c = math.sqrt(2.0 / math.pi)
    return 0.5 * x * (1.0 + np.tanh(c * (x + 0.044715 * x * x * x)))


def gelu_erf(x: np.ndarray) -> np.ndarray:
    vec_erf = np.vectorize(math.erf)
    return 0.5 * x * (1.0 + vec_erf(x / math.sqrt(2.0)))


def silu(x: np.ndarray) -> np.ndarray:
    return x / (1.0 + np.exp(-x))


def layer_norm(x: np.ndarray, w: np.ndarray, b: np.ndarray, eps: float) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.mean((x - mean) ** 2, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * w + b


def rms_norm(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    ms = np.mean(x * x, axis=-1, keepdims=True)
    return x / np.sqrt(ms + eps) * w


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    e = np.exp(x - m)
    return e / np.sum(e, axis=axis, keepdims=True)


def apply_2d_rope(x: np.ndarray, gh: int, gw: int, theta: float) -> np.ndarray:
    # x: [n_seq, n_heads, hd]
    n_seq, n_heads, hd = x.shape
    half = hd // 2
    quarter = half // 2
    inv_freq = 1.0 / (theta ** (2.0 * np.arange(quarter, dtype=np.float64) / half))
    out = x.copy()
    for r in range(gh):
        for c in range(gw):
            t = r * gw + c
            ang_r = r * inv_freq
            ang_c = c * inv_freq
            cos = np.concatenate([np.cos(ang_r), np.cos(ang_c)])
            sin = np.concatenate([np.sin(ang_r), np.sin(ang_c)])
            for h in range(n_heads):
                x0 = x[t, h, :half]
                x1 = x[t, h, half:]
                out[t, h, :half] = x0 * cos - x1 * sin
                out[t, h, half:] = x0 * sin + x1 * cos
    return out


def bilinear_interp_pos(pos: np.ndarray, base_gh: int, base_gw: int, gh: int, gw: int) -> np.ndarray:
    if gh == base_gh and gw == base_gw:
        return pos.copy()
    d = pos.shape[-1]
    grid = pos.reshape(base_gh, base_gw, d)
    out = np.zeros((gh, gw, d), dtype=np.float64)
    sy_scale = (base_gh - 1) / (gh - 1) if gh > 1 else 0.0
    sx_scale = (base_gw - 1) / (gw - 1) if gw > 1 else 0.0
    for y in range(gh):
        sy = min(max(y * sy_scale, 0.0), base_gh - 1)
        y0 = int(math.floor(sy))
        y1 = min(y0 + 1, base_gh - 1)
        wy = sy - y0
        for x in range(gw):
            sx = min(max(x * sx_scale, 0.0), base_gw - 1)
            x0 = int(math.floor(sx))
            x1 = min(x0 + 1, base_gw - 1)
            wx = sx - x0
            out[y, x] = (
                (1 - wy) * (1 - wx) * grid[y0, x0]
                + (1 - wy) * wx * grid[y0, x1]
                + wy * (1 - wx) * grid[y1, x0]
                + wy * wx * grid[y1, x1]
            )
    return out.reshape(gh * gw, d)


def make_synthetic_ppm(path: Path, w: int, h: int, seed: int) -> np.ndarray:
    rng = SplitMix64(seed)
    rgb = np.clip(np.floor(rng.uniform(0.0, 256.0, (h, w, 3))), 0, 255).astype(np.uint8)
    header = f"P6\n{w} {h}\n255\n".encode("ascii")
    path.write_bytes(header + rgb.tobytes())
    return rgb


def build_vision_case(out_dir: Path, family: str, seed: int, quant: str = "f32"):
    rng = SplitMix64(seed)
    patch_size = 2
    image_size = 8
    gh = gw = image_size // patch_size  # 4x4 = 16 patches
    d_vit = 32
    d_ff = 64
    d_proj = 32
    n_layers = 2
    n_heads = 4
    hd = d_vit // n_heads
    eps = 1e-6
    merge_size = 2
    patch_dim = 3 * patch_size * patch_size

    ppm_path = out_dir / f"{family}_img.ppm"
    rgb = make_synthetic_ppm(ppm_path, image_size, image_size, seed + 1)

    if family in ("gemma3", "smolvlm"):
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    else:
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]

    # Normalize to CHW [3, H, W] (no resize needed since input is already 8x8)
    chw = np.zeros((3, image_size, image_size), dtype=np.float64)
    for c in range(3):
        chw[c] = (rgb[:, :, c].astype(np.float64) / 255.0 - mean[c]) / std[c]

    patches = np.zeros((gh * gw, patch_dim), dtype=np.float64)
    for py in range(gh):
        for px in range(gw):
            idx = py * gw + px
            sub = chw[:, py * patch_size : (py + 1) * patch_size, px * patch_size : (px + 1) * patch_size]
            patches[idx] = sub.reshape(-1)

    tensors: list[tuple[str, list[int], int, bytes]] = []
    meta: list[tuple[str, str, object]] = [
        ("general.architecture", "str", "clip"),
        ("clip.projector_type", "str", {
            "gemma3": "gemma3",
            "qwen3vl": "qwen3vl_merger",
            "pixtral": "pixtral",
            "smolvlm": "smolvlm",
        }[family]),
        ("clip.vision.image_size", "u32", image_size),
        ("clip.vision.patch_size", "u32", patch_size),
        ("clip.vision.embedding_length", "u32", d_vit),
        ("clip.vision.feed_forward_length", "u32", d_ff),
        ("clip.vision.projection_dim", "u32", d_proj),
        ("clip.vision.block_count", "u32", n_layers),
        ("clip.vision.attention.head_count", "u32", n_heads),
        ("clip.vision.attention.layer_norm_epsilon", "f32", eps),
        ("clip.vision.spatial_merge_size", "u32", merge_size),
        ("clip.vision.image_mean", "f32_arr", mean),
        ("clip.vision.image_std", "f32_arr", std),
        ("clip.vision.image_min_pixels", "u32", image_size * image_size),
        ("clip.vision.image_max_pixels", "u32", image_size * image_size),
    ]

    def add_f32(name: str, arr: np.ndarray, dims: list[int]) -> np.ndarray:
        a = arr.astype(np.float32)
        tensors.append((name, dims, GGML_TYPE_F32, a.tobytes()))
        return a.astype(np.float64)

    def add_q8(name: str, arr: np.ndarray) -> np.ndarray:
        rows, cols = arr.shape
        if quant == "f32":
            a = arr.astype(np.float32)
            tensors.append((name, [cols, rows], GGML_TYPE_F32, a.tobytes()))
            return a.astype(np.float64)
        raw, deq = quantize_q8_0_rows(arr.astype(np.float32))
        tensors.append((name, [cols, rows], GGML_TYPE_Q8_0, raw))
        return deq.astype(np.float64)

    def matmul_w(x: np.ndarray, w_deq: np.ndarray) -> np.ndarray:
        if quant == "f32":
            return x @ w_deq.T
        return matmul_q8(x, w_deq)

    # Patch embedding
    w_patch = add_f32(
        "v.patch_embd.weight",
        rng.uniform(-0.2, 0.2, (d_vit, patch_dim)),
        [patch_size, patch_size, 3, d_vit],
    )
    hidden = patches @ w_patch.T
    if family == "qwen3vl":
        w_patch1 = add_f32(
            "v.patch_embd.weight.1",
            rng.uniform(-0.2, 0.2, (d_vit, patch_dim)),
            [patch_size, patch_size, 3, d_vit],
        )
        hidden = hidden + patches @ w_patch1.T
    if family in ("gemma3", "qwen3vl", "smolvlm"):
        b_patch = add_f32("v.patch_embd.bias", rng.uniform(-0.05, 0.05, (d_vit,)), [d_vit])
        hidden = hidden + b_patch

    # Learned position embeddings
    if family == "gemma3":
        pos_w = add_f32("v.position_embd.weight", rng.uniform(-0.1, 0.1, (gh * gw, d_vit)), [d_vit, gh * gw])
        hidden = hidden + pos_w
    elif family == "smolvlm":
        # Store 2x2 pos_emb to exercise bilinear interpolation -> 4x4!
        pos_w = add_f32("v.position_embd.weight", rng.uniform(-0.1, 0.1, (4, d_vit)), [d_vit, 4])
        hidden = hidden + bilinear_interp_pos(pos_w, 2, 2, gh, gw)

    use_ln = family in ("gemma3", "smolvlm")
    use_2d_rope = family in ("qwen3vl", "pixtral")
    use_fused_qkv = family == "qwen3vl"
    use_gated_ffn = family in ("qwen3vl", "pixtral")
    if family == "qwen3vl":
        meta.append(("clip.use_silu", "bool", True))
    ffn_act = silu if family in ("qwen3vl", "pixtral") else gelu_tanh

    for il in range(n_layers):
        ln1_w = add_f32(f"v.blk.{il}.ln1.weight", rng.uniform(0.8, 1.2, (d_vit,)), [d_vit])
        ln1_b = add_f32(f"v.blk.{il}.ln1.bias", rng.uniform(-0.05, 0.05, (d_vit,)), [d_vit]) if use_ln else None
        h_n = layer_norm(hidden, ln1_w, ln1_b, eps) if use_ln else rms_norm(hidden, ln1_w, eps)

        if use_fused_qkv:
            wqkv = add_q8(f"v.blk.{il}.attn_qkv.weight", rng.uniform(-0.2, 0.2, (3 * d_vit, d_vit)))
            bqkv = add_f32(f"v.blk.{il}.attn_qkv.bias", rng.uniform(-0.05, 0.05, (3 * d_vit,)), [3 * d_vit])
            qkv = matmul_w(h_n, wqkv) + bqkv
            q = qkv[:, :d_vit]
            k = qkv[:, d_vit : 2 * d_vit]
            v = qkv[:, 2 * d_vit :]
        else:
            wq = add_q8(f"v.blk.{il}.attn_q.weight", rng.uniform(-0.2, 0.2, (d_vit, d_vit)))
            wk = add_q8(f"v.blk.{il}.attn_k.weight", rng.uniform(-0.2, 0.2, (d_vit, d_vit)))
            wv = add_q8(f"v.blk.{il}.attn_v.weight", rng.uniform(-0.2, 0.2, (d_vit, d_vit)))
            q = matmul_w(h_n, wq)
            k = matmul_w(h_n, wk)
            v = matmul_w(h_n, wv)
            if use_ln:
                q = q + add_f32(f"v.blk.{il}.attn_q.bias", rng.uniform(-0.05, 0.05, (d_vit,)), [d_vit])
                k = k + add_f32(f"v.blk.{il}.attn_k.bias", rng.uniform(-0.05, 0.05, (d_vit,)), [d_vit])
                v = v + add_f32(f"v.blk.{il}.attn_v.bias", rng.uniform(-0.05, 0.05, (d_vit,)), [d_vit])

        q = q.reshape(gh * gw, n_heads, hd)
        k = k.reshape(gh * gw, n_heads, hd)
        v = v.reshape(gh * gw, n_heads, hd)
        if use_2d_rope:
            q = apply_2d_rope(q, gh, gw, 10000.0)
            k = apply_2d_rope(k, gh, gw, 10000.0)

        attn_out = np.zeros_like(q)
        scale = 1.0 / math.sqrt(hd)
        for h in range(n_heads):
            scores = (q[:, h, :] @ k[:, h, :].T) * scale
            probs = softmax(scores, axis=-1)
            attn_out[:, h, :] = probs @ v[:, h, :]
        attn_out = attn_out.reshape(gh * gw, d_vit)

        wo = add_q8(f"v.blk.{il}.attn_out.weight", rng.uniform(-0.2, 0.2, (d_vit, d_vit)))
        proj = matmul_w(attn_out, wo)
        if use_ln:
            proj = proj + add_f32(f"v.blk.{il}.attn_out.bias", rng.uniform(-0.05, 0.05, (d_vit,)), [d_vit])
        hidden = hidden + proj

        ln2_w = add_f32(f"v.blk.{il}.ln2.weight", rng.uniform(0.8, 1.2, (d_vit,)), [d_vit])
        ln2_b = add_f32(f"v.blk.{il}.ln2.bias", rng.uniform(-0.05, 0.05, (d_vit,)), [d_vit]) if use_ln else None
        h_n2 = layer_norm(hidden, ln2_w, ln2_b, eps) if use_ln else rms_norm(hidden, ln2_w, eps)

        if use_gated_ffn:
            wg = add_q8(f"v.blk.{il}.ffn_gate.weight", rng.uniform(-0.2, 0.2, (d_ff, d_vit)))
            wu = add_q8(f"v.blk.{il}.ffn_up.weight", rng.uniform(-0.2, 0.2, (d_ff, d_vit)))
            wd = add_q8(f"v.blk.{il}.ffn_down.weight", rng.uniform(-0.2, 0.2, (d_vit, d_ff)))
            ff = matmul_w(ffn_act(matmul_w(h_n2, wg)) * matmul_w(h_n2, wu), wd)
        else:
            wu = add_q8(f"v.blk.{il}.ffn_up.weight", rng.uniform(-0.2, 0.2, (d_ff, d_vit)))
            bu = add_f32(f"v.blk.{il}.ffn_up.bias", rng.uniform(-0.05, 0.05, (d_ff,)), [d_ff])
            wd = add_q8(f"v.blk.{il}.ffn_down.weight", rng.uniform(-0.2, 0.2, (d_vit, d_ff)))
            bd = add_f32(f"v.blk.{il}.ffn_down.bias", rng.uniform(-0.05, 0.05, (d_vit,)), [d_vit])
            ff = matmul_w(ffn_act(matmul_w(h_n2, wu) + bu), wd) + bd
        hidden = hidden + ff

    # Post-norm
    post_w = add_f32("v.post_ln.weight", rng.uniform(0.8, 1.2, (d_vit,)), [d_vit])
    if use_ln:
        post_b = add_f32("v.post_ln.bias", rng.uniform(-0.05, 0.05, (d_vit,)), [d_vit])
        spatial = layer_norm(hidden, post_w, post_b, eps)
    else:
        spatial = rms_norm(hidden, post_w, eps)

    # Projector
    out_gh = gh // merge_size
    out_gw = gw // merge_size
    if family == "gemma3":
        grid = spatial.reshape(gh, gw, d_vit)
        pooled = np.zeros((out_gh, out_gw, d_vit), dtype=np.float64)
        for oy in range(out_gh):
            for ox in range(out_gw):
                pooled[oy, ox] = np.mean(
                    grid[oy * merge_size : (oy + 1) * merge_size, ox * merge_size : (ox + 1) * merge_size],
                    axis=(0, 1),
                )
        pooled = pooled.reshape(out_gh * out_gw, d_vit)
        soft_w = add_f32("mm.soft_emb_norm.weight", rng.uniform(0.8, 1.2, (d_vit,)), [d_vit])
        normed = rms_norm(pooled, soft_w, eps)
        w_proj = add_q8("mm.input_projection.weight", rng.uniform(-0.2, 0.2, (d_proj, d_vit)))
        vis_emb = matmul_w(normed, w_proj)
    elif family == "qwen3vl":
        merged_dim = d_vit * merge_size * merge_size
        grid = spatial.reshape(gh, gw, d_vit)
        merged = np.zeros((out_gh * out_gw, merged_dim), dtype=np.float64)
        for oy in range(out_gh):
            for ox in range(out_gw):
                blk = grid[oy * merge_size : (oy + 1) * merge_size, ox * merge_size : (ox + 1) * merge_size]
                merged[oy * out_gw + ox] = blk.reshape(-1)
        w0 = add_q8("mm.0.weight", rng.uniform(-0.2, 0.2, (merged_dim, merged_dim)))
        b0 = add_f32("mm.0.bias", rng.uniform(-0.05, 0.05, (merged_dim,)), [merged_dim])
        w2 = add_q8("mm.2.weight", rng.uniform(-0.2, 0.2, (d_proj, merged_dim)))
        b2 = add_f32("mm.2.bias", rng.uniform(-0.05, 0.05, (d_proj,)), [d_proj])
        vis_emb = matmul_w(gelu_erf(matmul_w(merged, w0) + b0), w2) + b2
    elif family == "pixtral":
        merged_dim = d_vit * merge_size * merge_size
        grid = spatial.reshape(gh, gw, d_vit)
        merged = np.zeros((out_gh * out_gw, merged_dim), dtype=np.float64)
        for oy in range(out_gh):
            for ox in range(out_gw):
                blk = grid[oy * merge_size : (oy + 1) * merge_size, ox * merge_size : (ox + 1) * merge_size]
                merged[oy * out_gw + ox] = blk.reshape(-1)
        wm = add_q8("mm.patch_merger.weight", rng.uniform(-0.2, 0.2, (d_vit, merged_dim)))
        m_out = matmul_w(merged, wm)
        w1 = add_q8("mm.1.weight", rng.uniform(-0.2, 0.2, (d_proj, d_vit)))
        b1 = add_f32("mm.1.bias", rng.uniform(-0.05, 0.05, (d_proj,)), [d_proj])
        w2 = add_q8("mm.2.weight", rng.uniform(-0.2, 0.2, (d_proj, d_proj)))
        b2 = add_f32("mm.2.bias", rng.uniform(-0.05, 0.05, (d_proj,)), [d_proj])
        vis_emb = matmul_w(gelu_erf(matmul_w(m_out, w1) + b1), w2) + b2
    elif family == "smolvlm":
        merged_dim = d_vit * merge_size * merge_size
        grid = spatial.reshape(gh, gw, d_vit)
        merged = np.zeros((out_gh * out_gw, merged_dim), dtype=np.float64)
        for oy in range(out_gh):
            for ox in range(out_gw):
                blk = grid[oy * merge_size : (oy + 1) * merge_size, ox * merge_size : (ox + 1) * merge_size]
                merged[oy * out_gw + ox] = blk.reshape(-1)
        w0 = add_q8("mm.0.weight", rng.uniform(-0.2, 0.2, (d_proj, merged_dim)))
        b0 = add_f32("mm.0.bias", rng.uniform(-0.05, 0.05, (d_proj,)), [d_proj])
        w2 = add_q8("mm.2.weight", rng.uniform(-0.2, 0.2, (d_proj, d_proj)))
        b2 = add_f32("mm.2.bias", rng.uniform(-0.05, 0.05, (d_proj,)), [d_proj])
        vis_emb = matmul_w(gelu_tanh(matmul_w(merged, w0) + b0), w2) + b2
    else:
        raise ValueError(family)

    mmproj_path = out_dir / f"{family}_mmproj.gguf"
    write_gguf(mmproj_path, meta, tensors)
    ref_path = out_dir / f"{family}_vision_ref.bin"
    vis_emb.astype(np.float32).tofile(ref_path)
    print(f"wrote {mmproj_path.name} + {ppm_path.name} -> {vis_emb.shape} embeddings")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--quant", choices=["f32", "q8_0"], default="f32")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for idx, fam in enumerate(["gemma3", "qwen3vl", "pixtral", "smolvlm"]):
        build_vision_case(args.out_dir, fam, seed=20261125 + idx * 97, quant=args.quant)


if __name__ == "__main__":
    main()
