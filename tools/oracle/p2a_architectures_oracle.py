#!/usr/bin/env python3
"""Independent FP64 mathematical oracle & synthetic GGUF v3 builder for P2a architectures (`ADR-018`):
  1. `olmoe`            : NeoX RoPE, full-vector QK-Norm, 8-choose-2 Softmax MoE (`norm_w = False`),
                          testing BOTH separate 3D `ffn_{gate,up}_exps` (layer 0) and fused 3D
                          `ffn_gate_up_exps` (layer 1).
  2. `granitemoe`       : Norm RoPE, Granite 4x muP scales (`embedding_scale`, `attention.scale`,
                          `residual_scale`, `1/logit_scale`), Softmax MoE (`norm_w = True`),
                          and un-gated shared expert `ffn_{gate,up,down}_shexp`.
  3. `granitemoehybrid` : Mamba-2 SSM (layer 0) + Norm-RoPE GQA (layer 1) + Softmax MoE (`norm_w = True`)
                          + shared expert + Granite 4x muP scales (also supports `--arch granitehybrid`
                          with `expert_count > 0`!).
  4. `qwen2moe`         : NeoX RoPE, QKV biases, Softmax MoE (`norm_w = False`), and 1D
                          `blk.{l}.ffn_gate_inp_shexp.weight [n_embd]` sigmoid-gated shared expert.
  5. `qwen3moe`         : NeoX RoPE, per-head QK-Norm, Softmax MoE (`norm_w = True`).
  6. `bailingmoe2`      : Fused `attn_qkv.weight` + per-head QK-Norm + NeoX RoPE,
                          `leading_dense_block_count = 1` (layer 0 dense SwiGLU, layer 1 Sigmoid MoE
                          `expert_gating_func = 2` + `exp_probs_b.bias` + `expert_weights_norm = True`
                          + `expert_weights_norm_eps = 1e-6` + `expert_weights_scale = 2.5` + shared expert).
  7. `exaone`           : LGAI EXAONE 3.5 — NeoX RoPE, Pre-Norm, GQA, dense SwiGLU.
"""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path

import numpy as np

MOE_NORM_CLAMP_F16_MIN = 6.103515625e-5


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


def quantize_q4_0_rows(mat_f32: np.ndarray) -> tuple[bytes, np.ndarray]:
    rows, cols = mat_f32.shape
    assert cols % 32 == 0, f"Q4_0 requires cols % 32 == 0, got {cols}"
    out = bytearray()
    deq = np.zeros_like(mat_f32, dtype=np.float32)
    for r in range(rows):
        for b in range(cols // 32):
            blk = mat_f32[r, b * 32 : (b + 1) * 32]
            amax_idx = int(np.argmax(np.abs(blk)))
            max_val = float(blk[amax_idx])
            d = max_val / -8.0
            d_f16 = np.float16(d)
            d_eff = float(d_f16)
            inv_d = (1.0 / d) if d != 0.0 else 0.0
            qs = bytearray(16)
            for j in range(16):
                x0 = int(min(15, max(0, math.floor(float(blk[j]) * inv_d + 8.5))))
                x1 = int(min(15, max(0, math.floor(float(blk[j + 16]) * inv_d + 8.5))))
                qs[j] = (x0 & 0x0F) | ((x1 & 0x0F) << 4)
                deq[r, b * 32 + j] = np.float32((x0 - 8) * d_eff)
                deq[r, b * 32 + j + 16] = np.float32((x1 - 8) * d_eff)
            out.extend(struct.pack("<e", d_f16))
            out.extend(qs)
    return bytes(out), deq


def quantize_q8_0_f64(vec: np.ndarray) -> np.ndarray:
    x = vec.astype(np.float32)
    out = np.empty_like(x, dtype=np.float64)
    for b in range(x.size // 32):
        blk = x[b * 32 : (b + 1) * 32]
        amax = np.float32(np.max(np.abs(blk)))
        if amax == np.float32(0.0):
            out[b * 32 : (b + 1) * 32] = 0.0
            continue
        d = np.float32(amax / np.float32(127.0))
        inv_d = np.float32(np.float32(1.0) / d)
        d_eff = float(np.float16(d))
        scaled = (blk * inv_d).astype(np.float32)
        qs = np.trunc(
            scaled + np.where(scaled >= np.float32(0.0), np.float32(0.5), np.float32(-0.5))
        ).astype(np.int32)
        qs = np.clip(qs, -127, 127)
        out[b * 32 : (b + 1) * 32] = qs.astype(np.float64) * d_eff
    return out


def _gguf_str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def write_gguf_v3(
    path: Path,
    kv_items: list[tuple[str, int, object]],
    tensors: list[tuple[str, tuple[int, ...], int, bytes]],
) -> None:
    alignment = 32
    hdr = bytearray()
    hdr.extend(b"GGUF")
    hdr.extend(struct.pack("<IQQ", 3, len(tensors), len(kv_items)))

    for key, vtype, val in kv_items:
        hdr.extend(_gguf_str(key))
        hdr.extend(struct.pack("<I", vtype))
        if vtype == 4:  # U32
            hdr.extend(struct.pack("<I", int(val)))
        elif vtype == 5:  # I32
            hdr.extend(struct.pack("<i", int(val)))
        elif vtype == 6:  # F32
            hdr.extend(struct.pack("<f", float(val)))
        elif vtype == 7:  # BOOL
            hdr.extend(struct.pack("<B", 1 if val else 0))
        elif vtype == 8:  # STRING
            hdr.extend(_gguf_str(str(val)))
        elif vtype == 9:  # ARRAY
            elem_type, items = val
            hdr.extend(struct.pack("<IQ", elem_type, len(items)))
            for it in items:
                if elem_type == 4:
                    hdr.extend(struct.pack("<I", int(it)))
                elif elem_type == 6:
                    hdr.extend(struct.pack("<f", float(it)))
                else:
                    raise ValueError(f"unsupported array elem_type {elem_type}")
        else:
            raise ValueError(f"unsupported kv vtype {vtype}")

    offsets = []
    cur = 0
    for _, _, _, raw in tensors:
        cur = (cur + alignment - 1) // alignment * alignment
        offsets.append(cur)
        cur += len(raw)

    for (name, dims_ggml, ggml_type, _), off in zip(tensors, offsets):
        hdr.extend(_gguf_str(name))
        hdr.extend(struct.pack("<I", len(dims_ggml)))
        for d in dims_ggml:
            hdr.extend(struct.pack("<Q", int(d)))
        hdr.extend(struct.pack("<IQ", int(ggml_type), int(off)))

    while len(hdr) % alignment != 0:
        hdr.append(0)

    data_start = len(hdr)
    for (_, _, _, raw), off in zip(tensors, offsets):
        target = data_start + off
        if len(hdr) < target:
            hdr.extend(b"\x00" * (target - len(hdr)))
        hdr.extend(raw)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(hdr))


def write_tmlg(path: Path | None, tokens: list[int], vocab_size: int, all_logits: np.ndarray) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"TMLG")
        f.write(struct.pack("<III", 1, len(tokens), vocab_size))
        for tid in tokens:
            f.write(struct.pack("<I", tid))
        f.write(all_logits.astype(np.float32).tobytes())


def rms_norm_f64(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    ms = float(np.mean(x.astype(np.float64) ** 2))
    scale = 1.0 / math.sqrt(ms + float(eps))
    return (x.astype(np.float64) * scale) * w.astype(np.float64)


def silu_f64(x: np.ndarray) -> np.ndarray:
    xf = x.astype(np.float64)
    return xf / (1.0 + np.exp(-xf))


def softplus_f64(x: np.ndarray) -> np.ndarray:
    xf = x.astype(np.float64)
    return np.where(xf > 20.0, xf, np.log1p(np.exp(xf)))


def rope_apply_f64(
    vec: np.ndarray,
    n_heads: int,
    head_dim: int,
    rope_dim: int,
    pos: int,
    freq_base: float,
    is_neox: bool,
) -> np.ndarray:
    out = vec.astype(np.float64).copy()
    half = rope_dim // 2
    inv_freqs = np.array(
        [1.0 / (freq_base ** ((2 * i) / float(rope_dim))) for i in range(half)],
        dtype=np.float64,
    )
    angles = float(pos) * inv_freqs
    cos_t = np.cos(angles)
    sin_t = np.sin(angles)
    for h in range(n_heads):
        sub = out[h * head_dim : (h + 1) * head_dim]
        if is_neox:
            x0 = sub[:half].copy()
            x1 = sub[half : 2 * half].copy()
            sub[:half] = x0 * cos_t - x1 * sin_t
            sub[half : 2 * half] = x0 * sin_t + x1 * cos_t
        else:
            x0 = sub[0 : 2 * half : 2].copy()
            x1 = sub[1 : 2 * half : 2].copy()
            sub[0 : 2 * half : 2] = x0 * cos_t - x1 * sin_t
            sub[1 : 2 * half : 2] = x0 * sin_t + x1 * cos_t
    return out


def gqa_attention_step_f64(
    q: np.ndarray,
    k_cache_f16: list[np.ndarray],
    v_cache_f16: list[np.ndarray],
    n_head: int,
    n_head_kv: int,
    head_dim: int,
    attn_scale: float,
) -> np.ndarray:
    group = n_head // n_head_kv
    seq_len = len(k_cache_f16)
    out = np.zeros(n_head * head_dim, dtype=np.float64)
    for h in range(n_head):
        kv_h = h // group
        qh = q[h * head_dim : (h + 1) * head_dim]
        scores = np.empty(seq_len, dtype=np.float64)
        for t in range(seq_len):
            kt = k_cache_f16[t][kv_h * head_dim : (kv_h + 1) * head_dim].astype(np.float64)
            scores[t] = float(np.dot(qh, kt)) * attn_scale
        m = float(np.max(scores))
        probs = np.exp(scores - m)
        probs /= float(np.sum(probs))
        acc = np.zeros(head_dim, dtype=np.float64)
        for t in range(seq_len):
            vt = v_cache_f16[t][kv_h * head_dim : (kv_h + 1) * head_dim].astype(np.float64)
            acc += probs[t] * vt
        out[h * head_dim : (h + 1) * head_dim] = acc
    return out


def matvec_sim(mat_deq: np.ndarray, x: np.ndarray, use_q4: bool, act_q8: bool) -> np.ndarray:
    xv = quantize_q8_0_f64(x) if (use_q4 and act_q8) else x.astype(np.float64)
    return mat_deq.astype(np.float64) @ xv


def pack_2d_tensor(
    tensors: list[tuple[str, tuple[int, ...], int, bytes]],
    name: str,
    mat_f32: np.ndarray,
    use_q4: bool,
) -> np.ndarray:
    rows, cols = mat_f32.shape
    if use_q4 and cols % 32 == 0:
        raw, deq = quantize_q4_0_rows(mat_f32)
        tensors.append((name, (cols, rows), 2, raw))
        return deq.astype(np.float64)
    tensors.append((name, (cols, rows), 0, mat_f32.astype(np.float32).tobytes()))
    return mat_f32.astype(np.float64)


def pack_1d_linear_tensor(
    tensors: list[tuple[str, tuple[int, ...], int, bytes]],
    name: str,
    vec_f32: np.ndarray,
    use_q4: bool,
) -> np.ndarray:
    cols = vec_f32.size
    mat = vec_f32.reshape(1, cols)
    if use_q4 and cols % 32 == 0:
        raw, deq = quantize_q4_0_rows(mat)
        tensors.append((name, (cols,), 2, raw))
        return deq.astype(np.float64)
    tensors.append((name, (cols,), 0, vec_f32.astype(np.float32).tobytes()))
    return mat.astype(np.float64)


def pack_3d_experts(
    tensors: list[tuple[str, tuple[int, ...], int, bytes]],
    name: str,
    arr_f32: np.ndarray,  # [n_expert, rows, cols]
    use_q4: bool,
) -> np.ndarray:
    n_exp, rows, cols = arr_f32.shape
    flat = arr_f32.reshape(n_exp * rows, cols)
    if use_q4 and cols % 32 == 0:
        raw, deq = quantize_q4_0_rows(flat)
        tensors.append((name, (cols, rows, n_exp), 2, raw))
        return deq.reshape(n_exp, rows, cols).astype(np.float64)
    tensors.append((name, (cols, rows, n_exp), 0, flat.astype(np.float32).tobytes()))
    return arr_f32.astype(np.float64)


def pack_f32_1d(
    tensors: list[tuple[str, tuple[int, ...], int, bytes]], name: str, vec_f32: np.ndarray
) -> np.ndarray:
    tensors.append((name, (vec_f32.size,), 0, vec_f32.astype(np.float32).tobytes()))
    return vec_f32.astype(np.float64)


def eval_moe_ffn_f64(
    xn: np.ndarray,
    moe_dict: dict,
    use_q4: bool,
    act_q8: bool,
) -> np.ndarray:
    # Router logits (note: `gate_inp` may be F32 even when `use_q4` is True or Q4_0 when cols % 32 == 0)
    gate_is_q4 = moe_dict["gate_inp_q4"]
    logits = matvec_sim(moe_dict["gate_inp"], xn, gate_is_q4, act_q8)
    if moe_dict.get("gate_inp_bias") is not None:
        logits = logits + moe_dict["gate_inp_bias"]

    gating = moe_dict["gating_func"]  # 1 = Softmax, 2 = Sigmoid
    if gating == 1:
        m = float(np.max(logits))
        ex = np.exp(logits - m)
        probs = ex / float(np.sum(ex))
    elif gating == 2:
        probs = 1.0 / (1.0 + np.exp(-logits))
    else:
        raise ValueError(f"unsupported gating {gating}")

    sel = probs + moe_dict["exp_probs_b"] if moe_dict.get("exp_probs_b") is not None else probs.copy()
    n_exp = logits.size
    k = moe_dict["n_expert_used"]
    top_ids: list[int] = []
    top_w = np.empty(k, dtype=np.float64)
    for slot in range(k):
        best_e = 0
        best_v = -1e300
        for e in range(n_exp):
            if e in top_ids:
                continue
            if float(sel[e]) > best_v:
                best_v = float(sel[e])
                best_e = e
        top_ids.append(best_e)
        top_w[slot] = probs[best_e]

    if moe_dict["norm_weights"]:
        sum_w = float(np.sum(top_w))
        eps = float(moe_dict["norm_eps"])
        denom = (sum_w + eps) if eps > 0.0 else max(sum_w, MOE_NORM_CLAMP_F16_MIN)
        top_w = top_w / denom

    w_scale = float(moe_dict["weights_scale"])
    if w_scale != 0.0 and abs(w_scale - 1.0) > 1e-12:
        top_w = top_w * w_scale

    n_embd = xn.size
    out = np.zeros(n_embd, dtype=np.float64)
    for slot in range(k):
        eid = top_ids[slot]
        g = matvec_sim(moe_dict["gate_exps"][eid], xn, use_q4, act_q8)
        u = matvec_sim(moe_dict["up_exps"][eid], xn, use_q4, act_q8)
        act = silu_f64(g) * u
        y = matvec_sim(moe_dict["down_exps"][eid], act, use_q4, act_q8)
        out += top_w[slot] * y

    sh = moe_dict.get("shexp")
    if sh is not None:
        sg = matvec_sim(sh["w_gate"], xn, use_q4, act_q8)
        su = matvec_sim(sh["w_up"], xn, use_q4, act_q8)
        sact = silu_f64(sg) * su
        sy = matvec_sim(sh["w_down"], sact, use_q4, act_q8)
        if sh.get("w_gate_inp") is not None:
            gl = float(matvec_sim(sh["w_gate_inp"], xn, use_q4, act_q8)[0])
            scale_sh = 1.0 / (1.0 + math.exp(-gl))
            out += scale_sh * sy
        else:
            out += sy

    return out


def build_moe_layer_weights(
    rng: SplitMix64,
    tensors: list[tuple[str, tuple[int, ...], int, bytes]],
    ltag: str,
    n_embd: int,
    n_exp: int,
    k: int,
    n_ff_exp: int,
    use_q4: bool,
    *,
    fused_gate_up: bool = False,
    gating_func: int = 1,
    norm_weights: bool = False,
    norm_eps: float = 0.0,
    weights_scale: float = 0.0,
    with_exp_probs_b: bool = False,
    with_shexp: bool = False,
    with_shexp_gate: bool = False,
) -> dict:
    # Upstream `llama-quantize` always keeps `ffn_gate_inp.weight` in `F32`, and trained routers
    # specialize experts into distinct subspaces so `Q8_0` (CPU) vs `F32` (GPU) activations agree on top-k.
    gi_f32 = rng.uniform(-0.35, 0.35, (n_exp, n_embd))
    seg = max(1, n_embd // n_exp)
    for e in range(n_exp):
        gi_f32[e, e * seg : (e + 1) * seg] *= np.float32(3.5)
    gi_deq = pack_2d_tensor(tensors, f"{ltag}.ffn_gate_inp.weight", gi_f32, False)

    exp_probs_b = None
    if with_exp_probs_b:
        epb = np.linspace(0.28, -0.28, n_exp, dtype=np.float32)
        exp_probs_b = pack_f32_1d(tensors, f"{ltag}.exp_probs_b.bias", epb)

    gate_arr = rng.uniform(-0.18, 0.18, (n_exp, n_ff_exp, n_embd))
    up_arr = rng.uniform(-0.18, 0.18, (n_exp, n_ff_exp, n_embd))
    down_arr = rng.uniform(-0.18, 0.18, (n_exp, n_embd, n_ff_exp))

    if fused_gate_up:
        gu_arr = np.concatenate([gate_arr, up_arr], axis=1)
        gu_deq = pack_3d_experts(tensors, f"{ltag}.ffn_gate_up_exps.weight", gu_arr, use_q4)
        gate_deq = gu_deq[:, :n_ff_exp, :]
        up_deq = gu_deq[:, n_ff_exp:, :]
    else:
        gate_deq = pack_3d_experts(tensors, f"{ltag}.ffn_gate_exps.weight", gate_arr, use_q4)
        up_deq = pack_3d_experts(tensors, f"{ltag}.ffn_up_exps.weight", up_arr, use_q4)

    down_deq = pack_3d_experts(tensors, f"{ltag}.ffn_down_exps.weight", down_arr, use_q4)

    shexp = None
    if with_shexp:
        ff_sh = n_ff_exp
        sh_g = pack_2d_tensor(
            tensors, f"{ltag}.ffn_gate_shexp.weight", rng.uniform(-0.16, 0.16, (ff_sh, n_embd)), use_q4
        )
        sh_u = pack_2d_tensor(
            tensors, f"{ltag}.ffn_up_shexp.weight", rng.uniform(-0.16, 0.16, (ff_sh, n_embd)), use_q4
        )
        sh_d = pack_2d_tensor(
            tensors, f"{ltag}.ffn_down_shexp.weight", rng.uniform(-0.16, 0.16, (n_embd, ff_sh)), use_q4
        )
        sh_gi = None
        if with_shexp_gate:
            sh_gi = pack_1d_linear_tensor(
                tensors,
                f"{ltag}.ffn_gate_inp_shexp.weight",
                rng.uniform(-0.25, 0.25, (n_embd,)),
                use_q4,
            )
        shexp = {"w_gate": sh_g, "w_up": sh_u, "w_down": sh_d, "w_gate_inp": sh_gi}

    return {
        "gate_inp": gi_deq,
        "gate_inp_q4": False,
        "exp_probs_b": exp_probs_b,
        "gate_exps": gate_deq,
        "up_exps": up_deq,
        "down_exps": down_deq,
        "n_expert_used": k,
        "gating_func": gating_func,
        "norm_weights": norm_weights,
        "norm_eps": norm_eps,
        "weights_scale": weights_scale,
        "shexp": shexp,
    }


def build_and_eval_llama_moe_or_exaone(
    arch: str,
    gguf_path: Path,
    tmlg_path: Path | None,
    tokens: list[int],
    use_q4: bool,
    act_q8: bool,
) -> None:
    seed_map = {
        "olmoe": 20262001,
        "granitemoe": 20262002,
        "qwen2moe": 20262003,
        "qwen3moe": 20262004,
        "bailingmoe2": 20262005,
        "exaone": 20262006,
    }
    rng = SplitMix64(seed_map[arch])

    n_layers = 2
    n_embd = 64
    n_head = 2
    n_head_kv = 1
    head_dim = 32
    q_dim = n_head * head_dim  # 64
    kv_dim = n_head_kv * head_dim  # 32
    vocab_size = 96
    eps = 1e-5

    is_neox = arch != "granitemoe"
    freq_base = (
        1_000_000.0
        if arch in ("qwen2moe", "qwen3moe")
        else (500_000.0 if arch in ("olmoe", "exaone") else 10_000.0)
    )

    emb_scale = 8.0 if arch == "granitemoe" else 1.0
    attn_scale = 0.015625 if arch == "granitemoe" else (1.0 / math.sqrt(head_dim))
    res_scale = 0.22 if arch == "granitemoe" else 1.0
    logit_scale = 4.0 if arch == "granitemoe" else 1.0

    kv: list[tuple[str, int, object]] = [
        ("general.architecture", 8, arch),
        (f"{arch}.block_count", 4, n_layers),
        (f"{arch}.embedding_length", 4, n_embd),
        (f"{arch}.attention.head_count", 4, n_head),
        (f"{arch}.attention.head_count_kv", 4, n_head_kv),
        (f"{arch}.attention.key_length", 4, head_dim),
        (f"{arch}.attention.value_length", 4, head_dim),
        (f"{arch}.context_length", 4, 128),
        (f"{arch}.rope.dimension_count", 4, head_dim),
        (f"{arch}.rope.freq_base", 6, freq_base),
        (f"{arch}.attention.layer_norm_rms_epsilon", 6, eps),
        (f"{arch}.vocab_size", 4, vocab_size),
    ]

    if arch == "exaone":
        n_ff = 128
        kv.append((f"{arch}.feed_forward_length", 4, n_ff))
    else:
        n_exp = 8
        k_exp = 2
        n_ff_exp = 64
        n_ff_dense = 128 if arch == "bailingmoe2" else n_ff_exp
        kv.extend(
            [
                (f"{arch}.feed_forward_length", 4, n_ff_dense),
                (f"{arch}.expert_count", 4, n_exp),
                (f"{arch}.expert_used_count", 4, k_exp),
                (f"{arch}.expert_feed_forward_length", 4, n_ff_exp),
            ]
        )
        if arch == "granitemoe":
            kv.extend(
                [
                    (f"{arch}.embedding_scale", 6, emb_scale),
                    (f"{arch}.attention.scale", 6, attn_scale),
                    (f"{arch}.residual_scale", 6, res_scale),
                    (f"{arch}.logit_scale", 6, logit_scale),
                    (f"{arch}.shared_expert_feed_forward_length", 4, n_ff_exp),
                ]
            )
        elif arch == "bailingmoe2":
            kv.extend(
                [
                    (f"{arch}.leading_dense_block_count", 4, 1),
                    (f"{arch}.expert_gating_func", 4, 2),  # Sigmoid
                    (f"{arch}.expert_weights_norm", 7, True),
                    (f"{arch}.expert_weights_norm_eps", 6, 1e-6),
                    (f"{arch}.expert_weights_scale", 6, 2.5),
                ]
            )

    tensors: list[tuple[str, tuple[int, ...], int, bytes]] = []
    emb_deq = pack_2d_tensor(
        tensors, "token_embd.weight", rng.uniform(-0.15, 0.15, (vocab_size, n_embd)), use_q4
    )
    out_norm = pack_f32_1d(tensors, "output_norm.weight", rng.uniform(0.75, 1.25, (n_embd,)))
    out_deq = pack_2d_tensor(
        tensors, "output.weight", rng.uniform(-0.15, 0.15, (vocab_size, n_embd)), use_q4
    )

    layers = []
    for l in range(n_layers):
        ltag = f"blk.{l}"
        ld: dict = {}
        ld["attn_norm"] = pack_f32_1d(
            tensors, f"{ltag}.attn_norm.weight", rng.uniform(0.75, 1.25, (n_embd,))
        )
        if arch == "bailingmoe2":
            qkv_mat = rng.uniform(-0.18, 0.18, (q_dim + 2 * kv_dim, n_embd))
            qkv_deq = pack_2d_tensor(tensors, f"{ltag}.attn_qkv.weight", qkv_mat, use_q4)
            ld["wq"] = qkv_deq[:q_dim, :]
            ld["wk"] = qkv_deq[q_dim : q_dim + kv_dim, :]
            ld["wv"] = qkv_deq[q_dim + kv_dim :, :]
        else:
            ld["wq"] = pack_2d_tensor(
                tensors, f"{ltag}.attn_q.weight", rng.uniform(-0.18, 0.18, (q_dim, n_embd)), use_q4
            )
            ld["wk"] = pack_2d_tensor(
                tensors, f"{ltag}.attn_k.weight", rng.uniform(-0.18, 0.18, (kv_dim, n_embd)), use_q4
            )
            ld["wv"] = pack_2d_tensor(
                tensors, f"{ltag}.attn_v.weight", rng.uniform(-0.18, 0.18, (kv_dim, n_embd)), use_q4
            )
        ld["wo"] = pack_2d_tensor(
            tensors, f"{ltag}.attn_output.weight", rng.uniform(-0.18, 0.18, (n_embd, q_dim)), use_q4
        )

        if arch == "qwen2moe":
            ld["bq"] = pack_f32_1d(tensors, f"{ltag}.attn_q.bias", rng.uniform(-0.05, 0.05, (q_dim,)))
            ld["bk"] = pack_f32_1d(tensors, f"{ltag}.attn_k.bias", rng.uniform(-0.05, 0.05, (kv_dim,)))
            ld["bv"] = pack_f32_1d(tensors, f"{ltag}.attn_v.bias", rng.uniform(-0.05, 0.05, (kv_dim,)))

        if arch == "olmoe":
            ld["q_norm"] = pack_f32_1d(
                tensors, f"{ltag}.attn_q_norm.weight", rng.uniform(0.75, 1.25, (q_dim,))
            )
            ld["k_norm"] = pack_f32_1d(
                tensors, f"{ltag}.attn_k_norm.weight", rng.uniform(0.75, 1.25, (kv_dim,))
            )
            ld["qk_full"] = True
        elif arch in ("qwen3moe", "bailingmoe2"):
            ld["q_norm"] = pack_f32_1d(
                tensors, f"{ltag}.attn_q_norm.weight", rng.uniform(0.75, 1.25, (head_dim,))
            )
            ld["k_norm"] = pack_f32_1d(
                tensors, f"{ltag}.attn_k_norm.weight", rng.uniform(0.75, 1.25, (head_dim,))
            )
            ld["qk_full"] = False

        ld["ffn_norm"] = pack_f32_1d(
            tensors, f"{ltag}.ffn_norm.weight", rng.uniform(0.75, 1.25, (n_embd,))
        )

        is_dense_layer = arch == "exaone" or (arch == "bailingmoe2" and l == 0)
        if is_dense_layer:
            ff = 128
            ld["w_gate"] = pack_2d_tensor(
                tensors, f"{ltag}.ffn_gate.weight", rng.uniform(-0.18, 0.18, (ff, n_embd)), use_q4
            )
            ld["w_up"] = pack_2d_tensor(
                tensors, f"{ltag}.ffn_up.weight", rng.uniform(-0.18, 0.18, (ff, n_embd)), use_q4
            )
            ld["w_down"] = pack_2d_tensor(
                tensors, f"{ltag}.ffn_down.weight", rng.uniform(-0.18, 0.18, (n_embd, ff)), use_q4
            )
        else:
            ld["moe"] = build_moe_layer_weights(
                rng,
                tensors,
                ltag,
                n_embd,
                8,
                2,
                64,
                use_q4,
                fused_gate_up=(arch == "olmoe" and l == 1),
                gating_func=(2 if arch == "bailingmoe2" else 1),
                norm_weights=(arch in ("granitemoe", "qwen3moe", "bailingmoe2")),
                norm_eps=(1e-6 if arch == "bailingmoe2" else 0.0),
                weights_scale=(2.5 if arch == "bailingmoe2" else 0.0),
                with_exp_probs_b=(arch == "bailingmoe2"),
                with_shexp=(arch in ("granitemoe", "qwen2moe", "bailingmoe2")),
                with_shexp_gate=(arch == "qwen2moe"),
            )
        layers.append(ld)

    write_gguf_v3(gguf_path, kv, tensors)
    if tmlg_path is None:
        return

    k_caches: list[list[np.ndarray]] = [[] for _ in range(n_layers)]
    v_caches: list[list[np.ndarray]] = [[] for _ in range(n_layers)]
    all_logits = np.zeros((len(tokens), vocab_size), dtype=np.float32)

    for step, tok in enumerate(tokens):
        x = emb_deq[tok].copy() * emb_scale
        for l, ld in enumerate(layers):
            xn = rms_norm_f64(x, ld["attn_norm"], eps)
            q = matvec_sim(ld["wq"], xn, use_q4, act_q8)
            k = matvec_sim(ld["wk"], xn, use_q4, act_q8)
            v = matvec_sim(ld["wv"], xn, use_q4, act_q8)
            if "bq" in ld:
                q += ld["bq"]
                k += ld["bk"]
                v += ld["bv"]
            if "q_norm" in ld:
                if ld["qk_full"]:
                    q = rms_norm_f64(q, ld["q_norm"], eps)
                    k = rms_norm_f64(k, ld["k_norm"], eps)
                else:
                    for h in range(n_head):
                        q[h * head_dim : (h + 1) * head_dim] = rms_norm_f64(
                            q[h * head_dim : (h + 1) * head_dim], ld["q_norm"], eps
                        )
                    for h in range(n_head_kv):
                        k[h * head_dim : (h + 1) * head_dim] = rms_norm_f64(
                            k[h * head_dim : (h + 1) * head_dim], ld["k_norm"], eps
                        )
            q = rope_apply_f64(q, n_head, head_dim, head_dim, step, freq_base, is_neox)
            k = rope_apply_f64(k, n_head_kv, head_dim, head_dim, step, freq_base, is_neox)
            k_caches[l].append(k.astype(np.float16))
            v_caches[l].append(v.astype(np.float16))

            ao = gqa_attention_step_f64(
                q, k_caches[l], v_caches[l], n_head, n_head_kv, head_dim, attn_scale
            )
            ap = matvec_sim(ld["wo"], ao, use_q4, act_q8)
            x = x + res_scale * ap

            xfn = rms_norm_f64(x, ld["ffn_norm"], eps)
            if "moe" in ld:
                fp = eval_moe_ffn_f64(xfn, ld["moe"], use_q4, act_q8)
            else:
                g = matvec_sim(ld["w_gate"], xfn, use_q4, act_q8)
                u = matvec_sim(ld["w_up"], xfn, use_q4, act_q8)
                fp = matvec_sim(ld["w_down"], silu_f64(g) * u, use_q4, act_q8)
            x = x + res_scale * fp

        xn_final = rms_norm_f64(x, out_norm, eps)
        logits = matvec_sim(out_deq, xn_final, use_q4, act_q8) / logit_scale
        all_logits[step] = logits.astype(np.float32)

    write_tmlg(tmlg_path, tokens, vocab_size, all_logits)


def build_and_eval_granite_moe_hybrid(
    arch: str,
    gguf_path: Path,
    tmlg_path: Path | None,
    tokens: list[int],
    use_q4: bool,
    act_q8: bool,
) -> None:
    rng = SplitMix64(20262007 if arch == "granitemoehybrid" else 20262008)
    n_layers = 2
    n_embd = 64
    vocab_size = 96
    eps = 1e-5

    d_conv = 4
    d_inner = 64
    d_state = 64
    n_heads_ssm = 4
    n_groups = 2
    ssm_head_dim = d_inner // n_heads_ssm  # 16
    group_ratio = n_heads_ssm // n_groups  # 2
    bc_dim = n_groups * d_state  # 128
    conv_dim = d_inner + 2 * bc_dim  # 320
    d_in_proj = 2 * d_inner + 2 * bc_dim + n_heads_ssm  # 388

    n_head_attn = 2
    n_kv_attn = 1
    head_dim = 32
    q_dim = n_head_attn * head_dim  # 64
    kv_dim = n_kv_attn * head_dim  # 32

    emb_scale = 6.0
    attn_scale = 0.02
    res_scale = 0.25
    logit_scale = 4.0

    # Layer 0 = Mamba-2 SSM (`n_head_kv = 0`) + SparseMoeFfn; Layer 1 = Norm-RoPE GQA + SparseMoeFfn
    nh_arr = [0, n_head_attn]
    nkv_arr = [0, n_kv_attn]
    nff_arr = [64, 64]

    kv: list[tuple[str, int, object]] = [
        ("general.architecture", 8, arch),
        (f"{arch}.block_count", 4, n_layers),
        (f"{arch}.embedding_length", 4, n_embd),
        (f"{arch}.context_length", 4, 128),
        (f"{arch}.vocab_size", 4, vocab_size),
        (f"{arch}.attention.layer_norm_rms_epsilon", 6, eps),
        (f"{arch}.ssm.conv_kernel", 4, d_conv),
        (f"{arch}.ssm.inner_size", 4, d_inner),
        (f"{arch}.ssm.state_size", 4, d_state),
        (f"{arch}.ssm.time_step_rank", 4, n_heads_ssm),
        (f"{arch}.ssm.group_count", 4, n_groups),
        (f"{arch}.attention.head_count", 9, (4, nh_arr)),
        (f"{arch}.attention.head_count_kv", 9, (4, nkv_arr)),
        (f"{arch}.feed_forward_length", 9, (4, nff_arr)),
        (f"{arch}.expert_count", 4, 8),
        (f"{arch}.expert_used_count", 4, 2),
        (f"{arch}.attention.key_length", 4, head_dim),
        (f"{arch}.rope.dimension_count", 4, head_dim),
        (f"{arch}.rope.freq_base", 6, 10_000.0),
        (f"{arch}.embedding_scale", 6, emb_scale),
        (f"{arch}.attention.scale", 6, attn_scale),
        (f"{arch}.residual_scale", 6, res_scale),
        (f"{arch}.logit_scale", 6, logit_scale),
    ]

    tensors: list[tuple[str, tuple[int, ...], int, bytes]] = []
    emb_deq = pack_2d_tensor(
        tensors, "token_embd.weight", rng.uniform(-0.15, 0.15, (vocab_size, n_embd)), use_q4
    )
    out_norm = pack_f32_1d(tensors, "output_norm.weight", rng.uniform(0.75, 1.25, (n_embd,)))
    out_deq = pack_2d_tensor(
        tensors, "output.weight", rng.uniform(-0.15, 0.15, (vocab_size, n_embd)), use_q4
    )

    layers = []
    for l in range(n_layers):
        ltag = f"blk.{l}"
        ld: dict = {
            "attn_norm": pack_f32_1d(
                tensors, f"{ltag}.attn_norm.weight", rng.uniform(0.75, 1.25, (n_embd,))
            )
        }
        if l == 0:
            ssm: dict = {}
            ssm["ssm_in"] = pack_2d_tensor(
                tensors, f"{ltag}.ssm_in.weight", rng.uniform(-0.15, 0.15, (d_in_proj, n_embd)), use_q4
            )
            cw = rng.uniform(-0.2, 0.2, (conv_dim, d_conv))
            tensors.append((f"{ltag}.ssm_conv1d.weight", (d_conv, conv_dim), 0, cw.astype(np.float32).tobytes()))
            ssm["conv_w"] = cw.astype(np.float64)
            ssm["conv_b"] = pack_f32_1d(
                tensors, f"{ltag}.ssm_conv1d.bias", rng.uniform(-0.05, 0.05, (conv_dim,))
            )
            ssm["dt_bias"] = pack_f32_1d(
                tensors, f"{ltag}.ssm_dt.bias", rng.uniform(-1.2, -0.2, (n_heads_ssm,))
            )
            ssm["ssm_a"] = pack_f32_1d(
                tensors, f"{ltag}.ssm_a", rng.uniform(-1.5, -0.25, (n_heads_ssm,))
            )
            ssm["ssm_d"] = pack_f32_1d(tensors, f"{ltag}.ssm_d", rng.uniform(0.1, 0.6, (n_heads_ssm,)))
            ssm["ssm_norm"] = pack_f32_1d(
                tensors, f"{ltag}.ssm_norm.weight", rng.uniform(0.75, 1.25, (d_inner,))
            )
            ssm["ssm_out"] = pack_2d_tensor(
                tensors, f"{ltag}.ssm_out.weight", rng.uniform(-0.15, 0.15, (n_embd, d_inner)), use_q4
            )
            ssm["state"] = np.zeros((d_inner, d_state), dtype=np.float64)
            ssm["conv_hist"] = np.zeros((conv_dim, d_conv - 1), dtype=np.float64)
            ld["ssm"] = ssm
        else:
            attn: dict = {
                "wq": pack_2d_tensor(
                    tensors, f"{ltag}.attn_q.weight", rng.uniform(-0.18, 0.18, (q_dim, n_embd)), use_q4
                ),
                "wk": pack_2d_tensor(
                    tensors, f"{ltag}.attn_k.weight", rng.uniform(-0.18, 0.18, (kv_dim, n_embd)), use_q4
                ),
                "wv": pack_2d_tensor(
                    tensors, f"{ltag}.attn_v.weight", rng.uniform(-0.18, 0.18, (kv_dim, n_embd)), use_q4
                ),
                "wo": pack_2d_tensor(
                    tensors, f"{ltag}.attn_output.weight", rng.uniform(-0.18, 0.18, (n_embd, q_dim)), use_q4
                ),
                "k_cache": [],
                "v_cache": [],
            }
            ld["attn"] = attn

        ld["ffn_norm"] = pack_f32_1d(
            tensors, f"{ltag}.ffn_norm.weight", rng.uniform(0.75, 1.25, (n_embd,))
        )
        ld["moe"] = build_moe_layer_weights(
            rng,
            tensors,
            ltag,
            n_embd,
            8,
            2,
            64,
            use_q4,
            fused_gate_up=(l == 1),
            gating_func=1,
            norm_weights=True,
            with_shexp=True,
        )
        layers.append(ld)

    write_gguf_v3(gguf_path, kv, tensors)
    if tmlg_path is None:
        return

    all_logits = np.zeros((len(tokens), vocab_size), dtype=np.float32)
    group_dim = d_inner // n_groups

    for step, tok in enumerate(tokens):
        x = emb_deq[tok].copy() * emb_scale
        for ld in layers:
            xn = rms_norm_f64(x, ld["attn_norm"], eps)
            if "ssm" in ld:
                s = ld["ssm"]
                proj = matvec_sim(s["ssm_in"], xn, use_q4, act_q8)
                z = proj[:d_inner]
                xbc = proj[d_inner : d_inner + conv_dim]
                dt_raw = proj[d_inner + conv_dim :]

                full = np.concatenate([s["conv_hist"], xbc[:, None]], axis=1)
                conv_out = np.sum(full * s["conv_w"], axis=1) + s["conv_b"]
                s["conv_hist"] = full[:, 1:].copy()
                conv_act = silu_f64(conv_out)

                x_ssm = conv_act[:d_inner]
                b_ssm = conv_act[d_inner : d_inner + bc_dim]
                c_ssm = conv_act[d_inner + bc_dim :]
                dt = softplus_f64(dt_raw + s["dt_bias"])

                y_scan = np.zeros(d_inner, dtype=np.float64)
                for h in range(n_heads_ssm):
                    g = h // group_ratio
                    dA = math.exp(float(dt[h]) * float(s["ssm_a"][h]))
                    dt_h = float(dt[h])
                    d_h = float(s["ssm_d"][h])
                    b_g = b_ssm[g * d_state : (g + 1) * d_state]
                    c_g = c_ssm[g * d_state : (g + 1) * d_state]
                    for j in range(ssm_head_dim):
                        ch = h * ssm_head_dim + j
                        xv = float(x_ssm[ch])
                        st = dA * s["state"][ch] + (dt_h * xv) * b_g
                        s["state"][ch] = st
                        y_scan[ch] = float(np.dot(st, c_g)) + d_h * xv

                gated = y_scan * silu_f64(z)
                y_norm = np.empty_like(gated)
                for g in range(n_groups):
                    sl = slice(g * group_dim, (g + 1) * group_dim)
                    y_norm[sl] = rms_norm_f64(gated[sl], s["ssm_norm"][sl], eps)
                mx = matvec_sim(s["ssm_out"], y_norm, use_q4, act_q8)
            else:
                a = ld["attn"]
                q = matvec_sim(a["wq"], xn, use_q4, act_q8)
                k = matvec_sim(a["wk"], xn, use_q4, act_q8)
                v = matvec_sim(a["wv"], xn, use_q4, act_q8)
                q = rope_apply_f64(q, n_head_attn, head_dim, head_dim, step, 10_000.0, False)
                k = rope_apply_f64(k, n_kv_attn, head_dim, head_dim, step, 10_000.0, False)
                a["k_cache"].append(k.astype(np.float16))
                a["v_cache"].append(v.astype(np.float16))
                ao = gqa_attention_step_f64(
                    q, a["k_cache"], a["v_cache"], n_head_attn, n_kv_attn, head_dim, attn_scale
                )
                mx = matvec_sim(a["wo"], ao, use_q4, act_q8)

            x = x + res_scale * mx
            xfn = rms_norm_f64(x, ld["ffn_norm"], eps)
            fp = eval_moe_ffn_f64(xfn, ld["moe"], use_q4, act_q8)
            x = x + res_scale * fp

        xn_final = rms_norm_f64(x, out_norm, eps)
        logits = matvec_sim(out_deq, xn_final, use_q4, act_q8) / logit_scale
        all_logits[step] = logits.astype(np.float32)

    write_tmlg(tmlg_path, tokens, vocab_size, all_logits)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--arch",
        required=True,
        choices=[
            "olmoe",
            "granitemoe",
            "granitemoehybrid",
            "granitehybrid",
            "qwen2moe",
            "qwen3moe",
            "bailingmoe2",
            "exaone",
        ],
    )
    ap.add_argument("--out-gguf", required=True, type=Path)
    ap.add_argument("--out-logits", default=None, type=Path)
    ap.add_argument("--tokens", default="1,5,12,7,19,3,28,14")
    ap.add_argument("--quant", default="f32", choices=["f32", "q4_0"])
    ap.add_argument("--act-q8", action="store_true")
    args = ap.parse_args()

    tokens = [int(t) for t in args.tokens.split(",") if t.strip()]
    use_q4 = args.quant == "q4_0"

    if args.arch in ("granitemoehybrid", "granitehybrid"):
        build_and_eval_granite_moe_hybrid(
            args.arch, args.out_gguf, args.out_logits, tokens, use_q4, args.act_q8
        )
    else:
        build_and_eval_llama_moe_or_exaone(
            args.arch, args.out_gguf, args.out_logits, tokens, use_q4, args.act_q8
        )


if __name__ == "__main__":
    main()
