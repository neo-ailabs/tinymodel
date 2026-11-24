#!/usr/bin/env python3
"""Independent FP64 mathematical oracle & synthetic GGUF v3 builder for P1 sub-8B architectures:
  - `olmo2` / `olmo3` : NeoX RoPE, pure Post-Norm (`post_attention_norm` + `post_ffw_norm`, no pre-norms),
                        full-vector `[q_dim]`/`[kv_dim]` QK-Norm, and OLMo 3 3:1 SWA (`pattern = 4`)
                        with dual RoPE (`500_000.0` full vs `10_000.0` SWA).
  - `granite`         : Norm RoPE + LongRoPE (`rope_factors_short` / `rope_factors_long`), optional
                        `rope.scaling.finetuned`, QKV/O + MLP biases, and all 4 muP scaling multipliers
                        (`embedding_scale`, `attention.scale`, `residual_scale`, `1 / logit_scale`).
  - `mamba2`          : Pure Mamba-2 selective state space model (grouped B/C, causal Conv1d + SiLU,
                        `softplus(dt + dt_bias)`, `exp(dt * A)` scan, `silu(z)` gate + grouped RMSNorm).
  - `nemotron_h`      : Heterogeneous per-layer `Mamba-2 SSM` | `NoRoPE GQA` | `Squared-ReLU FFN`.
  - `falcon-h1`       : Parallel `NeoX-RoPE GQA + Mamba-2 SSM` mixer + sequential `SwiGLU FFN`
                        (ignores `rope_freqs.weight`, supports `blk.{l}.ffn_norm` without `.weight`
                        and optional `ssm_norm.weight`).
  - `granitehybrid`   : Interleaved `Mamba-2 | Norm-RoPE GQA` mixer + `SwiGLU FFN` + Granite muP scales.
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
        qs = np.trunc(scaled + np.where(scaled >= np.float32(0.0), np.float32(0.5), np.float32(-0.5))).astype(
            np.int32
        )
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
                elif elem_type == 5:
                    hdr.extend(struct.pack("<i", int(it)))
                elif elem_type == 6:
                    hdr.extend(struct.pack("<f", float(it)))
                elif elem_type == 7:
                    hdr.extend(struct.pack("<B", 1 if it else 0))
                elif elem_type == 8:
                    hdr.extend(_gguf_str(str(it)))
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


def softplus_scalar_f64(x: float) -> float:
    if x > 20.0:
        return x
    if x < -20.0:
        return math.exp(x)
    return math.log1p(math.exp(x))


def rope_apply_f64(
    vec: np.ndarray,
    n_heads: int,
    head_dim: int,
    rope_dim: int,
    pos: int,
    inv_freqs: np.ndarray,
    neox: bool,
) -> np.ndarray:
    out = vec.astype(np.float64).copy().reshape(n_heads, head_dim)
    half = rope_dim // 2
    for i in range(half):
        theta = float(pos) * float(inv_freqs[i])
        c = float(np.float32(math.cos(theta)))
        s = float(np.float32(math.sin(theta)))
        if neox:
            x0 = out[:, i].copy()
            x1 = out[:, i + half].copy()
            out[:, i] = x0 * c - x1 * s
            out[:, i + half] = x0 * s + x1 * c
        else:
            x0 = out[:, 2 * i].copy()
            x1 = out[:, 2 * i + 1].copy()
            out[:, 2 * i] = x0 * c - x1 * s
            out[:, 2 * i + 1] = x0 * s + x1 * c
    return out.reshape(-1)


def default_gpt2_tokenizer_kv(vocab_size: int) -> list[tuple[str, int, object]]:
    tok_strs = ["<|endoftext|>", "<|im_start|>", "<|im_end|>"] + [
        f"tok{i}" for i in range(3, vocab_size)
    ]
    tok_types = [3, 3, 3] + [1] * (vocab_size - 3)
    return [
        ("tokenizer.ggml.model", 8, "gpt2"),
        ("tokenizer.ggml.tokens", 9, (8, tok_strs)),
        ("tokenizer.ggml.token_type", 9, (5, tok_types)),
        ("tokenizer.ggml.merges", 9, (8, [])),
        ("tokenizer.ggml.eos_token_id", 4, 2),
    ]


def build_olmo_or_granite(
    arch: str,
    out_gguf: Path,
    out_tmlg: Path | None,
    tokens: list[int],
    *,
    quant: str,
    act_q8: bool,
    eval_ctx: int,
    rope_finetuned: bool,
    seed: int,
) -> np.ndarray:
    rng = SplitMix64(seed)
    use_q4 = quant == "q4_0"
    head_dim = 32
    n_head = 4
    n_head_kv = 2
    n_embd = n_head * head_dim  # 128
    n_ff = 256
    q_dim = n_head * head_dim
    kv_dim = n_head_kv * head_dim
    vocab_size = 128

    def make_mat(rows: int, cols: int, scale: float = 0.16):
        m = rng.uniform(-scale, scale, (rows, cols))
        if use_q4:
            raw, deq = quantize_q4_0_rows(m)
            return 2, raw, deq
        return 0, m.tobytes(), m

    def mv(w: np.ndarray, act: np.ndarray) -> np.ndarray:
        a = quantize_q8_0_f64(act) if (use_q4 and act_q8) else act.astype(np.float64)
        return w.astype(np.float64) @ a

    is_olmo = arch in ("olmo2", "olmo3")
    if is_olmo:
        n_layers = 4  # layers 0,1,2 SWA (10k), layer 3 full (500k)
        rope_dim = head_dim
        rope_freq_base = 500_000.0
        rope_freq_base_swa = 10_000.0
        rms_eps = 1e-6
        neox = True
        sliding_window = 3
        is_swa = [(il + 1) % 4 != 0 for il in range(n_layers)]
        use_rope = [True] * n_layers
        rope_orig_ctx = 512
        emb_scale = 1.0
        attn_scale = float(np.float32(1.0 / math.sqrt(float(head_dim))))
        res_scale = 1.0
        logit_scale = 1.0
    elif arch == "granite":
        n_layers = 2
        rope_dim = head_dim
        rope_freq_base = 10_000.0
        rope_freq_base_swa = 10_000.0
        rms_eps = 1e-5
        neox = False
        sliding_window = None
        is_swa = [False] * n_layers
        use_rope = [rope_finetuned] * n_layers
        rope_orig_ctx = 48
        emb_scale = 12.0
        attn_scale = 0.08
        res_scale = 0.22
        logit_scale = 8.0
    else:
        raise ValueError(f"unexpected dense P1 arch {arch}")

    emb_t, emb_r, emb_d = make_mat(vocab_size, n_embd, 0.16)
    out_t, out_r, out_d = make_mat(vocab_size, n_embd, 0.16)
    output_norm = rng.uniform(0.85, 1.15, (n_embd,))

    kv_items: list[tuple[str, int, object]] = [
        ("general.architecture", 8, arch),
        (f"{arch}.block_count", 4, n_layers),
        (f"{arch}.context_length", 4, 512),
        (f"{arch}.embedding_length", 4, n_embd),
        (f"{arch}.feed_forward_length", 4, n_ff),
        (f"{arch}.attention.head_count", 4, n_head),
        (f"{arch}.attention.head_count_kv", 4, n_head_kv),
        (f"{arch}.attention.layer_norm_rms_epsilon", 6, rms_eps),
        (f"{arch}.rope.dimension_count", 4, rope_dim),
        (f"{arch}.rope.freq_base", 6, rope_freq_base),
    ]
    if is_olmo:
        kv_items.append((f"{arch}.rope.freq_base_swa", 6, rope_freq_base_swa))
        kv_items.append((f"{arch}.attention.sliding_window", 4, sliding_window))
    else:
        kv_items.extend(
            [
                (f"{arch}.rope.scaling.original_context_length", 4, rope_orig_ctx),
                (f"{arch}.rope.scaling.finetuned", 7, rope_finetuned),
                (f"{arch}.embedding_scale", 6, emb_scale),
                (f"{arch}.attention.scale", 6, attn_scale),
                (f"{arch}.residual_scale", 6, res_scale),
                (f"{arch}.logit_scale", 6, logit_scale),
            ]
        )
    kv_items.extend(default_gpt2_tokenizer_kv(vocab_size))

    tensors: list[tuple[str, tuple[int, ...], int, bytes]] = [
        ("token_embd.weight", (n_embd, vocab_size), emb_t, emb_r),
        ("output.weight", (n_embd, vocab_size), out_t, out_r),
        ("output_norm.weight", (n_embd,), 0, output_norm.tobytes()),
    ]

    half = rope_dim // 2
    base_inv = np.array(
        [float(rope_freq_base) ** (-2.0 * float(i) / float(rope_dim)) for i in range(half)],
        dtype=np.float64,
    )
    swa_inv = np.array(
        [float(rope_freq_base_swa) ** (-2.0 * float(i) / float(rope_dim)) for i in range(half)],
        dtype=np.float64,
    )
    if arch == "granite":
        f_short = rng.uniform(1.0, 1.5, (half,))
        f_long = rng.uniform(2.5, 5.5, (half,))
        tensors.append(("rope_factors_short.weight", (half,), 0, f_short.tobytes()))
        tensors.append(("rope_factors_long.weight", (half,), 0, f_long.tobytes()))
        chosen = f_long if eval_ctx > rope_orig_ctx else f_short
        inv_full = base_inv / chosen.astype(np.float64)
        inv_swa = swa_inv / f_short.astype(np.float64)
    else:
        inv_full = base_inv
        inv_swa = swa_inv

    layers = []
    for l in range(n_layers):
        p = f"blk.{l}"
        wq_t, wq_r, wq_d = make_mat(q_dim, n_embd)
        wk_t, wk_r, wk_d = make_mat(kv_dim, n_embd)
        wv_t, wv_r, wv_d = make_mat(kv_dim, n_embd)
        wo_t, wo_r, wo_d = make_mat(n_embd, q_dim)
        wg_t, wg_r, wg_d = make_mat(n_ff, n_embd)
        wu_t, wu_r, wu_d = make_mat(n_ff, n_embd)
        wd_t, wd_r, wd_d = make_mat(n_embd, n_ff)
        tensors.extend(
            [
                (f"{p}.attn_q.weight", (n_embd, q_dim), wq_t, wq_r),
                (f"{p}.attn_k.weight", (n_embd, kv_dim), wk_t, wk_r),
                (f"{p}.attn_v.weight", (n_embd, kv_dim), wv_t, wv_r),
                (f"{p}.attn_output.weight", (q_dim, n_embd), wo_t, wo_r),
                (f"{p}.ffn_gate.weight", (n_embd, n_ff), wg_t, wg_r),
                (f"{p}.ffn_up.weight", (n_embd, n_ff), wu_t, wu_r),
                (f"{p}.ffn_down.weight", (n_ff, n_embd), wd_t, wd_r),
            ]
        )

        if is_olmo:
            # Pure Post-Norm (`attn_norm = None`, `ffn_norm = None`) + full-vector QK-Norm!
            pan = rng.uniform(0.8, 1.2, (n_embd,))
            pfn = rng.uniform(0.8, 1.2, (n_embd,))
            qn = rng.uniform(0.75, 1.25, (q_dim,))
            kn = rng.uniform(0.75, 1.25, (kv_dim,))
            tensors.extend(
                [
                    (f"{p}.post_attention_norm.weight", (n_embd,), 0, pan.tobytes()),
                    (f"{p}.post_ffw_norm.weight", (n_embd,), 0, pfn.tobytes()),
                    (f"{p}.attn_q_norm.weight", (q_dim,), 0, qn.tobytes()),
                    (f"{p}.attn_k_norm.weight", (kv_dim,), 0, kn.tobytes()),
                ]
            )
            layers.append(
                {
                    "an": None,
                    "fn": None,
                    "pan": pan,
                    "pfn": pfn,
                    "qn": qn,
                    "kn": kn,
                    "wq": wq_d,
                    "wk": wk_d,
                    "wv": wv_d,
                    "wo": wo_d,
                    "wg": wg_d,
                    "wu": wu_d,
                    "wd": wd_d,
                    "bq": None,
                    "bk": None,
                    "bv": None,
                    "bo": None,
                    "bg": None,
                    "bu": None,
                    "bd": None,
                }
            )
        else:
            an = rng.uniform(0.8, 1.2, (n_embd,))
            fn = rng.uniform(0.8, 1.2, (n_embd,))
            bq = rng.uniform(-0.15, 0.15, (q_dim,))
            bk = rng.uniform(-0.15, 0.15, (kv_dim,))
            bv = rng.uniform(-0.15, 0.15, (kv_dim,))
            bo = rng.uniform(-0.15, 0.15, (n_embd,))
            bg = rng.uniform(-0.15, 0.15, (n_ff,))
            bu = rng.uniform(-0.15, 0.15, (n_ff,))
            bd = rng.uniform(-0.15, 0.15, (n_embd,))
            tensors.extend(
                [
                    (f"{p}.attn_norm.weight", (n_embd,), 0, an.tobytes()),
                    (f"{p}.ffn_norm.weight", (n_embd,), 0, fn.tobytes()),
                    (f"{p}.attn_q.bias", (q_dim,), 0, bq.tobytes()),
                    (f"{p}.attn_k.bias", (kv_dim,), 0, bk.tobytes()),
                    (f"{p}.attn_v.bias", (kv_dim,), 0, bv.tobytes()),
                    (f"{p}.attn_output.bias", (n_embd,), 0, bo.tobytes()),
                    (f"{p}.ffn_gate.bias", (n_ff,), 0, bg.tobytes()),
                    (f"{p}.ffn_up.bias", (n_ff,), 0, bu.tobytes()),
                    (f"{p}.ffn_down.bias", (n_embd,), 0, bd.tobytes()),
                ]
            )
            layers.append(
                {
                    "an": an,
                    "fn": fn,
                    "pan": None,
                    "pfn": None,
                    "qn": None,
                    "kn": None,
                    "wq": wq_d,
                    "wk": wk_d,
                    "wv": wv_d,
                    "wo": wo_d,
                    "wg": wg_d,
                    "wu": wu_d,
                    "wd": wd_d,
                    "bq": bq,
                    "bk": bk,
                    "bv": bv,
                    "bo": bo,
                    "bg": bg,
                    "bu": bu,
                    "bd": bd,
                }
            )

    write_gguf_v3(out_gguf, kv_items, tensors)

    k_cache = [
        np.zeros((len(tokens), n_head_kv, head_dim), dtype=np.float16) for _ in range(n_layers)
    ]
    v_cache = [
        np.zeros((len(tokens), n_head_kv, head_dim), dtype=np.float16) for _ in range(n_layers)
    ]
    all_logits = np.zeros((len(tokens), vocab_size), dtype=np.float32)
    group = n_head // n_head_kv
    inv_logit = float(np.float32(1.0 / logit_scale))
    eff_emb_scale = float(np.float32(emb_scale))
    eff_attn_scale = float(np.float32(attn_scale))
    eff_res_scale = float(np.float32(res_scale))

    for pos, tid in enumerate(tokens):
        x = emb_d[tid].astype(np.float64) * eff_emb_scale
        for l_idx, lw in enumerate(layers):
            xb = rms_norm_f64(x, lw["an"], rms_eps) if lw["an"] is not None else x.copy()
            q = mv(lw["wq"], xb)
            k = mv(lw["wk"], xb)
            v = mv(lw["wv"], xb)
            if lw["bq"] is not None:
                q += lw["bq"].astype(np.float64)
                k += lw["bk"].astype(np.float64)
                v += lw["bv"].astype(np.float64)

            if lw["qn"] is not None:
                q = rms_norm_f64(q, lw["qn"], rms_eps)
                k = rms_norm_f64(k, lw["kn"], rms_eps)

            if use_rope[l_idx]:
                inv = inv_swa if is_swa[l_idx] else inv_full
                q = rope_apply_f64(q, n_head, head_dim, rope_dim, pos, inv, neox)
                k = rope_apply_f64(k, n_head_kv, head_dim, rope_dim, pos, inv, neox)

            k_cache[l_idx][pos] = k.reshape(n_head_kv, head_dim).astype(np.float16)
            v_cache[l_idx][pos] = v.reshape(n_head_kv, head_dim).astype(np.float16)

            start_kv = (
                max(0, pos + 1 - sliding_window)
                if (sliding_window is not None and is_swa[l_idx])
                else 0
            )
            qh = q.reshape(n_head, head_dim)
            attn_out = np.zeros((n_head, head_dim), dtype=np.float64)
            for h in range(n_head):
                kv_h = h // group
                kh = k_cache[l_idx][start_kv : pos + 1, kv_h, :].astype(np.float64)
                vh = v_cache[l_idx][start_kv : pos + 1, kv_h, :].astype(np.float64)
                scores = (kh @ qh[h]) * eff_attn_scale
                scores -= np.max(scores)
                probs = np.exp(scores)
                probs /= np.sum(probs)
                attn_out[h] = probs @ vh

            attn_proj = mv(lw["wo"], attn_out.reshape(-1))
            if lw["bo"] is not None:
                attn_proj += lw["bo"].astype(np.float64)
            if lw["pan"] is not None:
                attn_proj = rms_norm_f64(attn_proj, lw["pan"], rms_eps)
            x = x + eff_res_scale * attn_proj

            xb_ffn = rms_norm_f64(x, lw["fn"], rms_eps) if lw["fn"] is not None else x.copy()
            gate = mv(lw["wg"], xb_ffn)
            up = mv(lw["wu"], xb_ffn)
            if lw["bg"] is not None:
                gate += lw["bg"].astype(np.float64)
                up += lw["bu"].astype(np.float64)
            ffn_proj = mv(lw["wd"], silu_f64(gate) * up)
            if lw["bd"] is not None:
                ffn_proj += lw["bd"].astype(np.float64)
            if lw["pfn"] is not None:
                ffn_proj = rms_norm_f64(ffn_proj, lw["pfn"], rms_eps)
            x = x + eff_res_scale * ffn_proj

        xn = rms_norm_f64(x, output_norm, rms_eps)
        all_logits[pos] = (mv(out_d, xn) * inv_logit).astype(np.float32)

    write_tmlg(out_tmlg, tokens, vocab_size, all_logits)
    return all_logits


def build_mamba2_family(
    arch: str,
    out_gguf: Path,
    out_tmlg: Path | None,
    tokens: list[int],
    *,
    quant: str,
    act_q8: bool,
    seed: int,
) -> np.ndarray:
    rng = SplitMix64(seed)
    use_q4 = quant == "q4_0"

    n_embd = 128
    vocab_size = 128
    rms_eps = 1e-5
    ssm_norm_eps = 1e-5

    d_conv = 4
    d_inner = 128  # divisible by 32 for Q4_0 `ssm_out.weight`
    d_state = 32
    n_ssm_head = 4
    n_group = 2
    d_head = d_inner // n_ssm_head  # 32
    group_dim = d_inner // n_group  # 64
    bc_dim = n_group * d_state  # 64
    conv_dim = d_inner + 2 * bc_dim  # 256
    d_in_proj = 2 * d_inner + 2 * bc_dim + n_ssm_head  # 388

    head_dim = 32
    rope_dim = 32
    rope_freq_base = 10_000.0

    def make_mat(rows: int, cols: int, scale: float = 0.14):
        m = rng.uniform(-scale, scale, (rows, cols))
        if use_q4:
            raw, deq = quantize_q4_0_rows(m)
            return 2, raw, deq
        return 0, m.tobytes(), m

    def mv(w: np.ndarray, act: np.ndarray) -> np.ndarray:
        a = quantize_q8_0_f64(act) if (use_q4 and act_q8) else act.astype(np.float64)
        return w.astype(np.float64) @ a

    if arch == "mamba2":
        n_layers = 2
        n_head_arr = [0, 0]
        n_kv_arr = [0, 0]
        n_ff_arr = [0, 0]
        emb_scale = 1.0
        attn_scale = 1.0
        res_scale = 1.0
        logit_scale = 1.0
        use_rope = False
        neox = False
    elif arch == "nemotron_h":
        # Heterogeneous: blk.0 = Mamba-2 SSM, blk.1 = NoRoPE GQA, blk.2 = ReLU^2 FFN
        n_layers = 3
        n_head_arr = [0, 4, 0]
        n_kv_arr = [0, 2, 0]
        n_ff_arr = [0, 0, 256]
        emb_scale = 1.0
        attn_scale = float(np.float32(1.0 / math.sqrt(float(head_dim))))
        res_scale = 1.0
        logit_scale = 1.0
        use_rope = False
        neox = False
    elif arch == "falcon-h1":
        # Parallel (NeoX-RoPE GQA + Mamba-2 SSM) + sequential SwiGLU FFN
        n_layers = 2
        n_head_arr = [4, 4]
        n_kv_arr = [2, 2]
        n_ff_arr = [256, 256]
        emb_scale = 1.0
        attn_scale = float(np.float32(1.0 / math.sqrt(float(head_dim))))
        res_scale = 1.0
        logit_scale = 1.0
        use_rope = True
        neox = True
    elif arch == "granitehybrid":
        # Interleaved: blk.0 = Mamba-2 + SwiGLU, blk.1 = Norm-RoPE GQA + SwiGLU, with Granite muP
        n_layers = 2
        n_head_arr = [0, 4]
        n_kv_arr = [0, 2]
        n_ff_arr = [256, 256]
        emb_scale = 8.0
        attn_scale = 0.09
        res_scale = 0.25
        logit_scale = 6.0
        use_rope = True
        neox = False
    else:
        raise ValueError(f"unexpected Mamba-2 family arch {arch}")

    emb_t, emb_r, emb_d = make_mat(vocab_size, n_embd, 0.16)
    out_t, out_r, out_d = make_mat(vocab_size, n_embd, 0.16)
    output_norm = rng.uniform(0.85, 1.15, (n_embd,))

    kv_items: list[tuple[str, int, object]] = [
        ("general.architecture", 8, arch),
        (f"{arch}.block_count", 4, n_layers),
        (f"{arch}.context_length", 4, 512),
        (f"{arch}.embedding_length", 4, n_embd),
        (f"{arch}.attention.layer_norm_rms_epsilon", 6, rms_eps),
        (f"{arch}.ssm.conv_kernel", 4, d_conv),
        (f"{arch}.ssm.inner_size", 4, d_inner),
        (f"{arch}.ssm.state_size", 4, d_state),
        (f"{arch}.ssm.time_step_rank", 4, n_ssm_head),
        (f"{arch}.ssm.group_count", 4, n_group),
        (f"{arch}.ssm.norm_epsilon", 6, ssm_norm_eps),
        (f"{arch}.attention.head_count", 9, (4, n_head_arr)),
        (f"{arch}.attention.head_count_kv", 9, (4, n_kv_arr)),
        (f"{arch}.feed_forward_length", 9, (4, n_ff_arr)),
    ]
    if max(n_head_arr) > 0:
        kv_items.extend(
            [
                (f"{arch}.attention.key_length", 4, head_dim),
                (f"{arch}.rope.dimension_count", 4, rope_dim),
                (f"{arch}.rope.freq_base", 6, rope_freq_base),
            ]
        )
    if arch == "granitehybrid":
        kv_items.extend(
            [
                (f"{arch}.embedding_scale", 6, emb_scale),
                (f"{arch}.attention.scale", 6, attn_scale),
                (f"{arch}.residual_scale", 6, res_scale),
                (f"{arch}.logit_scale", 6, logit_scale),
            ]
        )
    kv_items.extend(default_gpt2_tokenizer_kv(vocab_size))

    tensors: list[tuple[str, tuple[int, ...], int, bytes]] = [
        ("token_embd.weight", (n_embd, vocab_size), emb_t, emb_r),
        ("output.weight", (n_embd, vocab_size), out_t, out_r),
        ("output_norm.weight", (n_embd,), 0, output_norm.tobytes()),
    ]
    if arch == "falcon-h1":
        # Verify `falcon-h1` ignores `rope_freqs.weight` (`falcon-h1.cpp:111`)!
        bogus_rope = np.full((rope_dim // 2,), 999.0, dtype=np.float32)
        tensors.append(("rope_freqs.weight", (rope_dim // 2,), 0, bogus_rope.tobytes()))

    half = rope_dim // 2
    inv_freqs = np.array(
        [float(rope_freq_base) ** (-2.0 * float(i) / float(rope_dim)) for i in range(half)],
        dtype=np.float64,
    )

    def build_ssm(p: str, l_idx: int, include_ssm_norm: bool) -> dict[str, object]:
        sin_t, sin_r, sin_d = make_mat(d_in_proj, n_embd)
        sout_t, sout_r, sout_d = make_mat(n_embd, d_inner)
        conv_w = rng.uniform(-0.22, 0.22, (conv_dim, d_conv))
        conv_b = rng.uniform(-0.10, 0.10, (conv_dim,))
        dt_bias = rng.uniform(-1.2, -0.2, (n_ssm_head,))
        ssm_a = -rng.uniform(0.3, 1.8, (n_ssm_head,))
        ssm_d = rng.uniform(0.1, 0.8, (n_ssm_head,))
        # Exercise BOTH 2D `[1, n_heads]` (`llama.cpp` canonical) and 1D `[n_heads]`!
        a_dims = (1, n_ssm_head) if l_idx == 0 else (n_ssm_head,)
        tensors.extend(
            [
                (f"{p}.ssm_in.weight", (n_embd, d_in_proj), sin_t, sin_r),
                (f"{p}.ssm_conv1d.weight", (d_conv, conv_dim), 0, conv_w.tobytes()),
                (f"{p}.ssm_conv1d.bias", (conv_dim,), 0, conv_b.tobytes()),
                (f"{p}.ssm_dt.bias", (n_ssm_head,), 0, dt_bias.tobytes()),
                (f"{p}.ssm_a", a_dims, 0, ssm_a.tobytes()),
                (f"{p}.ssm_d", a_dims, 0, ssm_d.tobytes()),
                (f"{p}.ssm_out.weight", (d_inner, n_embd), sout_t, sout_r),
            ]
        )
        if include_ssm_norm:
            sn = rng.uniform(0.8, 1.2, (d_inner,))
            sn_dims = (group_dim, n_group) if l_idx == 0 else (d_inner,)
            tensors.append((f"{p}.ssm_norm.weight", sn_dims, 0, sn.tobytes()))
        else:
            sn = None
        return {
            "ssm_in": sin_d,
            "conv_w": conv_w,
            "conv_b": conv_b,
            "dt_bias": dt_bias,
            "ssm_a": ssm_a,
            "ssm_d": ssm_d,
            "ssm_norm": sn,
            "ssm_out": sout_d,
        }

    def build_attn(p: str, nh: int, nh_kv: int) -> dict[str, object]:
        qd = nh * head_dim
        kvd = nh_kv * head_dim
        wq_t, wq_r, wq_d = make_mat(qd, n_embd)
        wk_t, wk_r, wk_d = make_mat(kvd, n_embd)
        wv_t, wv_r, wv_d = make_mat(kvd, n_embd)
        wo_t, wo_r, wo_d = make_mat(n_embd, qd)
        tensors.extend(
            [
                (f"{p}.attn_q.weight", (n_embd, qd), wq_t, wq_r),
                (f"{p}.attn_k.weight", (n_embd, kvd), wk_t, wk_r),
                (f"{p}.attn_v.weight", (n_embd, kvd), wv_t, wv_r),
                (f"{p}.attn_output.weight", (qd, n_embd), wo_t, wo_r),
            ]
        )
        return {"nh": nh, "nh_kv": nh_kv, "wq": wq_d, "wk": wk_d, "wv": wv_d, "wo": wo_d}

    def build_swiglu(p: str, n_ff: int) -> dict[str, object]:
        wg_t, wg_r, wg_d = make_mat(n_ff, n_embd)
        wu_t, wu_r, wu_d = make_mat(n_ff, n_embd)
        wd_t, wd_r, wd_d = make_mat(n_embd, n_ff)
        tensors.extend(
            [
                (f"{p}.ffn_gate.weight", (n_embd, n_ff), wg_t, wg_r),
                (f"{p}.ffn_up.weight", (n_embd, n_ff), wu_t, wu_r),
                (f"{p}.ffn_down.weight", (n_ff, n_embd), wd_t, wd_r),
            ]
        )
        return {"kind": "swiglu", "wg": wg_d, "wu": wu_d, "wd": wd_d}

    layers = []
    for l in range(n_layers):
        p = f"blk.{l}"
        an = rng.uniform(0.8, 1.2, (n_embd,))
        tensors.append((f"{p}.attn_norm.weight", (n_embd,), 0, an.tobytes()))

        if arch == "mamba2":
            layers.append({"kind": "ssm_only", "an": an, "ssm": build_ssm(p, l, True)})
        elif arch == "nemotron_h":
            if n_kv_arr[l] == 0 and n_ff_arr[l] == 0:
                layers.append({"kind": "ssm_only", "an": an, "ssm": build_ssm(p, l, True)})
            elif n_ff_arr[l] == 0:
                layers.append(
                    {
                        "kind": "attn_only",
                        "an": an,
                        "attn": build_attn(p, n_head_arr[l], n_kv_arr[l]),
                    }
                )
            else:
                n_ff = n_ff_arr[l]
                wu_t, wu_r, wu_d = make_mat(n_ff, n_embd)
                wd_t, wd_r, wd_d = make_mat(n_embd, n_ff)
                bu = rng.uniform(-0.1, 0.1, (n_ff,))
                bd = rng.uniform(-0.1, 0.1, (n_embd,))
                tensors.extend(
                    [
                        (f"{p}.ffn_up.weight", (n_embd, n_ff), wu_t, wu_r),
                        (f"{p}.ffn_down.weight", (n_ff, n_embd), wd_t, wd_r),
                        (f"{p}.ffn_up.bias", (n_ff,), 0, bu.tobytes()),
                        (f"{p}.ffn_down.bias", (n_embd,), 0, bd.tobytes()),
                    ]
                )
                layers.append(
                    {
                        "kind": "ffn_only",
                        "an": an,
                        "ffn": {"kind": "relu2", "wu": wu_d, "wd": wd_d, "bu": bu, "bd": bd},
                    }
                )
        elif arch == "granitehybrid":
            fn = rng.uniform(0.8, 1.2, (n_embd,))
            tensors.append((f"{p}.ffn_norm.weight", (n_embd,), 0, fn.tobytes()))
            ffn = build_swiglu(p, n_ff_arr[l])
            if n_kv_arr[l] == 0:
                layers.append(
                    {"kind": "ssm_ffn", "an": an, "ssm": build_ssm(p, l, True), "fn": fn, "ffn": ffn}
                )
            else:
                layers.append(
                    {
                        "kind": "attn_ffn",
                        "an": an,
                        "attn": build_attn(p, n_head_arr[l], n_kv_arr[l]),
                        "fn": fn,
                        "ffn": ffn,
                    }
                )
        elif arch == "falcon-h1":
            fn = rng.uniform(0.8, 1.2, (n_embd,))
            # Exercise BOTH `blk.0.ffn_norm` (no `.weight` suffix) and `blk.1.ffn_norm.weight`,
            # plus optional `ssm_norm` absent on layer 0 and present on layer 1!
            fn_name = f"{p}.ffn_norm" if l == 0 else f"{p}.ffn_norm.weight"
            tensors.append((fn_name, (n_embd,), 0, fn.tobytes()))
            attn = build_attn(p, n_head_arr[l], n_kv_arr[l])
            ssm = build_ssm(p, l, include_ssm_norm=(l == 1))
            ffn = build_swiglu(p, n_ff_arr[l])
            layers.append(
                {
                    "kind": "parallel_attn_ssm_ffn",
                    "an": an,
                    "attn": attn,
                    "ssm": ssm,
                    "fn": fn,
                    "ffn": ffn,
                }
            )

    write_gguf_v3(out_gguf, kv_items, tensors)

    hist_len = d_conv - 1
    ssm_states = [
        np.zeros((n_ssm_head, d_head, d_state), dtype=np.float64) for _ in range(n_layers)
    ]
    conv_hists = [np.zeros((conv_dim, hist_len), dtype=np.float64) for _ in range(n_layers)]
    k_caches = [
        np.zeros((len(tokens), max(1, n_kv_arr[l]), head_dim), dtype=np.float16)
        for l in range(n_layers)
    ]
    v_caches = [
        np.zeros((len(tokens), max(1, n_kv_arr[l]), head_dim), dtype=np.float16)
        for l in range(n_layers)
    ]

    all_logits = np.zeros((len(tokens), vocab_size), dtype=np.float32)
    eff_emb_scale = float(np.float32(emb_scale))
    eff_attn_scale = float(np.float32(attn_scale))
    eff_res_scale = float(np.float32(res_scale))
    inv_logit = float(np.float32(1.0 / logit_scale))
    heads_per_group = n_ssm_head // n_group

    def run_ssm_step(l_idx: int, ssm: dict[str, object], xn: np.ndarray) -> np.ndarray:
        proj = mv(ssm["ssm_in"], xn)
        z = proj[:d_inner]
        xbc = proj[d_inner : d_inner + conv_dim]
        dt_raw = proj[d_inner + conv_dim :]

        w_conv = ssm["conv_w"].astype(np.float64)
        b_conv = ssm["conv_b"].astype(np.float64)
        pre = (
            np.sum(conv_hists[l_idx] * w_conv[:, :hist_len], axis=1)
            + xbc * w_conv[:, hist_len]
            + b_conv
        )
        conv_hists[l_idx][:, :-1] = conv_hists[l_idx][:, 1:]
        conv_hists[l_idx][:, -1] = xbc
        act = silu_f64(pre)

        x_ssm = act[:d_inner].reshape(n_ssm_head, d_head)
        b_mat = act[d_inner : d_inner + bc_dim].reshape(n_group, d_state)
        c_mat = act[d_inner + bc_dim :].reshape(n_group, d_state)

        y_scan = np.zeros((n_ssm_head, d_head), dtype=np.float64)
        st = ssm_states[l_idx]
        for h in range(n_ssm_head):
            g = h // heads_per_group
            dt = softplus_scalar_f64(float(dt_raw[h]) + float(ssm["dt_bias"][h]))
            da = math.exp(dt * float(ssm["ssm_a"][h]))
            d_skip = float(ssm["ssm_d"][h])
            st[h] = da * st[h] + np.outer(dt * x_ssm[h], b_mat[g])
            y_scan[h] = st[h] @ c_mat[g] + d_skip * x_ssm[h]

        y = silu_f64(z) * y_scan.reshape(-1)
        if ssm["ssm_norm"] is not None:
            nw = ssm["ssm_norm"]
            for g in range(n_group):
                s = g * group_dim
                e = s + group_dim
                y[s:e] = rms_norm_f64(y[s:e], nw[s:e], ssm_norm_eps)
        return mv(ssm["ssm_out"], y)

    def run_attn_step(l_idx: int, attn: dict[str, object], xn: np.ndarray, pos: int) -> np.ndarray:
        nh = int(attn["nh"])
        nh_kv = int(attn["nh_kv"])
        grp = nh // nh_kv
        q = mv(attn["wq"], xn)
        k = mv(attn["wk"], xn)
        v = mv(attn["wv"], xn)
        if use_rope:
            q = rope_apply_f64(q, nh, head_dim, rope_dim, pos, inv_freqs, neox)
            k = rope_apply_f64(k, nh_kv, head_dim, rope_dim, pos, inv_freqs, neox)
        k_caches[l_idx][pos] = k.reshape(nh_kv, head_dim).astype(np.float16)
        v_caches[l_idx][pos] = v.reshape(nh_kv, head_dim).astype(np.float16)

        qh = q.reshape(nh, head_dim)
        attn_out = np.zeros((nh, head_dim), dtype=np.float64)
        for h in range(nh):
            kv_h = h // grp
            kh = k_caches[l_idx][: pos + 1, kv_h, :].astype(np.float64)
            vh = v_caches[l_idx][: pos + 1, kv_h, :].astype(np.float64)
            scores = (kh @ qh[h]) * eff_attn_scale
            scores -= np.max(scores)
            probs = np.exp(scores)
            probs /= np.sum(probs)
            attn_out[h] = probs @ vh
        return mv(attn["wo"], attn_out.reshape(-1))

    def run_ffn_step(ffn: dict[str, object], xn: np.ndarray) -> np.ndarray:
        if ffn["kind"] == "swiglu":
            gate = mv(ffn["wg"], xn)
            up = mv(ffn["wu"], xn)
            return mv(ffn["wd"], silu_f64(gate) * up)
        up = mv(ffn["wu"], xn) + ffn["bu"].astype(np.float64)
        r = np.maximum(up, 0.0)
        return mv(ffn["wd"], r * r) + ffn["bd"].astype(np.float64)

    for pos, tid in enumerate(tokens):
        x = emb_d[tid].astype(np.float64) * eff_emb_scale
        for l_idx, lw in enumerate(layers):
            xn = rms_norm_f64(x, lw["an"], rms_eps)
            kind = lw["kind"]
            if kind == "ssm_only":
                x = x + eff_res_scale * run_ssm_step(l_idx, lw["ssm"], xn)
            elif kind == "attn_only":
                x = x + eff_res_scale * run_attn_step(l_idx, lw["attn"], xn, pos)
            elif kind == "ffn_only":
                x = x + eff_res_scale * run_ffn_step(lw["ffn"], xn)
            elif kind == "ssm_ffn":
                x = x + eff_res_scale * run_ssm_step(l_idx, lw["ssm"], xn)
                x = x + eff_res_scale * run_ffn_step(
                    lw["ffn"], rms_norm_f64(x, lw["fn"], rms_eps)
                )
            elif kind == "attn_ffn":
                x = x + eff_res_scale * run_attn_step(l_idx, lw["attn"], xn, pos)
                x = x + eff_res_scale * run_ffn_step(
                    lw["ffn"], rms_norm_f64(x, lw["fn"], rms_eps)
                )
            elif kind == "parallel_attn_ssm_ffn":
                mix = run_attn_step(l_idx, lw["attn"], xn, pos) + run_ssm_step(l_idx, lw["ssm"], xn)
                x = x + eff_res_scale * mix
                x = x + eff_res_scale * run_ffn_step(
                    lw["ffn"], rms_norm_f64(x, lw["fn"], rms_eps)
                )
            else:
                raise ValueError(f"unknown block kind {kind}")

        xn_out = rms_norm_f64(x, output_norm, rms_eps)
        all_logits[pos] = (mv(out_d, xn_out) * inv_logit).astype(np.float32)

    write_tmlg(out_tmlg, tokens, vocab_size, all_logits)
    return all_logits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--arch",
        required=True,
        choices=[
            "olmo2",
            "olmo3",
            "granite",
            "mamba2",
            "nemotron_h",
            "falcon-h1",
            "granitehybrid",
        ],
    )
    ap.add_argument("--out-gguf", type=Path, required=True)
    ap.add_argument("--out-tmlg", type=Path, default=None)
    ap.add_argument("--tokens", type=str, default="1,5,12,7,19,3,28,14")
    ap.add_argument("--quant", choices=["f32", "q4_0"], default="f32")
    ap.add_argument("--act-q8", action="store_true")
    ap.add_argument("--eval-ctx", type=int, default=64)
    ap.add_argument("--no-rope-finetuned", action="store_true")
    ap.add_argument("--seed", type=int, default=20261125)
    args = ap.parse_args()

    toks = [int(x) for x in args.tokens.split(",") if x.strip()]
    if args.arch in ("olmo2", "olmo3", "granite"):
        build_olmo_or_granite(
            args.arch,
            args.out_gguf,
            args.out_tmlg,
            toks,
            quant=args.quant,
            act_q8=args.act_q8,
            eval_ctx=args.eval_ctx,
            rope_finetuned=not args.no_rope_finetuned,
            seed=args.seed,
        )
    else:
        build_mamba2_family(
            args.arch,
            args.out_gguf,
            args.out_tmlg,
            toks,
            quant=args.quant,
            act_q8=args.act_q8,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
