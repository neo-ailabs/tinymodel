#!/usr/bin/env python3
"""Independent FP64 mathematical oracle & synthetic GGUF builder for `qwen2` (Qwen2 & Qwen2.5).

Exercises all `qwen2` architectural features in GGUF (`third_party/llama.cpp/src/models/qwen2.cpp`):
  - `general.architecture == "qwen2"` (`qwen2.*` metadata keys)
  - NeoX split-half RoPE (`rope_freq_base = 1_000_000.0`, `rms_eps = 1e-6`)
  - Grouped-Query Attention (`n_head > n_head_kv`) with mandatory 1D FP32 `attn_q.bias`,
    `attn_k.bias`, `attn_v.bias` added before RoPE & F16 KV cache storage
  - Optional `attn_output.bias` and `output.bias`
  - Both tied (`0.5B` / `1.5B` / `3B`) and untied (`7B+`) `output.weight`
  - SwiGLU FFN (`ffn_gate`, `ffn_up`, `ffn_down`)
  - `qwen2` GPT-2 Byte-BPE tokenizer metadata (`tokenizer.ggml.pre = "qwen2"`, `<|im_end|>` EOT)

Usage:
  python3 tools/oracle/qwen2_oracle.py --out-gguf model.gguf --out-tmlg ref.bin --tokens 1,5,12,7,19,3
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
    """Quantizes `[rows, cols]` FP32 matrix to ggml `Q4_0` bytes and returns `(raw_bytes, dequant_f32)`."""
    rows, cols = mat_f32.shape
    assert cols % 32 == 0
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


def _gguf_str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def write_gguf_v3(path: Path, kv_items: list[tuple[str, int, object]], tensors: list[tuple[str, tuple[int, ...], int, bytes]]) -> None:
    alignment = 32
    hdr = bytearray()
    hdr.extend(b"GGUF")
    hdr.extend(struct.pack("<IQQ", 3, len(tensors), len(kv_items)))

    for key, vtype, val in kv_items:
        hdr.extend(_gguf_str(key))
        hdr.extend(struct.pack("<I", vtype))
        if vtype == 4:  # U32
            hdr.extend(struct.pack("<I", int(val)))
        elif vtype == 6:  # F32
            hdr.extend(struct.pack("<f", float(val)))
        elif vtype == 8:  # STRING
            hdr.extend(_gguf_str(str(val)))
        elif vtype == 9:  # ARRAY
            elem_type, items = val
            hdr.extend(struct.pack("<IQ", elem_type, len(items)))
            for it in items:
                if elem_type == 8:
                    hdr.extend(_gguf_str(str(it)))
                elif elem_type == 5:
                    hdr.extend(struct.pack("<i", int(it)))
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


def rms_norm_f64(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    ms = float(np.mean(x.astype(np.float64) ** 2))
    scale = 1.0 / math.sqrt(ms + float(eps))
    return (x.astype(np.float64) * scale) * w.astype(np.float64)


def quantize_q8_0_f64(vec: np.ndarray) -> np.ndarray:
    """Simulates ggml / `tm-cpu` `quantize_q8_0` block activation quantization (`QK8_0 = 32`)."""
    x = vec.astype(np.float32)
    out = np.empty_like(x, dtype=np.float64)
    for b in range(x.size // 32):
        blk = x[b * 32 : (b + 1) * 32]
        amax = float(np.max(np.abs(blk)))
        if amax == 0.0:
            out[b * 32 : (b + 1) * 32] = 0.0
            continue
        d = amax / 127.0
        inv_d = 1.0 / d
        d_eff = float(np.float16(d))
        qs = np.trunc(blk * inv_d + np.where(blk >= 0.0, 0.5, -0.5)).astype(np.int32)
        qs = np.clip(qs, -128, 127)
        out[b * 32 : (b + 1) * 32] = qs.astype(np.float64) * d_eff
    return out


def rope_neox_f64(vec: np.ndarray, n_heads: int, head_dim: int, rope_dim: int, pos: int, freq_base: float) -> np.ndarray:
    out = vec.astype(np.float64).copy().reshape(n_heads, head_dim)
    half = rope_dim // 2
    for i in range(half):
        freq = float(freq_base) ** (-2.0 * float(i) / float(rope_dim))
        theta = float(pos) * freq
        c = math.cos(theta)
        s = math.sin(theta)
        x0 = out[:, i].copy()
        x1 = out[:, i + half].copy()
        out[:, i] = x0 * c - x1 * s
        out[:, i + half] = x0 * s + x1 * c
    return out.reshape(-1)


def build_and_eval_qwen2(
    out_gguf: Path,
    out_tmlg: Path | None,
    tokens: list[int],
    *,
    quant: str = "f32",
    act_q8: bool = False,
    tied_output: bool = True,
    with_wo_bias: bool = True,
    with_output_bias: bool = True,
    seed: int = 20261123,
) -> np.ndarray:
    rng = SplitMix64(seed)
    n_layers = 2
    n_embd = 64
    n_ff = 128
    n_head = 4
    n_head_kv = 2
    head_dim = 16 if quant == "f32" else 32
    if head_dim == 32:
        n_embd = 128
        n_ff = 256
    q_dim = n_head * head_dim
    kv_dim = n_head_kv * head_dim
    vocab_size = 320
    rope_freq_base = 1_000_000.0
    rms_eps = 1e-6

    use_q4 = quant == "q4_0"

    def make_mat(rows: int, cols: int, scale: float = 0.15):
        m = rng.uniform(-scale, scale, (rows, cols))
        if use_q4:
            raw, deq = quantize_q4_0_rows(m)
            return 2, raw, deq
        return 0, m.tobytes(), m

    emb_type, emb_raw, emb_deq = make_mat(vocab_size, n_embd, 0.20)
    if tied_output:
        out_deq = emb_deq
        out_entry = None
    else:
        out_type, out_raw, out_deq = make_mat(vocab_size, n_embd, 0.18)
        out_entry = ("output.weight", (n_embd, vocab_size), out_type, out_raw)

    output_norm = rng.uniform(0.8, 1.2, (n_embd,))
    output_bias = rng.uniform(-0.25, 0.25, (vocab_size,)) if with_output_bias else None

    tok_strs = []
    tok_types = []
    for i in range(vocab_size):
        if i == 0:
            tok_strs.append("<|endoftext|>")
            tok_types.append(3)
        elif i == 1:
            tok_strs.append("<|im_start|>")
            tok_types.append(3)
        elif i == 2:
            tok_strs.append("<|im_end|>")
            tok_types.append(3)
        elif 3 <= i < 13:
            tok_strs.append(str(i - 3))
            tok_types.append(1)
        else:
            tok_strs.append(f"tok{i}")
            tok_types.append(1)

    kv_items = [
        ("general.architecture", 8, "qwen2"),
        ("general.name", 8, "synthetic-qwen2.5-oracle"),
        ("qwen2.block_count", 4, n_layers),
        ("qwen2.context_length", 4, 512),
        ("qwen2.embedding_length", 4, n_embd),
        ("qwen2.feed_forward_length", 4, n_ff),
        ("qwen2.attention.head_count", 4, n_head),
        ("qwen2.attention.head_count_kv", 4, n_head_kv),
        ("qwen2.attention.layer_norm_rms_epsilon", 6, rms_eps),
        ("qwen2.rope.freq_base", 6, rope_freq_base),
        ("tokenizer.ggml.model", 8, "gpt2"),
        ("tokenizer.ggml.pre", 8, "qwen2"),
        ("tokenizer.ggml.tokens", 9, (8, tok_strs)),
        ("tokenizer.ggml.token_type", 9, (5, tok_types)),
        ("tokenizer.ggml.merges", 9, (8, [])),
        ("tokenizer.ggml.eos_token_id", 4, 2),
        ("tokenizer.ggml.padding_token_id", 4, 0),
    ]

    tensors: list[tuple[str, tuple[int, ...], int, bytes]] = [
        ("token_embd.weight", (n_embd, vocab_size), emb_type, emb_raw),
        ("output_norm.weight", (n_embd,), 0, output_norm.tobytes()),
    ]
    if out_entry is not None:
        tensors.append(out_entry)
    if output_bias is not None:
        tensors.append(("output.bias", (vocab_size,), 0, output_bias.tobytes()))

    layers = []
    for l in range(n_layers):
        p = f"blk.{l}"
        attn_norm = rng.uniform(0.8, 1.2, (n_embd,))
        ffn_norm = rng.uniform(0.8, 1.2, (n_embd,))
        wq_t, wq_r, wq_d = make_mat(q_dim, n_embd)
        wk_t, wk_r, wk_d = make_mat(kv_dim, n_embd)
        wv_t, wv_r, wv_d = make_mat(kv_dim, n_embd)
        wo_t, wo_r, wo_d = make_mat(n_embd, q_dim)
        wg_t, wg_r, wg_d = make_mat(n_ff, n_embd)
        wu_t, wu_r, wu_d = make_mat(n_ff, n_embd)
        wd_t, wd_r, wd_d = make_mat(n_embd, n_ff)
        # Deliberately non-trivial biases (`[-0.6, +0.6]`) so omitting any bias fails immediately!
        bq = rng.uniform(-0.6, 0.6, (q_dim,))
        bk = rng.uniform(-0.6, 0.6, (kv_dim,))
        bv = rng.uniform(-0.6, 0.6, (kv_dim,))
        bo = rng.uniform(-0.3, 0.3, (n_embd,)) if with_wo_bias else None

        tensors.extend(
            [
                (f"{p}.attn_norm.weight", (n_embd,), 0, attn_norm.tobytes()),
                (f"{p}.ffn_norm.weight", (n_embd,), 0, ffn_norm.tobytes()),
                (f"{p}.attn_q.weight", (n_embd, q_dim), wq_t, wq_r),
                (f"{p}.attn_k.weight", (n_embd, kv_dim), wk_t, wk_r),
                (f"{p}.attn_v.weight", (n_embd, kv_dim), wv_t, wv_r),
                (f"{p}.attn_q.bias", (q_dim,), 0, bq.tobytes()),
                (f"{p}.attn_k.bias", (kv_dim,), 0, bk.tobytes()),
                (f"{p}.attn_v.bias", (kv_dim,), 0, bv.tobytes()),
                (f"{p}.attn_output.weight", (q_dim, n_embd), wo_t, wo_r),
                (f"{p}.ffn_gate.weight", (n_embd, n_ff), wg_t, wg_r),
                (f"{p}.ffn_up.weight", (n_embd, n_ff), wu_t, wu_r),
                (f"{p}.ffn_down.weight", (n_ff, n_embd), wd_t, wd_r),
            ]
        )
        if bo is not None:
            tensors.append((f"{p}.attn_output.bias", (n_embd,), 0, bo.tobytes()))

        layers.append(
            {
                "attn_norm": attn_norm,
                "ffn_norm": ffn_norm,
                "wq": wq_d,
                "wk": wk_d,
                "wv": wv_d,
                "bq": bq,
                "bk": bk,
                "bv": bv,
                "wo": wo_d,
                "bo": bo,
                "wg": wg_d,
                "wu": wu_d,
                "wd": wd_d,
            }
        )

    write_gguf_v3(out_gguf, kv_items, tensors)

    # Exact FP64 reference forward (matching F16 KV cache rounding!)
    k_cache = [np.zeros((len(tokens), n_head_kv, head_dim), dtype=np.float16) for _ in range(n_layers)]
    v_cache = [np.zeros((len(tokens), n_head_kv, head_dim), dtype=np.float16) for _ in range(n_layers)]
    all_logits = np.zeros((len(tokens), vocab_size), dtype=np.float32)
    group = n_head // n_head_kv
    scale_attn = 1.0 / math.sqrt(float(head_dim))

    def mv(w: np.ndarray, act: np.ndarray) -> np.ndarray:
        a = quantize_q8_0_f64(act) if (use_q4 and act_q8) else act.astype(np.float64)
        return w.astype(np.float64) @ a

    for pos, tid in enumerate(tokens):
        x = emb_deq[tid].astype(np.float64)
        for l_idx, lw in enumerate(layers):
            xb = rms_norm_f64(x, lw["attn_norm"], rms_eps)
            q = mv(lw["wq"], xb) + lw["bq"].astype(np.float64)
            k = mv(lw["wk"], xb) + lw["bk"].astype(np.float64)
            v = mv(lw["wv"], xb) + lw["bv"].astype(np.float64)

            q = rope_neox_f64(q, n_head, head_dim, head_dim, pos, rope_freq_base)
            k = rope_neox_f64(k, n_head_kv, head_dim, head_dim, pos, rope_freq_base)

            k_cache[l_idx][pos] = k.reshape(n_head_kv, head_dim).astype(np.float16)
            v_cache[l_idx][pos] = v.reshape(n_head_kv, head_dim).astype(np.float16)

            q_h = q.reshape(n_head, head_dim)
            attn_out = np.zeros((n_head, head_dim), dtype=np.float64)
            for h in range(n_head):
                kv_h = h // group
                kh = k_cache[l_idx][: pos + 1, kv_h, :].astype(np.float64)
                vh = v_cache[l_idx][: pos + 1, kv_h, :].astype(np.float64)
                scores = (kh @ q_h[h]) * scale_attn
                scores -= np.max(scores)
                probs = np.exp(scores)
                probs /= np.sum(probs)
                attn_out[h] = probs @ vh

            x = x + mv(lw["wo"], attn_out.reshape(-1))
            if lw["bo"] is not None:
                x = x + lw["bo"].astype(np.float64)

            xb_ffn = rms_norm_f64(x, lw["ffn_norm"], rms_eps)
            gate = mv(lw["wg"], xb_ffn)
            up = mv(lw["wu"], xb_ffn)
            silu_gate = gate / (1.0 + np.exp(-gate))
            x = x + mv(lw["wd"], silu_gate * up)

        xn = rms_norm_f64(x, output_norm, rms_eps)
        logits = mv(out_deq, xn)
        if output_bias is not None:
            logits = logits + output_bias.astype(np.float64)
        all_logits[pos] = logits.astype(np.float32)

    if out_tmlg is not None:
        out_tmlg.parent.mkdir(parents=True, exist_ok=True)
        with open(out_tmlg, "wb") as f:
            f.write(b"TMLG")
            f.write(struct.pack("<III", 1, len(tokens), vocab_size))
            for tid in tokens:
                f.write(struct.pack("<I", tid))
            f.write(all_logits.tobytes())

    return all_logits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-gguf", type=Path, required=True)
    ap.add_argument("--out-tmlg", type=Path, default=None)
    ap.add_argument("--tokens", type=str, default="1,5,12,7,19,3")
    ap.add_argument("--quant", choices=["f32", "q4_0"], default="f32")
    ap.add_argument("--act-q8", action="store_true")
    ap.add_argument("--untied", action="store_true")
    ap.add_argument("--no-wo-bias", action="store_true")
    ap.add_argument("--no-output-bias", action="store_true")
    ap.add_argument("--seed", type=int, default=20261123)
    args = ap.parse_args()

    toks = [int(x) for x in args.tokens.split(",") if x.strip()]
    build_and_eval_qwen2(
        args.out_gguf,
        args.out_tmlg,
        toks,
        quant=args.quant,
        act_q8=args.act_q8,
        tied_output=not args.untied,
        with_wo_bias=not args.no_wo_bias,
        with_output_bias=not args.no_output_bias,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
