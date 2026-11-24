#!/usr/bin/env python3
"""Independent FP64 mathematical oracle & synthetic GGUF v3 builder for P0 sub-8B architectures:
  - `smollm3`  : Norm RoPE (`5e6`), NoPE every 4th layer (`(il + 1) % 4 != 0`), GQA, SwiGLU
  - `phi3`     : Fused `attn_qkv.weight` + `attn_qkv.bias`, fused `ffn_up.weight` (`2*n_ff`),
                 NeoX partial RoPE + LongRoPE (`rope_factors_short` / `rope_factors_long`)
  - `mistral`  : Norm RoPE + causal `sliding_window` SWA, GQA, SwiGLU
  - `mistral3` : Norm RoPE + causal `sliding_window` SWA, GQA, SwiGLU
  - `gemma2`   : Alternating SWA (pattern 2), no QK-Norm, no V-Norm, `attn_logit_softcapping`
                 + `final_logit_softcapping`, 4 sandwich norms, GeGLU, `LlamaSpm` tokenizer
  - `gemma3`   : 5:1 SWA (pattern 6), dual RoPE (`1e6` / `1e4`), QK-Norm, no V-Norm,
                 `attn_scale = 1/sqrt(head_dim)`, 4 sandwich norms, GeGLU
  - `lfm3`     : 3-way hybrid (`ShortConv` + `sliding_attention` + `full_attention`), dual RoPE
                 (`1e6` + `+inf` partial rotary `rope_freqs` on full vs `1e4` on SWA), per-head
                 sigmoid `attn_gate.weight`, and `post_attention_norm` + `post_ffw_norm`
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


def quantize_q8_0_f64(vec: np.ndarray) -> np.ndarray:
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


def geglu_f64(gate: np.ndarray, up: np.ndarray) -> np.ndarray:
    g = gate.astype(np.float64)
    u = up.astype(np.float64)
    c = math.sqrt(2.0 / math.pi)
    gelu = 0.5 * g * (1.0 + np.tanh(c * (g + 0.044715 * g * g * g)))
    return gelu * u


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


def default_spm_tokenizer_kv(vocab_size: int) -> list[tuple[str, int, object]]:
    base = [
        "<pad>",
        "<eos>",
        "<bos>",
        "<unk>",
        "\u2581",
        "h",
        "e",
        "l",
        "o",
        "\u2581h",
        "he",
        "ll",
        "\u2581he",
        "\u2581hell",
        "\u2581hello",
    ]
    tok_strs = list(base) + [f"sp{i}" for i in range(len(base), vocab_size)]
    tok_types = [3, 3, 3, 2] + [1] * (vocab_size - 4)
    scores = [0.0, 0.0, 0.0, 0.0, -5.0, -5.0, -5.0, -5.0, -5.0, -4.0, -4.0, -3.5, -3.0, -2.0, -1.0]
    scores += [-10.0 - 0.01 * i for i in range(len(scores), vocab_size)]
    return [
        ("tokenizer.ggml.model", 8, "llama"),
        ("tokenizer.ggml.tokens", 9, (8, tok_strs)),
        ("tokenizer.ggml.token_type", 9, (5, tok_types)),
        ("tokenizer.ggml.scores", 9, (6, scores)),
        ("tokenizer.ggml.bos_token_id", 4, 2),
        ("tokenizer.ggml.eos_token_id", 4, 1),
    ]


def build_llama_family(
    arch: str,
    out_gguf: Path,
    out_tmlg: Path | None,
    tokens: list[int],
    *,
    quant: str,
    act_q8: bool,
    eval_ctx: int,
    seed: int,
) -> np.ndarray:
    rng = SplitMix64(seed)
    use_q4 = quant == "q4_0"
    head_dim = 32
    n_head = 4
    n_head_kv = 2
    n_embd = n_head * head_dim  # 128 (divisible by 32 for Q4_0)
    n_ff = 256
    q_dim = n_head * head_dim
    kv_dim = n_head_kv * head_dim
    vocab_size = 128

    if arch == "smollm3":
        n_layers = 4  # layer 3 exercises `(il + 1) % 4 == 0` -> NoPE!
        rope_dim = head_dim
        rope_freq_base = 5_000_000.0
        rms_eps = 1e-6
        neox = False
        sliding_window = None
        use_rope = [(il + 1) % 4 != 0 for il in range(n_layers)]
        rope_orig_ctx = 512
    elif arch == "phi3":
        n_layers = 2
        rope_dim = head_dim // 2  # partial rotary 50%
        rope_freq_base = 10_000.0
        rms_eps = 1e-5
        neox = True
        sliding_window = None
        use_rope = [True] * n_layers
        rope_orig_ctx = 48
    elif arch in ("mistral", "mistral3"):
        n_layers = 2
        rope_dim = head_dim
        rope_freq_base = 10_000.0
        rms_eps = 1e-5
        neox = False
        sliding_window = 3
        use_rope = [True] * n_layers
        rope_orig_ctx = 512
    else:
        raise ValueError(f"unexpected llama-family arch {arch}")

    def make_mat(rows: int, cols: int, scale: float = 0.16):
        m = rng.uniform(-scale, scale, (rows, cols))
        if use_q4:
            raw, deq = quantize_q4_0_rows(m)
            return 2, raw, deq
        return 0, m.tobytes(), m

    emb_t, emb_r, emb_d = make_mat(vocab_size, n_embd, 0.20)
    out_t, out_r, out_d = make_mat(vocab_size, n_embd, 0.18)
    output_norm = rng.uniform(0.8, 1.2, (n_embd,))

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
    if sliding_window is not None:
        kv_items.append((f"{arch}.attention.sliding_window", 4, sliding_window))
    if arch == "phi3":
        kv_items.append((f"{arch}.rope.scaling.original_context_length", 4, rope_orig_ctx))
        # Also include `phi3.attention.sliding_window` to verify `phi3` ignores it like `phi3.cpp`!
        kv_items.append((f"{arch}.attention.sliding_window", 4, 2))
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
    if arch == "phi3":
        f_short = rng.uniform(1.0, 1.6, (half,))
        f_long = rng.uniform(2.5, 6.5, (half,))
        tensors.append(("rope_factors_short.weight", (half,), 0, f_short.tobytes()))
        tensors.append(("rope_factors_long.weight", (half,), 0, f_long.tobytes()))
        chosen = f_long if eval_ctx > rope_orig_ctx else f_short
        inv_freqs = base_inv / chosen.astype(np.float64)
    else:
        inv_freqs = base_inv

    layers = []
    for l in range(n_layers):
        p = f"blk.{l}"
        attn_norm = rng.uniform(0.8, 1.2, (n_embd,))
        ffn_norm = rng.uniform(0.8, 1.2, (n_embd,))
        wo_t, wo_r, wo_d = make_mat(n_embd, q_dim)
        wd_t, wd_r, wd_d = make_mat(n_embd, n_ff)
        tensors.append((f"{p}.attn_norm.weight", (n_embd,), 0, attn_norm.tobytes()))
        tensors.append((f"{p}.ffn_norm.weight", (n_embd,), 0, ffn_norm.tobytes()))
        tensors.append((f"{p}.attn_output.weight", (q_dim, n_embd), wo_t, wo_r))
        tensors.append((f"{p}.ffn_down.weight", (n_ff, n_embd), wd_t, wd_r))

        if arch == "phi3":
            qkv_t, qkv_r, qkv_d = make_mat(q_dim + 2 * kv_dim, n_embd)
            qkv_b = rng.uniform(-0.35, 0.35, (q_dim + 2 * kv_dim,))
            fup_t, fup_r, fup_d = make_mat(2 * n_ff, n_embd)
            tensors.append((f"{p}.attn_qkv.weight", (n_embd, q_dim + 2 * kv_dim), qkv_t, qkv_r))
            tensors.append((f"{p}.attn_qkv.bias", (q_dim + 2 * kv_dim,), 0, qkv_b.tobytes()))
            tensors.append((f"{p}.ffn_up.weight", (n_embd, 2 * n_ff), fup_t, fup_r))
            wq_d = qkv_d[:q_dim]
            wk_d = qkv_d[q_dim : q_dim + kv_dim]
            wv_d = qkv_d[q_dim + kv_dim :]
            bq = qkv_b[:q_dim]
            bk = qkv_b[q_dim : q_dim + kv_dim]
            bv = qkv_b[q_dim + kv_dim :]
            wg_d = fup_d[:n_ff]
            wu_d = fup_d[n_ff:]
        else:
            wq_t, wq_r, wq_d = make_mat(q_dim, n_embd)
            wk_t, wk_r, wk_d = make_mat(kv_dim, n_embd)
            wv_t, wv_r, wv_d = make_mat(kv_dim, n_embd)
            wg_t, wg_r, wg_d = make_mat(n_ff, n_embd)
            wu_t, wu_r, wu_d = make_mat(n_ff, n_embd)
            tensors.extend(
                [
                    (f"{p}.attn_q.weight", (n_embd, q_dim), wq_t, wq_r),
                    (f"{p}.attn_k.weight", (n_embd, kv_dim), wk_t, wk_r),
                    (f"{p}.attn_v.weight", (n_embd, kv_dim), wv_t, wv_r),
                    (f"{p}.ffn_gate.weight", (n_embd, n_ff), wg_t, wg_r),
                    (f"{p}.ffn_up.weight", (n_embd, n_ff), wu_t, wu_r),
                ]
            )
            bq = bk = bv = None

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
                "wg": wg_d,
                "wu": wu_d,
                "wd": wd_d,
            }
        )

    write_gguf_v3(out_gguf, kv_items, tensors)

    k_cache = [np.zeros((len(tokens), n_head_kv, head_dim), dtype=np.float16) for _ in range(n_layers)]
    v_cache = [np.zeros((len(tokens), n_head_kv, head_dim), dtype=np.float16) for _ in range(n_layers)]
    all_logits = np.zeros((len(tokens), vocab_size), dtype=np.float32)
    group = n_head // n_head_kv
    scale_attn = float(np.float32(1.0 / math.sqrt(float(head_dim))))

    def mv(w: np.ndarray, act: np.ndarray) -> np.ndarray:
        a = quantize_q8_0_f64(act) if (use_q4 and act_q8) else act.astype(np.float64)
        return w.astype(np.float64) @ a

    for pos, tid in enumerate(tokens):
        x = emb_d[tid].astype(np.float64)
        for l_idx, lw in enumerate(layers):
            xb = rms_norm_f64(x, lw["attn_norm"], rms_eps)
            q = mv(lw["wq"], xb)
            k = mv(lw["wk"], xb)
            v = mv(lw["wv"], xb)
            if lw["bq"] is not None:
                q = q + lw["bq"].astype(np.float64)
                k = k + lw["bk"].astype(np.float64)
                v = v + lw["bv"].astype(np.float64)

            if use_rope[l_idx]:
                q = rope_apply_f64(q, n_head, head_dim, rope_dim, pos, inv_freqs, neox)
                k = rope_apply_f64(k, n_head_kv, head_dim, rope_dim, pos, inv_freqs, neox)

            k_cache[l_idx][pos] = k.reshape(n_head_kv, head_dim).astype(np.float16)
            v_cache[l_idx][pos] = v.reshape(n_head_kv, head_dim).astype(np.float16)

            start_kv = max(0, pos + 1 - sliding_window) if sliding_window is not None else 0
            q_h = q.reshape(n_head, head_dim)
            attn_out = np.zeros((n_head, head_dim), dtype=np.float64)
            for h in range(n_head):
                kv_h = h // group
                kh = k_cache[l_idx][start_kv : pos + 1, kv_h, :].astype(np.float64)
                vh = v_cache[l_idx][start_kv : pos + 1, kv_h, :].astype(np.float64)
                scores = (kh @ q_h[h]) * scale_attn
                scores -= np.max(scores)
                probs = np.exp(scores)
                probs /= np.sum(probs)
                attn_out[h] = probs @ vh

            x = x + mv(lw["wo"], attn_out.reshape(-1))
            xb_ffn = rms_norm_f64(x, lw["ffn_norm"], rms_eps)
            gate = mv(lw["wg"], xb_ffn)
            up = mv(lw["wu"], xb_ffn)
            silu_gate = gate / (1.0 + np.exp(-gate))
            x = x + mv(lw["wd"], silu_gate * up)

        xn = rms_norm_f64(x, output_norm, rms_eps)
        all_logits[pos] = mv(out_d, xn).astype(np.float32)

    write_tmlg(out_tmlg, tokens, vocab_size, all_logits)
    return all_logits


def build_gemma_family(
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
    head_dim = 32
    n_head = 4
    n_head_kv = 2
    n_embd = 128
    n_ff = 256
    q_dim = n_head * head_dim
    kv_dim = n_head_kv * head_dim
    vocab_size = 128
    rms_eps = 1e-6
    sliding_window = 3

    if arch == "gemma2":
        n_layers = 4  # default pattern 2 -> [True, False, True, False]
        is_swa = [(il + 1) % 2 != 0 for il in range(n_layers)]
        freq_full = 10_000.0
        freq_swa = 10_000.0
        has_qk_norm = False
        attn_softcap = 5.0  # tight cap so `tanh(s / 5) * 5` strongly saturates!
        final_softcap = 30.0
    elif arch == "gemma3":
        n_layers = 6  # default pattern 6 -> [True, True, True, True, True, False]
        is_swa = [(il + 1) % 6 != 0 for il in range(n_layers)]
        freq_full = 1_000_000.0
        freq_swa = 10_000.0
        has_qk_norm = True
        attn_softcap = None
        final_softcap = None
    else:
        raise ValueError(f"unexpected gemma arch {arch}")

    def make_mat(rows: int, cols: int, scale: float = 0.16):
        m = rng.uniform(-scale, scale, (rows, cols))
        if use_q4:
            raw, deq = quantize_q4_0_rows(m)
            return 2, raw, deq
        return 0, m.tobytes(), m

    emb_t, emb_r, emb_d = make_mat(vocab_size, n_embd, 0.18)
    output_norm = rng.uniform(0.8, 1.2, (n_embd,))

    kv_items: list[tuple[str, int, object]] = [
        ("general.architecture", 8, arch),
        (f"{arch}.block_count", 4, n_layers),
        (f"{arch}.context_length", 4, 512),
        (f"{arch}.embedding_length", 4, n_embd),
        (f"{arch}.feed_forward_length", 4, n_ff),
        (f"{arch}.attention.head_count", 4, n_head),
        (f"{arch}.attention.head_count_kv", 4, n_head_kv),
        (f"{arch}.attention.key_length", 4, head_dim),
        (f"{arch}.attention.sliding_window", 4, sliding_window),
        (f"{arch}.attention.layer_norm_rms_epsilon", 6, rms_eps),
        (f"{arch}.rope.freq_base", 6, freq_full),
    ]
    if arch == "gemma3":
        kv_items.append((f"{arch}.rope.freq_base_swa", 6, freq_swa))
    if attn_softcap is not None:
        kv_items.append((f"{arch}.attn_logit_softcapping", 6, attn_softcap))
    if final_softcap is not None:
        kv_items.append((f"{arch}.final_logit_softcapping", 6, final_softcap))
    kv_items.extend(default_spm_tokenizer_kv(vocab_size))

    tensors: list[tuple[str, tuple[int, ...], int, bytes]] = [
        ("token_embd.weight", (n_embd, vocab_size), emb_t, emb_r),
        ("output_norm.weight", (n_embd,), 0, output_norm.tobytes()),
    ]

    layers = []
    for l in range(n_layers):
        p = f"blk.{l}"
        an = rng.uniform(0.8, 1.2, (n_embd,))
        apn = rng.uniform(0.8, 1.2, (n_embd,))
        fn = rng.uniform(0.8, 1.2, (n_embd,))
        fpn = rng.uniform(0.8, 1.2, (n_embd,))
        qn = rng.uniform(0.8, 1.2, (head_dim,)) if has_qk_norm else None
        kn = rng.uniform(0.8, 1.2, (head_dim,)) if has_qk_norm else None
        wq_t, wq_r, wq_d = make_mat(q_dim, n_embd, 0.35 if attn_softcap else 0.16)
        wk_t, wk_r, wk_d = make_mat(kv_dim, n_embd, 0.35 if attn_softcap else 0.16)
        wv_t, wv_r, wv_d = make_mat(kv_dim, n_embd, 0.18)
        wo_t, wo_r, wo_d = make_mat(n_embd, q_dim, 0.16)
        wg_t, wg_r, wg_d = make_mat(n_ff, n_embd, 0.16)
        wu_t, wu_r, wu_d = make_mat(n_ff, n_embd, 0.16)
        wd_t, wd_r, wd_d = make_mat(n_embd, n_ff, 0.16)
        tensors.extend(
            [
                (f"{p}.attn_norm.weight", (n_embd,), 0, an.tobytes()),
                (f"{p}.post_attention_norm.weight", (n_embd,), 0, apn.tobytes()),
                (f"{p}.ffn_norm.weight", (n_embd,), 0, fn.tobytes()),
                (f"{p}.post_ffw_norm.weight", (n_embd,), 0, fpn.tobytes()),
                (f"{p}.attn_q.weight", (n_embd, q_dim), wq_t, wq_r),
                (f"{p}.attn_k.weight", (n_embd, kv_dim), wk_t, wk_r),
                (f"{p}.attn_v.weight", (n_embd, kv_dim), wv_t, wv_r),
                (f"{p}.attn_output.weight", (q_dim, n_embd), wo_t, wo_r),
                (f"{p}.ffn_gate.weight", (n_embd, n_ff), wg_t, wg_r),
                (f"{p}.ffn_up.weight", (n_embd, n_ff), wu_t, wu_r),
                (f"{p}.ffn_down.weight", (n_ff, n_embd), wd_t, wd_r),
            ]
        )
        if qn is not None and kn is not None:
            tensors.append((f"{p}.attn_q_norm.weight", (head_dim,), 0, qn.tobytes()))
            tensors.append((f"{p}.attn_k_norm.weight", (head_dim,), 0, kn.tobytes()))
        layers.append(
            {
                "an": an,
                "apn": apn,
                "fn": fn,
                "fpn": fpn,
                "qn": qn,
                "kn": kn,
                "wq": wq_d,
                "wk": wk_d,
                "wv": wv_d,
                "wo": wo_d,
                "wg": wg_d,
                "wu": wu_d,
                "wd": wd_d,
            }
        )

    write_gguf_v3(out_gguf, kv_items, tensors)

    half = head_dim // 2
    inv_full = np.array([float(freq_full) ** (-2.0 * i / head_dim) for i in range(half)], dtype=np.float64)
    inv_swa = np.array([float(freq_swa) ** (-2.0 * i / head_dim) for i in range(half)], dtype=np.float64)
    k_cache = [np.zeros((len(tokens), n_head_kv, head_dim), dtype=np.float16) for _ in range(n_layers)]
    v_cache = [np.zeros((len(tokens), n_head_kv, head_dim), dtype=np.float16) for _ in range(n_layers)]
    all_logits = np.zeros((len(tokens), vocab_size), dtype=np.float32)
    group = n_head // n_head_kv
    attn_scale = float(np.float32(1.0 / math.sqrt(float(head_dim))))
    emb_scale = float(np.float32(math.sqrt(float(n_embd))))

    def mv(w: np.ndarray, act: np.ndarray) -> np.ndarray:
        a = quantize_q8_0_f64(act) if (use_q4 and act_q8) else act.astype(np.float64)
        return w.astype(np.float64) @ a

    for pos, tid in enumerate(tokens):
        x = (emb_d[tid] * np.float32(emb_scale)).astype(np.float64)
        for l_idx, lw in enumerate(layers):
            xb = rms_norm_f64(x, lw["an"], rms_eps)
            q = mv(lw["wq"], xb)
            k = mv(lw["wk"], xb)
            v = mv(lw["wv"], xb)  # NOTE: Gemma 2 & Gemma 3 do NOT apply unit V-Norm!

            if lw["qn"] is not None:
                qh = q.reshape(n_head, head_dim)
                for h in range(n_head):
                    qh[h] = rms_norm_f64(qh[h], lw["qn"], rms_eps)
                q = qh.reshape(-1)
            if lw["kn"] is not None:
                kh = k.reshape(n_head_kv, head_dim)
                for h in range(n_head_kv):
                    kh[h] = rms_norm_f64(kh[h], lw["kn"], rms_eps)
                k = kh.reshape(-1)

            inv = inv_swa if is_swa[l_idx] else inv_full
            q = rope_apply_f64(q, n_head, head_dim, head_dim, pos, inv, True)
            k = rope_apply_f64(k, n_head_kv, head_dim, head_dim, pos, inv, True)

            k_cache[l_idx][pos] = k.reshape(n_head_kv, head_dim).astype(np.float16)
            v_cache[l_idx][pos] = v.reshape(n_head_kv, head_dim).astype(np.float16)

            start_kv = max(0, pos + 1 - sliding_window) if is_swa[l_idx] else 0
            q_h = q.reshape(n_head, head_dim)
            attn_out = np.zeros((n_head, head_dim), dtype=np.float64)
            for h in range(n_head):
                kv_h = h // group
                kh = k_cache[l_idx][start_kv : pos + 1, kv_h, :].astype(np.float64)
                vh = v_cache[l_idx][start_kv : pos + 1, kv_h, :].astype(np.float64)
                scores = (kh @ q_h[h]) * attn_scale
                if attn_softcap is not None:
                    cap = float(attn_softcap)
                    scores = np.tanh(scores / cap) * cap
                scores -= np.max(scores)
                probs = np.exp(scores)
                probs /= np.sum(probs)
                attn_out[h] = probs @ vh

            attn_proj = rms_norm_f64(mv(lw["wo"], attn_out.reshape(-1)), lw["apn"], rms_eps)
            x = x + attn_proj
            xb_ffn = rms_norm_f64(x, lw["fn"], rms_eps)
            ffn_act = geglu_f64(mv(lw["wg"], xb_ffn), mv(lw["wu"], xb_ffn))
            ffn_proj = rms_norm_f64(mv(lw["wd"], ffn_act), lw["fpn"], rms_eps)
            x = x + ffn_proj

        xn = rms_norm_f64(x, output_norm, rms_eps)
        logits = mv(emb_d, xn)
        if final_softcap is not None:
            fc = float(final_softcap)
            logits = np.tanh(logits / fc) * fc
        all_logits[pos] = logits.astype(np.float32)

    write_tmlg(out_tmlg, tokens, vocab_size, all_logits)
    return all_logits


def build_lfm3(
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
    n_layers = 3  # 0: ShortConv, 1: sliding_attention (w=3), 2: full_attention (+inf partial rotary)
    head_dim = 32
    n_head = 4
    n_embd = 128
    n_ff = 256
    q_dim = n_head * head_dim
    n_head_kv_arr = [0, 2, 2]
    is_swa = [False, True, False]
    l_cache = 4
    sliding_window = 3
    freq_full = 1_000_000.0
    freq_swa = 10_000.0
    rms_eps = 1e-5
    vocab_size = 128

    def make_mat(rows: int, cols: int, scale: float = 0.16):
        m = rng.uniform(-scale, scale, (rows, cols))
        if use_q4:
            raw, deq = quantize_q4_0_rows(m)
            return 2, raw, deq
        return 0, m.tobytes(), m

    emb_t, emb_r, emb_d = make_mat(vocab_size, n_embd, 0.20)
    output_norm = rng.uniform(0.8, 1.2, (n_embd,))

    kv_items: list[tuple[str, int, object]] = [
        ("general.architecture", 8, "lfm3"),
        ("lfm3.block_count", 4, n_layers),
        ("lfm3.context_length", 4, 512),
        ("lfm3.embedding_length", 4, n_embd),
        ("lfm3.feed_forward_length", 4, n_ff),
        ("lfm3.attention.head_count", 4, n_head),
        ("lfm3.attention.head_count_kv", 9, (4, n_head_kv_arr)),
        ("lfm3.attention.key_length", 4, head_dim),
        ("lfm3.attention.sliding_window", 4, sliding_window),
        ("lfm3.attention.sliding_window_pattern", 9, (7, is_swa)),
        ("lfm3.shortconv.l_cache", 4, l_cache),
        ("lfm3.attention.layer_norm_rms_epsilon", 6, rms_eps),
        ("lfm3.rope.dimension_count", 4, head_dim),
        ("lfm3.rope.freq_base", 6, freq_full),
        ("lfm3.rope.freq_base_swa", 6, freq_swa),
    ]
    kv_items.extend(default_gpt2_tokenizer_kv(vocab_size))

    tensors: list[tuple[str, tuple[int, ...], int, bytes]] = [
        ("token_embd.weight", (n_embd, vocab_size), emb_t, emb_r),
        ("token_embd_norm.weight", (n_embd,), 0, output_norm.tobytes()),
    ]

    half = head_dim // 2
    # Upper half `+inf` -> `inv_freq == 0.0` (exact `partial_rotary_factor = 0.5` on full attention!)
    rope_freqs_full = np.array(
        [1.0 if i < half // 2 else float("inf") for i in range(half)], dtype=np.float32
    )
    tensors.append(("blk.2.rope_freqs.weight", (half,), 0, rope_freqs_full.tobytes()))

    layers = []
    for l in range(n_layers):
        p = f"blk.{l}"
        an = rng.uniform(0.8, 1.2, (n_embd,))
        pan = rng.uniform(0.8, 1.2, (n_embd,))
        fn = rng.uniform(0.8, 1.2, (n_embd,))
        pfn = rng.uniform(0.8, 1.2, (n_embd,))
        wg_t, wg_r, wg_d = make_mat(n_ff, n_embd)
        wu_t, wu_r, wu_d = make_mat(n_ff, n_embd)
        wd_t, wd_r, wd_d = make_mat(n_embd, n_ff)
        tensors.extend(
            [
                (f"{p}.attn_norm.weight", (n_embd,), 0, an.tobytes()),
                (f"{p}.post_attention_norm.weight", (n_embd,), 0, pan.tobytes()),
                (f"{p}.ffn_norm.weight", (n_embd,), 0, fn.tobytes()),
                (f"{p}.post_ffw_norm.weight", (n_embd,), 0, pfn.tobytes()),
                (f"{p}.ffn_gate.weight", (n_embd, n_ff), wg_t, wg_r),
                (f"{p}.ffn_up.weight", (n_embd, n_ff), wu_t, wu_r),
                (f"{p}.ffn_down.weight", (n_ff, n_embd), wd_t, wd_r),
            ]
        )
        if n_head_kv_arr[l] == 0:
            inp_t, inp_r, inp_d = make_mat(3 * n_embd, n_embd)
            conv_w = rng.uniform(-0.3, 0.3, (n_embd, l_cache))
            op_t, op_r, op_d = make_mat(n_embd, n_embd)
            tensors.extend(
                [
                    (f"{p}.shortconv.in_proj.weight", (n_embd, 3 * n_embd), inp_t, inp_r),
                    (f"{p}.shortconv.conv.weight", (l_cache, n_embd), 0, conv_w.tobytes()),
                    (f"{p}.shortconv.out_proj.weight", (n_embd, n_embd), op_t, op_r),
                ]
            )
            layers.append(
                {
                    "kind": "conv",
                    "an": an,
                    "pan": pan,
                    "fn": fn,
                    "pfn": pfn,
                    "inp": inp_d,
                    "conv": conv_w,
                    "op": op_d,
                    "wg": wg_d,
                    "wu": wu_d,
                    "wd": wd_d,
                }
            )
        else:
            hkv = n_head_kv_arr[l]
            kv_dim = hkv * head_dim
            qn = rng.uniform(0.8, 1.2, (head_dim,))
            kn = rng.uniform(0.8, 1.2, (head_dim,))
            wq_t, wq_r, wq_d = make_mat(q_dim, n_embd)
            wk_t, wk_r, wk_d = make_mat(kv_dim, n_embd)
            wv_t, wv_r, wv_d = make_mat(kv_dim, n_embd)
            ag_t, ag_r, ag_d = make_mat(n_head, n_embd, 0.45)
            wo_t, wo_r, wo_d = make_mat(n_embd, q_dim)
            tensors.extend(
                [
                    (f"{p}.attn_q_norm.weight", (head_dim,), 0, qn.tobytes()),
                    (f"{p}.attn_k_norm.weight", (head_dim,), 0, kn.tobytes()),
                    (f"{p}.attn_q.weight", (n_embd, q_dim), wq_t, wq_r),
                    (f"{p}.attn_k.weight", (n_embd, kv_dim), wk_t, wk_r),
                    (f"{p}.attn_v.weight", (n_embd, kv_dim), wv_t, wv_r),
                    (f"{p}.attn_gate.weight", (n_embd, n_head), ag_t, ag_r),
                    (f"{p}.attn_output.weight", (q_dim, n_embd), wo_t, wo_r),
                ]
            )
            layers.append(
                {
                    "kind": "gqa",
                    "hkv": hkv,
                    "an": an,
                    "pan": pan,
                    "fn": fn,
                    "pfn": pfn,
                    "qn": qn,
                    "kn": kn,
                    "wq": wq_d,
                    "wk": wk_d,
                    "wv": wv_d,
                    "ag": ag_d,
                    "wo": wo_d,
                    "wg": wg_d,
                    "wu": wu_d,
                    "wd": wd_d,
                }
            )

    write_gguf_v3(out_gguf, kv_items, tensors)

    inv_full = np.array(
        [
            0.0 if math.isinf(float(rope_freqs_full[i])) else (float(freq_full) ** (-2.0 * i / head_dim)) / float(rope_freqs_full[i])
            for i in range(half)
        ],
        dtype=np.float64,
    )
    inv_swa = np.array([float(freq_swa) ** (-2.0 * i / head_dim) for i in range(half)], dtype=np.float64)

    hist_len = l_cache - 1
    conv_hist = [np.zeros((n_embd, hist_len), dtype=np.float64) for _ in range(n_layers)]
    k_cache = [
        np.zeros((len(tokens), max(1, n_head_kv_arr[l]), head_dim), dtype=np.float16)
        for l in range(n_layers)
    ]
    v_cache = [
        np.zeros((len(tokens), max(1, n_head_kv_arr[l]), head_dim), dtype=np.float16)
        for l in range(n_layers)
    ]
    all_logits = np.zeros((len(tokens), vocab_size), dtype=np.float32)
    attn_scale = float(np.float32(1.0 / math.sqrt(float(head_dim))))

    def mv(w: np.ndarray, act: np.ndarray) -> np.ndarray:
        a = quantize_q8_0_f64(act) if (use_q4 and act_q8) else act.astype(np.float64)
        return w.astype(np.float64) @ a

    for pos, tid in enumerate(tokens):
        x = emb_d[tid].astype(np.float64)
        for l_idx, lw in enumerate(layers):
            xb = rms_norm_f64(x, lw["an"], rms_eps)
            if lw["kind"] == "conv":
                bcx = mv(lw["inp"], xb)
                b_vec = bcx[:n_embd]
                c_vec = bcx[n_embd : 2 * n_embd]
                x_vec = bcx[2 * n_embd :]
                bx = b_vec * x_vec
                w_conv = lw["conv"].astype(np.float64)
                csum = np.sum(conv_hist[l_idx] * w_conv[:, :hist_len], axis=1) + bx * w_conv[:, hist_len]
                conv_hist[l_idx][:, :-1] = conv_hist[l_idx][:, 1:]
                conv_hist[l_idx][:, -1] = bx
                op_raw = mv(lw["op"], c_vec * csum)
            else:
                hkv = lw["hkv"]
                group = n_head // hkv
                q = mv(lw["wq"], xb)
                k = mv(lw["wk"], xb)
                v = mv(lw["wv"], xb)
                gate_raw = mv(lw["ag"], xb)

                qh = q.reshape(n_head, head_dim)
                for h in range(n_head):
                    qh[h] = rms_norm_f64(qh[h], lw["qn"], rms_eps)
                kh_pre = k.reshape(hkv, head_dim)
                for h in range(hkv):
                    kh_pre[h] = rms_norm_f64(kh_pre[h], lw["kn"], rms_eps)

                inv = inv_swa if is_swa[l_idx] else inv_full
                q = rope_apply_f64(qh.reshape(-1), n_head, head_dim, head_dim, pos, inv, True)
                k = rope_apply_f64(kh_pre.reshape(-1), hkv, head_dim, head_dim, pos, inv, True)

                k_cache[l_idx][pos] = k.reshape(hkv, head_dim).astype(np.float16)
                v_cache[l_idx][pos] = v.reshape(hkv, head_dim).astype(np.float16)

                start_kv = max(0, pos + 1 - sliding_window) if is_swa[l_idx] else 0
                q_h = q.reshape(n_head, head_dim)
                attn_out = np.zeros((n_head, head_dim), dtype=np.float64)
                for h in range(n_head):
                    kv_h = h // group
                    kh = k_cache[l_idx][start_kv : pos + 1, kv_h, :].astype(np.float64)
                    vh = v_cache[l_idx][start_kv : pos + 1, kv_h, :].astype(np.float64)
                    scores = (kh @ q_h[h]) * attn_scale
                    scores -= np.max(scores)
                    probs = np.exp(scores)
                    probs /= np.sum(probs)
                    sig = 1.0 / (1.0 + math.exp(-float(gate_raw[h])))
                    attn_out[h] = (probs @ vh) * sig

                op_raw = mv(lw["wo"], attn_out.reshape(-1))

            x = x + rms_norm_f64(op_raw, lw["pan"], rms_eps)
            xb_ffn = rms_norm_f64(x, lw["fn"], rms_eps)
            gate = mv(lw["wg"], xb_ffn)
            up = mv(lw["wu"], xb_ffn)
            silu_gate = gate / (1.0 + np.exp(-gate))
            ffn_raw = mv(lw["wd"], silu_gate * up)
            x = x + rms_norm_f64(ffn_raw, lw["pfn"], rms_eps)

        xn = rms_norm_f64(x, output_norm, rms_eps)
        all_logits[pos] = mv(emb_d, xn).astype(np.float32)

    write_tmlg(out_tmlg, tokens, vocab_size, all_logits)
    return all_logits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--arch",
        required=True,
        choices=["smollm3", "phi3", "mistral", "mistral3", "gemma2", "gemma3", "lfm3"],
    )
    ap.add_argument("--out-gguf", type=Path, required=True)
    ap.add_argument("--out-tmlg", type=Path, default=None)
    ap.add_argument("--tokens", type=str, default="1,5,12,7,19,3,28,14")
    ap.add_argument("--quant", choices=["f32", "q4_0"], default="f32")
    ap.add_argument("--act-q8", action="store_true")
    ap.add_argument("--eval-ctx", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20261124)
    args = ap.parse_args()

    toks = [int(x) for x in args.tokens.split(",") if x.strip()]
    if args.arch in ("smollm3", "phi3", "mistral", "mistral3"):
        build_llama_family(
            args.arch,
            args.out_gguf,
            args.out_tmlg,
            toks,
            quant=args.quant,
            act_q8=args.act_q8,
            eval_ctx=args.eval_ctx,
            seed=args.seed,
        )
    elif args.arch in ("gemma2", "gemma3"):
        build_gemma_family(
            args.arch,
            args.out_gguf,
            args.out_tmlg,
            toks,
            quant=args.quant,
            act_q8=args.act_q8,
            seed=args.seed,
        )
    elif args.arch == "lfm3":
        build_lfm3(
            args.out_gguf,
            args.out_tmlg,
            toks,
            quant=args.quant,
            act_q8=args.act_q8,
            seed=args.seed,
        )


if __name__ == "__main__":
    main()
