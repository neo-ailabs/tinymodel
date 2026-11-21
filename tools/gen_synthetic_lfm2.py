#!/usr/bin/env python3
"""
Deterministic zero-dependency synthetic GGUF v3 generator for Liquid AI LFM2 (`lfm2`).

Generates a compact 4-layer hybrid `lfm2` GGUF (`ShortConv` + `GQA` with QK-Norm, NeoX RoPE,
`sliding_window = 8`, `Q4_0` linear/embedding matrices + `F32` norms/conv kernels, 260-token
GPT-2 BPE vocabulary) that loads natively in BOTH `llama.cpp` and `tinymodel` so multi-token
end-to-end logits alignment (`tools/logits_diff.py`, `AGENTS.md` Rule #7) can be verified
deterministically.
"""

from __future__ import annotations

import argparse
import math
import random
import struct
from pathlib import Path

GGML_TYPE_F32 = 0
GGML_TYPE_Q4_0 = 2

GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9


def f32_to_f16_bits(val: float) -> int:
    return struct.unpack("<H", struct.pack("<e", val))[0]


def quantize_q4_0(values: list[float]) -> bytes:
    assert len(values) % 32 == 0
    out = bytearray()
    for i in range(0, len(values), 32):
        blk = values[i : i + 32]
        amax = max(abs(x) for x in blk)
        d = amax / 7.0 if amax > 0.0 else 0.0
        d_bits = f32_to_f16_bits(d)
        d_f32 = struct.unpack("<e", struct.pack("<H", d_bits))[0]
        inv_d = 1.0 / d_f32 if d_f32 != 0.0 else 0.0
        out.extend(struct.pack("<H", d_bits))
        qs = bytearray(16)
        for j in range(16):
            v0 = int(round(blk[j] * inv_d)) + 8
            v1 = int(round(blk[j + 16] * inv_d)) + 8
            v0 = max(0, min(15, v0))
            v1 = max(0, min(15, v1))
            qs[j] = (v0 & 0x0F) | ((v1 & 0x0F) << 4)
        out.extend(qs)
    return bytes(out)


def pack_f32(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def gpt2_byte_to_unicode() -> dict[int, str]:
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


class GgufWriter:
    def __init__(self) -> None:
        self.kv: list[tuple[str, int, bytes]] = []
        self.tensors: list[tuple[str, list[int], int, bytes]] = []

    def _s(self, s: str) -> bytes:
        b = s.encode("utf-8")
        return struct.pack("<Q", len(b)) + b

    def add_str(self, key: str, val: str) -> None:
        self.kv.append((key, GGUF_TYPE_STRING, self._s(val)))

    def add_u32(self, key: str, val: int) -> None:
        self.kv.append((key, GGUF_TYPE_UINT32, struct.pack("<I", val)))

    def add_f32(self, key: str, val: float) -> None:
        self.kv.append((key, GGUF_TYPE_FLOAT32, struct.pack("<f", val)))

    def add_u32_arr(self, key: str, vals: list[int]) -> None:
        payload = struct.pack("<IQ", GGUF_TYPE_UINT32, len(vals)) + struct.pack(
            f"<{len(vals)}I", *vals
        )
        self.kv.append((key, GGUF_TYPE_ARRAY, payload))

    def add_i32_arr(self, key: str, vals: list[int]) -> None:
        payload = struct.pack("<IQ", GGUF_TYPE_INT32, len(vals)) + struct.pack(
            f"<{len(vals)}i", *vals
        )
        self.kv.append((key, GGUF_TYPE_ARRAY, payload))

    def add_str_arr(self, key: str, vals: list[str]) -> None:
        parts = [struct.pack("<IQ", GGUF_TYPE_STRING, len(vals))]
        for v in vals:
            parts.append(self._s(v))
        self.kv.append((key, GGUF_TYPE_ARRAY, b"".join(parts)))

    def add_tensor(self, name: str, dims: list[int], ggml_type: int, raw: bytes) -> None:
        self.tensors.append((name, dims, ggml_type, raw))

    def write(self, path: Path) -> None:
        alignment = 32
        buf = bytearray()
        buf.extend(b"GGUF")
        buf.extend(struct.pack("<IQQ", 3, len(self.tensors), len(self.kv)))
        for k, vtype, vbytes in self.kv:
            buf.extend(self._s(k))
            buf.extend(struct.pack("<I", vtype))
            buf.extend(vbytes)

        offsets: list[int] = []
        cur = 0
        for _, _, _, raw in self.tensors:
            cur = (cur + alignment - 1) // alignment * alignment
            offsets.append(cur)
            cur += len(raw)

        for (name, dims, ggml_type, _), off in zip(self.tensors, offsets):
            buf.extend(self._s(name))
            buf.extend(struct.pack("<I", len(dims)))
            for d in dims:
                buf.extend(struct.pack("<Q", d))
            buf.extend(struct.pack("<IQ", ggml_type, off))

        pad = (alignment - (len(buf) % alignment)) % alignment
        buf.extend(b"\x00" * pad)
        data_base = len(buf)
        for (_, _, _, raw), off in zip(self.tensors, offsets):
            cur_rel = len(buf) - data_base
            if cur_rel < off:
                buf.extend(b"\x00" * (off - cur_rel))
            buf.extend(raw)

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(bytes(buf))


def build_synthetic_lfm2(out_path: Path, seed: int = 20261121) -> None:
    rng = random.Random(seed)
    n_vocab = 288  # multiple of 32 (>= 256 bytes + special tokens)
    n_embd = 128
    n_ff = 256
    n_layers = 4
    n_head = 4
    head_dim = n_embd // n_head  # 32
    head_count_kv = [0, 2, 0, 2]  # alternating ShortConv (0,2) and GQA (1,3)
    l_cache = 4
    sliding_window = 8

    b2u = gpt2_byte_to_unicode()
    tokens = ["<pad>", "<bos>", "<eos>", "<unk>"]
    token_types = [3, 3, 3, 3]
    for b in range(256):
        tokens.append(b2u[b])
        token_types.append(1)
    while len(tokens) < n_vocab:
        tokens.append(f"<extra_{len(tokens)}>")
        token_types.append(3)

    w = GgufWriter()
    w.add_str("general.architecture", "lfm2")
    w.add_str("general.name", "synthetic-lfm2-hybrid-test")
    w.add_u32("lfm2.context_length", 512)
    w.add_u32("lfm2.embedding_length", n_embd)
    w.add_u32("lfm2.feed_forward_length", n_ff)
    w.add_u32("lfm2.block_count", n_layers)
    w.add_u32("lfm2.attention.head_count", n_head)
    w.add_u32_arr("lfm2.attention.head_count_kv", head_count_kv)
    w.add_f32("lfm2.attention.layer_norm_rms_epsilon", 1e-5)
    w.add_u32("lfm2.attention.sliding_window", sliding_window)
    w.add_u32("lfm2.rope.dimension_count", head_dim)
    w.add_f32("lfm2.rope.freq_base", 1000000.0)
    w.add_u32("lfm2.shortconv.l_cache", l_cache)

    w.add_str("tokenizer.ggml.model", "gpt2")
    w.add_str("tokenizer.ggml.pre", "gpt-2")
    w.add_str_arr("tokenizer.ggml.tokens", tokens)
    w.add_i32_arr("tokenizer.ggml.token_type", token_types)
    w.add_str_arr("tokenizer.ggml.merges", [])
    w.add_u32("tokenizer.ggml.bos_token_id", 1)
    w.add_u32("tokenizer.ggml.eos_token_id", 2)

    def rand_q4(in_dim: int, out_dim: int, scale: float | None = None) -> bytes:
        s = scale if scale is not None else (0.6 / math.sqrt(in_dim))
        vals = [rng.gauss(0.0, s) for _ in range(in_dim * out_dim)]
        return quantize_q4_0(vals)

    def rand_norm(dim: int) -> bytes:
        vals = [1.0 + rng.uniform(-0.15, 0.15) for _ in range(dim)]
        return pack_f32(vals)

    w.add_tensor("token_embd.weight", [n_embd, n_vocab], GGML_TYPE_Q4_0, rand_q4(n_embd, n_vocab, 0.25))
    w.add_tensor("token_embd_norm.weight", [n_embd], GGML_TYPE_F32, rand_norm(n_embd))

    for il in range(n_layers):
        w.add_tensor(f"blk.{il}.attn_norm.weight", [n_embd], GGML_TYPE_F32, rand_norm(n_embd))
        w.add_tensor(f"blk.{il}.ffn_norm.weight", [n_embd], GGML_TYPE_F32, rand_norm(n_embd))
        if head_count_kv[il] == 0:
            w.add_tensor(
                f"blk.{il}.shortconv.in_proj.weight",
                [n_embd, 3 * n_embd],
                GGML_TYPE_Q4_0,
                rand_q4(n_embd, 3 * n_embd),
            )
            conv_vals = [rng.uniform(-0.35, 0.35) for _ in range(l_cache * n_embd)]
            w.add_tensor(
                f"blk.{il}.shortconv.conv.weight",
                [l_cache, n_embd],
                GGML_TYPE_F32,
                pack_f32(conv_vals),
            )
            w.add_tensor(
                f"blk.{il}.shortconv.out_proj.weight",
                [n_embd, n_embd],
                GGML_TYPE_Q4_0,
                rand_q4(n_embd, n_embd),
            )
        else:
            kv_dim = head_count_kv[il] * head_dim
            w.add_tensor(f"blk.{il}.attn_q_norm.weight", [head_dim], GGML_TYPE_F32, rand_norm(head_dim))
            w.add_tensor(f"blk.{il}.attn_k_norm.weight", [head_dim], GGML_TYPE_F32, rand_norm(head_dim))
            w.add_tensor(f"blk.{il}.attn_q.weight", [n_embd, n_embd], GGML_TYPE_Q4_0, rand_q4(n_embd, n_embd))
            w.add_tensor(f"blk.{il}.attn_k.weight", [n_embd, kv_dim], GGML_TYPE_Q4_0, rand_q4(n_embd, kv_dim))
            w.add_tensor(f"blk.{il}.attn_v.weight", [n_embd, kv_dim], GGML_TYPE_Q4_0, rand_q4(n_embd, kv_dim))
            w.add_tensor(f"blk.{il}.attn_output.weight", [n_embd, n_embd], GGML_TYPE_Q4_0, rand_q4(n_embd, n_embd))

        w.add_tensor(f"blk.{il}.ffn_gate.weight", [n_embd, n_ff], GGML_TYPE_Q4_0, rand_q4(n_embd, n_ff))
        w.add_tensor(f"blk.{il}.ffn_up.weight", [n_embd, n_ff], GGML_TYPE_Q4_0, rand_q4(n_embd, n_ff))
        w.add_tensor(f"blk.{il}.ffn_down.weight", [n_ff, n_embd], GGML_TYPE_Q4_0, rand_q4(n_ff, n_embd))

    w.write(out_path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic LFM2 GGUF fixture")
    ap.add_argument("-o", "--output", type=Path, required=True, help="Output GGUF path")
    ap.add_argument("--seed", type=int, default=20261121)
    args = ap.parse_args()
    build_synthetic_lfm2(args.output, args.seed)
    print(f"Wrote synthetic LFM2 GGUF ({args.output.stat().st_size} bytes) -> {args.output}")


if __name__ == "__main__":
    main()
