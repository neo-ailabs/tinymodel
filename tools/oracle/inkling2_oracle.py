#!/usr/bin/env python3
"""Independent FP64/FP32 Reference Oracle & Synthetic GGUF Generator for `inkling2_loop` (`AGENTS.md` Rules #3 & #7).

Implements the exact mathematical specification of Thinking Machines Lab's `Inkling-2-Mini`
(`general.architecture = "inkling2_loop"` / `"inkling2"`, upstream MIT reference:
`ggml-org/llama.cpp` `src/models/inkling2.cpp` & `src/models/inkling2-loop.cpp`).

Usage:
  1. Generate a synthetic `inkling2_loop` Q8_0 GGUF + multi-token golden `TMLG` v1 reference:
       python3 tools/oracle/inkling2_oracle.py \
           --gen-gguf target/synthetic_inkling2_loop_q8_0.gguf \
           --tokens 5,12,29,41,7,63,18 \
           --out target/synthetic_inkling2_loop_ref.tmlg

  2. Run the FP64/FP32 Python oracle directly on any existing `inkling2` / `inkling2_loop` GGUF:
       python3 tools/oracle/inkling2_oracle.py \
           --gguf target/synthetic_inkling2_loop_q8_0.gguf \
           --tokens 5,12,29,41,7,63,18 \
           --out target/ref.tmlg
"""

from __future__ import annotations

import argparse
import math
import struct
from pathlib import Path
from typing import Any

import numpy as np

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3

# GGUF metadata value types
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9

# GGML tensor types
GGML_TYPE_F32 = 0
GGML_TYPE_Q8_0 = 8

LOOP_CTRL_NAMES = ["q", "k", "v", "ffn"]


def quantize_q8_0_rows(mat_f32: np.ndarray) -> bytes:
    """Quantize a 2D/3D array `[..., cols]` (`cols % 32 == 0`) to GGML `Q8_0` bytes."""
    arr = np.asarray(mat_f32, dtype=np.float32)
    cols = arr.shape[-1]
    assert cols % 32 == 0, f"Q8_0 requires cols % 32 == 0, got {cols}"
    flat = arr.reshape(-1, 32)
    amax = np.max(np.abs(flat), axis=1)
    d_f32 = amax / 127.0
    d_f16 = d_f32.astype(np.float16)
    d_back = d_f16.astype(np.float32)
    inv_d = np.where(d_back != 0.0, 1.0 / d_back, 0.0)
    qs = np.rint(flat * inv_d[:, None]).clip(-128, 127).astype(np.int8)
    out = bytearray(flat.shape[0] * 34)
    for i in range(flat.shape[0]):
        off = i * 34
        out[off : off + 2] = d_f16[i].tobytes()
        out[off + 2 : off + 34] = qs[i].tobytes()
    return bytes(out)


def dequantize_q8_0_bytes(raw: bytes, shape_row_major: tuple[int, ...]) -> np.ndarray:
    """Dequantize GGML `Q8_0` bytes into `float64` with `shape_row_major`."""
    n_elem = int(np.prod(shape_row_major))
    assert n_elem % 32 == 0
    n_blocks = n_elem // 32
    assert len(raw) == n_blocks * 34
    dt = np.dtype([("d", "<f2"), ("qs", "i1", (32,))])
    blocks = np.frombuffer(raw, dtype=dt, count=n_blocks)
    vals = blocks["d"].astype(np.float64)[:, None] * blocks["qs"].astype(np.float64)
    return vals.reshape(shape_row_major)


class GgufWriter:
    def __init__(self) -> None:
        self.kv: list[tuple[str, int, bytes]] = []
        self.tensors: list[tuple[str, tuple[int, ...], int, bytes]] = []

    def _pack_str(self, s: str) -> bytes:
        b = s.encode("utf-8")
        return struct.pack("<Q", len(b)) + b

    def add_str(self, key: str, val: str) -> None:
        self.kv.append((key, GGUF_TYPE_STRING, self._pack_str(val)))

    def add_u32(self, key: str, val: int) -> None:
        self.kv.append((key, GGUF_TYPE_UINT32, struct.pack("<I", val)))

    def add_f32(self, key: str, val: float) -> None:
        self.kv.append((key, GGUF_TYPE_FLOAT32, struct.pack("<f", val)))

    def add_bool(self, key: str, val: bool) -> None:
        self.kv.append((key, GGUF_TYPE_BOOL, struct.pack("<?", val)))

    def add_arr_str(self, key: str, vals: list[str]) -> None:
        body = bytearray(struct.pack("<IQ", GGUF_TYPE_STRING, len(vals)))
        for v in vals:
            body.extend(self._pack_str(v))
        self.kv.append((key, GGUF_TYPE_ARRAY, bytes(body)))

    def add_arr_u32(self, key: str, vals: list[int]) -> None:
        body = struct.pack("<IQ", GGUF_TYPE_UINT32, len(vals)) + struct.pack(f"<{len(vals)}I", *vals)
        self.kv.append((key, GGUF_TYPE_ARRAY, body))

    def add_arr_i32(self, key: str, vals: list[int]) -> None:
        body = struct.pack("<IQ", GGUF_TYPE_INT32, len(vals)) + struct.pack(f"<{len(vals)}i", *vals)
        self.kv.append((key, GGUF_TYPE_ARRAY, body))

    def add_arr_bool(self, key: str, vals: list[bool]) -> None:
        body = struct.pack("<IQ", GGUF_TYPE_BOOL, len(vals)) + struct.pack(f"<{len(vals)}?", *vals)
        self.kv.append((key, GGUF_TYPE_ARRAY, body))

    def add_tensor_f32(self, name: str, arr_row_major: np.ndarray, ggml_ne: tuple[int, ...]) -> None:
        f32 = np.asarray(arr_row_major, dtype=np.float32)
        assert f32.size == int(np.prod(ggml_ne)), f"{name}: {f32.shape} vs ne={ggml_ne}"
        self.tensors.append((name, ggml_ne, GGML_TYPE_F32, f32.tobytes()))

    def add_tensor_q8_0(self, name: str, arr_row_major: np.ndarray, ggml_ne: tuple[int, ...]) -> None:
        f32 = np.asarray(arr_row_major, dtype=np.float32)
        assert f32.size == int(np.prod(ggml_ne)), f"{name}: {f32.shape} vs ne={ggml_ne}"
        assert f32.shape[-1] == ggml_ne[0], f"{name}: last dim {f32.shape[-1]} != ne[0] {ggml_ne[0]}"
        self.tensors.append((name, ggml_ne, GGML_TYPE_Q8_0, quantize_q8_0_rows(f32)))

    def write(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        hdr = bytearray()
        hdr.extend(GGUF_MAGIC)
        hdr.extend(struct.pack("<IQQ", GGUF_VERSION, len(self.tensors), len(self.kv)))
        for k, vtype, payload in self.kv:
            hdr.extend(self._pack_str(k))
            hdr.extend(struct.pack("<I", vtype))
            hdr.extend(payload)

        offsets: list[int] = []
        cur_off = 0
        for _, _, _, raw in self.tensors:
            cur_off = (cur_off + 31) & ~31
            offsets.append(cur_off)
            cur_off += len(raw)

        for (name, ne, ttype, _), off in zip(self.tensors, offsets):
            hdr.extend(self._pack_str(name))
            hdr.extend(struct.pack("<I", len(ne)))
            hdr.extend(struct.pack(f"<{len(ne)}Q", *ne))
            hdr.extend(struct.pack("<IQ", ttype, off))

        while len(hdr) % 32 != 0:
            hdr.append(0)

        with open(path, "wb") as f:
            f.write(hdr)
            written = 0
            for (_, _, _, raw), off in zip(self.tensors, offsets):
                if off > written:
                    f.write(b"\x00" * (off - written))
                    written = off
                f.write(raw)
                written += len(raw)


def generate_synthetic_inkling2_loop_gguf(out_path: Path, seed: int = 20261122) -> None:
    """Create a compact synthetic `inkling2_loop` Q8_0+F32 GGUF exercising every architectural branch:
    - `n_phys = 8`, `core = [2..6)`, `n_passes = 3` -> `16` logical cache slots
    - `attn_res.block_size = 2` -> 2 blocks inside core (`bank` is non-empty on passes 1 & 2!)
    - Mixed SWA + full-attention (`is_swa = [T, F, T, F, T, F, T, F]`) across prelude, core, and coda
    - `n_swa = 4`, `n_rel_extent = 6`, `log_scale_floor = 3` (all triggered on a 7-token sequence!)
    - `vocab_padded = 96`, `vocab_unpadded = 70` (verifies padded tail rows are dropped)
    """
    rng = np.random.default_rng(seed)

    def rand_w(shape: tuple[int, ...], scale: float = 0.08) -> np.ndarray:
        return rng.standard_normal(shape).astype(np.float32) * scale

    def rand_norm(dim: int) -> np.ndarray:
        return (1.0 + 0.15 * rng.standard_normal((dim,))).astype(np.float32)

    arch = "inkling2_loop"
    n_phys = 8
    n_embd = 64
    n_head = 4
    n_head_kv = 2
    head_dim = 16
    q_dim = n_head * head_dim  # 64
    kv_dim = n_head_kv * head_dim  # 32
    n_ff_exp = 32
    n_expert = 8
    n_expert_used = 2
    n_rel = 8
    n_swa = 4
    n_rel_extent = 6
    n_conv = 4
    core_start = 2
    core_end = 6
    n_passes = 3
    gate_rank = 8
    block_size = 2
    vocab_padded = 96
    vocab_unpadded = 70
    is_swa = [True, False, True, False, True, False, True, False]

    w = GgufWriter()
    w.add_str("general.architecture", arch)
    w.add_str("general.name", "synthetic-inkling2-loop-mini")
    w.add_u32(f"{arch}.block_count", n_phys)
    w.add_u32(f"{arch}.context_length", 128)
    w.add_u32(f"{arch}.embedding_length", n_embd)
    w.add_u32(f"{arch}.expert_feed_forward_length", n_ff_exp)
    w.add_u32(f"{arch}.expert_count", n_expert)
    w.add_u32(f"{arch}.expert_used_count", n_expert_used)
    w.add_f32(f"{arch}.expert_weights_scale", 8.0)
    w.add_arr_u32(f"{arch}.attention.head_count", [n_head] * n_phys)
    w.add_arr_u32(f"{arch}.attention.head_count_kv", [n_head_kv] * n_phys)
    w.add_u32(f"{arch}.attention.key_length", head_dim)
    w.add_u32(f"{arch}.attention.value_length", head_dim)
    w.add_f32(f"{arch}.attention.layer_norm_rms_epsilon", 1e-6)
    w.add_u32(f"{arch}.rope.dimension_count", 0)
    w.add_u32(f"{arch}.attention.sliding_window", n_swa)
    w.add_arr_bool(f"{arch}.attention.sliding_window_pattern", is_swa)
    w.add_u32(f"{arch}.shortconv.l_cache", n_conv)
    w.add_f32(f"{arch}.activation.situ_beta", 4.0)
    w.add_f32(f"{arch}.activation.situ_linear_beta", 25.0)
    w.add_u32(f"{arch}.attn_res.block_size", block_size)
    w.add_f32(f"{arch}.final_logit_softcapping", 30.0)
    w.add_u32(f"{arch}.attention.relative_length", n_rel)
    w.add_u32(f"{arch}.attention.relative_extent", n_rel_extent)
    w.add_f32(f"{arch}.attention.qk_clip", 22.6274166)
    w.add_f32(f"{arch}.attention.relative_clip", 8.0)
    w.add_u32(f"{arch}.attention.log_scale_floor", 3)
    w.add_f32(f"{arch}.attention.log_scale_alpha", 0.15)
    w.add_u32(f"{arch}.vocab_size", vocab_padded)
    w.add_u32(f"{arch}.vocab_size_unpadded", vocab_unpadded)
    w.add_u32(f"{arch}.loop.start", core_start)
    w.add_u32(f"{arch}.loop.end", core_end)
    w.add_u32(f"{arch}.loop.count", n_passes)
    w.add_u32(f"{arch}.loop.gate_rank", gate_rank)
    w.add_bool(f"{arch}.loop.varying", False)

    # Minimal valid GPT-4o / inkling2 BPE vocabulary (`vocab_unpadded = 70` tokens)
    toks = [f"t{i:02d}" for i in range(vocab_unpadded - 2)] + ["<|end_of_text|>", "<|im_end|>"]
    ttypes = [1] * (vocab_unpadded - 2) + [3, 3]
    w.add_str("tokenizer.ggml.model", "gpt2")
    w.add_str("tokenizer.ggml.pre", "inkling2")
    w.add_arr_str("tokenizer.ggml.tokens", toks)
    w.add_arr_i32("tokenizer.ggml.token_type", ttypes)
    w.add_arr_str("tokenizer.ggml.merges", [])
    w.add_u32("tokenizer.ggml.eos_token_id", vocab_unpadded - 2)
    w.add_bool("tokenizer.ggml.add_bos_token", False)

    # Global tensors (`ggml_ne = (cols, rows)`)
    embd = rand_w((vocab_padded, n_embd), 0.12)
    embd[vocab_unpadded:] = 9.0  # Sentinel on padded rows to catch accidental use!
    out_w = rand_w((vocab_padded, n_embd), 0.12)
    out_w[vocab_unpadded:] = 9.0  # Sentinel on padded rows to verify truncation to vocab_unpadded!
    w.add_tensor_q8_0("token_embd.weight", embd, (n_embd, vocab_padded))
    w.add_tensor_f32("token_embd_norm.weight", rand_norm(n_embd), (n_embd,))
    w.add_tensor_f32("output_res_query.weight", rand_w((n_embd,), 0.2), (n_embd,))
    w.add_tensor_f32("output_norm.weight", rand_norm(n_embd), (n_embd,))
    w.add_tensor_q8_0("output.weight", out_w, (n_embd, vocab_padded))

    for i in range(n_phys):
        swa = is_swa[i]
        extent = n_swa if swa else n_rel_extent
        w.add_tensor_q8_0(f"blk.{i}.attn_q.weight", rand_w((q_dim, n_embd)), (n_embd, q_dim))
        w.add_tensor_q8_0(f"blk.{i}.attn_k.weight", rand_w((kv_dim, n_embd)), (n_embd, kv_dim))
        w.add_tensor_q8_0(f"blk.{i}.attn_v.weight", rand_w((kv_dim, n_embd)), (n_embd, kv_dim))
        w.add_tensor_q8_0(f"blk.{i}.attn_output.weight", rand_w((n_embd, q_dim)), (q_dim, n_embd))
        w.add_tensor_q8_0(f"blk.{i}.attn_gate.weight", rand_w((q_dim, n_embd)), (n_embd, q_dim))
        w.add_tensor_f32(
            f"blk.{i}.attn_q_clip.weight",
            (1.0 + 0.3 * rng.standard_normal((n_head, head_dim))).astype(np.float32),
            (head_dim, n_head),
        )
        w.add_tensor_q8_0(f"blk.{i}.attn_sink_proj.weight", rand_w((n_head, n_embd), 0.1), (n_embd, n_head))
        w.add_tensor_q8_0(f"blk.{i}.attn_rel.weight", rand_w((n_rel * n_head, n_embd), 0.12), (n_embd, n_rel * n_head))
        w.add_tensor_f32(f"blk.{i}.attn_rel_bias.weight", rand_w((extent, n_rel), 0.25), (n_rel, extent))
        w.add_tensor_f32(
            f"blk.{i}.attn_rel_clip.weight",
            (1.0 + 0.2 * rng.standard_normal((n_head, n_rel))).astype(np.float32),
            (n_rel, n_head),
        )

        w.add_tensor_f32(f"blk.{i}.attn_k_conv.weight", rand_w((kv_dim, n_conv), 0.2), (n_conv, kv_dim))
        w.add_tensor_f32(f"blk.{i}.attn_v_conv.weight", rand_w((kv_dim, n_conv), 0.2), (n_conv, kv_dim))
        w.add_tensor_f32(f"blk.{i}.attn_post_conv.weight", rand_w((n_embd, n_conv), 0.2), (n_conv, n_embd))
        w.add_tensor_f32(f"blk.{i}.ffn_post_conv.weight", rand_w((n_embd, n_conv), 0.2), (n_conv, n_embd))

        if swa:
            w.add_tensor_f32(f"blk.{i}.attn_norm.weight", rand_norm(n_embd), (n_embd,))
        else:
            w.add_tensor_f32(f"blk.{i}.attn_q_inp_norm.weight", rand_norm(n_embd), (n_embd,))
            w.add_tensor_f32(f"blk.{i}.attn_k_inp_norm.weight", rand_norm(n_embd), (n_embd,))
            w.add_tensor_f32(f"blk.{i}.attn_v_inp_norm.weight", rand_norm(n_embd), (n_embd,))
            w.add_tensor_q8_0(
                f"blk.{i}.attn_k_conv_proj.weight",
                rand_w((n_conv * n_head_kv, n_embd), 0.15),
                (n_embd, n_conv * n_head_kv),
            )
            w.add_tensor_q8_0(
                f"blk.{i}.attn_v_conv_proj.weight",
                rand_w((n_conv * n_head_kv, n_embd), 0.15),
                (n_embd, n_conv * n_head_kv),
            )

        if i > 0:
            if swa:
                w.add_tensor_f32(f"blk.{i}.attn_res_query.weight", rand_w((n_embd,), 0.2), (n_embd,))
            else:
                w.add_tensor_f32(f"blk.{i}.attn_res_query.weight", rand_w((3, n_embd), 0.2), (n_embd, 3))
        w.add_tensor_f32(f"blk.{i}.ffn_res_query.weight", rand_w((n_embd,), 0.2), (n_embd,))

        # MoE
        w.add_tensor_f32(f"blk.{i}.ffn_norm.weight", rand_norm(n_embd), (n_embd,))
        w.add_tensor_f32(f"blk.{i}.ffn_gate_inp.weight", rand_w((n_expert, n_embd), 0.3), (n_embd, n_expert))
        w.add_tensor_f32(f"blk.{i}.exp_probs_b.bias", rand_w((n_expert,), 0.15), (n_expert,))
        w.add_tensor_q8_0(
            f"blk.{i}.ffn_gate_up_exps.weight",
            rand_w((n_expert, 2 * n_ff_exp, n_embd), 0.10),
            (n_embd, 2 * n_ff_exp, n_expert),
        )
        w.add_tensor_q8_0(
            f"blk.{i}.ffn_down_exps.weight",
            rand_w((n_expert, n_embd, n_ff_exp), 0.10),
            (n_ff_exp, n_embd, n_expert),
        )

        # Loop controllers on core `[core_start..core_end)`
        if core_start <= i < core_end:
            w.add_tensor_f32(f"blk.{i}.loop_msg_query.weight", rand_w((4, n_embd), 0.2), (n_embd, 4))
            for ctrl in LOOP_CTRL_NAMES:
                w.add_tensor_f32(
                    f"blk.{i}.loop_{ctrl}_ctrl_down.weight",
                    rand_w((gate_rank, 2 * n_embd), 0.12),
                    (2 * n_embd, gate_rank),
                )
                w.add_tensor_f32(
                    f"blk.{i}.loop_{ctrl}_ctrl_gain.weight",
                    rand_w((n_embd, gate_rank), 0.12),
                    (gate_rank, n_embd),
                )
                w.add_tensor_f32(
                    f"blk.{i}.loop_{ctrl}_ctrl_forget.weight",
                    rand_w((n_embd, gate_rank), 0.12),
                    (gate_rank, n_embd),
                )

    w.write(out_path)


class GgufReader:
    """Minimal self-contained GGUF v3 reader supporting F32 & Q8_0 tensors."""

    def __init__(self, path: Path) -> None:
        self.raw = path.read_bytes()
        pos = 0
        magic, ver, n_tensors, n_kv = struct.unpack_from("<4sIQQ", self.raw, pos)
        assert magic == GGUF_MAGIC and ver == 3
        pos += 24
        self.kv: dict[str, Any] = {}
        for _ in range(n_kv):
            k, pos = self._read_str(pos)
            vtype = struct.unpack_from("<I", self.raw, pos)[0]
            pos += 4
            val, pos = self._read_val(vtype, pos)
            self.kv[k] = val
        self.tensor_meta: dict[str, tuple[tuple[int, ...], int, int]] = {}
        for _ in range(n_tensors):
            name, pos = self._read_str(pos)
            nd = struct.unpack_from("<I", self.raw, pos)[0]
            pos += 4
            ne = struct.unpack_from(f"<{nd}Q", self.raw, pos)
            pos += 8 * nd
            ttype, rel_off = struct.unpack_from("<IQ", self.raw, pos)
            pos += 12
            self.tensor_meta[name] = (ne, ttype, rel_off)
        align = int(self.kv.get("general.alignment", 32))
        self.data_offset = (pos + align - 1) // align * align

    def _read_str(self, pos: int) -> tuple[str, int]:
        ln = struct.unpack_from("<Q", self.raw, pos)[0]
        pos += 8
        return self.raw[pos : pos + ln].decode("utf-8"), pos + ln

    def _read_val(self, vtype: int, pos: int) -> tuple[Any, int]:
        if vtype == 0:
            return struct.unpack_from("<B", self.raw, pos)[0], pos + 1
        if vtype == 1:
            return struct.unpack_from("<b", self.raw, pos)[0], pos + 1
        if vtype == 2:
            return struct.unpack_from("<H", self.raw, pos)[0], pos + 2
        if vtype == 3:
            return struct.unpack_from("<h", self.raw, pos)[0], pos + 2
        if vtype == 4:
            return struct.unpack_from("<I", self.raw, pos)[0], pos + 4
        if vtype == 5:
            return struct.unpack_from("<i", self.raw, pos)[0], pos + 4
        if vtype == 6:
            return struct.unpack_from("<f", self.raw, pos)[0], pos + 4
        if vtype == 7:
            return bool(self.raw[pos]), pos + 1
        if vtype == 8:
            return self._read_str(pos)
        if vtype == 10:
            return struct.unpack_from("<Q", self.raw, pos)[0], pos + 8
        if vtype == 11:
            return struct.unpack_from("<q", self.raw, pos)[0], pos + 8
        if vtype == 12:
            return struct.unpack_from("<d", self.raw, pos)[0], pos + 8
        if vtype == 9:
            elem_t, cnt = struct.unpack_from("<IQ", self.raw, pos)
            pos += 12
            arr = []
            for _ in range(cnt):
                v, pos = self._read_val(elem_t, pos)
                arr.append(v)
            return arr, pos
        raise ValueError(f"unsupported GGUF vtype {vtype}")

    def tensor_f64(self, name: str) -> np.ndarray:
        ne, ttype, rel_off = self.tensor_meta[name]
        # GGML `ne = (ne0, ne1, ne2)` corresponds to row-major `(ne2, ne1, ne0)`
        rm_shape = tuple(reversed(ne))
        n_elem = int(np.prod(rm_shape))
        abs_off = self.data_offset + rel_off
        if ttype == GGML_TYPE_F32:
            byte_len = n_elem * 4
            arr = np.frombuffer(self.raw, dtype="<f4", count=n_elem, offset=abs_off)
            assert abs_off + byte_len <= len(self.raw)
            return arr.astype(np.float64).reshape(rm_shape)
        if ttype == GGML_TYPE_Q8_0:
            byte_len = (n_elem // 32) * 34
            return dequantize_q8_0_bytes(self.raw[abs_off : abs_off + byte_len], rm_shape)
        raise ValueError(f"{name}: unsupported ttype={ttype}")


def rms_norm_unit(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    ms = np.mean(x * x, axis=-1, keepdims=True)
    return x / np.sqrt(ms + eps)


def rms_norm_weighted(x: np.ndarray, w: np.ndarray, eps: float) -> np.ndarray:
    return rms_norm_unit(x, eps) * w


def clip_rows(x: np.ndarray, opt_scale: np.ndarray | None, threshold: float) -> np.ndarray:
    y = x * opt_scale if opt_scale is not None else x.copy()
    norm = np.sqrt(np.sum(y * y, axis=-1, keepdims=True)) + 1e-6
    factor = np.clip(threshold / norm, 0.0, 1.0)
    return y * factor


def mix_depth(blocks: list[np.ndarray], cur: np.ndarray, query: np.ndarray) -> np.ndarray:
    """`inkling2.cpp:179-192` depth attention over `[blocks..., cur]`.
    `query`: `[n_queries, n_embd]` or `[n_embd]`. Returns `[n_queries, n_embd]` or `[n_embd]`.
    """
    squeeze = query.ndim == 1
    q2 = query[None, :] if squeeze else query
    # Order matches `inkling2.cpp:180-183`: prepend each block in order -> [blocks[-1], ..., blocks[0], cur]
    states = np.stack([*reversed(blocks), cur], axis=0)  # [M, n_embd]
    states_hat = rms_norm_unit(states, 1e-6)
    n_embd = cur.shape[0]
    scores = (q2 @ states_hat.T) / math.sqrt(float(n_embd))  # [Q, M]
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    probs = np.exp(scores)
    probs = probs / np.sum(probs, axis=-1, keepdims=True)
    out = probs @ states_hat  # [Q, n_embd]
    return out[0] if squeeze else out


class Inkling2Oracle:
    """Exact token-by-token FP64 reference implementation of `inkling2` & `inkling2_loop`."""

    def __init__(self, gguf: GgufReader) -> None:
        self.g = gguf
        kv = gguf.kv
        arch = kv["general.architecture"]
        assert arch in ("inkling2", "inkling2_loop")
        self.arch = arch
        self.n_phys = int(kv[f"{arch}.block_count"])
        self.n_embd = int(kv[f"{arch}.embedding_length"])
        self.n_ff_exp = int(kv[f"{arch}.expert_feed_forward_length"])
        self.n_expert = int(kv[f"{arch}.expert_count"])
        eu = kv[f"{arch}.expert_used_count"]
        self.n_expert_used = int(eu[0] if isinstance(eu, list) else eu)
        self.expert_weights_scale = float(kv.get(f"{arch}.expert_weights_scale", 1.0))
        hc = kv[f"{arch}.attention.head_count"]
        hkv = kv[f"{arch}.attention.head_count_kv"]
        self.n_head = [int(x) for x in hc] if isinstance(hc, list) else [int(hc)] * self.n_phys
        self.n_head_kv = [int(x) for x in hkv] if isinstance(hkv, list) else [int(hkv)] * self.n_phys
        self.head_dim = int(kv[f"{arch}.attention.key_length"])
        self.rms_eps = float(kv[f"{arch}.attention.layer_norm_rms_epsilon"])
        self.n_swa = int(kv[f"{arch}.attention.sliding_window"])
        self.is_swa = [bool(x) for x in kv[f"{arch}.attention.sliding_window_pattern"]]
        self.n_conv = int(kv[f"{arch}.shortconv.l_cache"])
        self.situ_beta = float(kv[f"{arch}.activation.situ_beta"])
        self.situ_lb = float(kv[f"{arch}.activation.situ_linear_beta"])
        self.block_size = int(kv[f"{arch}.attn_res.block_size"])
        self.softcap = float(kv.get(f"{arch}.final_logit_softcapping", 0.0))
        self.n_rel = int(kv[f"{arch}.attention.relative_length"])
        self.n_rel_extent = int(kv[f"{arch}.attention.relative_extent"])
        self.qk_clip = float(kv[f"{arch}.attention.qk_clip"])
        self.rel_clip = float(kv[f"{arch}.attention.relative_clip"])
        self.log_floor = float(kv[f"{arch}.attention.log_scale_floor"])
        self.log_alpha = float(kv[f"{arch}.attention.log_scale_alpha"])
        self.vocab_padded = int(kv[f"{arch}.vocab_size"])
        self.vocab_unpadded = int(kv[f"{arch}.vocab_size_unpadded"])

        if arch == "inkling2_loop":
            self.core_start = int(kv[f"{arch}.loop.start"])
            self.core_end = int(kv[f"{arch}.loop.end"])
            self.n_passes = int(kv[f"{arch}.loop.count"])
            self.gate_rank = int(kv[f"{arch}.loop.gate_rank"])
            self.n_slots = self.n_phys + (self.n_passes - 1) * (self.core_end - self.core_start)
        else:
            self.core_start = 0
            self.core_end = 0
            self.n_passes = 1
            self.gate_rank = 0
            self.n_slots = self.n_phys

        self.tok_embd = gguf.tensor_f64("token_embd.weight")
        self.tok_norm = gguf.tensor_f64("token_embd_norm.weight")
        self.out_res_q = gguf.tensor_f64("output_res_query.weight")
        self.out_norm = gguf.tensor_f64("output_norm.weight")
        self.out_w = gguf.tensor_f64("output.weight")

        self.reset_state()

    def reset_state(self) -> None:
        self.k_cache: list[list[np.ndarray]] = [[] for _ in range(self.n_slots)]
        self.v_cache: list[list[np.ndarray]] = [[] for _ in range(self.n_slots)]
        # Per-slot conv history: 4 conv stages (`k`, `v`, `attn_post`, `ffn_post`), each `[n_conv - 1, channels]`
        self.conv_hist: list[dict[str, np.ndarray]] = []
        for il in range(self.n_slots):
            ip = self.layer_phys(il)
            kv_dim = self.n_head_kv[ip] * self.head_dim
            self.conv_hist.append(
                {
                    "k": np.zeros((self.n_conv - 1, kv_dim), dtype=np.float64),
                    "v": np.zeros((self.n_conv - 1, kv_dim), dtype=np.float64),
                    "attn": np.zeros((self.n_conv - 1, self.n_embd), dtype=np.float64),
                    "ffn": np.zeros((self.n_conv - 1, self.n_embd), dtype=np.float64),
                }
            )

    def layer_phys(self, il: int) -> int:
        if il < self.n_phys:
            return il
        return self.core_start + (il - self.n_phys) % (self.core_end - self.core_start)

    def _convolve(
        self,
        il: int,
        stage: str,
        cur: np.ndarray,
        weight: np.ndarray,
        dynamic: np.ndarray | None,
    ) -> np.ndarray:
        """`inkling2.cpp:196-238` causal short conv with residual & optional dynamic taps."""
        hist = self.conv_hist[il][stage]  # [n_conv - 1, C] oldest -> newest
        x_win = np.vstack([hist, cur[None, :]])  # [n_conv, C], index 0 is t-(n_conv-1), index n_conv-1 is t
        self.conv_hist[il][stage] = x_win[1:].copy()

        # `weight` row-major is `[C, n_conv]` (`ne = [n_conv, C]`)
        out = np.sum(x_win.T * weight, axis=1)  # [C]
        if dynamic is not None:
            # `dynamic` has length `n_conv * n_head_kv`, reshaped in ggml `(n_conv, n_heads)` -> row-major `(n_heads, n_conv)`
            n_heads = dynamic.shape[0] // self.n_conv
            dyn = dynamic.reshape(n_heads, self.n_conv)
            ch_per_head = cur.shape[0] // n_heads
            for lag in range(self.n_conv):
                win_lag = x_win[self.n_conv - 1 - lag].reshape(n_heads, ch_per_head)
                out += (win_lag * dyn[:, lag : lag + 1]).reshape(-1)

        denom = math.sqrt(float((2 if dynamic is not None else 1) * self.n_conv + 1))
        return (cur + out) / denom

    def _control(self, ip: int, ctrl_idx: int, cur: np.ndarray, msg: np.ndarray) -> np.ndarray:
        ctrl = LOOP_CTRL_NAMES[ctrl_idx]
        gate_in = np.concatenate([rms_norm_unit(cur, 1e-6), rms_norm_unit(msg, 1e-6)], axis=0)
        w_down = self.g.tensor_f64(f"blk.{ip}.loop_{ctrl}_ctrl_down.weight")
        w_gain = self.g.tensor_f64(f"blk.{ip}.loop_{ctrl}_ctrl_gain.weight")
        w_forget = self.g.tensor_f64(f"blk.{ip}.loop_{ctrl}_ctrl_forget.weight")
        g = w_down @ gate_in
        gain = np.tanh(w_gain @ g)
        forget = np.tanh(w_forget @ g)
        return (cur - forget * cur) + gain * msg

    def _run_layer(
        self,
        il: int,
        ip: int,
        pos: int,
        tau: float,
        cur: np.ndarray,
        blocks: list[np.ndarray],
        messages: np.ndarray | None,
    ) -> np.ndarray:
        swa = self.is_swa[ip]
        nh = self.n_head[ip]
        nh_kv = self.n_head_kv[ip]
        hd = self.head_dim

        inp_sa = cur.copy()
        inputs = [cur.copy(), cur.copy(), cur.copy()]
        if ip > 0:
            rq = self.g.tensor_f64(f"blk.{ip}.attn_res_query.weight")
            mixed = mix_depth(blocks, cur, rq)
            if swa:
                inputs = [mixed.copy(), mixed.copy(), mixed.copy()]
            else:
                inputs = [mixed[0].copy(), mixed[1].copy(), mixed[2].copy()]

        if messages is not None:
            for j in range(3):
                inputs[j] = self._control(ip, j, inputs[j], messages[j])

        if swa and messages is not None:
            an = self.g.tensor_f64(f"blk.{ip}.attn_norm.weight")
            for j in range(3):
                inputs[j] = rms_norm_weighted(inputs[j], an, self.rms_eps)
        elif swa:
            an = self.g.tensor_f64(f"blk.{ip}.attn_norm.weight")
            inputs[0] = rms_norm_weighted(inputs[0], an, self.rms_eps)
            inputs[1] = inputs[0]
            inputs[2] = inputs[0]
        else:
            inputs[0] = rms_norm_weighted(inputs[0], self.g.tensor_f64(f"blk.{ip}.attn_q_inp_norm.weight"), self.rms_eps)
            inputs[1] = rms_norm_weighted(inputs[1], self.g.tensor_f64(f"blk.{ip}.attn_k_inp_norm.weight"), self.rms_eps)
            inputs[2] = rms_norm_weighted(inputs[2], self.g.tensor_f64(f"blk.{ip}.attn_v_inp_norm.weight"), self.rms_eps)

        # Self-attention
        q_raw = self.g.tensor_f64(f"blk.{ip}.attn_q.weight") @ inputs[0]
        k_mm = self.g.tensor_f64(f"blk.{ip}.attn_k.weight") @ inputs[1]
        v_mm = self.g.tensor_f64(f"blk.{ip}.attn_v.weight") @ inputs[2]
        k_dyn = None if swa else (self.g.tensor_f64(f"blk.{ip}.attn_k_conv_proj.weight") @ inputs[1])
        v_dyn = None if swa else (self.g.tensor_f64(f"blk.{ip}.attn_v_conv_proj.weight") @ inputs[2])
        k_conv = self._convolve(il, "k", k_mm, self.g.tensor_f64(f"blk.{ip}.attn_k_conv.weight"), k_dyn)
        v_conv = self._convolve(il, "v", v_mm, self.g.tensor_f64(f"blk.{ip}.attn_v_conv.weight"), v_dyn)
        r_raw = self.g.tensor_f64(f"blk.{ip}.attn_rel.weight") @ inputs[0]

        q_cur = clip_rows(
            q_raw.reshape(nh, hd),
            self.g.tensor_f64(f"blk.{ip}.attn_q_clip.weight"),
            self.qk_clip,
        )
        k_cur = clip_rows(k_conv.reshape(nh_kv, hd), None, self.qk_clip)
        # Match F16 KV cache rounding used by `tinymodel` default `KvType::F16` and `llama.cpp`
        v_cur = rms_norm_unit(v_conv.reshape(nh_kv, hd), self.rms_eps)
        r_cur = clip_rows(
            r_raw.reshape(nh, self.n_rel),
            self.g.tensor_f64(f"blk.{ip}.attn_rel_clip.weight"),
            self.rel_clip,
        )

        if not swa:
            q_cur = q_cur * tau
            r_cur = r_cur * tau

        # Store in F16-quantized KV cache (exact match with default F16 KV cache in llama.cpp & tinymodel)
        k_f16 = k_cur.astype(np.float16).astype(np.float64)
        v_f16 = v_cur.astype(np.float16).astype(np.float64)
        self.k_cache[il].append(k_f16)
        self.v_cache[il].append(v_f16)
        assert len(self.k_cache[il]) == pos + 1

        sinks = self.g.tensor_f64(f"blk.{ip}.attn_sink_proj.weight") @ inputs[0]  # [nh]
        rel_bias_w = self.g.tensor_f64(f"blk.{ip}.attn_rel_bias.weight")  # [extent, n_rel]
        extent = rel_bias_w.shape[0]
        # `table`: `[nh, extent + 1]`, last column is 0 for `distance >= extent`
        table = np.zeros((nh, extent + 1), dtype=np.float64)
        table[:, :extent] = r_cur @ rel_bias_w.T
        table = table - sinks[:, None]

        kv_start = max(0, pos + 1 - self.n_swa) if swa else 0
        gqa_ratio = nh // nh_kv
        scale = 1.0 / math.sqrt(float(hd))
        attn_heads = np.zeros((nh, hd), dtype=np.float64)
        k_stack = np.stack(self.k_cache[il][kv_start : pos + 1], axis=0)  # [L, nh_kv, hd]
        v_stack = np.stack(self.v_cache[il][kv_start : pos + 1], axis=0)  # [L, nh_kv, hd]
        kv_positions = np.arange(kv_start, pos + 1, dtype=np.int64)
        dists = pos - kv_positions
        rel_idx = np.where((dists >= 0) & (dists < extent), dists, extent)

        for h in range(nh):
            kvh = h // gqa_ratio
            logits = (k_stack[:, kvh, :] @ q_cur[h]) * scale + table[h, rel_idx]
            # Zero sink logit: `exp(s - m) / (exp(0 - m) + sum exp(s - m))`
            m = max(0.0, float(np.max(logits)))
            exps = np.exp(logits - m)
            denom = math.exp(0.0 - m) + float(np.sum(exps))
            probs = exps / denom
            attn_heads[h] = probs @ v_stack[:, kvh, :]

        attn_vec = attn_heads.reshape(-1)
        gate = 1.0 / (1.0 + np.exp(-(self.g.tensor_f64(f"blk.{ip}.attn_gate.weight") @ inputs[0])))
        attn_vec = attn_vec * gate
        wo_out = self.g.tensor_f64(f"blk.{ip}.attn_output.weight") @ attn_vec
        cur = self._convolve(il, "attn", wo_out, self.g.tensor_f64(f"blk.{ip}.attn_post_conv.weight"), None)

        if ip % self.block_size == 0:
            blocks.append(inp_sa)
        else:
            cur = inp_sa + cur

        ffn_inp = cur.copy()
        cur = mix_depth(blocks, cur, self.g.tensor_f64(f"blk.{ip}.ffn_res_query.weight"))
        if messages is not None:
            cur = self._control(ip, 3, cur, messages[3])
        cur = rms_norm_weighted(cur, self.g.tensor_f64(f"blk.{ip}.ffn_norm.weight"), self.rms_eps)

        # MoE FFN (`LLM_FFN_SITU`, sigmoid router + `exp_probs_b` selection bias + `norm_w` + `expert_weights_scale`)
        router_logits = self.g.tensor_f64(f"blk.{ip}.ffn_gate_inp.weight") @ cur
        probs = 1.0 / (1.0 + np.exp(-router_logits))
        sel_probs = probs + self.g.tensor_f64(f"blk.{ip}.exp_probs_b.bias")
        top_idx = np.argsort(-sel_probs, kind="stable")[: self.n_expert_used]
        top_w = probs[top_idx]
        top_w = (top_w / (float(np.sum(top_w)) + 1e-20)) * self.expert_weights_scale

        gate_up_exps = self.g.tensor_f64(f"blk.{ip}.ffn_gate_up_exps.weight")  # [E, 2*n_ff, n_embd]
        down_exps = self.g.tensor_f64(f"blk.{ip}.ffn_down_exps.weight")  # [E, n_embd, n_ff]
        moe_out = np.zeros((self.n_embd,), dtype=np.float64)
        for e_id, w_e in zip(top_idx, top_w):
            gu = gate_up_exps[e_id] @ cur
            gate_v = gu[: self.n_ff_exp]
            up_v = gu[self.n_ff_exp :]
            act_g = self.situ_beta * np.tanh(gate_v / self.situ_beta) * (1.0 / (1.0 + np.exp(-gate_v)))
            act_u = self.situ_lb * np.tanh(up_v / self.situ_lb) if self.situ_lb > 0.0 else up_v
            hidden = act_g * act_u
            moe_out += w_e * (down_exps[e_id] @ hidden)

        cur = self._convolve(il, "ffn", moe_out, self.g.tensor_f64(f"blk.{ip}.ffn_post_conv.weight"), None)
        return ffn_inp + cur

    def forward_token(self, token_id: int, pos: int) -> np.ndarray:
        assert 0 <= token_id < self.vocab_unpadded
        tau = 1.0 + self.log_alpha * math.log(max(1.0, float(pos + 1) / self.log_floor))
        cur = rms_norm_weighted(self.tok_embd[token_id], self.tok_norm, self.rms_eps)

        blocks: list[np.ndarray] = []
        if self.arch == "inkling2":
            for il in range(self.n_phys):
                cur = self._run_layer(il, il, pos, tau, cur, blocks, None)
        else:
            for il in range(self.core_start):
                cur = self._run_layer(il, il, pos, tau, cur, blocks, None)

            prelude = cur.copy()
            prelude_blocks = [b.copy() for b in blocks]
            for p in range(self.n_passes):
                bank: list[np.ndarray] = []
                bank_out: np.ndarray | None = None
                if p > 0:
                    bank = [b.copy() for b in blocks[len(prelude_blocks) + 1 :]]
                    bank_out = cur.copy()
                    cur = prelude.copy()
                    blocks = [b.copy() for b in prelude_blocks]

                for ip in range(self.core_start, self.core_end):
                    il = ip if p == 0 else self.n_phys + (p - 1) * (self.core_end - self.core_start) + (ip - self.core_start)
                    if p == 0:
                        cur = self._run_layer(il, ip, pos, tau, cur, blocks, None)
                    else:
                        assert bank_out is not None
                        mq = self.g.tensor_f64(f"blk.{ip}.loop_msg_query.weight")
                        msgs = mix_depth(bank, bank_out, mq)
                        cur = self._run_layer(il, ip, pos, tau, cur, blocks, msgs)

            for il in range(self.core_end, self.n_phys):
                cur = self._run_layer(il, il, pos, tau, cur, blocks, None)

        cur = mix_depth(blocks, cur, self.out_res_q)
        cur = rms_norm_weighted(cur, self.out_norm, self.rms_eps)
        logits = (self.out_w @ cur)[: self.vocab_unpadded]
        if self.softcap != 0.0:
            logits = self.softcap * np.tanh(logits / self.softcap)
        return logits.astype(np.float32)


def write_tmlg(path: Path, tokens: list[int], logits_list: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_steps = len(tokens)
    n_vocab = int(logits_list[0].shape[0])
    with open(path, "wb") as f:
        f.write(b"TMLG")
        f.write(struct.pack("<III", 1, n_steps, n_vocab))
        f.write(struct.pack(f"<{n_steps}I", *tokens))
        for lg in logits_list:
            f.write(np.asarray(lg, dtype=np.float32).tobytes())


def main() -> None:
    ap = argparse.ArgumentParser(description="Inkling-2-Mini (`inkling2_loop`) FP64 Oracle & Synthetic GGUF Generator")
    ap.add_argument("--gen-gguf", type=Path, help="Generate synthetic inkling2_loop Q8_0 GGUF at this path")
    ap.add_argument("--gguf", type=Path, help="Input GGUF path (defaults to --gen-gguf if omitted)")
    ap.add_argument("--tokens", type=str, default="5,12,29,41,7,63,18", help="Comma-separated token IDs (>=2)")
    ap.add_argument("--out", type=Path, required=True, help="Output TMLG v1 reference file")
    ap.add_argument("--seed", type=int, default=20261122)
    args = ap.parse_args()

    if args.gen_gguf is not None:
        generate_synthetic_inkling2_loop_gguf(args.gen_gguf, seed=args.seed)
        print(f"[gen-gguf] wrote {args.gen_gguf} ({args.gen_gguf.stat().st_size} bytes)")

    gguf_path = args.gguf or args.gen_gguf
    if gguf_path is None:
        raise SystemExit("Must pass --gguf or --gen-gguf")

    toks = [int(x) for x in args.tokens.split(",") if x.strip()]
    if len(toks) < 2:
        raise SystemExit("AGENTS.md Rule #7 requires >= 2 tokens")

    reader = GgufReader(gguf_path)
    oracle = Inkling2Oracle(reader)
    logits_steps: list[np.ndarray] = []
    for step, tid in enumerate(toks):
        lg = oracle.forward_token(tid, step)
        logits_steps.append(lg)
        top1 = int(np.argmax(lg))
        print(f"  step {step} (tok={tid:3d}): top1={top1:3d} max_logit={float(lg[top1]):+.6f} l2={float(np.linalg.norm(lg)):.6f}")

    write_tmlg(args.out, toks, logits_steps)
    print(f"[oracle] wrote {args.out} ({len(toks)} steps, vocab={logits_steps[0].shape[0]})")


if __name__ == "__main__":
    main()
