#!/usr/bin/env python3
"""
L2 Oracle for tinymodel `BlockQ4_0x4` (`Q4_0_4x8` offline zero-copy repacked layout).

Verifies `TESTING.md` L2 invariant:
    dequant_4rows(repack_q4_0x4(ggml_q4_0_4rows)) == dequant_ggml_q4_0(ggml_q4_0_4rows)
bit-for-bit across all 16 nibble values, negative/positive f16 scales, and random matrices,
and verifies integer dot products against `BlockQ8_0` match ggml's `vec_dot_q4_0_q8_0`.
"""

from __future__ import annotations

import random
import struct
import sys
from typing import List, Tuple

QK4_0 = 32
BLOCK_Q4_0_BYTES = 18
BLOCK_Q4_0X4_BYTES = 72


def f16_to_f32(bits: int) -> float:
    return struct.unpack("<e", struct.pack("<H", bits))[0]


def f32_to_f16(val: float) -> int:
    return struct.unpack("<H", struct.pack("<e", val))[0]


def dequant_ggml_q4_0_row(row_bytes: bytes, cols: int) -> List[float]:
    assert cols % QK4_0 == 0
    nb = cols // QK4_0
    assert len(row_bytes) == nb * BLOCK_Q4_0_BYTES
    out: List[float] = []
    for ib in range(nb):
        blk = row_bytes[ib * BLOCK_Q4_0_BYTES : (ib + 1) * BLOCK_Q4_0_BYTES]
        d = f16_to_f32(struct.unpack_from("<H", blk, 0)[0])
        qs = blk[2:18]
        lo = [((qs[j] & 0x0F) - 8) * d for j in range(16)]
        hi = [((qs[j] >> 4) - 8) * d for j in range(16)]
        out.extend(lo)
        out.extend(hi)
    return out


def repack_q4_0_to_q4_0x4(src_bytes: bytes, rows: int, cols: int) -> bytes:
    """
    Repacks `[rows, cols_blocks]` standard ggml `BlockQ4_0` into `[rows / 4, cols_blocks]`
    `BlockQ4_0x4` (72 bytes per 4x32 super-block, zero byte overhead):
      - bytes 0..8:   `u16 d[4]` (`[d0, d1, d2, d3]`)
      - bytes 8..16:  `r0.qs[0..8]`   (low nibbles k=0..8,  high nibbles k=16..24)
      - bytes 16..24: `r1.qs[0..8]`   (low nibbles k=0..8,  high nibbles k=16..24)
      - bytes 24..32: `r0.qs[8..16]`  (low nibbles k=8..16, high nibbles k=24..32)
      - bytes 32..40: `r1.qs[8..16]`  (low nibbles k=8..16, high nibbles k=24..32)
      - bytes 40..48: `r2.qs[0..8]`
      - bytes 48..56: `r3.qs[0..8]`
      - bytes 56..64: `r2.qs[8..16]`
      - bytes 64..72: `r3.qs[8..16]`
    """
    assert rows % 4 == 0 and cols % QK4_0 == 0
    nb = cols // QK4_0
    row_stride = nb * BLOCK_Q4_0_BYTES
    assert len(src_bytes) == rows * row_stride

    out = bytearray()
    for r4 in range(0, rows, 4):
        for ib in range(nb):
            blks = [
                src_bytes[(r4 + m) * row_stride + ib * 18 : (r4 + m) * row_stride + (ib + 1) * 18]
                for m in range(4)
            ]
            # 4 x u16 scales
            for m in range(4):
                out.extend(blks[m][0:2])
            # Pair (r0, r1): [r0_0..8, r1_0..8, r0_8..16, r1_8..16]
            out.extend(blks[0][2:10])
            out.extend(blks[1][2:10])
            out.extend(blks[0][10:18])
            out.extend(blks[1][10:18])
            # Pair (r2, r3): [r2_0..8, r3_0..8, r2_8..16, r3_8..16]
            out.extend(blks[2][2:10])
            out.extend(blks[3][2:10])
            out.extend(blks[2][10:18])
            out.extend(blks[3][10:18])

    assert len(out) == len(src_bytes)
    return bytes(out)


def dequant_q4_0x4_row(repacked: bytes, rows: int, cols: int, row: int) -> List[float]:
    assert 0 <= row < rows and rows % 4 == 0 and cols % QK4_0 == 0
    nb = cols // QK4_0
    r_group = row // 4
    sub = row % 4
    pair_off = 0 if sub < 2 else 32
    lane_off = 0 if (sub % 2 == 0) else 8

    out: List[float] = []
    group_base = r_group * nb * BLOCK_Q4_0X4_BYTES
    for ib in range(nb):
        sb = repacked[group_base + ib * 72 : group_base + (ib + 1) * 72]
        d = f16_to_f32(struct.unpack_from("<H", sb, sub * 2)[0])
        qs = sb[8:72]
        b0 = qs[pair_off + lane_off : pair_off + lane_off + 8]
        b1 = qs[pair_off + 16 + lane_off : pair_off + 16 + lane_off + 8]
        q_lo = [((b & 0x0F) - 8) * d for b in b0] + [((b & 0x0F) - 8) * d for b in b1]
        q_hi = [((b >> 4) - 8) * d for b in b0] + [((b >> 4) - 8) * d for b in b1]
        out.extend(q_lo)
        out.extend(q_hi)
    return out


def smmla_2x2(a16: List[int], b16: List[int]) -> Tuple[int, int, int, int]:
    """Simulates ARMv8.6 `smmla v.4s, a.16b, b.16b` (`a`=[2x8], `b`=[2x8])."""
    c00 = sum(a16[k] * b16[k] for k in range(8))
    c01 = sum(a16[k] * b16[8 + k] for k in range(8))
    c10 = sum(a16[8 + k] * b16[k] for k in range(8))
    c11 = sum(a16[8 + k] * b16[8 + k] for k in range(8))
    return c00, c01, c10, c11


def main() -> int:
    rng = random.Random(0x2026_1121)
    for rows, cols in [(4, 32), (8, 64), (12, 96), (20, 256)]:
        nb = cols // QK4_0
        raw = bytearray()
        for r in range(rows):
            for ib in range(nb):
                scale = (rng.randint(-500, 500) / 250.0) if (r + ib > 0) else 0.125
                raw.extend(struct.pack("<H", f32_to_f16(scale)))
                if r == 0 and ib == 0:
                    raw.extend(bytes((j | ((15 - j) << 4)) for j in range(16)))
                else:
                    raw.extend(bytes(rng.randrange(256) for _ in range(16)))

        repacked = repack_q4_0_to_q4_0x4(bytes(raw), rows, cols)

        # 1. Verify exact bit-for-bit dequantization equality across every row & element
        for r in range(rows):
            orig_row = dequant_ggml_q4_0_row(
                bytes(raw[r * nb * 18 : (r + 1) * nb * 18]), cols
            )
            got_row = dequant_q4_0x4_row(repacked, rows, cols, r)
            if orig_row != got_row:
                print(f"FAIL: dequant mismatch at rows={rows} cols={cols} r={r}", file=sys.stderr)
                return 1

        # 2. Verify zero-shuffle `smmla` integer dot product matches scalar dot product exactly
        q8_tok0 = [rng.randint(-127, 127) for _ in range(32)]
        q8_tok1 = [rng.randint(-127, 127) for _ in range(32)]
        sb0 = repacked[0:72]
        qs = sb0[8:72]
        v0, v1 = qs[0:16], qs[16:32]
        la0 = [(b & 0x0F) - 8 for b in v0]
        la2 = [(b >> 4) - 8 for b in v0]
        la1 = [(b & 0x0F) - 8 for b in v1]
        la3 = [(b >> 4) - 8 for b in v1]
        rr0 = q8_tok0[0:8] + q8_tok1[0:8]
        rr1 = q8_tok0[8:16] + q8_tok1[8:16]
        rr2 = q8_tok0[16:24] + q8_tok1[16:24]
        rr3 = q8_tok0[24:32] + q8_tok1[24:32]
        acc = [0, 0, 0, 0]
        for la, rr in [(la0, rr0), (la1, rr1), (la2, rr2), (la3, rr3)]:
            p = smmla_2x2(la, rr)
            acc = [acc[i] + p[i] for i in range(4)]

        for m, tok in [(0, q8_tok0), (0, q8_tok1), (1, q8_tok0), (1, q8_tok1)]:
            blk_m = raw[m * nb * 18 : m * nb * 18 + 18]
            q_orig = [((blk_m[2 + j] & 0x0F) - 8) for j in range(16)] + [
                ((blk_m[2 + j] >> 4) - 8) for j in range(16)
            ]
            ref_sumi = sum(q_orig[k] * tok[k] for k in range(32))
            idx = m * 2 + (0 if tok is q8_tok0 else 1)
            assert acc[idx] == ref_sumi, f"smmla mismatch: {acc[idx]} != {ref_sumi}"

    print("PASS: Q4_0x4 oracle dequantization & zero-shuffle smmla verified bit-exact.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
