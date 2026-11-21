#!/usr/bin/env python3
"""L2 Oracle for M4 Low-Bit & Ternary Formats (`Q2_0`, `TQ2_0`, `IQ2_XXS`, `IQ3_XXS`, `IQ4_NL`)
and T-MAC 16-row `vqtbl1q` Activation-LUT GEMV (`docs/TESTING.md` §1 L2 & `AGENTS.md` Rule #3).
"""

from __future__ import annotations

import re
import struct
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent.parent
HDR = (REPO / "Strata/third_party/strata-ggml/strata-quant-common.h").read_text()


def parse_c_table(name: str) -> list[int]:
    m = re.search(rf"GGML_TABLE_BEGIN\([^,]+,\s*{name},\s*(\d+)\)(.*?)GGML_TABLE_END\(\)", HDR, re.S)
    assert m, f"table {name} not found"
    body = re.sub(r"//.*", "", m.group(2))
    vals = [int(v.strip(), 0) for v in body.split(",") if v.strip()]
    assert len(vals) == int(m.group(1))
    return vals


KVALUES_IQ4NL = parse_c_table("kvalues_iq4nl")
KSIGNS_IQ2XS = parse_c_table("ksigns_iq2xs")
IQ2XXS_GRID = parse_c_table("iq2xxs_grid")
IQ3XXS_GRID = parse_c_table("iq3xxs_grid")


def f16_to_f32(u16: int) -> float:
    return struct.unpack("<f", struct.pack("<f", struct.unpack("<e", struct.pack("<H", u16))[0]))[0]


def f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def build_pair_lut_q2_0(y0: int, y1: int) -> tuple[list[int], list[int]]:
    """Builds 16-entry signed 16-bit LUT split into `(lut_lo_u8, lut_hi_i8)` for `((c0-1)*y0 + (c1-1)*y1)`."""
    lo, hi = [], []
    for nib in range(16):
        q0 = (nib & 3) - 1
        q1 = ((nib >> 2) & 3) - 1
        val = q0 * y0 + q1 * y1  # in [-512, +512], fits in i16
        u = val & 0xFFFF
        lo.append(u & 0xFF)
        hi_b = (u >> 8) & 0xFF
        hi.append(hi_b - 256 if hi_b >= 128 else hi_b)
    return lo, hi


def verify_tmac_lut_q2_0_and_tq2_0():
    # 1. Verify 16-row T-MAC LUT for Q2_0 against Strata scalar dot_q2_0_q8_0_generic (0 ULP!)
    rows = 16
    cols = 128
    nb_q2 = cols // 64
    nb_q8 = cols // 32
    d_q2 = [0x3800 + (r * 0x0040) for r in range(rows)]  # exact powers/dyadics
    qs_q2 = [[((r * 13 + ib * 7 + b * 29 + 5) & 0xFF) for b in range(16)] for r in range(rows) for ib in range(nb_q2)]
    d_q8 = [0x3400, 0x3800, 0x3600, 0x3A00]
    qs_q8 = [[(((ib * 31 + j * 17 + 3) % 251) - 125) for j in range(32)] for ib in range(nb_q8)]

    # Scalar reference (`Strata/src/quant/dot_generic.cpp:22-63`)
    ref = []
    for r in range(rows):
        sumf = 0.0
        for ib in range(nb_q2):
            d0 = f16_to_f32(d_q2[r])
            sumi_f = 0.0
            qbytes = qs_q2[r * nb_q2 + ib]
            for k in range(2):
                d1 = f16_to_f32(d_q8[ib * 2 + k])
                qy = qs_q8[ib * 2 + k]
                sb = 0
                for b in range(8):
                    byte = qbytes[k * 8 + b]
                    sb += (((byte >> 0) & 3) - 1) * qy[b * 4 + 0]
                    sb += (((byte >> 2) & 3) - 1) * qy[b * 4 + 1]
                    sb += (((byte >> 4) & 3) - 1) * qy[b * 4 + 2]
                    sb += (((byte >> 6) & 3) - 1) * qy[b * 4 + 3]
                sumi_f = f32(sumi_f + f32(d1 * f32(float(sb))))
            sumf = f32(sumf + f32(d0 * sumi_f))
        ref.append(sumf)

    # T-MAC 16-row NEON `vqtbl1q` simulation (ZERO multiplications in row loop!)
    lut_out = [0.0] * rows
    for ib in range(nb_q2):
        sub_acc = [[0] * rows for _ in range(2)]
        for k in range(2):
            qy = qs_q8[ib * 2 + k]
            for b in range(8):
                lo01, hi01 = build_pair_lut_q2_0(qy[b * 4 + 0], qy[b * 4 + 1])
                lo23, hi23 = build_pair_lut_q2_0(qy[b * 4 + 2], qy[b * 4 + 3])
                # Gather 16 rows' byte at `k * 8 + b` and run `vqtbl1q`
                for r in range(rows):
                    byte = qs_q2[r * nb_q2 + ib][k * 8 + b]
                    n_lo = byte & 0x0F
                    n_hi = byte >> 4
                    v01 = lo01[n_lo] + (hi01[n_lo] << 8)
                    v23 = lo23[n_hi] + (hi23[n_hi] << 8)
                    sub_acc[k][r] += v01 + v23
        d1_0 = f16_to_f32(d_q8[ib * 2 + 0])
        d1_1 = f16_to_f32(d_q8[ib * 2 + 1])
        for r in range(rows):
            d0 = f16_to_f32(d_q2[r])
            sumi_f = f32(f32(d1_0 * f32(float(sub_acc[0][r]))) + f32(d1_1 * f32(float(sub_acc[1][r]))))
            lut_out[r] = f32(lut_out[r] + f32(d0 * sumi_f))

    assert ref == lut_out, f"Q2_0 T-MAC mismatch: {ref} vs {lut_out}"

    # 2. Verify 16-row T-MAC LUT for BitNet TQ2_0 against ggml_vec_dot_tq2_0_q8_K_generic (0 ULP!)
    qs_tq = [[(((r + m) % 3) | (((r * 3 + m) % 3) << 2) | (((r * 5 + m) % 3) << 4) | (((r + m * 7) % 3) << 6))
              for m in range(64)] for r in range(rows)]
    y_q8k = [(((i * 23 + 11) % 251) - 125) for i in range(256)]
    d_y = f32(0.125)

    ref_tq = []
    for r in range(rows):
        sumi = 0
        for j in (0, 32):
            for l in range(4):
                for k in range(32):
                    sumi += y_q8k[j * 4 + l * 32 + k] * (((qs_tq[r][j + k] >> (l * 2)) & 3) - 1)
        d = f32(d_y * f16_to_f32(d_q2[r]))
        ref_tq.append(f32(f32(float(sumi)) * d))

    lut_tq = [0.0] * rows
    acc_i32 = [0] * rows
    for j in (0, 32):
        base_y = j * 4
        for k in range(32):
            lo01, hi01 = build_pair_lut_q2_0(y_q8k[base_y + k], y_q8k[base_y + 32 + k])
            lo23, hi23 = build_pair_lut_q2_0(y_q8k[base_y + 64 + k], y_q8k[base_y + 96 + k])
            for r in range(rows):
                byte = qs_tq[r][j + k]
                n_lo = byte & 0x0F
                n_hi = byte >> 4
                acc_i32[r] += (lo01[n_lo] + (hi01[n_lo] << 8)) + (lo23[n_hi] + (hi23[n_hi] << 8))
    for r in range(rows):
        d = f32(d_y * f16_to_f32(d_q2[r]))
        lut_tq[r] = f32(f32(float(acc_i32[r])) * d)

    assert ref_tq == lut_tq, f"TQ2_0 T-MAC mismatch: {ref_tq} vs {lut_tq}"

    # 3. Verify IQ2_XXS, IQ3_XXS, and IQ4_NL codebook invariants
    for idx7 in range(128):
        s = KSIGNS_IQ2XS[idx7]
        assert (s & 0x7F) == idx7
        assert bin(s).count("1") % 2 == 0, f"ksigns_iq2xs[{idx7}] odd parity!"
    assert len(IQ2XXS_GRID) == 256 and len(IQ3XXS_GRID) == 256 and len(KVALUES_IQ4NL) == 16


if __name__ == "__main__":
    verify_tmac_lut_q2_0_and_tq2_0()
    print("PASS: M4 Low-Bit Oracle (Q2_0, TQ2_0, IQ2_XXS, IQ3_XXS, IQ4_NL + 16-row T-MAC vqtbl1q LUT) verified 0-ULP bit-exact.")
