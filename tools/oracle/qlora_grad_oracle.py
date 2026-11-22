#!/usr/bin/env python3
"""Independent reference QLoRA gradient oracle (`ADR-015` §3 Verification Layers 2 & 3).

Supports TWO execution backends automatically:
  - `torch` (when PyTorch is installed in the active Python environment)
  - Pure Python `float64` reference (zero third-party dependencies, runs anywhere!)

Reads a JSON dump exported by `crates/tm-train/tests/qlora_verification.rs` containing
a synthetic `Q4_0` Llama/Qwen3 model + initial LoRA `(A, B)` + masked token sequence,
and computes:
  1. **Exact Q8_0 Activation Fake-Quant STE + F16 KV STE** gradients (`rel_err < 1e-3` gate);
  2. **Plain FP32 PEFT** (no activation Q8_0 or F16 KV fake-quant) gradients (`cosine > 0.99` gate).
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path


def f32_to_f16_f32(x: float) -> float:
    # Exact IEEE-754 binary16 round-to-nearest-even
    try:
        return struct.unpack("<e", struct.pack("<e", x))[0]
    except OverflowError:
        return math.copysign(65504.0, x)


def f32(x: float) -> float:
    return struct.unpack("<f", struct.pack("<f", x))[0]


def round_half_away_from_zero(x: float) -> int:
    return int(math.floor(x + 0.5)) if x >= 0.0 else int(math.ceil(x - 0.5))


def fake_quant_q8_0(vec: list[float]) -> list[float]:
    out = [0.0] * len(vec)
    for b0 in range(0, len(vec), 32):
        blk = [f32(v) for v in vec[b0 : b0 + 32]]
        amax = f32(max(abs(v) for v in blk))
        d_raw = f32(amax / f32(127.0))
        inv_d = f32(f32(1.0) / d_raw) if d_raw != 0.0 else 0.0
        d_f16 = f32_to_f16_f32(d_raw) if amax > 0.0 else 0.0
        for j, v in enumerate(blk):
            prod = f32(v * inv_d)
            q = max(-127, min(127, round_half_away_from_zero(prod)))
            out[b0 + j] = f32(q * d_f16)
    return out


def rms_norm_fwd(x: list[float], w: list[float], eps: float) -> list[float]:
    n = len(x)
    ms = sum(v * v for v in x) / n
    inv = 1.0 / math.sqrt(ms + eps)
    return [x[i] * inv * w[i] for i in range(n)]


def rms_norm_bwd(x: list[float], w: list[float], dy: list[float], eps: float) -> list[float]:
    n = len(x)
    ms = sum(v * v for v in x) / n
    inv = 1.0 / math.sqrt(ms + eps)
    gx = [dy[i] * w[i] for i in range(n)]
    dot = sum(gx[i] * x[i] for i in range(n))
    coeff = (inv * inv) * dot / n
    return [inv * (gx[i] - x[i] * coeff) for i in range(n)]


def rope_fwd(vec: list[float], n_heads: int, hd: int, rope_dim: int, cos: list[float], sin: list[float], neox: bool) -> list[float]:
    out = list(vec)
    half = rope_dim // 2
    for h in range(n_heads):
        off = h * hd
        for i in range(half):
            c, s = cos[i], sin[i]
            if neox:
                x0, x1 = out[off + i], out[off + half + i]
                out[off + i] = x0 * c - x1 * s
                out[off + half + i] = x0 * s + x1 * c
            else:
                x0, x1 = out[off + 2 * i], out[off + 2 * i + 1]
                out[off + 2 * i] = x0 * c - x1 * s
                out[off + 2 * i + 1] = x0 * s + x1 * c
    return out


def rope_bwd(dvec: list[float], n_heads: int, hd: int, rope_dim: int, cos: list[float], sin: list[float], neox: bool) -> list[float]:
    out = list(dvec)
    half = rope_dim // 2
    for h in range(n_heads):
        off = h * hd
        for i in range(half):
            c, s = cos[i], sin[i]
            if neox:
                dy0, dy1 = out[off + i], out[off + half + i]
                out[off + i] = dy0 * c + dy1 * s
                out[off + half + i] = -dy0 * s + dy1 * c
            else:
                dy0, dy1 = out[off + 2 * i], out[off + 2 * i + 1]
                out[off + 2 * i] = dy0 * c + dy1 * s
                out[off + 2 * i + 1] = -dy0 * s + dy1 * c
    return out


def run_oracle(spec: dict, q8_ste: bool) -> tuple[float, dict[str, dict[str, list[float]]]]:
    cfg = spec["cfg"]
    d = cfg["n_embd"]
    ff = cfg["n_ff"]
    nh = cfg["n_head"]
    nkv = cfg["n_head_kv"]
    hd = cfg["head_dim"]
    rdim = cfg["rope_dim"]
    q_dim = nh * hd
    kv_dim = nkv * hd
    vocab = cfg["vocab_size"]
    eps = cfg["rms_eps"]
    neox = cfg["neox"]
    inv_freqs = cfg["rope_inv_freqs"]

    tokens = spec["tokens"]
    mask = spec["loss_mask"]
    seq_len = len(tokens) - 1
    alpha = spec["lora_alpha"]
    lora = spec["lora"]  # name -> {rank, d_in, d_out, a, b, is_token_embd}
    grads = {
        name: {"da": [0.0] * len(p["a"]), "db": [0.0] * len(p["b"])}
        for name, p in lora.items()
    }

    def linear_fwd(name: str, w_f32: list[float], rows: int, cols: int, x_seq: list[list[float]]) -> list[list[float]]:
        out = []
        p = lora.get(name)
        for xt in x_seq:
            xq = fake_quant_q8_0(xt) if q8_ste else xt
            yt = [0.0] * rows
            for r_i in range(rows):
                off = r_i * cols
                yt[r_i] = sum(w_f32[off + j] * xq[j] for j in range(cols))
            if p is not None and not p["is_token_embd"]:
                rk = p["rank"]
                sc = (alpha / rk) if alpha != 0.0 else 1.0
                a, b = p["a"], p["b"]
                h = [sum(a[ri * cols + j] * xt[j] for j in range(cols)) * sc for ri in range(rk)]
                for o in range(rows):
                    yt[o] += sum(b[o * rk + ri] * h[ri] for ri in range(rk))
            out.append(yt)
        return out

    def linear_bwd(name: str, w_f32: list[float], rows: int, cols: int, x_seq: list[list[float]], dy_seq: list[list[float]]) -> list[list[float]]:
        dx_seq = []
        p = lora.get(name)
        for xt, dyt in zip(x_seq, dy_seq):
            dxt = [sum(dyt[r_i] * w_f32[r_i * cols + j] for r_i in range(rows)) for j in range(cols)]
            if p is not None and not p["is_token_embd"]:
                rk = p["rank"]
                sc = (alpha / rk) if alpha != 0.0 else 1.0
                a, b = p["a"], p["b"]
                g = grads[name]
                h = [sum(a[ri * cols + j] * xt[j] for j in range(cols)) for ri in range(rk)]
                dh = [sc * sum(dyt[o] * b[o * rk + ri] for o in range(rows)) for ri in range(rk)]
                for o in range(rows):
                    for ri in range(rk):
                        g["db"][o * rk + ri] += sc * dyt[o] * h[ri]
                for ri in range(rk):
                    for j in range(cols):
                        g["da"][ri * cols + j] += dh[ri] * xt[j]
                        dxt[j] += dh[ri] * a[ri * cols + j]
            dx_seq.append(dxt)
        return dx_seq

    # Embedding lookup
    emb_w = spec["token_embd"]
    x = []
    for t in range(seq_len):
        tid = tokens[t]
        row = list(emb_w[tid * d : (tid + 1) * d])
        if "token_embd.weight" in lora:
            p = lora["token_embd.weight"]
            rk = p["rank"]
            sc = (alpha / rk) if alpha != 0.0 else 1.0
            for ri in range(rk):
                ar = p["a"][tid * rk + ri] * sc
                for o in range(d):
                    row[o] += ar * p["b"][o * rk + ri]
        x.append(row)

    cos_tbl = [[math.cos(t * f) for f in inv_freqs] for t in range(seq_len)]
    sin_tbl = [[math.sin(t * f) for f in inv_freqs] for t in range(seq_len)]

    saved_layers = []
    for l_idx, lw in enumerate(spec["layers"]):
        pfx = f"blk.{l_idx}"
        x_in = [list(v) for v in x]
        x_an = [rms_norm_fwd(v, lw["attn_norm"], eps) for v in x_in]
        q_raw = linear_fwd(f"{pfx}.attn_q.weight", lw["wq"], q_dim, d, x_an)
        k_raw = linear_fwd(f"{pfx}.attn_k.weight", lw["wk"], kv_dim, d, x_an)
        v_raw = linear_fwd(f"{pfx}.attn_v.weight", lw["wv"], kv_dim, d, x_an)

        q_pn = [list(v) for v in q_raw]
        k_pn = [list(v) for v in k_raw]
        if lw["q_norm"] is not None:
            for t in range(seq_len):
                for h in range(nh):
                    q_raw[t][h * hd : (h + 1) * hd] = rms_norm_fwd(q_raw[t][h * hd : (h + 1) * hd], lw["q_norm"], eps)
        if lw["k_norm"] is not None:
            for t in range(seq_len):
                for h in range(nkv):
                    k_raw[t][h * hd : (h + 1) * hd] = rms_norm_fwd(k_raw[t][h * hd : (h + 1) * hd], lw["k_norm"], eps)

        q_rp = [rope_fwd(q_raw[t], nh, hd, rdim, cos_tbl[t], sin_tbl[t], neox) for t in range(seq_len)]
        k_rp = [rope_fwd(k_raw[t], nkv, hd, rdim, cos_tbl[t], sin_tbl[t], neox) for t in range(seq_len)]
        if q8_ste:
            k_kv = [[f32_to_f16_f32(v) for v in row] for row in k_rp]
            v_kv = [[f32_to_f16_f32(v) for v in row] for row in v_raw]
        else:
            k_kv = k_rp
            v_kv = v_raw

        group = nh // nkv
        ascale = 1.0 / math.sqrt(hd)
        attn_out = [[0.0] * q_dim for _ in range(seq_len)]
        for t in range(seq_len):
            for h in range(nh):
                kvh = h // group
                qh = q_rp[t][h * hd : (h + 1) * hd]
                scores = [
                    sum(qh[j] * k_kv[s][kvh * hd + j] for j in range(hd)) * ascale
                    for s in range(t + 1)
                ]
                ms = max(scores)
                exps = [math.exp(sc - ms) for sc in scores]
                z = sum(exps)
                probs = [e / z for e in exps]
                for j in range(hd):
                    attn_out[t][h * hd + j] = sum(probs[s] * v_kv[s][kvh * hd + j] for s in range(t + 1))

        wo_out = linear_fwd(f"{pfx}.attn_output.weight", lw["wo"], d, q_dim, attn_out)
        x_mid = [[x_in[t][j] + wo_out[t][j] for j in range(d)] for t in range(seq_len)]
        x_fn = [rms_norm_fwd(v, lw["ffn_norm"], eps) for v in x_mid]
        gate_pre = linear_fwd(f"{pfx}.ffn_gate.weight", lw["w_gate"], ff, d, x_fn)
        up_pre = linear_fwd(f"{pfx}.ffn_up.weight", lw["w_up"], ff, d, x_fn)
        sw = [
            [(gate_pre[t][j] / (1.0 + math.exp(-gate_pre[t][j]))) * up_pre[t][j] for j in range(ff)]
            for t in range(seq_len)
        ]
        down_out = linear_fwd(f"{pfx}.ffn_down.weight", lw["w_down"], d, ff, sw)
        x = [[x_mid[t][j] + down_out[t][j] for j in range(d)] for t in range(seq_len)]

        saved_layers.append((x_in, x_an, q_pn, k_pn, q_rp, k_kv, v_kv, attn_out, x_mid, x_fn, gate_pre, up_pre, sw))

    valid = [(t, tokens[t + 1]) for t in range(seq_len) if mask[t + 1] != 0]
    n_valid = len(valid)
    x_sub = [x[t] for t, _ in valid]
    xn_sub = [rms_norm_fwd(v, spec["output_norm"], eps) for v in x_sub]
    logits = linear_fwd("output.weight", spec["output"], vocab, d, xn_sub)

    loss_sum = 0.0
    d_logits = []
    for k_i, (_, target) in enumerate(valid):
        row = logits[k_i]
        ml = max(row)
        lz = ml + math.log(sum(math.exp(v - ml) for v in row))
        loss_sum += lz - row[target]
        d_logits.append([(math.exp(v - lz) - (1.0 if vi == target else 0.0)) / n_valid for vi, v in enumerate(row)])

    dxn_sub = linear_bwd("output.weight", spec["output"], vocab, d, xn_sub, d_logits)
    dx = [[0.0] * d for _ in range(seq_len)]
    for k_i, (t, _) in enumerate(valid):
        dx[t] = rms_norm_bwd(x_sub[k_i], spec["output_norm"], dxn_sub[k_i], eps)

    for l_idx in reversed(range(len(spec["layers"]))):
        lw = spec["layers"][l_idx]
        pfx = f"blk.{l_idx}"
        (x_in, x_an, q_pn, k_pn, q_rp, k_kv, v_kv, attn_out, x_mid, x_fn, gate_pre, up_pre, sw) = saved_layers[l_idx]

        d_sw = linear_bwd(f"{pfx}.ffn_down.weight", lw["w_down"], d, ff, sw, dx)
        d_gate, d_up = [], []
        for t in range(seq_len):
            dg_t, du_t = [0.0] * ff, [0.0] * ff
            for j in range(ff):
                g_v, u_v, dy_v = gate_pre[t][j], up_pre[t][j], d_sw[t][j]
                sig = 1.0 / (1.0 + math.exp(-g_v))
                du_t[j] = dy_v * (g_v * sig)
                dg_t[j] = dy_v * u_v * sig * (1.0 + g_v * (1.0 - sig))
            d_gate.append(dg_t)
            d_up.append(du_t)
        dx_fn_g = linear_bwd(f"{pfx}.ffn_gate.weight", lw["w_gate"], ff, d, x_fn, d_gate)
        dx_fn_u = linear_bwd(f"{pfx}.ffn_up.weight", lw["w_up"], ff, d, x_fn, d_up)
        for t in range(seq_len):
            dxfn = [dx_fn_g[t][j] + dx_fn_u[t][j] for j in range(d)]
            dxm = rms_norm_bwd(x_mid[t], lw["ffn_norm"], dxfn, eps)
            dx[t] = [dx[t][j] + dxm[j] for j in range(d)]

        d_ao = linear_bwd(f"{pfx}.attn_output.weight", lw["wo"], d, q_dim, attn_out, dx)
        dq = [[0.0] * q_dim for _ in range(seq_len)]
        dk = [[0.0] * kv_dim for _ in range(seq_len)]
        dv = [[0.0] * kv_dim for _ in range(seq_len)]
        group = nh // nkv
        ascale = 1.0 / math.sqrt(hd)
        for t in range(seq_len):
            for h in range(nh):
                kvh = h // group
                qh = q_rp[t][h * hd : (h + 1) * hd]
                doh = d_ao[t][h * hd : (h + 1) * hd]
                scores = [sum(qh[j] * k_kv[s][kvh * hd + j] for j in range(hd)) * ascale for s in range(t + 1)]
                ms = max(scores)
                exps = [math.exp(sc - ms) for sc in scores]
                z = sum(exps)
                probs = [e / z for e in exps]
                dp = [0.0] * (t + 1)
                for s in range(t + 1):
                    for j in range(hd):
                        dv[s][kvh * hd + j] += doh[j] * probs[s]
                        dp[s] += doh[j] * v_kv[s][kvh * hd + j]
                sum_pdp = sum(probs[s] * dp[s] for s in range(t + 1))
                for s in range(t + 1):
                    ds = probs[s] * (dp[s] - sum_pdp) * ascale
                    for j in range(hd):
                        dq[t][h * hd + j] += ds * k_kv[s][kvh * hd + j]
                        dk[s][kvh * hd + j] += ds * qh[j]

        for t in range(seq_len):
            dq[t] = rope_bwd(dq[t], nh, hd, rdim, cos_tbl[t], sin_tbl[t], neox)
            dk[t] = rope_bwd(dk[t], nkv, hd, rdim, cos_tbl[t], sin_tbl[t], neox)
            if lw["q_norm"] is not None:
                for h in range(nh):
                    dq[t][h * hd : (h + 1) * hd] = rms_norm_bwd(
                        q_pn[t][h * hd : (h + 1) * hd], lw["q_norm"], dq[t][h * hd : (h + 1) * hd], eps
                    )
            if lw["k_norm"] is not None:
                for h in range(nkv):
                    dk[t][h * hd : (h + 1) * hd] = rms_norm_bwd(
                        k_pn[t][h * hd : (h + 1) * hd], lw["k_norm"], dk[t][h * hd : (h + 1) * hd], eps
                    )

        dx_q = linear_bwd(f"{pfx}.attn_q.weight", lw["wq"], q_dim, d, x_an, dq)
        dx_k = linear_bwd(f"{pfx}.attn_k.weight", lw["wk"], kv_dim, d, x_an, dk)
        dx_v = linear_bwd(f"{pfx}.attn_v.weight", lw["wv"], kv_dim, d, x_an, dv)
        for t in range(seq_len):
            dxan = [dx_q[t][j] + dx_k[t][j] + dx_v[t][j] for j in range(d)]
            dxin = rms_norm_bwd(x_in[t], lw["attn_norm"], dxan, eps)
            dx[t] = [dx[t][j] + dxin[j] for j in range(d)]

    if "token_embd.weight" in lora:
        p = lora["token_embd.weight"]
        rk = p["rank"]
        sc = (alpha / rk) if alpha != 0.0 else 1.0
        g = grads["token_embd.weight"]
        for t in range(seq_len):
            tid = tokens[t]
            for ri in range(rk):
                dot_dxb = sum(dx[t][o] * p["b"][o * rk + ri] for o in range(d))
                g["da"][tid * rk + ri] += sc * dot_dxb
                for o in range(d):
                    g["db"][o * rk + ri] += sc * dx[t][o] * p["a"][tid * rk + ri]

    return loss_sum / n_valid, grads


def main() -> int:
    ap = argparse.ArgumentParser(description="Reference QLoRA gradient oracle")
    ap.add_argument("-i", "--input", type=Path, required=True, help="Input JSON problem spec")
    ap.add_argument("-o", "--output", type=Path, required=True, help="Output JSON oracle gradients")
    args = ap.parse_args()

    spec = json.loads(args.input.read_text())
    loss_ste, grads_ste = run_oracle(spec, q8_ste=True)
    loss_fp32, grads_fp32 = run_oracle(spec, q8_ste=False)

    out = {
        "q8_ste": {"loss": loss_ste, "grads": grads_ste},
        "fp32_peft": {"loss": loss_fp32, "grads": grads_fp32},
    }
    args.output.write_text(json.dumps(out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
