#!/usr/bin/env python3
"""Multi-token end-to-end logits comparison tool (`docs/TESTING.md` L3 & `AGENTS.md` Rule #7).

Supports two modes:
  1. Direct binary comparison of two `TMLG` v1 files:
       python3 tools/logits_diff.py --ref ref.bin --cand cand.bin
  2. Automated end-to-end comparison of `tinymodel` (CPU and/or Metal) against `llama.cpp`:
       python3 tools/logits_diff.py -m models/Qwen3.5-0.8B-Q4_0.gguf \
           -p "Qwen3.5 uses Gated DeltaNet linear attention in 18 of 24 layers: 2 + 2 = 4." \
           --both
"""

from __future__ import annotations

import argparse
import math
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import numpy as np
except ImportError:
    np = None  # Pure-stdlib fallback supported below

ORACLE_C_SRC = r"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "llama.h"

static void silent_log(enum ggml_log_level level, const char * text, void * user_data) {
    (void)level; (void)text; (void)user_data;
}

int main(int argc, char ** argv) {
    if (argc < 5) {
        fprintf(stderr, "usage: %s <model.gguf> <out.bin> <ngl> <t0,t1,...>\n", argv[0]);
        return 1;
    }
    const char * model_path = argv[1];
    const char * out_path   = argv[2];
    int ngl                 = atoi(argv[3]);
    char * tok_csv          = argv[4];

    llama_log_set(silent_log, NULL);
    llama_backend_init();

    int cap = 1024, n_toks = 0;
    llama_token * toks = (llama_token *)malloc(sizeof(llama_token) * cap);
    char * saveptr = NULL;
    for (char * p = strtok_r(tok_csv, ",", &saveptr); p != NULL; p = strtok_r(NULL, ",", &saveptr)) {
        if (n_toks >= cap) {
            cap *= 2;
            toks = (llama_token *)realloc(toks, sizeof(llama_token) * cap);
        }
        toks[n_toks++] = (llama_token)strtoul(p, NULL, 10);
    }

    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = ngl;
    if (ngl == 0) {
        mparams.use_extra_bufts = false;
    }
    struct llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) {
        fprintf(stderr, "failed to load model: %s\n", model_path);
        return 2;
    }

    const struct llama_vocab * vocab = llama_model_get_vocab(model);
    int32_t n_vocab = llama_vocab_n_tokens(vocab);

    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = (uint32_t)(n_toks + 64 < 512 ? 512 : n_toks + 64);
    cparams.n_batch = 512;
    cparams.n_threads = 4;
    struct llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        fprintf(stderr, "failed to init llama_context\n");
        return 3;
    }

    FILE * fp = fopen(out_path, "wb");
    if (!fp) {
        fprintf(stderr, "failed to open output: %s\n", out_path);
        return 4;
    }

    uint32_t ver = 1, u_steps = (uint32_t)n_toks, u_vocab = (uint32_t)n_vocab;
    fwrite("TMLG", 1, 4, fp);
    fwrite(&ver, 4, 1, fp);
    fwrite(&u_steps, 4, 1, fp);
    fwrite(&u_vocab, 4, 1, fp);
    for (int i = 0; i < n_toks; ++i) {
        uint32_t tid = (uint32_t)toks[i];
        fwrite(&tid, 4, 1, fp);
    }

    for (int i = 0; i < n_toks; ++i) {
        struct llama_batch batch = llama_batch_get_one(&toks[i], 1);
        if (llama_decode(ctx, batch) != 0) {
            fprintf(stderr, "llama_decode failed at step %d\n", i);
            fclose(fp);
            return 5;
        }
        const float * logits = llama_get_logits_ith(ctx, -1);
        fwrite(logits, sizeof(float), (size_t)n_vocab, fp);
    }

    fclose(fp);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    free(toks);
    return 0;
}
"""


def read_tmlg(path: Path):
    with open(path, "rb") as f:
        hdr = f.read(16)
        if len(hdr) < 16:
            raise ValueError(f"{path}: truncated header")
        magic, ver, n_steps, n_vocab = struct.unpack("<4sIII", hdr)
        if magic != b"TMLG" or ver != 1:
            raise ValueError(f"{path}: invalid magic/version {magic!r} v{ver}")
        tok_bytes = f.read(4 * n_steps)
        tokens = list(struct.unpack(f"<{n_steps}I", tok_bytes))
        raw_logits = f.read(4 * n_steps * n_vocab)
        if len(raw_logits) != 4 * n_steps * n_vocab:
            raise ValueError(f"{path}: truncated logits payload")
    if np is not None:
        mat = np.frombuffer(raw_logits, dtype=np.float32).reshape((n_steps, n_vocab)).copy()
    else:
        import array

        arr = array.array("f")
        arr.frombytes(raw_logits)
        mat = [arr[i * n_vocab : (i + 1) * n_vocab] for i in range(n_steps)]
    return tokens, n_vocab, mat


def ensure_llama_oracle(repo_root: Path) -> Path:
    bin_dir = repo_root / "third_party" / "llama.cpp" / "build" / "bin"
    inc_llama = repo_root / "third_party" / "llama.cpp" / "include"
    inc_ggml = repo_root / "third_party" / "llama.cpp" / "ggml" / "include"
    out_bin = repo_root / "target" / "llama_cpp_logits_oracle"
    src_file = repo_root / "target" / "llama_cpp_logits_oracle.c"
    out_bin.parent.mkdir(parents=True, exist_ok=True)

    if not src_file.exists() or src_file.read_text() != ORACLE_C_SRC:
        src_file.write_text(ORACLE_C_SRC)

    if not out_bin.exists() or out_bin.stat().st_mtime < src_file.stat().st_mtime:
        cmd = [
            "cc",
            "-O2",
            "-Wno-deprecated-declarations",
            f"-I{inc_llama}",
            f"-I{inc_ggml}",
            str(src_file),
            f"-L{bin_dir}",
            "-lllama",
            "-lggml",
            f"-Wl,-rpath,{bin_dir}",
            "-o",
            str(out_bin),
        ]
        subprocess.run(cmd, check=True)
    return out_bin


def compare_dumps(label: str, ref_toks, cand_toks, ref_mat, cand_mat, verbose: bool = True):
    if ref_toks != cand_toks:
        raise ValueError(f"Token sequence mismatch: ref={ref_toks} vs cand={cand_toks}")
    n_steps = len(ref_toks)
    if n_steps < 2:
        raise ValueError(
            f"AGENTS.md Rule #7 violation: multi-token test requires >= 2 tokens, got {n_steps}"
        )

    kls = []
    max_diffs = []
    maes = []
    top1_hits = 0
    top5_overlaps = []

    if verbose:
        print(f"\n=== Multi-Token Logits Diff: {label} ({n_steps} tokens) ===")
        print(
            f"{'Step':>4} {'InTok':>8} {'RefTop1':>9} {'CandTop1':>9} {'Match':>5} "
            f"{'KL(ref||cand)':>14} {'Max|dL|':>10} {'Top5Ovlp':>9}"
        )
        print("-" * 76)

    for t in range(n_steps):
        if np is not None:
            r = ref_mat[t].astype(np.float64)
            c = cand_mat[t].astype(np.float64)
            diff = np.abs(r - c)
            max_d = float(np.max(diff))
            mae_d = float(np.mean(diff))

            r_shift = r - np.max(r)
            c_shift = c - np.max(c)
            log_z_r = math.log(float(np.sum(np.exp(r_shift))))
            log_z_c = math.log(float(np.sum(np.exp(c_shift))))
            log_p = r_shift - log_z_r
            log_q = c_shift - log_z_c
            p = np.exp(log_p)
            kl = float(np.sum(p * (log_p - log_q)))
            kl = max(0.0, kl)

            r_top1 = int(np.argmax(r))
            c_top1 = int(np.argmax(c))
            r_top5 = set(int(x) for x in np.argpartition(r, -5)[-5:])
            c_top5 = set(int(x) for x in np.argpartition(c, -5)[-5:])
        else:
            r = ref_mat[t]
            c = cand_mat[t]
            max_d = max(abs(a - b) for a, b in zip(r, c))
            mae_d = sum(abs(a - b) for a, b in zip(r, c)) / len(r)
            mr = max(r)
            mc = max(c)
            er = [math.exp(x - mr) for x in r]
            ec = [math.exp(x - mc) for x in c]
            zr = math.log(sum(er))
            zc = math.log(sum(ec))
            inv_sr = 1.0 / sum(er)
            kl = max(
                0.0,
                sum(
                    (er_i * inv_sr) * ((r_i - mr - zr) - (c_i - mc - zc))
                    for er_i, r_i, c_i in zip(er, r, c)
                ),
            )
            r_top1 = max(range(len(r)), key=r.__getitem__)
            c_top1 = max(range(len(c)), key=c.__getitem__)
            r_top5 = set(sorted(range(len(r)), key=r.__getitem__, reverse=True)[:5])
            c_top5 = set(sorted(range(len(c)), key=c.__getitem__, reverse=True)[:5])

        hit = r_top1 == c_top1
        if hit:
            top1_hits += 1
        ovlp = len(r_top5 & c_top5) / 5.0
        kls.append(kl)
        max_diffs.append(max_d)
        maes.append(mae_d)
        top5_overlaps.append(ovlp)

        if verbose:
            print(
                f"{t:4d} {ref_toks[t]:8d} {r_top1:9d} {c_top1:9d} "
                f"{'  OK' if hit else 'DIFF':>5} {kl:14.3e} {max_d:10.4f} {ovlp * 100:8.1f}%"
            )

    mean_kl = sum(kls) / n_steps
    max_kl = max(kls)
    top1_rate = top1_hits / n_steps
    mean_top5 = sum(top5_overlaps) / n_steps
    worst_diff = max(max_diffs)
    mean_mae = sum(maes) / n_steps

    print("-" * 76)
    print(
        f"Summary ({label}): steps={n_steps} | Top-1={top1_hits}/{n_steps} ({top1_rate * 100:.1f}%) | "
        f"Top-5={mean_top5 * 100:.1f}% | mean_KL={mean_kl:.3e} | max_KL={max_kl:.3e} | "
        f"max|dL|={worst_diff:.4f} | MAE={mean_mae:.5f}"
    )
    return {
        "mean_kl": mean_kl,
        "max_kl": max_kl,
        "top1_rate": top1_rate,
        "top5_overlap": mean_top5,
        "max_diff": worst_diff,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Multi-token logits diff tool for tinymodel")
    ap.add_argument("--ref", type=Path, help="Reference TMLG v1 binary file")
    ap.add_argument("--cand", type=Path, help="Candidate TMLG v1 binary file")
    ap.add_argument("-m", "--model", type=Path, help="GGUF model path for automated llama.cpp comparison")
    ap.add_argument(
        "-p",
        "--prompt",
        default="The Gated DeltaNet hybrid architecture in Qwen3.5 computes 2 + 2 = 4 in Rust!",
        help="Multi-token prompt string",
    )
    ap.add_argument("--tokens", help="Explicit comma-separated token IDs")
    ap.add_argument("--metal", action="store_true", help="Test tinymodel Metal backend")
    ap.add_argument("--both", action="store_true", help="Test BOTH tinymodel CPU and Metal against llama.cpp")
    ap.add_argument("--ngl", type=int, default=99, help="n_gpu_layers for llama.cpp reference (default: 99)")
    ap.add_argument("--max-kl", type=float, default=1e-3, help="Max allowed mean KL divergence")
    ap.add_argument("--min-top1", type=float, default=0.95, help="Min allowed Top-1 agreement rate")
    args = ap.parse_args()

    if args.ref and args.cand:
        r_toks, _, r_mat = read_tmlg(args.ref)
        c_toks, _, c_mat = read_tmlg(args.cand)
        stats = compare_dumps(f"{args.ref.name} vs {args.cand.name}", r_toks, c_toks, r_mat, c_mat)
        if stats["mean_kl"] > args.max_kl or stats["top1_rate"] < args.min_top1:
            return 1
        return 0

    if not args.model:
        ap.error("provide either (--ref and --cand) or (-m <model.gguf>)")

    repo_root = Path(__file__).resolve().parent.parent
    tm_bin = repo_root / "target" / "release" / "tinymodel"
    if not tm_bin.exists():
        subprocess.run(["cargo", "build", "--release", "-p", "tm-cli"], cwd=repo_root, check=True)

    oracle_bin = ensure_llama_oracle(repo_root)

    with tempfile.TemporaryDirectory(prefix="tm_logits_") as tmpdir:
        tmp = Path(tmpdir)
        backends = ["cpu", "metal"] if args.both else (["metal"] if args.metal else ["cpu"])
        cand_files = {}
        toks_ref = None

        for bk in backends:
            out_f = tmp / f"tm_{bk}.bin"
            cmd = [str(tm_bin), "logits", "-m", str(args.model), "-o", str(out_f)]
            if args.tokens:
                cmd += ["--tokens", args.tokens]
            else:
                cmd += ["-p", args.prompt]
            if bk == "metal":
                cmd.append("--metal")
            subprocess.run(cmd, check=True)
            toks, _, mat = read_tmlg(out_f)
            toks_ref = toks
            cand_files[bk] = (toks, mat)

        ref_cache = {}
        failed = False
        for bk, (c_toks, c_mat) in cand_files.items():
            ref_ngl = 0 if (args.both and bk == "cpu") else args.ngl
            if ref_ngl not in ref_cache:
                ref_f = tmp / f"llama_cpp_ngl{ref_ngl}.bin"
                tok_csv = ",".join(str(x) for x in toks_ref)
                subprocess.run(
                    [str(oracle_bin), str(args.model), str(ref_f), str(ref_ngl), tok_csv],
                    check=True,
                )
                ref_cache[ref_ngl] = read_tmlg(ref_f)

            r_toks, _, r_mat = ref_cache[ref_ngl]
            kl_limit = args.max_kl if bk == "metal" else max(args.max_kl, 5e-3)
            top1_limit = args.min_top1 if bk == "metal" else min(args.min_top1, 0.90)
            stats = compare_dumps(
                f"llama.cpp(ngl={ref_ngl}) vs tinymodel({bk}) [{args.model.name}]",
                r_toks,
                c_toks,
                r_mat,
                c_mat,
            )
            if stats["mean_kl"] > kl_limit or stats["top1_rate"] < top1_limit:
                print(
                    f"FAIL [{bk}]: mean_kl={stats['mean_kl']:.3e} (limit {kl_limit:.3e}), "
                    f"top1={stats['top1_rate']:.3f} (limit {top1_limit:.3f})",
                    file=sys.stderr,
                )
                failed = True

        if "cpu" in cand_files and "metal" in cand_files:
            compare_dumps(
                f"tinymodel(cpu) vs tinymodel(metal) [{args.model.name}]",
                cand_files["cpu"][0],
                cand_files["metal"][0],
                cand_files["cpu"][1],
                cand_files["metal"][1],
            )

        return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
