#!/usr/bin/env bash
# M0/V0 baseline on a Linux AMD GPU host: llama.cpp Vulkan and HIP backends.
# Protocol: llama-bench -p 512 -n 128 -r 5 -ngl 99 -fa 1 (median reported by llama-bench).
#
# Usage: bench/run_amd_gpu_baseline.sh <out.json> [models_dir]
# Env:   LLAMA_DIR (default ~/llama.cpp), VK_DEVICE (GGML_VK_VISIBLE_DEVICES index, default 1)
set -euo pipefail
OUT=${1:?usage: $0 <out.json> [models_dir]}
MODELS=${2:-"$(dirname "$0")/../models"}
LLAMA_DIR=${LLAMA_DIR:-$HOME/llama.cpp}
if [ -z "${VK_DRIVER_FILES:-}" ] && [ -f "$HOME/.local/mesa-25.2.8/env.sh" ]; then
    # shellcheck disable=SC1091
    . "$HOME/.local/mesa-25.2.8/env.sh"
fi
VK_DEVICE=${VK_DEVICE:-1}
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

run() { # backend bin env...
    local backend=$1 bin=$2; shift 2
    for m in "$MODELS"/*.gguf; do
        echo "[$backend] $(basename "$m")" >&2
        env "$@" "$bin" -m "$m" -p 512 -n 128 -r 5 -ngl 99 -fa 1 -o json \
            > "$TMP/$backend-$(basename "$m").json" 2> "$TMP/$backend-$(basename "$m").log" \
            || echo "  failed (see log)" >&2
    done
}

run vulkan "$LLAMA_DIR/build-vulkan/bin/llama-bench" GGML_VK_VISIBLE_DEVICES="$VK_DEVICE"
run hip "$LLAMA_DIR/build-hip/bin/llama-bench" HIP_VISIBLE_DEVICES=0

python3 - "$TMP" "$OUT" "$LLAMA_DIR" <<'EOF'
import json, sys, glob, os, subprocess, platform
tmp, out, llama = sys.argv[1:4]
rows = []
for f in sorted(glob.glob(f"{tmp}/*.json")):
    backend = os.path.basename(f).split("-", 1)[0]
    try:
        data = json.load(open(f))
    except Exception:
        continue
    for r in data:
        rows.append({
            "backend": backend,
            "model": os.path.basename(r["model_filename"]),
            "model_size": r["model_size"],
            "n_prompt": r["n_prompt"],
            "n_gen": r["n_gen"],
            "avg_ts": r["avg_ts"],
            "stddev_ts": r["stddev_ts"],
            "gpu_info": r.get("gpu_info", ""),
            "build_commit": r.get("build_commit", ""),
        })
rev = subprocess.run(["git", "-C", llama, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
json.dump({"protocol": "llama-bench -p 512 -n 128 -r 5 -ngl 99 -fa 1",
           "llama_cpp_commit": rev, "host": platform.node(), "results": rows}, open(out, "w"), indent=1)
print(f"wrote {out} ({len(rows)} rows)")
EOF
