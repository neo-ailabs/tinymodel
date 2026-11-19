#!/usr/bin/env python3
"""
tinymodel M0 Baseline Benchmark Runner (llama.cpp).

Implements the M0 benchmark protocol in docs/TESTING.md:
1. Verifies the frozen prompt corpus against bench/data/manifest.json (SHA-256).
2. Measures sustained STREAM memory bandwidth on the host if bench/stream_bw.c is compiled.
3. Collects hardware & thermal telemetry (OS, CPU topology, ISA flags, thermal state on macOS / temp & freq on Linux).
4. Runs llama-bench (pp512, tg128, 5 repetitions) across configured backends:
   - cpu_kleidiai (pure CPU build, repack=1, 4 P-core threads)
   - cpu_norepack (pure CPU build, repack=0, zero-copy mmap)
   - metal        (Metal GPU build, ngl=99, flash_attn=1) [macOS only]
5. Runs deterministic greedy end-to-end completion (llama-completion, temp=0, seed=42, n=128)
   on the frozen prompt corpus, recording load_ms, pp_tps, tg_tps, peak RSS, peak footprint,
   and SHA-256 of generated text (for future L3 greedy token-for-token parity checks!).
6. Emits schema-versioned JSON to bench/baselines/<device>-llama-cpp-<date>.json
   and generates a human-readable Markdown summary table.
"""

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import re
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "bench" / "data" / "manifest.json"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_manifest(manifest_path: Path) -> Dict[str, Any]:
    if not manifest_path.exists():
        raise RuntimeError(f"Missing prompt manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_dir = manifest_path.parent
    for fname, meta in manifest["files"].items():
        fpath = data_dir / fname
        if not fpath.exists():
            raise RuntimeError(f"Missing frozen fixture: {fpath}")
        raw = fpath.read_bytes()
        actual_hash = sha256_bytes(raw)
        if len(raw) != meta["bytes"] or actual_hash != meta["sha256"]:
            raise RuntimeError(
                f"Fixture mismatch for {fname}: expected {meta['bytes']}B / {meta['sha256']}, "
                f"got {len(raw)}B / {actual_hash}"
            )
    return manifest


def run_cmd(cmd: List[str], env: Optional[Dict[str, str]] = None, timeout: int = 600) -> subprocess.CompletedProcess:
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=merged_env, timeout=timeout)


def sysctl_str(key: str) -> Optional[str]:
    r = subprocess.run(["sysctl", "-n", key], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def collect_hardware_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "os": platform.system(),
        "os_release": platform.release(),
        "arch": platform.machine(),
    }
    if platform.system() == "Darwin":
        info["os_version"] = subprocess.check_output(["sw_vers", "-productVersion"], text=True).strip()
        info["cpu_brand"] = sysctl_str("machdep.cpu.brand_string")
        mem = sysctl_str("hw.memsize")
        info["ram_bytes"] = int(mem) if mem else None
        info["ncpu"] = int(sysctl_str("hw.ncpu") or 0)
        info["p_cores"] = int(sysctl_str("hw.perflevel0.physicalcpu") or 0)
        info["e_cores"] = int(sysctl_str("hw.perflevel1.physicalcpu") or 0)
        feats = {}
        for f in ["FEAT_DotProd", "FEAT_I8MM", "FEAT_FP16", "FEAT_BF16", "FEAT_SME", "FEAT_SME2"]:
            v = sysctl_str(f"hw.optional.arm.{f}")
            if v is not None:
                feats[f] = bool(int(v))
        info["arm_features"] = feats
    elif platform.system() == "Linux":
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    info["ram_bytes"] = int(line.split()[1]) * 1024
                    break
        except OSError:
            pass
        try:
            model = Path("/proc/device-tree/model").read_text().rstrip("\x00\n")
            info["board_model"] = model
        except OSError:
            pass
    return info


def read_thermal_and_freq() -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if platform.system() == "Darwin":
        # Non-privileged thermal pressure state via Foundation NSProcessInfo.thermalState
        # 0 = Nominal, 1 = Fair, 2 = Serious, 3 = Critical
        try:
            import ctypes
            objc = ctypes.cdll.LoadLibrary("/usr/lib/libobjc.A.dylib")
            ctypes.cdll.LoadLibrary("/System/Library/Frameworks/Foundation.framework/Foundation")
            objc.objc_getClass.restype = ctypes.c_void_p
            objc.objc_getClass.argtypes = [ctypes.c_char_p]
            objc.sel_registerName.restype = ctypes.c_void_p
            objc.sel_registerName.argtypes = [ctypes.c_char_p]
            objc.objc_msgSend.restype = ctypes.c_void_p
            objc.objc_msgSend.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
            pi = objc.objc_msgSend(objc.objc_getClass(b"NSProcessInfo"), objc.sel_registerName(b"processInfo"))
            objc.objc_msgSend.restype = ctypes.c_long
            val = int(objc.objc_msgSend(pi, objc.sel_registerName(b"thermalState")))
            names = {0: "nominal", 1: "fair", 2: "serious", 3: "critical"}
            out["macos_thermal_state"] = names.get(val, str(val))
        except Exception:
            pass
    elif platform.system() == "Linux":
        tpath = Path("/sys/class/thermal/thermal_zone0/temp")
        if tpath.exists():
            out["cpu_temp_c"] = round(int(tpath.read_text().strip()) / 1000.0, 2)
        fpath = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq")
        if fpath.exists():
            out["cpu0_freq_mhz"] = round(int(fpath.read_text().strip()) / 1000.0, 1)
    return out


def measure_stream_bandwidth(threads_list: List[int]) -> List[Dict[str, Any]]:
    src = REPO_ROOT / "bench" / "stream_bw.c"
    bin_dir = REPO_ROOT / "bench" / "tmp"
    bin_dir.mkdir(parents=True, exist_ok=True)
    exe = bin_dir / "stream_bw"
    if not exe.exists() and src.exists():
        cc = "clang" if platform.system() == "Darwin" else "cc"
        subprocess.run([cc, "-O3", "-pthread", str(src), "-o", str(exe)], check=True)
    results = []
    if exe.exists():
        for t in threads_list:
            r = run_cmd([str(exe), str(t)])
            if r.returncode == 0:
                results.append(json.loads(r.stdout.strip()))
    return results


def run_llama_bench(
    bench_bin: Path,
    model_path: Path,
    extra_args: List[str],
    reps: int = 5,
    pp: int = 512,
    tg: int = 128,
) -> Dict[str, Any]:
    cmd = [
        str(bench_bin),
        "-m",
        str(model_path),
        "-p",
        str(pp),
        "-n",
        str(tg),
        "-r",
        str(reps),
        "-o",
        "json",
    ] + extra_args
    r = run_cmd(cmd, timeout=900)
    if r.returncode != 0:
        raise RuntimeError(f"llama-bench failed ({cmd}):\n{r.stderr[-1000:]}")
    entries = json.loads(r.stdout)
    out: Dict[str, Any] = {}
    for e in entries:
        samples = e.get("samples_ts") or []
        med = statistics.median(samples) if samples else e.get("avg_ts")
        rec = {
            "median_tps": round(med, 2) if med is not None else None,
            "avg_tps": round(e["avg_ts"], 2),
            "stddev_tps": round(e["stddev_ts"], 2),
            "samples_tps": [round(x, 2) for x in samples],
        }
        if e["n_prompt"] > 0 and e["n_gen"] == 0:
            out[f"pp{pp}"] = rec
            out["model_size_bytes"] = e["model_size"]
            out["model_n_params"] = e["model_n_params"]
            out["build_commit"] = e["build_commit"]
        elif e["n_gen"] > 0 and e["n_prompt"] == 0:
            out[f"tg{tg}"] = rec
    return out


def parse_completion_and_rss(
    comp_bin: Path,
    model_path: Path,
    prompt_file: Path,
    extra_args: List[str],
    n_predict: int = 128,
) -> Dict[str, Any]:
    out_txt = REPO_ROOT / "bench" / "tmp" / "last_output.txt"
    if out_txt.exists():
        out_txt.unlink()
    time_cmd = ["/usr/bin/time", "-l"] if platform.system() == "Darwin" else ["/usr/bin/time", "-v"]
    cmd = time_cmd + [
        str(comp_bin),
        "-m",
        str(model_path),
        "-f",
        str(prompt_file),
        "-n",
        str(n_predict),
        "--temp",
        "0",
        "-s",
        "42",
        "-c",
        "2048",
        "--no-display-prompt",
        "--no-conversation",
        "--no-warmup",
        "--perf",
    ] + extra_args
    r = run_cmd(cmd, timeout=600)
    if r.returncode != 0:
        raise RuntimeError(f"llama-completion failed ({cmd}):\n{r.stderr[-1000:]}")

    combined = r.stderr + "\n" + r.stdout
    res: Dict[str, Any] = {
        "generated_sha256": sha256_bytes(r.stdout.encode("utf-8")),
        "generated_preview": r.stdout[:160].strip(),
    }

    m_load = re.search(r"load time\s*=\s*([\d.]+)\s*ms", combined)
    if m_load:
        res["load_ms"] = round(float(m_load.group(1)), 2)

    m_pp = re.search(
        r"prompt eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*tokens\s*\(\s*[\d.]+\s*ms per token,\s*([\d.]+)\s*tokens per second\)",
        combined,
    )
    if m_pp:
        res["prompt_eval_ms"] = round(float(m_pp.group(1)), 2)
        res["prompt_tokens"] = int(m_pp.group(2))
        res["prompt_tps"] = round(float(m_pp.group(3)), 2)

    m_tg = re.search(
        r"(?<!prompt )eval time\s*=\s*([\d.]+)\s*ms\s*/\s*(\d+)\s*runs\s*\(\s*[\d.]+\s*ms per token,\s*([\d.]+)\s*tokens per second\)",
        combined,
    )
    if m_tg:
        res["eval_ms"] = round(float(m_tg.group(1)), 2)
        res["eval_runs"] = int(m_tg.group(2))
        res["eval_tps"] = round(float(m_tg.group(3)), 2)

    if platform.system() == "Darwin":
        m_rss = re.search(r"(\d+)\s+maximum resident set size", combined)
        m_fp = re.search(r"(\d+)\s+peak memory footprint", combined)
        if m_rss:
            res["max_rss_mib"] = round(int(m_rss.group(1)) / (1024 * 1024), 2)
        if m_fp:
            res["peak_footprint_mib"] = round(int(m_fp.group(1)) / (1024 * 1024), 2)
    else:
        m_rss = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", combined)
        if m_rss:
            res["max_rss_mib"] = round(int(m_rss.group(1)) / 1024.0, 2)

    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", type=Path, default=REPO_ROOT / "models")
    ap.add_argument("--cpu-bin-dir", type=Path, default=REPO_ROOT / "third_party" / "llama.cpp" / "build-cpu" / "bin")
    ap.add_argument("--metal-bin-dir", type=Path, default=REPO_ROOT / "third_party" / "llama.cpp" / "build" / "bin")
    ap.add_argument("--cpu-threads", type=int, default=4, help="CPU threads for edge/host CPU runs (4 = M4 P-cores / Pi 5 cores)")
    ap.add_argument("--reps", type=int, default=5, help="llama-bench repetitions (protocol requires >= 5)")
    ap.add_argument("--tag", type=str, default="apple-m4-16gb")
    args = ap.parse_args()

    manifest = verify_manifest(MANIFEST_PATH)
    print(f"[1/4] Verified frozen prompt manifest ({manifest['protocol_id']})", flush=True)

    hw = collect_hardware_info()
    thermal_before = read_thermal_and_freq()
    stream_bw = measure_stream_bandwidth([1, 4, 8, 10] if hw.get("ncpu", 4) >= 8 else [1, 2, 4])
    peak_copy_gbs = max((x["Copy_gbs"] for x in stream_bw), default=None)
    peak_read_gbs = max((x["Read_gbs"] for x in stream_bw), default=None)
    print(f"[2/4] Hardware: {hw.get('cpu_brand') or hw.get('board_model')}, STREAM peak Copy={peak_copy_gbs} GB/s, Read={peak_read_gbs} GB/s", flush=True)

    ggufs = sorted(args.models_dir.glob("*.gguf"))
    if not ggufs:
        raise RuntimeError(f"No .gguf models found in {args.models_dir}")

    prompt_short = REPO_ROOT / "bench" / "data" / "prompt_short.txt"
    model_results: List[Dict[str, Any]] = []

    for idx, mpath in enumerate(ggufs, 1):
        print(f"[3/4] ({idx}/{len(ggufs)}) Benchmarking {mpath.name} ...", flush=True)
        file_bytes = mpath.stat().st_size
        file_sha = sha256_file(mpath)
        m_entry: Dict[str, Any] = {
            "filename": mpath.name,
            "file_bytes": file_bytes,
            "file_mib": round(file_bytes / (1024 * 1024), 2),
            "sha256": file_sha,
            "modes": {},
        }

        # Mode A: Pure CPU zero-copy mmap (repack=0) — directly mirrors Pi Zero 2W / non-repacked path
        cpu_bench = args.cpu_bin_dir / "llama-bench"
        cpu_comp = args.cpu_bin_dir / "llama-completion"
        if cpu_bench.exists():
            print(f"      -> cpu_norepack (t={args.cpu_threads})", flush=True)
            b_nr = run_llama_bench(
                cpu_bench, mpath, ["-t", str(args.cpu_threads), "--repack", "0"], reps=args.reps
            )
            c_nr = parse_completion_and_rss(
                cpu_comp, mpath, prompt_short, ["-t", str(args.cpu_threads), "--no-repack"], n_predict=128
            )
            if peak_read_gbs and b_nr.get("tg128", {}).get("median_tps"):
                eff_bw = (b_nr["model_size_bytes"] * b_nr["tg128"]["median_tps"]) / 1e9
                b_nr["effective_decode_bw_gbs"] = round(eff_bw, 2)
                b_nr["stream_read_bw_util_pct"] = round(100.0 * eff_bw / peak_read_gbs, 1)
            m_entry["model_size_bytes"] = b_nr.get("model_size_bytes")
            m_entry["model_n_params"] = b_nr.get("model_n_params")
            m_entry["llama_cpp_commit"] = b_nr.get("build_commit")
            m_entry["modes"]["cpu_norepack"] = {
                "threads": args.cpu_threads,
                "bench": b_nr,
                "e2e_greedy": c_nr,
                "thermal": read_thermal_and_freq(),
            }

            # Mode B: Pure CPU with KleidiAI repack=1 (SME2/I8MM/dotprod)
            print(f"      -> cpu_kleidiai (t={args.cpu_threads}, repack=1)", flush=True)
            b_kl = run_llama_bench(
                cpu_bench, mpath, ["-t", str(args.cpu_threads), "--repack", "1"], reps=args.reps
            )
            c_kl = parse_completion_and_rss(
                cpu_comp, mpath, prompt_short, ["-t", str(args.cpu_threads)], n_predict=128
            )
            if peak_read_gbs and b_kl.get("tg128", {}).get("median_tps"):
                eff_bw = (b_kl["model_size_bytes"] * b_kl["tg128"]["median_tps"]) / 1e9
                b_kl["effective_decode_bw_gbs"] = round(eff_bw, 2)
                b_kl["stream_read_bw_util_pct"] = round(100.0 * eff_bw / peak_read_gbs, 1)
            m_entry["modes"]["cpu_kleidiai"] = {
                "threads": args.cpu_threads,
                "bench": b_kl,
                "e2e_greedy": c_kl,
                "thermal": read_thermal_and_freq(),
            }

        # Mode C: Metal GPU (ngl=99, flash_attn=1)
        mtl_bench = args.metal_bin_dir / "llama-bench"
        mtl_comp = args.metal_bin_dir / "llama-completion"
        if platform.system() == "Darwin" and mtl_bench.exists():
            print("      -> metal (ngl=99, fa=1)", flush=True)
            b_mt = run_llama_bench(
                mtl_bench, mpath, ["-ngl", "99", "-fa", "1", "-t", str(args.cpu_threads)], reps=args.reps
            )
            c_mt = parse_completion_and_rss(
                mtl_comp, mpath, prompt_short, ["-ngl", "99", "-fa", "on", "-t", str(args.cpu_threads)], n_predict=128
            )
            if peak_read_gbs and b_mt.get("tg128", {}).get("median_tps"):
                eff_bw = (b_mt["model_size_bytes"] * b_mt["tg128"]["median_tps"]) / 1e9
                b_mt["effective_decode_bw_gbs"] = round(eff_bw, 2)
                b_mt["stream_read_bw_util_pct"] = round(100.0 * eff_bw / peak_read_gbs, 1)
            m_entry["modes"]["metal"] = {
                "ngl": 99,
                "flash_attn": True,
                "bench": b_mt,
                "e2e_greedy": c_mt,
                "thermal": read_thermal_and_freq(),
            }

        model_results.append(m_entry)

    today = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    payload = {
        "schema": "tinymodel-m0-baseline",
        "schema_version": 1,
        "protocol_id": manifest["protocol_id"],
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "device_tag": args.tag,
        "engine": {
            "name": "llama.cpp",
            "commit": model_results[0].get("llama_cpp_commit") if model_results else None,
            "cpu_cmake_flags": "-DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_CPU_KLEIDIAI=ON -DGGML_METAL=OFF -DGGML_BLAS=OFF",
            "metal_cmake_flags": "-DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_CPU_KLEIDIAI=ON -DGGML_METAL=ON -DGGML_BLAS=ON",
        },
        "hardware": hw,
        "thermal_initial": thermal_before,
        "thermal_final": read_thermal_and_freq(),
        "stream_bandwidth": stream_bw,
        "prompt_manifest": manifest,
        "models": model_results,
    }

    out_dir = REPO_ROOT / "bench" / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{args.tag}-llama-cpp-{today}.json"
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[4/4] Saved baseline JSON -> {json_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
