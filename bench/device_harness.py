#!/usr/bin/env python3
"""
Unified External Device Benchmark & Verification Harness for `tinymodel`.

Supports three pluggable transports under one protocol (`docs/TESTING.md` §2–§4):
  1. `local`                  — Local host (Apple M4 CPU / Metal, Linux x86_64/arm64)
  2. `adb` / `adb:<serial>`   — Android devices over USB or Wireless ADB (e.g. Google Pixel 8)
  3. `ssh://[user@]host[:p]`  — Linux edge boards over SSH (Raspberry Pi 5 / Pi 4B / Zero 2W)

Examples:
  # List connected ADB devices & cross-compiled binaries
  python3 bench/device_harness.py list

  # Probe hardware, heterogeneous CPU clusters, L1 kernel self-check & STREAM BW
  python3 bench/device_harness.py probe --target local
  python3 bench/device_harness.py probe --target adb --build
  python3 bench/device_harness.py probe --target ssh://pi@raspberrypi.local --profile micro --build

  # Full E2E benchmark (manifest check + probe + multi-round pp/tg/TTFT + L3 parity + JSON report)
  python3 bench/device_harness.py run --target adb -m models/Qwen3.5-0.8B-Q4_0.gguf \
      --cluster-policy no-little --reps 5 --compare-local --build
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import re
import resource
import shlex
import shutil
import statistics
import subprocess
import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent
MANIFEST_PATH = REPO_ROOT / "bench" / "data" / "manifest.json"
RESULTS_ROOT = REPO_ROOT / "bench" / "results"

LITTLE_CORE_NAMES = {"Cortex-A53", "Cortex-A55", "Cortex-A510", "Cortex-A520"}
PRIME_CORE_PREFIXES = ("Cortex-X",)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_prompt_manifest(manifest_path: Path = MANIFEST_PATH) -> Dict[str, Any]:
    """Verifies frozen prompt fixtures against SHA-256 manifest (`docs/TESTING.md` §3.1)."""
    if not manifest_path.exists():
        raise RuntimeError(f"Missing frozen prompt manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    data_dir = manifest_path.parent
    for fname, meta in manifest.get("files", {}).items():
        fpath = data_dir / fname
        if not fpath.exists():
            raise RuntimeError(f"Missing frozen prompt fixture: {fpath}")
        raw = fpath.read_bytes()
        digest = sha256_bytes(raw)
        if len(raw) != meta["bytes"] or digest != meta["sha256"]:
            raise RuntimeError(
                f"Frozen prompt fixture mismatch for '{fname}': "
                f"expected {meta['bytes']}B / {meta['sha256']}, got {len(raw)}B / {digest}"
            )
    return manifest


def git_short_commit() -> str:
    r = subprocess.run(
        ["git", "rev-parse", "--short=7", "HEAD"],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else "unknown"


def slugify_device(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return s or "unknown-device"


def find_adb_binary() -> Optional[str]:
    on_path = shutil.which("adb")
    if on_path:
        return on_path
    candidates = [
        Path.home() / "Library" / "Android" / "sdk" / "platform-tools" / "adb",
        Path.home() / "Android" / "Sdk" / "platform-tools" / "adb",
    ]
    for env_key in ("ANDROID_HOME", "ANDROID_SDK_ROOT"):
        val = os.environ.get(env_key)
        if val:
            candidates.insert(0, Path(val) / "platform-tools" / "adb")
    for c in candidates:
        if c.is_file() and os.access(c, os.X_OK):
            return str(c)
    return None


def list_adb_devices(adb_bin: Optional[str] = None) -> List[Dict[str, str]]:
    adb = adb_bin or find_adb_binary()
    if not adb:
        return []
    r = subprocess.run([adb, "devices", "-l"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if r.returncode != 0:
        return []
    devices: List[Dict[str, str]] = []
    for line in r.stdout.splitlines()[1:]:
        line = line.strip()
        if not line or line.startswith("*"):
            continue
        parts = line.split()
        if len(parts) >= 2:
            entry: Dict[str, str] = {"serial": parts[0], "state": parts[1]}
            for kv in parts[2:]:
                if ":" in kv:
                    k, v = kv.split(":", 1)
                    entry[k] = v
            devices.append(entry)
    return devices


def select_cluster_mask_and_threads(
    clusters: List[Dict[str, Any]],
    policy: str,
    explicit_mask: Optional[str] = None,
    explicit_threads: Optional[int] = None,
) -> Tuple[Optional[str], int]:
    """
    Computes `(taskset_hex_mask, recommended_threads)` from probed CPU clusters.
    Policies:
      - `all`: use all logical cores (no taskset mask unless explicit_mask given)
      - `no-little`: exclude LITTLE cores (e.g. Pixel 8 -> 4x A715 + 1x X3 = 5 threads, mask 0x1f0)
      - `mid-only`: power-efficient decode on mid cores (e.g. Pixel 8 -> 4x A715 = 4 threads, mask 0xf0)
    """
    if explicit_mask:
        if explicit_threads is not None:
            return explicit_mask, explicit_threads
        mask_int = int(explicit_mask, 16) if explicit_mask.startswith("0x") else int(explicit_mask, 16)
        return explicit_mask, max(1, bin(mask_int).count("1"))

    if not clusters or len(clusters) <= 1 or policy == "all":
        total_cpus = sum(int(c.get("count", 0)) for c in clusters) or 4
        return None, explicit_threads or min(total_cpus, 8)

    selected_cpus: List[int] = []
    if policy == "no-little":
        non_little = [c for c in clusters if c.get("uarch") not in LITTLE_CORE_NAMES]
        target = non_little if non_little else clusters
        for c in target:
            selected_cpus.extend(c.get("cpus", []))
    elif policy == "mid-only":
        mid = [
            c
            for c in clusters
            if c.get("uarch") not in LITTLE_CORE_NAMES
            and not str(c.get("uarch", "")).startswith(PRIME_CORE_PREFIXES)
        ]
        target = mid if mid else [c for c in clusters if c.get("uarch") not in LITTLE_CORE_NAMES] or clusters
        for c in target:
            selected_cpus.extend(c.get("cpus", []))
    else:
        raise ValueError(f"Unknown cluster policy: {policy}")

    mask_val = 0
    for cpu in selected_cpus:
        mask_val |= 1 << int(cpu)
    mask_hex = f"0x{mask_val:x}" if mask_val > 0 else None
    n_threads = explicit_threads or max(1, len(selected_cpus))
    return mask_hex, n_threads


def parse_last_json_line(stdout: str) -> Dict[str, Any]:
    for line in reversed(stdout.splitlines()):
        s = line.strip()
        if s.startswith("{") and s.endswith("}"):
            return json.loads(s)
    raise RuntimeError(f"No JSON object found in command output:\n{stdout[:800]}")


class DeviceTransport(ABC):
    """Abstract execution & file-staging interface for local, ADB, and SSH targets."""

    kind: str

    @abstractmethod
    def describe(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def prepare_binary(self, profile: str, auto_build: bool) -> str:
        """Ensures the `tinymodel` binary is built and staged on the target; returns target path."""
        raise NotImplementedError

    @abstractmethod
    def stage_file(self, local_path: Path, subdir: str = "") -> str:
        """Ensures `local_path` is available on the target (skipping upload if size matches); returns target path."""
        raise NotImplementedError

    @abstractmethod
    def run_tinymodel(
        self,
        remote_bin: str,
        args: List[str],
        env: Optional[Dict[str, str]] = None,
        cpu_mask: Optional[str] = None,
        timeout: int = 900,
    ) -> Tuple[subprocess.CompletedProcess, Optional[int]]:
        """Runs `tinymodel` on the target and returns `(completed_proc, host_measured_peak_rss_bytes)`."""
        raise NotImplementedError

    def extra_device_telemetry(self) -> Dict[str, Any]:
        return {}


class LocalTransport(DeviceTransport):
    kind = "local"

    def describe(self) -> str:
        return f"local ({platform.system()}/{platform.machine()})"

    def prepare_binary(self, profile: str, auto_build: bool) -> str:
        exe = REPO_ROOT / "target" / "release" / "tinymodel"
        if auto_build or not exe.exists():
            cmd = ["cargo", "build", "--release", "-p", "tm-cli"]
            if profile == "micro":
                cmd.extend(["--no-default-features", "--features", "micro"])
            subprocess.run(cmd, cwd=REPO_ROOT, check=True)
        if not exe.exists():
            raise RuntimeError(f"Local release binary not found at {exe}; pass --build")
        return str(exe)

    def stage_file(self, local_path: Path, subdir: str = "") -> str:
        if not local_path.exists():
            raise RuntimeError(f"Local file does not exist: {local_path}")
        return str(local_path.resolve())

    def run_tinymodel(
        self,
        remote_bin: str,
        args: List[str],
        env: Optional[Dict[str, str]] = None,
        cpu_mask: Optional[str] = None,
        timeout: int = 900,
    ) -> Tuple[subprocess.CompletedProcess, Optional[int]]:
        merged_env = os.environ.copy()
        if env:
            merged_env.update(env)
        cmd = [remote_bin] + args
        if cpu_mask and platform.system() == "Linux" and shutil.which("taskset"):
            cmd = ["taskset", cpu_mask] + cmd

        # On macOS, `/usr/bin/time -l` captures exact child `maximum resident set size`.
        if platform.system() == "Darwin" and Path("/usr/bin/time").exists():
            wrapped = ["/usr/bin/time", "-l"] + cmd
            proc = subprocess.run(
                wrapped,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=merged_env,
                timeout=timeout,
            )
            m = re.search(r"(\d+)\s+maximum resident set size", proc.stderr)
            peak_rss = int(m.group(1)) if m else None
            return proc, peak_rss

        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=merged_env,
            timeout=timeout,
        )
        return proc, None


def build_musl_binary(profile: str) -> Path:
    exe = REPO_ROOT / "target" / "aarch64-unknown-linux-musl" / "release" / "tinymodel"
    cmd = [
        "cargo",
        "zigbuild",
        "--release",
        "-p",
        "tm-cli",
        "--target",
        "aarch64-unknown-linux-musl",
    ]
    if profile == "micro":
        cmd.extend(["--no-default-features", "--features", "micro"])
    else:
        cmd.extend(["--no-default-features", "--features", "edge"])
    env = os.environ.copy()
    if profile == "micro" and "RUSTFLAGS" not in env:
        env["RUSTFLAGS"] = "-C target-cpu=cortex-a53"
    subprocess.run(cmd, cwd=REPO_ROOT, env=env, check=True)
    if not exe.exists():
        raise RuntimeError(f"Cross-compiled binary missing after build: {exe}")
    return exe


class AdbTransport(DeviceTransport):
    """Android external device transport via `adb` (e.g., Google Pixel 8)."""

    kind = "adb"

    def __init__(self, serial: Optional[str] = None, remote_dir: str = "/data/local/tmp/tinymodel"):
        adb = find_adb_binary()
        if not adb:
            raise RuntimeError(
                "adb not found on PATH or ~/Library/Android/sdk/platform-tools/adb; install Android platform-tools"
            )
        self.adb = adb
        self.remote_dir = remote_dir.rstrip("/")
        devices = list_adb_devices(adb)
        ready = [d for d in devices if d["state"] == "device"]
        if serial:
            matched = [d for d in devices if d["serial"] == serial]
            if not matched:
                raise RuntimeError(f"ADB device '{serial}' not found (available: {devices})")
            if matched[0]["state"] != "device":
                raise RuntimeError(
                    f"ADB device '{serial}' is in state '{matched[0]['state']}' (accept USB debugging prompt on phone)"
                )
            self.serial = serial
        else:
            if not ready:
                raise RuntimeError(
                    f"No authorized Android ADB devices connected (detected: {devices}). "
                    "Connect Google Pixel 8 via USB or `adb connect <ip:port>` and enable USB debugging."
                )
            self.serial = ready[0]["serial"]

        self._adb(["shell", f"mkdir -p {shlex.quote(self.remote_dir)}"], check=True)

    def _adb(self, args: List[str], check: bool = False, timeout: int = 900) -> subprocess.CompletedProcess:
        cmd = [self.adb, "-s", self.serial] + args
        return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=check, timeout=timeout)

    def describe(self) -> str:
        return f"adb:{self.serial} ({self.remote_dir})"

    def _remote_size(self, rpath: str) -> Optional[int]:
        r = self._adb(["shell", f"stat -c %s {shlex.quote(rpath)} 2>/dev/null"])
        s = r.stdout.strip()
        return int(s) if r.returncode == 0 and s.isdigit() else None

    def prepare_binary(self, profile: str, auto_build: bool) -> str:
        local_exe = REPO_ROOT / "target" / "aarch64-unknown-linux-musl" / "release" / "tinymodel"
        if auto_build or not local_exe.exists():
            local_exe = build_musl_binary(profile)
        rbin = f"{self.remote_dir}/tinymodel"
        # Always push the compact (~750 KB) binary so code changes take effect immediately
        self._adb(["push", str(local_exe), rbin], check=True)
        self._adb(["shell", f"chmod 755 {shlex.quote(rbin)}"], check=True)
        return rbin

    def stage_file(self, local_path: Path, subdir: str = "") -> str:
        if not local_path.exists():
            raise RuntimeError(f"Local file missing: {local_path}")
        target_dir = f"{self.remote_dir}/{subdir}".rstrip("/")
        if subdir:
            self._adb(["shell", f"mkdir -p {shlex.quote(target_dir)}"], check=True)
        rpath = f"{target_dir}/{local_path.name}"
        expected_size = local_path.stat().st_size
        if self._remote_size(rpath) != expected_size:
            print(f"[adb:{self.serial}] pushing {local_path.name} ({expected_size / 1048576.0:.1f} MiB) -> {rpath}", file=sys.stderr)
            self._adb(["push", str(local_path), rpath], check=True, timeout=1800)
        return rpath

    def run_tinymodel(
        self,
        remote_bin: str,
        args: List[str],
        env: Optional[Dict[str, str]] = None,
        cpu_mask: Optional[str] = None,
        timeout: int = 900,
    ) -> Tuple[subprocess.CompletedProcess, Optional[int]]:
        env_prefix = ""
        if env:
            env_prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items()) + " "
        taskset_prefix = f"taskset {shlex.quote(cpu_mask)} " if cpu_mask else ""
        quoted_cmd = " ".join(shlex.quote(x) for x in [remote_bin] + args)
        # Preserve remote exit status explicitly across older Android `adb shell` versions
        shell_cmd = f"{env_prefix}{taskset_prefix}{quoted_cmd}"
        proc = self._adb(["shell", shell_cmd], check=False, timeout=timeout)
        return proc, None

    def extra_device_telemetry(self) -> Dict[str, Any]:
        extra: Dict[str, Any] = {}
        for prop, key in [
            ("ro.product.model", "android_model"),
            ("ro.soc.model", "android_soc"),
            ("ro.build.version.release", "android_release"),
        ]:
            r = self._adb(["shell", f"getprop {prop}"])
            if r.returncode == 0 and r.stdout.strip():
                extra[key] = r.stdout.strip()
        return extra


class SshTransport(DeviceTransport):
    """Linux SBC / Raspberry Pi (Zero 2W, 4B, 5) external device transport via SSH + SCP."""

    kind = "ssh"

    def __init__(self, ssh_uri: str, remote_dir: str = "/tmp/tinymodel-bench"):
        parsed = urlparse(ssh_uri if "://" in ssh_uri else f"ssh://{ssh_uri}")
        if not parsed.hostname:
            raise ValueError(f"Invalid SSH target '{ssh_uri}' (expected ssh://[user@]host[:port])")
        self.user_host = f"{parsed.username}@{parsed.hostname}" if parsed.username else parsed.hostname
        self.port = parsed.port
        self.remote_dir = remote_dir.rstrip("/")
        self._ssh(f"mkdir -p {shlex.quote(self.remote_dir)}", check=True)

    def _ssh_base(self) -> List[str]:
        base = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
        if self.port:
            base.extend(["-p", str(self.port)])
        base.append(self.user_host)
        return base

    def _ssh(self, remote_cmd: str, check: bool = False, timeout: int = 900) -> subprocess.CompletedProcess:
        return subprocess.run(
            self._ssh_base() + [remote_cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=check,
            timeout=timeout,
        )

    def _scp(self, local_path: Path, remote_path: str, timeout: int = 1800) -> None:
        cmd = ["scp", "-q", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
        if self.port:
            cmd.extend(["-P", str(self.port)])
        cmd.extend([str(local_path), f"{self.user_host}:{remote_path}"])
        subprocess.run(cmd, check=True, timeout=timeout)

    def describe(self) -> str:
        port_s = f":{self.port}" if self.port else ""
        return f"ssh://{self.user_host}{port_s} ({self.remote_dir})"

    def _remote_size(self, rpath: str) -> Optional[int]:
        r = self._ssh(f"stat -c %s {shlex.quote(rpath)} 2>/dev/null")
        s = r.stdout.strip()
        return int(s) if r.returncode == 0 and s.isdigit() else None

    def prepare_binary(self, profile: str, auto_build: bool) -> str:
        local_exe = REPO_ROOT / "target" / "aarch64-unknown-linux-musl" / "release" / "tinymodel"
        if auto_build or not local_exe.exists():
            local_exe = build_musl_binary(profile)
        rbin = f"{self.remote_dir}/tinymodel"
        self._scp(local_exe, rbin)
        self._ssh(f"chmod 755 {shlex.quote(rbin)}", check=True)
        return rbin

    def stage_file(self, local_path: Path, subdir: str = "") -> str:
        if not local_path.exists():
            raise RuntimeError(f"Local file missing: {local_path}")
        target_dir = f"{self.remote_dir}/{subdir}".rstrip("/")
        if subdir:
            self._ssh(f"mkdir -p {shlex.quote(target_dir)}", check=True)
        rpath = f"{target_dir}/{local_path.name}"
        expected_size = local_path.stat().st_size
        if self._remote_size(rpath) != expected_size:
            print(f"[ssh:{self.user_host}] uploading {local_path.name} ({expected_size / 1048576.0:.1f} MiB) -> {rpath}", file=sys.stderr)
            self._scp(local_path, rpath)
        return rpath

    def run_tinymodel(
        self,
        remote_bin: str,
        args: List[str],
        env: Optional[Dict[str, str]] = None,
        cpu_mask: Optional[str] = None,
        timeout: int = 900,
    ) -> Tuple[subprocess.CompletedProcess, Optional[int]]:
        env_prefix = ""
        if env:
            env_prefix = " ".join(f"{k}={shlex.quote(v)}" for k, v in env.items()) + " "
        taskset_prefix = f"taskset {shlex.quote(cpu_mask)} " if cpu_mask else ""
        quoted_cmd = " ".join(shlex.quote(x) for x in [remote_bin] + args)
        proc = self._ssh(f"{env_prefix}{taskset_prefix}{quoted_cmd}", check=False, timeout=timeout)
        return proc, None

    def extra_device_telemetry(self) -> Dict[str, Any]:
        extra: Dict[str, Any] = {}
        r = self._ssh("vcgencmd get_throttled 2>/dev/null")
        if r.returncode == 0 and "throttled=" in r.stdout:
            extra["rpi_throttled"] = r.stdout.strip().split("=", 1)[-1]
        return extra


def make_transport(target: str, remote_dir: Optional[str] = None) -> DeviceTransport:
    if target == "local":
        return LocalTransport()
    if target == "adb" or target.startswith("adb:"):
        serial = target.split(":", 1)[1] if ":" in target else None
        return AdbTransport(serial=serial, remote_dir=remote_dir or "/data/local/tmp/tinymodel")
    if target.startswith("ssh://") or "@" in target:
        return SshTransport(target, remote_dir=remote_dir or "/tmp/tinymodel-bench")
    raise ValueError(
        f"Unsupported --target '{target}'. Expected 'local', 'adb[:<serial>]', or 'ssh://[user@]host[:port]'."
    )


def summarize_rounds(
    rounds: List[Dict[str, Any]],
    stream_read_gbs: Optional[float],
) -> Dict[str, Any]:
    pp_vals = [float(r["pp_tps"]) for r in rounds]
    tg_vals = [float(r["tg_tps"]) for r in rounds]
    ttft_vals = [float(r["ttft_ms"]) for r in rounds]
    load_vals = [float(r["load_ms"]) for r in rounds]
    bw_vals = [float(r["decode_eff_bw_gbs"]) for r in rounds]
    rss_vals = [int(r["peak_rss_bytes"]) for r in rounds if r.get("peak_rss_bytes") is not None]

    med_bw = statistics.median(bw_vals)
    bw_util_pct = (
        round(100.0 * med_bw / stream_read_gbs, 2)
        if stream_read_gbs and stream_read_gbs > 0
        else None
    )

    # Estimate average battery/PMIC power & Joules/token when power sensor is present
    pwr_samples: List[float] = []
    for r in rounds:
        for key in ("telemetry_before", "telemetry_after"):
            p = (r.get(key) or {}).get("battery_power_mw")
            if p is not None:
                pwr_samples.append(float(p))
    avg_pwr_mw = round(statistics.mean(pwr_samples), 1) if pwr_samples else None
    med_tg = statistics.median(tg_vals)
    joules_per_tok = (
        round((avg_pwr_mw / 1000.0) / med_tg, 4)
        if avg_pwr_mw is not None and med_tg > 0
        else None
    )

    temps_after = [
        float(r["telemetry_after"]["max_temp_c"])
        for r in rounds
        if (r.get("telemetry_after") or {}).get("max_temp_c") is not None
    ]

    hashes = sorted({r["greedy_tokens_fnv1a64"] for r in rounds})
    return {
        "repetitions": len(rounds),
        "load_ms_median": round(statistics.median(load_vals), 2),
        "ttft_ms_median": round(statistics.median(ttft_vals), 2),
        "pp_tokens": rounds[0]["pp_tokens"],
        "pp_tps_median": round(statistics.median(pp_vals), 2),
        "pp_tps_min": round(min(pp_vals), 2),
        "pp_tps_max": round(max(pp_vals), 2),
        "tg_tokens": rounds[0]["tg_tokens"],
        "tg_tps_median": round(med_tg, 2),
        "tg_tps_min": round(min(tg_vals), 2),
        "tg_tps_max": round(max(tg_vals), 2),
        "decode_eff_bw_gbs_median": round(med_bw, 2),
        "stream_bw_utilization_pct": bw_util_pct,
        "peak_rss_bytes_max": max(rss_vals) if rss_vals else None,
        "max_temp_c_peak": round(max(temps_after), 2) if temps_after else None,
        "avg_power_mw": avg_pwr_mw,
        "decode_joules_per_token": joules_per_tok,
        "deterministic_self_consistent": len(hashes) == 1,
        "greedy_tokens_fnv1a64": hashes[0] if len(hashes) == 1 else hashes,
    }


def cmd_list(_args: argparse.Namespace) -> int:
    adb = find_adb_binary()
    print("=== External Device Discovery ===")
    print(f"ADB binary: {adb or 'NOT FOUND'}")
    devices = list_adb_devices(adb)
    if devices:
        for d in devices:
            model = d.get("model", "?")
            print(f"  adb:{d['serial']}  state={d['state']}  model={model}")
    else:
        print("  (no Android ADB devices currently attached)")

    local_bin = REPO_ROOT / "target" / "release" / "tinymodel"
    musl_bin = REPO_ROOT / "target" / "aarch64-unknown-linux-musl" / "release" / "tinymodel"
    print("\n=== Staged Host Binaries ===")
    print(f"  host release:  {'READY' if local_bin.exists() else 'missing'} ({local_bin})")
    if musl_bin.exists():
        kb = musl_bin.stat().st_size / 1024.0
        print(f"  aarch64-musl:  READY ({kb:.1f} KB, {musl_bin})")
    else:
        print(f"  aarch64-musl:  missing ({musl_bin})")
    return 0


def cmd_probe(args: argparse.Namespace) -> int:
    transport = make_transport(args.target, args.remote_dir)
    rbin = transport.prepare_binary(args.profile, args.build)
    probe_args = ["probe", "--json"]
    if args.threads:
        probe_args.extend(["-t", str(args.threads)])
    if args.stream_mib:
        probe_args.extend(["--stream-mib", str(args.stream_mib)])
    if args.no_stream:
        probe_args.append("--no-stream")

    env = {"TM_FORCE_ISA": args.force_isa} if args.force_isa else None
    proc, _ = transport.run_tinymodel(rbin, probe_args, env=env, cpu_mask=args.cpu_mask)
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr or proc.stdout)
        raise RuntimeError(f"Remote probe failed with exit code {proc.returncode}")

    info = parse_last_json_line(proc.stdout)
    info["transport"] = transport.describe()
    info["extra_telemetry"] = transport.extra_device_telemetry()
    print(json.dumps(info, indent=2, ensure_ascii=False))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    manifest = verify_prompt_manifest()
    transport = make_transport(args.target, args.remote_dir)
    rbin = transport.prepare_binary(args.profile, args.build)
    env = {"TM_FORCE_ISA": args.force_isa} if args.force_isa else None

    # 1. Probe target device capabilities, clusters & STREAM bandwidth
    probe_args = ["probe", "--json"]
    if args.stream_mib:
        probe_args.extend(["--stream-mib", str(args.stream_mib)])
    probe_proc, _ = transport.run_tinymodel(rbin, probe_args, env=env, cpu_mask=args.cpu_mask)
    if probe_proc.returncode != 0:
        sys.stderr.write(probe_proc.stderr or probe_proc.stdout)
        raise RuntimeError(f"Target probe failed (code {probe_proc.returncode})")
    probe_info = parse_last_json_line(probe_proc.stdout)
    if not probe_info.get("kernel_self_check", {}).get("passed", False):
        raise RuntimeError(f"Target L1 kernel self-check failed: {probe_info.get('kernel_self_check')}")

    cpu_mask, eff_threads = select_cluster_mask_and_threads(
        probe_info.get("clusters", []),
        policy=args.cluster_policy,
        explicit_mask=args.cpu_mask,
        explicit_threads=args.threads,
    )

    # Re-measure STREAM Read BW under the exact selected `(cpu_mask, eff_threads)` if it differs
    if cpu_mask or eff_threads != (probe_info.get("stream_bw") or {}).get("threads"):
        sp_args = ["probe", "--json", "-t", str(eff_threads)]
        if args.stream_mib:
            sp_args.extend(["--stream-mib", str(args.stream_mib)])
        sp_proc, _ = transport.run_tinymodel(rbin, sp_args, env=env, cpu_mask=cpu_mask)
        if sp_proc.returncode == 0:
            probe_info = parse_last_json_line(sp_proc.stdout)

    stream_read_gbs = (probe_info.get("stream_bw") or {}).get("read_gbs")
    extra_tel = transport.extra_device_telemetry()

    # Stage frozen prompt fixture on target
    local_prompt = REPO_ROOT / "bench" / "data" / args.prompt_fixture
    remote_prompt = transport.stage_file(local_prompt, subdir="data")

    local_ref_transport: Optional[LocalTransport] = None
    local_ref_bin: Optional[str] = None
    if args.compare_local and transport.kind != "local":
        local_ref_transport = LocalTransport()
        local_ref_bin = local_ref_transport.prepare_binary("edge", args.build)

    model_reports: List[Dict[str, Any]] = []
    for model_str in args.models:
        model_path = Path(model_str)
        remote_model = transport.stage_file(model_path, subdir="models")

        def run_suite(corpus_flag: bool) -> List[Dict[str, Any]]:
            suite_rounds: List[Dict[str, Any]] = []
            for rep in range(args.reps):
                b_args = [
                    "bench",
                    "-m",
                    remote_model,
                    "-n",
                    str(args.n_predict),
                    "-t",
                    str(eff_threads),
                    "-c",
                    str(args.max_ctx),
                    "--json",
                ]
                if args.metal:
                    b_args.append("--metal")
                if corpus_flag:
                    b_args.extend(["-f", remote_prompt])

                proc, host_rss = transport.run_tinymodel(
                    rbin, b_args, env=env, cpu_mask=cpu_mask
                )
                if proc.returncode != 0:
                    sys.stderr.write(proc.stderr or proc.stdout)
                    raise RuntimeError(
                        f"Benchmark failed on round {rep + 1}/{args.reps} for {model_path.name}"
                    )
                rec = parse_last_json_line(proc.stdout)
                if rec.get("peak_rss_bytes") is None and host_rss is not None:
                    rec["peak_rss_bytes"] = host_rss
                suite_rounds.append(rec)
            return suite_rounds

        synth_rounds = run_suite(corpus_flag=False)
        corpus_rounds = run_suite(corpus_flag=True)

        synth_summary = summarize_rounds(synth_rounds, stream_read_gbs)
        corpus_summary = summarize_rounds(corpus_rounds, stream_read_gbs)

        parity_check: Optional[Dict[str, Any]] = None
        if local_ref_transport and local_ref_bin:
            ref_args = [
                "bench",
                "-m",
                str(model_path.resolve()),
                "-f",
                str(local_prompt.resolve()),
                "-n",
                str(args.n_predict),
                "-t",
                "4",
                "-c",
                str(args.max_ctx),
                "--json",
            ]
            ref_proc, _ = local_ref_transport.run_tinymodel(local_ref_bin, ref_args)
            if ref_proc.returncode == 0:
                ref_rec = parse_last_json_line(ref_proc.stdout)
                dev_hash = corpus_summary["greedy_tokens_fnv1a64"]
                ref_hash = ref_rec["greedy_tokens_fnv1a64"]
                parity_check = {
                    "host_isa": ref_rec["isa"],
                    "host_greedy_fnv1a64": ref_hash,
                    "device_greedy_fnv1a64": dev_hash,
                    "exact_match": dev_hash == ref_hash,
                }

        model_reports.append(
            {
                "model": model_path.name,
                "arch": synth_rounds[0]["arch"],
                "backend": synth_rounds[0]["backend"],
                "isa": synth_rounds[0]["isa"],
                "threads": eff_threads,
                "cpu_mask": cpu_mask,
                "synthetic_pp_tg": synth_summary,
                "frozen_corpus": {
                    "fixture": args.prompt_fixture,
                    "summary": corpus_summary,
                },
                "host_l3_parity": parity_check,
                "raw_rounds": {
                    "synthetic": synth_rounds,
                    "corpus": corpus_rounds,
                },
            }
        )

    commit = git_short_commit()
    today = dt.date.today().isoformat()
    dev_slug = slugify_device(
        extra_tel.get("android_model") or probe_info.get("device_model", transport.kind)
    )
    out_dir = RESULTS_ROOT / dev_slug
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{today}-{commit}.json"

    full_report = {
        "schema_version": 1,
        "protocol_id": manifest.get("protocol_id", "tinymodel-m0-v1"),
        "timestamp_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_commit": commit,
        "target": transport.describe(),
        "cluster_policy": args.cluster_policy,
        "effective_cpu_mask": cpu_mask,
        "effective_threads": eff_threads,
        "device_probe": probe_info,
        "extra_telemetry": extra_tel,
        "models": model_reports,
    }
    out_json.write_text(json.dumps(full_report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Print concise Markdown table summary
    print(f"\n### External Device Benchmark: `{probe_info.get('device_model')}` (`{transport.describe()}`)")
    print(
        f"- Active ISA: `{probe_info.get('active_isa')}` | Threads: `{eff_threads}` "
        f"| Mask: `{cpu_mask or 'all'}` | STREAM Read: `{stream_read_gbs or 0:.2f} GB/s`\n"
    )
    print("| Model | Backend/ISA | pp512 (tok/s) | tg (tok/s) | Corpus TTFT (ms) | Eff BW (GB/s) | BW Util | Peak RSS (MiB) |")
    print("|---|---|---:|---:|---:|---:|---:|---:|")
    for mr in model_reports:
        s = mr["synthetic_pp_tg"]
        c = mr["frozen_corpus"]["summary"]
        rss_mib = (
            f"{c['peak_rss_bytes_max'] / 1048576.0:.1f}"
            if c["peak_rss_bytes_max"] is not None
            else "-"
        )
        util_s = f"{s['stream_bw_utilization_pct']:.1f}%" if s["stream_bw_utilization_pct"] is not None else "-"
        print(
            f"| `{mr['model']}` | `{mr['backend']}/{mr['isa']}` | {s['pp_tps_median']:.2f} | "
            f"**{s['tg_tps_median']:.2f}** | {c['ttft_ms_median']:.1f} | "
            f"{s['decode_eff_bw_gbs_median']:.2f} | {util_s} | {rss_mib} |"
        )
    print(f"\nSaved schema-v1 result JSON -> {out_json}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="tinymodel External Device Test & Benchmark Harness")
    sub = p.add_subparsers(dest="subcmd", required=True)

    sub.add_parser("list", help="Discover attached Android ADB devices and check staged binaries")

    p_probe = sub.add_parser("probe", help="Probe remote/local device capabilities, clusters, L1 self-check & STREAM BW")
    p_probe.add_argument("--target", default="local", help="local | adb[:<serial>] | ssh://[user@]host[:port]")
    p_probe.add_argument("--remote-dir", default=None, help="Remote staging directory on device")
    p_probe.add_argument("--profile", choices=["edge", "micro"], default="edge", help="Cargo feature profile for cross-build")
    p_probe.add_argument("--build", action="store_true", help="Automatically build/cross-compile tinymodel before probing")
    p_probe.add_argument("-t", "--threads", type=int, default=None, help="Thread count for STREAM bandwidth test")
    p_probe.add_argument("--stream-mib", type=int, default=None, help="Override STREAM array size in MiB")
    p_probe.add_argument("--no-stream", action="store_true", help="Skip STREAM bandwidth test")
    p_probe.add_argument("--cpu-mask", default=None, help="Hex CPU affinity mask for taskset (e.g. 0xf0)")
    p_probe.add_argument("--force-isa", default=None, help="Set TM_FORCE_ISA on target (generic|neon|dotprod|i8mm)")

    p_run = sub.add_parser("run", help="Run full protocol benchmark + telemetry + L3 parity check on target device")
    p_run.add_argument("--target", default="local", help="local | adb[:<serial>] | ssh://[user@]host[:port]")
    p_run.add_argument("--remote-dir", default=None, help="Remote staging directory on device")
    p_run.add_argument("--profile", choices=["edge", "micro"], default="edge", help="Cargo feature profile for cross-build")
    p_run.add_argument("--build", action="store_true", help="Automatically build/cross-compile tinymodel before running")
    p_run.add_argument("-m", "--models", action="append", required=True, help="GGUF model path(s) to benchmark")
    p_run.add_argument("--reps", type=int, default=3, help="Number of benchmark repetitions (median reported)")
    p_run.add_argument("-n", "--n-predict", type=int, default=64, help="Decode tokens per run")
    p_run.add_argument("-t", "--threads", type=int, default=None, help="Override thread count")
    p_run.add_argument("-c", "--max-ctx", type=int, default=1024, help="Context window size")
    p_run.add_argument("--metal", action="store_true", help="Use Metal GPU backend (macOS local target)")
    p_run.add_argument(
        "--cluster-policy",
        choices=["all", "no-little", "mid-only"],
        default="no-little",
        help="Heterogeneous CPU scheduling policy (e.g. 'no-little' or 'mid-only' on Google Pixel 8)",
    )
    p_run.add_argument("--cpu-mask", default=None, help="Explicit hex CPU affinity mask for taskset")
    p_run.add_argument("--force-isa", default=None, help="Set TM_FORCE_ISA on target")
    p_run.add_argument("--stream-mib", type=int, default=None, help="Override STREAM array size in MiB")
    p_run.add_argument("--prompt-fixture", default="prompt_short.txt", help="Frozen fixture filename in bench/data/")
    p_run.add_argument("--compare-local", action="store_true", help="Compare greedy token hash against local host")

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.subcmd == "list":
        return cmd_list(args)
    if args.subcmd == "probe":
        return cmd_probe(args)
    if args.subcmd == "run":
        return cmd_run(args)
    return 1


if __name__ == "__main__":
    sys.exit(main())
