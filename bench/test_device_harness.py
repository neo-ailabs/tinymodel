#!/usr/bin/env python3
"""Unit tests for `bench/device_harness.py` (manifest verification, Pixel 8 cluster policies, summarization)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import device_harness as dh


class TestDeviceHarness(unittest.TestCase):
    def test_frozen_manifest_validates(self) -> None:
        manifest = dh.verify_prompt_manifest()
        self.assertEqual(manifest["schema_version"], 1)
        self.assertIn("prompt_short.txt", manifest["files"])

    def test_pixel8_cluster_policies(self) -> None:
        # Google Pixel 8 (Tensor G3): cpus 0..3 = A510, cpus 4..7 = A715, cpu 8 = X3
        pixel8_clusters = [
            {"uarch": "Cortex-A510", "max_freq_mhz": 1704, "count": 4, "cpus": [0, 1, 2, 3], "cpu_mask_hex": "0xf"},
            {"uarch": "Cortex-A715", "max_freq_mhz": 2367, "count": 4, "cpus": [4, 5, 6, 7], "cpu_mask_hex": "0xf0"},
            {"uarch": "Cortex-X3", "max_freq_mhz": 2910, "count": 1, "cpus": [8], "cpu_mask_hex": "0x100"},
        ]
        mask_nl, thr_nl = dh.select_cluster_mask_and_threads(pixel8_clusters, "no-little")
        self.assertEqual(mask_nl, "0x1f0")
        self.assertEqual(thr_nl, 5)

        mask_mid, thr_mid = dh.select_cluster_mask_and_threads(pixel8_clusters, "mid-only")
        self.assertEqual(mask_mid, "0xf0")
        self.assertEqual(thr_mid, 4)

        mask_all, thr_all = dh.select_cluster_mask_and_threads(pixel8_clusters, "all")
        self.assertIsNone(mask_all)
        self.assertEqual(thr_all, 8)

    def test_summarize_rounds_computes_bw_util_and_joules(self) -> None:
        rounds = [
            {
                "load_ms": 40.0,
                "ttft_ms": 210.0,
                "pp_tokens": 292,
                "pp_tps": 1390.0,
                "tg_tokens": 64,
                "tg_tps": 50.0,
                "decode_eff_bw_gbs": 24.0,
                "peak_rss_bytes": 550 * 1024 * 1024,
                "greedy_tokens_fnv1a64": "0123456789abcdef",
                "telemetry_before": {"battery_power_mw": 3800.0, "max_temp_c": 36.0},
                "telemetry_after": {"battery_power_mw": 4200.0, "max_temp_c": 41.5},
            },
            {
                "load_ms": 38.0,
                "ttft_ms": 200.0,
                "pp_tokens": 292,
                "pp_tps": 1460.0,
                "tg_tokens": 64,
                "tg_tps": 52.0,
                "decode_eff_bw_gbs": 25.0,
                "peak_rss_bytes": 552 * 1024 * 1024,
                "greedy_tokens_fnv1a64": "0123456789abcdef",
                "telemetry_before": {"battery_power_mw": 4000.0, "max_temp_c": 41.5},
                "telemetry_after": {"battery_power_mw": 4000.0, "max_temp_c": 43.0},
            },
        ]
        summary = dh.summarize_rounds(rounds, stream_read_gbs=30.0)
        self.assertEqual(summary["tg_tps_median"], 51.0)
        self.assertAlmostEqual(summary["stream_bw_utilization_pct"], 81.67, places=2)
        self.assertEqual(summary["avg_power_mw"], 4000.0)
        self.assertAlmostEqual(summary["decode_joules_per_token"], 4.0 / 51.0, places=3)
        self.assertTrue(summary["deterministic_self_consistent"])


if __name__ == "__main__":
    unittest.main()
