import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "benchmark_streaming_gi", Path(__file__).resolve().parents[1] / "scripts" / "benchmark_streaming_gi.py")
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)

report_spec = importlib.util.spec_from_file_location(
    "report_streaming_gi_benchmark", Path(__file__).resolve().parents[1] / "scripts" / "report_streaming_gi_benchmark.py")
report = importlib.util.module_from_spec(report_spec)
report_spec.loader.exec_module(report)


class BenchmarkParsing(unittest.TestCase):
    def test_native_summary_units_and_phase_keys(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "model").mkdir()
            (root / "model" / "TrackingTime.txt").write_text("0.020\n0.040\n")
            (root / "gpu.csv").write_text("timestamp, 15, 200\ntimestamp, 25, 201\n")
            (root / "run.log").write_text(
                "[UltrafastRotation SUMMARY] event=save ultrafast_lod_001_steps=2 "
                "ultrafast_total_normal_steps=10 tail_training_steps=0 mean_mapping_training_ms=5\n"
                "[GlobalGI] event=final iteration=10 global_gi_bank_total_ms=20 "
                "global_gi_incremental_total_ms=3 global_gi_topology_cumulative_ms=1 "
                "global_gi_full_refresh_total_ms=6\n"
                "[Mapping PERF FINAL] complete_mapping_mean_ms=8 complete_mapping_p95_ms=12 "
                "gi_teacher_host_ms_per_step=0.1 gi_loss_host_ms_per_step=0.05\n")
            result = benchmark.summarize(root, 1.0, 0)
            self.assertEqual(result["ultrafast_lod_001_steps"], 2)
            self.assertEqual(result["gi_maintenance_ms_per_step"], 3)
            self.assertAlmostEqual(result["gi_including_teacher_ms_per_step"], 3.15)
            self.assertEqual(result["tracking_mean_ms"], 30)
            self.assertEqual(result["tracking_p95_ms"], 40)
            self.assertEqual(result["complete_mapping_p95_ms"], 12)
            self.assertEqual(result["gpu_mean_percent"], 20)

    def test_per_run_normalization_and_failed_run_exclusion(self):
        def row(steps, total_ms, mean_ms):
            return {"returncode": 0, "ultrafast_total_normal_steps": steps,
                    "complete_mapping_mean_ms": mean_ms,
                    "global_gi_bank_snapshot_total_ms": total_ms,
                    "global_gi_bank_selection_total_ms": 0,
                    "global_gi_incremental_total_ms": 0,
                    "global_gi_topology_cumulative_ms": 0,
                    "global_gi_full_refresh_total_ms": 0}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = [row(10, 20, 8), row(20, 20, 10), {"returncode": 134}]
            for index, value in enumerate(rows):
                directory = root / f"run_{index + 1}"
                directory.mkdir()
                (directory / "summary.json").write_text(json.dumps(value))
            result = report.batch(root)
            self.assertEqual(len(result["runs"]), 2)
            self.assertEqual(len(result["failures"]), 1)
            self.assertEqual(result["medians"]["gi_maintenance_ms_per_step"], 1.5)
            self.assertEqual(result["medians"]["complete_mapping_mean_ms"], 9)
            self.assertEqual(result["runs"][0]["gi_maintenance_share_percent"], 25)


if __name__ == "__main__":
    unittest.main()
