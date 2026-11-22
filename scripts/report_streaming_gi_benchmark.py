"""Aggregate per-run normalized GI measurements; failed runs remain visible."""
import argparse
import json
import statistics
from pathlib import Path


def normalize(row):
    row = dict(row)
    steps = row["ultrafast_total_normal_steps"]
    mapping = row["complete_mapping_mean_ms"] if "complete_mapping_mean_ms" in row else row["mapping_iteration_ms_mean"]
    row["complete_mapping_mean_ms"] = mapping
    row["mapping_steps_per_second"] = 1000 / mapping
    stages = {
        "snapshot": "global_gi_bank_snapshot_total_ms",
        "bank_selection": "global_gi_bank_selection_total_ms",
        "incremental": "global_gi_incremental_total_ms",
        "topology": "global_gi_topology_cumulative_ms",
        "refresh": "global_gi_full_refresh_total_ms",
    }
    for stage, key in stages.items():
        row[stage + "_ms_per_step"] = row[key] / steps
    row["gi_maintenance_total_ms"] = sum(row[key] for key in stages.values())
    row["gi_maintenance_ms_per_step"] = row["gi_maintenance_total_ms"] / steps
    row["gi_maintenance_share_percent"] = 100 * row["gi_maintenance_ms_per_step"] / mapping
    if "gi_teacher_host_ms_per_step" in row:
        row["gi_including_teacher_ms_per_step"] = (row["gi_maintenance_ms_per_step"]
            + row["gi_teacher_host_ms_per_step"] + row["gi_loss_host_ms_per_step"])
        row["gi_including_teacher_share_percent"] = 100 * row["gi_including_teacher_ms_per_step"] / mapping
    return row


def batch(root):
    runs = []
    failures = []
    for path in sorted(root.glob("run_*/summary.json")):
        row = json.loads(path.read_text())
        if row["returncode"] != 0:
            failures.append({"path": str(path), "returncode": row["returncode"]})
            continue
        row = normalize(row)
        row["path"] = str(path.parent)
        runs.append(row)
    keys = set.intersection(*(set(row) for row in runs)) if runs else set()
    medians = {key: statistics.median(row[key] for row in runs)
               for key in keys if all(isinstance(row[key], (int, float)) for row in runs)}
    return {"runs": runs, "failures": failures, "medians": medians}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--baseline", default="baseline_retry")
    parser.add_argument("--optimized", default="optimized")
    args = parser.parse_args()
    report = {"baseline": batch(args.root / args.baseline),
              "instrumented_baseline": batch(args.root / "profile_baseline"),
              "optimized": batch(args.root / args.optimized),
              "validation": batch(args.root / "validate_snapshot")}
    before = report["baseline"]["medians"]
    after = report["optimized"]["medians"]
    if before and after:
        report["mapping_speedup"] = before["complete_mapping_mean_ms"] / after["complete_mapping_mean_ms"]
        report["normalized_gi_speedup"] = before["gi_maintenance_ms_per_step"] / after["gi_maintenance_ms_per_step"]
        report["normalized_gi_reduction_percent"] = 100 * (1 - 1 / report["normalized_gi_speedup"])
    for name in ("quality_baseline", "quality_optimized"):
        path = args.root / name / "summary.json"
        if path.exists():
            report[name] = json.loads(path.read_text())
    (args.root / "comparison.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if not isinstance(value, (dict, list))}, indent=2))


if __name__ == "__main__":
    main()
