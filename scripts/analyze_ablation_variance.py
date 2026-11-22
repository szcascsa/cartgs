"""Quantify run-to-run variance of CaRtGS ablation runs and expose uncontrolled variables.

The script never decides which arm is better. It answers two questions from the
artifacts an existing run already writes:

1. How large is the repeat-to-repeat spread inside one arm (the noise floor)?
2. Which *causal* variables (mapping steps, ChangeDistill dose, TF start, evaluation
   frame set, Gaussian population, trajectory length) still vary between those
   repeats, i.e. which knob was not actually held fixed?

Usage:
  python scripts/analyze_ablation_variance.py \
      --arm baseline=results/change_distill_ablation/run_a \
      --arm epochs20=results/change_distill_ablation/run_b \
      [--json variance.json] [--target-delta 0.2]

Each --arm points at a directory holding one sub-directory per repeat. A repeat is
any directory containing ``eval.txt`` (run.py output) or ``transform_state.json``
(mapper shutdown dump). Sibling artifacts are searched in the repeat directory and
up to ``--search-up`` parents; model-side state is searched only inside the repeat
directory so one repeat can never borrow another repeat's numbers.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import sys
from pathlib import Path

# Measured outcome.
OUTCOME_FIELDS = ("psnr", "ssim", "lpips", "translation_rmse", "rotation_rmse")
# Variables that describe *why* two runs can differ. They must agree across
# repeats before an arm-to-arm difference can be called an effect of the knob.
CONTROL_FIELDS = (
    "mapping_steps",
    "eval_frames",
    "traj_rows",
    "total_gaussians",
    "cd_selector_steps",
    "cd_tf_steps",
    "cd_events",
    "tf_start_mapping_iteration",
    "tail_completed_steps",
)

FLOAT = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?"


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def parse_key_values(path: Path) -> dict:
    values = {}
    for key, value in re.findall(r"^\s*([A-Za-z][\w ]*?)\s*:\s*(" + FLOAT + r")",
                                 read_text(path), re.M):
        values[key.strip()] = float(value)
    return values


def parse_traj_rmse(path: Path) -> dict:
    # evo writes one "rmse" row per call; run.py calls it for translation then rotation.
    rmse = [float(v) for v in
            re.findall(r"^\s*rmse\s+(" + FLOAT + r")", read_text(path), re.M)]
    result = {}
    if len(rmse) >= 1:
        result["translation_rmse"] = rmse[0]
    if len(rmse) >= 2:
        result["rotation_rmse"] = rmse[1]
    return result


def parse_state(path: Path) -> dict:
    try:
        data = json.loads(read_text(path) or "{}")
    except json.JSONDecodeError:
        return {}
    flat = {}
    for key, value in data.items():
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            flat[key] = float(value)
        elif isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, (int, float)) and not isinstance(sub_value, bool):
                    flat["{}.{}".format(key, sub_key)] = float(sub_value)
    return flat


def ply_vertex_count(path: Path) -> float | None:
    with path.open("rb") as stream:
        head = stream.read(2048).decode("ascii", errors="replace")
    found = re.search(r"element vertex (\d+)", head)
    return float(found.group(1)) if found else None


def find_ancestor(start: Path, names, levels: int) -> Path | None:
    """Look for names in start and its first `levels` parents (non-recursive)."""
    for name in names:
        for folder in [start] + list(start.parents)[:levels]:
            candidate = folder / name
            if candidate.is_file():
                return candidate
    return None


def find_inside(start: Path, names) -> Path | None:
    """Look for names only inside start's own subtree; never outside the repeat."""
    for name in names:
        matches = sorted(path for path in start.rglob(name) if path.is_file())
        if matches:
            return matches[-1]
    return None


def parse_log(path: Path) -> dict:
    text = read_text(path)
    if not text:
        return {}
    result = {}
    steps = re.findall(r"^(\d+) iterations completed\.$", text, re.M)
    if steps:
        result["mapping_steps"] = float(steps[-1])
    # Data-dependent events: these decide how often an ablation knob is exercised.
    for marker, key in (
        ("[CD BEGIN]", "cd_events"),
        ("[Gaussian Mapper]Loop Closure Detected.", "loop_closures"),
        ("[Gaussian Mapper]Scale refinement Detected.", "scale_refinements"),
        ("[ChangeCoordinator] skip", "cd_skips"),
        ("[TF CD SKIP]", "tf_cd_skips"),
        ("[SELECTOR CD SKIP]", "selector_cd_skips"),
    ):
        count = text.count(marker)
        if count:
            result[key] = float(count)
    summary = re.findall(r"\[UltrafastRotation SUMMARY\][^\n]*", text)
    if summary:
        for key, value in re.findall(r"(\w+)=(" + FLOAT + r")", summary[-1]):
            result.setdefault("summary_" + key, float(value))
        result.setdefault("mapping_steps", result.get("summary_ultrafast_total_normal_steps"))
    return result


def discover_repeats(folder: Path, name: str, depth: int | None) -> list[Path]:
    """Return one directory per repeat.

    Nested evaluations (``lod_eval/ratio_0p05/eval.txt``) must not be counted as
    extra repeats, so only the shallowest artifact depth is used unless --depth
    says otherwise.
    """
    found = [{path.parent for path in folder.rglob(name)} for name in
             ("eval.txt", "transform_state.json")]
    candidates = found[0] or found[1]
    if not candidates:
        return []
    depths = {len(path.relative_to(folder).parts): path for path in candidates}
    selected = depth if depth is not None else min(depths)
    return sorted({path for path in candidates
                   if len(path.relative_to(folder).parts) == selected})


def collect_run(run_dir: Path, label: str, search_up: int) -> dict:
    row = {"run": label}

    evaluation = find_ancestor(run_dir, ["eval.txt"], search_up)
    if evaluation:
        row.update(parse_key_values(evaluation))

    rmse = find_ancestor(run_dir, ["metrics_traj.txt"], search_up)
    if rmse:
        row.update(parse_traj_rmse(rmse))

    frame_set = find_ancestor(run_dir, ["evaluation_frames.json"], search_up)
    if frame_set:
        try:
            indices = json.loads(read_text(frame_set)).get("gt_indices", [])
            row["eval_frames"] = float(len(indices))
        except (json.JSONDecodeError, TypeError):
            pass
    if "eval_frames" not in row:
        # run.py always writes one line per evaluated frame next to eval.txt.
        per_frame = find_ancestor(run_dir, ["psnr.txt"], search_up)
        if per_frame:
            lines = [line for line in read_text(per_frame).splitlines() if line.strip()]
            row["eval_frames"] = float(len(lines))

    trajectory = find_ancestor(run_dir, ["CameraTrajectory_TUM.txt"], search_up)
    if trajectory:
        lines = [line for line in read_text(trajectory).splitlines() if line.strip()]
        row["traj_rows"] = float(len(lines))

    tracking = find_ancestor(run_dir, ["TrackingTime.txt"], search_up)
    if tracking:
        values = [float(v) for v in re.findall(FLOAT, read_text(tracking))]
        if values:
            row["tracked_frames"] = float(len(values))
            row["tracking_mean_ms"] = statistics.mean(values) * 1000.0

    stats = find_inside(run_dir, ["selector_stats.json"])
    if stats:
        parsed = parse_state(stats)
        for key in ("total_gaussians", "active_gaussians", "actual_vs_target"):
            if key in parsed:
                row[key] = parsed[key]

    state = find_inside(run_dir, ["transform_state.json", "mapping_stage_state.json"])
    if state:
        parsed = parse_state(state)
        row.update({key: parsed[key] for key in (
            "cd_selector_steps", "cd_tf_steps", "tf_start_mapping_iteration",
            "tf_initialization_iteration", "first_ready_population_iteration",
            "tail_completed_steps") if key in parsed})

    if "total_gaussians" not in row:
        ply = find_inside(run_dir, ["point_cloud.ply"])
        if ply:
            count = ply_vertex_count(ply)
            if count is not None:
                row["total_gaussians"] = count

    log = find_ancestor(run_dir, ["train.log", "stdout.log", "run.log"], search_up)
    if log:
        row.update(parse_log(log))
    return row


def distribution(values):
    values = [v for v in values if isinstance(v, (int, float))]
    if not values:
        return None
    mean = statistics.mean(values)
    std = statistics.stdev(values) if len(values) > 1 else None
    return {"n": len(values), "mean": mean, "std": std, "min": min(values),
            "max": max(values), "values": values,
            "cv": (std / abs(mean)) if std is not None and mean else None}


def mde(sigma: float, n_per_arm: int) -> float | None:
    """Minimum detectable effect, two-sided alpha=0.05, power=0.80, normal approx."""
    if not sigma or n_per_arm < 2:
        return None
    return (1.959964 + 0.841621) * sigma * math.sqrt(2.0 / n_per_arm)


def required_n(sigma: float, delta: float) -> int | None:
    if not sigma or delta <= 0:
        return None
    return max(2, int(math.ceil(2.0 * ((1.959964 + 0.841621) * sigma / delta) ** 2)))


def format_cell(item) -> str:
    if not item:
        return "missing"
    if item["std"] is None:
        return "{:.4f} (single run)".format(item["mean"])
    return "{:.4f} +/- {:.4f} [{:.4f}, {:.4f}]".format(
        item["mean"], item["std"], item["min"], item["max"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--arm", action="append", required=True, metavar="NAME=DIR",
                        help="repeatable; DIR holds one sub-directory per repeat")
    parser.add_argument("--json", default=None, help="write the full table as JSON")
    parser.add_argument("--search-up", type=int, default=2,
                        help="parents searched for sibling artifacts (default 2)")
    parser.add_argument("--depth", type=int, default=None,
                        help="artifact depth under each arm that defines a repeat; default = "
                             "shallowest found, which keeps nested lod_eval out of the count")
    parser.add_argument("--target-delta", type=float, default=None,
                        help="effect size to resolve, e.g. 0.2 (dB PSNR)")
    args = parser.parse_args()

    arms = {}
    for item in args.arm:
        if "=" not in item:
            parser.error("--arm expects NAME=DIR, got {!r}".format(item))
        name, folder = item.split("=", 1)
        arms[name] = Path(folder).expanduser().resolve()

    report = {"arms": {}, "checks": [], "power": {}}
    for name, folder in arms.items():
        if not folder.is_dir():
            print("missing arm directory: {}".format(folder), file=sys.stderr)
            return 2
        repeat_dirs = discover_repeats(folder, "eval.txt", args.depth)
        if not repeat_dirs:
            print("no eval.txt / transform_state.json under {}".format(folder), file=sys.stderr)
            return 2
        rows = [collect_run(repeat, str(repeat.relative_to(folder)), args.search_up)
                for repeat in repeat_dirs]
        fields = OUTCOME_FIELDS + CONTROL_FIELDS + (
            "loop_closures", "scale_refinements", "cd_skips", "tf_cd_skips",
            "tracked_frames", "tracking_mean_ms", "active_gaussians", "cd_selector_steps")
        seen = dict.fromkeys(fields)
        stats = {field: distribution([row.get(field) for row in rows]) for field in seen}
        report["arms"][name] = {"directory": str(folder), "runs": rows, "statistics": stats}

    # --- 1. noise floor ------------------------------------------------------
    print("\n## 1. Repeat-to-repeat spread inside each arm (noise floor)")
    header = "| metric | " + " | ".join(
        "{} (n={})".format(name, report["arms"][name]["statistics"]["psnr"]["n"]
                           if report["arms"][name]["statistics"]["psnr"] else 0)
        for name in arms) + " |"
    print(header)
    print("|---" * (len(arms) + 1) + "|")
    for field in OUTCOME_FIELDS:
        cells = [format_cell(report["arms"][name]["statistics"].get(field)) for name in arms]
        print("| {} | {} |".format(field, " | ".join(cells)))

    # --- 2. were the causal variables actually fixed? -----------------------
    print("\n## 2. Causal variables: fixed, or still moving between repeats?")
    print("| variable | " + " | ".join(arms) + " |")
    print("|---" * (len(arms) + 1) + "|")
    for field in CONTROL_FIELDS + ("loop_closures", "scale_refinements", "cd_skips",
                                   "tracked_frames", "tracking_mean_ms", "active_gaussians"):
        cells, present = [], False
        for name in arms:
            item = report["arms"][name]["statistics"].get(field)
            if not item:
                cells.append("-")
                continue
            present = True
            unique = sorted(set(round(v, 6) for v in item["values"]))
            if len(unique) == 1:
                cells.append("{} (fixed)".format(round(item["mean"], 4)))
            elif item["mean"]:
                spread = 100.0 * (item["max"] - item["min"]) / abs(item["mean"])
                cells.append("{}..{} (spread {:.1f}%)".format(
                    round(item["min"], 4), round(item["max"], 4), spread))
            else:
                cells.append("{}..{}".format(item["min"], item["max"]))
        if present:
            print("| {} | {} |".format(field, " | ".join(cells)))

    # --- 3. explicit verdicts ----------------------------------------------
    print("\n## 3. Verdicts")
    for name in arms:
        stats = report["arms"][name]["statistics"]

        def spread(field):
            item = stats.get(field)
            if not item or item["std"] is None:
                return None
            return item["max"] - item["min"]

        steps, frames = spread("mapping_steps"), spread("eval_frames")
        dose, population = spread("cd_tf_steps"), spread("total_gaussians")
        report["checks"].append({"arm": name, "mapping_step_spread": steps,
                                 "eval_frame_spread": frames, "cd_tf_dose_spread": dose,
                                 "population_spread": population})
        if steps:
            print("- {}: normal training steps are not fixed (range {:.0f}). The online "
                  "driver derives the iteration count from the SLAM stop moment, not from "
                  "a budget.".format(name, steps))
        if frames:
            print("- {}: the evaluated frame set differs (range {:.0f} frames), so mean "
                  "PSNR is averaged over different pixels; paired arms must pass "
                  "--evaluation-frame-set.".format(name, frames))
        if dose:
            print("- {}: actual ChangeDistill dose differs (cd_tf_steps range {:.0f}). "
                  "Dose = events x epochs x batches, and the event count is driven by "
                  "SLAM loop closure / scale refinement, not by the config.".format(name, dose))
        if population:
            mean = stats["total_gaussians"]["mean"] or 1
            print("- {}: final Gaussian population differs (range {:.0f}, {:.1f}% of mean). "
                  "Densification is a discrete threshold event; this difference alone "
                  "produces a structurally different model.".format(
                      name, population, 100.0 * population / mean))

    # --- 4. statistical power ----------------------------------------------
    print("\n## 4. What this sample size can resolve (alpha=0.05, power=0.80)")
    print("| metric | pooled sigma (within arm) | MDE at n=3 | MDE at n=5 | n needed for target {} |".format(
        args.target_delta if args.target_delta is not None else "-"))
    print("|---|---|---|---|---|")
    for field in ("psnr", "ssim", "lpips", "translation_rmse"):
        sigmas = [report["arms"][n]["statistics"][field]["std"] for n in arms
                  if report["arms"][n]["statistics"].get(field)
                  and report["arms"][n]["statistics"][field]["std"]]
        if not sigmas:
            continue
        sigma = math.sqrt(sum(s * s for s in sigmas) / len(sigmas))
        need = required_n(sigma, args.target_delta) if args.target_delta else None
        report["power"][field] = {"pooled_sigma": sigma, "mde_n3": mde(sigma, 3),
                                  "mde_n5": mde(sigma, 5), "required_n": need}
        print("| {} | {:.4f} | {:.4f} | {:.4f} | {} |".format(
            field, sigma, mde(sigma, 3) or float("nan"), mde(sigma, 5) or float("nan"),
            need if need else "-"))

    print("\nNote: sigma above is the pooled standard deviation of same-config repeats. "
          "Any target effect smaller than the n=3 column cannot be established by three "
          "repeats; either raise n or remove a noise source first.")

    if args.json:
        Path(args.json).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                   encoding="utf-8")
        print("\nJSON written to {}".format(args.json))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
