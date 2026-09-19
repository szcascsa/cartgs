"""Evaluate selector LOD ratios by reusing the standard ``run.py`` pipeline.

The source result directory is never modified. Each ratio gets an independent
output directory containing rendered images and the standard metric files.
"""

import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys


def read_selector_config(config_path):
    """Read LOD ratios and selector maturity settings from an OpenCV YAML file."""
    with open(config_path, "r", encoding="utf-8") as stream:
        text = stream.read()

    ratio_match = re.search(
        r"Selector\.(?:lod_ratios|target_ratios)\s*:\s*\[([^]]+)\]",
        text,
    )
    if not ratio_match:
        raise ValueError(
            "config has no Selector.lod_ratios or Selector.target_ratios: {}".format(
                config_path
            )
        )
    ratios = [float(value) for value in re.findall(r"[-+]?\d*\.??\d+", ratio_match.group(1))]
    if not ratios or any(ratio <= 0.0 or ratio > 1.0 for ratio in ratios):
        raise ValueError("selector ratios must be in (0, 1]: {}".format(ratios))

    def read_int(key, default):
        match = re.search(r"{}\s*:\s*(-?\d+)".format(re.escape(key)), text)
        return int(match.group(1)) if match else default

    def read_float(key, default):
        match = re.search(
            r"{}\s*:\s*([-+]?\d*\.?\d+)".format(re.escape(key)), text
        )
        return float(match.group(1)) if match else default

    return {
        "ratios": ratios,
        "min_age": read_int("Selector.min_age", 10000),
        "min_seen": read_int("Selector.min_seen", 8000),
        "protection_enabled": read_int("Selector.protection_enabled", 1) != 0,
        "temperature": read_float("Selector.temperature", 1.0),
    }


def find_selector_checkpoint(result_path, explicit_path):
    if explicit_path:
        return os.path.abspath(explicit_path)
    candidates = sorted(
        glob.glob(os.path.join(result_path, "**", "selector.pt"), recursive=True)
    )
    if not candidates:
        raise FileNotFoundError(
            "selector.pt was not found below result path: {}".format(result_path)
        )
    return candidates[-1]


def ratio_directory_name(ratio):
    return "ratio_{}".format(format(ratio, ".6f").rstrip("0").rstrip(".").replace(".", "p"))


def parse_eval_file(path):
    metrics = {}
    if not os.path.isfile(path):
        return metrics
    with open(path, "r", encoding="utf-8") as stream:
        for line in stream:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            try:
                metrics[key.strip()] = float(value.strip())
            except ValueError:
                continue
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate selector LOD ratios")
    parser.add_argument("result_path", help="completed CaRtGS result directory")
    parser.add_argument("gt_path", help="dataset ground-truth directory")
    parser.add_argument(
        "--config",
        default=None,
        help="CaRtGS config; defaults to cfg/colmap/gaussian_splatting.yaml",
    )
    parser.add_argument("--selector", default=None, help="path to selector.pt")
    parser.add_argument("--output", default=None, help="LOD output directory")
    parser.add_argument("--run-script", default=None, help="path to run.py")
    parser.add_argument("--ratios", nargs="+", type=float, default=None)
    parser.add_argument("--correct-scale", action="store_true")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(script_dir)
    config_path = os.path.abspath(
        args.config or os.path.join(repo_dir, "cfg", "colmap", "gaussian_splatting.yaml")
    )
    run_script = os.path.abspath(args.run_script or os.path.join(script_dir, "run.py"))
    result_path = os.path.abspath(args.result_path)
    gt_path = os.path.abspath(args.gt_path)
    output_root = os.path.abspath(args.output or os.path.join(result_path, "lod_eval"))
    config = read_selector_config(config_path)
    ratios = args.ratios or config["ratios"]
    selector_path = find_selector_checkpoint(result_path, args.selector)
    os.makedirs(output_root, exist_ok=True)

    summary = []
    for ratio in ratios:
        if ratio <= 0.0 or ratio > 1.0:
            raise ValueError("ratio must be in (0, 1]: {}".format(ratio))
        ratio_output = os.path.join(output_root, ratio_directory_name(ratio))
        os.makedirs(ratio_output, exist_ok=True)
        command = [
            sys.executable,
            run_script,
            result_path,
            gt_path,
            "--output_path",
            ratio_output,
            "--selector",
            selector_path,
            "--selector-ratio",
            str(ratio),
            "--selector-min-age",
            str(config["min_age"]),
            "--selector-min-seen",
            str(config["min_seen"]),
            "--selector-protection-enabled",
            str(int(config["protection_enabled"])),
            "--selector-temperature",
            str(config["temperature"]),
        ]
        if args.correct_scale:
            command.append("--correct_scale")
        subprocess.run(command, check=True)

        row = {
            "ratio": float(ratio),
            "output_path": ratio_output,
            **parse_eval_file(os.path.join(ratio_output, "eval.txt")),
        }
        stats_path = os.path.join(ratio_output, "selector_stats.json")
        if os.path.isfile(stats_path):
            with open(stats_path, "r", encoding="utf-8") as stream:
                row.update(json.load(stream))
        summary.append(row)

    summary_json = os.path.join(output_root, "summary.json")
    with open(summary_json, "w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2)

    summary_csv = os.path.join(output_root, "summary.csv")
    fieldnames = sorted({key for row in summary for key in row})
    with open(summary_csv, "w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary)

    print("LOD summary: {}".format(summary_json))
    print("LOD summary CSV: {}".format(summary_csv))


if __name__ == "__main__":
    main()
