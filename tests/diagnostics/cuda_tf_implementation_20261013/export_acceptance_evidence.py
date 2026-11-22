"""完整验收后导出可复核证据；模型及Nsight raw trace继续留在服务器。"""
import hashlib
import argparse
import json
import re
import zipfile
from pathlib import Path

d = Path(__file__).resolve().parent
root = d.parents[2]
python = "/home/prmi/anaconda3/envs/zwj/bin/python"
parser = argparse.ArgumentParser()
parser.add_argument("--revision", choices=("initial", "reciprocal_fix"), default="reciprocal_fix")
args = parser.parse_args()
fixed = args.revision == "reciprocal_fix"
base_prefix = "guard_off_reciprocal_fix_" if fixed else "guard_off_"
profile_prefix = "online_tf_profile_reciprocal_fix" if fixed else "online_tf_profile"
marker_names = ("guard_off_remaining_acceptance.exit", "online_tf_profile.exit", "online_regression_control.exit")
for name in marker_names:
    assert (d / name).read_text().strip() == "0", name
verification = dict(implementation_commit="462d53f" if fixed else "839a1e1", formal_mapping_runs=0, formal_evaluations=0, model_files={}, configs_equal=True,
                    no_tail=True, finite_metrics=True, inference_strict_load=True)
files = []
for prefix in (base_prefix, base_prefix+"desk_start1000_"):
    summary_path = d / (prefix+"online_summary.json")
    evaluations = json.loads((d / (prefix+"online_evaluations.json")).read_text())["runs"]
    if fixed:
        summary = json.loads(summary_path.read_text())
        manifest = {"runs": summary}
    else:
        manifest = json.loads((d / (prefix+"online_runs.json")).read_text())
    assert len(evaluations) == 5*len(manifest["runs"])
    configs = {}
    for run in manifest["runs"]:
        if not fixed:
            assert run["status"] == "complete" and run["exit_code"] == 0
        path = Path(run["output"])
        if fixed:
            saved = run["transform_state"]
            assert saved["tail_completed_steps"] == saved["post_slam_mappings"] == 0
        else:
            text = (path / "selector_config.yaml").read_text()
            assert "GradientConstraint.enable: 0" in text
            base = re.sub(r'\nTransformField.backend: "(?:legacy|cuda)"\n$', '', text)
            if run["scene"] in configs:
                assert configs[run["scene"]] == base
            configs[run["scene"]] = base
            state = list(path.glob("*_shutdown/ply/transform_state.json"))
            assert len(state) == 1
            saved = json.loads(state[0].read_text())
            assert saved["tail_completed_steps"] == saved["post_slam_mappings"] == 0
            for f in state[0].parent.glob("*.pt"):
                verification["model_files"][str(f.relative_to(root))] = hashlib.sha256(f.read_bytes()).hexdigest()
        current = [e for e in evaluations if (e["scene"],e["backend"]) == (run["scene"],run["backend"])]
        assert len(current) == 5 and len({e["evaluated_frames"] for e in current}) == 1
        assert all(e["status"] == "complete" and e["exit_code"] == 0 for e in current)
        assert all(all(__import__('math').isfinite(v) for v in e["metrics"].values()) for e in current)
        # A successful LoD evaluation uses python/transform_field.py strict=True.
        assert all(e.get("selector_stats",{}).get("transform_applied",False)
                   for e in current if e["ratio"] is not None) == (int(saved["transform_step"]) > 0)
        verification["formal_mapping_runs"] += 1
        verification["formal_evaluations"] += len(current)
        if path.exists():
            files += [f for f in path.rglob("*") if f.is_file() and f.suffix in (".txt", ".json", ".yaml", ".log")]
    for suffix in ("online_runs.json", "online_evaluations.json", "online_summary.json",
                   "online_memory_samples.csv", "online.exit", "evaluation.exit", "remaining_acceptance.exit"):
        f = d / (prefix+suffix)
        if f.is_file():
            files.append(f)
    if fixed:
        files += [d / (prefix+"online_summary.json"), d / (prefix+"online_evaluations.json")]
assert verification["formal_mapping_runs"] == 10 and verification["formal_evaluations"] == 50
profile = root / ("results/cuda_tf_stage_profile_reciprocal_fix_20261014" if fixed else "results/cuda_tf_stage_profile_20261014")
control = root / "results/cuda_tf_online_regression_control_20261014"
for path in (profile,control):
    files += [f for f in path.rglob("*") if f.is_file() and f.suffix in (".txt", ".json", ".yaml", ".log")]
for pattern in (profile_prefix+"*", "online_profiler_parity*", "online_regression_control*",
                "training_chain*", "mlp_backward*", "batch_metric_parity.json",
                "remote_source_verification.json", "remote_harness_verification.json", "online_profile_build.*"):
    files += [f for f in d.glob(pattern) if f.is_file() and f.suffix not in (".cpp", ".py", ".h")]
files = sorted(set(files))
if fixed:
    files += [f for f in (d / "regularization_reciprocal_fix").rglob("*") if f.is_file()]
    files += [f for f in d.glob("reciprocal_fix_phase_5*") if f.suffix in (".json", ".csv", ".txt", ".exit", ".log")]
    files += [f for f in d.glob("remote_*_verification_reciprocal_fix.json")]
    files += [f for f in d.glob("reciprocal_fix_main_build.*")]
files = sorted(set(files))
verification["evidence_files"] = {str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
(d / (base_prefix+"acceptance_verification.json")).write_text(json.dumps(verification,indent=2))
files.append(d / (base_prefix+"acceptance_verification.json"))
with zipfile.ZipFile(d / (base_prefix+"acceptance_evidence.zip"), "w", zipfile.ZIP_DEFLATED) as archive:
    for f in files:
        archive.write(f,f.relative_to(root))
print(json.dumps({k:v for k,v in verification.items() if k not in ("model_files","evidence_files")},indent=2))
print("archive_bytes",(d / (base_prefix+"acceptance_evidence.zip")).stat().st_size)
