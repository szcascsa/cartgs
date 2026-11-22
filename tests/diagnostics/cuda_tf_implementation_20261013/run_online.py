"""同一完整RGB-D序列与fast配置，仅backend不同；串行占用GPU。"""
import hashlib
import argparse
import json
import os
import subprocess
import time
from pathlib import Path

root = Path(__file__).resolve().parents[3]
diagnostics = Path(__file__).resolve().parent
output = root / "results" / "cuda_tf_online_guard_off_20261014"
datasets = Path("/home/prmi/datasets/TUM_RGBD")
parser = argparse.ArgumentParser()
parser.add_argument("--protocol", choices=("original", "desk_start1000"), default="original")
args = parser.parse_args()
prefix = "guard_off_" if args.protocol == "original" else "guard_off_desk_start1000_"
scenes = [
    ("long", "rgbd_dataset_freiburg3_long_office_household", "tum_freiburg3_long_office_household"),
    ("room", "rgbd_dataset_freiburg1_room", "tum_freiburg1_desk"),
    ("desk", "rgbd_dataset_freiburg1_desk", "tum_freiburg1_desk"),
    ("xyz", "rgbd_dataset_freiburg2_xyz", "tum_freiburg2_xyz"),
]
environment = os.environ.copy()
environment["LD_LIBRARY_PATH"] = ":".join([
    str(root / "third_party/libtorch-cu118/libtorch/lib"),
    "/usr/local/cuda-11.8/lib64", str(root / "lib"),
    str(root / "third_party/ORB-SLAM3/lib"),
    environment.get("LD_LIBRARY_PATH", ""),
])
base = (root / "cfg/selector/fast.yaml").read_text()
assert "Training.force_tail_optimization_enabled: 0" in base
assert "Training.ultra_light_enabled: 0" in base
assert "Selector.detach_gaussian_parameters: 0" in base
assert "TransformField.start_iter: 3000" in base
assert "TransformField.backend:" not in base
assert "GradientConstraint.enable: 0" in base
tf_start = 3000
if args.protocol == "desk_start1000":
    # User-authorized supplementary pair, kept separate from original fast runs.
    base = base.replace("TransformField.start_iter: 3000", "TransformField.start_iter: 1000")
    tf_start = 1000
    scenes = [entry for entry in scenes if entry[0] == "desk"]
    output = root / "results" / "cuda_tf_online_guard_off_desk_start1000_20261014"
output.mkdir(parents=True, exist_ok=True)
manifest = {"binary": str(root / "bin/tum_rgbd_cuda_tf_online"),
            "binary_sha256": hashlib.sha256((root / "bin/tum_rgbd_cuda_tf_online").read_bytes()).hexdigest(),
            "base_config_sha256": hashlib.sha256(base.encode()).hexdigest(),
            "tail_training": False, "tf_start": tf_start, "gradient_constraint": False, "runs": []}
saved_manifest = diagnostics / (prefix + "online_runs.json")
if saved_manifest.exists():
    previous = json.loads(saved_manifest.read_text())
    for key in ("binary_sha256", "base_config_sha256", "tail_training", "tf_start", "gradient_constraint"):
        assert previous[key] == manifest[key], ("Resume protocol changed", key)
    manifest = previous
(diagnostics / (prefix + "online.exit")).unlink(missing_ok=True)

def save():
    saved_manifest.write_text(json.dumps(manifest, indent=2))

try:
    for scene, dataset_name, orb_name in scenes:
        dataset = datasets / dataset_name
        association = (dataset / "tum_freiburg1_room.txt" if scene == "room" else
                       root / "cfg/ORB_SLAM3/RGB-D/TUM/associations" / (orb_name + ".txt"))
        rows = [line.split() for line in association.read_text().splitlines()
                if line.strip() and not line.startswith("#")]
        assert rows, association
        assert all(len(row) == 4 and (dataset / row[1]).is_file() and
                   (dataset / row[3]).is_file() for row in rows), association
        for backend in ("legacy", "cuda"):
            directory = output / scene / backend
            completed = [run for run in manifest["runs"]
                         if run["scene"] == scene and run["backend"] == backend]
            if completed:
                assert len(completed) == 1 and completed[0]["status"] == "complete", completed
                assert completed[0]["association"] == str(association)
                assert (directory / "selector_config.yaml").read_text() == base + f'\nTransformField.backend: "{backend}"\n'
                assert list(directory.glob("*_shutdown/ply/transform_state.json")), directory
                assert (directory / "CameraTrajectory_TUM.txt").is_file(), directory
                continue
            assert not directory.exists(), f"Refuse to overwrite completed/partial run: {directory}"
            directory.mkdir(parents=True)
            config = directory / "selector_config.yaml"
            config.write_text(base + f'\nTransformField.backend: "{backend}"\n')
            command = [str(root / "bin/tum_rgbd_cuda_tf_online"),
                       str(root / "third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt"),
                       str(root / "cfg/ORB_SLAM3/RGB-D/TUM" / (orb_name + ".yaml")),
                       str(root / "cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml"),
                       str(dataset), str(association), str(directory), "no_viewer",
                       "--selector-config", str(config)]
            run = dict(scene=scene, backend=backend, dataset=str(dataset),
                       association=str(association), frames=len(rows), command=command,
                       output=str(directory), status="running", start=time.time())
            manifest["runs"].append(run)
            save()
            with (directory / "mapping.log").open("w") as log:
                result = subprocess.run(command, cwd=root, env=environment, stdout=log,
                                        stderr=subprocess.STDOUT, timeout=1800)
            run.update(exit_code=result.returncode, wall_seconds=time.time()-run["start"],
                       status="complete" if result.returncode == 0 else "failed")
            save()
            assert result.returncode == 0, f"Mapping failed: {scene}/{backend}"
            assert list(directory.glob("*_shutdown/ply/transform_state.json")), directory
            assert (directory / "CameraTrajectory_TUM.txt").is_file(), directory
            print(scene, backend, "passed", run["wall_seconds"], flush=True)
    (diagnostics / (prefix + "online.exit")).write_text("0\n")
except BaseException:
    (diagnostics / (prefix + "online.exit")).write_text("1\n")
    raise
