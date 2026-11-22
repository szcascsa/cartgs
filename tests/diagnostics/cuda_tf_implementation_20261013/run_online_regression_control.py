"""固定输入训练链与同配置legacy复跑；排查在线质量差异，不调整生产节奏。"""
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np

d = Path(__file__).resolve().parent
root = d.parents[2]
python = "/home/prmi/anaconda3/envs/zwj/bin/python"
environment = os.environ.copy()
environment["LD_LIBRARY_PATH"] = ":".join([
    str(root / "third_party/libtorch-cu118/libtorch/lib"), "/usr/local/cuda-11.8/lib64",
    str(root / "lib"), str(root / "third_party/ORB-SLAM3/lib"), environment.get("LD_LIBRARY_PATH", "")])

try:
    while not (d / "online_tf_profile.exit").exists():
        time.sleep(2)
    assert (d / "online_tf_profile.exit").read_text().strip() == "0"
    assert (d / "training_chain_build.exit").read_text().strip() == "0"
    with (d / "training_chain_parity.log").open("w") as log:
        result = subprocess.run([str(d / "build/training_chain_parity"),
            str(root / "results/protection_recheck_20261012/fr3_long_office_household/3400_shutdown/ply"),
            str(root / "tests/diagnostics/original_tf_cuda_audit_20261013/input.bin"),
            str(d / "training_chain_parity.csv")], stdout=log, stderr=subprocess.STDOUT,
            cwd=root, env=environment, timeout=300)
    (d / "training_chain_parity.exit").write_text(str(result.returncode)+"\n")
    source = next(r for r in json.loads((d / "guard_off_online_runs.json").read_text())["runs"]
                  if r["scene"] == "long" and r["backend"] == "legacy")
    output = root / "results/cuda_tf_online_regression_control_20261014/long/legacy_repeat"
    assert not output.exists(), output
    output.mkdir(parents=True)
    command = source["command"].copy()
    command[6] = str(output)
    config = output / "selector_config.yaml"
    config.write_bytes(Path(source["output"]).joinpath("selector_config.yaml").read_bytes())
    command[-1] = str(config)
    manifest = dict(protocol="same legacy binary/config/dataset; unrestricted online cycles; no tail",
                    training_chain_exit=result.returncode, command=command, output=str(output),
                    status="running", start=time.time(), evaluations=[])
    def save():
        (d / "online_regression_control.json").write_text(json.dumps(manifest, indent=2))
    save()
    with (output / "mapping.log").open("w") as log:
        result = subprocess.run(command, cwd=root, env=environment, stdout=log,
                                stderr=subprocess.STDOUT, timeout=1800)
    assert result.returncode == 0, result.returncode
    manifest["wall_seconds"] = time.time()-manifest["start"]
    checkpoints = list(output.glob("*_shutdown"))
    assert len(checkpoints) == 1, checkpoints
    for ratio in (None, .01, .05, .10, .15):
        label = "full" if ratio is None else f"lod_{int(ratio*100):02d}"
        target = output / "evaluation" / label
        target.mkdir(parents=True)
        command = [python, str(d / "evaluate_headless_batched.py"), str(output), source["dataset"],
                   "--output_path", str(target), "--shutdown-model", checkpoints[0].name]
        if ratio is not None:
            command += ["--selector", str(checkpoints[0] / "ply/selector.pt"),
                        "--selector-ratio", str(ratio), "--selector-config", str(config)]
        with (target / "evaluation.log").open("w") as log:
            result = subprocess.run(command, cwd=root, env=environment, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=1800)
        assert result.returncode == 0, (label, result.returncode)
        metrics = {}; counts = []
        for key in ("psnr", "ssim", "lpips"):
            values = np.atleast_1d(np.loadtxt(target / (key+".txt")))
            assert np.isfinite(values).all() and values.size
            metrics[key] = float(values.mean()); counts.append(values.size)
        assert len(set(counts)) == 1
        manifest["evaluations"].append(dict(label=label, frames=counts[0], metrics=metrics))
        save(); print(label, metrics, flush=True)
    manifest["status"] = "complete"; save()
    (d / "online_regression_control.exit").write_text("0\n")
except BaseException:
    (d / "online_regression_control.exit").write_text("1\n")
    raise
