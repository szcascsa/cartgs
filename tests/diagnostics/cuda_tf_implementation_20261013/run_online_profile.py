"""独立在线TF阶段采样；诊断时间不替代无插桩Mapping验收时间。"""
import json
import os
import subprocess
import time
from pathlib import Path

diagnostics = Path(__file__).resolve().parent
root = diagnostics.parents[2]
try:
    while not (diagnostics / "guard_off_remaining_acceptance.exit").exists():
        time.sleep(2)
    assert (diagnostics / "guard_off_remaining_acceptance.exit").read_text().strip() == "0"
    assert (diagnostics / "online_profile_build.exit").read_text().strip() == "0"
    assert (diagnostics / "online_profiler_parity.exit").read_text().strip() == "0"
    original = json.loads((diagnostics / "guard_off_online_runs.json").read_text())
    manifest = {"protocol": "separate diagnostic graph brackets; sampled current-stream CUDA events; no explicit synchronize", "runs": []}
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = ":".join([
        str(root / "third_party/libtorch-cu118/libtorch/lib"), "/usr/local/cuda-11.8/lib64",
        str(root / "lib"), str(root / "third_party/ORB-SLAM3/lib"), environment.get("LD_LIBRARY_PATH", "")])
    for source in original["runs"]:
        if source["scene"] != "long":
            continue
        output = root / "results/cuda_tf_stage_profile_20261014" / source["backend"]
        assert not output.exists(), output
        output.mkdir(parents=True)
        command = source["command"].copy()
        command[0] = str(root / "bin/tum_rgbd_cuda_tf_profile")
        command[6] = str(output)
        run = {"backend": source["backend"], "scene": "long", "command": command,
               "output": str(output), "status": "running", "start": time.time()}
        manifest["runs"].append(run)
        (diagnostics / "online_tf_profile_runs.json").write_text(json.dumps(manifest, indent=2))
        with (output / "mapping.log").open("w") as log:
            result = subprocess.run(command, cwd=root, env=environment, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=1800)
        assert result.returncode == 0, (source["backend"], result.returncode)
        text = (output / "mapping.log").read_text()
        run["stage_reports"] = [line for line in text.splitlines() if line.startswith("[TF DIAGNOSTIC]")]
        assert len(run["stage_reports"]) == 6, run
        assert not any("calls=0 " in line for line in run["stage_reports"][:5]), run
        run.update(status="complete", wall_seconds=time.time()-run["start"])
        (diagnostics / "online_tf_profile_runs.json").write_text(json.dumps(manifest, indent=2))
        print(source["backend"], run["stage_reports"], flush=True)
    (diagnostics / "online_tf_profile.exit").write_text("0\n")
except BaseException:
    (diagnostics / "online_tf_profile.exit").write_text("1\n")
    raise
