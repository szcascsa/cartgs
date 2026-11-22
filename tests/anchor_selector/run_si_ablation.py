"""执行预注册的 Anchor +/- Streaming SI 两 seed 对照，不按质量调整参数。"""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

repo, dataset = map(Path, sys.argv[1:3])
root = repo.parent
protocol = json.loads((repo/"tests/anchor_selector/si_ablation_protocol.json").read_text())
base = (repo/"cfg/selector/fast.yaml").read_text()
jobs = []
for seed in protocol["seeds"]:
    # Alternate ordering between paired seeds; no concurrent GPU workloads.
    for arm in (("no_si", "si") if seed == protocol["seeds"][0] else ("si", "no_si")):
        name = f"{arm}_seed{seed}"
        output = root/"runs"/name
        output.mkdir(parents=True, exist_ok=False)
        settings = {"Training.force_tail_optimization_enabled": "0",
                    "Selector.max_mapping_iterations": str(protocol["training"]["normal_mapping_limit"]),
                    "GlobalGI.performance_logging": "1", **protocol["arms"][arm]}
        config = base
        for key, value in settings.items():
            value = '"'+value+'"' if isinstance(value, str) and not value.isnumeric() else str(value)
            config, count = re.subn(r"^"+re.escape(key)+r":.*$", key+": "+value, config, flags=re.M)
            assert count == 1, key
        config_path = output/"input_selector.yaml"
        config_path.write_text(config)
        command = [repo/"bin/tum_rgbd", repo/"third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt",
                   repo/"cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg1_xyz.yaml",
                   repo/"cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml", dataset,
                   repo/"cfg/ORB_SLAM3/RGB-D/TUM/associations/fr1_xyz.txt", output,
                   "no_viewer", "--selector-config", config_path]
        start = time.monotonic()
        with (output/"training.log").open("w") as log:
            run = subprocess.run(list(map(str, command)), cwd=repo, stdout=log, stderr=subprocess.STDOUT,
                env=dict(os.environ, OMP_NUM_THREADS="2", QT_QPA_PLATFORM="offscreen", CARTGS_AUDIT_SEED=str(seed)))
        job = dict(arm=arm, seed=seed, output=str(output), exit_code=run.returncode,
                   wall_seconds=time.monotonic()-start, command=list(map(str, command)))
        jobs.append(job)
        (root/"validation/training_jobs.json").write_text(json.dumps(jobs, indent=2))
        assert run.returncode == 0, job
        log = (output/"training.log").read_text()
        assert "[Anchor Selector]" in log
        assert "SI/GI="+protocol["arms"][arm]["GlobalGI.mode"] in log
        assert "10000 iterations completed." in log
        checkpoint = output/"10000_shutdown/ply"
        saved = (checkpoint/"selector_config.yaml").read_text()
        assert f'GlobalGI.mode: "{protocol["arms"][arm]["GlobalGI.mode"]}"' in saved
        assert (checkpoint/"selector.pt").is_file()
        print(name, f'{job["wall_seconds"]:.2f}s', "completed", flush=True)
(root/"validation/training.exit").write_text("0")
