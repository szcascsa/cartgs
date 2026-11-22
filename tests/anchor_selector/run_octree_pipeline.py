"""按 correctness -> 训练 -> 质量顺序执行；gate 失败即停止。"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
r=Path(__file__).resolve().parents[3]
repo=r/"source"
env=dict(os.environ,OMP_NUM_THREADS="2",MPLBACKEND="Agg",PATH="/home/prmi/anaconda3/envs/zwj/bin:/usr/local/cuda-11.8/bin:"+os.environ["PATH"],LD_LIBRARY_PATH=str(repo/"lib")+":"+os.environ.get("LD_LIBRARY_PATH",""))
status=r/"validation/pipeline_status.json"
def state(stage,**extra):status.write_text(json.dumps(dict(stage=stage,**extra),indent=2))
state("WAITING_FOR_BUILD")
while not (r/"validation/build.exit").exists():time.sleep(5)
assert (r/"validation/build.exit").read_text()=="0","build failed"
manifest=json.loads((r/"validation/source_manifest.json").read_text())
assert all(hashlib.sha256((repo/p).read_bytes()).hexdigest()==h for p,h in manifest["files"].items())
gates={}
for device in ("cpu","cuda"):
    for name,cmd in [
        ("octree_"+device,[repo/"bin/anchor_octree_test",device]),
        ("octree_python_"+device,["/home/prmi/anaconda3/envs/zwj/bin/python",repo/"tests/anchor_selector/octree_python_parity.py",r/"validation",device]),
        ("fixed_"+device,[repo/"bin/anchor_attribute_test",device])]:
        state("GATES",test=name)
        with (r/"validation"/(name+".log")).open("w") as log:
            run=subprocess.run(list(map(str,cmd)),env=env,cwd=r/"validation",stdout=log,stderr=subprocess.STDOUT)
        gates[name]=dict(passed=run.returncode==0,exit_code=run.returncode)
        (r/"validation/correctness.json").write_text(json.dumps(gates,indent=2))
        if run.returncode:state("FAILED",test=name);raise SystemExit(run.returncode)
for name,cmd in [
    ("adaptive_mapping",[repo/"bin/anchor_octree_mapping_test","--parity"]),
    ("fixed_mapping",[repo/"bin/anchor_attribute_mapping_test","--parity"]),
    ("adaptive_change_distill",[repo/"bin/anchor_change_distill_test","attributes","adaptive"])]:
    state("GATES",test=name)
    with (r/"validation"/(name+".log")).open("w") as log:
        run=subprocess.run(list(map(str,cmd)),env=env,cwd=r/"validation",stdout=log,stderr=subprocess.STDOUT)
    gates[name]=dict(passed=run.returncode==0,exit_code=run.returncode)
    (r/"validation/correctness.json").write_text(json.dumps(gates,indent=2))
    if run.returncode:state("FAILED",test=name);raise SystemExit(run.returncode)
state("TRAINING_AND_EVALUATION")
with (r/"validation/audit.log").open("w") as log:
    run=subprocess.run(["/home/prmi/anaconda3/envs/zwj/bin/python",str(repo/"tests/anchor_selector/run_octree_audit.py"),str(repo),"/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz"],env=env,cwd=repo,stdout=log,stderr=subprocess.STDOUT)
state("COMPLETED" if run.returncode==0 else "FAILED",exit_code=run.returncode)
