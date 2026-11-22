"""30维/child-local 的必要正确性验证与锁定实验。"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
r=Path(__file__).resolve().parents[3];repo=r/"source";v=r/"validation"
env=dict(os.environ,OMP_NUM_THREADS="2",MPLBACKEND="Agg",PATH="/home/prmi/anaconda3/envs/zwj/bin:/usr/local/cuda-11.8/bin:"+os.environ["PATH"],LD_LIBRARY_PATH=str(repo/"lib")+":"+os.environ.get("LD_LIBRARY_PATH",""))
def state(stage,**extra):(v/"pipeline_status.json").write_text(json.dumps(dict(stage=stage,**extra),indent=2))
state("WAITING_FOR_BUILD")
while not (v/"build.exit").exists():time.sleep(5)
assert (v/"build.exit").read_text()=="0","build failed"
m=json.loads((v/"source_manifest.json").read_text())
assert all(hashlib.sha256((repo/p).read_bytes()).hexdigest()==h for p,h in m["files"].items())
gates={}
def run(name,command):
    state("GATES",test=name)
    with (v/(name+".log")).open("w") as f:p=subprocess.run(list(map(str,command)),env=env,cwd=v,stdout=f,stderr=subprocess.STDOUT)
    gates[name]=dict(passed=p.returncode==0,exit_code=p.returncode)
    (v/"correctness.json").write_text(json.dumps(gates,indent=2))
    if p.returncode:state("FAILED",test=name);raise SystemExit(p.returncode)
for device in ("cpu","cuda"):
    run("xyz_local_"+device,[repo/"bin/anchor_octree_test",device,"global_xyz"])
    run("xyz_local_python_"+device,["/home/prmi/anaconda3/envs/zwj/bin/python",repo/"tests/anchor_selector/xyz_local_python_parity.py",v,device])
    run("xyz_fixed_"+device,[repo/"bin/anchor_attribute_test",device,"global_xyz"])
    run("legacy27_"+device,[repo/"bin/anchor_attribute_test",device])
run("xyz_local_mapping",[repo/"bin/anchor_xyz_local_mapping_test","--parity"])
run("xyz_local_change_distill",[repo/"bin/anchor_change_distill_test","attributes","adaptive","global_xyz"])
state("TRAINING_AND_EVALUATION")
with (v/"audit.log").open("w") as f:
    p=subprocess.run(["/home/prmi/anaconda3/envs/zwj/bin/python",str(repo/"tests/anchor_selector/run_octree_audit.py"),str(repo),"/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz",str(repo/"tests/anchor_selector/xyz_local_protocol.json")],env=env,cwd=repo,stdout=f,stderr=subprocess.STDOUT)
state("COMPLETED" if p.returncode==0 else "FAILED",exit_code=p.returncode)
