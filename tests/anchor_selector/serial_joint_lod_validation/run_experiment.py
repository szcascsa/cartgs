"""串行 Full Master + joint LoD：功能门控后一次固定预算在线训练。"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

root=Path('/home/prmi/zwj/serial-joint-lod-20261021')
previous=Path('/home/prmi/zwj/anchor-slot-deactivation-20261021')
repo=root/'source';out=root/'validation';out.mkdir(exist_ok=True)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest=json.loads((root/'source_manifest.json').read_text())
def status(stage,**kw):(out/'status.json').write_text(json.dumps(dict(stage=stage,**kw),indent=2))
if not repo.exists():
    shutil.copytree(Path('/home/prmi/zwj/serial-full-lod-20261021/source'),repo,symlinks=True)
    for p,h in manifest.items():
        target=repo/p;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(root/'overlay'/p,target)
assert all(sha(repo/p)==h for p,h in manifest.items())
env=dict(os.environ,PATH='/usr/local/cuda-11.8/bin:/home/prmi/anaconda3/envs/zwj/bin:'+os.environ['PATH'],
    OMP_NUM_THREADS='2',QT_QPA_PLATFORM='offscreen',MPLBACKEND='Agg')
cmake='/home/prmi/.local/bin/cmake'
cmd=[cmake,'-S',str(repo),'-B',str(root/'build'),'-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CUDA_ARCHITECTURES=86',
    '-DCMAKE_CXX_COMPILER=/usr/bin/g++-9','-DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc',
    '-DTorch_DIR=/home/prmi/zwj/mycartgs/third_party/libtorch-cu118/libtorch/share/cmake/Torch',
    '-DOpenCV_DIR=/home/prmi/zwj/mycartgs/third_party/install/opencv-cuda118-sm86-gcc9/lib/cmake/opencv4',
    '-Djsoncpp_DIR=/home/prmi/zwj/mycartgs/third_party/install/jsoncpp-1.9.5/lib/cmake/jsoncpp',
    '-DEigen3_DIR=/home/prmi/zwj/mycartgs/third_party/eigen-3.3.7/build-cmake',
    '-DCARTGS_BUILD_TESTS=ON','-DCARTGS_EXPERIMENTAL_SELECTOR_CUDA=ON']
status('BUILDING')
with (out/'build.log').open('w') as f:
    subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    subprocess.run([cmake,'--build',str(root/'build'),'--target','tum_rgbd','post_topology_full_test',
        'full_recovery_mapping_test','serial_joint_lod_mapping_test','serial_full_lod_mapping_test','slot_deactivation_mapping_test','anchor_opacity_pruning_test',
        'anchor_checkpoint_probe','anchor_change_distill_test','-j4'],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
gates={}
def test(name,cmd,cwd):
    status('CORRECTNESS',test=name)
    with (out/(name+'.log')).open('w') as f:
        r=subprocess.run(list(map(str,cmd)),cwd=cwd,env=env,stdout=f,stderr=subprocess.STDOUT)
    gates[name]=dict(passed=r.returncode==0,code=r.returncode)
    (out/'correctness.json').write_text(json.dumps(gates,indent=2));assert r.returncode==0,name
test('schedule',[repo/'bin/post_topology_full_test'],out)
test('joint_mapping',[repo/'bin/serial_joint_lod_mapping_test','--parity'],out)

test('serial_mapping',[repo/'bin/serial_full_lod_mapping_test','--parity'],out)
test('full100_regression',[repo/'bin/full_recovery_mapping_test','--parity'],out)
test('disabled_regression',[repo/'bin/slot_deactivation_mapping_test','--parity'],out)
for device in ['cpu','cuda']:
    folder=out/device;folder.mkdir(exist_ok=True)
    test('pruning_'+device,[repo/'bin/anchor_opacity_pruning_test',device],folder)
test('change_distill',[repo/'bin/anchor_change_distill_test','attributes'],out)
run=root/'joint_seed2026';run.mkdir(exist_ok=False)
base=previous/'enabled_seed2026/input_selector.yaml'
config=run/'input_selector.yaml';config.write_text(base.read_text()+'\nTraining.post_topology_full_steps: 0\nTraining.serial_full_lod: 1\nTraining.serial_lod_detach_master: 0\n')
protocol=dict(scene='fr1_xyz',seed=2026,total_mapping_steps=10000,training_runs=1,
    baseline=str(previous/'enabled_seed2026'),serial_detached='/home/prmi/zwj/serial-full-lod-20261021/serial_seed2026',additional_reference='/home/prmi/zwj/octree-anchor-pruning-20261021/smoke_seed2026',
    only_config_change='versus serial detached: Training.serial_lod_detach_master: 0; serial schedule retained; Full100=0',slot_deactivation=True,
    event='every Mapping: Full render/backward/Master commit, existing Full topology, re-read updated live Master, one LoD render/backward/Gaussian+auxiliary commit',
    normal_updates='Full: Master only; LoD: Master Gaussian + Selector/Anchor feature/TF; two Master Adam commits per Mapping',
    clocks='one Mapping/Gaussian LR tick per pair; four LoD ratios rotate across online/tail boundary; 10000 Mapping budget',
    resolution='Full preserves GausPyramid; online LoD uses existing 0.5 scale, tail LoD native; no resolution tuning',
    full100=False,
    no_changes=['loss','network','pruning threshold','growth rule','Cached Bernoulli','TF/CD algorithm','slot eligibility'],
    evaluation='final checkpoint; common pre-quality locked views and absolute K; no tuning or best checkpoint selection',
    source_sha256=manifest,base_config_sha256=sha(base))
(out/'preregistration.json').write_text(json.dumps(protocol,indent=2))
status('TRAINING')
cmd=[repo/'bin/tum_rgbd',repo/'third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt',
    repo/'cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg1_xyz.yaml',repo/'cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml',
    '/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz',repo/'cfg/ORB_SLAM3/RGB-D/TUM/associations/fr1_xyz.txt',
    run,'no_viewer','--selector-config',config]
start=time.monotonic()
with (run/'training.log').open('w') as f:
    subprocess.run(list(map(str,cmd)),cwd=repo,env=dict(env,CARTGS_AUDIT_SEED='2026'),stdout=f,stderr=subprocess.STDOUT,check=True)
assert '10000 iterations completed.' in (run/'training.log').read_text()
(out/'training_runtime.json').write_text(json.dumps(dict(wall_seconds=time.monotonic()-start,strict_benchmark=False),indent=2))
status('TRAINED');print('TRAINED_SERIAL_JOINT',flush=True)
