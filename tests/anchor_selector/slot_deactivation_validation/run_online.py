"""隔离构建/功能门控/一次在线训练；保留原训练源码与模型。"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

root=Path('/home/prmi/zwj/anchor-slot-deactivation-20261021')
baseline=Path('/home/prmi/zwj/octree-anchor-pruning-20261021')
repo=root/'source';out=root/'validation';out.mkdir(exist_ok=True)
def status(stage,**kw): (out/'status.json').write_text(json.dumps(dict(stage=stage,**kw),indent=2))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest=json.loads((root/'source_manifest.json').read_text())
if not repo.exists():
    shutil.copytree(baseline/'source',repo,symlinks=True)
    for p,h in manifest.items():
        target=repo/p;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(root/'overlay'/p,target)
        assert sha(target)==h
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
with (out/'build.log').open('w') as log:
    subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    subprocess.run([cmake,'--build',str(root/'build'),'--target','tum_rgbd','slot_deactivation_test',
        'slot_deactivation_mapping_test','anchor_opacity_pruning_test','explicit_anchor_test',
        'explicit_anchor_mapping_test','anchor_attribute_mapping_test','anchor_change_distill_test',
        'anchor_checkpoint_probe','-j2'],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
gates={}
def test(name,cmd,cwd):
    status('CORRECTNESS',test=name)
    with (out/(name+'.log')).open('w') as f:
        p=subprocess.run(list(map(str,cmd)),env=env,cwd=cwd,stdout=f,stderr=subprocess.STDOUT)
    gates[name]=dict(passed=p.returncode==0,code=p.returncode)
    (out/'correctness.json').write_text(json.dumps(gates,indent=2))
    assert p.returncode==0,name
for device in ['cpu','cuda']:
    folder=out/device;folder.mkdir(exist_ok=True)
    test('slot_'+device,[repo/'bin/slot_deactivation_test',device],folder)
    test('slot_python_'+device,[sys.executable,repo/'tests/anchor_selector/slot_deactivation_validation/python_parity.py',folder,device],folder)
    test('pruning_'+device,[repo/'bin/anchor_opacity_pruning_test',device],folder)
    test('legacy_explicit_'+device,[repo/'bin/explicit_anchor_test',device],folder)
    test('legacy_python_'+device,[sys.executable,repo/'tests/anchor_selector/explicit_python_parity.py',folder,device],folder)
test('slot_mapping',[repo/'bin/slot_deactivation_mapping_test','--parity'],out)
test('legacy_mapping',[repo/'bin/explicit_anchor_mapping_test','--parity'],out)
test('legacy_attributes',[repo/'bin/anchor_attribute_mapping_test','--parity'],out)
test('change_distill',[repo/'bin/anchor_change_distill_test','attributes'],out)
# Backward compatibility: native loader accepts a real checkpoint without slot buffers.
checkpoint=baseline/'smoke_seed2026/10000_shutdown/ply'
cloud=checkpoint/'point_cloud/iteration_10000'
test('old_checkpoint',[repo/'bin/anchor_checkpoint_probe',checkpoint/'selector.pt',cloud/'point_cloud.ply',
    cloud/'selector_metadata.ply',out/'old_probe.pt','explicit_blocks'],out)
result=root/'enabled_seed2026';result.mkdir(exist_ok=False)
config=result/'input_selector.yaml'
base=checkpoint/'selector_config.yaml'
config.write_text(base.read_text()+'\nSelector.slot_deactivation: 1\n')
protocol=dict(scene='fr1_xyz',seed=2026,mapping_steps=10000,baseline=str(baseline/'smoke_seed2026'),
    config_base_sha256=sha(base),config_only_change='Selector.slot_deactivation: 1',
    ratios=[.01,.05,.1,.15],pruning='unchanged Octree-GS anchor opacity',
    budget='ratio * total stored Gaussian count; masked probability ratio loss',
    sampler='unchanged cached Bernoulli; no count calibration',
    training_runs=1,same_k='offline only; common absolute K locked before rendering',
    online_trajectory='same seed/config but asynchronous SLAM may change poses, KF sampling, topology',
    no_tuning=True,no_early_stop=True,source_sha256=manifest)
(out/'training_protocol.json').write_text(json.dumps(protocol,indent=2))
status('TRAINING',seed=2026)
cmd=[repo/'bin/tum_rgbd',repo/'third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt',
    repo/'cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg1_xyz.yaml',repo/'cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml',
    '/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz',repo/'cfg/ORB_SLAM3/RGB-D/TUM/associations/fr1_xyz.txt',
    result,'no_viewer','--selector-config',config]
start=time.monotonic()
with (result/'training.log').open('w') as f:
    subprocess.run(list(map(str,cmd)),cwd=repo,env=dict(env,CARTGS_AUDIT_SEED='2026'),stdout=f,stderr=subprocess.STDOUT,check=True)
assert '10000 iterations completed.' in (result/'training.log').read_text()
(out/'training_runtime.json').write_text(json.dumps(dict(wall_seconds=time.monotonic()-start,
    note='single run; not an uncontaminated speed benchmark'),indent=2))
status('TRAINED')
print('ONLINE_TRAINING_COMPLETE',flush=True)
