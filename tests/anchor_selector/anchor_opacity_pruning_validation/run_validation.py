"""隔离构建与规则验证；不覆盖原训练或 production，不按质量调整 pruning。"""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import torch

root=Path('/home/prmi/zwj/octree-anchor-pruning-20261021')
repo=root/'source';out=root/'validation'
env=dict(os.environ,PATH='/usr/local/cuda-11.8/bin:/home/prmi/anaconda3/envs/zwj/bin:'+os.environ['PATH'],
         OMP_NUM_THREADS='2',QT_QPA_PLATFORM='offscreen',MPLBACKEND='Agg')
def state(stage,**kw): (out/'status.json').write_text(json.dumps(dict(stage=stage,**kw),indent=2))
cmake='/home/prmi/.local/bin/cmake'
state('BUILDING')
cmd=[cmake,'-S',str(repo),'-B',str(root/'build'),'-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CUDA_ARCHITECTURES=86',
     '-DCMAKE_CXX_COMPILER=/usr/bin/g++-9','-DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc',
     '-DTorch_DIR=/home/prmi/zwj/mycartgs/third_party/libtorch-cu118/libtorch/share/cmake/Torch',
     '-DOpenCV_DIR=/home/prmi/zwj/mycartgs/third_party/install/opencv-cuda118-sm86-gcc9/lib/cmake/opencv4',
     '-Djsoncpp_DIR=/home/prmi/zwj/mycartgs/third_party/install/jsoncpp-1.9.5/lib/cmake/jsoncpp',
     '-DEigen3_DIR=/home/prmi/zwj/mycartgs/third_party/eigen-3.3.7/build-cmake',
     '-DCARTGS_BUILD_TESTS=ON','-DCARTGS_EXPERIMENTAL_SELECTOR_CUDA=ON']
with (out/'build.log').open('w') as log:
    subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    subprocess.run([cmake,'--build',str(root/'build'),'--target','tum_rgbd','anchor_opacity_pruning_test',
        'explicit_anchor_test','explicit_anchor_mapping_test','anchor_attribute_mapping_test','anchor_change_distill_test','-j2'],
        env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
gates={}
def run(name,command,cwd):
    state('CORRECTNESS',test=name)
    with (out/(name+'.log')).open('w') as log:
        p=subprocess.run(list(map(str,command)),cwd=cwd,env=env,stdout=log,stderr=subprocess.STDOUT)
    gates[name]=dict(passed=p.returncode==0,exit_code=p.returncode)
    (out/'correctness.json').write_text(json.dumps(gates,indent=2))
    assert p.returncode==0,name
for device in ['cpu','cuda']:
    directory=out/device;directory.mkdir(exist_ok=True)
    run('pruning_'+device,[repo/'bin/anchor_opacity_pruning_test',device],directory)
    run('explicit_'+device,[repo/'bin/explicit_anchor_test',device],directory)
    run('python_parity_'+device,[sys.executable,repo/'tests/anchor_selector/explicit_python_parity.py',directory,device],directory)
    fixture=torch.jit.load(str(directory/'anchor_pruning_fixture.pt'),map_location='cpu')
    f=dict(fixture.named_buffers());expected=((f['anchor_visits']>4)&(f['opacity_accum']<.005*f['anchor_visits'])).repeat_interleave(10)
    assert torch.equal(f['prune_mask'],expected)
    fixture=torch.jit.load(str(directory/'anchor_history_before_check.pt'),map_location='cpu')
    f=dict(fixture.named_buffers());visits=f['anchor_visits'].double().unsqueeze(1)
    assert torch.allclose(f['slot_opacity_accum'].sum(1),f['opacity_accum'],atol=1e-12,rtol=1e-12)
    assert torch.allclose(f['slot_mean_opacity'],f['slot_opacity_accum']/visits)
    assert (f['slot_below_cutoff_fraction'][5,:8]==1).all() and (f['slot_below_cutoff_fraction'][5,8:]==0).all()
    assert torch.allclose(f['slot_below_cutoff_fraction'][4],torch.full((10,),.6,dtype=torch.float64))
    assert (f['slot_below_cutoff_fraction'][2]==1).all() and f['anchor_visits'][2]==2
    fixture=torch.jit.load(str(directory/'anchor_history_newborn.pt'),map_location='cpu')
    f=dict(fixture.named_buffers());new=f['anchor_visits']==0
    assert torch.isnan(f['slot_mean_opacity'][new]).all() and torch.isnan(f['slot_below_cutoff_fraction'][new]).all()
    gates['slot_history_'+device]=dict(passed=True,official_rule_unchanged=True,never_observed_is_nan=True,
        effective_anchor_with_8_always_low_slots_retained=True,temporary_low_fraction=.6)
    (out/'correctness.json').write_text(json.dumps(gates,indent=2))
run('explicit_mapping',[repo/'bin/explicit_anchor_mapping_test','--parity'],out)
run('legacy_mapping',[repo/'bin/anchor_attribute_mapping_test','--parity'],out)
run('legacy_change_distill',[repo/'bin/anchor_change_distill_test','attributes'],out)
state('ONLINE_SMOKE')
result=root/'smoke_seed2026';result.mkdir()
config=Path('/home/prmi/zwj/octree-explicit-20261021/explicit_seed2026/input_selector.yaml')
command=[repo/'bin/tum_rgbd',repo/'third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt',
    repo/'cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg1_xyz.yaml',repo/'cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml',
    '/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz',repo/'cfg/ORB_SLAM3/RGB-D/TUM/associations/fr1_xyz.txt',
    result,'no_viewer','--selector-config',config]
start=time.monotonic()
with (result/'training.log').open('w') as log:
    subprocess.run(list(map(str,command)),cwd=repo,env=dict(env,CARTGS_AUDIT_SEED='2026'),stdout=log,stderr=subprocess.STDOUT,check=True)
assert '10000 iterations completed.' in (result/'training.log').read_text()
(out/'smoke_runtime.json').write_text(json.dumps(dict(seed=2026,wall_seconds=time.monotonic()-start,
    mapping_steps=10000,purpose='functional smoke; not a matched quality comparison'),indent=2))
state('COMPLETED')
print('PRUNING_VALIDATION_COMPLETE',flush=True)
