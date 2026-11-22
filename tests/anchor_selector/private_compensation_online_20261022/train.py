"""真实fr1_xyz SLAM：先R3双seed，再R2双seed，原joint与自然停止。"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root=Path('/home/prmi/zwj/private-compensation-online-20261022');repo=root/'source';out=root/'validation'
assert json.loads((out/'correctness.json').read_text())['passed']
commit=json.loads((root/'commit.json').read_text())
manifest=json.loads((root/'source_manifest.json').read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert all(sha(repo/p)==h for p,h in manifest.items())
base=(repo/'cfg/selector/selector_config.yaml').read_text()
for line in ['Selector.compensation_method: "r3_independent14"','Selector.max_mapping_iterations: 0','Training.serial_full_lod: 0',
             'TransformField.enable: 0','ChangeDistill.enable: 0','Selector.compensation_start_iter: 3000',
             'Selector.anchor_only_pruning: 1','Selector.anchor_ratio_features: 0','Selector.anchor_feature_dim: 16']:
    assert line in base,line
frames=json.loads(Path('/home/prmi/zwj/budget-shared-audit-20261022/validation/evaluation_plan.json').read_text())['frames']
protocol=dict(scene='fr1_xyz',training='from scratch real SLAM, online joint LoD, Full Master only',
    methods=['r3_independent14','r2_shared14'],seeds=[2026,2027],
    runs=[['r3_independent14',2026],['r3_independent14',2027],['r2_shared14',2026],['r2_shared14',2027]],
    training_ratios=[.01,.05,.1,.15],evaluation_ratios=sorted([.01,.025,.075,.125]+[i/100 for i in range(5,101,5)]),
    evaluation_frames=frames,initialization='C output zero and frozen, private residual zero;Selector/Encoder/Anchor/Master train normally',
    topology='Variable Anchor1..10;clone/split inherit residual with fresh UID/new zero moments;observed append zero;prune/reorder aligned',
    geometry='XYZ residual transported by scale/rotation;raw quaternion conjugated;affected Adam rows reset;no new ChangeDistill',
    interpolation='R3 linear raw-ratio interpolation of14D coefficient vectors, endpoint clamp outside1..15%;R2 same table',
    optimizer='existing joint LibTorch Adam group eps1e-15/lr0.001;R3 inactive tables gradNone;R2 receives allratio updates',
    budget='max_mapping_iterations0, original stopping;no forced10000;report actual optimization clocks and schedules',
    metrics=['PSNR','SSIM','LPIPS','photo','actualK','visible','effective opacity proxy','Full','ATE','runtime','memory'],
    evaluation='shared64 RGB, own final poses,nonBank excludes union of all4 Bank paths;matched absoluteK plusCached',
    limit='different online topology/poses/update counts;not frozen map;>15% is out-of-range Selector inference and endpoint compensation',
    source_commit=commit,source_sha256=manifest,no_eval_tuning=True,no_push=True)
if (out/'preregistration.json').exists():assert json.loads((out/'preregistration.json').read_text())==protocol
else:(out/'preregistration.json').write_text(json.dumps(protocol,indent=2))
runtime=json.loads((out/'training_runtime.json').read_text()) if (out/'training_runtime.json').exists() else {}
for method,seed in protocol['runs']:
    name=f'{method}_seed{seed}'
    if name in runtime:continue
    run=root/name;run.mkdir(exist_ok=False)
    config=base.replace('Selector.compensation_method: "r3_independent14"',f'Selector.compensation_method: "{method}"')
    (run/'input_selector.yaml').write_text(config)
    env=dict(os.environ,OMP_NUM_THREADS='2',QT_QPA_PLATFORM='offscreen',MPLBACKEND='Agg',CARTGS_AUDIT_SEED=str(seed),
        LD_LIBRARY_PATH=str(repo/'lib')+':'+os.environ.get('LD_LIBRARY_PATH',''),
        PATH='/usr/local/cuda-11.8/bin:/home/prmi/anaconda3/envs/zwj/bin:'+os.environ['PATH'])
    for key in ['CARTGS_COMPENSATION_SPLIT4','CARTGS_BUDGET_METHOD','CARTGS_BUDGET_AUDIT','CARTGS_MLP_GRAD_AUDIT']:env.pop(key,None)
    ldd=subprocess.check_output(['ldd',str(repo/'bin/tum_rgbd')],env=env,text=True)
    assert str(repo/'lib/libgaussian_mapper.so') in ldd;(run/'ldd.txt').write_text(ldd)
    cmd=[repo/'bin/tum_rgbd',repo/'third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt',repo/'cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg1_xyz.yaml',
        repo/'cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml','/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz',
        repo/'cfg/ORB_SLAM3/RGB-D/TUM/associations/fr1_xyz.txt',run,'no_viewer','--selector-config',run/'input_selector.yaml']
    (out/'training_status.json').write_text(json.dumps(dict(stage='TRAINING',run=name,completed=len(runtime))))
    memory=[];start=time.monotonic()
    with (run/'training.log').open('w') as log:
        process=subprocess.Popen(list(map(str,cmd)),cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
        while process.poll() is None:
            usage=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True).strip()
            memory.append(dict(seconds=time.monotonic()-start,memory_mib=int(usage)));time.sleep(1)
        assert process.returncode==0,(name,process.returncode)
    cps=list(run.glob('*_shutdown/ply'));assert len(cps)==1;cp=cps[0];cloud=next((cp/'point_cloud').glob('iteration_*'))
    anchor=json.loads((cloud/'anchor_selector_state.json').read_text())
    rotation=json.loads((cp/'ultrafast_rotation_state.json').read_text());stages=json.loads((cp/'mapping_stage_state.json').read_text())
    assert anchor['compensation_method']==method and anchor['private_compensation_rows']==anchor['gaussian_count']
    assert anchor['private_compensation_tables']==(4 if method=='r3_independent14' else 1)
    assert all(x['finite'] and x['adam_step']>0 for x in anchor['private_compensation_state'])
    assert anchor['feature_dim']==16 and anchor['input_dim']==27 and anchor['anchor_only_pruning']
    assert not rotation['serial_full_lod'] and rotation['post_topology_full_steps']==0 and not (cp/'transform_field.pt').exists()
    assert sum(stages[s]['lod_nonfinite'] for s in ['online','tail'])==0
    runtime[name]=dict(method=method,seed=seed,checkpoint=str(cp),wall_seconds=time.monotonic()-start,
        peak_gpu_mib_sampled=max(x['memory_mib'] for x in memory),anchor=anchor,schedule=rotation,stages=stages,config_sha256=sha(run/'input_selector.yaml'))
    (run/'gpu_memory_samples.json').write_text(json.dumps(memory));(out/'training_runtime.json').write_text(json.dumps(runtime,indent=2))
    print('TRAINED',name,anchor['iteration'],anchor['gaussian_count'],anchor['anchor_count'],flush=True)
assert all(sha(repo/p)==h for p,h in manifest.items())
(out/'training_status.json').write_text(json.dumps(dict(stage='TRAINED',completed=len(runtime))))
