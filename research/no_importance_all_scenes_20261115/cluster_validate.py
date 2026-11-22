"""新3090节点验收：完整性、相同CUDA微图、各模式短程训练及数值渲染。"""
from pathlib import Path
import copy
import csv
import json
import os
import subprocess
import sys
import time
import torch
import run
from prepare import R,read,write,sha


def main():
    worker=sys.argv[1];assert worker in ['cloud_a','cloud_b']
    os.environ['CARTGS_CLUSTER_WORKER']=worker
    assert read('/root/autodl-tmp/no_importance_20261115/receipts/FILES_VERIFIED.json')['passed']
    for name,target in read(R/'ALIASES.json').items():
        p=Path(name);p.parent.mkdir(parents=True,exist_ok=True)
        assert not p.exists() or (p.is_symlink() and p.resolve()==Path(target).resolve()),str(p)
        if not p.is_symlink():p.symlink_to(target,target_is_directory=True)
    assert torch.__version__=='2.3.1+cu118' and torch.version.cuda=='11.8'
    assert torch.cuda.get_device_name().endswith('3090')
    (R/'logs').mkdir(exist_ok=True);(R/'validation').mkdir(exist_ok=True)
    tests=[]
    for binary,mode in [('lod_health_guard_test',''),('joint_view_conditioning_test','unit'),
        ('joint_view_conditioning_test','view_d'),('joint_view_conditioning_test','view_li_isolation')]:
        name=binary+'_'+mode;cmd=[str(R/'runtime/bin'/binary)]+([mode,str(R/'validation'/name)] if mode else [])
        with (R/'logs'/('cloud_validate_'+name+'.log')).open('w') as log:
            subprocess.run(cmd,cwd=R,env=run.environment(),stdout=log,stderr=subprocess.STDOUT,check=True)
        tests.append(name)
    result=R/'validation/cloud_cuda.pt'
    with (R/'logs/cloud_cuda.log').open('w') as log:
        subprocess.run([str(R/'runtime/bin/joint_view_cuda_parity_test'),str(result)],env=run.environment(),stdout=log,stderr=subprocess.STDOUT,check=True)
    a=torch.jit.load(str(R/'validation/cuda_D.pt'),map_location='cpu').state_dict()
    b=torch.jit.load(str(result),map_location='cpu').state_dict();assert a.keys()==b.keys()
    differences=[]
    for k,v in a.items():
        if not torch.equal(v.contiguous().reshape(-1).view(torch.uint8),b[k].contiguous().reshape(-1).view(torch.uint8)):
            differences.append(dict(name=k,max_abs=float((v-b[k]).abs().max())))
    write(R/'reports/CLOUD_CUDA_PARITY.json',dict(passed=not differences,tensors=len(a),differences=differences))
    assert not differences,'跨节点固定CUDA微图不逐bit一致；停止正式训练'
    state=read(R/'LOD_PIPELINE.json');seen=[];records=[]
    def save(value):write(R/'reports/CLOUD_VALIDATION_PROGRESS.json',dict(worker=worker,tests=tests,validated_modes=seen,current=value))
    oldsave=run.save;run.save=save
    try:
        for original in state['jobs']:
            if original['assigned_worker']!=worker:continue
            mode=original['dataset']+'/'+original['mode']
            if mode in seen:continue
            job=copy.deepcopy(original);job.update(state='pending',attempt_history=[],evaluations=[],oom_failures=0)
            smoke=dict(jobs=[job],phase='validation')
            folder=R/'validation/cloud'/mode
            out=run.train(job,smoke,folder,smoke=6);assert out is not None
            with (Path(job['reference_evaluations'][0]['result']['output'])/'per_frame_metrics.csv').open() as f:rows=list(csv.DictReader(f))
            frame_set=folder/'frames.json';write(frame_set,dict(gt_indices=[int(rows[i]['rgb_index']) for i in [0,len(rows)//2,len(rows)-1]]))
            for scope,ratio in [('full',1.),('global',.01),('global',.03),('global',.7),('view_ratio',.01)]:
                item=dict(scope=scope,ratio=ratio,state='pending',kind='validation',oom_failures=0);job['evaluations'].append(item)
                run.evaluate(job,item,smoke,folder/(scope+'_'+str(ratio)),frame_set)
                assert item['state']=='complete'
            records.append(dict(mode=mode,job=job['id'],initialization=read(out/'initialization_check.json') if (out/'initialization_check.json').exists() else 'passed_log',
                manifest_sha256=sha(out/'manifest.json'),evaluations=[e['result'] for e in job['evaluations']]))
            seen.append(mode);save(smoke);print('CLOUD_SMOKE_PASSED',worker,mode,flush=True)
    finally:run.save=oldsave
    write(R/'reports/CLOUD_VALIDATION.json',dict(passed=True,worker=worker,tests=tests,cuda_bitwise_tensors=len(a),modes=seen,
        smoke_steps_each=6,evaluated_frames_each=3,records=records,verified_unix=time.time()))
    print('CLOUD_VALIDATION_PASSED',worker,flush=True)


if __name__=='__main__':main()
