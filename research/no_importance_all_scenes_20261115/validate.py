"""原生健康策略、旧开关数值回归、六种入口与3帧推理验收。"""
from pathlib import Path
import copy
import csv
import json
import shutil
import subprocess
import sys
import torch
from prepare import R,B,read,write,sha
from run import environment,train,evaluate


def main():
    assert read(R/'BUILD_READY.json')['passed'] and not (R/'reports/VALIDATION.json').exists()
    # This generated evaluator only hashes selected IDs instead of storing large ID arrays.
    q=B/'source/python/run_metrics_only.py';shutil.copy2(q,R/'source/python/run_metrics_only.py')
    hashes=read(R/'reports/SOURCE_HASHES.json');hashes['python/run_metrics_only.py']=sha(q);write(R/'reports/SOURCE_HASHES.json',hashes)
    tests=[]
    for binary,mode in [('lod_health_guard_test',''),('joint_view_conditioning_test','unit'),
        ('joint_view_conditioning_test','view_d'),('joint_view_conditioning_test','view_li_isolation')]:
        name=binary+'_'+mode;cmd=[str(R/'runtime/bin'/binary)]+([mode,str(R/'validation'/name)] if mode else [])
        with (R/'logs'/('validate_'+name+'.log')).open('w') as f:
            p=subprocess.run(cmd,cwd=R,env=environment(),stdout=f,stderr=subprocess.STDOUT)
        assert p.returncode==0,(name,p.returncode)
        tests.append(dict(name=name,passed=True));write(R/'reports/VALIDATION_PROGRESS.json',dict(tests=tests,state='native'))
    archives=[]
    for name,lib in [('original',B/'runtime/lib'),('D',R/'runtime/lib')]:
        path=R/'validation'/('cuda_'+name+'.pt');env=environment();env['LD_LIBRARY_PATH']=str(lib)+':'+env['LD_LIBRARY_PATH']
        with (R/'logs'/('cuda_'+name+'.log')).open('w') as f:
            subprocess.run([str(R/'runtime/bin/joint_view_cuda_parity_test'),str(path)],env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
        archives.append(torch.jit.load(str(path),map_location='cpu').state_dict())
    assert archives[0].keys()==archives[1].keys()
    assert all(torch.equal(v.contiguous().reshape(-1).view(torch.uint8),archives[1][k].contiguous().reshape(-1).view(torch.uint8)) for k,v in archives[0].items())
    queue=read(R/'LOD_PIPELINE.json');state=dict(jobs=[],counts={},phase='validation');done=[]
    for old in queue['jobs']:
        mode=old['dataset']+'/'+old['mode']
        if mode in done:continue
        job=copy.deepcopy(old);job.update(state='pending',attempt_history=[],evaluations=[],oom_failures=0)
        state['jobs'].append(job)
        # Keep validation progress separate from the formal queue.
        import run
        original_save=run.save
        run.save=lambda value:write(R/'reports/VALIDATION_PROGRESS.json',dict(state='smoke',mode=mode,details=value))
        try:
            out=train(job,state,R/'validation'/mode,smoke=6);assert out is not None
            reference=Path(old['reference_evaluations'][0]['result']['output'])/'per_frame_metrics.csv'
            with reference.open() as f:rows=list(csv.DictReader(f))
            frame_set=R/'validation'/mode/'frames.json'
            write(frame_set,dict(gt_indices=[int(rows[k]['rgb_index']) for k in [0,len(rows)//2,len(rows)-1]]))
            for scope,ratio in [('full',1.),('global',.01),('global',.03),('global',.7),('view_ratio',.01)]:
                item=dict(scope=scope,ratio=ratio,kind='validation',state='pending',oom_failures=0);job['evaluations'].append(item)
                evaluate(job,item,state,R/'validation'/mode/('eval_'+scope+'_'+str(ratio)),frame_set)
                assert item['state']=='complete'
        finally:run.save=original_save
        done.append(mode)
        write(R/'reports/VALIDATION_PROGRESS.json',dict(state='smoke_complete',validated_modes=done,tests=tests))
    assert len(done)==6
    queue['validated_modes']=done;write(R/'LOD_PIPELINE.json',queue)
    write(R/'reports/VALIDATION.json',dict(passed=True,native_tests=tests,cuda_bitwise_tensors=len(archives[0]),
        modes=done,smoke_steps_each=6,evaluation_cases=30,evaluation_frames_each=3,images_saved=0))
    print('VALIDATION_PASSED',done,flush=True)


if __name__=='__main__':main()
