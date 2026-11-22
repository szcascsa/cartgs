"""GPU 空闲后运行冻结教师／学生的配对部署测速。"""
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch


def require_idle_gpu():
    output=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    foreign=[int(line) for line in output.splitlines() if line.strip() and int(line)!=os.getpid()]
    if foreign:
        raise RuntimeError('GPU has other compute processes. Retry after it is idle. PIDs '+str(foreign))


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description='空闲 GPU 上的 Compensator 蒸馏配对测速')
    parser.add_argument('checkpoint',type=Path)
    parser.add_argument('--dataset',type=Path)
    parser.add_argument('--student',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():
        parser.error('Use a new output directory')
    require_idle_gpu()
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    from compensation_student import load_student
    from deployment_runtime import DeploymentRenderer
    from distill_compensation import TeacherSamples
    from renderer import render_prepared
    source=TeacherSamples(args.checkpoint,args.dataset)
    student,_=load_student(args.student,'cuda',source.fingerprint)
    engines=dict(teacher=source.runtime,
        student=DeploymentRenderer(source.scene.model,source.scene.gaussians,source.scene.config,student=student))
    warm_until=time.perf_counter()+2.;i=0
    while time.perf_counter()<warm_until:
        cam=source.camera((i*73+19)%len(source.scene.views))
        for engine in engines.values():
            packet,_,_=engine.prepare(cam,budget=5000)
            render_prepared(cam,packet,source.scene.background)
        torch.cuda.synchronize();i+=1
    conditions=[('1000',dict(budget=1000)),('5000',dict(budget=5000)),
        ('1pct',dict(ratio_budget=.01)),('10pct',dict(ratio_budget=.1)),('70pct',dict(ratio_budget=.7))]
    rows=[]
    for repeat in range(3):
        for condition,request in conditions:
            require_idle_gpu()
            for i in range(150):
                frame=(i*73+19)%len(source.scene.views);cam=source.camera(frame)
                for name in (['teacher','student'] if (i+repeat)%4 in (0,3) else ['student','teacher']):
                    torch.cuda.synchronize();start=time.perf_counter()
                    packet,ids,stats=engines[name].prepare(cam,**request)
                    prepared=time.perf_counter()
                    render_prepared(cam,packet,source.scene.background);torch.cuda.synchronize()
                    if i>=20:rows.append(dict(repeat=repeat,condition=condition,version=name,frame=frame,
                        N=stats['candidate_gaussians'],K=len(ids),prepare_ms=(prepared-start)*1000,
                        total_ms=(time.perf_counter()-start)*1000))
            require_idle_gpu()
    args.output.mkdir(parents=True)
    with (args.output/'samples.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=[]
    for condition,_ in conditions:
        for name in engines:
            group=[r for r in rows if r['condition']==condition and r['version']==name]
            summary.append(dict(condition=condition,version=name,samples=len(group),
                prepare_ms=float(np.median([r['prepare_ms'] for r in group])),
                total_ms=float(np.median([r['total_ms'] for r in group]))))
    (args.output/'summary.json').write_text(json.dumps(dict(teacher_fingerprint=source.fingerprint,
        idle_gpu_checks=True,results=summary),indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
