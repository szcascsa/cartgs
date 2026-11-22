"""按场景分派尚未开始的D组，制作无凭据、可核验的独立运行包。"""
from pathlib import Path
import collections
import copy
import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
import time
from prepare import R,read,write,sha

C=R/'cluster'


def plan():
    C.mkdir(exist_ok=True)
    s=read(R/'LOD_PIPELINE.json');groups=collections.defaultdict(list)
    for j in s['jobs']:
        if j['state']=='pending':groups[(j['dataset'],j['scene'])].append(j)
    loads=dict(original=0.,cloud_a=0.,cloud_b=0.);assignment={}
    weights={}
    for key,jobs in groups.items():
        seconds=0
        for j in jobs:
            p=Path(j['reference_model']).parent/'exit.json'
            seconds+=read(p).get('seconds',0) if p.exists() else 0
            seconds+=sum(e['attempt_history'][-1].get('seconds',0) for e in j['reference_evaluations'])
        weights[key]=seconds
    # 同场景的两种相机模式同节点，减少重复传图；按原实际训练和评价时间平衡。
    for key in sorted(groups,key=lambda k:-weights[k]):
        worker=min(loads,key=loads.get)
        for j in groups[key]:assignment[j['id']]=worker
        loads[worker]+=weights[key]
    for j in s['jobs']:j['assigned_worker']=assignment.get(j['id'],'original')
    write(C/'ASSIGNMENTS.json',dict(created_unix=time.time(),estimated_reference_seconds=loads,
        assignments={j['id']:j['assigned_worker'] for j in s['jobs']},nodes=dict(original='172.19.163.58',
        cloud_a=dict(host='connect.nmb2.seetacloud.com',port=43433),cloud_b=dict(host='connect.nmb2.seetacloud.com',port=14086))))
    write(C/'INITIAL_PIPELINE.json',s)
    print(json.dumps({w:[j['id'] for j in s['jobs'] if j['assigned_worker']==w] for w in loads},ensure_ascii=False),flush=True)


def add_files(table,folder):
    for p in Path(folder).rglob('*'):
        if p.is_file() and '__pycache__' not in p.parts:table[str(p).lstrip('/')]=p


def archive(label,files):
    dest=C/(label+'.tar.gz');manifest={};begin=time.time()
    with dest.open('wb') as stream:
        zipproc=subprocess.Popen(['pigz','-p','6','-1'],stdin=subprocess.PIPE,stdout=stream)
        with tarfile.open(fileobj=zipproc.stdin,mode='w|',dereference=True) as tf:
            for name,p in sorted(files.items()):
                assert Path(name).is_absolute() is False and '..' not in Path(name).parts
                assert name.startswith('home/prmi/')
                manifest[name]=dict(size=p.stat().st_size,sha256=sha(p))
                tf.add(p,arcname=name,recursive=False)
        zipproc.stdin.close();assert zipproc.wait()==0
    write(C/(label+'.files.json'),manifest)
    write(C/(label+'.receipt.json'),dict(bytes=dest.stat().st_size,sha256=sha(dest),files=len(manifest),seconds=time.time()-begin))
    print('ARCHIVED',label,dest.stat().st_size,len(manifest),round(time.time()-begin),flush=True)


def common():
    files={};add_files(files,'/home/prmi/anaconda3/envs/zwj')
    archive('environment',files)
    files={}
    for p in R.glob('*.py'):files[str(p).lstrip('/')]=p
    for d in ['source','configs','runtime','evaluation_runtime']:
        # third_party下为历史只读链接；只复制源码本身，不把依赖工程全量带走。
        for p in (R/d).rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:files[str(p).lstrip('/')]=p
    for name in ['reports/VALIDATION.json','reports/SOURCE_HASHES.json','reports/BUILD_HASHES.json','BUILD_READY.json','PROTOCOL.md']:
        files[str(R/name).lstrip('/')]=R/name
    libroot=Path('/home/prmi/zwj/mycartgs/third_party/libtorch-cu118/libtorch/lib')
    add_files(files,libroot)
    output=subprocess.check_output(['ldd',str(R/'runtime/bin/lod_from_full')],text=True)
    dependencies=re.findall(r'=> (/\S+)',output)
    core={'libpthread.so.0','libstdc++.so.6','libm.so.6','libgcc_s.so.1','libc.so.6','libdl.so.2','librt.so.1'}
    for entry in dependencies:
        p=Path(entry)
        if str(p).startswith('/home/prmi/'):files[str(p).lstrip('/')]=p
        elif p.name not in core:
            name=str(R/'cluster_deps/lib'/p.name).lstrip('/');files[name]=p
    # cuDNN子库由dlopen载入，不能仅依靠ldd。
    for p in libroot.glob('libcudnn*'):files[str(p).lstrip('/')]=p
    for p in Path('/home/prmi/.cache/torch/hub/checkpoints').glob('*'):
        if p.is_file():files[str(p).lstrip('/')]=p
    archive('runtime',files)


def worker(which):
    s=read(C/'INITIAL_PIPELINE.json');jobs=[j for j in s['jobs'] if j['assigned_worker']==which];files={};aliases={}
    assert jobs and all(j['state']=='pending' for j in jobs)
    for j in jobs:
        full=Path(j['full_job']['result']['checkpoint']);add_files(files,full)
        for n in ['CameraTrajectory_TUM.txt','TrackingTime.txt']:
            p=Path(j['full_evaluation']['model'])/n
            if p.exists():files[str(p).lstrip('/')]=p
        ref=Path(j['reference_model'])
        for p in [ref/'training_keyframes.csv',*ref.glob('*_shutdown/ply/real_lod_state.json')]:files[str(p).lstrip('/')]=p
        for e in j['reference_evaluations']:
            for n in ['per_frame_metrics.csv','image_preprocessing.json']:
                p=Path(e['result']['output'])/n
                if p.exists():files[str(p).lstrip('/')]=p
        data=Path(j['full_evaluation']['data'])
        for p in data.iterdir():
            if p.is_file() and p.suffix in ['.txt','.json']:files[str(p).lstrip('/')]=p
        rgb=data/('results' if j['dataset']=='Replica' else 'rgb')
        for p in rgb.iterdir():
            if p.is_file() and (j['dataset']!='Replica' or p.name.startswith('frame')):
                files[str(p).lstrip('/')]=p
        # 保持cameras.json字节不变；绝对图像别名只在新节点建立。
        for cam in read(full/'cameras.json'):
            img=Path(cam['img_name'])
            if img.is_absolute() and not str(img).startswith(str(data)):
                part='results' if j['dataset']=='Replica' else 'rgb'
                prefix=Path(str(img).split('/'+part+'/')[0]);aliases[str(prefix)]=str(data)
    local=C/which;local.mkdir(exist_ok=True)
    write(local/'ALIASES.json',aliases);write(local/'LOD_PIPELINE.json',s)
    for name in ['ALIASES.json','LOD_PIPELINE.json']:
        files[str(R/name).lstrip('/')]=local/name
    files[str(R/'cluster/ASSIGNMENTS.json').lstrip('/')]=C/'ASSIGNMENTS.json'
    archive(which,files)


if __name__=='__main__':
    mode=sys.argv[1]
    if mode=='plan':plan()
    elif mode=='common':common()
    else:worker(mode)
