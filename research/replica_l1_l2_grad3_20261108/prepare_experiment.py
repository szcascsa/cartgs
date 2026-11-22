"""服务器恢复连接后准备隔离的16组Full＋tail任务；不复用旧模型继续Fine。"""
from pathlib import Path
from collections import Counter
import copy
import hashlib
import json
import shutil
import time
from patch_growth import apply

R=Path('/home/prmi/zwj/realLOD/checkpoint/replica_l1_l2_grad3_20261108')
F=Path('/home/prmi/zwj/realLOD/saved_experiments/multi_dataset_full_tail_prune1_seed0_20261103')
SCENES=['office0','office1','office2','office3','office4','room0','room1','room2']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def planned_jobs(original):
    selected=[copy.deepcopy(j) for j in original if j.get('dataset')=='Replica' and j.get('repeat')==1]
    if Counter((j['scene'],j['mode']) for j in selected)!=Counter((s,m) for s in SCENES for m in ['mono','rgbd']):
        raise ValueError('必须恰好包含Replica八场景的Mono和RGB-D各一遍')
    jobs=[]
    for job in selected:
        if job['seed']!=0 or job['prune']!=1:raise ValueError('只允许seed0、only Anchor prune=1')
        job={k:job[k] for k in ['id','dataset','scene','mode','prune','seed','repeat','frames','command','environment']}
        job['command']=list(job['command']);job['environment']=dict(job['environment'])
        config=R/'configs/Replica'/job['scene']
        output=R/'jobs'/job['id']/'attempt_01'
        command=job['command']
        if len(command)!=9 or command[7]!='--selector-config':raise ValueError('Replica启动参数不匹配')
        command[0]=str(R/'runtime/bin'/('replica_'+job['mode']))
        command[1]=str(R/'runtime/third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt')
        command[2]=str(config/(job['mode']+'_camera.yaml'));command[3]=str(config/(job['mode']+'_mapping.yaml'))
        command[4]=str(F/'datasets/Replica'/job['scene']);command[5]=str(output/'model')
        command[8]=str(R/'configs/selector_prune1.yaml')
        job.update(state='held',output=str(output),input_sha256={},attempt_history=[])
        job['environment'].update(CARTGS_AUDIT_SEED='0',CARTGS_BATCH_FULL_TAIL_ONLY='1')
        jobs.append(job)
    return jobs


def main():
    assert Path(__file__).resolve().parent==R
    assert not (R/'PREPARED.json').exists() and not (R/'source').exists(),'已有现场不能覆盖'
    assert read(F/'BUILD_READY.json')['passed']
    assert shutil.disk_usage(R).free>5*2**30
    for folder in ['reports','logs','objects','configs','jobs','validation','evaluation']:(R/folder).mkdir(exist_ok=True)
    (R/'HOLD').write_text('独立编译、原生增长边界及入口检查通过后才能启动\n',encoding='utf-8')
    protected={}
    for folder in ['source','runtime','evaluation_runtime']:
        shutil.copytree(F/folder,R/folder,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for folder in ['source','runtime']:
        for path in (F/folder).rglob('*'):
            if path.is_file() and not path.is_symlink() and not any(a.is_symlink() for a in path.parents if a!=F):
                protected[str(path)]=sha(path)
    for folder in ['configs/Replica']:
        shutil.copytree(F/folder,R/folder,symlinks=True)
    shutil.copy2(F/'configs/selector_prune1.yaml',R/'configs/selector_prune1.yaml')
    # 原训练、评价及OOM重试工具只复制到本实验；原队列继续只读。
    for name in ['queue.py','evaluate_trajectory.py','trajectory_eval_entry.py','smoke_eval.py','oom_retry.py']:
        text=(F/name).read_text(encoding='utf-8')
        if name in ['queue.py','evaluate_trajectory.py']:text=text.replace('== 66','== 16')
        (R/name).write_text(text,encoding='utf-8')
    jobs=planned_jobs(read(F/'QUEUE.json'))
    for job in jobs:
        inputs=[Path(job['command'][i]) for i in [2,3,8]]+[F/'datasets/Replica'/job['scene']/'pose_TUM.txt']
        assert Path(job['command'][4]).is_dir()
        job['input_sha256']={str(p):sha(p) for p in inputs}
        for p in inputs:
            if not p.is_relative_to(R):protected[str(p)]=sha(p)
    inventory=[copy.deepcopy(x) for x in read(F/'SCENE_INVENTORY.json') if x['dataset']=='Replica']
    assert len(inventory)==8
    for item in inventory:item['data']=str(F/'datasets/Replica'/item['scene'])
    write(R/'SCENE_INVENTORY.json',inventory);write(R/'QUEUE.json',jobs)
    changed=apply(R/'source',R/'reports')
    write(R/'PROTECTED_BEFORE.json',protected)
    write(R/'PREPARED.json',dict(passed=True,prepared_unix=time.time(),jobs=16,seed=0,repeats=1,
          full_slam=True,normal_tail=True,lod_fine=False,save_images=False,
          l1_l2_tau0_multiplier=3,baseline_full_source=str(F/'source'),
          source_patch=changed,compiled=False,native_tests_run=False,training_started=False))
    print('PREPARED_16_FULL_TAIL_JOBS_BUILD_AND_NATIVE_VALIDATION_REQUIRED',flush=True)


if __name__=='__main__':main()
