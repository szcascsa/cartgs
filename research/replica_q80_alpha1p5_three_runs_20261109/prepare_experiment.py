"""补齐其余六场景的重复2、3，汇总Replica八场景各模式三遍。"""
from collections import Counter
from pathlib import Path
import copy
import hashlib
import json
import os
import shutil
import subprocess
import time

R=Path('/home/prmi/zwj/realLOD/checkpoint/replica_q80_alpha1p5_three_runs_20261109')
Q=Path('/home/prmi/zwj/realLOD/checkpoint/replica_q80_alpha1p5_20261109')
A=Path('/home/prmi/zwj/realLOD/checkpoint/replica_q80_alpha1p5_room01_repeats_20261109')
F=Path('/home/prmi/zwj/realLOD/saved_experiments/multi_dataset_full_tail_prune1_seed0_20261103')
SCENES=['office0','office1','office2','office3','office4','room2']


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path,value):
    path=Path(path);temporary=path.with_name(path.name+'.tmp')
    with temporary.open('w',encoding='utf-8') as stream:
        json.dump(value,stream,ensure_ascii=False,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def planned_jobs(original):
    originals=[j for j in original if j['dataset']=='Replica' and j['scene'] in SCENES and j['repeat']==1]
    assert Counter((j['scene'],j['mode']) for j in originals)==Counter((s,m) for s in SCENES for m in ['mono','rgbd'])
    jobs=[]
    for repeat in [2,3]:
        for prior in originals:
            assert prior['seed']==0 and prior['prune']==1
            job={k:copy.deepcopy(prior[k]) for k in ['dataset','scene','mode','prune','seed','frames','command','environment']}
            job.update(id=f"Replica/{job['scene']}/{job['mode']}/prune1/seed0_repeat{repeat}",repeat=repeat,
                state='held',input_sha256={},attempt_history=[])
            job['command']=[x.replace(str(Q),str(R)) for x in job['command']]
            output=R/'jobs'/job['id']/'attempt_01';job['output']=str(output);job['command'][5]=str(output/'model')
            job['environment'].update(CARTGS_AUDIT_SEED='0',CARTGS_BATCH_FULL_TAIL_ONLY='1')
            assert len(job['command'])==9 and job['command'][7]=='--selector-config'
            jobs.append(job)
    return jobs


def main():
    assert Path(__file__).resolve().parent==R and not (R/'PREPARED.json').exists()
    assert read(Q/'FINISHED.json')['passed'] and read(A/'FINISHED.json')['passed']
    assert read(Q/'NO_TRAIN_RUNTIME_VERIFICATION.json')['passed']
    assert shutil.disk_usage(R).free>5*2**30
    for folder in ['reports','logs','jobs','validation','evaluation']:(R/folder).mkdir(exist_ok=True)
    (R/'HOLD').write_text('独立加载、配置和机器码核验通过后启动\n',encoding='utf-8')
    for folder in ['source','runtime','evaluation_runtime','configs']:
        shutil.copytree(Q/folder,R/folder,symlinks=True,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    protected={p:sha(Path(p)) for p in read(A/'PROTECTED_BEFORE.json')}
    for name in ['QUEUE.json','TRAJECTORY_EVALUATION.json','FINISHED.json','BUILD_READY.json']:
        protected[str(Q/name)]=sha(Q/name)
    for folder in ['source','runtime','evaluation_runtime','configs','jobs']:
        for path in (Q/folder).rglob('*'):
            if path.is_file() and not path.is_symlink() and not any(p.is_symlink() for p in path.parents if p!=Q):
                if folder!='jobs' or path.suffix in ['.pt','.ply','.yaml']:
                    protected[str(path)]=sha(path)
    for name in ['QUEUE.json','TRAJECTORY_EVALUATION.json','FINISHED.json','BUILD_READY.json']:
        protected[str(A/name)]=sha(A/name)
    for folder in ['source','runtime','configs','jobs']:
        for path in (A/folder).rglob('*'):
            if path.is_file() and not path.is_symlink() and not any(p.is_symlink() for p in path.parents if p!=A):
                if folder!='jobs' or path.suffix in ['.pt','.ply','.yaml']:
                    protected[str(path)]=sha(path)
    source_hashes={str(p.relative_to(Q/'source')):sha(p) for p in (Q/'source').rglob('*') if p.is_file() and not p.is_symlink()}
    assert all(sha(R/'source'/name)==value for name,value in source_hashes.items())
    write(R/'reports/SOURCE_AUDIT.json',dict(passed=True,source=str(Q/'source'),unchanged_files=len(source_hashes),sha256=source_hashes))
    for name in ['queue.py','evaluate_trajectory.py','trajectory_eval_entry.py','smoke_eval.py','oom_retry.py']:
        text=(Q/name).read_text(encoding='utf-8')
        if name in ['queue.py','evaluate_trajectory.py']:text=text.replace('== 16','== 24')
        if name=='queue.py':
            text=text.replace("experiment='replica_q80_alpha1p5_20261109'",f"experiment='{R.name}'")
        (R/name).write_text(text,encoding='utf-8')
    shutil.copy2(Q/'reports/GROWTH_THRESHOLD_CHANGE.json',R/'reports/GROWTH_THRESHOLD_CHANGE.json')
    jobs=planned_jobs(read(Q/'QUEUE.json'))
    for job in jobs:
        paths=[Path(job['command'][i]) for i in [2,3,8]]+[Path(job['command'][4])/'pose_TUM.txt']
        job['input_sha256']={str(p):sha(p) for p in paths}
        assert Path(job['command'][4]).is_dir()
    write(R/'QUEUE.json',jobs)
    write(R/'SCENE_INVENTORY.json',[x for x in read(Q/'SCENE_INVENTORY.json') if x['scene'] in SCENES])
    env=dict(os.environ,LD_LIBRARY_PATH=str(R/'runtime/lib')+':'+os.environ.get('LD_LIBRARY_PATH',''))
    links={};code_hashes={};patchelf='/home/prmi/anaconda3/bin/patchelf'
    binaries=[R/'runtime/bin'/name for name in ['replica_mono','replica_rgbd','no_train_gate','growth_threshold_test']]
    libraries=[p for p in (R/'runtime/lib').iterdir() if p.is_file() and p.name.endswith('.so')]
    for path in binaries+libraries:
        prior=Q/path.relative_to(R)
        old_code=subprocess.check_output(['readelf','-x','.text',str(prior)])
        rpath=subprocess.check_output([patchelf,'--print-rpath',str(path)],text=True).strip()
        subprocess.run([patchelf,'--set-rpath',rpath.replace(str(Q),str(R)),str(path)],check=True)
        new_code=subprocess.check_output(['readelf','-x','.text',str(path)])
        assert old_code==new_code,'机器码改变: '+str(path)
        code_hashes[str(path.relative_to(R))]=hashlib.sha256(new_code).hexdigest()
    for path in binaries:
        text=subprocess.check_output(['ldd',str(path)],text=True,env=env)
        assert str(R/'runtime/lib/libgaussian_mapper.so') in text and 'not found' not in text
        links[path.name]=text
    write(R/'BUILD_READY.json',dict(passed=True,recompiled=False,source_runtime_copied_from=str(Q),
        only_rpath_relocated=True,text_sections_identical=True,text_section_sha256=code_hashes,ldd=links,
        binaries={str(p.relative_to(R)):sha(p) for p in binaries+libraries}))
    write(R/'PROTECTED_BEFORE.json',protected)
    write(R/'PREPARED.json',dict(passed=True,jobs=24,seed=0,repeats=[2,3],original_repeat=str(Q),room01_repeats=str(A),quantile=.8,alpha=1.5,
        strict_upper=True,full_slam=True,normal_tail=True,lod_fine=False,save_images=False,prepared_unix=time.time()))
    print('PREPARED_24_REPEATS_IDENTICAL_MACHINE_CODE',flush=True)


if __name__=='__main__':main()
