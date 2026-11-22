"""重复实验只核验原配置和独立运行时，不更改算法。"""
from pathlib import Path
import importlib.util
import os
import subprocess
import sys
import time
from prepare_experiment import R,Q,read,write,sha


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


def main():
    assert (R/'HOLD').exists() and read(R/'BUILD_READY.json')['passed']
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',QT_QPA_PLATFORM='offscreen',
        CARTGS_BATCH_FULL_TAIL_ONLY='1',PYTHONDONTWRITEBYTECODE='1',
        LD_LIBRARY_PATH=str(R/'runtime/lib')+':'+os.environ.get('LD_LIBRARY_PATH',''))
    checks=[]
    for device in ['cpu','cuda']:
        with (R/'logs'/('growth_native_'+device+'.log')).open('w') as log:
            subprocess.run([str(R/'runtime/bin/growth_threshold_test'),device],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        checks.append(dict(test='actual_Anchor_growth_and_calibration',device=device,passed=True))
    jobs=read(R/'QUEUE.json');assert len(jobs)==8
    for job in jobs:
        config=Path(job['command'][8]);mapping=Path(job['command'][3])
        assert config.read_bytes()==(Q/'configs/selector_prune1.yaml').read_bytes()
        assert mapping.read_bytes()==(Q/'configs/Replica'/job['scene']/mapping.name).read_bytes()
        label=job['scene']+'_'+job['mode']+'_repeat'+str(job['repeat'])
        with (R/'logs'/('config_gate_'+label+'.log')).open('w') as log:
            subprocess.run([str(R/'runtime/bin/no_train_gate'),str(mapping),str(config),str(R/'validation'/label),'1'],
                cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
        stage=read(R/'validation'/label/'real_lod_state.json')
        assert stage['fine_skipped_by_batch'] and stage['lod_steps']==stage['optimizer_commits']==0
        checks.append(dict(test='Full_tail_only_config',scene=job['scene'],mode=job['mode'],repeat=job['repeat'],passed=True))
    q=load('training_queue',R/'queue.py');e=load('evaluation_queue',R/'evaluate_trajectory.py');e.prepare();q.summary(jobs)
    before=read(R/'PROTECTED_BEFORE.json');assert all(sha(Path(p))==value for p,value in before.items())
    write(R/'NO_TRAIN_RUNTIME_VERIFICATION.json',dict(passed=True,checks=checks,train_steps=0,render_calls=0,optimizer_steps=0))
    write(R/'START_AUTHORIZATION.json',dict(approved_by_user=True,authorized_unix=time.time(),
        request='room0／room1各Mono／RGB-D，q80 alpha1.5配置，每组seed0额外重跑两次；不保存渲染图片',queue_sha256=sha(R/'QUEUE.json')))
    print('PREFLIGHT_PASS_8_REPEAT_CONFIGS_CPU_CUDA',flush=True)


if __name__=='__main__':main()
