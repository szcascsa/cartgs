"""独立运行真实Anchor边界测试和16组无训练配置检查，校验后才授权队列。"""
from pathlib import Path
import importlib.util
import os
import subprocess
import sys
import time
from prepare_experiment import R,F,read,write,sha
from patch_growth import CONFIG_APPEND


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module);return module


def main():
    assert Path(__file__).resolve().parent==R and (R/'HOLD').exists()
    assert read(R/'BUILD_READY.json')['passed']
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',QT_QPA_PLATFORM='offscreen',
             CARTGS_BATCH_FULL_TAIL_ONLY='1',PYTHONDONTWRITEBYTECODE='1')
    env['LD_LIBRARY_PATH']=str(R/'runtime/lib')+':'+env.get('LD_LIBRARY_PATH','')
    records=[]
    for mode in ['cpu','cuda']:
        with (R/'logs'/('growth_native_'+mode+'.log')).open('w') as log:
            result=subprocess.run([str(R/'runtime/bin/growth_threshold_test'),mode],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
        assert result.returncode==0,mode
        records.append(dict(test='actual_AnchorBank_growBlocks',device=mode,passed=True))
    jobs=read(R/'QUEUE.json')
    for job in jobs:
        config=Path(job['command'][8]);mapping=Path(job['command'][3])
        assert config.read_text()==(F/'configs/selector_prune1.yaml').read_text()+CONFIG_APPEND
        assert mapping.read_bytes()==(F/'configs/Replica'/job['scene']/mapping.name).read_bytes()
        label=job['scene']+'_'+job['mode']
        with (R/'logs'/('config_gate_'+label+'.log')).open('w') as log:
            result=subprocess.run([str(R/'runtime/bin/no_train_gate'),str(mapping),str(config),
                                   str(R/'validation'/label),'1'],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
        assert result.returncode==0,label
        stage=read(R/'validation'/label/'real_lod_state.json')
        assert stage['fine_skipped_by_batch'] and stage['lod_steps']==stage['optimizer_commits']==0
        records.append(dict(test='Full_tail_only_gate',scene=job['scene'],mode=job['mode'],passed=True))
    q=load('full_training_queue',R/'queue.py');e=load('full_evaluation_queue',R/'evaluate_trajectory.py')
    e.prepare();q.summary(jobs)
    before=read(R/'PROTECTED_BEFORE.json');assert all(sha(Path(p))==h for p,h in before.items())
    write(R/'NO_TRAIN_RUNTIME_VERIFICATION.json',dict(passed=True,checks=records,
          train_steps=0,render_calls=0,optimizer_steps=0,native_anchor_tests=2,configuration_checks=16))
    write(R/'START_AUTHORIZATION.json',dict(approved_by_user=True,authorized_unix=time.time(),
          request='Replica八场景×Mono/RGB-D，seed0各一次，只重建Full＋正常tail；首个有效L0窗口q80校准t后固定，alpha=1.5，严格超过上界细化；不追加LoD，不保存图片',
          queue_sha256=sha(R/'QUEUE.json')))
    print('PREFLIGHT_PASS_NATIVE_CPU_CUDA_AND_16_CONFIGS',flush=True)


if __name__=='__main__':main()
