"""32组新联合Fine队列：每步三路监督、单次Adam；全局与局部全轨迹数值评价。"""
from pathlib import Path
from collections import Counter
import fcntl
import json
import math
import os
import resource
import shutil
import subprocess
import sys
import time
import pipeline as p

R=Path(__file__).resolve().parent
p.config=lambda:R/'configs/joint.yaml'
original_environment=p.environment
original_run=p.run_process
original_verify=p.verify_training


def environment():
    env=original_environment()
    env['CARTGS_AUDIT_SEED']='0'
    for key in list(env):
        if key.startswith('CARTGS_LOD_DIAGNOSTIC_'):env.pop(key)
    return env


p.environment=environment


def manifest(job,output):
    full=Path(job['full_job']['result']['checkpoint'])
    cloud,=(full/'point_cloud').glob('iteration_*/point_cloud.ply')
    value=dict(full_ply=str(full),point_cloud=str(cloud),anchor_state=str(cloud.parent/'anchor_selector_state.json'),
        camera_config=job['camera_config'],mapping_config=job['mapping_config'],selector_config=str(p.config()),
        initial_iteration=int(full.parent.name.split('_')[0]),output=str(output),mode=job['mode'],dataset=job['dataset'],
        dataset_root=job['full_evaluation']['data'],optimizer_protocol='fresh Adam/RNG; not complete SLAM resume')
    value['input_sha256']={str(path):p.sha(path) for path in [cloud,cloud.parent/'anchor_selector_state.json',
        cloud.parent/'anchor_pruning_state.pt',full/'selector.pt',full/'cameras.json',full/'real_lod_state.json',
        Path(value['camera_config']),Path(value['mapping_config']),p.config()]}
    return value


p.manifest=manifest


def run_process(argv,directory,item,state,evaluation=False):
    # 固定地图离线Fine与异步SLAM不同；共享负载只影响速度，不能改训练量或降低分辨率避OOM。
    while True:
        free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
        assert shutil.disk_usage(R).free>5*2**30,'剩余磁盘不足5GiB，暂停'
        if free>=6144:break
        state.update(phase='waiting_gpu_memory',gpu_free_mib=free);p.save(state)
        time.sleep(30)
    return original_run(argv,directory,item,state,evaluation)


p.run_process=run_process


def verify_training(out,target=None):
    ply=original_verify(out,target)
    stage=p.read(ply/'real_lod_state.json')
    assert stage['view_budget_enabled'] and stage['view_budget_loss_weight']==1
    assert stage['view_budget_training_renders']>0
    assert stage['view_budget_training_backwards']==stage['view_budget_training_renders']
    assert stage['view_budget_training_renders']<=stage['lod_steps']
    return ply


p.verify_training=verify_training


def evaluate(job,item,state,root):
    model=Path(job['trained_model']);ply,=model.glob('*_shutdown/ply')
    def command(out):
        argv=[sys.executable,str(R/'eval_joint_entry.py'),str(R/'source/python/run.py'),str(model),
            job['full_evaluation']['data'],'--shutdown-model',ply.parent.name,'--output_path',str(out),'--no-save-images']
        if item['scope']!='full':
            argv+=['--selector',str(ply/'selector.pt'),'--selector-ratio',str(item['ratio']),'--selector-config',str(p.config())]
        if item['scope']=='view_ratio':argv.append('--joint-local-ratio')
        if job['mode']=='mono':argv.append('--correct_scale')
        if job['mode']=='stereo':argv+=['--stereo-camera-config',job['camera_config'],'--stereo-native-maps',str(R/'configs/vector_stereo_native_maps.yml.gz')]
        return argv
    out=p.run_process(command,root,item,state,evaluation=True)
    if out is None:return
    result=p.read(out/'metric_summary.json')
    assert result['evaluated_frames']==job['full_evaluation']['result']['evaluated_frames']
    assert all(math.isfinite(result[k]) for k in ['psnr','ssim','lpips','ate_rmse'])
    assert not any(x.suffix.lower() in ['.png','.jpg','.jpeg'] for x in out.rglob('*'))
    result.update(scope=item['scope'],output=str(out),saved_images=0)
    item.update(state='complete',result=result);p.save(state)
    # 全局PLY已完成渲染，只保留最终模型与数值结果；不会删除输入或历史实验。
    packet=out/'point_cloud.ply'
    if packet.exists():
        p.write(out/'temporary_export_removed.json',{'sha256':p.sha(packet),'bytes':packet.stat().st_size,'reason':'final metrics only'})
        packet.unlink()


def evaluations():
    return [dict(scope=s,ratio=r,kind=s+'_'+('train' if r in p.TRAIN else 'unseen'),state='pending')
            for s in ['global','view_ratio'] for r in sorted(p.TRAIN+p.UNSEEN)]+[dict(scope='full',ratio=1.,kind='full',state='pending')]


def verify_inputs():
    before=p.read(R/'PROTECTED_BEFORE.json')
    after={name:{path:p.sha(Path(path)) for path in values} for name,values in before.items()}
    assert before==after,'输入或生产文件发生改变'
    p.write(R/'PROTECTED_AFTER.json',after)


def main():
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    lock=(R/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p.write(R/'LOD_CONTROLLER.json',p.identity(os.getpid()))
    assert p.read(R/'PREPARED.json')['passed']
    state=p.read(R/'LOD_PIPELINE.json')
    try:
        validated=set(state['validated_modes'])
        for job in state['jobs']:
            if job['state'] in ['complete','quality_failed','skipped_oom','partial_evaluation']:continue
            assert job['state'] in ['pending','retry_pending','trained'],'中断现场必须核查，不能覆盖running'
            kind=job['dataset']+'/'+job['mode']
            if kind not in validated:
                state['phase']='validating_'+kind
                v=dict(job,state='pending',attempt_history=[],evaluations=[])
                state['active_validation']=v;p.save(state)
                root=R/'validation'/kind
                out=p.launch_training(v,state,root,smoke=6)
                assert out is not None,'验证OOM，不跳过验证进入正式训练'
                verify_training(out,6)
                check=subprocess.run([sys.executable,str(R/'verify_initialization.py'),str(out/'manifest.json')],env=environment(),capture_output=True,text=True)
                (out/'initialization_check.log').write_text(check.stdout+check.stderr)
                assert check.returncode==0,'初始Master/网络逐位校验失败'
                v['trained_model']=str(out/'model')
                # 首次相机入口分别验证Full、全局低/高ratio、局部低ratio全轨迹。
                for scope,ratio in [('full',1.),('global',.01),('global',.7),('view_ratio',.01)]:
                    item=dict(scope=scope,ratio=ratio,kind='validation',state='pending')
                    v['evaluations'].append(item);p.save(state)
                    evaluate(v,item,state,root/('eval_'+scope+'_'+str(ratio)))
                    assert item['state']=='complete'
                validated.add(kind);state['validated_modes']=sorted(validated)
                state.setdefault('validation_history',[]).append(v);state.pop('active_validation');p.save(state)
                print('VALIDATED',kind,flush=True)
            if job['state'] in ['pending','retry_pending']:
                state['phase']='training';state['active_job']=job['id'];p.save(state)
                try:out=p.launch_training(job,state,R/'runs'/job['id'])
                except RuntimeError:
                    failure=Path(job['attempt'])/'model/lod_health_failure.json'
                    if not failure.exists():raise
                    job.update(state='quality_failed',failure=p.read(failure));p.save(state)
                    print('HEALTH_STOP',job['id'],job['failure'],flush=True);continue
                if out is None:continue
                verify_training(out,job['target_steps'])
                job.update(state='trained',trained_model=str(out/'model'),evaluations=evaluations());p.save(state)
                print('TRAINED',job['id'],job['target_steps'],flush=True)
            state['phase']='evaluating';p.save(state)
            for item in job['evaluations']:
                if item['state'] in ['complete','skipped_oom']:continue
                assert item['state'] in ['pending','retry_pending']
                evaluate(job,item,state,R/'evaluation'/job['id']/(item['scope']+'_'+str(item['ratio'])))
            job['state']='complete' if all(x['state']=='complete' for x in job['evaluations']) else 'partial_evaluation'
            p.save(state);print('COMPLETE',job['id'],flush=True)
        verify_inputs();state['phase']='finished';p.save(state)
        lines=['# 新联合方案批量实验最终状态','','每组按Full全轨迹PSNR选择seed0三次中的最佳checkpoint；其他指标绑定同次。',
            'Fine步数等于原online Mapping；Full+全局LoD+局部LoD，1:1:1，一次Adam；Master更新、拓扑固定。',
            '训练六档ratio与五档未见ratio分别做全局、名义Anchor局部严格预算评价，另评Full。双目GT使用原生矫正，mono沿用尺度校正。',
            '不保存渲染图片，不以评价指标挑训练终点。逐ratio PSNR/SSIM/LPIPS/ATE/N/K见LOD_METRICS.csv。',
            '',str(Counter(j['state'] for j in state['jobs']))]
        (R/'reports/JOINT_BATCH_REPORT.md').write_text('\n'.join(lines)+'\n')
        p.write(R/'FINISHED.json',dict(passed=all(j['state']=='complete' for j in state['jobs']),counts=state['counts']))
    except Exception as error:
        state.update(phase='blocked',error=repr(error));p.save(state);raise


if __name__=='__main__':main()
