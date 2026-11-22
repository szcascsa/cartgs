"""6组VECtor双网络真实视角＋LI独立队列，持久化重试、只保留最终模型。"""
from pathlib import Path
import collections
import fcntl
import json
import math
import os
import resource
import shutil
import subprocess
import sys
import threading
import time
import pipeline as p

R=Path(__file__).resolve().parent
p.R=R;p.TRAIN=[.01,.05,.1,.3,.5,.7];p.UNSEEN=[.03,.08,.2,.4,.6]
p.config=lambda:R/'configs/dual_view_li.yaml'
original_verify=p.verify_training
TERMINAL={'complete','skipped_oom','skipped_health','partial_evaluation','skipped_by_user'}


def environment():
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MPLBACKEND='Agg',QT_QPA_PLATFORM='offscreen',
             PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',CARTGS_AUDIT_SEED='0')
    env['LD_LIBRARY_PATH']=':'.join([str(R/'runtime/lib'),
        '/home/prmi/zwj/experiments/real_lod_full_joint_20261103/source/third_party/ORB-SLAM3/lib',
        '/home/prmi/zwj/mycartgs/third_party/install/opencv-cuda118-sm86-gcc9/lib',
        '/home/prmi/zwj/mycartgs/third_party/install/jsoncpp-1.9.5/lib',
        '/home/prmi/zwj/mycartgs/third_party/libtorch-cu118/libtorch/lib','/usr/local/cuda-11.8/lib64'])
    for k in list(env):
        if k.startswith('CARTGS_LOD_DIAGNOSTIC_') or k in ['GEOMETRY_HEAD_VARIANT','LD_PRELOAD','CARTGS_BATCH_FULL_TAIL_ONLY']:env.pop(k)
    return env


def manifest(job,output):
    full=Path(job['full_job']['result']['checkpoint']);cloud,=(full/'point_cloud').glob('iteration_*/point_cloud.ply')
    value=dict(full_ply=str(full),point_cloud=str(cloud),anchor_state=str(cloud.parent/'anchor_selector_state.json'),
        camera_config=job['camera_config'],mapping_config=job['mapping_config'],selector_config=str(p.config()),
        initial_iteration=int(full.parent.name.split('_')[0]),output=str(output),mode=job['mode'],dataset=job['dataset'],
        dataset_root=job['full_evaluation']['data'],optimizer_protocol='fresh Adam/RNG; not complete SLAM resume')
    value['input_sha256']={str(n):p.sha(n) for n in [cloud,cloud.parent/'anchor_selector_state.json',cloud.parent/'anchor_pruning_state.pt',
        full/'selector.pt',full/'cameras.json',full/'real_lod_state.json',Path(value['camera_config']),Path(value['mapping_config']),p.config()]}
    return value


def discard_initial_copies(model):
    for n in ['converted_selector_init.pt','step0_selector.pt','step0_master.ply']:
        file=model/n;file.resolve().relative_to(R.resolve())
        if file.is_file():file.unlink()


def health_decision(item,max_retries):
    return item.get('health_failures',0)<=max_retries


def run_process(argv,directory,item,state,evaluation=False):
    while True:
        if (R/'HOLD').exists():raise RuntimeError('用户HOLD存在，保留队列')
        assert shutil.disk_usage(R).free>5*2**30,'剩余磁盘不足5GiB，保留模型并暂停，不清理历史数据'
        free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
        if free>=6144:break
        state.update(phase='waiting_gpu_memory',gpu_free_mib=free);p.save(state);time.sleep(30)
    number=len(item.get('attempt_history',[]))+1
    out=directory/f'attempt_{number:02d}';out.mkdir(parents=True,exist_ok=False)
    command=argv(out)
    guard=None
    if item.get('id',state.get('active_job'))=='VECtor/corridors-dolly/stereo':
        import memory_guard
        guard=p.read(R/'configs/corridors_stereo_memory.json')
        memory_before=memory_guard.snapshot(guard)
        command=[sys.executable,str(R/'memory_guard.py'),str(R/'configs/corridors_stereo_memory.json'),str(out),*command]
    p.write(out/'launch.json',dict(command=command,started_unix=time.time()))
    started=time.monotonic();log_name='evaluate.log' if evaluation else 'train.log'
    with (out/log_name).open('w') as log:
        child=subprocess.Popen(command,cwd=R/'source',env=environment(),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        item.update(state='running',process=p.identity(child.pid),attempt=str(out));p.save(state)
        if guard:
            stop=threading.Event()
            watcher=threading.Thread(target=memory_guard.watch,args=(child,item['process'],guard,out,stop),daemon=True)
            watcher.start()
        while child.poll() is None:
            item.update(observed_at=time.time(),log_bytes=(out/log_name).stat().st_size)
            trace=out/'model/real_lod_steps.csv'
            if trace.exists():item['trace_bytes']=trace.stat().st_size
            maps=Path(f'/proc/{child.pid}/maps')
            if maps.exists():
                content=maps.read_text()
                if 'libgaussian_mapper.so' in content:
                    assert str(R/'runtime/lib/libgaussian_mapper.so') in content
                    if not (out/'runtime_maps.txt').exists():(out/'runtime_maps.txt').write_text(content)
            p.save(state);time.sleep(5)
    if guard:
        stop.set();watcher.join()
    info=dict(returncode=child.returncode,seconds=time.monotonic()-started)
    p.write(out/'exit.json',info)
    if evaluation:(out/'train.log').symlink_to(log_name)
    failure=out/'model/lod_health_failure.json'
    health=p.read(failure) if failure.exists() else None
    oom=None
    if guard:
        memory_after=memory_guard.snapshot(guard)
        p.write(out/'memory_usage.json',dict(before=memory_before,after=memory_after))
        if child.returncode in [-9,137] and memory_after['oom_kill']>memory_before['oom_kill']:
            assert p.read(out/'memory_guard.json')['pid']==item['process']['pid']
            oom=dict(kind='cgroup_memory_oom',evidence=[dict(cgroup=guard['cgroup'],before=memory_before,after=memory_after)])
        if (out/'memory_reserve_stop.json').exists():
            oom=dict(kind='system_memory_reserve_stop',evidence=[p.read(out/'memory_reserve_stop.json')])
    if oom is None:oom=p.classify_oom(dict(output=str(out),process=item['process']),info)
    record=dict(output=str(out),process=item['process'],oom=oom,health=health,**info)
    item.setdefault('attempt_history',[]).append(record);p.save(state)
    if child.returncode==0:return out
    if health:
        item['health_failures']=item.get('health_failures',0)+1
        retry=health_decision(item,state['protocol']['max_health_retries'])
        item.update(state='retry_pending' if retry else 'skipped_health',error=health['error'])
        discard_initial_copies(out/'model');p.save(state)
        print('HEALTH_RETRY' if retry else 'HEALTH_RETRIES_EXHAUSTED',directory,health['error'],flush=True)
        return None
    if oom:
        item['oom_failures']=item.get('oom_failures',0)+1
        item.update(state='retry_pending' if item['oom_failures']<=2 else 'skipped_oom',error='明确OOM，重试计数跨重启保留')
        discard_initial_copies(out/'model');p.save(state);return None
    item.update(state='failed',error=f'非健康停止／非OOM退出{child.returncode}，保留{out}');p.save(state)
    raise RuntimeError(item['error'])


p.environment=environment;p.manifest=manifest;p.run_process=run_process


def verify_training(out,target):
    ply=original_verify(out,target);stage=p.read(ply/'real_lod_state.json')
    assert stage['view_budget_conditioning']==2 and abs(stage['view_budget_li_weight']-.1)<1e-6
    assert stage['view_budget_enabled'] and not stage['view_budget_view_residual']
    assert stage['view_budget_li_full_renders']==target
    assert stage['view_budget_li_backwards']<=stage['view_budget_training_backwards']<=target
    assert stage['total_training_renders']==stage['full_training_renders']+stage['lod_training_renders']+stage['view_budget_training_renders']
    assert not stage['master_frozen_in_lod'] and not stage['maturity_gate'] and not stage['maturity_loss']
    return ply


def evaluate(job,item,state,root):
    model=Path(job['trained_model']);ply,=model.glob('*_shutdown/ply')
    def command(out):
        cmd=[sys.executable,str(R/'eval_joint_entry.py'),str(R/'source/python/run.py'),str(model),job['full_evaluation']['data'],
             '--shutdown-model',ply.parent.name,'--output_path',str(out),'--no-save-images']
        if item['scope']!='full':cmd+=['--selector',str(ply/'selector.pt'),'--selector-ratio',str(item['ratio']),'--selector-config',str(p.config())]
        if item['scope']=='view_ratio':cmd.append('--joint-local-ratio')
        if job['mode']=='mono':cmd.append('--correct_scale')
        if job['mode']=='stereo':cmd+=['--stereo-camera-config',job['camera_config'],'--stereo-native-maps',str(R/'configs/vector_stereo_native_maps.yml.gz')]
        return cmd
    while item['state'] in ['pending','retry_pending']:
        out=run_process(command,root,item,state,evaluation=True)
        if out is None:continue
        result=p.read(out/'metric_summary.json')
        assert result['evaluated_frames']==job['full_evaluation']['result']['evaluated_frames']
        assert all(math.isfinite(result[k]) for k in ['psnr','ssim','lpips','ate_rmse'])
        if result.get('gaussian_count') is None:result['gaussian_count']=job['full_job']['result']['gaussian_count']
        result['anchor_count']=job['full_job']['result']['anchor_count']
        if item['scope']=='full':
            result.update(actual_K_mean=result['gaussian_count'],actual_K_min=result['gaussian_count'],actual_K_max=result['gaussian_count'],target_K=result['gaussian_count'])
        assert not any(v.suffix.lower() in ['.png','.jpg','.jpeg','.bmp'] for v in out.rglob('*'))
        result.update(scope=item['scope'],output=str(out),saved_images=0)
        item.update(state='complete',result=result);p.save(state)
        temp=out/'point_cloud.ply'
        if temp.exists():
            temp.resolve().relative_to(R.resolve());p.write(out/'temporary_export_removed.json',dict(bytes=temp.stat().st_size,sha256=p.sha(temp)));temp.unlink()


def initialization_check(out):
    result=subprocess.run([sys.executable,str(R/'verify_initialization.py'),str(out/'manifest.json')],env=environment(),capture_output=True,text=True)
    (out/'initialization_check.log').write_text(result.stdout+result.stderr);assert result.returncode==0,'初态逐bit校验失败'


def cases():
    return [dict(scope=s,ratio=r,kind='train' if r in p.TRAIN else 'unseen',state='pending')
            for s in ['global','view_ratio'] for r in sorted(p.TRAIN+p.UNSEEN)]+[dict(scope='full',ratio=1.,kind='full',state='pending')]


def train(job,state,folder,smoke=0):
    while job['state'] in ['pending','retry_pending']:
        out=p.launch_training(job,state,folder,smoke=smoke)
        if out is None:continue
        verify_training(out,smoke or job['target_steps']);initialization_check(out);discard_initial_copies(out/'model')
        job.update(state='trained',trained_model=str(out/'model'));p.save(state);return out
    return None


def report(state):
    lines=['# 双网络真实视角＋LI：批量结果','',
        'Full赢家按每个数据集／场景／模式的seed0三次全轨迹PSNR选取，only anchor prune=1。',
        'Full＋全局LoD＋局部LoD，LI=0.1；Master可更新、固定拓扑；未见ratio只评价。',
        '每个工作项保留最终模型和全轨迹数值指标，不保存渲染图片；健康重跑与OOM计数见attempt_history。','',
        '|场景／模式|Full起点PSNR|Fine步数|状态|','|---|---:|---:|---|']
    for j in state['jobs']:lines.append(f"|{j['id']}|{j['full_psnr']:.6f}|{j['target_steps']}|{j['state']}|")
    lines+=['','逐ratio PSNR/SSIM/LPIPS/ATE/N/实际K见LOD_METRICS.csv；逐帧指标位于evaluation目录。',
        '未因评价质量回选checkpoint或更改训练终点。最高Full结果的三次轨迹覆盖可能不同，不能称共同帧统计检验。']
    (R/'reports/DUAL_VIEW_LI_BATCH_REPORT.md').write_text('\n'.join(lines)+'\n')


def pause_completed_job(state):
    request = R/'PAUSE_AFTER_JOB.json'
    if not request.exists(): return False
    target = p.read(request)['job']
    if not any(j['id'] == target and j['state'] in TERMINAL for j in state['jobs']): return False
    (R/'HOLD').write_text('用户要求：当前场景评价结束后暂停VECtor，恢复原Replica/TUM队列\n')
    state.update(phase='held', active_job=None, error='')
    p.save(state); report(state)
    print('PAUSED_AFTER_JOB', target, flush=True)
    return True


def main():
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    lock=(R/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    p.write(R/'LOD_CONTROLLER.json',p.identity(os.getpid()))
    state=p.read(R/'LOD_PIPELINE.json')
    try:
        assert p.read(R/'reports/BUILD_HASHES.json')
        assert len(state['jobs'])==6 and all(j['dataset']=='VECtor' for j in state['jobs'])
        state['phase']='running';p.save(state)
        for job in state['jobs']:
            if pause_completed_job(state): return
            if job['state'] in TERMINAL:continue
            assert job['state'] in ['pending','retry_pending','trained'],'中断现场不能覆盖running任务'
            kind=job['dataset']+'/'+job['mode']
            if kind not in state['validated_modes']:
                v=state.get('active_validation') or dict(job,state='pending',attempt_history=[],evaluations=[],health_failures=0,oom_failures=0)
                assert v['dataset']+'/'+v['mode']==kind
                state.update(phase='validating_'+kind,active_validation=v);p.save(state)
                base=R/'validation'/kind
                out=Path(v['attempt']) if v['state']=='trained' else train(v,state,base,smoke=6)
                assert out is not None,'入口验证重试用尽，暂停，不能未验收进入正式任务'
                for scope,ratio in [('full',1.),('global',.01),('global',.03),('global',.7),('view_ratio',.01)]:
                    item=next((e for e in v['evaluations'] if e['scope']==scope and e['ratio']==ratio),None)
                    if item is None:
                        item=dict(scope=scope,ratio=ratio,kind='validation',state='pending');v['evaluations'].append(item);p.save(state)
                    if item['state']=='complete':continue
                    evaluate(v,item,state,base/('eval_'+scope+'_'+str(ratio)));assert item['state']=='complete'
                state['validated_modes'].append(kind);state.setdefault('validation_history',[]).append(v);state.pop('active_validation');p.save(state)
                print('VALIDATED',kind,flush=True)
            if job['state'] in ['pending','retry_pending']:
                state.update(phase='training',active_job=job['id']);p.save(state)
                out=train(job,state,R/'runs'/job['id'])
                if out is None:report(state);continue
                job['evaluations']=cases();p.save(state);print('TRAINED',job['id'],flush=True)
            state.update(phase='evaluating',active_job=job['id']);p.save(state)
            for item in job['evaluations']:
                if item['state'] in ['complete','skipped_oom']:continue
                assert item['state'] in ['pending','retry_pending']
                evaluate(job,item,state,R/'evaluation'/job['id']/(item['scope']+'_'+str(item['ratio'])))
            job['state']='complete' if all(v['state']=='complete' for v in job['evaluations']) else 'partial_evaluation'
            p.save(state);report(state);print('COMPLETE',job['id'],flush=True)
        before=p.read(R/'PROTECTED_BEFORE.json');after={k:{n:p.sha(Path(n)) for n in v} for k,v in before.items()}
        assert after==before,'只读Full或生产文件发生变动';p.write(R/'PROTECTED_AFTER.json',after)
        state.update(phase='finished',active_job=None);p.save(state);report(state)
        p.write(R/'FINISHED.json',dict(passed=all(v['state'] in ['complete','skipped_by_user'] for v in state['jobs']),counts=state['counts'],images_saved=0))
    except Exception as e:
        state.update(phase='held' if (R/'HOLD').exists() else 'blocked',error=repr(e));p.save(state);report(state);raise


if __name__=='__main__':main()
