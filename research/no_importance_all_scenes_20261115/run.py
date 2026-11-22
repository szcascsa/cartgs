"""D组32场景调度；独立目录、持久化OOM计数、逐组初始化与曝光核验。"""
from pathlib import Path
import collections
import concurrent.futures
import copy
import csv
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
from prepare import R,B,V,read,write,sha
from oom_retry import classify_oom

TRAIN=[.01,.05,.1,.3,.5,.7]
UNSEEN=[.03,.08,.2,.4,.6]
LOCK=threading.RLock()
TERMINAL={'complete','skipped_zero_selection','skipped_oom','failed','partial_evaluation'}
p.R=R;p.TRAIN=TRAIN;p.UNSEEN=UNSEEN


def update(item,**values):
    with LOCK:item.update(values)


def environment():
    env=dict(os.environ,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',MPLBACKEND='Agg',QT_QPA_PLATFORM='offscreen',
             PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',CARTGS_AUDIT_SEED='0')
    if os.environ.get('CARTGS_CLUSTER_WORKER','original')!='original':
        env['TORCH_HOME']='/home/prmi/.cache/torch'
    env['LD_LIBRARY_PATH']=':'.join([str(R/'runtime/lib'),
        *([str(R/'cluster_deps/lib')] if (R/'cluster_deps/lib').exists() else []),
        '/home/prmi/zwj/experiments/real_lod_full_joint_20261103/source/third_party/ORB-SLAM3/lib',
        '/home/prmi/zwj/mycartgs/third_party/install/opencv-cuda118-sm86-gcc9/lib',
        '/home/prmi/zwj/mycartgs/third_party/install/jsoncpp-1.9.5/lib',
        '/home/prmi/zwj/mycartgs/third_party/libtorch-cu118/libtorch/lib','/usr/local/cuda-11.8/lib64'])
    for k in list(env):
        if k.startswith('CARTGS_LOD_DIAGNOSTIC_') or k in ['GEOMETRY_HEAD_VARIANT','LD_PRELOAD','CARTGS_BATCH_FULL_TAIL_ONLY']:env.pop(k)
    return env


def save(state):
    with LOCK:
        p.save(state)
        rows=[]
        for j in state['jobs']:
            for v in j['evaluations']:
                if v['state']!='complete':continue
                old=next(t for t in j['reference_evaluations'] if t['scope']==v['scope'] and t['ratio']==v['ratio'])
                a,d=old['result'],v['result']
                row=dict(id=j['id'],dataset=j['dataset'],scene=j['scene'],mode=j['mode'],scope=v['scope'],ratio=v['ratio'],
                    reference_model=j['reference_model'],model=j.get('trained_model'),evaluated_frames=d['evaluated_frames'])
                for key in ['psnr','ssim','lpips','ate_rmse','actual_K_mean']:
                    if a.get(key) is not None and d.get(key) is not None:row.update({key+'_A':a[key],key+'_D':d[key],'delta_'+key:d[key]-a[key]})
                rows.append(row)
        if rows:
            file=R/'reports/A_VS_D_METRICS.csv';keys=list(dict.fromkeys(k for row in rows for k in row))
            with file.with_suffix('.tmp').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
            file.with_suffix('.tmp').replace(file)


def manifest(job,output):
    full=Path(job['full_job']['result']['checkpoint']);cloud,=(full/'point_cloud').glob('iteration_*/point_cloud.ply')
    value=dict(full_ply=str(full),point_cloud=str(cloud),anchor_state=str(cloud.parent/'anchor_selector_state.json'),
        camera_config=job['camera_config'],mapping_config=job['mapping_config'],selector_config=str(R/'configs/no_importance.yaml'),
        initial_iteration=int(full.parent.name.split('_')[0]),output=str(output),mode=job['mode'],dataset=job['dataset'],
        dataset_root=job['full_evaluation']['data'],optimizer_protocol='fresh Adam/RNG; not complete SLAM resume')
    value['input_sha256']={str(n):sha(n) for n in [cloud,cloud.parent/'anchor_selector_state.json',cloud.parent/'anchor_pruning_state.pt',
        full/'selector.pt',full/'cameras.json',full/'real_lod_state.json',Path(value['camera_config']),Path(value['mapping_config']),R/'configs/no_importance.yaml']}
    return value


def process(command,directory,item,state,evaluation=False):
    while True:
        assert shutil.disk_usage(R).free>5*2**30,'剩余磁盘不足5GiB；暂停，不清理模型'
        if (R/'HOLD').exists():raise RuntimeError('HOLD存在，保留队列')
        free=int(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
        if free>=4096:break
        update(item,waiting_gpu_memory_mib=free);save(state);time.sleep(10)
    number=len(item.get('attempt_history',[]))+1;out=directory/f'attempt_{number:02d}';out.mkdir(parents=True,exist_ok=False)
    argv=command(out);write(out/'launch.json',dict(command=argv,started_unix=time.time()))
    logname='evaluate.log' if evaluation else 'train.log';start=time.monotonic();last_bytes=-1;last_progress=start
    with (out/logname).open('w') as f:
        child=subprocess.Popen(argv,cwd=R/'source',env=environment(),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        update(item,state='running',process=p.identity(child.pid),attempt=str(out));save(state)
        while child.poll() is None:
            update(item,observed_at=time.time(),log_bytes=(out/logname).stat().st_size)
            trace=out/'model/real_lod_steps.csv'
            if trace.exists():
                update(item,trace_bytes=trace.stat().st_size)
                with trace.open('rb') as stream:
                    stream.seek(max(0,trace.stat().st_size-1024));lines=stream.read().decode(errors='replace').splitlines()
                update(item,step=next((int(s.split(',')[0]) for s in reversed(lines) if s.split(',')[0].isdigit()),0))
            progress=item.get('trace_bytes',item['log_bytes'])
            if progress!=last_bytes:last_bytes=progress;last_progress=time.monotonic()
            update(item,no_progress_seconds=round(time.monotonic()-last_progress,1))
            maps=Path(f'/proc/{child.pid}/maps')
            if not evaluation and maps.exists():
                text=maps.read_text()
                if 'libgaussian_mapper.so' in text:
                    assert str((R/'runtime/lib/libgaussian_mapper.so').resolve()) in text,'加载错误Mapper'
                    if not (out/'runtime_maps.txt').exists():(out/'runtime_maps.txt').write_text(text)
            save(state);time.sleep(5)
    exit_info=dict(returncode=child.returncode,seconds=time.monotonic()-start);write(out/'exit.json',exit_info)
    if evaluation:(out/'train.log').symlink_to(logname)
    healthfile=out/'model/lod_health_failure.json';health=read(healthfile) if healthfile.exists() else None
    oom=classify_oom(dict(output=str(out),process=item['process']),exit_info)
    with LOCK:item.setdefault('attempt_history',[]).append(dict(output=str(out),process=item['process'],oom=oom,health=health,**exit_info))
    save(state)
    if child.returncode==0:return out
    if oom:
        update(item,oom_failures=item.get('oom_failures',0)+1)
        update(item,state='retry_pending' if item['oom_failures']<=2 else 'skipped_oom',error='有明确OOM证据；最多额外2次')
        save(state);return None
    if health and health['error'].endswith('consecutive_zero_selection_with_candidates'):
        update(item,state='skipped_zero_selection',error=health['error']);save(state);return None
    update(item,state='failed',error=dict(health=health,returncode=child.returncode,log=str(out/logname)));save(state)
    raise RuntimeError(str(item['error']))


def verify_training(out,target,reference=None):
    stage=p.verify_training(out,target);v=read(stage/'real_lod_state.json')
    assert v['view_budget_conditioning']==2 and v['view_budget_li_weight']==0 and not v['master_frozen_in_lod']
    assert v['view_budget_li_full_renders']==v['view_budget_li_backwards']==0
    assert v['view_budget_enabled'] and not v['view_budget_view_residual']
    assert v['view_budget_training_renders']==v['view_budget_training_backwards']==target
    assert v['total_training_renders']==3*target
    with (out/'model/real_lod_steps.csv').open() as f:steps=list(csv.DictReader(f))
    with (out/'model/view_budget_steps.jsonl').open() as f:local=[json.loads(s) for s in f]
    assert len(steps)==len(local)==target
    assert all(not s['li_supervised'] and s['li_weight']==0 and s['li_loss']==0 for s in local)
    config=(stage/'selector_config.yaml').read_text()
    assert 'Selector.gi_loss_weight: 0' in config and 'LoDHealth.zero_selection_only: 1' in config
    if reference is not None:
        reference=Path(reference)
        assert (out/'model/training_keyframes.csv').read_bytes()==(reference/'training_keyframes.csv').read_bytes()
        old=read(next(reference.glob('*_shutdown/ply/real_lod_state.json')))
        assert old['ratio_steps']==v['ratio_steps'] and old['importance_full_renders']==v['importance_full_renders']
    write(out/'D_VERIFIED.json',dict(passed=True,steps=target,gi_weight=0,li_weight=0,li_capture=0,li_vjp=0,
        three_branch_renders=3*target,ratio_steps=v['ratio_steps'],importance_renders=v['importance_full_renders'],
        zero_selection_only=True,selection_exposure_global=sum(int(s['K']) for s in steps),
        selection_exposure_local=sum(s['K'] for s in local),paired_camera_ratio_sequence=reference is not None))
    return stage


def train(job,state,folder,smoke=0):
    while job['state'] in ['pending','retry_pending']:
        def command(out):
            model=out/'model';model.mkdir();value=manifest(job,model);write(out/'manifest.json',value)
            for n in ['CameraTrajectory_TUM.txt','TrackingTime.txt']:shutil.copy2(Path(job['full_evaluation']['model'])/n,model/n)
            return [str(R/'runtime/bin/lod_from_full'),str(out/'manifest.json')]+([str(smoke)] if smoke else [])
        out=process(command,folder,job,state)
        if out is None:continue
        verify_training(out,smoke or job['target_steps'],None if smoke else job['reference_model'])
        with (out/'initialization_check.log').open('w') as log:
            q=subprocess.run([sys.executable,str(R/'verify_initialization.py'),str(out/'manifest.json')],env=environment(),stdout=log,stderr=subprocess.STDOUT)
        assert q.returncode==0,'Full初始状态逐bit核验失败'
        update(job,state='trained',trained_model=str(out/'model'));save(state);return out
    return None


def cases():
    return [dict(scope=s,ratio=r,kind='train' if r in TRAIN else 'unseen',state='pending',oom_failures=0)
        for s in ['global','view_ratio'] for r in sorted(TRAIN+UNSEEN)]+[dict(scope='full',ratio=1.,kind='full',state='pending',oom_failures=0)]


def evaluate(job,item,state,root,frame_set=None):
    model=Path(job['trained_model']);ply,=model.glob('*_shutdown/ply')
    def command(out):
        cmd=[sys.executable,str(R/'eval_joint_entry.py'),str(R/'source/python/run.py'),str(model),job['full_evaluation']['data'],
            '--shutdown-model',ply.parent.name,'--output_path',str(out),'--no-save-images']
        if item['scope']!='full':cmd+=['--selector',str(ply/'selector.pt'),'--selector-ratio',str(item['ratio']),'--selector-config',str(R/'configs/no_importance.yaml')]
        if item['scope']=='view_ratio':cmd+=['--joint-local-ratio']
        if job['mode']=='mono':cmd+=['--correct_scale']
        if frame_set is not None:cmd+=['--evaluation-frame-set',str(frame_set)]
        # All datasets use CaRtGS original RGB; stereo keeps its original training rectification only.
        return cmd
    while item['state'] in ['pending','retry_pending']:
        out=process(command,root,item,state,evaluation=True)
        if out is None:continue
        finish_evaluation(job,item,state,out,frame_set)


def finish_evaluation(job,item,state,out,frame_set=None):
        v=read(out/'metric_summary.json');expected=len(read(frame_set)['gt_indices']) if frame_set else job['full_evaluation']['result']['evaluated_frames']
        assert v['evaluated_frames']==expected and all(math.isfinite(v[k]) for k in ['psnr','ssim','lpips','ate_rmse'])
        assert read(out/'image_preprocessing.json')['protocol']=='original_rgb'
        if v.get('gaussian_count') is None:v['gaussian_count']=job['full_job']['result']['gaussian_count']
        v['anchor_count']=job['full_job']['result']['anchor_count']
        if item['scope']=='full':v.update(actual_K_mean=v['gaussian_count'],actual_K_min=v['gaussian_count'],actual_K_max=v['gaussian_count'],target_K=v['gaussian_count'])
        assert v['gi_loss_weight']==v['li_weight']==0
        assert not any(q.suffix.lower() in ['.png','.jpg','.jpeg','.bmp'] for q in out.rglob('*'))
        # Keep paired per-frame coverage explicit; quality never selects a frame or checkpoint.
        reference=next((e for e in job['reference_evaluations'] if e['scope']==item['scope'] and e['ratio']==item['ratio']),None)
        if frame_set is None and reference:
            old=Path(reference['result']['output'])/'per_frame_metrics.csv'
            if old.exists():
                with old.open() as f:a=list(csv.DictReader(f))
                with (out/'per_frame_metrics.csv').open() as f:d=list(csv.DictReader(f))
                keys=['rgb_index','trajectory_timestamp','rgb_timestamp']
                assert [[t[k] for k in keys] for t in a]==[[t[k] for k in keys] for t in d]
        v.update(scope=item['scope'],output=str(out),saved_images=0,evaluation_protocol='cartgs_original_rgb_full_trajectory')
        update(item,state='complete',result=v);save(state)


def work(job,state):
    try:
        if job['state'] in ['pending','retry_pending']:
            if train(job,state,R/'runs'/job['id']) is None:return
            update(job,evaluations=cases());save(state)
        evaluate_job(job,state)
    except Exception as error:
        update(job,state='failed',error=repr(error));save(state);print('JOB_FAILED',job['id'],repr(error),flush=True)


def evaluate_job(job,state):
    for item in job['evaluations']:
        if item['state'] in ['complete','skipped_oom','failed']:continue
        assert item['state'] in ['pending','retry_pending'],'评价现场需人工核查后恢复'
        evaluate(job,item,state,R/'evaluation'/job['id']/(item['scope']+'_'+str(item['ratio'])))
    update(job,state='complete' if all(t['state']=='complete' for t in job['evaluations']) else 'partial_evaluation');save(state)
    print('JOB_FINISHED',job['id'],job['state'],flush=True)


def training_work(job,state):
    try:
        if train(job,state,R/'runs'/job['id']) is None:return False
        update(job,evaluations=cases());save(state);return True
    except Exception as error:
        update(job,state='failed',error=repr(error));save(state);print('JOB_FAILED',job['id'],repr(error),flush=True)
        return False


def evaluation_work(job,state):
    try:evaluate_job(job,state)
    except Exception as error:
        update(job,state='failed',error=repr(error));save(state);print('JOB_FAILED',job['id'],repr(error),flush=True)


def pipelined_work(assigned,state):
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as evaluations, concurrent.futures.ThreadPoolExecutor(max_workers=2) as training:
        eval_futures=[evaluations.submit(evaluation_work,j,state) for j in assigned if j['state']=='trained']
        train_futures={training.submit(training_work,j,state):j for j in assigned if j['state'] in ['pending','retry_pending']}
        for future in concurrent.futures.as_completed(train_futures):
            if future.result():eval_futures.append(evaluations.submit(evaluation_work,train_futures[future],state))
        for future in concurrent.futures.as_completed(eval_futures):future.result()


def main():
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    lock=(R/'controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert read(R/'reports/VALIDATION.json')['passed']
    state=read(R/'LOD_PIPELINE.json')
    worker=os.environ.get('CARTGS_CLUSTER_WORKER','original')
    overlap=os.environ.get('CARTGS_PIPELINE_OVERLAP')=='1'
    assert not overlap or worker in ['cloud_a','cloud_b'],'重叠调度只用于AutoDL节点'
    if overlap:
        session_lock=(R/'cloud_session.lock').open('a');fcntl.flock(session_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if worker!='original':assert read(R/'reports/CLOUD_VALIDATION.json')['passed']
    assigned=[j for j in state['jobs'] if j.get('assigned_worker','original')==worker]
    assert len(state['jobs'])==32 and assigned
    assert all(j['state'] in TERMINAL|{'pending','retry_pending','trained'} for j in assigned),'旧running现场需先核查，禁止重复训练'
    write(R/'LOD_CONTROLLER.json',p.identity(os.getpid()));state.update(phase='running',workers=4 if overlap else 2);save(state)
    if overlap:
        state.update(scheduling='independent_train_and_evaluation_pools',train_workers=2,evaluation_workers=2);save(state)
        write(R/'reports/CLOUD_SESSION.json',dict(worker=worker,phase='training',controller=p.identity(os.getpid()),started_unix=time.time(),scheduling=state['scheduling']))
        pipelined_work(assigned,state)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(work,j,state) for j in assigned if j['state'] not in TERMINAL]
            for f in concurrent.futures.as_completed(futures):f.result()
    state.update(phase='local_finished',worker=worker);save(state)
    if overlap:write(R/'reports/CLOUD_SESSION.json',dict(worker=worker,phase='local_finished',finished_unix=time.time()))
    if worker=='original' and all(j['state'] in TERMINAL for j in state['jobs']):
        subprocess.run([sys.executable,str(R/'finish.py')],cwd=R,env=environment(),check=True)


if __name__=='__main__':main()
