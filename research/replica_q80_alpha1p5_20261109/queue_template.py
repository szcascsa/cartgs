"""只在显式授权文件存在且执行start时运行；无常驻监控/自动启动。"""
from pathlib import Path
import argparse
import csv
import fcntl
import hashlib
import json
import math
import os
import resource
import shutil
import subprocess
import sys
import time
from oom_retry import classify_oom, record_attempt, schedule_failure, next_attempt

R = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def process_identity(pid):
    fields = (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()
    return dict(pid=pid, start_ticks=fields[19], boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())


def alive(value):
    try:
        return process_identity(value['pid']) == {k:value[k] for k in ['pid','start_ticks','boot_id']}
    except (FileNotFoundError, ProcessLookupError):
        return False


def summary(jobs):
    counts = {}
    for job in jobs:
        counts[job['state']] = counts.get(job['state'], 0) + 1
    value = dict(counts=counts, total=len(jobs), started=sum(job['state'] != 'held' for job in jobs),
                 auto_start=False)
    write(R / 'STATUS.json', value)
    rows = [dict(id=j['id'], scene=j['scene'], mode=j['mode'], prune=j['prune'], seed=j['seed'], repeat=j.get('repeat', 1),
                 dataset=j.get('dataset'), state=j['state'], error=j.get('error', ''),
                 oom_retry_count=j.get('oom_retry_count',0),attempts_finished=len(j.get('attempt_history',[])),
                 attempt_output=j['output'], **j.get('result', {})) for j in jobs]
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with (R / 'reports/EXPERIMENTS.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, keys)
        writer.writeheader(); writer.writerows(rows)
    return value


def verify(job):
    model = Path(job['output']) / 'model'
    shutdown, = model.glob('*_shutdown')
    state = read(shutdown / 'ply/real_lod_state.json')
    stages = read(shutdown / 'ply/mapping_stage_state.json')
    assert state['fine_skipped_by_batch'] and state['lod_steps'] == state['optimizer_commits'] == 0
    assert state['importance_refreshes'] == state['full_training_renders'] == 0
    assert state['online_full_steps'] > 0 and state['tail_full_steps'] > 0
    for phase in ['online', 'tail']:
        assert stages[phase]['lod_renders'] == stages[phase]['selector_commits'] == 0
    anchor = read(next((shutdown / 'ply').rglob('anchor_selector_state.json')))
    assert anchor['anchor_only_pruning'] == job['prune']
    assert anchor['gaussian_count'] > 0 and anchor['anchor_count'] > 0
    assert math.isfinite(anchor['growth_tau0']) and anchor['growth_tau0']>0 and anchor['growth_alpha']==1.5
    assert anchor['growth_threshold_mode']=='quantile' and anchor['growth_quantile']==.8 and anchor['growth_strict_upper']
    training_log=(Path(job['output'])/'train.log').read_text()
    assert '[Anchor Growth Policy] mode=quantile tau0=0 quantile=0.8 alpha=1.5 strict_upper=1' in training_log
    calibrations=[line for line in training_log.splitlines() if line.startswith('[Explicit Anchor Calibration]')]
    assert len(calibrations)==1 and 'quantile=0.8' in calibrations[0] and 'alpha=1.5' in calibrations[0]
    growth_lines=[line for line in training_log.splitlines() if line.startswith('[Explicit Anchor Growing]')]
    assert growth_lines and all('alpha=1.5 strict_upper=1' in line for line in growth_lines)
    thresholds={line.split('tau0=',1)[1].split()[0] for line in growth_lines}
    assert len(thresholds)==1
    logged_threshold=float(next(iter(thresholds)))
    assert math.isclose(logged_threshold,anchor['growth_tau0'],rel_tol=1e-5)
    assert all(item['adam_step'] == 0 for item in anchor['compensation_feature_state'])
    assert all((model / name).is_file() for name in ['CameraTrajectory_TUM.txt', 'TrackingTime.txt'])
    tracked = len((model / 'TrackingTime.txt').read_text().splitlines())
    assert tracked == job['frames'], (tracked, job['frames'])
    pictures = [p for p in model.rglob('*') if p.suffix.lower() in ['.png', '.jpg', '.jpeg']]
    assert not pictures, 'Unexpected saved images'
    def metrics(name):
        rows = [line.split() for line in (shutdown / name).read_text().splitlines()
                if line.strip() and not line.startswith('#')]
        values = [float(row[1]) for row in rows]
        assert values and all(math.isfinite(x) for x in values)
        return sum(values) / len(values)
    write(shutdown/'ply/checkpoint_experiment.json',dict(experiment='replica_q80_alpha1p5_20261109',
          manual_threshold=0,quantile=.8,calibrated_threshold=anchor['growth_tau0'],alpha=1.5,strict_upper=True,
          baseline_threshold_mode='q90',baseline_alpha=2,
          full_slam=True,normal_tail=True,lod_fine=False,
          source_change=read(R/'reports/GROWTH_THRESHOLD_CHANGE.json')))
    return dict(online_steps=state['online_full_steps'], tail_steps=state['tail_full_steps'],
       fine_steps=0, gaussian_count=anchor['gaussian_count'], anchor_count=anchor['anchor_count'],
       keyframe_psnr=metrics('psnr.txt'), keyframe_ssim=metrics('dssim.txt'),
       checkpoint=str(shutdown / 'ply'), frames=tracked, saved_images=0)


def start():
    assert read(R / 'START_AUTHORIZATION.json')['approved_by_user'] is True
    assert not (R / 'HOLD').exists(), 'Held by user: do not start until explicitly released'
    assert read(R / 'BUILD_READY.json')['passed'], 'Build the batch runtime first'
    assert read(R / 'NO_TRAIN_RUNTIME_VERIFICATION.json')['passed'], 'Validate the no-training runtime gate first'
    lock = (R / 'queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    jobs = read(R / 'QUEUE.json')
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', QT_QPA_PLATFORM='offscreen',
               PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
    env['LD_LIBRARY_PATH'] = str(R / 'runtime/lib') + ':' + env.get('LD_LIBRARY_PATH', '')
    env.pop('LD_PRELOAD', None)
    for key in list(env):
        if key.startswith('CARTGS_'):
            env.pop(key)
    for job in jobs:
        if job['state'] == 'running':
            assert not alive(job['process']), 'A previous native child is still alive; do not duplicate it'
            job.update(state='interrupted', error='Previous controller exited; native return code unknown; preserve attempt')
        if job['state']=='failed' and not job.get('attempt_history'):
            previous=read(Path(job['output'])/'exit.json')
            oom=classify_oom(job,previous)
            record_attempt(job,previous,oom)
            schedule_failure(job,oom)
            write(R/'QUEUE.json',jobs);summary(jobs)
        while job['state'] in ['held','retry_pending']:
            if (R/'HOLD').exists():
                write(R/'QUEUE.json',jobs);summary(jobs);lock.close();return
            if shutil.disk_usage(R).free<5*1024**3:
                write(R/'RESOURCE_HOLD.json',dict(reason='Less than 5 GiB free; stop before original/retry; no deletion'))
                write(R/'QUEUE.json',jobs);summary(jobs);lock.close();return
            assert all(sha(Path(name))==value for name,value in job['input_sha256'].items()),job['id']
            if job['state']=='retry_pending':
                next_attempt(job)
            folder=Path(job['output']);folder.mkdir(parents=True,exist_ok=False)
            write(folder/'launch.json',job)
            start_time=time.monotonic()
            with (folder/'train.log').open('w') as log:
                child=subprocess.Popen(job['command'],cwd=R/'runtime',env=dict(env,**job['environment']),
                    stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                job.update(state='running',process=process_identity(child.pid))
                write(R/'QUEUE.json',jobs);summary(jobs)
                while child.poll() is None:
                    maps=Path(f'/proc/{child.pid}/maps')
                    if maps.exists():
                        content=maps.read_text()
                        if 'libgaussian_mapper.so' in content:
                            assert str(R/'runtime/lib/libgaussian_mapper.so') in content,'Wrong Mapper loaded'
                            if not (folder/'runtime_maps.txt').exists():(folder/'runtime_maps.txt').write_text(content)
                    time.sleep(1)
                code=child.returncode
            seconds=time.monotonic()-start_time
            exit_info=dict(returncode=code,seconds=seconds)
            write(folder/'exit.json',exit_info)
            try:
                assert code==0,f'Native exit code {code}'
                result=verify(job);result['seconds']=seconds
                job.update(state='complete',result=result,error='')
                write(folder/'verified.json',dict(passed=True,**result))
                record_attempt(job,exit_info,None)
            except Exception as error:
                job.update(state='failed',error=str(error))
                oom=classify_oom(job,exit_info)
                record_attempt(job,exit_info,oom)
                schedule_failure(job,oom)
                write(folder/'failure.json',dict(error=str(error),oom=oom,
                      oom_retry_count=job.get('oom_retry_count',0),next_state=job['state']))
            write(R/'QUEUE.json',jobs);summary(jobs)
        write(R / 'QUEUE.json', jobs); summary(jobs)
        if job['state']=='complete':
            entry=Path(job['command'][0]).name
            if not (R / ('SMOKE_'+entry+'.json')).exists():
                with (R / 'logs' / ('smoke_'+entry+'.log')).open('w') as smoke_log:
                    code=subprocess.call([sys.executable,'-B','-u',str(R/'smoke_eval.py'),job['id']],
                        cwd=R,env=env,stdout=smoke_log,stderr=subprocess.STDOUT)
                assert code==0, 'New entry full-trajectory evaluation failed: '+entry
    print(json.dumps(summary(jobs)), flush=True)
    lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['status', 'verify-inputs', 'start'])
    args = parser.parse_args()
    if args.action == 'start':
        start()
    else:
        jobs = read(R / 'QUEUE.json')
        if args.action == 'verify-inputs':
            assert len(jobs) == 16 and len({j['id'] for j in jobs}) == 16
            assert all(sha(Path(name)) == value for job in jobs for name, value in job['input_sha256'].items())
        print(json.dumps(summary(jobs)), flush=True)
