"""单写者队列、场景级并发，接管既有子进程并保留完整训练序列。"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import csv
import fcntl
import json
import shutil
import subprocess
import sys
import threading
import time


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
GUARD = threading.RLock()


def write(path,value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False))
    temporary.replace(path)


def alive(task):
    path = Path(f'/proc/{task["pid"]}/stat')
    if not path.exists():
        return False
    parts = path.read_text().rpartition(')')[2].split()
    return parts[0]!='Z' and int(parts[19])==int(task['start_ticks']) and task['boot_id']==Path('/proc/sys/kernel/random/boot_id').read_text().strip()


def success_evidence(scene,task,protocol):
    log = Path(task['log']).read_text(errors='replace')
    name = task['name']
    folder = ROOT/'reports'/scene['slug']
    if name in [s['name'] for s in protocol['stages']]:
        stage = next(s for s in protocol['stages'] if s['name']==name)
        output = ROOT/'models'/scene['slug']/name
        assert 'DISTILLATION_COMPLETE '+str(output) in log,'Adopted training has no completion marker'
        rows = list(csv.DictReader((output/'training.csv').open()))
        assert [int(r['step']) for r in rows]==list(range(1,stage['steps']+1))
        assert (output/'best.pt').is_file() and (output/'last.pt').stat().st_mtime>=task['started']
    elif name=='preflight':
        assert 'PREFLIGHT_PASS' in log and json.loads((folder/'PREFLIGHT.json').read_text())['passed']
    elif name=='evaluation':
        assert 'CROSS_EVALUATION_COMPLETE' in log and len(json.loads((folder/'QUALITY_SUMMARY.json').read_text())['summaries'])==88
    elif name=='deployment_validation':
        assert 'DEPLOYMENT_VALIDATION_PASS' in log and json.loads((folder/'DEPLOYMENT_VALIDATION.json').read_text())['passed']
    else:
        raise ValueError('Unknown adopted phase '+name)


def execute_scene(scene,job,state,protocol):
    destination = ROOT/'reports/BATCH_STATUS.json'

    def save():
        write(destination,state)

    def run(name,arguments,output=None):
        existing = [t for t in job['tasks'] if t['name']==name]
        if existing and (existing[-1].get('returncode')==0 or existing[-1].get('adopted_success')):
            return
        if existing and 'ended' not in existing[-1]:
            task = existing[-1]
            while alive(task):
                time.sleep(2)
            success_evidence(scene,task,protocol)
            with GUARD:
                task.update(ended=time.time(),adopted_success=True,
                    completion_evidence='process exited, completion marker, result files and complete CSV budget',
                    exit_code_observed=False)
                save()
            print('ADOPTED_COMPLETE',scene['id'],name,flush=True)
            return
        first_attempt = len(existing)+1
        for attempt in range(first_attempt,4):
            assert shutil.disk_usage(ROOT).free>5*2**30,'Disk below 5 GiB'
            log_path = ROOT/'logs'/f'{scene["slug"]}__{name}__attempt{attempt}.log'
            with log_path.open('x') as log:
                with GUARD:
                    p = subprocess.Popen([sys.executable,'-u',*arguments],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                    task = dict(name=name,attempt=attempt,command=arguments,pid=p.pid,started=time.time(),
                        start_ticks=int(Path(f'/proc/{p.pid}/stat').read_text().rpartition(')')[2].split()[19]),
                        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),log=str(log_path),
                        scheduler='parallel_scene')
                    job['status'] = 'running'
                    job['tasks'].append(task)
                    save()
                code = p.wait()
            text = log_path.read_text(errors='replace')
            oom = code!=0 and any(marker in text for marker in ['CUDA out of memory','OutOfMemoryError','Cannot allocate memory'])
            with GUARD:
                task.update(ended=time.time(),returncode=code,oom=oom)
                save()
            print('PARALLEL_PHASE',scene['id'],name,attempt,code,flush=True)
            if code==0:
                return
            if not oom or attempt==3 or output is None:
                raise RuntimeError(f'{scene["id"]} {name} failed with {code}, OOM={oom}')
            if output.exists():
                archived = output.with_name(output.name+f'_oom_attempt{attempt}')
                assert output.resolve().is_relative_to((ROOT/'models').resolve()) and not archived.exists()
                output.rename(archived)
                with GUARD:
                    task['archived_output'] = str(archived)
                    save()
        raise RuntimeError('OOM retries already exhausted')

    try:
        run('preflight',[str(HERE/'preflight.py'),scene['slug']])
        previous = None
        for stage in protocol['stages']:
            output = ROOT/'models'/scene['slug']/stage['name']
            command = [str(HERE/'scene_runtime.py'),'train',scene['checkpoint'],'--dataset',scene['dataset'],
                '--output',str(output),'--seed','0','--steps',str(stage['steps']),'--lr',str(stage['lr']),
                '--batch-points','4096','--calibration-cases','12','--image-every',str(stage['image_every']),
                '--image-weight',str(stage['image_weight']),'--residual-weight',str(stage['residual_weight']),
                '--validation-metric',stage['validation_metric'],'--validate-every',str(stage['validate_every']),
                '--ratio-jitter',str(stage['ratio_jitter'])]
            if stage['paired']:
                command += ['--paired-view-sampling','--validation-ratios',*[str(r) for r in protocol['continuous_validation_ratios']]]
            if previous is not None:
                command += ['--initialize',str(previous)]
            run(stage['name'],command,output)
            previous = output/'best.pt'
            assert previous.is_file()
        with GUARD:
            job['training_complete'] = True
            save()
        run('evaluation',[str(HERE/'evaluate_scene.py'),scene['slug']])
        run('deployment_validation',[str(HERE/'validate_deployment.py'),scene['slug']])
        with GUARD:
            job['status'] = 'complete'
            save()
    except Exception as error:
        with GUARD:
            job.update(status='failed',error=str(error))
            save()
        print('PARALLEL_SCENE_FAILED',scene['id'],str(error),flush=True)


def main():
    lock = (ROOT/'reports/CONTROLLER.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state = json.loads((ROOT/'reports/BATCH_STATUS.json').read_text())
    assert state['status']=='running'
    scenes = json.loads((ROOT/'reports/SCENES.json').read_text())
    protocol = json.loads((ROOT/'reports/PROTOCOL.json').read_text())
    by_id = {j['id']:j for j in state['scenes']}
    pending = []
    for scene in scenes:
        job = by_id.get(scene['id'])
        if job is None:
            job = dict(id=scene['id'],slug=scene['slug'],status='pending',tasks=[],reused=scene['reused'])
            state['scenes'].append(job)
        if job['status'] in ['pending','running']:
            pending.append((scene,job))
    state.update(scheduler='parallel_scene',parallel_started=time.time(),
        scheduling_authorization='User requested concurrent experiments',concurrency_history=[])
    write(ROOT/'reports/BATCH_STATUS.json',state)
    active = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        while pending or active:
            settings = json.loads((ROOT/'reports/PARALLEL_SETTINGS.json').read_text())
            limit = int(settings['max_workers'])
            assert 1<=limit<=4
            with GUARD:
                if not state['concurrency_history'] or state['concurrency_history'][-1]['max_workers']!=limit:
                    state['concurrency_history'].append(dict(time=time.time(),max_workers=limit))
                    write(ROOT/'reports/BATCH_STATUS.json',state)
            while pending and len(active)<limit:
                free_mib = float(subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True).splitlines()[0])
                if free_mib<3072:
                    break
                scene,job = pending.pop(0)
                future = pool.submit(execute_scene,scene,job,state,protocol)
                active[future] = scene['id']
            if active:
                done,_ = wait(active,timeout=5,return_when=FIRST_COMPLETED)
                for future in done:
                    future.result()
                    del active[future]
            else:
                time.sleep(5)
    with GUARD:
        state.update(status='finished',ended=time.time(),completed=sum(j['status']=='complete' for j in state['scenes']),
            failed=sum(j['status']=='failed' for j in state['scenes']))
        write(ROOT/'reports/BATCH_STATUS.json',state)
    print('PARALLEL_BATCH_FINISHED',state['completed'],state['failed'],flush=True)


if __name__=='__main__':
    main()
