"""Replica十六组固定协议，复用通过核验的两组历史结果。"""
from pathlib import Path
import fcntl
import json
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def save(path,value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,indent=2,ensure_ascii=False))
    temp.replace(path)


def main():
    lock = (ROOT/'reports/CONTROLLER.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    scenes = json.loads((ROOT/'reports/SCENES.json').read_text())
    protocol = json.loads((ROOT/'reports/PROTOCOL.json').read_text())
    destination = ROOT/'reports/BATCH_STATUS.json'
    assert not destination.exists(),'Existing queue state must be audited before restart'
    state = dict(started=time.time(),status='running',scenes=[],reused=0)
    for scene in scenes:
        job = dict(id=scene['id'],slug=scene['slug'],status='running',tasks=[],reused=scene['reused'])
        state['scenes'].append(job)
        if scene['reused']:
            quality = json.loads((ROOT/'reports'/scene['slug']/'QUALITY_SUMMARY.json').read_text())
            assert json.loads((ROOT/'reports'/scene['slug']/'DEPLOYMENT_VALIDATION.json').read_text())['passed']
            job.update(status='complete',training_complete=True,reused_from=scene['reused_from'])
            state['reused'] += 1
            save(destination,state)
            print('REUSED',scene['id'],flush=True)
            continue
        save(destination,state)

        def run(name,arguments,output=None):
            for attempt in range(1,4):
                assert shutil.disk_usage(ROOT).free>5*2**30,'Disk below 5 GiB'
                log_path = ROOT/'logs'/f'{scene["slug"]}__{name}__attempt{attempt}.log'
                with log_path.open('x') as log:
                    p = subprocess.Popen([sys.executable,'-u',*arguments],cwd=ROOT,
                        stdout=log,stderr=subprocess.STDOUT)
                    task = dict(name=name,attempt=attempt,command=arguments,pid=p.pid,started=time.time(),
                        start_ticks=int(Path(f'/proc/{p.pid}/stat').read_text().rpartition(')')[2].split()[19]),
                        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),log=str(log_path))
                    job['tasks'].append(task)
                    save(destination,state)
                    code = p.wait()
                text = log_path.read_text(errors='replace')
                oom = code!=0 and any(marker in text for marker in ['CUDA out of memory','OutOfMemoryError','Cannot allocate memory'])
                task.update(ended=time.time(),returncode=code,oom=oom)
                save(destination,state)
                print(scene['id'],name,attempt,code,flush=True)
                if code==0:
                    return
                if not oom or attempt==3 or output is None:
                    raise RuntimeError(f'{scene["id"]} {name} failed with {code}, OOM={oom}')
                if output.exists():
                    archived = output.with_name(output.name+f'_oom_attempt{attempt}')
                    assert output.resolve().is_relative_to((ROOT/'models').resolve()) and not archived.exists()
                    output.rename(archived)
                    task['archived_output'] = str(archived)
                    save(destination,state)

        try:
            run('preflight',[str(HERE/'preflight.py'),scene['slug']])
            assert json.loads((ROOT/'reports'/scene['slug']/'PREFLIGHT.json').read_text())['passed']
            previous = None
            for stage in protocol['stages']:
                output = ROOT/'models'/scene['slug']/stage['name']
                command = [str(HERE/'scene_runtime.py'),'train',scene['checkpoint'],
                    '--dataset',scene['dataset'],'--output',str(output),'--seed','0',
                    '--steps',str(stage['steps']),'--lr',str(stage['lr']),'--batch-points','4096',
                    '--calibration-cases','12','--image-every',str(stage['image_every']),
                    '--image-weight',str(stage['image_weight']),'--residual-weight',str(stage['residual_weight']),
                    '--validation-metric',stage['validation_metric'],'--validate-every',str(stage['validate_every']),
                    '--ratio-jitter',str(stage['ratio_jitter'])]
                if stage['paired']:
                    command += ['--paired-view-sampling','--validation-ratios',
                        *[str(r) for r in protocol['continuous_validation_ratios']]]
                if previous is not None:
                    command += ['--initialize',str(previous)]
                run(stage['name'],command,output)
                previous = output/'best.pt'
                assert previous.exists()
            job['training_complete'] = True
            save(destination,state)
            run('evaluation',[str(HERE/'evaluate_scene.py'),scene['slug']])
            run('deployment_validation',[str(HERE/'validate_deployment.py'),scene['slug']])
            job['status'] = 'complete'
        except Exception as error:
            job.update(status='failed',error=str(error))
            print('SCENE_FAILED',str(error),flush=True)
        save(destination,state)
    state.update(status='finished',ended=time.time(),
        completed=sum(j['status']=='complete' for j in state['scenes']),
        failed=sum(j['status']=='failed' for j in state['scenes']))
    save(destination,state)
    print('REPLICA_BATCH_FINISHED',state['completed'],state['failed'],'reused',state['reused'],flush=True)


if __name__=='__main__':
    main()
