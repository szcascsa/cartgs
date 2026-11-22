"""四场景固定蒸馏协议，串行执行并保留失败现场。"""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def save(path, value):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temp.replace(path)


def main():
    scenes = json.loads((ROOT/'reports/SCENES.json').read_text())
    protocol = json.loads((ROOT/'reports/PROTOCOL.json').read_text())
    destination = ROOT/'reports/BATCH_STATUS.json'
    assert not destination.exists()
    state = dict(started=time.time(), status='running', scenes=[])
    for scene in scenes:
        job = dict(id=scene['id'], slug=scene['slug'], status='running', tasks=[])
        state['scenes'].append(job)
        save(destination, state)

        def run(name, arguments):
            assert shutil.disk_usage(ROOT).free > 5*2**30, 'Disk below 5 GiB'
            log_path = ROOT/'logs'/f'{scene["slug"]}__{name}.log'
            with log_path.open('x') as log:
                p = subprocess.Popen([sys.executable, '-u', *arguments], cwd=ROOT,
                    stdout=log, stderr=subprocess.STDOUT)
                task = dict(name=name, command=arguments, pid=p.pid, started=time.time(),
                    start_ticks=int(Path(f'/proc/{p.pid}/stat').read_text().rpartition(')')[2].split()[19]),
                    boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), log=str(log_path))
                job['tasks'].append(task)
                save(destination, state)
                code = p.wait()
            task.update(ended=time.time(), returncode=code)
            save(destination, state)
            print(scene['id'], name, code, flush=True)
            if code:
                raise RuntimeError(f'{scene["id"]} {name} failed with {code}')

        try:
            preflight = ROOT/'reports'/scene['slug']/'PREFLIGHT.json'
            if not preflight.exists():
                run('preflight', [str(HERE/'preflight.py'), scene['slug']])
            assert json.loads(preflight.read_text())['passed']
            previous = None
            for stage in protocol['stages']:
                output = ROOT/'models'/scene['slug']/stage['name']
                command = [str(HERE/'scene_runtime.py'), 'train', scene['checkpoint'],
                    '--dataset', scene['dataset'], '--output', str(output), '--seed', '0',
                    '--steps', str(stage['steps']), '--lr', str(stage['lr']), '--batch-points', '4096',
                    '--calibration-cases', '12', '--image-every', str(stage['image_every']),
                    '--image-weight', str(stage['image_weight']), '--residual-weight', str(stage['residual_weight']),
                    '--validation-metric', stage['validation_metric'], '--validate-every', str(stage['validate_every']),
                    '--ratio-jitter', str(stage['ratio_jitter'])]
                if stage['paired']:
                    command += ['--paired-view-sampling', '--validation-ratios',
                        *[str(r) for r in protocol['continuous_validation_ratios']]]
                if previous is not None:
                    command += ['--initialize', str(previous)]
                run(stage['name'], command)
                previous = output/'best.pt'
                assert previous.exists()
            job['training_complete'] = True
            save(destination, state)
            run('evaluation', [str(HERE/'evaluate_scene.py'), scene['slug']])
            run('deployment_validation', [str(HERE/'validate_deployment.py'), scene['slug']])
            job['status'] = 'complete'
        except Exception as error:
            job.update(status='failed', error=str(error))
            print('SCENE_FAILED', str(error), flush=True)
        save(destination, state)
    state.update(status='finished', ended=time.time(),
        completed=sum(j['status']=='complete' for j in state['scenes']),
        failed=sum(j['status']=='failed' for j in state['scenes']))
    save(destination, state)
    print('CROSS_SCENE_BATCH_FINISHED', state['completed'], state['failed'], flush=True)


if __name__ == '__main__':
    main()
