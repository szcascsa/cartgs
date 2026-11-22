"""单GPU串行队列；每组独立smoke、全轨迹三轮及退出核验。"""
from pathlib import Path
import fcntl
import json
import os
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parent
PYTHON = '/home/prmi/anaconda3/envs/zwj/bin/python'


def save(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temporary.replace(path)


def ticks(pid):
    return Path(f'/proc/{pid}/stat').read_text().rpartition(')')[2].split()[19]


def main():
    lock = (ROOT/'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    identity = dict(pid=os.getpid(), start_ticks=ticks(os.getpid()),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), state='running', started_unix=time.time())
    save(ROOT/'CONTROLLER.json', identity)
    scenes = json.loads((ROOT/'SCENES_FROZEN.json').read_text())
    queue_path = ROOT/'QUEUE.json'
    queue = json.loads(queue_path.read_text()) if queue_path.exists() else dict(
        state='running', scenes=[dict(scene=row['id'], slug=row['slug'], smoke='pending', formal='pending') for row in scenes])
    save(queue_path, queue)
    try:
        for row in queue['scenes']:
            for phase in ['smoke', 'formal']:
                output = ROOT/'runs'/row['slug']/phase
                validation = output/'VALIDATION.json'
                if row[phase] == 'complete':
                    assert json.loads(validation.read_text())['passed']
                    continue
                assert row[phase] in ['pending', 'failed'], (row, '禁止覆盖running任务')
                assert not output.exists(), (output, '已有现场需要先定位')
                if shutil.disk_usage(ROOT).free < 5*2**30:
                    raise RuntimeError('剩余磁盘不足5GiB；保留现场，禁止自动清理')
                name = row['slug']+'_'+phase
                job_path = ROOT/'logs'/f'{name}_job.json'
                command = [PYTHON, '-B', '-u', str(ROOT/'worker.py'), '--scene', row['slug']]
                if phase == 'smoke':
                    command.append('--smoke')
                job = dict(name=name, command=command, output_dir=str(output.relative_to(ROOT)),
                    measured=True, scene=row['scene'], phase=phase, training=False)
                save(job_path, job)
                row[phase] = 'running'
                queue.update(active_scene=row['scene'], active_phase=phase, updated_unix=time.time())
                save(queue_path, queue)
                with (ROOT/'logs'/f'{name}_runner.log').open('x') as log:
                    process = subprocess.Popen([PYTHON, '-B', '-u', str(ROOT/'job_runner.py'), str(job_path)],
                        cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                    identity.update(active_runner_pid=process.pid, active_scene=row['scene'], active_phase=phase)
                    save(ROOT/'CONTROLLER.json', identity)
                    returncode = process.wait()
                result = json.loads(job_path.read_text())
                if returncode or result.get('returncode') != 0 or not result.get('measurement_valid'):
                    row[phase] = 'failed'
                    save(queue_path, queue)
                    raise RuntimeError(f"{row['scene']} {phase}未通过，需读取日志定位；不盲目重跑")
                check = json.loads(validation.read_text())
                assert check['passed'] and check['warm_plan_misses'] == 0
                row[phase] = 'complete'
                row[phase+'_elapsed_s'] = result['elapsed_s']
                queue['updated_unix'] = time.time()
                save(queue_path, queue)
                print('SCENE_PHASE_COMPLETE', row['scene'], phase, result['elapsed_s'], flush=True)
        queue.update(state='complete', active_scene=None, active_phase=None, completed_unix=time.time())
        save(queue_path, queue)
        identity.update(state='complete', completed_unix=time.time())
    except BaseException as error:
        queue.update(state='failed', error=f'{type(error).__name__}: {error}', updated_unix=time.time())
        save(queue_path, queue)
        identity.update(state='failed', error=queue['error'])
        raise
    finally:
        save(ROOT/'CONTROLLER.json', identity)


if __name__ == '__main__':
    main()
