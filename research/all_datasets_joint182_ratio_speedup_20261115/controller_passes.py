"""用户指定优先级：全部组先单轮交付，之后顺序补足三轮。"""
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


def live(job, prefix):
    path = Path(f"/proc/{job.get(prefix+'_pid', 0)}/stat")
    if not path.exists():
        return False
    return ticks(job[prefix+'_pid']) == job[prefix+'_start_ticks']


def main():
    lock = (ROOT/'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    identity = dict(pid=os.getpid(), start_ticks=ticks(os.getpid()),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), state='running', started_unix=time.time(),
        priority='all groups first repeat, then remaining repeats')
    save(ROOT/'CONTROLLER.json', identity)
    queue_path = ROOT/'QUEUE.json'
    queue = json.loads(queue_path.read_text())
    queue.update(state='running_first_pass', execution_order='all groups round0 -> first-pass plots -> rounds1/2')
    try:
        # 接收旧控制器已启动的任务；先确认它真实结束，不重复启动。
        for row in queue['scenes']:
            for phase in ['smoke', 'formal']:
                if row[phase] != 'running':
                    continue
                path = ROOT/'logs'/f"{row['slug']}_{phase}_job.json"
                while True:
                    job = json.loads(path.read_text())
                    if job.get('returncode') is not None:
                        break
                    assert live(job, 'runner'), ('旧任务未记录退出且runner已退出', row['scene'], phase)
                    time.sleep(1)
                assert job.get('measurement_valid') and job['returncode'] == 0, job
                row[phase] = 'complete'
                row[phase+'_elapsed_s'] = job['elapsed_s']
        save(queue_path, queue)

        def run(row, phase):
            output = ROOT/'runs'/row['slug']/phase
            validation = output/'VALIDATION.json'
            if row.get(phase) == 'complete':
                assert json.loads(validation.read_text())['passed']
                return
            assert row.get(phase, 'pending') == 'pending' and not output.exists(), (row, phase, output)
            assert shutil.disk_usage(ROOT).free >= 5*2**30, '剩余磁盘不足5GiB，不自动清理'
            name = row['slug']+'_'+phase
            job_path = ROOT/'logs'/f'{name}_job.json'
            command = [PYTHON, '-B', '-u', str(ROOT/'worker_passes.py'), '--scene', row['slug']]
            if phase == 'smoke':
                command.append('--smoke')
            elif phase == 'formal':
                command += ['--repeat-start', '0', '--repeat-count', '1']
            else:
                command += ['--stage', 'remaining', '--repeat-start', '1', '--repeat-count', '2']
            save(job_path, dict(name=name, command=command, output_dir=str(output.relative_to(ROOT)),
                measured=True, scene=row['scene'], phase=phase, training=False))
            row[phase] = 'running'
            queue.update(active_scene=row['scene'], active_phase=phase, updated_unix=time.time())
            save(queue_path, queue)
            with (ROOT/'logs'/f'{name}_runner.log').open('x') as log:
                process = subprocess.Popen([PYTHON, '-B', '-u', str(ROOT/'job_runner.py'), str(job_path)],
                    cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
                identity.update(active_runner_pid=process.pid, active_scene=row['scene'], active_phase=phase)
                save(ROOT/'CONTROLLER.json', identity)
                returncode = process.wait()
            job = json.loads(job_path.read_text())
            if returncode or job.get('returncode') != 0 or not job.get('measurement_valid'):
                row[phase] = 'failed'
                save(queue_path, queue)
                raise RuntimeError(f"{row['scene']} {phase}失败，停止队列定位，不盲目重跑")
            check = json.loads(validation.read_text())
            assert check['passed'] and check['warm_plan_misses'] == 0
            row[phase] = 'complete'
            row[phase+'_elapsed_s'] = job['elapsed_s']
            save(queue_path, queue)
            print('SCENE_PHASE_COMPLETE', row['scene'], phase, job['elapsed_s'], flush=True)

        for row in queue['scenes']:
            run(row, 'smoke')
            run(row, 'formal')
        queue.update(state='first_pass_complete', active_scene=None, active_phase=None, first_pass_completed_unix=time.time())
        save(queue_path, queue)
        with (ROOT/'logs/first_pass_analysis.log').open('w') as log:
            subprocess.run([PYTHON, '-B', '-u', str(ROOT/'analyze_passes.py'), '--first-pass'],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        save(ROOT/'FIRST_PASS_READY.json', dict(passed=True, groups=32, repeats=1, completed_unix=time.time(),
            output='first_pass', remaining_rounds='continuing automatically as authorized'))
        queue['state'] = 'running_remaining_rounds'
        save(queue_path, queue)
        for row in queue['scenes']:
            formal = json.loads((ROOT/'runs'/row['slug']/'formal/VALIDATION.json').read_text())
            if formal['repeats'] == 3:
                row['remaining'] = 'already_in_formal'
                save(queue_path, queue)
                continue
            run(row, 'remaining')
        queue.update(state='complete', active_scene=None, active_phase=None, completed_unix=time.time())
        save(queue_path, queue)
        with (ROOT/'logs/final_analysis.log').open('w') as log:
            subprocess.run([PYTHON, '-B', '-u', str(ROOT/'analyze_passes.py')],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
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
