"""有界并发采集确定性计数，不做GPU测速；任一校验失败停止派发。"""
from pathlib import Path
import fcntl
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent/'all_datasets_joint182_ratio_speedup_20261115'


def ticks(pid):
    return Path(f'/proc/{pid}/stat').read_text().rpartition(')')[2].split()[19]


def save(path, data):
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps(data, indent=2)+'\n')
    pending.replace(path)


def main():
    lock = (ROOT/'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    scenes = json.loads((PRIOR/'SCENES_FROZEN.json').read_text())
    assert len(scenes) == 32
    leads = ['Replica/office2/rgbd', 'Replica/office0/mono', 'TUM_RGBD/freiburg1_xyz/rgbd',
        'TUM_RGBD/freiburg1_room/mono', 'VECtor/corner-slow/mono', 'VECtor/robot-normal/stereo']
    identity = dict(pid=os.getpid(), start_ticks=ticks(os.getpid()),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), started=time.time(),
        command=sys.argv, state='running')
    save(ROOT/'CONTROLLER.json', identity)
    queue = dict(state='smoke', concurrency=3, scenes=[dict(scene=s['id'], slug=s['slug'],
        frames=s['coverage']['evaluated_views'], smoke='pending' if s['id'] in leads else 'not_required',
        formal='pending') for s in scenes])
    logs = ROOT/'logs'
    logs.mkdir(exist_ok=True)
    running = {}

    def launch(row, stage):
        name = row['slug']+'_'+stage
        command = [sys.executable, '-B', '-u', str(ROOT/'collect.py'), '--scene', row['slug']]
        if stage == 'smoke':
            command.append('--smoke')
        log = (logs/(name+'.log')).open('a')
        process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
        row[stage] = 'running'
        row[stage+'_process'] = dict(pid=process.pid, start_ticks=ticks(process.pid), command=command,
            started=time.time(), log=str(log.name))
        running[process.pid] = (process, log, row, stage)

    failed = False
    while True:
        for pid, (process, log, row, stage) in list(running.items()):
            code = process.poll()
            if code is None:
                continue
            log.close()
            validation = ROOT/'runs'/row['slug']/stage/'VALIDATION.json'
            passed = code == 0 and validation.is_file() and json.loads(validation.read_text())['passed']
            row[stage] = 'complete' if passed else 'failed'
            row[stage+'_process'].update(returncode=code, ended=time.time())
            failed |= not passed
            del running[pid]
        if not failed:
            smoke = [r for r in queue['scenes'] if r['smoke'] == 'pending']
            if smoke:
                if not running:
                    launch(smoke[0], 'smoke')
            elif not any(r['smoke'] == 'running' for r in queue['scenes']):
                queue['state'] = 'formal'
                for row in [r for r in queue['scenes'] if r['formal'] == 'pending'][:max(0, queue['concurrency']-len(running))]:
                    launch(row, 'formal')
        queue['updated'] = time.time()
        save(ROOT/'QUEUE.json', queue)
        if failed and not running:
            queue['state'] = 'failed'
            break
        if all(r['formal'] == 'complete' for r in queue['scenes']):
            queue['state'] = 'complete'
            break
        time.sleep(2)
    save(ROOT/'QUEUE.json', queue)
    identity.update(state=queue['state'], ended=time.time())
    save(ROOT/'CONTROLLER.json', identity)
    print(json.dumps(dict(state=queue['state'], complete=sum(r['formal']=='complete' for r in queue['scenes']))), flush=True)
    assert not failed


if __name__ == '__main__':
    main()
