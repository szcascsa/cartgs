"""终验单次接续与汇总；没有训练任务，不替换已有控制器。"""
from pathlib import Path
from datetime import datetime
import fcntl
import json
import os
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
R = HERE / 'reports'


def run(name):
    subprocess.run([sys.executable, '-u', '-B', name], cwd=HERE, check=True)


def wait_stage(name):
    while True:
        deadline = datetime.fromisoformat(json.loads((R / 'PROTOCOL.json').read_text())['deadline']).timestamp()
        assert time.time() < deadline
        path = R / (name + '_CONTROLLER.json')
        if not path.exists():
            time.sleep(1)
            continue
        state = json.loads(path.read_text())
        assert state['state'] != 'failed', state
        if state['state'] == 'complete':
            with (R / 'controller.lock').open('a') as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    time.sleep(1)
                    continue
            return
        time.sleep(10)


def main():
    with (R / 'RELEASE_FINISH_PIPELINE_IDENTITY.json').open('x') as stream:
        json.dump(dict(pid=os.getpid(), start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().rpartition(')')[2].split()[19],
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), start=time.time()), stream)
    wait_stage('RELEASE_JOINT_CONTINUED_20261115')
    run('merge_release_controller.py')
    for script in ['summarize_release_joint.py', 'summarize_release_thin.py', 'audit_release_results.py',
                   'analyze_release_scaling.py', 'analyze_release_workloads.py']:
        run(script)
    run('launch_unseen_validation.py')
    wait_stage('RELEASE_UNSEEN_20261115')
    run('summarize_unseen_validation.py')
    print('RELEASE_FINISH_PIPELINE_COMPLETE', time.time(), flush=True)


if __name__ == '__main__':
    main()
