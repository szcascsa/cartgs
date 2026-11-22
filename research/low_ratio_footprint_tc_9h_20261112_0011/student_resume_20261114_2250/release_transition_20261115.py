"""单次阶段接续；原验收完成且控制器锁释放后才启动Student联合队列。"""
from pathlib import Path
from datetime import datetime
import fcntl
import json
import os
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent


def main():
    reports = HERE / 'reports'
    path = reports / 'RELEASE_TRANSITION_IDENTITY.json'
    with path.open('x') as stream:
        json.dump(dict(pid=os.getpid(), start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().rpartition(')')[2].split()[19],
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), started_at=time.time()), stream)
    while True:
        protocol = json.loads((reports / 'PROTOCOL.json').read_text())
        assert time.time() < datetime.fromisoformat(protocol['final_validation_reserve']).timestamp()
        state = json.loads((reports / 'RELEASE_RENDER_20261115_CONTROLLER.json').read_text())
        assert state['state'] != 'failed', state
        if state['state'] == 'complete':
            with (reports / 'controller.lock').open('a') as lock:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    time.sleep(1)
                    continue
            assert not (reports / 'RELEASE_JOINT_20261115_CONTROLLER.json').exists()
            assert not (reports / 'SELECTED_RASTER_20261115.json').exists()
            subprocess.run([sys.executable, '-u', '-B', 'select_and_launch_joint_20261115.py'], cwd=HERE, check=True)
            print('RELEASE_TRANSITION_COMPLETE', time.time(), flush=True)
            return
        time.sleep(10)


if __name__ == '__main__':
    main()
