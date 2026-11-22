"""训练完整结束后顺序执行留出评价、部署回归与完整性核验。"""
from pathlib import Path
import json
import os
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def main():
    controller = json.loads((ROOT/'reports/CONTINUATION_CONTROLLER.json').read_text())
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == controller['boot_id']
    while True:
        state = json.loads((ROOT/'reports/PAIRED_RUNS_STATUS.json').read_text())
        if state['state'] == 'all_complete' and state.get('steps_per_arm') == 6000:
            break
        proc = Path(f'/proc/{controller["pid"]}/stat')
        assert proc.exists() and int(proc.read_text().rpartition(')')[2].split()[19]) == controller['start_ticks'], 'Training controller exited'
        assert state['state'] != 'failed', state
        time.sleep(5)
    result = dict(pid=os.getpid(), tasks=[])
    for name in ['compare_quality', 'final_validation', 'integrity']:
        with (ROOT/'logs'/f'{name}.log').open('x') as log:
            code = subprocess.call([sys.executable, '-u', str(HERE/(name+'.py'))],
                cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        result['tasks'].append(dict(name=name, returncode=code))
        (ROOT/'reports/DELIVERY_TESTS.json').write_text(json.dumps(result, indent=2))
        print(name, code, flush=True)
        if code:
            raise SystemExit(code)
    result['complete'] = True
    (ROOT/'reports/DELIVERY_TESTS.json').write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
