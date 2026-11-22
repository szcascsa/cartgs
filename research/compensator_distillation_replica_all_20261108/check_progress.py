"""核验本轮进程身份与真实训练进度。"""
from pathlib import Path
import csv
import json
import time


ROOT = Path(__file__).resolve().parents[3]


def process(identity):
    path = Path(f'/proc/{identity["pid"]}/stat')
    if not path.exists():
        return dict(alive=False)
    parts = path.read_text().rpartition(')')[2].split()
    same = int(parts[19]) == int(identity['start_ticks'])
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    return dict(alive=same and boot==identity['boot_id'] and parts[0]!='Z',
        state=parts[0],identity_match=same,cmdline=Path(f'/proc/{identity["pid"]}/cmdline').read_bytes().replace(b'\0',b' ').decode())


def main():
    state = json.loads((ROOT/'reports/BATCH_STATUS.json').read_text())
    result = dict(time=time.time(),status=state['status'],scenes=[])
    controller = json.loads((ROOT/'reports/CONTROLLER.json').read_text())
    result['controller'] = process(controller)
    for job in state['scenes']:
        summary = dict(id=job['id'],status=job['status'])
        if job['tasks']:
            task = job['tasks'][-1]
            summary['task'] = task['name']
            summary['process'] = process(task)
            log = Path(task['log'])
            summary['log_bytes'] = log.stat().st_size
            summary['log_age_seconds'] = time.time()-log.stat().st_mtime
            summary['log_tail'] = log.read_text(errors='replace').splitlines()[-3:]
            trace = ROOT/'models'/job['slug']/task['name']/'training.csv'
            if trace.exists():
                rows = list(csv.DictReader(trace.open()))
                if rows:
                    summary.update(step=int(rows[-1]['step']),last_row=rows[-1],
                        trace_age_seconds=time.time()-trace.stat().st_mtime)
        result['scenes'].append(summary)
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__ == '__main__':
    main()
