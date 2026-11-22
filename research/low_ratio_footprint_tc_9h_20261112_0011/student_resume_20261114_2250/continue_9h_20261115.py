"""记录用户新授权的九小时窗口，仅接续未完成验收，不覆盖既有控制器和结果。"""
from pathlib import Path
from datetime import datetime, timedelta
import fcntl
import json
import shutil
import subprocess
import time

HERE = Path(__file__).resolve().parent


def main():
    reports = HERE / 'reports'
    record = reports / 'BUDGET_RESET_20261115_0244.json'
    assert not record.exists()
    lock = (reports / 'controller.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    audited = []
    for path in reports.glob('*CONTROLLER.json'):
        state = json.loads(path.read_text())
        proc = Path(f"/proc/{state.get('pid', 0)}/stat")
        alive = False
        if proc.exists():
            fields = proc.read_text().rpartition(')')[2].split()
            alive = (fields[0] != 'Z' and fields[19] == str(state.get('start_ticks'))
                     and state.get('boot_id') == boot)
        assert not alive, (str(path), '已有控制器仍存活')
        audited.append(dict(path=path.name, state=state.get('state'), alive=alive))
    assert shutil.disk_usage(HERE).free >= 5 * (1 << 30)
    protocol_path = reports / 'PROTOCOL.json'
    old = json.loads(protocol_path.read_text())
    start = datetime.fromisoformat('2026-11-15T02:44:06+08:00')
    amendment = dict(recorded_at=time.time(), authority='用户最新指令：实验从现在开始计时9个小时',
                     previous_protocol=old, controller_audit=audited,
                     start=start.isoformat(), deadline=(start + timedelta(hours=9)).isoformat(),
                     final_validation_reserve=(start + timedelta(hours=8)).isoformat())
    record.write_text(json.dumps(amendment, indent=2, ensure_ascii=False))
    old.update({key: amendment[key] for key in ['start', 'deadline', 'final_validation_reserve']})
    old['budget_authority'] = amendment['authority']
    old['budget_reset_record'] = str(record)
    protocol_path.write_text(json.dumps(old, indent=2, ensure_ascii=False))
    previous = json.loads((reports / 'RESEARCH_ACCEPTANCE_CONTROLLER.json').read_text())
    assert previous['state'] == 'failed' and previous['current_task'] == 'student_execution_identities'
    plan = json.loads((reports / 'RESEARCH_ACCEPTANCE_PLAN.json').read_text())
    pending = []
    for task in plan['jobs']:
        if any(name.startswith(task['name'] + '_a') for name in previous['accepted']):
            continue
        pending.append(dict(task, name=task['name'] + '_continued'))
    assert pending and pending[0]['name'] == 'student_execution_identities_continued'
    plan_path = reports / 'ACCEPTANCE_CONTINUED_PLAN.json'
    assert not plan_path.exists()
    plan_path.write_text(json.dumps(dict(name='ACCEPTANCE_CONTINUED', jobs=pending), indent=2))
    fcntl.flock(lock, fcntl.LOCK_UN)
    with (HERE / 'logs/acceptance_continued_controller.log').open('x') as log:
        child = subprocess.Popen(['/home/prmi/anaconda3/envs/zwj/bin/python', '-u', '-B',
                                  'run_stage.py', str(plan_path)], cwd=HERE,
                                 stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                 start_new_session=True)
    print('CONTINUED', child.pid, len(pending), amendment['start'], amendment['deadline'], flush=True)


if __name__ == '__main__':
    main()
