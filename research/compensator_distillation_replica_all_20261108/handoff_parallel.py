"""仅移交本轮控制器，保留正在运行的训练或评价子进程。"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import signal
import subprocess
import time


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def verify(record,expected):
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==record['boot_id']
    path = Path(f'/proc/{record["pid"]}/stat')
    if not path.exists():
        return False
    parts = path.read_text().rpartition(')')[2].split()
    assert int(parts[19])==int(record['start_ticks']),'PID was reused'
    if parts[0]=='Z':
        return False
    cmd = Path(f'/proc/{record["pid"]}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    if not cmd:
        if not path.exists() or path.read_text().rpartition(')')[2].split()[0]=='Z':
            return False
    assert str(HERE/expected) in cmd,cmd
    return True


def identity(pid):
    parts = Path(f'/proc/{pid}/stat').read_text().rpartition(')')[2].split()
    return dict(pid=pid,start_ticks=int(parts[19]),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())


def launch(record):
    assert 'new_controller' not in record
    (ROOT/'reports/PARALLEL_SETTINGS.json').write_text(json.dumps(dict(max_workers=2),indent=2))
    log = (ROOT/'logs/parallel_batch.log').open('x')
    p = subprocess.Popen(['/home/prmi/anaconda3/envs/zwj/bin/python','-u',str(HERE/'parallel_batch.py')],
        cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    new = identity(p.pid)
    (ROOT/'reports/CONTROLLER.json').write_text(json.dumps(new,indent=2))
    with (ROOT/'logs/parallel_finalize.log').open('x') as output:
        f = subprocess.Popen(['/home/prmi/anaconda3/envs/zwj/bin/python','-u',str(HERE/'finalize.py')],
            cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,start_new_session=True)
    (ROOT/'reports/FINALIZER_PROCESS.json').write_text(json.dumps(identity(f.pid),indent=2))
    record.update(new_controller=new,new_finalizer=identity(f.pid),max_workers=2)
    (ROOT/'reports/PARALLEL_HANDOFF.json').write_text(json.dumps(record,indent=2))
    print('HANDOFF_COMPLETE',json.dumps(new),'preserved_children',[t['pid'] for t in record['preserved_children']],flush=True)


def main():
    receipt = ROOT/'reports/PARALLEL_HANDOFF.json'
    if receipt.exists():
        record = json.loads(receipt.read_text())
        assert not verify(record['old_controller'],'batch.py'),'Old controller is still alive'
        record['launch_recovery'] = 'old controller exit raced with cmdline inspection, launch not previously performed'
        launch(record)
        return
    old = json.loads((ROOT/'reports/CONTROLLER.json').read_text())
    assert verify(old,'batch.py')
    os.kill(old['pid'],signal.SIGSTOP)
    for _ in range(100):
        if Path(f'/proc/{old["pid"]}/stat').read_text().rpartition(')')[2].split()[0]=='T':
            break
        time.sleep(.01)
    else:
        raise RuntimeError('Controller did not stop before snapshot')
    state = json.loads((ROOT/'reports/BATCH_STATUS.json').read_text())
    assert state['status']=='running'
    active = [j for j in state['scenes'] if j['status']=='running']
    assert len(active)<=1
    children = [j['tasks'][-1] for j in active if j['tasks']]
    snapshot = ROOT/'reports/serial_handoff'
    snapshot.mkdir()
    for name in ['BATCH_STATUS.json','CONTROLLER.json','FINALIZER_PROCESS.json','PROTOCOL.json']:
        shutil.copy2(ROOT/'reports'/name,snapshot/name)
    finalizer = json.loads((ROOT/'reports/FINALIZER_PROCESS.json').read_text())
    if verify(finalizer,'finalize.py'):
        os.kill(finalizer['pid'],signal.SIGTERM)
    record = dict(time=time.time(),old_controller=old,old_finalizer=finalizer,
        preserved_children=children,snapshot=str(snapshot),scope='only this experiment controller and finalizer',
        algorithm_source_sha256=hashlib.sha256((ROOT/'source/python/distill_compensation.py').read_bytes()).hexdigest())
    (ROOT/'reports/PARALLEL_HANDOFF.json').write_text(json.dumps(record,indent=2))
    os.kill(old['pid'],signal.SIGTERM)
    os.kill(old['pid'],signal.SIGCONT)
    for _ in range(100):
        if not verify(old,'batch.py'):
            break
        time.sleep(.02)
    else:
        raise RuntimeError('Old controller remains alive')
    launch(record)


if __name__=='__main__':
    main()
