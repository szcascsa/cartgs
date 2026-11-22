"""并发交接后的锁、状态、任务唯一性和算法配置核验。"""
from pathlib import Path
import fcntl
import hashlib
import json
import tempfile
import threading
import time
from parallel_batch import alive,write,GUARD


ROOT = Path(__file__).resolve().parents[3]


def main():
    record = json.loads((ROOT/'reports/PARALLEL_HANDOFF.json').read_text())
    assert not alive(record['old_controller'])
    assert not alive(record['old_finalizer'])
    assert alive(record['new_controller'])
    assert alive(record['new_finalizer'])
    assert hashlib.sha256((ROOT/'source/python/distill_compensation.py').read_bytes()).hexdigest()==record['algorithm_source_sha256']
    with (ROOT/'reports/CONTROLLER.lock').open('a+') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            pass
        else:
            raise AssertionError('Parallel controller lock is not held')
    state = json.loads((ROOT/'reports/BATCH_STATUS.json').read_text())
    assert len(state['scenes'])==len({j['id'] for j in state['scenes']})==16
    active = [j for j in state['scenes'] if j['status']=='running']
    assert 1<=len(active)<=2
    pids = [j['tasks'][-1]['pid'] for j in active]
    assert len(pids)==len(set(pids)) and all(alive(j['tasks'][-1]) for j in active)
    before = json.loads((ROOT/'reports/serial_handoff/BATCH_STATUS.json').read_text())
    for old_job in before['scenes']:
        if old_job['status']=='complete':
            new_job = next(j for j in state['scenes'] if j['id']==old_job['id'])
            assert new_job==old_job,'Completed job was rewritten'
    protocol = json.loads((ROOT/'reports/PROTOCOL.json').read_text())
    baseline = json.loads((ROOT/'reports/serial_handoff/PROTOCOL.json').read_text())
    assert protocol['stages']==baseline['stages'] and protocol['train_ratios']==baseline['train_ratios']
    with tempfile.TemporaryDirectory(dir=ROOT/'reports') as folder:
        target = Path(folder)/'state.json'
        seen = []
        def update(worker):
            for step in range(25):
                with GUARD:
                    write(target,dict(worker=worker,step=step))
                    assert json.loads(target.read_text())==dict(worker=worker,step=step)
                    seen.append((worker,step))
        threads = [threading.Thread(target=update,args=(i,)) for i in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert len(seen)==100
    report = dict(passed=True,time=time.time(),new_controller=record['new_controller'],
        old_controller_exited=True,old_finalizer_exited=True,controller_lock_exclusive=True,
        completed_jobs_preserved=True,algorithm_sha_unchanged=True,stage_configs_unchanged=True,
        unique_scene_ids=True,active_unique_pids=pids,serialized_atomic_state_writes=100,
        adopted=[(j['id'],t['name'],t['pid']) for j in state['scenes'] for t in j['tasks'] if t.get('adopted_success')])
    (ROOT/'reports/PARALLEL_VALIDATION.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
