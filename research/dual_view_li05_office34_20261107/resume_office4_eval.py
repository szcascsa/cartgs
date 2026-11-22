"""只补完原队列office4 RGB-D评价；结束后保持原队列暂停。"""
from pathlib import Path
import fcntl
import json
import sys
import time

B = Path('/home/prmi/zwj/realLOD/saved_experiments/dual_view_li_best_full_20261107')


def run(b):
    p = b.p
    root = b.R
    with (root/'controller.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = p.read(root/'LOD_PIPELINE.json')
        assert state['phase'] in ['held', 'held_for_li05_controls']
        job = next(j for j in state['jobs'] if j['id'] == 'Replica/office4/rgbd')
        assert job['state'] in ['trained', 'complete']
        assert all(e['state'] in ['pending', 'retry_pending', 'complete'] for e in job['evaluations'])
        p.write(root/'LOD_CONTROLLER.json', p.identity(__import__('os').getpid()))
        hold = root/'HOLD'
        if hold.exists():
            (root/'reports/HOLD_BEFORE_LI05.txt').write_text(hold.read_text())
            hold.unlink()
        try:
            state.update(phase='evaluating', active_job=job['id'], error='')
            p.save(state)
            for item in job['evaluations']:
                if item['state'] == 'complete':
                    continue
                b.evaluate(job, item, state, root/'evaluation'/job['id']/(item['scope']+'_'+str(item['ratio'])))
                assert item['state'] == 'complete', item
            job['state'] = 'complete'
            state.update(phase='held_for_li05_controls', active_job=None)
            p.write(root/'reports/OFFICE4_EVALUATION_COMPLETED.json', dict(
                completed_at=time.time(), evaluation_count=len(job['evaluations']), model=job['trained_model']))
        except Exception as error:
            state.update(phase='blocked', error=repr(error))
            raise
        finally:
            if not hold.exists():
                hold.write_text('按用户顺序等待office3/office4 RGB-D LI=0.5额外对照及结果汇报；不得先恢复原队列。\n')
            p.save(state)
            b.report(state)


if __name__ == '__main__':
    sys.path.insert(0, str(B))
    import run_batch
    run(run_batch)
