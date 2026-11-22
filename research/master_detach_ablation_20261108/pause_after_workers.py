"""接管已停止发射的控制器，等待现有worker完整结束后设置HOLD。"""
from pathlib import Path
import fcntl
import os
import signal
import time
import parallel_queue as q

R=q.R;p=q.p


def main():
    lock=(R/'pause_after_workers.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    request=p.read(R/'reports/PAUSE_AFTER_WORKERS_REQUEST.json')
    controller=request['controller']
    p.write(R/'reports/PAUSE_WAITER.json',p.identity(os.getpid()))
    errors=[]
    while True:
        assert q.supervisor.alive(controller),'原控制器身份变化，禁止改写队列'
        fields=(Path('/proc')/str(controller['pid'])/'stat').read_text().rsplit(')',1)[1].split()
        assert fields[0] in ['T','t'],'控制器未处于停止发射状态'
        state=p.read(R/'LOD_PIPELINE.json')
        assert all(info==request['workers'][lane] for lane,info in state['parallel_workers'].items())
        for lane in list(state['parallel_workers']):
            error=q.collect(state,lane)
            if error:errors.append(dict(lane=lane,error=error))
        state.update(phase='parallel_draining',pause_after_workers=True,
                     active_training_job=state['parallel_workers'].get('train',{}).get('job'),
                     active_evaluation_job=state['parallel_workers'].get('eval',{}).get('job'))
        state['active_job']=state['active_training_job'] or state['active_evaluation_job']
        p.save(state);q.b.report(state)
        if not state['parallel_workers']:
            assert not any(q.supervisor.alive(item.get('process',{}))
                           for _,item,_ in q.supervisor.running(state))
            (R/'HOLD').write_text('用户要求：当前训练和整组评价结束后暂停，等待明确恢复指令\n')
            state.update(phase='held',error='',pause_after_workers=False)
            p.save(state)
            p.write(R/'reports/PAUSE_AFTER_WORKERS_COMPLETED.json',dict(
                completed_unix=time.time(),workers=request['workers'],counts=state['counts'],
                errors=errors,held=True,no_active_children=True))
            # 只结束已核验且停止发射的本批控制器；不发送进程组信号。
            assert q.supervisor.alive(controller)
            os.kill(controller['pid'],signal.SIGTERM)
            os.kill(controller['pid'],signal.SIGCONT)
            print('PAUSED_AFTER_CURRENT_WORKERS',state['counts'],flush=True)
            return
        time.sleep(5)


if __name__=='__main__':main()
