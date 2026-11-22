"""让当前评价自然结束后切换调度器，保留评价结果和尝试记录。"""
from pathlib import Path
import fcntl
import os
import signal
import subprocess
import sys
import time
import traceback

ROOT='/home/prmi/zwj/experiments/no_importance_all_scenes_20261115'
R=Path(ROOT);worker=sys.argv[1]
os.environ.update(CARTGS_CLUSTER_ROOT=ROOT,CARTGS_CLUSTER_WORKER=worker,CARTGS_PIPELINE_OVERLAP='1',TORCH_HOME='/home/prmi/.cache/torch')
import run as r
from prepare import read,write


def fields(pid):
    p=Path('/proc')/str(pid)/'stat'
    return p.read_text().split(') ',1)[1].split() if p.exists() else []


def main():
    lock=(R/'pipeline_handoff.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    old=read(R/'LOD_CONTROLLER.json');outer=read(R/'reports/CLOUD_LAUNCH.json')['identity']
    assert old==r.p.identity(old['pid']) and outer==r.p.identity(outer['pid'])
    assert worker in ['cloud_a','cloud_b'] and not (R/'reports/PIPELINE_HANDOFF_DONE.json').exists()
    assert str(R/'run.py').encode() in (Path('/proc')/str(old['pid'])/'cmdline').read_bytes().split(b'\0')
    paused=False;killed=False
    try:
        os.kill(old['pid'],signal.SIGSTOP);paused=True
        for _ in range(100):
            if fields(old['pid'])[0]=='T':break
            time.sleep(.02)
        else:raise RuntimeError('旧调度器未暂停')
        state=read(R/'LOD_PIPELINE.json');write(R/'reports/PIPELINE_HANDOFF_BEFORE.json',state)
        active=[]
        for job in state['jobs']:
            if job.get('assigned_worker')!=worker:continue
            assert job['state']!='running','发现训练仍运行，保留原调度器，稍后切换'
            for item in job['evaluations']:
                if item['state']=='running':active.append((job,item))
        children={int(n) for task in (Path('/proc')/str(old['pid'])/'task').iterdir() for n in (task/'children').read_text().split()}
        assert children=={item['process']['pid'] for _,item in active},'子进程与状态不匹配'
        ended={};start=time.time()
        while len(ended)<len(active):
            for job,item in active:
                ident=item['process'];pid=ident['pid']
                if pid in ended:continue
                assert ident==r.p.identity(pid)
                f=fields(pid);assert int(f[1])==old['pid']
                if f[0]=='Z':ended[pid]=dict(returncode=os.waitstatus_to_exitcode(int(f[-1])),ended_unix=time.time())
            write(R/'reports/PIPELINE_HANDOFF_PROGRESS.json',dict(phase='draining_current_evaluations',previous_controller=old,
                active=[dict(id=j['id'],scope=e['scope'],ratio=e['ratio'],process=e['process']) for j,e in active],finished=len(ended),elapsed=time.time()-start))
            assert time.time()-start<1800,'当前评价30分钟未结束；恢复原调度器并保留现场'
            if len(ended)<len(active):time.sleep(1)
        previous_save=r.save;r.save=lambda _:None;receipts=[]
        try:
            for job,item in active:
                out=Path(item['attempt']);exit_info=ended[item['process']['pid']]
                info=dict(returncode=exit_info['returncode'],seconds=exit_info['ended_unix']-read(out/'launch.json')['started_unix'],recovered_during_pipeline_handoff=True)
                receipts.append((out,info))
                oom=r.classify_oom(dict(output=str(out),process=item['process']),info)
                assert not any(a['output']==str(out) for a in item.get('attempt_history',[]))
                item.setdefault('attempt_history',[]).append(dict(output=str(out),process=item['process'],oom=oom,health=None,**info))
                if info['returncode']==0:r.finish_evaluation(job,item,state,out)
                elif oom:
                    item['oom_failures']=item.get('oom_failures',0)+1
                    item.update(state='retry_pending' if item['oom_failures']<=2 else 'skipped_oom',error='明确OOM；沿用累计最多额外2次重试')
                else:
                    item.update(state='failed',error=dict(returncode=info['returncode'],log=str(out/'evaluate.log')))
                    job.update(state='failed',error=item['error'])
            write(R/'reports/PIPELINE_HANDOFF_PREPARED.json',state)
        finally:r.save=previous_save
        assert old==r.p.identity(old['pid']);os.kill(old['pid'],signal.SIGKILL);killed=True
        for _ in range(200):
            f=fields(outer['pid'])
            if not f or f[0]=='Z':break
            assert outer==r.p.identity(outer['pid']);time.sleep(.05)
        else:raise RuntimeError('旧外层入口尚未退出')
        for out,info in receipts:
            write(out/'exit.json',info)
            if not (out/'train.log').exists():(out/'train.log').symlink_to('evaluate.log')
        r.save(state)
        with (R/'logs/cloud_training_overlap.log').open('a') as log:
            child=subprocess.Popen([sys.executable,'-u',str(R/'run.py')],cwd=R,env=dict(os.environ),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        identity=r.p.identity(child.pid)
        for _ in range(100):
            if read(R/'LOD_CONTROLLER.json')==identity:break
            assert fields(child.pid) and fields(child.pid)[0]!='Z','新调度器启动失败'
            time.sleep(.1)
        else:raise RuntimeError('新调度器未获得控制权')
        write(R/'reports/CLOUD_LAUNCH.json',dict(worker=worker,identity=identity,started_unix=time.time(),scheduling='2_train_plus_2_evaluate'))
        write(R/'reports/PIPELINE_HANDOFF_DONE.json',dict(passed=True,previous_controller=old,new_controller=identity,
            preserved_evaluations=len(active),train_workers=2,evaluation_workers=2,completed_unix=time.time()))
        print('PIPELINE_HANDOFF_PASSED',worker,identity,flush=True)
    except Exception:
        write(R/'reports/PIPELINE_HANDOFF_ERROR.json',dict(time=time.time(),traceback=traceback.format_exc(),previous_controller_killed=killed))
        if paused and not killed and old==r.p.identity(old['pid']):os.kill(old['pid'],signal.SIGCONT)
        raise


if __name__=='__main__':main()
