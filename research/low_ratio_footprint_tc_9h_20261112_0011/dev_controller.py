"""首批候选顺序执行；污染整段重测，结果不覆盖，不并发占用GPU。"""
from pathlib import Path
import fcntl
import argparse
import json
import os
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',default='02');parser.add_argument('--modes',default=','.join(map(str,range(15))))
    parser.add_argument('--wait-build');args=parser.parse_args()
    lock=(HERE/'reports/controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    path=HERE/'reports'/('DEV_CONTROLLER'+args.stage+'.json')
    if path.exists():
        old=json.loads(path.read_text());assert old['state']=='failed'
        p=Path(f'/proc/{old["pid"]}/stat')
        if p.exists():assert p.read_text().rpartition(')')[2].split()[19]!=str(old['start_ticks'])
        archived=path.with_name(path.stem+'_FAILED_'+str(old['pid'])+'.json');assert not archived.exists();path.rename(archived)
    state=dict(state='running',pid=os.getpid(),start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().rpartition(')')[2].split()[19],
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),started_at=time.time(),accepted=[],rejected=[])
    def save():
        tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2));tmp.replace(path)
    save()
    try:
        if args.wait_build:
            while True:
                build=json.loads((HERE/'reports/jobs'/(args.wait_build+'.json')).read_text())
                if build['state'] in ('complete','failed'):break
                time.sleep(1)
            assert build['returncode']==0,build
        for workload in ('lod','nested'):
            plans=[(f'quality_{workload}{args.stage}','quality.py',['--tune'],False)]
            plans += [(f'screen_{workload}{args.stage}_r{r}','screen.py',['--round',str(r)],True) for r in (1,2)]
            for label,script,extra,measured in plans:
                for attempt in range(1,5):
                    name=f'{label}_a{attempt}'
                    command=[sys.executable,'-B','launch.py']+(['--measure'] if measured else [])+[
                        name,sys.executable,'-u','-B',script,'--output',name,'--modes',args.modes,*extra]
                    if workload=='nested':command.append('--nested')
                    job_path=HERE/'reports/jobs'/(name+'.json')
                    if not job_path.exists():subprocess.run(command,cwd=HERE,check=True)
                    while True:
                        try:job=json.loads(job_path.read_text())
                        except json.JSONDecodeError:
                            time.sleep(.1);continue
                        if job['state'] in ('complete','failed'):break
                        time.sleep(1)
                    assert job['returncode']==0,(name,job)
                    if job['measurement_valid']:state['accepted'].append(name);save();break
                    state['rejected'].append(name);save()
                else:raise RuntimeError('持续外部负载，已保留现场，不盲目计入性能')
        state.update(state='complete',ended_at=time.time());save()
    except BaseException as error:
        state.update(state='failed',error=repr(error),ended_at=time.time());save();raise
    print('DEV_SEQUENCE_COMPLETE',flush=True)


if __name__=='__main__':main()
