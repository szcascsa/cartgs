from pathlib import Path
import json
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent


def main():
    tasks=[('compile_smoke.py',[]),('micro.py',[]),('benchmark.py',['1','--ablation']),('benchmark.py',['2','--ablation'])]
    if '--validation' in sys.argv:
        tasks=[('prefix_edges.py',[]),('validate.py',[]),('extra_checks.py',['tum']),('extra_checks.py',['replica']),
            ('graph_boundaries.py',[]),('graph_workspace_test.py',[]),('host_capacity.py',[]),
            ('control_audit.py',['tum']),('control_audit.py',['replica']),('trajectory.py',['tum']),('trajectory.py',['replica']),('space_cost.py',[]),('benchmark.py',['7']),('benchmark.py',['8']),('benchmark.py',['9'])]
    state=dict(pid=__import__('os').getpid(),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),tasks=[])
    state['start_ticks']=Path(f'/proc/{state["pid"]}/stat').read_text().split()[21]
    for index,(name,args) in enumerate(tasks):
        label=name.removesuffix('.py')+'_'+('_'.join(args) or 'all')
        path=ROOT/'logs'/f'{label}.log'
        with path.open('w') as out:
            p=subprocess.Popen([sys.executable,'-u',str(HERE/name),*args],stdout=out,stderr=subprocess.STDOUT)
            row=dict(name=name,args=args,pid=p.pid,started=time.time(),log=str(path));state['tasks'].append(row)
            (ROOT/'reports/RUNNER_STATUS.json').write_text(json.dumps(state,indent=2))
            code=p.wait();row.update(exit=code,ended=time.time())
        (ROOT/'reports/RUNNER_STATUS.json').write_text(json.dumps(state,indent=2))
        print(label,'EXIT',code,flush=True)
        if code:raise SystemExit(code)
    state['complete']=True;(ROOT/'reports/RUNNER_STATUS.json').write_text(json.dumps(state,indent=2))


if __name__=='__main__':main()
