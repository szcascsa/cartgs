from common import *
from baseline import previous_runtime
from deployment_runtime import DeploymentRenderer
from large_scene import LARGE
import subprocess


def foreign_processes():
    value=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    return [int(line) for line in value.splitlines() if line.strip() and int(line)!=os.getpid()]


@torch.no_grad()
def main():
    # Keep only batches bracketed by two observations with no other CUDA process.
    results=[];deadline=time.monotonic()+180
    for group in ['tum','replica']:
        s=scene() if group=='tum' else BudgetScene(LARGE,None,5000,0,None,'pytorch')
        engines=[previous_runtime(s),DeploymentRenderer(s.model,s.gaussians,s.config)]
        conditions=[('1000',dict(budget=1000)),('5000',dict(budget=5000)),('1pct',dict(ratio_budget=.01)),('70pct',dict(ratio_budget=.7))]
        rows={name:dict(condition=name,group=group,prepare={'v27':[],'optimized':[]},total={'v27':[],'optimized':[]},frames=[]) for name,_ in conditions}
        attempts=0;accepted=0
        group_deadline=min(deadline,time.monotonic()+90)
        while time.monotonic()<group_deadline and min(len(r['frames']) for r in rows.values())<64:
            if foreign_processes():time.sleep(.15);continue
            name,request=conditions[attempts%len(conditions)];pending=[]
            for j in range(12):
                frame=((attempts*12+j)*73+19)%len(s.views);cam=camera(s,frame)
                values={}
                for index in ([0,1] if (j+attempts)%2 else [1,0]):
                    torch.cuda.synchronize();start=time.perf_counter()
                    packet,_,_=engines[index].prepare(cam,**request)
                    torch.cuda.synchronize();prepared=time.perf_counter()
                    render_prepared(cam,packet,s.background);torch.cuda.synchronize()
                    values[['v27','optimized'][index]]=[(prepared-start)*1000,(time.perf_counter()-start)*1000]
                pending.append((frame,values))
            attempts+=1
            if foreign_processes():continue
            for frame,values in pending:
                rows[name]['frames'].append(frame)
                for version,(prepare,total) in values.items():
                    rows[name]['prepare'][version].append(prepare);rows[name]['total'][version].append(total)
            accepted+=1
        results.extend(rows.values())
        save(ROOT/'reports/IDLE_WINDOWS.json',dict(results=results,group_attempts=attempts,accepted_batches=accepted,
            criterion='No foreign CUDA PID before and after each 12-request batch',complete=all(len(r['frames'])>=64 for r in results)))
        print('IDLE_WINDOWS',group,[(r['condition'],len(r['frames'])) for r in rows.values()],flush=True)
        del engines,s;torch.cuda.empty_cache()


if __name__=='__main__':main()
