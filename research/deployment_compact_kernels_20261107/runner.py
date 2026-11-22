from pathlib import Path
import subprocess,json,time,os
r=Path(__file__).resolve().parents[3]
here=Path(__file__).resolve().parent
py="/home/prmi/anaconda3/envs/zwj/bin/python"
tasks=[["numeric_edges.py"],["candidate_edges.py"],["selection_fused.py"],["extra_checks.py","tum"],["extra_checks.py","replica"],["graph_boundaries.py"],["validate.py"],
       ["host_capacity.py"],["control_audit.py","tum"],["control_audit.py","replica"],
       ["benchmark.py","1"],["benchmark.py","2"],["benchmark.py","3"],["trajectory.py","tum"],["trajectory.py","replica"],
       ["memory.py","v21","tum"],["memory.py","optimized","tum"],["memory.py","v21","replica"],["memory.py","optimized","replica"],["graph_workspace_test.py"]]
states=[]
for args in tasks:
 name="_".join(args).replace(".py","")
 row=dict(task=args,state="running",start=time.time())
 states.append(row)
 print("START",args,flush=True)
 with (r/"logs"/(name+".log")).open("w") as f:
  p=subprocess.Popen([py,"-u",str(here/args[0]),*args[1:]],stdout=f,stderr=subprocess.STDOUT)
  row.update(pid=p.pid,start_ticks=Path(f"/proc/{p.pid}/stat").read_text().split()[21])
  (r/"reports/RUNNER_STATUS.json").write_text(json.dumps(states,indent=2))
  code=p.wait()
 row.update(exit=code,state="complete" if code==0 else "failed",end=time.time())
 (r/"reports/RUNNER_STATUS.json").write_text(json.dumps(states,indent=2))
 print("END",args,code,flush=True)
 if code:raise SystemExit(code)
print("ALL_PASS",flush=True)
