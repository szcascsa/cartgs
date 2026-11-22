from pathlib import Path
import subprocess,sys,json,time
r=Path(__file__).resolve().parents[3];h=Path(__file__).resolve().parent
jobs=[['validate_topology'],['validate_summary'],['benchmark_dynamic','1'],['benchmark_dynamic','2'],['benchmark_dynamic','3'],['microbenchmark'],['benchmark_viewer'],['memory','baseline'],['memory','fused'],['unit_tests']]
state=json.loads((r/'reports/ANALYSIS_STATUS.json').read_text()) if (r/'reports/ANALYSIS_STATUS.json').exists() else dict(completed=[])
state['state']='running'
state.pop('returncode',None)
for job in jobs:
 label='_'.join(job)
 if label in state['completed']:continue
 state['current']=label;(r/'reports/ANALYSIS_STATUS.json').write_text(json.dumps(state))
 with (r/'logs'/(label+'.log')).open('w') as f:code=subprocess.call([sys.executable,str(h/(job[0]+'.py')),*job[1:]],stdout=f,stderr=subprocess.STDOUT)
 if code:
  state.update(state='failed',returncode=code);(r/'reports/ANALYSIS_STATUS.json').write_text(json.dumps(state));sys.exit(code)
 state['completed'].append(label)
state['current']='nsight'
cmd=['/usr/local/cuda-11.8/bin/nsys','profile','--force-overwrite=true','--trace=cuda,nvtx,osrt','--sample=none','--cpuctxsw=none','-o',str(r/'reports/PROFILE'),sys.executable,str(h/'profile.py')]
with (r/'logs/nsight.log').open('w') as f:code=subprocess.call(cmd,stdout=f,stderr=subprocess.STDOUT)
if code:raise RuntimeError('nsight '+str(code))
subprocess.check_call(['/usr/local/cuda-11.8/bin/nsys','export','--force-overwrite=true','--type=sqlite','-o',str(r/'reports/PROFILE.sqlite'),str(r/'reports/PROFILE.nsys-rep')])
subprocess.check_call([sys.executable,str(h/'analyze_profile.py')])
state.update(state='complete',current=None);(r/'reports/ANALYSIS_STATUS.json').write_text(json.dumps(state))
