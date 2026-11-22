"""编译结束后正反顺序比较，每个分支每版本 10 warm samples。"""
from pathlib import Path
import json
import statistics
import subprocess
import sys
D=Path(__file__).resolve().parent
compiler_names={'cc1plus','nvcc','cicc','ptxas','g++','cmake','ctest'}
ps=subprocess.check_output(['ps','-eo','comm='],text=True).splitlines()
# run.py applies a sustained idle gate and monitors the full run.
candidate=sys.argv[1] if len(sys.argv)>1 else 'final'
baseline=sys.argv[2] if len(sys.argv)>2 else 'baseline'
prefix='paired_'+candidate
for direction,stages in [('forward',[baseline,candidate]),('reverse',[candidate,baseline])]:
    for branch in ['tf','selector','both']:
        for stage in stages:
            subprocess.run([sys.executable,str(D/'run.py'),f'{stage}:{branch}:{prefix}_{direction}_{branch}:native:5'],check=True)
result={}
for stage in [baseline,candidate]:
    result[stage]={}
    for branch in ['tf','selector','both']:
        rows=[]
        for direction in ['forward','reverse']:
            rows.extend(r for r in json.loads((D/stage/'raw'/f'{prefix}_{direction}_{branch}/runs.json').read_text()) if r['label'].startswith('warm_'))
        assert len(rows)==10
        for r in rows:
            assert r['batch_size']==8192 and r['ratio_count']==4 and r['epochs']==5
            assert r['selector_steps']==(20 if branch!='tf' else 0)
            assert r['tf_steps']==(20 if branch!='selector' else 0)
            assert r['identical_initial_parameters'] and r['gaussian_unchanged'] and r['normal_optimizer_fresh_and_unchanged']
        rec={}
        for field in ['cpu_wall_ms','cuda_elapsed_ms','peak_allocated_bytes','additional_allocated_bytes']:
            values=[r[field] for r in rows]
            rec[field]={'median':statistics.median(values),'min':min(values),'max':max(values),'std':statistics.pstdev(values),'n':len(values)}
        result[stage][branch]=rec
(D/('paired_'+candidate+'_summary.json')).write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
