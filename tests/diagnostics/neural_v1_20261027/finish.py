"""仅等待已登记任务；顺序完成机制检查与汇总，不启动新研究分支。"""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent

def run(name,*args):subprocess.run([sys.executable,str(ROOT/name),*args],cwd=ROOT,check=True)

def main():
    plans=['confirm','reference_control','controls','fr2_confirm','baseline_sets','long36k','node_clock','folded','input_cause','repeat','fr2_local_followup']
    while True:
        missing=['plan_'+name+'_complete.json' for name in plans if not(ROOT/('plan_'+name+'_complete.json')).exists()]
        (ROOT/'pending.json').write_text(json.dumps(missing,indent=2))
        if not missing:break
        time.sleep(15)
    plan=dict(scene='fr1_xyz',source_seed=3100,models=[dict(directory=str(ROOT/'results'/tier/'fr1_xyz'/name/'5100')) for tier,name in [
        ('reference_control','Reference59TrainEncoder'),('confirm','SpatialFiLM'),('confirm','SpatialOutputKnots'),('controls','SpatialLocalXYZ')]])
    (ROOT/'mechanism_plan.json').write_text(json.dumps(plan,indent=2))
    run('mechanism_probe.py','--plan','mechanism_plan.json');run('summarize.py');run('final_audit.py');run('plots.py')
    (ROOT/'finish_complete.json').write_text(json.dumps(dict(all_registered_formal_jobs_complete=True,production_changed=False),indent=2))
    print('FINISH_COMPLETE',flush=True)

if __name__=='__main__':main()
