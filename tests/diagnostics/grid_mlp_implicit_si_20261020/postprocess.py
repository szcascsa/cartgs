"""本次任务的顺序pipeline；等训练完成后锁定集合、评价、汇总，不启动后续研究。"""
from pathlib import Path
import json,subprocess,sys,time
root=Path(__file__).resolve().parent
try:
    while True:
        state=json.loads((root/'status.json').read_text())
        if state['stage']in ['failed','optimization_gate_failed']:raise RuntimeError(state)
        if state['stage']=='training_complete':break
        time.sleep(5)
    for script in ['collect.py','evaluate.py','summarize.py']:
        with open(root/(script.removesuffix('.py')+'.log'),'w')as out:
            subprocess.run([sys.executable,str(root/script)],stdout=out,stderr=subprocess.STDOUT,check=True)
    (root/'status.json').write_text(json.dumps(dict(stage='analysis_complete'))+'\n')
except Exception as error:
    (root/'status.json').write_text(json.dumps(dict(stage='failed',error=str(error)))+'\n');raise
