from pathlib import Path
import json,subprocess,sys,time
r=Path(__file__).resolve().parents[3];h=Path(__file__).resolve().parent
status=r/'reports/FINISH_STATUS.json'
try:
 for _ in range(600):
  state=json.loads((r/'reports/ANALYSIS_STATUS.json').read_text())
  if state['state']=='failed':raise RuntimeError(state)
  if state['state']=='complete':break
  time.sleep(2)
 else:raise TimeoutError('analysis still running')
 for step in ['identity','report','finalize']:
  status.write_text(json.dumps(dict(state='running',current=step)))
  subprocess.check_call([sys.executable,str(h/(step+'.py'))])
 status.write_text(json.dumps(dict(state='complete')))
except Exception as error:
 status.write_text(json.dumps(dict(state='failed',error=str(error))));raise
