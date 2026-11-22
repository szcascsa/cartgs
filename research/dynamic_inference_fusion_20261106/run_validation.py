from pathlib import Path
import subprocess,sys,json,time
r=Path(__file__).resolve().parents[3];here=Path(__file__).resolve().parent
state=dict(state='running',completed=[])
for name in ['validate_edges','validate_layers','validate_pipeline','validate_state','validate_topology','validate_purity','validate_dynamic']:
 state.update(current=name);(r/'reports/VALIDATION_STATUS.json').write_text(json.dumps(state))
 with (r/'logs'/f'{name}.log').open('w') as f:code=subprocess.call([sys.executable,str(here/(name+'.py'))],stdout=f,stderr=subprocess.STDOUT)
 if code:
  state.update(state='failed',returncode=code);(r/'reports/VALIDATION_STATUS.json').write_text(json.dumps(state));sys.exit(code)
 state['completed'].append(name)
state.update(state='complete');(r/'reports/VALIDATION_STATUS.json').write_text(json.dumps(state))
