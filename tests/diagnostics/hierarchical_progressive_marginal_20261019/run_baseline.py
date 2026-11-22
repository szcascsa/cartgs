from pathlib import Path
import subprocess,json,traceback
ROOT=Path(__file__).resolve().parent
try:
    for stage in ['prepare','generate_rankings','baseline']:
        (ROOT/'driver_status.json').write_text(json.dumps({'stage':stage,'complete':False})+'\n')
        with open(ROOT/(stage+'.log'),'w')as log:
            subprocess.run(['/home/prmi/anaconda3/envs/zwj/bin/python',str(ROOT/(stage+'.py'))],stdout=log,stderr=subprocess.STDOUT,check=True)
        print('PASS',stage,flush=True)
    (ROOT/'driver_status.json').write_text(json.dumps({'baseline_complete':True,'all_passed':True})+'\n')
except Exception:
    (ROOT/'driver_status.json').write_text(json.dumps({'all_passed':False,'failure':traceback.format_exc()})+'\n')
    raise
