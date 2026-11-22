import time,subprocess,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
while not(ROOT/'complete.json').exists():
    if(ROOT/'postprocess_failure.json').exists():raise RuntimeError('postprocess_failure.json')
    time.sleep(5)
for stage in ['verify.py','pack.py']:
    with open(ROOT/(stage+'.log'),'w')as f:r=subprocess.run(['/home/prmi/anaconda3/envs/zwj/bin/python',str(ROOT/stage)],stdout=f,stderr=subprocess.STDOUT)
    assert r.returncode==0,stage
