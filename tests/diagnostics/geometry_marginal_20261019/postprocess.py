import time,subprocess,os,json,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
try:
    while not(ROOT/'online_complete.json').exists():
        if(ROOT/'online_failure.json').exists():raise RuntimeError('online_failure.json; diagnose before postprocess')
        time.sleep(5)
    for stage in ['benchmark.py','analyze_snapshots.py','reports.py']:
        while not(ROOT/stage).exists():time.sleep(3)
        with open(ROOT/(stage+'.log'),'w')as f:r=subprocess.run(['/home/prmi/anaconda3/envs/zwj/bin/python',str(ROOT/stage)],stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2'))
        assert r.returncode==0,stage
    (ROOT/'complete.json').write_text(json.dumps(dict(all_passed=True)))
except BaseException:(ROOT/'postprocess_failure.json').write_text(json.dumps(dict(traceback=traceback.format_exc())));raise
