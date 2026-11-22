from pathlib import Path
import subprocess,sys,time,json
ROOT=Path(__file__).resolve().parent
start=time.monotonic()
while not (ROOT/'independent_checks.json').exists():
    if time.monotonic()-start>1800:raise RuntimeError('独立验证未完成，停止打包')
    time.sleep(10)
assert json.loads((ROOT/'independent_checks.json').read_text())['all_passed']
for stage in ['report','pack']:
    with open(ROOT/(stage+'.log'),'w')as log:subprocess.run([sys.executable,str(ROOT/(stage+'.py'))],stdout=log,stderr=subprocess.STDOUT,check=True,cwd=ROOT)
print('FINAL_ARTIFACTS_READY',flush=True)
