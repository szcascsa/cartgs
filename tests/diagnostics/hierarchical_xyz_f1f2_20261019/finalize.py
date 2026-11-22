"""空统计组输出空值；只重跑诊断/分析，ranking与render保持冻结。"""
from pathlib import Path
import json,shutil,subprocess,sys
ROOT=Path(__file__).resolve().parent
assert json.loads((ROOT/'status.json').read_text())['all_passed']
for scene in ['fr1_xyz','fr2_xyz']:
    out=ROOT/scene;before=(out/'mechanism_result.json').read_bytes()
    shutil.copy2(out/'diagnostics.log',out/'diagnostics_original.log')
    shutil.copy2(ROOT/'diagnostics.py',out/'diagnostics.py')
    for stage in ['diagnostics','analyze']:
        with open(out/(stage+'_final.log'),'w')as log:subprocess.run([sys.executable,str(out/(stage+'.py'))],cwd=out,stdout=log,stderr=subprocess.STDOUT,check=True)
    assert (out/'mechanism_result.json').read_bytes()==before,'Empty-group presentation changed mechanism result'
    print('FINAL_DIAGNOSTICS_PASS',scene,flush=True)
