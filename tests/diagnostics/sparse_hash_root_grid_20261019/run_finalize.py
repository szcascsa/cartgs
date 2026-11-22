from pathlib import Path
import subprocess,json,shutil,traceback
ROOT=Path(__file__).resolve().parent
CM='/home/prmi/.local/lib/python3.13/site-packages/cmake/data/bin/cmake'
PY='/home/prmi/anaconda3/envs/zwj/bin/python'
def run(stage,args):
    with open(ROOT/(stage+'.log'),'w') as log:
        subprocess.run(args,check=True,stdout=log,stderr=subprocess.STDOUT)
    print('PASS',stage,flush=True)
if (ROOT/'analysis.log').exists() and not (ROOT/'analysis_before_json_scalar_fix.log').exists():
    shutil.copyfile(ROOT/'analysis.log',ROOT/'analysis_before_json_scalar_fix.log')
try:
    run('final_build',[CM,'--build',str(ROOT/'build'),'-j1'])
    run('final_sparse_synthetic',[str(ROOT/'build/sparse_root_test')])
    run('final_hierarchy_cpu',[str(ROOT/'build/hierarchy_test')])
    run('final_hierarchy_cuda',[str(ROOT/'build/hierarchy_test'),'cuda'])
    run('final_structural_fixture',[str(ROOT/'build/sparse_root_test'),str(ROOT/'fr1_room_input.bin'),str(ROOT/'fixture')])
    run('analysis',[PY,str(ROOT/'analyze_structure.py')])
    (ROOT/'final_status.json').write_text(json.dumps({'complete':True,'all_passed':True})+'\n')
except Exception:
    (ROOT/'final_status.json').write_text(json.dumps({'complete':True,'all_passed':False,'failure':traceback.format_exc()})+'\n')
    raise
