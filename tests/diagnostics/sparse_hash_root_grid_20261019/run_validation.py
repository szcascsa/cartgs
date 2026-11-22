from pathlib import Path
import json,subprocess,os,traceback
ROOT=Path(__file__).resolve().parent
PY='/home/prmi/anaconda3/envs/zwj/bin/python';CM='/home/prmi/.local/lib/python3.13/site-packages/cmake/data/bin/cmake'
ENV=dict(os.environ,CUDA_HOME='/usr/local/cuda-11.8',PATH='/usr/local/cuda-11.8/bin:'+os.environ['PATH'])
def run(stage,args):
    (ROOT/'status.json').write_text(json.dumps({'stage':stage,'complete':False})+'\n')
    with open(ROOT/(stage+'.log'),'w')as out:subprocess.run(args,env=ENV,check=True,stdout=out,stderr=subprocess.STDOUT)
    print('PASS',stage,flush=True)
try:
    run('configure',[CM,'-S',str(ROOT),'-B',str(ROOT/'build'),'-DCMAKE_PREFIX_PATH=/home/prmi/zwj/mycartgs/third_party/libtorch-cu118/libtorch','-DCMAKE_CXX_COMPILER=/usr/bin/g++-9','-DCMAKE_BUILD_TYPE=Release','-DCUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-11.8'])
    run('build',[CM,'--build',str(ROOT/'build'),'-j1'])
    run('hierarchy_cpu',[str(ROOT/'build/hierarchy_test')])
    run('hierarchy_cuda',[str(ROOT/'build/hierarchy_test'),'cuda'])
    run('sparse_synthetic',[str(ROOT/'build/sparse_root_test')])
    run('global_gi_regression',[str(ROOT/'build/global_gi_regression')])
    run('mapper_compile',[PY,str(ROOT/'check_mapper_compile.py')])
    run('structural_fixture',[str(ROOT/'build/sparse_root_test'),str(ROOT/'fr1_room_input.bin'),str(ROOT/'fixture')])
    (ROOT/'status.json').write_text(json.dumps({'complete':True,'all_passed':True})+'\n')
except Exception:
    (ROOT/'status.json').write_text(json.dumps({'complete':True,'all_passed':False})+'\n')
    (ROOT/'failure.txt').write_text(traceback.format_exc());raise
