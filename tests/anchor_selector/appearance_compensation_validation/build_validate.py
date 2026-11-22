"""隔离编译和功能门控；任何失败即停止，不进入训练。"""
import json
import os
from pathlib import Path
import subprocess
import sys

root=Path('/home/prmi/zwj/shared-selector-appearance-20261021')
repo=root/'source';out=root/'validation';build=root/'build'
env=dict(os.environ,PATH='/usr/local/cuda-11.8/bin:/home/prmi/anaconda3/envs/zwj/bin:'+os.environ['PATH'],
         OMP_NUM_THREADS='2',QT_QPA_PLATFORM='offscreen',MPLBACKEND='Agg')
def run(name,cmd):
    (out/'build_status.json').write_text(json.dumps(dict(stage=name)))
    with (out/(name+'.log')).open('w') as log:
        subprocess.run(list(map(str,cmd)),cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('PASSED',name,flush=True)
try:
    run('configure',['/home/prmi/.local/bin/cmake','-S',repo,'-B',build,'-DCMAKE_BUILD_TYPE=Release',
        '-DCMAKE_CUDA_ARCHITECTURES=86','-DCMAKE_CXX_COMPILER=/usr/bin/g++-9',
        '-DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc',
        '-DTorch_DIR=/home/prmi/zwj/mycartgs/third_party/libtorch-cu118/libtorch/share/cmake/Torch',
        '-DOpenCV_DIR=/home/prmi/zwj/mycartgs/third_party/install/opencv-cuda118-sm86-gcc9/lib/cmake/opencv4',
        '-Djsoncpp_DIR=/home/prmi/zwj/mycartgs/third_party/install/jsoncpp-1.9.5/lib/cmake/jsoncpp',
        '-DEigen3_DIR=/home/prmi/zwj/mycartgs/third_party/eigen-3.3.7/build-cmake',
        '-DCARTGS_BUILD_TESTS=ON','-DCARTGS_EXPERIMENTAL_SELECTOR_CUDA=ON'])
    targets=['selector_compensation_test','selector_compensation_mapping_test','variable_anchor_test',
             'variable_anchor_mapping_test','anchor_change_distill_test','anchor_checkpoint_probe','tum_rgbd']
    run('build',['/home/prmi/.local/bin/cmake','--build',build,'-j2','--target']+targets)
    for device in ['cpu','cuda']:
        run('compensation_'+device,[repo/'bin/selector_compensation_test',device])
        run('python_'+device,[sys.executable,repo/'tests/anchor_selector/selector_compensation_python_test.py',device])
        run('variable_'+device,[repo/'bin/variable_anchor_test',device])
    for device in ['cpu','cuda']:
        run('v1_'+device,[repo/'bin/selector_compensation_test',device,'legacy'])
        run('v1_python_'+device,[sys.executable,repo/'tests/anchor_selector/selector_compensation_python_test.py',device])
    run('v1_real',[sys.executable,root/'verify_v1.py'])
    run('mapping',[repo/'bin/selector_compensation_mapping_test','--parity'])
    run('legacy_mapping',[repo/'bin/variable_anchor_mapping_test','--parity'])
    run('legacy_cd',[repo/'bin/anchor_change_distill_test','attributes'])
    (out/'correctness.json').write_text(json.dumps(dict(passed=True,tests=14),indent=2))
    (out/'build_status.json').write_text(json.dumps(dict(stage='CORRECTNESS_PASSED')))
except Exception as error:
    (out/'build_status.json').write_text(json.dumps(dict(stage='FAILED',error=str(error))))
    raise
