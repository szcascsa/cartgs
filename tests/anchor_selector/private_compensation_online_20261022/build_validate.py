"""构建准确的本地源码副本，过native/Python门控后才允许真实SLAM。"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

root=Path('/home/prmi/zwj/private-compensation-online-20261022');repo=root/'source';out=root/'validation';out.mkdir(exist_ok=True)
env=dict(os.environ,OMP_NUM_THREADS='2',QT_QPA_PLATFORM='offscreen',MPLBACKEND='Agg',PATH='/usr/local/cuda-11.8/bin:/home/prmi/anaconda3/envs/zwj/bin:'+os.environ['PATH'])
def run(name,cmd):
    (out/'build_status.json').write_text(json.dumps(dict(stage=name)))
    with (out/(name+'.log')).open('w') as log:subprocess.run(list(map(str,cmd)),cwd=out,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('PASS',name,flush=True)

if not repo.exists():
    repo.mkdir()
    with tarfile.open(root/'source.tar.gz') as archive:archive.extractall(repo,filter='data')
    (repo/'third_party').symlink_to('/home/prmi/zwj/mycartgs/third_party',target_is_directory=True)
    for name in ['cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg1_xyz.yaml','cfg/ORB_SLAM3/RGB-D/TUM/associations/fr1_xyz.txt']:
        destination=repo/name;destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(Path('/home/prmi/zwj/budget-shared-audit-20261022/source')/name,destination)
    old=Path('/home/prmi/zwj/budget-shared-audit-20261022')
    shutil.copytree(old/'build',root/'build',symlinks=True)
    for path in (root/'build').rglob('*'):
        if path.is_file() and not path.is_symlink() and path.suffix in ['.txt','.cmake','.make','.ninja','.internal','.ts','.d']:
            try:s=path.read_text()
            except UnicodeDecodeError:continue
            if str(old) in s:path.write_text(s.replace(str(old),str(root)))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
manifest=json.loads((root/'source_manifest.json').read_text())
assert all(sha(repo/p)==h for p,h in manifest.items())
run('configure',['/home/prmi/.local/bin/cmake','-S',repo,'-B',root/'build','-DCMAKE_BUILD_TYPE=Release','-DCMAKE_CUDA_ARCHITECTURES=86',
    '-DCMAKE_CXX_COMPILER=/usr/bin/g++-9','-DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc',
    '-DTorch_DIR=/home/prmi/zwj/mycartgs/third_party/libtorch-cu118/libtorch/share/cmake/Torch',
    '-DOpenCV_DIR=/home/prmi/zwj/mycartgs/third_party/install/opencv-cuda118-sm86-gcc9/lib/cmake/opencv4',
    '-Djsoncpp_DIR=/home/prmi/zwj/mycartgs/third_party/install/jsoncpp-1.9.5/lib/cmake/jsoncpp',
    '-DEigen3_DIR=/home/prmi/zwj/mycartgs/third_party/eigen-3.3.7/build-cmake','-DCARTGS_BUILD_TESTS=ON','-DCARTGS_EXPERIMENTAL_SELECTOR_CUDA=ON'])
run('compile',['/home/prmi/.local/bin/cmake','--build',root/'build','-j2','--target','tum_rgbd','private_compensation_test',
    'selector_compensation_test','selector_compensation_mapping_test','local_anchor_migration_test','variable_anchor_test','private_compensation_mapping_test','anchor_checkpoint_probe'])
for device in ['cpu','cuda']:
    run('private_'+device,[repo/'bin/private_compensation_test',device])
    run('private_python_'+device,[sys.executable,repo/'tests/anchor_selector/private_compensation_python_test.py',device])
    run('legacy_'+device,[repo/'bin/selector_compensation_test',device])
run('legacy_mapping',[repo/'bin/selector_compensation_mapping_test','--parity'])
run('migration',[repo/'bin/local_anchor_migration_test','cuda'])
run('variable',[repo/'bin/variable_anchor_test','cuda'])
run('private_mapping',[repo/'bin/private_compensation_mapping_test','cuda'])
(out/'correctness.json').write_text(json.dumps(dict(passed=True,native_cpu_cuda=True,python_parity=True,legacy_regression=True,
    identity_lifecycle=True,gaussian_transport=True,source_sha256=manifest,online_training='subsequent run only')))
(out/'build_status.json').write_text(json.dumps(dict(stage='PASSED')))
print('ALL_CORRECTNESS_PASSED',flush=True)
