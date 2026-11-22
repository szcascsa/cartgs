"""严格allowlist源码部署，保留旧库/源码；本任务不构建自定义kernel。"""
import os,subprocess,shutil,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parent;REPO=Path('/home/prmi/zwj/mycartgs')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert json.loads((ROOT/'optimized_exact_validation.json').read_text())['all_passed']
spec=json.loads((ROOT/'source_deployment.json').read_text());before={}
for name,item in spec.items():
    p=REPO/name
    if item['before'] is not None:assert sha(p)==item['before'],('REMOTE_SOURCE_DRIFT',name)
    else:assert not p.exists(),('NEW_FILE_ALREADY_EXISTS',name)
for name in spec:
    p=REPO/name;target=ROOT/'source_before'/name;target.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():shutil.copy2(p,target);before[name]=sha(p)
runtime=ROOT/'runtime_before';(runtime/'lib').mkdir(parents=True,exist_ok=True);(runtime/'bin').mkdir(exist_ok=True)
for p in (REPO/'lib').glob('*.so*'):
    if p.is_file():shutil.copy2(p,runtime/'lib'/p.name)
shutil.copy2(REPO/'bin/tum_rgbd',runtime/'bin/tum_rgbd')
for name,item in spec.items():
    p=REPO/name;p.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/'source_after'/name,p);assert sha(p)==item['after']
(ROOT/'deployment_checks.json').write_text(json.dumps(dict(before=before,after={n:sha(REPO/n)for n in spec},prior_libraries={p.name:sha(p)for p in(runtime/'lib').iterdir()}),indent=2))
cmake='/home/prmi/.local/lib/python3.13/site-packages/cmake/data/bin/cmake';build=REPO/'build-cuda118-sm86-gcc9'
with open(ROOT/'production_build.log','w')as f:
    r=subprocess.run([cmake,'--build',str(build),'--target','tum_rgbd','-j2'],stdout=f,stderr=subprocess.STDOUT,env=dict(os.environ,CC='/usr/bin/gcc-9',CXX='/usr/bin/g++-9'))
assert r.returncode==0,'production_build.log'
(ROOT/'build_complete.json').write_text(json.dumps(dict(all_passed=True,binary_sha=sha(REPO/'bin/tum_rgbd'),library_sha=sha(REPO/'lib/libgaussian_mapper.so'))))
print('PRODUCTION_BUILD_PASS',flush=True)
