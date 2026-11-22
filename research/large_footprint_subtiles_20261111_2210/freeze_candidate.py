"""仅依据开发集冻结候选，不读取保留场景性能来调参。"""
from footprint_study import *

path=HERE/'reports/FROZEN.json'
assert not path.exists()
verify_baseline()
quality=json.loads((HERE/'reports/validation13/SUMMARY.json').read_text())
assert not quality['failed']
for job in ('screen_clean01','screen_clean02','screen_clean03','screen_clean04','screen_clean05','screen_clean06','screen_clean07','screen_clean08'):
    assert json.loads((HERE/'reports/jobs'/(job+'.json')).read_text())['measurement_valid'],job
files={str(p.relative_to(HERE)):sha(p) for p in (HERE/'source').rglob('*') if p.is_file()}
files['subtile_runtime.py']=sha(HERE/'subtile_runtime.py')
save(path,dict(frozen=True,mode=49,name='exact_common32',binary=extension().__file__,binary_sha256=sha(extension().__file__),
               sources=files,tune=TUNE,heldout=[s['id'] for s in SCENES if s['slug'] not in TUNE],
               scope='候选尚未接受；需跨场景双轮和极端footprint验证',
               comparison_mode=18,reason='精确保持原SASS FMA方向的横向公共乘积复用',time=time.time()))
print('RESEARCH_CANDIDATE_FROZEN',49,flush=True)
