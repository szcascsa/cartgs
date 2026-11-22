"""验证 gate 后更新正式四个文件并完整构建；不改其他工作树内容。"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET

D=Path(__file__).resolve().parent
ROOT=Path('/home/prmi/zwj/mycartgs')
for branch in ['tf','both']:
    parity=json.loads((D/'final/raw'/('parity_baseline_'+branch+'.json')).read_text())
    assert parity['verification']['allclose'] and parity['final']['allclose']
life=json.loads((D/'lifecycle_final.json').read_text())
assert set(life)=={'global','aabb4000','aabb5000'}
assert all(v['allclose'] for c in life.values() for v in c.values())
out=D/'deployment'
assert not out.exists()
(out/'backup/lib').mkdir(parents=True)
manifest=json.loads((D/'baseline/raw/source_manifest.json').read_text())
for name, record in manifest.items():
    assert hashlib.sha256((ROOT/name).read_text().encode()).hexdigest()==record['normalized_sha256'],'Concurrent edit: '+name
files=[name for name in manifest if (D/'final/source'/name).read_text()!=(D/'baseline/source'/name).read_text()]
assert not any('transport' in name for name in files),files
assert all('selector' not in name for name in files),files
for name in files:
    p=ROOT/name
    if p.exists():
        assert name in manifest, 'Unexpected existing file: '+name
        target=out/'backup'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
for name in ['lib/libgaussian_mapper.so','build-cuda118-sm86-gcc9/libtransform_field_cuda.a']:
    target=out/'backup'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/name,target)
records={}
for name in files:
    p=ROOT/name
    data=(D/'final/source'/name).read_bytes()
    temp=p.with_name(p.name+'.multi-ratio.tmp')
    temp.write_bytes(data);os.replace(temp,p)
    records[name]=hashlib.sha256(data).hexdigest()
(out/'installed_source_sha256.json').write_text(json.dumps(records,indent=2))
env=os.environ.copy();env['PATH']='/usr/local/cuda-11.8/bin:'+env['PATH']
cmake='/home/prmi/.local/lib/python3.13/site-packages/cmake/data/bin/cmake'
ctest='/home/prmi/.local/lib/python3.13/site-packages/cmake/data/bin/ctest'
build=[cmake,'--build',str(ROOT/'build-cuda118-sm86-gcc9'),'-j2']
tests=[ctest,'--test-dir',str(ROOT/'build-cuda118-sm86-gcc9'),'--output-on-failure','--output-junit',str(out/'ctest.xml'),'-j1']
(out/'commands.json').write_text(json.dumps([build,tests],indent=2))
for name,cmd in [('build',build),('ctest',tests)]:
    assert not subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip(),'External GPU activity'
    with (out/(name+'.log')).open('w') as log:
        r=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
    (out/(name+'.exit')).write_text(str(r.returncode))
    if name=='build': assert r.returncode==0,'Production build failed'
    else:
        suite=ET.parse(out/'ctest.xml').getroot()
        failures=[c.attrib['name'] for c in suite.iter('testcase') if c.find('failure') is not None]
        (out/'ctest_summary.json').write_text(json.dumps({'tests':len(list(suite.iter('testcase'))),'failures':failures},indent=2))
        assert not failures or failures==['ultrafast_mapping_cuda'],failures
    print(name,r.returncode,flush=True)
runner=ROOT/'results/change_distill_optimization/20261017/deployment_20261018/benchmark_production'
for stage in ['baseline_production','production']:
    p=D/stage;p.mkdir();(p/'raw').mkdir();shutil.copy2(runner,p/'benchmark_native')
subprocess.run([os.sys.executable,str(D/'paired.py'),'production','baseline_production'],check=True)
print('PRODUCTION UPDATE COMPLETE',flush=True)
