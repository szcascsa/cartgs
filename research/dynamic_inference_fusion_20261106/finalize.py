from pathlib import Path,PureWindowsPath
import hashlib,json,tarfile,zipfile,shutil,time,subprocess,sys
ROOT=Path(__file__).resolve().parents[3];r=ROOT/'reports'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False))
assert json.loads((r/'VALIDATION_STATUS.json').read_text())['state']=='complete'
assert json.loads((r/'ANALYSIS_STATUS.json').read_text())['state']=='complete'
checks={}
for name in ['PROBE','QUATERNION_PROBE','EDGE_VALIDATION','LAYER_VALIDATION','PIPELINE_VALIDATION','STATE_VALIDATION','TOPOLOGY_VALIDATION','PURITY_VALIDATION','DYNAMIC_VALIDATION','SUMMARY_VALIDATION','UNIT_VALIDATION','VIEWER_BENCHMARK']:
 v=json.loads((r/(name+'.json')).read_text());assert v['passed'],name;checks[name]=v.get('comparisons',v.get('tests'))
assert checks['DYNAMIC_VALIDATION']==150480 and checks['UNIT_VALIDATION']==24
for j in [1,2,3]:
 v=json.loads((r/f'DYNAMIC_BENCHMARK_{j}.json').read_text());assert len(v['results'])==5
 for row in v['results']:assert row['summary']['fused']['median_ms']<row['summary']['baseline']['median_ms'],(j,row['case'])
for prefix in ['PRODUCTION','INPUTS']:
 before=json.loads((r/(prefix+'_BEFORE.json')).read_text());after={}
 for p,v in before.items():after[p]=dict(bytes=Path(p).stat().st_size,sha256=sha(p)) if isinstance(v,dict) else sha(p)
 assert before==after,prefix;save(r/(prefix+'_AFTER.json'),after)
legacy=json.loads(Path('/home/prmi/zwj/experiments/joint_view_inference_20261106/reports/LEGACY_CHECKPOINT_INTEGRITY.json').read_text())['files']
assert all(sha(p)==h for p,h in legacy.items());save(r/'LEGACY_INPUTS_AFTER.json',legacy)
local=json.loads((r/'LOCAL_BEFORE.json').read_text())
for name,h in local.items():assert sha(ROOT/'reference/python'/PureWindowsPath(name).name)==h,name
identity=json.loads((r/'BINARY_IDENTITY.json').read_text());assert all(sha(ROOT/'source/python'/n)==h for n,h in identity['source_hashes'].items())
files=[p for p in (ROOT/'source').rglob('*') if p.is_file() and p.suffix in ['.py','.cu','.cpp','.html'] and '__pycache__' not in p.parts]
source={str(p.relative_to(ROOT/'source')):sha(p) for p in files};save(r/'SOURCE_FINAL_HASHES.json',source)
with tarfile.open(ROOT/'source_final.tar.gz','w:gz') as t:
 for p in sorted(files):t.add(p,arcname=str(p.relative_to(ROOT/'source')),recursive=False)
images=[str(p) for p in (ROOT/'evaluation').rglob('*') if p.suffix.lower() in ['.png','.jpg','.jpeg','.webp']];assert not images
assert (r/'DYNAMIC_INFERENCE_FUSION_OPTIMIZATION_REPORT.md').exists()
result=dict(passed=True,checks=checks,protected_production_files=604,production_changes=[],protected_input_files=42,additional_legacy_input_files=len(legacy),images_saved=0,training_runs=0,
 production_deployed=False,commit=False,push=False,free_GiB=shutil.disk_usage(ROOT).free/1024**3,completed_unix=time.time(),source_archive_sha256=sha(ROOT/'source_final.tar.gz'))
save(r/'FINAL_INTEGRITY.json',result)
save(r/'FINISH_STATUS.json',dict(state='complete'))
dest=ROOT/'DYNAMIC_INFERENCE_FUSION_RESULTS.zip'
with zipfile.ZipFile(dest,'w',zipfile.ZIP_DEFLATED) as z:
 for directory in ['reports','logs','reference','evaluation']:
  for p in (ROOT/directory).rglob('*'):
   if p.is_file() and p != r/'ARCHIVE_SHA256.json' and p.suffix.lower() not in ['.pt','.ply','.sqlite','.nsys-rep'] and '__pycache__' not in p.parts:z.write(p,p.relative_to(ROOT))
 z.write(ROOT/'source_final.tar.gz','source_final.tar.gz')
save(r/'ARCHIVE_SHA256.json',dict(path=str(dest),sha256=sha(dest),bytes=dest.stat().st_size))
print(json.dumps(result),flush=True)
