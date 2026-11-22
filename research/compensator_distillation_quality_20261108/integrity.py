from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parents[3]
old=Path('/home/prmi/zwj/experiments/compensator_distillation_20261108')
result={}
for prefix in ['PRODUCTION','INPUTS','LEGACY_INPUTS','LARGE_INPUTS','STUDENT']:
 before=json.loads((ROOT/'reports'/f'{prefix}_BEFORE.json').read_text());after={};changed=[]
 for name,value in before.items():
  p=Path(name)
  digest=hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
  after[name]=dict(bytes=p.stat().st_size,sha256=digest) if isinstance(value,dict) and digest else digest
  if after[name]!=value:changed.append(name)
 (ROOT/'reports'/f'{prefix}_AFTER.json').write_text(json.dumps(after,indent=2))
 result[prefix]=dict(files=len(before),passed=not changed,changed=changed)
binary=old/'build/mycartgs_deployment_fp32_v31/mycartgs_deployment_fp32_v31.so'
expected='fa7b855685ecd4f75ce7118ee07fadd27640af08099a6775ffee5fe87fa7d601'
digest=hashlib.sha256(binary.read_bytes()).hexdigest()
result['DEPLOYMENT_BINARY']=dict(passed=digest==expected,sha256=digest,path=str(binary))
raster=Path('/home/prmi/zwj/experiments/global_view_joint_lod_20261105/evaluation/python/diff-gaussian-rasterization/diff_gaussian_rasterization/_C.cpython-311-x86_64-linux-gnu.so')
expected='27b5b468cf385dd1c96aeabdd65426da38a938a38d52d8181b1b1da485090c86'
digest=hashlib.sha256(raster.read_bytes()).hexdigest()
result['RASTERIZER_BINARY']=dict(passed=digest==expected,sha256=digest,path=str(raster))
report_figures={ROOT/'reports/COMPENSATOR_DISTILLATION_QUALITY.png', ROOT/'reports/TRAINING_IMAGE_MSE.png'}
result['report_figures']=[str(p) for p in sorted(report_figures) if p.exists()]
result['images_saved']=len([p for folder in ['models','reports'] for p in (ROOT/folder).rglob('*') if p.suffix.lower() in ['.png','.jpg','.jpeg'] and p not in report_figures])
result['passed']=all(v['passed'] for v in result.values() if isinstance(v,dict)) and result['images_saved']==0
(ROOT/'reports/FINAL_INTEGRITY.json').write_text(json.dumps(result,indent=2));assert result['passed'],result
manifest={str(p.relative_to(ROOT/'source')):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'source').rglob('*') if p.is_file() and '__pycache__' not in str(p)}
(ROOT/'reports/SOURCE_MANIFEST.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(result,indent=2))
