"""仅打包复现源码、二进制和实测证据，不复制模型/数据集/共享环境。"""
from pathlib import Path
import json,hashlib,tarfile,time

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
assert HERE==Path('/home/prmi/zwj/experiments/tensor_core_inference_20261110_2200/preparation_phase')
assert json.loads((HERE/'reports/INTEGRITY_FINAL.json').read_text())['passed']


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()


allowed={'.py','.cpp','.cu','.cuh','.h','.inc','.json','.jsonl','.csv','.md','.log','.png','.svg','.patch','.so','.nsys-rep','.txt'}
paths={str(p.relative_to(HERE)):p for p in HERE.rglob('*') if p.is_file() and p.suffix in allowed and
       p.name not in ['ARTIFACT_MANIFEST.json','LOCAL_DOWNLOAD_VERIFICATION.json'] and 'ARTIFACTS.sha256' not in p.name}
for name in ['build.ninja']:
    paths['build/'+name]=HERE/'build'/name
old=json.loads((ROOT/'reports/FROZEN_CANDIDATE.json').read_text())
paths['baselines/A/mycartgs_deployment_fp32_v31.so']=Path(old['baseline'])
paths['baselines/B/mycartgs_tensor_deployment_v1.so']=ROOT/'development_v1/mycartgs_tensor_deployment_v1.so'
paths['baselines/C/mycartgs_tensor_deployment_v1.so']=Path(old['binary'])
for label,root in [('B',ROOT/'development_v1/source'),('C',ROOT/'source')]:
    for p in root.iterdir():
        if p.is_file():paths[f'baselines/{label}/source/'+p.name]=p
for name in ['PROTECTED_BEFORE.json','FROZEN_CANDIDATE.json','RESULTS_SUMMARY.json','SANITIZER_ATTRIBUTION.json']:
    paths['baselines/C/evidence/'+name]=ROOT/'reports'/name
manifest=dict(created_at=time.time(),models_copied=False,datasets_copied=False,objects_included=False,
    files={name:dict(bytes=p.stat().st_size,sha256=sha(p)) for name,p in sorted(paths.items())})
man=HERE/'reports/ARTIFACT_MANIFEST.json';man.write_text(json.dumps(manifest,ensure_ascii=False,indent=2));paths['reports/ARTIFACT_MANIFEST.json']=man
archive=HERE/'TENSOR_CORE_PREPARATION_9H_ARTIFACTS.tar.gz'
with tarfile.open(archive,'w:gz',compresslevel=6) as tar:
    for name,p in sorted(paths.items()):tar.add(p,arcname=name,recursive=False)
summary=dict(archive=archive.name,bytes=archive.stat().st_size,sha256=sha(archive),files=len(paths))
(HERE/'TENSOR_CORE_PREPARATION_9H_ARTIFACTS.sha256.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
