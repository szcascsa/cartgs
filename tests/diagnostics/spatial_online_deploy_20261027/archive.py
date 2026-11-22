"""生产部署证据归档；旧二进制备份留服务器，不复制数据集与图像。"""
import hashlib,json,shutil,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parent;REPO=Path('/home/prmi/zwj/mycartgs')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    # 收录新生产实现及直接接入点。完整同步清单另存4395个文件SHA。
    names=['CMakeLists.txt','cfg/selector/selector_config.yaml','improvements/selector/gumbel_network.h',
        'improvements/selector/spatial_compensation.h','include/spatial_compensation_cuda.h','include/gaussian_mapper.h',
        'include/mapping_profiler.h','src/gaussian_mapper.cpp','src/spatial_compensation_cuda.cpp','src/spatial_compensation_cuda.cu',
        'src/spatial_compensation_live.cu','python/spatial_compensation.py','python/selector.py','python/run.py',
        'tests/anchor_selector/spatial_compensation_test.cpp','tests/anchor_selector/spatial_mapping_test.cpp',
        'tests/anchor_selector/spatial_live_benchmark.cpp','tests/anchor_selector/checkpoint_probe.cpp','tests/ultrafast_mapping_test.cpp']
    for name in names:
        target=ROOT/'source_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(REPO/name,target)
    files=[]
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file() or any(v in p.relative_to(ROOT).parts for v in ['backup','__pycache__']):continue
        if p.suffix in ['.zip','.gz','.tar','.png','.jpg','.jpeg','.mp4'] or p.name in ['artifact_manifest.json','archive_integrity.json','delivery_verification.json']:continue
        files.append(p)
    manifest=dict(files=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in files],source='working tree based on 74fc152; uncommitted production adaptation',backup_on_server=str(ROOT/'backup'))
    (ROOT/'artifact_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    path=ROOT/'spatial_production_evidence.tar.gz'
    with tarfile.open(path,'w:gz',compresslevel=3) as tar:
        for p in files+[ROOT/'artifact_manifest.json']:tar.add(p,arcname=str(p.relative_to(ROOT)),recursive=False)
    info=dict(file=path.name,sha256=sha(path),bytes=path.stat().st_size,files=len(files))
    (ROOT/'archive_integrity.json').write_text(json.dumps(info,indent=2));print(json.dumps(info),flush=True)
if __name__=='__main__':main()
