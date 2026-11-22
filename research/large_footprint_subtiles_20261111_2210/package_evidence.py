"""源码、加载so及哈希、逐样本证据与图表归档；不包含模型、数据或渲染RGB。"""
from pathlib import Path
import hashlib
import json
import tarfile
import time

HERE=Path(__file__).resolve().parent


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    integrity=json.loads((HERE/'reports/INTEGRITY_FINAL.json').read_text())
    assert integrity['passed'] and len(integrity['controllers'])==3
    frozen=json.loads((HERE/'reports/FROZEN.json').read_text())
    assert sha(Path(frozen['binary']))==frozen['binary_sha256']
    for name in ('FORMAL_CONTROLLER.json','POST_CHECKS.json','ATTRIBUTION_CONTROLLER.json'):
        assert json.loads((HERE/'reports'/name).read_text())['state']=='complete'
    for name in ('memcheck01','initcheck01','synccheck01'):
        assert 'ERROR SUMMARY: 0 errors' in (HERE/'logs'/(name+'.log')).read_text()
    files=[]
    for path in sorted(HERE.rglob('*')):
        if not path.is_file() or path.is_symlink():continue
        relative=path.relative_to(HERE)
        if '__pycache__' in relative.parts or path.suffix in ('.gz','.so','.o'):
            if path.suffix=='.gz' and not path.name.endswith('.tar.gz'):pass
            elif path.resolve()==Path(frozen['binary']).resolve():pass
            else:continue
        if path.name in ('DELIVERY_MANIFEST.json','DELIVERY_SHA256.json'):continue
        if relative.parts[0]=='build' and path.resolve()!=Path(frozen['binary']).resolve():continue
        files.append(path)
    rows=[dict(path=str(p.relative_to(HERE)),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
    manifest=HERE/'reports/DELIVERY_MANIFEST.json'
    manifest.write_text(json.dumps(dict(created_at=time.time(),files=rows,contains_models=False,
        contains_datasets=False,contains_rendered_images=False,previous_research_dependencies='见INPUT_CATALOG、INPUT_SHA256和BASELINE_SHA256',
        entry='LOW_RATIO_LARGE_FOOTPRINT_OPTIMIZATION_REPORT.md'),indent=2,ensure_ascii=False))
    archive=HERE/'reports/LOW_RATIO_LARGE_FOOTPRINT_FINAL_EVIDENCE.tar.gz';assert not archive.exists()
    with tarfile.open(archive,'w:gz',compresslevel=6) as bundle:
        for path in files+[manifest]:bundle.add(path,arcname=str(path.relative_to(HERE)),recursive=False)
    result=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),files=len(rows)+1)
    (HERE/'reports/DELIVERY_SHA256.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
