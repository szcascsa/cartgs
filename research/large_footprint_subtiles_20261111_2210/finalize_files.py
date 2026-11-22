"""交付前只处理本轮草稿归档，附源码补丁，不接触原研究基线。"""
from pathlib import Path
import difflib
import hashlib
import json
import time

HERE=Path(__file__).resolve().parent


def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    assert not (HERE/'reports/LOW_RATIO_LARGE_FOOTPRINT_FINAL_EVIDENCE.tar.gz').exists()
    path=HERE/'reports/CLEANUP.json';record=json.loads(path.read_text())
    files=[]
    for name in ('LOW_RATIO_LARGE_FOOTPRINT_EVIDENCE.tar.gz','figures_transfer.tar.gz','figures_final_transfer.tar.gz'):
        item=HERE/'reports'/name;assert item.is_file() and not item.is_symlink()
        assert item.resolve().is_relative_to((HERE/'reports').resolve())
        files.append(dict(path=str(item.resolve()),bytes=item.stat().st_size,sha256=sha(item),
            reason='本轮草稿归档或已提取图表传输包，被最终归档替代。'))
    assert files[0]['sha256']=='2ecc3459bbeb22d209766eec8e4a0a884ef6c0a88fe6c4d11de7079fee410b01'
    record.update(state='final_cleanup_planned',additional_files=files);path.write_text(json.dumps(record,indent=2,ensure_ascii=False))
    for item in files:Path(item['path']).unlink()
    record['files']+=files;record.pop('additional_files');record['total_bytes']=sum(item['bytes'] for item in record['files'])
    record.update(state='complete',finalized_at=time.time());path.write_text(json.dumps(record,indent=2,ensure_ascii=False))
    prior=HERE.parent/'render_count_scaling_20261111_2022/background_stage/source'
    current=HERE/'source';diff=[]
    names=sorted({str(p.relative_to(prior)) for p in prior.rglob('*') if p.is_file()}|
                 {str(p.relative_to(current)) for p in current.rglob('*') if p.is_file()})
    for name in names:
        a=prior/name;b=current/name
        old=a.read_text().splitlines(keepends=True) if a.exists() else []
        new=b.read_text().splitlines(keepends=True) if b.exists() else []
        diff.extend(difflib.unified_diff(old,new,fromfile='a/source/'+name,tofile='b/source/'+name))
    (HERE/'reports/FINAL_RESEARCH.patch').write_text(''.join(diff))
    print('FINAL_OWN_FILES',len(record['files']),record['total_bytes'],'PATCH_LINES',len(diff))


if __name__=='__main__':main()
