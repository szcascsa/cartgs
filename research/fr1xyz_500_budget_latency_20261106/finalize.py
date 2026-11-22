"""核验输入及逐样本数据，归档测时结果；不含模型或渲染图片。"""
import csv
import hashlib
import json
from pathlib import Path
import tarfile
from zipfile import ZipFile,ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[3]


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()


def save(name,value):
    (ROOT/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def main():
    r=ROOT/'reports';status=json.loads((ROOT/'STATUS.json').read_text());assert status['state']=='complete'
    before=json.loads((r/'INPUTS_BEFORE.json').read_text());after={n:sha(n) for n in before};assert before==after
    assert after==json.loads((r/'INPUTS_AFTER.json').read_text())
    with (r/'TIMINGS.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==11880 and len({(v['method'],v['repeat'],v['frame']) for v in rows})==11880
    assert all(int(v['K'])==500 and v['cache_hit']=='False' for v in rows if v['method'] in ['old_joint_500','default_500'])
    summary=json.loads((r/'SUMMARY.json').read_text());assert len(summary)==20
    assert json.loads((r/'PROFILE_SUMMARY.json').read_text())['all_kernels_attributed']
    assert (r/'LOCAL500_RENDER_LATENCY_REPORT.md').exists() and len(list((r/'figures').glob('*.png')))==2
    images=[p for p in ROOT.rglob('*') if p.is_file() and p.suffix.lower() in ['.jpg','.jpeg','.png','.bmp','.webp'] and r/'figures' not in p.parents]
    assert not images,images
    source={str(p.relative_to(ROOT/'source')):sha(p) for p in (ROOT/'source').rglob('*') if p.is_file()}
    save('reports/SOURCE_FINAL_HASHES.json',source)
    with tarfile.open(ROOT/'source_final.tar.gz','w:gz') as a:
        for n in sorted(source):a.add(ROOT/'source'/n,arcname=n,recursive=False)
    save('reports/FINAL_DELIVERY.json',dict(passed=True,protected_files=len(before),production_files=604,changed_files=[],
        timed_calls=11880,frames=792,repeats=3,event_calls=480,quality_calls=3168,methods=5,
        strict500_every_call=True,old_and_current_default_measured=True,images_saved=0,training_steps=0,
        production_code_updated=False,commit_or_push=False,shared_load_disclosed=True))
    archive=r/'LOCAL500_RENDER_LATENCY_RESULTS.zip';assert not archive.exists()
    with ZipFile(archive,'w',ZIP_DEFLATED) as z:
        for p in sorted(r.rglob('*')):
            if p.is_file() and p!=archive and p.suffix!='.gz':z.write(p,p.relative_to(ROOT))
        for p in sorted((ROOT/'logs').glob('*.log')):z.write(p,p.relative_to(ROOT))
        z.write(ROOT/'STATUS.json','STATUS.json')
        for p in sorted((ROOT/'source/research/fr1xyz_500_budget_latency_20261106').iterdir()):
            if p.is_file():z.write(p,p.relative_to(ROOT/'source'))
    save('reports/ARCHIVE_SHA256.json',dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),
        source_sha256=sha(ROOT/'source_final.tar.gz')))
    print(json.dumps(dict(passed=True,archive_bytes=archive.stat().st_size,source_files=len(source))),flush=True)


if __name__=='__main__':main()
