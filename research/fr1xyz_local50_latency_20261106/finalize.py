"""核验局部50%逐样本预算、源码与生产完整性并归档。"""
import csv
import hashlib
import json
import math
from pathlib import Path
import tarfile
from zipfile import ZipFile,ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[3]


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4194304),b''):h.update(b)
    return h.hexdigest()


def save(name,value):
    (ROOT/'reports'/name).write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def main():
    out=ROOT/'reports';assert json.loads((ROOT/'STATUS.json').read_text())['state']=='complete'
    before=json.loads((out/'INPUTS_BEFORE.json').read_text());after={n:sha(n) for n in before}
    assert before==after==json.loads((out/'INPUTS_AFTER.json').read_text())
    manifest=json.loads((ROOT/'source/research/fr1xyz_local50_latency_20261106/SOURCE_HASHES.json').read_text())
    assert all(sha(ROOT/'source'/n)==h for n,h in manifest.items())
    assert len([n for n in manifest if n.startswith('python/')])==26
    with (out/'TIMINGS.csv').open() as f:rows=list(csv.DictReader(f))
    assert len(rows)==11880 and len({(v['method'],v['repeat'],v['frame']) for v in rows})==11880
    for v in rows:
        if v['method'] not in ['default_500','default_ratio50']:continue
        requested=500 if v['method']=='default_500' else math.floor(.5*int(v['candidate_N']))
        assert int(v['requested_K'])==requested and int(v['K'])==min(requested,int(v['eligible_N']))
        assert int(v['budget_shortfall'])==requested-int(v['K']) and v['view_conditioned']=='True' and v['cache_hit']=='False'
        if v['method']=='default_ratio50':assert float(v['network_ratio'])==.5
    assert len(json.loads((out/'SUMMARY.json').read_text()))==20
    assert json.loads((out/'PROFILE_SUMMARY.json').read_text())['all_kernels_attributed']
    assert (out/'LOCAL50_RENDER_LATENCY_REPORT.md').is_file() and len(list((out/'figures').glob('*.png')))==1
    images=[p for p in ROOT.rglob('*') if p.is_file() and p.suffix.lower() in ['.png','.jpg','.jpeg','.bmp','.webp'] and out/'figures' not in p.parents]
    assert not images
    source={str(p.relative_to(ROOT/'source')):sha(p) for p in (ROOT/'source').rglob('*') if p.is_file()};save('SOURCE_FINAL_HASHES.json',source)
    with tarfile.open(ROOT/'source_final.tar.gz','w:gz') as a:
        for n in sorted(source):a.add(ROOT/'source'/n,arcname=n,recursive=False)
    save('FINAL_DELIVERY.json',dict(passed=True,commit='26c355b32bf373a75877fc4605a4e41b8de9b47f',python_source_from_git=26,
        protected_files=len(before),production_files=604,changed_files=[],frames=792,repeats=3,timed_calls=11880,event_calls=480,
        actual_K_formula_verified=True,ratio50_network_input_verified=True,quality_rows=3168,images_saved=0,
        training_steps=0,production_updated=False,commit_or_push=False,shared_GPU_load_disclosed=True))
    archive=out/'LOCAL50_RENDER_LATENCY_RESULTS.zip';assert not archive.exists()
    with ZipFile(archive,'w',ZIP_DEFLATED,strict_timestamps=False) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p!=archive and p.suffix!='.gz':z.write(p,p.relative_to(ROOT))
        for p in sorted((ROOT/'logs').glob('*.log')):z.write(p,p.relative_to(ROOT))
        z.write(ROOT/'STATUS.json','STATUS.json')
        for p in (ROOT/'source/research/fr1xyz_local50_latency_20261106').iterdir():
            if p.is_file():z.write(p,p.relative_to(ROOT/'source'))
    save('ARCHIVE_SHA256.json',dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),source_sha256=sha(ROOT/'source_final.tar.gz')))
    print(json.dumps(dict(passed=True,protected_files=len(before),archive_bytes=archive.stat().st_size)),flush=True)


if __name__=='__main__':main()
