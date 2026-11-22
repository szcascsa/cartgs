"""记录独立实现与已通过验收的身份，核验生产完整性并打包证据。"""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import time
from zipfile import ZipFile,ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[3]
HERE=ROOT/'source/research/view_budget_residual_20261105'


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def main():
    native=json.loads((ROOT/'reports/VALIDATION_STATUS.json').read_text())
    inference=json.loads((ROOT/'reports/INFERENCE_VALIDATION.json').read_text())
    pairing=json.loads((ROOT/'reports/PAIRING_VALIDATION.json').read_text())
    assert native['state']=='native_smoke_complete' and len(native['tests'])==16
    assert inference['passed'] and inference['cases']==54 and pairing['passed']
    before=json.loads((ROOT/'reports/PRODUCTION_BEFORE.json').read_text())
    after={p:dict(bytes=Path(p).stat().st_size,sha256=sha(Path(p))) for p in before}
    assert before==after,'受保护生产文件变化'
    (ROOT/'reports/PRODUCTION_AFTER.json').write_text(json.dumps(after,indent=2))
    built=json.loads((ROOT/'reports/BUILD_HASHES.json').read_text())
    assert all(sha(ROOT/name)==h for name,h in built.items()),'验收后二进制被改动'
    for p,h in inference['checkpoint_identity'].items():assert sha(Path(p))==h
    for group in ['off','view_no_li','view_li01']:
        manifest=json.loads((ROOT/'configs'/f'{group}.json').read_text())
        assert all(sha(Path(manifest[k]))==h for k,h in manifest['input_sha256'].items())
    images=[str(p) for p in (ROOT/'runs').rglob('*') if p.is_file() and p.suffix.lower() in ['.png','.jpg','.jpeg','.bmp']]
    assert not images
    names=set(json.loads((ROOT/'SOURCE_HASHES.json').read_text()))
    names.update(str(p.relative_to(ROOT/'source')) for p in HERE.iterdir() if p.is_file())
    sources={name:sha(ROOT/'source'/name) for name in sorted(names)}
    (ROOT/'reports/SOURCE_FINAL_HASHES.json').write_text(json.dumps(sources,indent=2))
    with tarfile.open(ROOT/'source_final.tar.gz','w:gz') as archive:
        for name in sources:archive.add(ROOT/'source'/name,arcname=name,recursive=False)
    for name in ['PROTOCOL.md','IMPLEMENTATION_REPORT.md']:shutil.copyfile(HERE/name,ROOT/'reports'/name)
    record=dict(passed=True,completed_unix=time.time(),protected_production_files=len(before),changed_production_files=[],
        native_tests=16,checkpoint_smoke_groups=3,steps_per_group=36,inference_cases=54,render_images_saved=0,
        binary_objects_unchanged=True,source_files=len(sources),free_GiB=shutil.disk_usage(ROOT).free/1024**3)
    (ROOT/'reports/FINAL_INTEGRITY.json').write_text(json.dumps(record,indent=2))
    archive=ROOT/'reports/VIEW_BUDGET_RESIDUAL_EVIDENCE.zip';assert not archive.exists()
    with ZipFile(archive,'w',ZIP_DEFLATED) as stream:
        for directory in ['reports','configs','logs']:
            for p in sorted((ROOT/directory).rglob('*')):
                if p.is_file() and p!=archive and p.suffix!='.lock':stream.write(p,p.relative_to(ROOT))
        for group in ['off','view_no_li','view_li01']:
            folder=ROOT/'runs'/f'smoke_{group}'
            for name in ['initialization.json','training_verified.json','training_keyframes.csv','real_lod_steps.csv','view_budget_steps.jsonl']:
                stream.write(folder/name,(folder/name).relative_to(ROOT))
            for p in (folder/'12016_shutdown/ply').rglob('*.json'):stream.write(p,p.relative_to(ROOT))
        for p in HERE.iterdir():
            if p.is_file():stream.write(p,Path('scripts')/p.name)
    print(json.dumps(dict(integrity=record,archive_bytes=archive.stat().st_size,archive_sha256=sha(archive)),indent=2))


if __name__=='__main__':main()
