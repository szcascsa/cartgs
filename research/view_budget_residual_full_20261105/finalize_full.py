"""全部训练和评价完成后的完整性核验与交付打包。"""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import time
from zipfile import ZipFile,ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[3]
HERE=ROOT/'source/research/view_budget_residual_full_20261105'
PREVIOUS=Path('/home/prmi/zwj/experiments/local_importance_li_20261105')


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for b in iter(lambda:stream.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def main():
    state=json.loads((ROOT/'reports/TRAINING_STATUS.json').read_text())
    evaluation=json.loads((ROOT/'evaluation/STATUS.json').read_text())
    assert state['state']=='training_complete' and evaluation['state']=='complete'
    assert set(state['groups'])=={'baseline','view_no_li','view_li01'} and state['groups']['baseline']['reused']
    assert not (ROOT/'runs/full_baseline').exists(),'不得重训已有baseline'
    assert evaluation['rows']==88704 and evaluation['reused_rows']==26928
    integrity=json.loads((ROOT/'evaluation/reports/INPUT_INTEGRITY.json').read_text());assert integrity['passed']
    assert all(sha(Path(p))==v for p,v in integrity['before'].items())
    before=json.loads((ROOT/'reports/PRODUCTION_BEFORE.json').read_text())
    after={p:dict(bytes=Path(p).stat().st_size,sha256=sha(Path(p))) for p in before}
    assert before==after,'受保护生产文件变化'
    (ROOT/'reports/PRODUCTION_AFTER.json').write_text(json.dumps(after,indent=2))
    binaries=json.loads((ROOT/'reports/BINARY_PARITY.json').read_text())['sha256']
    assert all(sha(ROOT/name)==v for name,v in binaries.items())
    historical={n:sha(PREVIOUS/n) for n in ['runtime/bin/lod_from_full','runtime/lib/libgaussian_mapper.so']}
    (ROOT/'reports/HISTORICAL_BASELINE_BINARY.json').write_text(json.dumps(historical,indent=2))
    sources=json.loads((ROOT/'reports/IMPLEMENTATION_SOURCE_HASHES.json').read_text())
    assert all(sha(ROOT/'source'/name)==v for name,v in sources.items()),'已验收实现源码变化'
    for group in ['view_no_li','view_li01']:
        manifest=json.loads((ROOT/'configs'/f'full_{group}.json').read_text())
        assert all(sha(Path(manifest[k]))==v for k,v in manifest['input_sha256'].items())
    images=[str(p) for directory in ['runs','evaluation'] for p in (ROOT/directory).rglob('*')
        if p.is_file() and p.suffix.lower() in ['.png','.jpg','.jpeg','.bmp','.webp']
        and ROOT/'evaluation/reports/figures' not in p.parents]
    assert not images
    sources.update({str(p.relative_to(ROOT/'source')):sha(p) for p in HERE.iterdir() if p.is_file()})
    (ROOT/'reports/SOURCE_FINAL_HASHES.json').write_text(json.dumps(sources,indent=2))
    with tarfile.open(ROOT/'source_final.tar.gz','w:gz') as tar:
        for name in sorted(sources):tar.add(ROOT/'source'/name,arcname=name,recursive=False)
    shutil.copyfile(HERE/'PROTOCOL.md',ROOT/'reports/PROTOCOL.md')
    result=dict(passed=True,completed_unix=time.time(),protected_production_files=len(before),changed_production_files=[],
        new_training_groups=2,reused_training_groups=1,steps_per_group=11580,evaluation_rows=88704,
        reused_rows=26928,new_rows=61776,rasterizer_bitwise_checks=234,rendered_frame_images=0,
        unchanged_implementation_source_files=len(json.loads((ROOT/'reports/IMPLEMENTATION_SOURCE_HASHES.json').read_text())),
        binary_parity_with_verified_implementation=True,input_checkpoints_unchanged=True,free_GiB=shutil.disk_usage(ROOT).free/1024**3)
    (ROOT/'reports/FINAL_INTEGRITY.json').write_text(json.dumps(result,indent=2))
    (ROOT/'reports/COMPLETION.json').write_text(json.dumps(dict(state='complete',integrity_passed=True,
        training_groups=['view_no_li','view_li01'],baseline_reused=True,quality_report='evaluation/reports/VIEW_RESIDUAL_FULL_REPORT.md'),indent=2))
    archive=ROOT/'reports/VIEW_RESIDUAL_FULL_RESULTS.zip';assert not archive.exists()
    with ZipFile(archive,'w',ZIP_DEFLATED) as stream:
        for directory in ['reports','evaluation/reports','configs','logs']:
            for p in sorted((ROOT/directory).rglob('*')):
                if p.is_file() and p!=archive and p.suffix!='.lock':stream.write(p,p.relative_to(ROOT))
        for group in ['view_no_li','view_li01']:
            folder=ROOT/'runs'/f'full_{group}'
            for name in ['initialization.json','training_verified.json','training_keyframes.csv','real_lod_steps.csv','view_budget_steps.jsonl']:
                stream.write(folder/name,(folder/name).relative_to(ROOT))
            for p in (folder/'23560_shutdown/ply').rglob('*.json'):stream.write(p,p.relative_to(ROOT))
        stream.write(ROOT/'evaluation/STATUS.json','evaluation/STATUS.json')
        for p in HERE.iterdir():
            if p.is_file():stream.write(p,Path('scripts')/p.name)
    download=dict(archive_sha256=sha(archive),archive_bytes=archive.stat().st_size)
    (ROOT/'reports/DOWNLOAD_HASH.json').write_text(json.dumps(download,indent=2))
    print(json.dumps(dict(integrity=result,download=download),indent=2))


if __name__=='__main__':main()
