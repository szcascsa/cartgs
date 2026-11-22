"""评价完成后验证受保护文件并打包数值证据，不修改模型。"""
import hashlib
import json
from pathlib import Path
import shutil
import time
from zipfile import ZipFile,ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[3]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def main():
    assert json.loads((ROOT/'evaluation/STATUS.json').read_text())['state']=='complete'
    assert (ROOT/'evaluation/reports/LI_WEIGHT_05_REPORT.md').exists()
    before=json.loads((ROOT/'reports/PRODUCTION_BEFORE.json').read_text())
    after={name:dict(bytes=Path(name).stat().st_size,sha256=sha(Path(name))) for name in before}
    changed=[name for name in before if before[name]!=after[name]]
    assert not changed,changed
    (ROOT/'reports/PRODUCTION_AFTER.json').write_text(json.dumps(after,indent=2))
    original=json.loads((ROOT/'reports/PREVIOUS_SOURCE_FINAL_HASHES.json').read_text())
    source_changed=[name for name,value in original.items() if sha(ROOT/'source'/name)!=value]
    assert not source_changed,source_changed
    binary=json.loads((ROOT/'reports/BINARY_PARITY.json').read_text())['sha256']
    assert all(sha(ROOT/name)==value for name,value in binary.items())
    inputs=json.loads((ROOT/'evaluation/reports/INPUT_INTEGRITY.json').read_text())
    assert inputs['passed'] and inputs['before']==inputs['after']
    assert all(sha(Path(name))==value for name,value in inputs['before'].items())
    manifest=json.loads((ROOT/'configs/full_li05.json').read_text())
    assert all(sha(Path(manifest[key]))==value for key,value in manifest['input_sha256'].items())
    frames=[str(p) for directory in [ROOT/'runs',ROOT/'evaluation'] for p in directory.rglob('*')
        if p.is_file() and p.suffix.lower() in ['.png','.jpg','.jpeg','.bmp','.webp']
        and ROOT/'evaluation/reports/figures' not in p.parents]
    assert not frames,frames
    record=dict(passed=True,checked_unix=time.time(),protected_production_files=len(before),
        changed_production_files=changed,unchanged_source_files=len(original),source_changes=source_changed,
        binary_parity_with_previous=True,evaluation_inputs_unchanged=True,
        input_manifest_unchanged=True,rendered_frame_images=len(frames),free_GiB=shutil.disk_usage(ROOT).free/1024**3)
    (ROOT/'reports/FINAL_INTEGRITY.json').write_text(json.dumps(record,indent=2))
    (ROOT/'reports/COMPLETION.json').write_text(json.dumps(dict(state='complete',checked_unix=time.time(),
        train_steps=11580,evaluation_rows=80784,retrained_after_postprocess_fix=False,
        rerendered_after_postprocess_fix=False,postprocess_fix='compare configured LI weight in exact FP32 representation',
        failure_log='logs/summarize_weight05.failed_fp32_assertion.log',integrity_passed=True),indent=2))
    protocol=ROOT/'source/research/local_importance_li05_20261105/PROTOCOL.md'
    shutil.copyfile(protocol,ROOT/'reports/PROTOCOL.md')
    scripts=protocol.parent
    (ROOT/'reports/EXPERIMENT_SCRIPT_HASHES.json').write_text(json.dumps(
        {p.name:sha(p) for p in scripts.iterdir() if p.is_file()},indent=2))
    archive=ROOT/'reports/LI_WEIGHT_05_EVIDENCE.zip'
    assert not archive.exists()
    with ZipFile(archive,'w',ZIP_DEFLATED) as stream:
        for directory in ['reports','evaluation/reports','configs','logs']:
            for p in sorted((ROOT/directory).rglob('*')):
                if p.is_file() and p!=archive and p.suffix!='.lock':stream.write(p,p.relative_to(ROOT))
        for group in ['smoke_li05','full_li05']:
            for name in ['training_verified.json','training_keyframes.csv','real_lod_steps.csv','view_budget_steps.jsonl']:
                p=ROOT/'runs'/group/name
                if p.exists():stream.write(p,p.relative_to(ROOT))
        stream.write(ROOT/'evaluation/STATUS.json','evaluation/STATUS.json')
        for p in (ROOT/'source/research/local_importance_li05_20261105').glob('*'):
            if p.is_file():stream.write(p,Path('scripts')/p.name)
    print(json.dumps(dict(integrity=record,archive=str(archive),archive_bytes=archive.stat().st_size),indent=2))


if __name__=='__main__':main()
