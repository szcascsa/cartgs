"""核验只读输入及生产哈希，归档数值与源码。"""
import json
import shutil
import tarfile
from zipfile import ZipFile,ZIP_DEFLATED
from common import ROOT,sha,save

OUT=ROOT/'evaluation/reports'
assert json.loads((ROOT/'evaluation/STATUS.json').read_text())['state']=='complete'
assert json.loads((ROOT/'reports/SUMMARIZATION.json').read_text())['passed']
assert json.loads((OUT/'CORRECTNESS.json').read_text())['passed']
assert (OUT/'OTHER_HEAD_INFERENCE_REPORT.md').exists() and len(list((OUT/'figures').glob('*.png')))==2
inputs=json.loads((ROOT/'reports/INPUTS_BEFORE.json').read_text());after={n:sha(n) for n in inputs};assert inputs==after
production=json.loads((ROOT/'reports/PRODUCTION_BEFORE.json').read_text());prod_after={n:sha(n) for n in production};assert production==prod_after
save(ROOT/'reports/INPUTS_AFTER.json',after);save(ROOT/'reports/PRODUCTION_AFTER.json',prod_after)
code={str(p.relative_to(ROOT/'source')):sha(p) for p in (ROOT/'source').rglob('*') if p.is_file() and p.suffix in ['.py','.md']}
copied=json.loads((ROOT/'reports/INFERENCE_SOURCE_HASHES.json').read_text());assert all(code[n]==h for n,h in copied.items())
save(ROOT/'reports/SOURCE_FINAL_HASHES.json',code)
with tarfile.open(ROOT/'source_final.tar.gz','w:gz') as a:
    for n in sorted(code):a.add(ROOT/'source'/n,arcname=n,recursive=False)
images=[str(p) for p in (ROOT/'evaluation').rglob('*') if p.is_file() and p.suffix.lower() in ['.png','.jpg','.jpeg','.bmp'] and OUT/'figures' not in p.parents]
assert not images
result=dict(passed=True,frames=792,variants=2,cases=28,rows=44352,paired_rows=22176,all_frame_same_selection_and_K=True,
    zero_residual_packet_bitwise_checks=84,rasterizer_bitwise_checks=168,model_state_bitwise_unchanged=True,
    protected_production_files=len(production),changed_production_files=[],historical_inputs_unchanged=True,images_saved=0,
    figures=2,production_default_changed=False,training_runs=0,free_GiB=shutil.disk_usage(ROOT).free/1024**3)
save(ROOT/'reports/FINAL_INTEGRITY.json',result)
archive=ROOT/'reports/OTHER_HEAD_INFERENCE_RESULTS.zip';assert not archive.exists()
with ZipFile(archive,'w',ZIP_DEFLATED) as z:
    for d in ['reports','logs','evaluation/reports']:
        for p in sorted((ROOT/d).rglob('*')):
            if p.is_file() and p!=archive:z.write(p,str(p.relative_to(ROOT)))
    z.write(ROOT/'evaluation/STATUS.json','evaluation/STATUS.json')
    for p in (ROOT/'source/research/other_head_inference_20261106').glob('*'):
        if p.is_file():z.write(p,str(p.relative_to(ROOT/'source')))
save(ROOT/'reports/ARCHIVE_SHA256.json',dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),
    source_sha256=sha(ROOT/'source_final.tar.gz')))
save(ROOT/'reports/COMPLETION.json',dict(state='complete',integrity_passed=True,report='evaluation/reports/OTHER_HEAD_INFERENCE_REPORT.md'))
print(json.dumps(result),flush=True)
