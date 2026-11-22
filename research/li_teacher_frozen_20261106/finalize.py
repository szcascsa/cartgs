"""核验冻结状态、来源与生产完整性，归档数值报告。"""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import time
from zipfile import ZipFile,ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def main():
    status=json.loads((ROOT/'evaluation/STATUS.json').read_text())
    assert status['passed'] and status['rows']==9504 and status['optimizer_updates']==0
    assert json.loads((ROOT/'reports/PIPELINE.json').read_text())['state']=='evaluation_complete'
    payload=json.loads((ROOT/'reports/TEACHER_PAYLOAD_PARITY.json').read_text());assert payload['passed'] and payload['cases']==16
    inputs=json.loads((ROOT/'reports/INPUTS_BEFORE.json').read_text());after={n:sha(Path(n)) for n in inputs}
    assert inputs==after,'输入模型或历史评价改变'
    (ROOT/'reports/INPUTS_AFTER.json').write_text(json.dumps(after,indent=2))
    before=json.loads((ROOT/'reports/PRODUCTION_BEFORE.json').read_text())
    protected={n:dict(bytes=Path(n).stat().st_size,sha256=sha(Path(n))) for n in before}
    assert before==protected,'生产保护文件改变'
    (ROOT/'reports/PRODUCTION_AFTER.json').write_text(json.dumps(protected,indent=2))
    build=json.loads((ROOT/'reports/BUILD.json').read_text());assert sha(Path(build['path']))==build['sha256']
    source=json.loads((ROOT/'reports/SOURCE_HASHES.json').read_text())
    assert all(sha(ROOT/'source'/n)==h for n,h in source.items() if not n.startswith('research/')),'原算法源码改变'
    assert all(sha(ROOT/'source/cuda_rasterizer'/n)==h for n,h in build['cuda_source_hashes'].items())
    source.update({str(p.relative_to(ROOT/'source')):sha(p) for p in HERE.iterdir() if p.is_file()})
    (ROOT/'reports/SOURCE_FINAL_HASHES.json').write_text(json.dumps(source,indent=2))
    with tarfile.open(ROOT/'source_final.tar.gz','w:gz') as t:
        for n in sorted(source):t.add(ROOT/'source'/n,arcname=n,recursive=False)
    images=[str(p) for d in ['runs','evaluation'] for p in (ROOT/d).rglob('*') if p.is_file() and p.suffix.lower() in ['.jpg','.png','.jpeg','.webp'] and ROOT/'evaluation/reports/figures' not in p.parents]
    assert not images
    assert len(list((ROOT/'evaluation/reports/figures').glob('*.png')))==2
    shutil.copyfile(HERE/'PROTOCOL.md',ROOT/'reports/PROTOCOL.md')
    value=dict(passed=True,completed_unix=time.time(),protected_production_files=len(before),changed_production_files=[],
        model_and_metrics_unchanged=True,cuda_sources_unchanged=True,original_algorithm_sources_unchanged=True,
        evaluation_rows=9504,frames=792,rasterizer_bitwise_checks=status['rasterizer_bitwise_checks'],
        teacher_payload_bitwise_checks=payload['cases'],
        native_teacher_sort_checks=status['native_teacher_sort_checks'],previous_psnr_max_abs_error=status['prior_psnr_max_abs_error'],
        optimizer_updates=0,rendered_frame_images=0,source_files=len(source),free_GiB=shutil.disk_usage(ROOT).free/1024**3)
    (ROOT/'reports/FINAL_INTEGRITY.json').write_text(json.dumps(value,indent=2))
    (ROOT/'reports/COMPLETION.json').write_text(json.dumps(dict(state='complete',report='evaluation/reports/LI_TEACHER_VALIDATION_REPORT.md'),indent=2))
    archive=ROOT/'reports/LI_TEACHER_VALIDATION_RESULTS.zip';assert not archive.exists()
    with ZipFile(archive,'w',ZIP_DEFLATED) as z:
        for d in ['reports','evaluation/reports','configs','logs']:
            for p in sorted((ROOT/d).rglob('*')):
                if p.is_file() and p!=archive and p.suffix!='.lock':z.write(p,p.relative_to(ROOT))
        z.write(ROOT/'evaluation/STATUS.json','evaluation/STATUS.json')
        for p in HERE.iterdir():
            if p.is_file():z.write(p,Path('scripts')/p.name)
    result=dict(sha256=sha(archive),bytes=archive.stat().st_size)
    (ROOT/'reports/DOWNLOAD_HASH.json').write_text(json.dumps(result,indent=2));print(json.dumps(dict(integrity=value,download=result),indent=2))


if __name__=='__main__':main()
