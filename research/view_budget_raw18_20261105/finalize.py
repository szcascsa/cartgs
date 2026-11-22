"""完整评价和绘图后的最终SHA检查、中文报告及数据归档。"""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import time
from zipfile import ZipFile,ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
OLD=Path('/home/prmi/zwj/experiments/view_budget_residual_full_20261105')
BASE=Path('/home/prmi/zwj/experiments/local_importance_li_20261105')


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()


def main():
    training=json.loads((ROOT/'reports/TRAINING_STATUS.json').read_text())
    evaluation=json.loads((ROOT/'evaluation/STATUS.json').read_text())
    assert training['state']=='training_complete' and evaluation['state']=='complete'
    assert evaluation['rows']==150480 and evaluation['new_rows']==61776 and evaluation['reused_rows']==88704
    assert evaluation['parity_checks']==234
    assert (ROOT/'evaluation/reports/RAW18_VIEW_RESIDUAL_REPORT.md').exists()
    assert len(list((ROOT/'evaluation/reports/figures').glob('*.png')))==4
    integrity=json.loads((ROOT/'evaluation/reports/INPUT_INTEGRITY.json').read_text());assert integrity['passed']
    assert all(sha(Path(n))==h for n,h in integrity['before'].items())
    before=json.loads((ROOT/'reports/PRODUCTION_BEFORE.json').read_text())
    after={n:dict(bytes=Path(n).stat().st_size,sha256=sha(Path(n))) for n in before}
    assert before==after,'受保护生产文件变化'
    (ROOT/'reports/PRODUCTION_AFTER.json').write_text(json.dumps(after,indent=2))
    build=json.loads((ROOT/'reports/BUILD_HASHES.json').read_text())
    assert all(sha(ROOT/n)==h for n,h in build.items()),'已验收构建变化'
    initial=json.loads((ROOT/'reports/SOURCE_HASHES.json').read_text())
    implementation={n:h for n,h in initial.items() if not n.startswith('research/')}
    assert all(sha(ROOT/'source'/n)==h for n,h in implementation.items()),'已编译实现源码发生变化'
    (ROOT/'reports/IMPLEMENTATION_SOURCE_HASHES.json').write_text(json.dumps(implementation,indent=2))
    historical={str(p):sha(p) for d in [BASE,OLD] for p in [d/'runtime/bin/lod_from_full',d/'runtime/lib/libgaussian_mapper.so']}
    (ROOT/'reports/HISTORICAL_BINARIES.json').write_text(json.dumps(historical,indent=2))
    for group in ['raw_no_li','raw_li01']:
        manifest=json.loads((ROOT/'configs'/f'full_{group}.json').read_text())
        assert all(sha(Path(manifest[k]))==v for k,v in manifest['input_sha256'].items())
    import torch
    folders={'baseline':BASE/'runs/full_no_li',**{g:OLD/'runs'/f'full_{g}' for g in ['view_no_li','view_li01']},
        **{g:ROOT/'runs'/f'full_{g}' for g in ['raw_no_li','raw_li01']}}
    reference=torch.jit.load(str(folders['baseline']/'step0_selector.pt'),map_location='cpu').state_dict()
    sequence=None;master=sha(folders['baseline']/'step0_master.ply')
    for group,folder in folders.items():
        values=torch.jit.load(str(folder/'step0_selector.pt'),map_location='cpu').state_dict()
        assert len(values)==(61 if group=='baseline' else 68)
        assert all(torch.equal(a.reshape(-1).view(torch.uint8),values[n].reshape(-1).view(torch.uint8)) for n,a in reference.items())
        assert master==sha(folder/'step0_master.ply')
        current=[(r['step'],r['keyframe_id'],r['ratio']) for r in map(json.loads,(folder/'view_budget_steps.jsonl').read_text().splitlines())]
        if sequence is None:sequence=current
        assert current==sequence and len(current)==11580
    (ROOT/'reports/FIVE_GROUP_INITIALIZATION.json').write_text(json.dumps(dict(passed=True,groups=list(folders),
        shared_tensors_bitwise=61,master_sha256=master,identical_camera_ratio_steps=11580),indent=2))
    frame_images=[str(p) for d in ['runs','evaluation'] for p in (ROOT/d).rglob('*') if p.is_file()
        and p.suffix.lower() in ['.png','.jpg','.jpeg','.bmp','.webp'] and ROOT/'evaluation/reports/figures' not in p.parents]
    assert not frame_images
    source={n:sha(ROOT/'source'/n) for n in initial}
    source.update({str(p.relative_to(ROOT/'source')):sha(p) for p in HERE.iterdir() if p.is_file()})
    (ROOT/'reports/SOURCE_FINAL_HASHES.json').write_text(json.dumps(source,indent=2))
    with tarfile.open(ROOT/'source_final.tar.gz','w:gz') as t:
        for n in sorted(source):t.add(ROOT/'source'/n,arcname=n,recursive=False)
    result=dict(passed=True,completed_unix=time.time(),new_training_groups=2,reused_training_groups=3,
        steps_per_new_group=11580,evaluation_rows=150480,new_rows=61776,reused_rows=88704,
        protected_production_files=len(before),changed_production_files=[],rasterizer_bitwise_checks=234,
        rendered_frame_images=0,figures=4,compiled_source_unchanged=True,binaries_unchanged=True,
        input_checkpoints_and_historical_metrics_unchanged=True,free_GiB=shutil.disk_usage(ROOT).free/1024**3)
    (ROOT/'reports/FINAL_INTEGRITY.json').write_text(json.dumps(result,indent=2))
    (ROOT/'reports/COMPLETION.json').write_text(json.dumps(dict(state='complete',integrity_passed=True,
        quality_report='evaluation/reports/RAW18_VIEW_RESIDUAL_REPORT.md'),indent=2))
    archive=ROOT/'reports/RAW18_VIEW_RESIDUAL_RESULTS.zip';assert not archive.exists()
    with ZipFile(archive,'w',ZIP_DEFLATED) as z:
        for d in ['reports','evaluation/reports','configs','logs']:
            for p in sorted((ROOT/d).rglob('*')):
                if p.is_file() and p!=archive and p.suffix!='.lock':z.write(p,p.relative_to(ROOT))
        for group in ['raw_no_li','raw_li01']:
            folder=ROOT/'runs'/f'full_{group}'
            for name in ['initialization.json','training_verified.json','training_keyframes.csv','real_lod_steps.csv','view_budget_steps.jsonl']:
                z.write(folder/name,(folder/name).relative_to(ROOT))
            for p in (folder/'23560_shutdown/ply').rglob('*.json'):z.write(p,p.relative_to(ROOT))
        z.write(ROOT/'evaluation/STATUS.json','evaluation/STATUS.json')
        for p in HERE.iterdir():
            if p.is_file():z.write(p,Path('scripts')/p.name)
    download=dict(sha256=sha(archive),bytes=archive.stat().st_size)
    (ROOT/'reports/DOWNLOAD_HASH.json').write_text(json.dumps(download,indent=2))
    print(json.dumps(dict(integrity=result,download=download),indent=2))


if __name__=='__main__':main()
