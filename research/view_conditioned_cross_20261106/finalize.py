"""核验生产完整性、固定输入、构建身份及无渲染帧图片，归档结果。"""
import json
from pathlib import Path
import shutil
import tarfile
import time
from zipfile import ZipFile,ZIP_DEFLATED
from common import ROOT,HERE,GROUPS,sha,save


def main():
    training=json.loads((ROOT/'reports/TRAINING_STATUS.json').read_text())
    evaluation=json.loads((ROOT/'evaluation/STATUS.json').read_text())
    assert training['state']=='training_complete' and evaluation['state']=='complete'
    assert evaluation['rows']==134640 and evaluation['new_rows']==53856 and evaluation['reused_rows']==80784 and evaluation['parity_checks']==204
    assert (ROOT/'evaluation/reports/CROSS_VIEW_CONDITIONING_REPORT.md').exists()
    assert len(list((ROOT/'evaluation/reports/figures').glob('*.png')))==4
    assert json.loads((ROOT/'reports/FINAL_CONDITIONING_VALIDATION.json').read_text())['passed']
    assert json.loads((ROOT/'evaluation/reports/HISTORICAL_METRIC_PARITY.json').read_text())['passed']
    baseline_geometry=json.loads((ROOT/'evaluation/reports/BASELINE_GEOMETRY_SUMMARY.json').read_text())
    assert baseline_geometry['passed'] and baseline_geometry['renderer_calls']==0 and baseline_geometry['rows']==3168
    tail=json.loads((ROOT/'evaluation/reports/TAIL_DIAGNOSTIC_CASES.json').read_text())
    assert len(tail['checks'])==24 and all(v['xyz_matches_direct_network_bitwise'] for v in tail['checks'])
    before=json.loads((ROOT/'reports/PRODUCTION_BEFORE.json').read_text())
    after={n:dict(bytes=Path(n).stat().st_size,sha256=sha(n)) for n in before};assert after==before
    save(ROOT/'reports/PRODUCTION_AFTER.json',after)
    historical=json.loads((ROOT/'reports/HISTORICAL_INPUTS.json').read_text());assert all(sha(n)==h for n,h in historical.items())
    integrity=json.loads((ROOT/'evaluation/reports/INPUT_INTEGRITY.json').read_text());assert integrity['passed'] and integrity['before']==integrity['after']
    assert all(sha(n)==h for n,h in integrity['after'].items())
    build=json.loads((ROOT/'reports/BUILD_HASHES.json').read_text());assert all(sha(ROOT/n)==h for n,h in build.items())
    initial=json.loads((ROOT/'reports/SOURCE_HASHES.json').read_text())
    implementation={n:h for n,h in initial.items() if not n.startswith('research/') or n=='research/global_view_joint_lod_20261105/lod_from_full.cpp'}
    assert all(sha(ROOT/'source'/n)==h for n,h in implementation.items()),'已编译实现发生变化'
    save(ROOT/'reports/IMPLEMENTATION_SOURCE_HASHES.json',implementation)
    for group in GROUPS:
        manifest=json.loads((ROOT/'configs'/f'full_{group}.json').read_text())
        assert all(sha(manifest[k])==h for k,h in manifest['input_sha256'].items())
        state=json.loads((ROOT/'runs'/f'full_{group}'/'23560_shutdown/ply/real_lod_state.json').read_text())
        assert state['importance_full_renders']==864 and state['importance_refreshes']==12
        assert state['view_budget_li_full_renders']==0 and state['view_budget_li_backwards']==0
        assert state['total_training_renders']==34740 and state['lod_nonfinite']==0
    images=[str(p) for d in ['runs','evaluation'] for p in (ROOT/d).rglob('*') if p.is_file() and
        p.suffix.lower() in ['.png','.jpg','.jpeg','.bmp','.webp'] and ROOT/'evaluation/reports/figures' not in p.parents]
    assert not images
    source={n:sha(ROOT/'source'/n) for n in initial}
    source.update({str(p.relative_to(ROOT/'source')):sha(p) for p in HERE.iterdir() if p.is_file() and p.suffix in ['.py','.md']})
    save(ROOT/'reports/SOURCE_FINAL_HASHES.json',source)
    with tarfile.open(ROOT/'source_final.tar.gz','w:gz') as archive:
        for n in sorted(source):archive.add(ROOT/'source'/n,arcname=n,recursive=False)
    result=dict(passed=True,completed_unix=time.time(),new_training_groups=2,reused_training_groups=3,steps_per_new_group=11580,
        evaluation_rows=134640,new_rows=53856,reused_rows=80784,protected_production_files=len(before),changed_production_files=[],
        rasterizer_bitwise_checks=204,rendered_frame_images=0,figures=4,compiled_source_unchanged=True,binaries_unchanged=True,
        historical_inputs_unchanged=True,reused_baseline_geometry_rows=3168,baseline_rerenders=0,
        extra_new_model_tail_renders=30,tail_xyz_bitwise_checks=24,geometry_results_reported=True,production_default_changed=False,
        free_GiB=shutil.disk_usage(ROOT).free/1024**3)
    save(ROOT/'reports/FINAL_INTEGRITY.json',result)
    archive=ROOT/'reports/CROSS_VIEW_CONDITIONING_RESULTS.zip';assert not archive.exists()
    with ZipFile(archive,'w',ZIP_DEFLATED) as z:
        for d in ['reports','evaluation/reports','configs','logs']:
            for p in sorted((ROOT/d).rglob('*')):
                if p.is_file() and p!=archive and p.suffix not in ['.lock','.pt']:z.write(p,p.relative_to(ROOT))
        for g in GROUPS:
            d=ROOT/'runs'/f'full_{g}'
            for n in ['initialization.json','training_verified.json','training_keyframes.csv','real_lod_steps.csv','view_budget_steps.jsonl']:z.write(d/n,(d/n).relative_to(ROOT))
            for p in (d/'23560_shutdown/ply').rglob('*.json'):z.write(p,p.relative_to(ROOT))
        z.write(ROOT/'evaluation/STATUS.json','evaluation/STATUS.json')
        for p in HERE.iterdir():
            if p.is_file() and p.suffix in ['.py','.md']:z.write(p,p.relative_to(ROOT/'source'))
    save(ROOT/'reports/ARCHIVE_SHA256.json',dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),
        source_path=str(ROOT/'source_final.tar.gz'),source_sha256=sha(ROOT/'source_final.tar.gz')))
    save(ROOT/'reports/COMPLETION.json',dict(state='complete',integrity_passed=True,quality_report='evaluation/reports/CROSS_VIEW_CONDITIONING_REPORT.md'))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
