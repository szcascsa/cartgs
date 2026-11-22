"""交付前检查原始输入、生产项目、源码、构建、训练与评价完整性。"""
import json
from pathlib import Path
import shutil
import tarfile
import time
from zipfile import ZipFile,ZIP_DEFLATED
from common import ROOT,HERE,sha,save


def main():
    training=json.loads((ROOT/'reports/TRAINING_STATUS.json').read_text());assert training['state']=='training_complete'
    evaluation=json.loads((ROOT/'evaluation/STATUS.json').read_text());assert evaluation['state']=='complete' and evaluation['rows']==55440
    assert evaluation['new_rows']==28512 and evaluation['reused_rows']==26928 and evaluation['parity_checks']==108
    assert json.loads((ROOT/'reports/PYTHON_PARITY.json').read_text())['passed']
    assert json.loads((ROOT/'reports/INITIALIZATION_PARITY.json').read_text())['identical_camera_ratio_steps']==11580
    assert json.loads((ROOT/'evaluation/reports/HISTORICAL_METRIC_PARITY.json').read_text())['passed']
    speed=json.loads((ROOT/'speed/STATUS.json').read_text());assert speed['state']=='complete' and speed['rows']==14256 and speed['micro_rows']==1600 and speed['forward_backward_rows']==640
    before=json.loads((ROOT/'reports/PRODUCTION_BEFORE.json').read_text())
    after={n:dict(bytes=Path(n).stat().st_size,sha256=sha(n)) for n in before};assert after==before
    save(ROOT/'reports/PRODUCTION_AFTER.json',after)
    historical=json.loads((ROOT/'reports/HISTORICAL_INPUTS.json').read_text());assert all(sha(n)==h for n,h in historical.items())
    integrity=json.loads((ROOT/'evaluation/reports/INPUT_INTEGRITY.json').read_text());assert integrity['passed'] and integrity['before']==integrity['after']
    assert all(sha(n)==h for n,h in integrity['after'].items())
    build=json.loads((ROOT/'reports/BUILD_HASHES.json').read_text());assert all(sha(ROOT/n)==h for n,h in build.items())
    initial=json.loads((ROOT/'reports/SOURCE_HASHES.json').read_text())
    implementation={n:h for n,h in initial.items() if not n.startswith(('research/','tests/'))}
    cache=json.loads((ROOT/'reports/PYTHON_LAYOUT_CACHE_PATCH.json').read_text())
    assert cache['before']==initial['python/spatial_compensation.py'] and cache['parity_passed']
    implementation['python/spatial_compensation.py']=cache['after']
    implementation['research/global_view_joint_lod_20261105/lod_from_full.cpp']=initial['research/global_view_joint_lod_20261105/lod_from_full.cpp']
    assert all(sha(ROOT/'source'/n)==h for n,h in implementation.items()),'已编译主体实现发生变化'
    save(ROOT/'reports/IMPLEMENTATION_SOURCE_HASHES.json',implementation)
    import torch
    checkpoint=ROOT/'runs/full_no_other/23560_shutdown/ply'
    state=torch.jit.load(str(checkpoint/'selector.pt'),map_location='cpu').state_dict()
    assert int(state['spatial_compensation.geometry_head_schema'])==1 and not any('.heads.other.' in n for n in state)
    assert len(state)==64
    for value in state.values():assert not value.is_floating_point() or torch.isfinite(value).all()
    fine=json.loads((checkpoint/'real_lod_state.json').read_text())
    assert fine['optimizer_commits']==11580 and fine['importance_full_renders']==864 and fine['importance_refreshes']==12
    assert fine['total_training_renders']==34740 and fine['lod_nonfinite']==0
    assert fine['view_budget_li_backwards']==fine['view_budget_li_full_renders']==11580
    images=[str(p) for d in ['runs','evaluation'] for p in (ROOT/d).rglob('*') if p.is_file() and p.suffix.lower() in ['.png','.jpg','.jpeg','.bmp','.webp'] and ROOT/'evaluation/reports/figures' not in p.parents]
    assert not images
    assert (ROOT/'evaluation/reports/NO_OTHER_HEAD_REPORT.md').exists()
    assert len(list((ROOT/'evaluation/reports/figures').glob('*.png')))==4
    source={n:sha(ROOT/'source'/n) for n in initial}
    source.update({str(p.relative_to(ROOT/'source')):sha(p) for p in HERE.iterdir() if p.is_file() and p.suffix in ['.py','.md']})
    save(ROOT/'reports/SOURCE_FINAL_HASHES.json',source)
    with tarfile.open(ROOT/'source_final.tar.gz','w:gz') as archive:
        for n in sorted(source):archive.add(ROOT/'source'/n,arcname=n,recursive=False)
    save(ROOT/'reports/FINAL_INTEGRITY.json',dict(passed=True,completed_unix=time.time(),protected_production_files=len(before),changed_production_files=[],
        new_training_groups=1,reused_training_groups=1,steps=11580,evaluation_rows=55440,new_rows=28512,reused_rows=26928,frames=792,
        rasterizer_bitwise_checks=108,speed_rows=14256,forward_micro_rows=1600,forward_backward_micro_rows=640,rendered_frame_images=0,
        removed_parameters=4420,compiled_source_unchanged=True,binaries_unchanged=True,historical_inputs_unchanged=True,production_default_changed=False,
        commit_push_performed=False,free_GiB=shutil.disk_usage(ROOT).free/2**30,model_path=str(checkpoint)))
    archive=ROOT/'reports/NO_OTHER_HEAD_RESULTS.zip';assert not archive.exists()
    with ZipFile(archive,'w',ZIP_DEFLATED) as z:
        for d in ['reports','evaluation/reports','configs','logs','speed']:
            for p in sorted((ROOT/d).rglob('*')):
                if p.is_file() and p!=archive and p.suffix not in ['.lock','.pt']:z.write(p,p.relative_to(ROOT))
        for name in ['initialization.json','training_verified.json','training_keyframes.csv','real_lod_steps.csv','view_budget_steps.jsonl']:
            p=ROOT/'runs/full_no_other'/name;z.write(p,p.relative_to(ROOT))
        for p in checkpoint.rglob('*.json'):z.write(p,p.relative_to(ROOT))
        z.write(ROOT/'evaluation/STATUS.json','evaluation/STATUS.json')
        for p in HERE.iterdir():
            if p.is_file() and p.suffix in ['.py','.md']:z.write(p,p.relative_to(ROOT/'source'))
    save(ROOT/'reports/ARCHIVE_SHA256.json',dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),source_sha256=sha(ROOT/'source_final.tar.gz')))
    save(ROOT/'reports/COMPLETION.json',dict(state='complete',integrity_passed=True,report='evaluation/reports/NO_OTHER_HEAD_REPORT.md'))
    print('FINAL_INTEGRITY_PASS',flush=True)


if __name__=='__main__':main()
