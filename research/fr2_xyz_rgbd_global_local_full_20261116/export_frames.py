"""固定轨迹分位点抽五帧，原始全局／局部／Full路径渲染并匹配历史数值。"""
from pathlib import Path
import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parent
BATCH = Path('/home/prmi/zwj/realLOD/saved_experiments/dual_view_li_best_full_20261107')
MODEL = BATCH / 'runs/TUM_RGBD/freiburg2_xyz/rgbd/attempt_01/model'
EVAL = BATCH / 'evaluation/TUM_RGBD/freiburg2_xyz/rgbd'
DIRECTORIES = {'full': 'full_1.0', 'global': 'global_0.5', 'view_ratio': 'view_ratio_0.5'}
OUTPUTS = {'full': 'full_0', 'global': 'global_0.5', 'view_ratio': 'local_0.5'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def read_csv(path):
    return list(csv.DictReader(path.open()))


def main():
    full_rows = read_csv(EVAL / 'full_1.0/attempt_01/per_frame_metrics.csv')
    fractions = [.1, .3, .5, .7, .9]
    positions = [round((len(full_rows) - 1) * f) for f in fractions]
    selected = [full_rows[p] for p in positions]
    indices = [int(r['rgb_index']) for r in selected]
    launches = {scope: load(EVAL / folder / 'attempt_01/launch.json')['command']
                for scope, folder in DIRECTORIES.items()}
    protected = {MODEL / 'CameraTrajectory_TUM.txt', MODEL / 'TrackingTime.txt',
                 BATCH / 'configs/dual_view_li.yaml', BATCH / 'source/python/run_metrics_only.py',
                 BATCH / 'eval_joint_entry.py'}
    protected.update(p for p in (MODEL / '101065_shutdown/ply').rglob('*') if p.is_file())
    protected.update(BATCH / 'source/python' / name for name in
                     ['renderer.py', 'selector.py', 'anchor_budget.py', 'anchor_frustum.py', 'selector_sampling.py'])
    for folder in DIRECTORIES.values():
        protected.update(p for p in (EVAL / folder / 'attempt_01').iterdir() if p.is_file())
    protected.update(Path(r['rgb_path']) for r in selected)
    before = {str(p): sha(p) for p in sorted(protected)}
    write(ROOT / 'INPUTS_BEFORE.json', before)
    metadata = dict(dataset='TUM_RGBD', scene='freiburg2_xyz', mode='rgbd',
        batch=str(BATCH), model=str(MODEL), shutdown='101065_shutdown', master_N=15634,
        sampling='按原3615帧有效评价轨迹的10%/30%/50%/70%/90%分位选帧，未按质量筛选',
        positions=positions, fractions=fractions, available_views=len(full_rows), gt_indices=indices,
        launch_arguments={scope: command[3:] for scope, command in launches.items()},
        original_evaluations={scope: str(EVAL / folder / 'attempt_01') for scope, folder in DIRECTORIES.items()},
        image_protocol='原始RGB；相同估计位姿；无裁剪、调色、曝光或额外位姿优化；原分辨率PNG',
        global_budget='原全图FlexGS采样导出，ratio=.5，零视角／零模式条件；复核原selected_ids',
        local_budget='名义Anchor视锥候选池，K=min(floor(.5*Ncandidate),eligible)，两网络真实视角；逐帧哈希核验',
        full='同一最终Master未选择、未补偿的Full渲染，不是CaRtGS Vanilla')
    write(ROOT / 'EXPERIMENT.json', metadata)
    write(ROOT / 'FRAME_SET.json', dict(gt_indices=indices))
    print('FRAMES', indices, flush=True)
    env = dict(os.environ, OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='2',
               MPLBACKEND='Agg', QT_QPA_PLATFORM='offscreen', PYTHONDONTWRITEBYTECODE='1', CARTGS_AUDIT_SEED='0')
    env['LD_LIBRARY_PATH'] = ':'.join([str(BATCH / 'runtime/lib'),
        '/home/prmi/zwj/mycartgs/third_party/install/opencv-cuda118-sm86-gcc9/lib',
        '/home/prmi/zwj/mycartgs/third_party/install/jsoncpp-1.9.5/lib',
        '/home/prmi/zwj/mycartgs/third_party/libtorch-cu118/libtorch/lib', '/usr/local/cuda-11.8/lib64'])
    for key in list(env):
        if key.startswith('CARTGS_LOD_DIAGNOSTIC_') or key in ['GEOMETRY_HEAD_VARIANT', 'LD_PRELOAD', 'CARTGS_BATCH_FULL_TAIL_ONLY']:
            env.pop(key)
    (ROOT / 'logs').mkdir(exist_ok=True)
    for scope, folder in OUTPUTS.items():
        if not (ROOT / folder / 'SELECTED_FRAME_RESULTS.json').exists():
            with (ROOT / 'logs' / (scope + '.log')).open('w') as stream:
                subprocess.run([sys.executable, '-B', str(ROOT / 'render_scope.py'), str(ROOT), scope],
                               cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
        print('SCOPE_COMPLETE', scope, flush=True)
    import numpy as np
    results = {scope: load(ROOT / folder / 'SELECTED_FRAME_RESULTS.json') for scope, folder in OUTPUTS.items()}
    np.testing.assert_array_equal(np.load(EVAL / 'global_0.5/attempt_01/selector_selected_ids.npy'),
                                  np.load(ROOT / 'global_0.5/selector_selected_ids.npy'))
    original_hashes = load(EVAL / 'view_ratio_0.5/attempt_01/selection_hashes.json')
    assert load(ROOT / 'local_0.5/selection_hashes.json') == [original_hashes[p] for p in positions]
    differences = {}
    for scope, folder in DIRECTORIES.items():
        previous = read_csv(EVAL / folder / 'attempt_01/per_frame_metrics.csv')
        assert [(r['rgb_index'],r['synchronized_trajectory_index']) for r in previous] == \
               [(r['rgb_index'],r['synchronized_trajectory_index']) for r in full_rows]
        reference = {int(r['rgb_index']): r for r in previous}
        differences[scope] = {metric: max(abs(row[metric] - float(reference[row['gt_index']][metric]))
                               for row in results[scope]) for metric in ['psnr', 'ssim', 'lpips']}
        assert differences[scope]['psnr'] < .001 and differences[scope]['ssim'] < 1e-5 and differences[scope]['lpips'] < 1e-5
    images = ROOT / 'images'
    for folder in ['GT', 'Global_0.5', 'Local_0.5', 'Full']:
        (images / folder).mkdir(parents=True, exist_ok=True)
    rows = []
    for position, rgb_index in enumerate(indices):
        full, glob, local = [results[s][position] for s in ['full', 'global', 'view_ratio']]
        assert full['c2w'] == glob['c2w'] == local['c2w']
        stem = f'view_{position+1:02d}_frame_{rgb_index:06d}.png'
        shutil.copy2(Path(full['rgb_path']), images / 'GT' / stem)
        for result, name in [(glob, 'Global_0.5'), (local, 'Local_0.5'), (full, 'Full')]:
            shutil.copy2(result['image'], images / name / stem)
        rows.append(dict(view=position+1, rgb_index=rgb_index, trajectory_index=full['trajectory_index'],
            timestamp=full['timestamp'], global_psnr=glob['psnr'],local_psnr=local['psnr'],full_psnr=full['psnr'],
            global_ssim=glob['ssim'],local_ssim=local['ssim'],full_ssim=full['ssim'],
            global_lpips=glob['lpips'],local_lpips=local['lpips'],full_lpips=full['lpips'],
            global_K=glob['K'],local_K=local['K'],full_N=full['K'],local_candidate_N=local['candidate_N'],image=stem))
    with (ROOT / 'FRAME_METRICS.csv').open('w', newline='') as stream:
        writer=csv.DictWriter(stream,list(rows[0])); writer.writeheader(); writer.writerows(rows)
    after = {p: sha(Path(p)) for p in before}
    assert before == after
    write(ROOT / 'INPUTS_AFTER.json',after)
    write(ROOT / 'VERIFIED.json',dict(complete=True,views=5,images=20,training_steps=0,
        same_poses=True,global_ids_equal=True,local_hashes_equal=True,input_hashes_unchanged=len(before),
        maximum_differences= differences,means={s:{m:float(np.mean([r[m] for r in values]))
            for m in ['psnr','ssim','lpips']} for s,values in results.items()}))
    with tarfile.open(ROOT / 'RENDER_RESULTS.tar.gz','w:gz') as archive:
        for name in ['images','FRAME_METRICS.csv','EXPERIMENT.json','FRAME_SET.json','VERIFIED.json','INPUTS_BEFORE.json','INPUTS_AFTER.json']:
            archive.add(ROOT/name,arcname=name)
        for scope,folder in OUTPUTS.items():
            archive.add(ROOT/folder/'SELECTED_FRAME_RESULTS.json',arcname=folder+'/SELECTED_FRAME_RESULTS.json')
    print('ALL_VERIFIED',differences,flush=True)


if __name__=='__main__':
    main()
