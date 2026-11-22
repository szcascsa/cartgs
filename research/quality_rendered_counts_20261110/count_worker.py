"""复用冻结评价的输入与选点，补测每帧实际Alpha贡献点数，不计算整套质量指标。"""
from pathlib import Path
import argparse
import ast
import csv
import gc
import hashlib
import importlib.util
import json
import math
import sys
import time

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent/'all_scene_quality_figures_20261108/data/ALL_SCENE_RESULTS.json'
BINDING = Path('/home/prmi/zwj/experiments/fr1xyz_viewer_budget_comparison_20261105/build/fr1xyz_production_counts.so')
CAMERAS = {}  # 每个独立场景进程内，仅缓存两个冻结模型的有限轨迹相机。
ORIGINAL_BINDING = None


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def setup(group, item, output, vanilla=False):
    import numpy as np
    import torch
    from evo.tools.settings import SETTINGS
    SETTINGS.plot_backend = 'Agg'

    batch = Path(group['trained_model'].split('/runs/')[0])
    entry = batch/'eval_joint_entry.py'
    script = batch/'source/python/run_metrics_only.py'
    command = read(Path(item['output'])/'launch.json')['command']
    entry_index = command.index(str(entry))
    argv = list(command[entry_index:])
    argv[argv.index('--output_path')+1] = str(output)
    if vanilla:
        checkpoint = Path(group['vanilla']['checkpoint'])
        argv[2] = str(checkpoint.parent)
        argv[argv.index('--shutdown-model')+1] = checkpoint.name
    assert '--stereo-camera-config' not in argv
    assert '--no-save-images' in argv
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    sys.argv = argv
    # 逐字复用原入口的局部分支安装及随机数设置，止于其runpy调用前。
    prefix = entry.read_text().split('\nrunpy.run_path(sys.argv[0],run_name=')[0]
    assert prefix != entry.read_text()
    # 此任务不测模块耗时；仅关闭现有计时开关引入的额外同步，保留全部数据计算。
    call='ratio_budget=self.ratio,view_conditioned=True)'
    assert prefix.count(call)==1
    prefix=prefix.replace(call,'ratio_budget=self.ratio,view_conditioned=True,measure_time=False)')
    exec(compile(prefix, str(entry), 'exec'), {'__file__': str(entry), '__name__': 'count_entry_setup'})
    sys.argv = sys.argv[1:]
    text = script.read_text()
    marker = '    psnr_list, ssim_list, lpips_list, time_list = [], [], [], []'
    assert text.count(marker) == 1
    prefix = ast.parse(text.split(marker)[0], filename=str(script))
    # 三个评价网络在计数过程中没有消费者，不加载LPIPS权重或重复质量计算。
    prefix.body = [node for node in prefix.body if not (
        isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in
        ['calc_psnr', 'calc_ssim', 'calc_lpips'] for t in node.targets))]
    namespace = {'__name__': '__main__', '__file__': str(script)}
    exec(compile(prefix, str(script), 'exec'), namespace)
    if vanilla:
        expected = read(Path(group['vanilla']['evaluation'])/'evaluation_frames.json')
        assert expected['gt_indices'] == [int(pair[1]) for pair in namespace['associations']]
    else:
        expected = read(Path(item['output'])/'evaluation_frames.json')
        assert expected['gt_indices'] == [int(pair[1]) for pair in namespace['associations']]
        if item['scope'] == 'global':
            assert np.array_equal(np.load(Path(item['output'])/'selector_selected_ids.npy'),
                                  np.load(output/'selector_selected_ids.npy'))
    return namespace


def counted(binding, camera, data, background):
    import torch
    inputs = data['inputs']
    empty = torch.Tensor([])
    value = lambda name: inputs[name] if inputs[name] is not None else empty
    return binding.render_with_counts(background, value('means3D'), value('colors_precomp'),
        value('opacities'), value('scales'), value('rotations'), 1., value('cov3D_precomp'),
        camera.world_view_transform, camera.full_proj_transform,
        math.tan(camera.FoVx*.5), math.tan(camera.FoVy*.5), camera.image_height, camera.image_width,
        value('dc'), value('shs'), data['sh_degree'], camera.camera_center,
        False, False, True, binding.Mode.Inference)


def measure(group, item, binding, limit, vanilla=False):
    import cv2
    import numpy as np
    import torch
    from renderer import render_prepared
    from utils import MiniCam

    scope = 'vanilla' if vanilla else item['scope']
    name = 'vanilla_full' if vanilla else f"{scope}_{item['ratio']:g}"
    output = ROOT/('smoke' if limit else 'counts')/group['id']/name
    receipt = output/'summary.json'
    if receipt.exists():
        return read(receipt)
    assert not output.exists(), f'未完成现场需检查：{output}'
    output.mkdir(parents=True)
    namespace = setup(group, item, output, vanilla)
    associations = namespace['associations']
    previous = group['vanilla'] if vanilla else item
    expected_frames = previous['frames']
    assert len(associations) == expected_frames
    full_N = group['vanilla']['gaussians'] if vanilla else item['gaussian_count']
    if vanilla:
        stored = [{key: value for key, value in zip(['psnr', 'ssim', 'lpips'], values)}
                  for values in zip(*[[float(x) for x in (Path(previous['evaluation'])/(key+'.txt')).read_text().splitlines()]
                                      for key in ['psnr', 'ssim', 'lpips']])]
    else:
        stored = list(csv.DictReader((Path(item['output'])/'per_frame_metrics.csv').open()))
    expected_hashes = read(Path(item['output'])/'selection_hashes.json') if scope == 'view_ratio' else None
    expected_local = list(csv.DictReader((Path(item['output'])/'anchor_frustum_per_view.csv').open())) if expected_hashes else None
    frame_indices = list(range(min(limit, len(associations)))) if limit else list(range(len(associations)))
    probes = {frame_indices[0], frame_indices[len(frame_indices)//2], frame_indices[-1]}
    checks = []
    started = time.monotonic()
    gpu_counts, rows = [], []
    output_fields = ['frame', 'trajectory_index', 'rgb_index', 'submitted_K', 'projected_K', 'contributing_K',
                     'psnr', 'ssim', 'lpips']
    with (output/'PER_VIEW.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, output_fields)
        writer.writeheader()
        for frame in frame_indices:
            trajectory_index, rgb_index = associations[frame]
            if not vanilla:
                assert int(stored[frame]['synchronized_trajectory_index']) == trajectory_index
                assert int(stored[frame]['rgb_index']) == rgb_index
            camera_key = (namespace['source_result_path'], trajectory_index)
            if camera_key not in CAMERAS:
                CAMERAS[camera_key] = MiniCam(namespace['width'], namespace['height'], namespace['fovx'], namespace['fovy'],
                                             np.linalg.inv(namespace['poses'][trajectory_index]))
            camera = CAMERAS[camera_key]
            if namespace['frustum_renderer'] is not None:
                data, ids, statistics = namespace['frustum_renderer'].prepare(camera)
                digest = hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()
                assert digest == expected_hashes[frame], (group['id'], scope, frame, '选点身份变化')
                assert len(ids) == int(expected_local[frame]['active_gaussians'])
            else:
                data = namespace['render_data']
            K = int(data['inputs']['means3D'].shape[0])
            rendered = counted(binding, camera, data, namespace['background'])
            gpu_counts.append(torch.stack([(rendered[3] > 0).sum(), (rendered[5] > 0).sum()]))
            rows.append(dict(frame=frame, trajectory_index=int(trajectory_index), rgb_index=int(rgb_index),
                             submitted_K=K, **{key: float(stored[frame][key]) for key in ['psnr', 'ssim', 'lpips']}))
            if frame in probes:
                normal = render_prepared(camera, data, namespace['background'], return_radii=True)
                assert torch.equal(normal['render'].view(torch.int32), rendered[2].view(torch.int32))
                assert torch.equal(normal['radii'], rendered[3])
                if binding is not ORIGINAL_BINDING:
                    original=counted(ORIGINAL_BINDING,camera,data,namespace['background'])
                    assert torch.equal(original[2].view(torch.int32),rendered[2].view(torch.int32))
                    assert torch.equal(original[5]>0,rendered[5]>0)
                    del original
                truth = cv2.cvtColor(cv2.imread(namespace['gt_color_paths'][rgb_index], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                truth = torch.from_numpy(truth).to('cuda').permute(2, 0, 1)/255.
                quality = float(-10*torch.log10((rendered[2].clamp(0, 1)-truth).square().mean()))
                error = abs(quality-float(stored[frame]['psnr']))
                assert error < 2e-4, (group['id'], scope, frame, quality, stored[frame]['psnr'])
                checks.append(dict(frame=frame, render_and_radii_bitwise_equal=True, stored_psnr_abs_error=error))
                del normal, truth
            del rendered, camera
            if len(gpu_counts) >= 128 or frame == frame_indices[-1]:
                counters = torch.stack(gpu_counts).cpu().tolist()
                for row, (projected, contributed) in zip(rows[-len(counters):], counters):
                    assert 0 <= contributed <= projected <= row['submitted_K'] <= full_N
                    row.update(projected_K=projected, contributing_K=contributed)
                    writer.writerow(row)
                gpu_counts.clear()
                stream.flush()
                progress = dict(group=group['id'], scope=scope, ratio=item['ratio'], frame=frame+1,
                                frames=len(frame_indices), seconds=time.monotonic()-started)
                write(output/'PROGRESS.json', progress)
                print('PROGRESS', json.dumps(progress), flush=True)
    summary = dict(group=group['id'], scope=scope, ratio=1. if vanilla else item['ratio'],
                   ratio_kind='full' if vanilla else item['ratio_kind'], frames=len(rows),
                   gaussian_N=full_N, model_total_N=full_N,
                   psnr=previous['psnr'], ssim=previous['ssim'], lpips=previous['lpips'],
                   checkpoint=previous['checkpoint'] if vanilla else group['trained_model'],
                   quality_source=previous['evaluation'] if vanilla else item['output'],
                   count_source=str(output/'PER_VIEW.csv'),
                   checks=checks, images_saved=0, elapsed_seconds=time.monotonic()-started,
                   count_backend='legacy_GI' if binding is ORIGINAL_BINDING else 'boolean_contribution_flag')
    for key in ['submitted_K', 'projected_K', 'contributing_K']:
        values = [row[key] for row in rows]
        summary.update({key+'_mean': sum(values)/len(values), key+'_min': min(values), key+'_max': max(values)})
    summary['exported_gaussians'] = summary['submitted_K_mean'] if scope in ['global', 'vanilla', 'full'] else None
    if not limit:
        for key in ['psnr', 'ssim', 'lpips']:
            assert abs(sum(row[key] for row in rows)/len(rows)-summary[key]) < 1e-9
        if not vanilla:
            assert summary['submitted_K_mean'] == item['actual_K_mean']
            assert summary['submitted_K_min'] == item['actual_K_min']
            assert summary['submitted_K_max'] == item['actual_K_max']
    # 仅删除此次计数入口为验证而导出的临时PLY，不接触任何checkpoint或历史导出。
    exported = output/'point_cloud.ply'
    if exported.exists():
        assert exported.parent.resolve() == output.resolve()
        write(output/'TEMPORARY_EXPORT_REMOVED.json', dict(path=str(exported), sha256=sha(exported)))
        exported.unlink()
    write(receipt, summary)
    print('RECORD_COMPLETE', group['id'], name, 'contributing_mean', summary['contributing_K_mean'], flush=True)
    del namespace, data, stored
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def main():
    global ORIGINAL_BINDING
    parser = argparse.ArgumentParser()
    parser.add_argument('group')
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--one-ratio', type=float)
    args = parser.parse_args()
    group = next(g for g in read(SOURCE)['groups'] if g['id'] == args.group)
    batch = Path(group['trained_model'].split('/runs/')[0])
    sys.path[:0] = [str(batch/'source/python'), str(batch/'evaluation_runtime')]
    import torch
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    specification = importlib.util.spec_from_file_location('fr1xyz_production_counts', BINDING)
    binding = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(binding)
    ORIGINAL_BINDING=binding
    marker=ROOT/'BOOLEAN_COUNTER_VALIDATED.json'
    if marker.exists():
        fast_path=ROOT/'boolean_counter_build/boolean_render_counts.so'
        assert sha(fast_path)==read(marker)['sha256']
        specification=importlib.util.spec_from_file_location('boolean_render_counts',fast_path)
        binding=importlib.util.module_from_spec(specification)
        specification.loader.exec_module(binding)
    full, = [item for item in group['ours'] if item['scope'] == 'full']
    items = [item for item in group['ours'] if not args.one_ratio or item['scope'] == 'full' or item['ratio'] == args.one_ratio]
    results = []
    with torch.no_grad():
        results.append(measure(group, full, binding, args.limit, vanilla=True))
        for item in sorted(items, key=lambda item:(item['scope']!='full', item['scope'], item['ratio'])):
            results.append(measure(group, item, binding, args.limit))
    write(ROOT/('smoke' if args.limit else 'counts')/group['id']/'COMPLETE.json', results)


if __name__ == '__main__':
    main()
