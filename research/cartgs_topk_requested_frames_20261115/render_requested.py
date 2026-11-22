"""只读复用已评价的CaRtGS Top-K集合，重渲染用户指定的两个视角。"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import sys
import time

ROOT = Path(__file__).resolve().parent
BASELINE = Path('/home/prmi/zwj/realLOD/saved_experiments/cartgs_pruning_baselines_20261112')
sys.path.insert(0, str(BASELINE))
from common import load, write, sha, budget, order_points
from worker import setup

CASES = {
    'replica_room0_mono': dict(group='Replica/room0/mono', image='frame000950.jpg',
                              input_frame=950, evaluation_frame=947),
    'tum_fr2_xyz_rgbd': dict(group='TUM_RGBD/freiburg2_xyz/rgbd', image='1311867188.831350.png',
                           input_frame=551, evaluation_frame=551),
}
RATIOS = ['0.01', '0.05', '0.1', '0.3']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', required=True, choices=CASES)
    args = parser.parse_args()
    case = CASES[args.case]
    folder = ROOT/'runs'/args.case
    folder.mkdir(parents=True, exist_ok=True)
    assert not (folder/'COMPLETE.json').exists(), '已有完整结果，禁止覆盖'
    group = next(g for g in load(BASELINE/'EXPERIMENTS.json')['groups'] if g['id'] == case['group'])
    histories = {r: BASELINE/'runs'/case['group']/'evaluation/opacity'/r for r in RATIOS}
    protected = {BASELINE/'EXPERIMENTS.json', BASELINE/'common.py', BASELINE/'worker.py', Path(group['ply'])}
    checkpoint = Path(group['vanilla']['checkpoint'])
    protected.update(p for p in checkpoint.rglob('*') if p.is_file() and p.suffix in ['.json', '.txt', '.yaml'])
    protected.update(p for p in (BASELINE/'runtime/python').glob('*.py'))
    protected.add(Path(group['vanilla']['evaluation'])/'evaluation_frames.json')
    for directory in histories.values():
        protected.update(directory/name for name in ['selection.json', 'selected_ids.npy', 'per_frame_metrics.csv', 'metric_summary.json'])
    before = {str(p): sha(p) for p in protected}
    write(folder/'INPUTS_BEFORE.json', before)

    import numpy as np
    import torch
    import cv2
    torch.set_num_threads(2)
    torch.manual_seed(0)
    started = time.monotonic()
    namespace, _ = setup(group, folder/'setup')
    from renderer import prepare_render_data, render_prepared
    from utils import MiniCam
    associations = namespace['associations']
    matches = [(i, pair) for i, pair in enumerate(associations)
               if Path(namespace['gt_color_paths'][pair[1]]).name == case['image']]
    assert len(matches) == 1, (case, matches)
    evaluation_frame, (trajectory_index, gt_index) = matches[0]
    assert evaluation_frame == case['evaluation_frame'] and gt_index == case['input_frame']
    camera = MiniCam(namespace['width'], namespace['height'], namespace['fovx'], namespace['fovy'],
                     np.linalg.inv(namespace['poses'][trajectory_index]))
    truth_path = Path(namespace['gt_color_paths'][gt_index])
    before[str(truth_path)] = sha(truth_path)
    write(folder/'INPUTS_BEFORE.json', before)
    truth_bgr = cv2.imread(str(truth_path), cv2.IMREAD_COLOR)
    truth_rgb = cv2.cvtColor(truth_bgr, cv2.COLOR_BGR2RGB)
    truth = torch.from_numpy(np.array(truth_rgb)).to('cuda').permute(2,0,1)[None]/255.0
    assert cv2.imwrite(str(folder/'GT.png'), truth_bgr)
    count = namespace['gaussians'].get_xyz.shape[0]
    opacity = namespace['gaussians'].get_opacity.detach().flatten().cpu().numpy()
    ranking = order_points('opacity', opacity)
    records = []
    with torch.no_grad():
        for ratio, directory in histories.items():
            selection = load(directory/'selection.json')
            selected = np.load(directory/'selected_ids.npy')
            k = budget(ratio, count)
            selected_sha = hashlib.sha256(selected.tobytes()).hexdigest()
            assert len(selected) == k and np.array_equal(selected, np.sort(ranking[:k]))
            assert selection['selected_sha256'] == selected_sha
            assert selection['checkpoint'] == group['vanilla']['checkpoint']
            assert selection['N'] == count and selection['K'] == k
            mask = torch.zeros(count, dtype=torch.bool, device='cuda')
            mask[torch.from_numpy(selected).to('cuda')] = True
            inputs = prepare_render_data(namespace['gaussians'], mask)
            assert inputs['inputs']['means3D'].shape[0] == k
            rendered = render_prepared(camera, inputs, namespace['background'])['render'].clamp(0,1)
            assert torch.isfinite(rendered).all()
            historical = list(csv.DictReader((directory/'per_frame_metrics.csv').open()))[evaluation_frame]
            assert int(historical['frame']) == evaluation_frame and int(historical['rgb_index']) == gt_index
            assert int(historical['trajectory_index']) == trajectory_index and int(historical['K']) == k
            metrics, errors = {}, {}
            for name in ['psnr', 'ssim', 'lpips']:
                metric = namespace['calc_'+name]
                metric.reset()
                metrics[name] = metric(rendered[None], truth).item()
                errors[name] = abs(metrics[name]-float(historical[name]))
                assert errors[name] < 2e-5, (case['group'], ratio, name, metrics[name], historical[name])
            output = folder/f'TOPK_{int(float(ratio)*100):02d}pct.png'
            rgb = np.uint8(rendered.permute(1,2,0).cpu().numpy()*255)
            assert cv2.imwrite(str(output), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            records.append(dict(case=args.case,group=case['group'],ratio=ratio,N=count,K=k,
                input_frame=int(gt_index),evaluation_frame=evaluation_frame,trajectory_index=int(trajectory_index),
                image_name=truth_path.name,reference_image=str(truth_path),checkpoint=group['vanilla']['checkpoint'],
                selected_ids_sha256=selected_sha,image_protocol='original_rgb',correct_scale=group['correct_scale'],
                **metrics,history_max_abs_error=max(errors.values()),image=str(output),image_sha256=sha(output)))
            print('RENDER_COMPLETE',args.case,ratio,'K',k,'PSNR',metrics['psnr'],flush=True)
    after = {p: sha(p) for p in before}
    assert after == before
    write(folder/'INPUTS_AFTER.json', after)
    with (folder/'METRICS.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer = csv.DictWriter(stream,list(records[0]));writer.writeheader();writer.writerows(records)
    write(folder/'COMPLETE.json',dict(complete=True,case=case,records=records,
        source_unchanged=True,selected_ids_match_history=True,metrics_match_history=True,
        width=int(namespace['width']),height=int(namespace['height']),seconds=time.monotonic()-started,
        png_conversion='原分辨率RGB，clamp[0,1]后乘255转uint8，不缩放不增强；PNG无损保存',
        training_steps=0,renders=4))
    print('CASE_COMPLETE',args.case,flush=True)


if __name__ == '__main__':
    main()
