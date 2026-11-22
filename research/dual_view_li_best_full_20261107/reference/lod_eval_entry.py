"""以无窗口、无图片方式执行批次快照中的原有全轨迹评价。"""
from pathlib import Path
import csv
import json
import math
import runpy
import sys

# 批次queue.py不能遮蔽PyTorch依赖的Python标准库queue。
batch = Path(__file__).resolve().parent
sys.path[:] = [path for path in sys.path if Path(path or '.').resolve() != batch]

from evo.tools.settings import SETTINGS

SETTINGS.plot_backend = 'Agg'
script = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(script.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / 'evaluation_runtime'))
sys.argv = sys.argv[1:]
assert '--no-save-images' in sys.argv
assert '--output_path' in sys.argv
assert '--ply' not in sys.argv
values = runpy.run_path(str(script), run_name='__main__')
output = Path(values['output_result_path'])
rows = []
for index, (trajectory_index, rgb_index) in enumerate(values['associations']):
    row = dict(synchronized_trajectory_index=int(trajectory_index), rgb_index=int(rgb_index),
               trajectory_timestamp=float(values['tstamp'][trajectory_index]),
               rgb_timestamp=float(values['gt_tstamp'][rgb_index]),
               rgb_path=values['gt_color_paths'][rgb_index],
               psnr=float(values['psnr_list'][index]), ssim=float(values['ssim_list'][index]),
               lpips=float(values['lpips_list'][index]), render_ms=float(values['time_list'][index]))
    assert all(math.isfinite(row[name]) for name in ['psnr', 'ssim', 'lpips', 'render_ms'])
    rows.append(row)
assert rows
with (output / 'per_frame_metrics.csv').open('w', newline='') as stream:
    writer = csv.DictWriter(stream, list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
metrics = dict(translation=values['result'].stats, rotation=values['result_rotation_part'].stats,
               evaluated_frames=len(rows), gt_associated_trajectory_frames=len(values['tstamp']),
               raw_trajectory_frames=sum(bool(line.strip()) and not line.lstrip().startswith('#')
                   for line in (Path(values['source_result_path']) / 'CameraTrajectory_TUM.txt').read_text().splitlines()),
               dataset_rgb_frames=len(values['gt_tstamp']),
               correct_scale=values['args'].correct_scale,
               protocol='原有run.py：GT同步后的估计轨迹；RGB最近时间戳关联，阈值0.08秒；无帧采样')
(output / 'trajectory_metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + '\n')
summary = dict(evaluated_frames=len(rows), ate_rmse=float(metrics['translation']['rmse']),
               correct_scale=metrics['correct_scale'], saved_images=0)
for name in ['psnr', 'ssim', 'lpips', 'render_ms']:
    summary[name] = sum(row[name] for row in rows)/len(rows)
selector = values.get('selector_stats') or {}
summary.update(gaussian_count=selector.get('total_gaussians'),
    actual_K_mean=selector.get('active_gaussians'),
    actual_K_min=selector.get('active_gaussians_min', selector.get('active_gaussians')),
    actual_K_max=selector.get('active_gaussians_max', selector.get('active_gaussians')),
    target_K=selector.get('target_active', selector.get('target_total_gaussians')),
    budget_shortfall_views=selector.get('budget_shortfall_views', 0))
(output/'metric_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
