"""真实Student同输入Rasterizer结果，条件等权、独立轮次及全部慢例。"""
from pathlib import Path
import json
import pandas as pd

HERE = Path(__file__).resolve().parent


def main():
    state = json.loads((HERE / 'reports/RASTER_CONTROLLER.json').read_text())
    assert state['state'] == 'complete'
    frames = {}
    for kind in ['samples', 'events', 'quality', 'checks', 'workloads', 'memory']:
        paths = [HERE / 'reports' / name / (kind + '.csv') for name in state['accepted']]
        frames[kind] = pd.concat([pd.read_csv(path) for path in paths if path.exists()], ignore_index=True)
        frames[kind].to_csv(HERE / 'reports' / ('RASTER_' + kind.upper() + '.csv'), index=False)
    keys = ['scene', 'condition', 'variant', 'round']
    stats = frames['samples'].groupby(keys).ms.agg(p50='median', p95=lambda x: x.quantile(.95), mean='mean').reset_index()
    ref = stats[stats.variant == 'F2'].drop(columns='variant').rename(columns={'p50': 'f2_p50', 'p95': 'f2_p95', 'mean': 'f2_mean'})
    stats = stats.merge(ref, on=['scene', 'condition', 'round'])
    stats['speedup'] = stats.f2_p50 / stats.p50
    stats.to_csv(HERE / 'reports/RASTER_SUMMARY.csv', index=False)
    quality = frames['quality'].groupby(['scene', 'condition', 'variant']).agg(
        psnr_loss=('psnr_loss', 'mean'), worst_frame=('psnr_loss', 'max'),
        ssim_loss=('ssim', 'mean'), baseline_ssim=('baseline_ssim', 'mean'),
        lpips=('lpips', 'mean'), baseline_lpips=('baseline_lpips', 'mean')).reset_index()
    quality['ssim_loss'] = quality.baseline_ssim - quality.ssim_loss
    quality['lpips_increase'] = quality.lpips - quality.baseline_lpips
    quality.to_csv(HERE / 'reports/RASTER_QUALITY_CONDITIONS.csv', index=False)
    primary = [f'local{x:g}' for x in [.1, .2, .3, .4, .5]]
    overview = []
    for name, part in stats[stats.condition.isin(primary)].groupby('variant'):
        q = quality[quality.variant == name]
        overview.append(dict(variant=name, primary_speedup=float(part.speedup.mean()),
            worst_speedup=float(part.speedup.min()), faster=int((part.speedup > 1).sum()), tests=len(part),
            quality_loss=float(q.psnr_loss.mean()) if len(q) else 0,
            worst_condition=float(q.psnr_loss.max()) if len(q) else 0,
            worst_frame=float(q.worst_frame.max()) if len(q) else 0))
    print(json.dumps(overview, indent=2))
    (HERE / 'reports/RASTER_OVERVIEW.json').write_text(json.dumps(overview, indent=2))
    assert frames['checks'].radii_bitwise.all()
    assert len(frames['samples'].scene.unique()) == 18


if __name__ == '__main__':
    main()
