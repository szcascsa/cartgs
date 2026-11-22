"""纯渲染开发集全部候选，保留同二进制控制与逐轮慢例。"""
from pathlib import Path
import argparse
import json
import pandas as pd
import numpy as np

HERE=Path(__file__).resolve().parent


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage');parser.add_argument('--prefix',default='');args=parser.parse_args()
    label=args.stage+('_'+args.prefix if args.prefix else '')
    state=json.loads((HERE/'reports'/(args.stage+'_CONTROLLER.json')).read_text())
    assert state['state']=='complete'
    paths=[HERE/'reports'/name for name in state['accepted'] if name.startswith(args.prefix)]
    samples=pd.concat([pd.read_csv(path/'samples.csv') for path in paths])
    stats=samples.groupby(['scene','condition','round','variant']).ms.agg(p50='median',p95=lambda x:x.quantile(.95)).reset_index()
    for variant,ref_label in [('F2','f2'),('tile_safe_bitexp_stop','old38')]:
        ref=stats.loc[stats.variant==variant,['scene','condition','round','p50','p95']].rename(columns={'p50':ref_label+'_p50','p95':ref_label+'_p95'})
        stats=stats.merge(ref,on=['scene','condition','round'])
        stats[ref_label+'_speedup']=stats[ref_label+'_p50']/stats.p50
    stats.to_csv(HERE/'reports'/(label+'_SUMMARY.csv'),index=False)
    quality=pd.concat([pd.read_csv(path/'quality.csv') for path in paths if (path/'quality.csv').exists()])
    q=quality.groupby(['scene','condition','variant']).agg(psnr_loss=('psnr_loss','mean'),worst_frame=('psnr_loss','max'),
        baseline_ssim=('baseline_ssim','mean'),ssim=('ssim','mean'),baseline_lpips=('baseline_lpips','mean'),lpips=('lpips','mean')).reset_index()
    q.to_csv(HERE/'reports'/(label+'_QUALITY.csv'),index=False)
    rows=[]
    primary=[f'local{x:g}' for x in [.1,.2,.3,.4,.5]]
    for name,part in stats[stats.condition.isin(primary)].groupby('variant'):
        values=q[q.variant==name]
        rows.append(dict(variant=name,f2_speedup=float(np.exp(np.log(part.f2_speedup).mean())),
            old38_speedup=float(np.exp(np.log(part.old38_speedup).mean())),worst_old38_speedup=float(part.old38_speedup.min()),
            mean_loss=float(values.psnr_loss.mean()) if len(values) else 0,
            worst_condition=float(values.psnr_loss.max()) if len(values) else 0,
            worst_frame=float(values.worst_frame.max()) if len(values) else 0))
    output=pd.DataFrame(rows);output.to_csv(HERE/'reports'/(label+'_OVERVIEW.csv'),index=False)
    print(output.to_string(index=False))


if __name__=='__main__':main()
