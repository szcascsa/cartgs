"""开发集按同场景同Ratio配对，不用网络理论吞吐替代实测。"""
from pathlib import Path
import argparse,json
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('run');args=p.parse_args();root=HERE/'reports'/args.run
d=pd.read_csv(root/'samples.csv');q=pd.read_csv(root/'quality.csv')
t=d.groupby(['scene','ratio','variant']).ms.median().unstack();rows=[]
for variant in t.columns:
    if variant=='original':continue
    g=q[q.variant==variant];per=g.groupby(['scene','ratio']).psnr_loss.mean();speedup=t.original/t[variant]
    rows.append(dict(variant=variant,speedup=float(np.exp(np.log(speedup).mean())),slower_conditions=int((speedup<1).sum()),conditions=len(speedup),
        mean_psnr_loss=float(g.psnr_loss.mean()),worst_frame_loss=float(g.psnr_loss.max()),worst_condition_mean_loss=float(per.max()),
        same_ids_fraction=float(g.same_ids.mean()),quality_passed=bool(g.psnr_loss.mean()<=.05 and per.max()<=.2)))
result=pd.DataFrame(rows).sort_values('speedup',ascending=False);result.to_csv(root/'PAIRED_SUMMARY.csv',index=False);print(result.to_string(index=False))
