"""读取已有日志，区分真实时间样本、启动前零样本和 checkpoint 重建样本。"""
import json,re
from pathlib import Path
import numpy as np
repo=Path('/home/prmi/zwj/mycartgs')
dest=Path(__file__).resolve().parent
paths=[repo/'results/protection_recheck_20261012/fr3_long_office_household/runtime.log']
paths+=list((repo/'results/gi_runtime_20261012').glob('*/run_*/run.log'))
paths+=list((repo/'results/newestfast/tum_rgbd/fr1_room').glob('*.log'))
paths+=list(Path('/home/prmi/zwj/chcl-validation-20261013/full-online/original').glob('*/train.log'))
rows=[]
for path in paths:
    for line in path.read_text(errors='replace').splitlines():
        if '[LOD Compact]' not in line and '[UltrafastRotation] normal_training_step=' not in line:continue
        if 'is_lod_step=0' in line:continue
        line=line.replace('mapping_iteration=','mapping_iter=').replace('tf_transformed_gaussians=','tf=')
        values=dict(re.findall(r'(mapping_iter|ratio|tf|active_gaussians)=([0-9.eE+-]+)',line))
        if len(values)==4:rows.append(dict(path=str(path),**{k:float(v) if k=='ratio' else int(v) for k,v in values.items()}))
summaries=[]
for ratio in sorted(set(v['ratio'] for v in rows)):
    all_rows=[v for v in rows if v['ratio']==ratio]
    a=np.array([v['tf'] for v in all_rows if v['tf']>0])
    summaries.append(dict(ratio=ratio,logged_rows=len(all_rows),zero_tf_rows=sum(v['tf']==0 for v in all_rows),
        active_tf_samples=len(a),min=int(a.min()) if len(a) else None,
        median=float(np.median(a)) if len(a) else None,p90=float(np.percentile(a,90)) if len(a) else None,
        max=int(a.max()) if len(a) else None))
(dest/'logged_tf_shapes.json').write_text(json.dumps(dict(paths=list(map(str,paths)),samples=rows,
    summaries=summaries,warning='sparse periodic logging and ratio scheduling bias; not a full iteration distribution'),indent=2))
print(json.dumps(summaries,indent=2))
