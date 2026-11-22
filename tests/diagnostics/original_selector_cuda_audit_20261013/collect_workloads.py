"""当前 evidence-aware 日志；保留逐行来源，拒绝填充未记录的 Mature N。"""
import json,re
from collections import defaultdict
from pathlib import Path
import numpy as np
dest=Path(__file__).resolve().parent;repo=Path('/home/prmi/zwj/mycartgs')
paths=[repo/'results/protection_recheck_20261012/fr3_long_office_household/runtime.log']
paths+=list((repo/'results/gi_runtime_20261012').glob('*/run_*/run.log'))
paths+=list(Path('/home/prmi/zwj/chcl-validation-20261013/full-online/original').glob('*/train.log'))
rows=[];mapping=[];gi=[];protection={}
for path in paths:
    if not path.exists():continue
    local=[]
    for line_no,line in enumerate(path.read_text(errors='replace').splitlines(),1):
        if '[Protection STATE]' in line:
            v={k:float(x) for k,x in re.findall(r'(\w+)=([-+0-9.eE]+)',line)};protection[(str(path),int(v['mapping_iter']))]=v
        if '[Mapping PERF]' in line: mapping.append(dict(path=str(path),line=line_no,text=line))
        if '[SGI ' in line and ('PERF' in line or 'mapping_' in line):gi.append(dict(path=str(path),line=line_no,text=line))
        if '[LOD Compact]' not in line and '[UltrafastRotation] normal_training_step=' not in line:continue
        if 'is_lod_step=0' in line:continue
        v={k:float(x) for k,x in re.findall(r'(\w+)=([-+0-9.eE]+)',line)}
        if 'ratio' not in v:continue
        v.update(path=str(path),line=line_no)
        if 'mapping_iteration' in v:v['mapping_iter']=v['mapping_iteration']
        if 'tf_transformed_gaussians' in v:v['tf']=v['tf_transformed_gaussians']
        local.append(v)
    for v in local:
        state=protection.get((str(path),int(v.get('mapping_iter',-1))))
        if state:
            v['N_network']=state['mature_gaussians'];v['protected_population']=state['protected_gaussians'];v['total_gaussians']=v['N_network']+v['protected_population']
    rows+=local
def summarize(subset):
    out={}
    for key in ['total_gaussians','N_network','active_gaussians','protected_population','protection_actual','mature_actual','mature_budget_target','tf','training_ms','render_ms']:
        values=[v[key] for v in subset if key in v]
        if values:out[key]=dict(samples=len(values),**dict(zip(['min','median','p90','p95','max'],map(float,np.quantile(values,[0,.5,.9,.95,1])))))
    return out
payload=dict(paths=list(map(str,paths)),rows=len(rows),ratios={str(r):summarize([v for v in rows if abs(v['ratio']-r)<1e-5]) for r in [.01,.05,.1,.15]},
    network_quantiles_note='Only same-iteration Protection STATE is accepted as exact Mature/network population; absent values stay missing. Normal and UltrafastRotation are separated in paths and per_source.',
    per_source={str(p):{str(r):summarize([v for v in rows if v['path']==str(p) and abs(v['ratio']-r)<1e-5]) for r in [.01,.05,.1,.15]} for p in paths})
(dest/'workload_rows.json').write_text(json.dumps(rows,indent=2));(dest/'workload_quantiles.json').write_text(json.dumps(payload,indent=2));(dest/'mapping_gi_logs.json').write_text(json.dumps(dict(mapping=mapping,gi=gi),indent=2))
print(json.dumps({k:payload[k] for k in ['rows','ratios']},indent=2))
