"""将实际网络Tensor Kernel与渲染Kernel按名称分开归因。"""
from pathlib import Path
import sqlite3,json,csv
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE/'reports/profiles'
rows=[];summary=[]
for case in json.loads((ROOT/'MANIFEST.json').read_text()):
    conn=sqlite3.connect(case['sqlite'])
    begin,end=conn.execute("SELECT start,end FROM NVTX_EVENTS WHERE text='TC_SQUARE_MEASURE'").fetchone()
    perstage={}
    for name,count,total in conn.execute('SELECT s.value,count(*),sum(k.end-k.start)/1000. FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName WHERE k.start>=? AND k.end<=? GROUP BY k.demangledName',(begin,end)):
        if 'selectorTensor' in name:stage='tensor_selector_kernel'
        elif 'compensationTensor' in name:stage='tensor_compensator_kernel_including_decode_publish'
        elif 'preparationProjectAnchors' in name:stage='dynamic_projection_cuda'
        else:stage='other_preparation_or_rendering'
        values=[r[0]/1000 for r in conn.execute('SELECT k.end-k.start FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName WHERE k.start>=? AND k.end<=? AND s.value=?',(begin,end,name))]
        rows.append(dict(profile=case['name'],scene=case['scene'],ratio=case['ratio'],stage=stage,kernel=name,
            count=count,total_us=total,mean_us=total/count,p50_us=float(np.median(values)),p95_us=float(np.percentile(values,95)),us_per_frame=total/case['frames']))
        perstage[stage]=perstage.get(stage,0)+total
    networks={k:v/case['frames'] for k,v in perstage.items() if k.startswith('tensor_')}
    assert len(networks)==2,(case,perstage)
    for stage,us in networks.items():summary.append(dict(profile=case['name'],scene=case['scene'],ratio=case['ratio'],stage=stage,us_per_frame=us,frames=case['frames']))
    conn.close()
for name,data in [('PROFILE_KERNELS.csv',rows),('PROFILE_NETWORKS.csv',summary)]:
    with (HERE/'reports'/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
print('PROFILE_ANALYSIS_PASS',len(rows),len(summary))
