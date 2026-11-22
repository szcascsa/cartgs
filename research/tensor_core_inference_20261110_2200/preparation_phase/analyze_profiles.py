from pathlib import Path
import sqlite3,json,csv,hashlib,subprocess,re
import numpy as np

HERE=Path(__file__).resolve().parent;R=HERE/'reports';rows=[];api=[];copies=[]
for label in ['profile_B','profile_C','profile_D','profile_C_large','profile_D_large']:
    con=sqlite3.connect(R/(label+'.sqlite'));begin,end=con.execute("SELECT start,end FROM NVTX_EVENTS WHERE text='PREPARATION_MEASURE'").fetchone()
    for name,count,total,average in con.execute('SELECT s.value,count(*),sum(k.end-k.start)/1000.,avg(k.end-k.start)/1000. FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName WHERE k.start>=? AND k.end<=? GROUP BY k.demangledName',(begin,end)):
        stage='other'
        if 'selector' in name:stage='selector'
        elif 'compensation' in name:stage='compensator'
        elif 'preparationProjectAnchors' in name:stage='projected_dynamic_input'
        elif 'viewKernel' in name:stage='view'
        elif 'shrinkingRadix' in name or 'emitSelected' in name or 'countThreshold' in name:stage='selection'
        elif 'gemm' in name:stage='camera_or_frustum_matrix'
        elif 'preparationFrustum' in name or 'makePlanes' in name or 'Frustum' in name or 'testAndCount' in name:stage='frustum'
        elif 'cub::' in name or 'Candidates' in name or 'count' in name.lower():stage='candidates'
        elif 'frameControls' in name:stage='controls'
        durations=[x[0]/1000 for x in con.execute('SELECT k.end-k.start FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName WHERE k.start>=? AND k.end<=? AND s.value=?',(begin,end,name))]
        rows.append(dict(profile=label,stage=stage,kernel=name,count=count,total_us=total,mean_us=average,us_per_frame=total/36,
            p50_us=float(np.median(durations)),p95_us=float(np.percentile(durations,95))))
    for name,count,total in con.execute('SELECT s.value,count(*),sum(k.end-k.start)/1000. FROM CUPTI_ACTIVITY_KIND_RUNTIME k JOIN StringIds s ON s.id=k.nameId WHERE k.start>=? AND k.end<=? GROUP BY k.nameId',(begin,end)):
        api.append(dict(profile=label,api=name,count=count,total_us=total))
    tables={x[0] for x in con.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    if 'CUPTI_ACTIVITY_KIND_MEMCPY' in tables:
        for kind,count,bytes_,total in con.execute('SELECT copyKind,count(*),sum(bytes),sum(end-start)/1000. FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE start>=? AND end<=? GROUP BY copyKind',(begin,end)):
            copies.append(dict(profile=label,copy_kind=kind,count=count,bytes=bytes_,total_us=total))
    con.close()
for name,data in [('PROFILE_KERNELS.csv',rows),('PROFILE_RUNTIME.csv',api),('PROFILE_MEMCPY.csv',copies)]:
    with (R/name).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(data[0]));writer.writeheader();writer.writerows(data)
binary=HERE/'build/mycartgs_tensor_preparation_v1.so'
sass=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-sass',str(binary)],text=True)
functions=[]
for part in re.split(r'Function : ',sass)[1:]:
    name=part.splitlines()[0]
    if 'selectorTensor' not in name and 'compensationTensor' not in name and 'preparationTensorDistance' not in name:continue
    instructions=[x.strip() for x in part.splitlines() if 'HMMA.' in x]
    functions.append(dict(function=name,hmma=len(instructions),examples=instructions[:2]))
assert any(x['hmma'] and 'compensationTensor' in x['function'] for x in functions)
(R/'TENSOR_SASS_EVIDENCE.json').write_text(json.dumps(dict(binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest(),functions=functions),indent=2))
resources=subprocess.check_output(['/usr/local/cuda/bin/cuobjdump','--dump-resource-usage',str(binary)],text=True)
(R/'CUDA_RESOURCE_USAGE.txt').write_text(resources)
print('PROFILE_AND_HMMA_ANALYSIS_PASS',len(rows),len(api),len(functions))
