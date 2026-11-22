from pathlib import Path
import csv
import json
import sqlite3
import sys
D=Path(__file__).resolve().parent
result={}
for stage in sys.argv[1:]:
    p=D/stage/'raw/nsys_tf/trace.sqlite'
    if not p.exists(): continue
    db=sqlite3.connect(p)
    # SQLite is a derived export; the original .nsys-rep remains unchanged.
    db.execute('CREATE INDEX IF NOT EXISTS cd_runtime_tid_start ON CUPTI_ACTIVITY_KIND_RUNTIME(globalTid,start)')
    db.execute('CREATE INDEX IF NOT EXISTS cd_kernel_corr ON CUPTI_ACTIVITY_KIND_KERNEL(correlationId)')
    db.execute('CREATE INDEX IF NOT EXISTS cd_memcpy_corr ON CUPTI_ACTIVITY_KIND_MEMCPY(correlationId)')
    db.commit()
    names=['geometry_transport','transport_plan_population','transport_compact_gather','transport_xyz_gemm','transport_fused_postprocess_scatter']
    total=db.execute('SELECT count(*),sum(end-start)/1e6,sum(end-start<10000) FROM CUPTI_ACTIVITY_KIND_KERNEL').fetchone()
    rec={'kernel_calls':total[0],'gpu_kernel_ms':total[1],'tiny_lt_10us':total[2],'scopes':{}}
    for name in names:
        selector='r.globalTid=n.globalTid AND r.start>=n.start AND r.start<n.end'
        args=['tf/'+name]
        row=db.execute('SELECT count(*),sum(k.end-k.start)/1e6,sum(k.end-k.start<10000) FROM NVTX_EVENTS n JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON '+selector+' JOIN CUPTI_ACTIVITY_KIND_KERNEL k ON k.correlationId=r.correlationId WHERE n.text=?',args).fetchone()
        copies=db.execute('SELECT m.copyKind,count(*),sum(m.bytes),sum(m.end-m.start)/1e6 FROM NVTX_EVENTS n JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON '+selector+' JOIN CUPTI_ACTIVITY_KIND_MEMCPY m ON m.correlationId=r.correlationId WHERE n.text=? GROUP BY m.copyKind',args).fetchall()
        rt=db.execute('SELECT count(*),sum(r.end-r.start)/1e6 FROM NVTX_EVENTS n JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON '+selector+' WHERE n.text=?',args).fetchone()
        intervals=db.execute('SELECT k.start,k.end FROM NVTX_EVENTS n JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON '+selector+' JOIN CUPTI_ACTIVITY_KIND_KERNEL k ON k.correlationId=r.correlationId WHERE n.text=? ORDER BY k.start',args).fetchall()
        # Includes gaps between separate transport invocations. Report only gaps within each invocation below.
        gaps=0;cpu_gaps=0
        for start,end,tid in db.execute('SELECT start,end,globalTid FROM NVTX_EVENTS WHERE text=?',args):
            ks=db.execute('SELECT k.start,k.end FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN CUPTI_ACTIVITY_KIND_KERNEL k ON k.correlationId=r.correlationId WHERE r.globalTid=? AND r.start>=? AND r.start<? ORDER BY k.start',(tid,start,end)).fetchall()
            gaps+=sum(max(0,ks[i][0]-ks[i-1][1]) for i in range(1,len(ks)))
            launches=db.execute('SELECT r.start,r.end FROM CUPTI_ACTIVITY_KIND_RUNTIME r WHERE r.globalTid=? AND r.start>=? AND r.start<? AND EXISTS(SELECT 1 FROM CUPTI_ACTIVITY_KIND_KERNEL k WHERE k.correlationId=r.correlationId) ORDER BY r.start',(tid,start,end)).fetchall()
            cpu_gaps+=sum(max(0,launches[i][0]-launches[i-1][1]) for i in range(1,len(launches)))
        categories=db.execute('SELECT s.value,count(*),sum(k.end-k.start)/1e6 FROM NVTX_EVENTS n JOIN CUPTI_ACTIVITY_KIND_RUNTIME r ON '+selector+' JOIN CUPTI_ACTIVITY_KIND_KERNEL k ON k.correlationId=r.correlationId JOIN StringIds s ON k.demangledName=s.id WHERE n.text=? GROUP BY k.demangledName ORDER BY sum(k.end-k.start) DESC',args).fetchall()
        rec['scopes'][name]={'kernel_calls':row[0],'gpu_kernel_ms':row[1],'tiny_lt_10us':row[2],'memcpy_kind_count_bytes_ms':copies,'runtime_api_calls':rt[0],'runtime_api_ms':rt[1],'inter_kernel_gap_ms':gaps/1e6,'between_launch_api_gap_ms':cpu_gaps/1e6,'kernels':categories}
    rec['all_memcpy']=db.execute('SELECT copyKind,count(*),sum(bytes),sum(end-start)/1e6 FROM CUPTI_ACTIVITY_KIND_MEMCPY GROUP BY copyKind').fetchall()
    kernels=db.execute('SELECT s.value,count(*),sum(k.end-k.start)/1e6,avg(k.end-k.start)/1000.0 FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id GROUP BY k.demangledName ORDER BY sum(k.end-k.start) DESC').fetchall()
    with (p.parent/'kernels.csv').open('w') as file:
        w=csv.writer(file);w.writerow(['kernel','calls','total_gpu_ms','average_us']);w.writerows(kernels)
    db.close()
    result[stage]=rec
(D/'nsys_summary.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
