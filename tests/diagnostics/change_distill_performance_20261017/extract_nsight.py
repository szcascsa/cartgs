"""从原始Nsight SQLite导出单表CSV，保留原始stats stdout。"""
from pathlib import Path
import csv
import json
import sqlite3

D=Path(__file__).resolve().parent
summary={}
for branch in ['selector','tf']:
    db=sqlite3.connect(D/'raw'/f'nsys_{branch}.sqlite')
    fields=['kernel','calls','total_gpu_us','average_gpu_us','min_gpu_us','max_gpu_us']
    rows=db.execute('''SELECT s.value,count(*),sum(k.end-k.start)/1000.0,
                      avg(k.end-k.start)/1000.0,min(k.end-k.start)/1000.0,
                      max(k.end-k.start)/1000.0
                      FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s
                      ON k.demangledName=s.id GROUP BY k.demangledName
                      ORDER BY sum(k.end-k.start) DESC''').fetchall()
    with (D/'raw'/f'nsys_{branch}_kernels.csv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.writer(file);writer.writerow(fields);writer.writerows(rows)
    apis=db.execute('''SELECT s.value,count(*),sum(k.end-k.start)/1000.0,avg(k.end-k.start)/1000.0
                      FROM CUPTI_ACTIVITY_KIND_RUNTIME k JOIN StringIds s
                      ON k.nameId=s.id GROUP BY k.nameId ORDER BY sum(k.end-k.start) DESC''').fetchall()
    with (D/'raw'/f'nsys_{branch}_api.csv').open('w',newline='',encoding='utf-8') as file:
        writer=csv.writer(file);writer.writerow(['api','calls','total_cpu_us','average_cpu_us']);writer.writerows(apis)
    count,total,tiny=db.execute('''SELECT count(*),sum(end-start)/1e6,
                                 sum(CASE WHEN end-start<10000 THEN 1 ELSE 0 END)
                                 FROM CUPTI_ACTIVITY_KIND_KERNEL''').fetchone()
    summary[branch]={'kernel_calls':count,'total_kernel_gpu_ms':total,'tiny_kernel_calls_lt_10us':tiny}
    db.close()
(D/'raw/nsys_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
