"""仅统计TC_MEASURE内实际GPU kernel，Nsight扰动不作为墙钟收益。"""
from pathlib import Path
import sqlite3,json,csv,argparse
HERE=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('reports',nargs='+');args=p.parse_args();rows=[]
for label in args.reports:
    db=HERE/'reports'/(label+'.sqlite');con=sqlite3.connect(db)
    begin,end=con.execute("SELECT start,end FROM NVTX_EVENTS WHERE text='TC_MEASURE'").fetchone()
    for name,count,total,average in con.execute('SELECT s.value,count(*),sum(k.end-k.start)/1000.,avg(k.end-k.start)/1000. FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName WHERE k.start>=? AND k.end<=? GROUP BY k.demangledName',(begin,end)):
        rows.append(dict(profile=label,kernel=name,count=count,total_us=total,mean_us=average))
    con.close()
out=HERE/'reports/FINAL_PROFILE_KERNELS.csv'
with out.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
for label in args.reports:
    x=[r for r in rows if r['profile']==label];print(label,'launches',sum(r['count'] for r in x),'kernel_us',sum(r['total_us'] for r in x))
    for r in x:
        if 'selector' in r['kernel'] or 'compensation' in r['kernel']:print(r['kernel'][:85],r['count'],r['mean_us'])
