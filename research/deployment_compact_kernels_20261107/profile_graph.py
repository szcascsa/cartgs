from common import ROOT,save
import sqlite3
import statistics
from collections import defaultdict


def main():
    db=sqlite3.connect(ROOT/'reports/PREPARE_PROFILE.sqlite')
    db.row_factory=sqlite3.Row
    strings=dict(db.execute('select id,value from StringIds'))
    kernels=list(db.execute('select * from CUPTI_ACTIVITY_KIND_KERNEL'))
    api=list(db.execute('select * from CUPTI_ACTIVITY_KIND_RUNTIME'))
    copies=list(db.execute('select * from CUPTI_ACTIVITY_KIND_MEMCPY'))
    groups=defaultdict(list)
    for scope in db.execute('select * from NVTX_EVENTS where end is not null'):
        label=scope['text'] or strings.get(scope['textId'],'')
        if not label.startswith('prepare/'):continue
        # The harness synchronizes before entering and before leaving each range.
        # It has one request in flight, so graph child nodes belong by timestamps.
        work=[x for x in kernels if scope['start']<=x['start'] and x['end']<=scope['end']]
        calls=[x for x in api if x['globalTid']==scope['globalTid'] and scope['start']<=x['start'] and x['end']<=scope['end']]
        mem=[x for x in copies if scope['start']<=x['start'] and x['end']<=scope['end']]
        by_name=defaultdict(lambda:dict(calls=0,gpu_us=0.))
        for kernel in work:
            name=strings[kernel['demangledName']]
            by_name[name]['calls']+=1
            by_name[name]['gpu_us']+=(kernel['end']-kernel['start'])/1000
        groups[label].append(dict(wall_us=(scope['end']-scope['start'])/1000,
            kernel_nodes=len(work),kernel_us=sum(x['end']-x['start'] for x in work)/1000,
            cpu_launches=sum('Launch' in strings[x['nameId']] for x in calls),
            syncs=sum('Synchronize' in strings[x['nameId']] for x in calls),
            sync_us=sum(x['end']-x['start'] for x in calls if 'Synchronize' in strings[x['nameId']])/1000,
            dtoh=sum(x['copyKind']==2 for x in mem),dtoh_bytes=sum(x['bytes'] for x in mem if x['copyKind']==2),
            by_name=by_name))
    result=[]
    for label,rows in groups.items():
        totals=defaultdict(lambda:dict(calls=0.,gpu_us=0.))
        for row in rows:
            for name,values in row['by_name'].items():
                for key,value in values.items():totals[name][key]+=value/len(rows)
        row=dict(label=label,samples=len(rows),
            **{key:statistics.median(x[key] for x in rows) for key in rows[0] if key!='by_name'},
            top=sorted([dict(name=name,**x) for name,x in totals.items()],key=lambda x:x['gpu_us'],reverse=True))
        result.append(row)
    save(ROOT/'reports/PREPARE_PROFILE_ANALYSIS.json',dict(summary=result,attribution='synchronized NVTX range timestamps'))
    for row in result:print({k:v for k,v in row.items() if k!='top'},flush=True)


if __name__=='__main__':main()
