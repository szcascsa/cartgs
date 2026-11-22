"""Profile按NVTX窗口关联实际CUDA Graph节点；不可代替独占墙钟。"""
from pathlib import Path
import argparse
import sqlite3
import pandas as pd

HERE=Path(__file__).resolve().parent
a=argparse.ArgumentParser();a.add_argument('name');args=a.parse_args()
con=sqlite3.connect(HERE/'reports'/(args.name+'.sqlite'))
out=HERE/'reports/PROFILE';out.mkdir(exist_ok=False)
ranges=con.execute("SELECT n.start,n.end,COALESCE(n.text,s.value) FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id=n.textId WHERE COALESCE(n.text,s.value) LIKE 'footprint/%' AND n.end IS NOT NULL").fetchall()
rows,calls,apis=[],[],[]
for number,(begin,end,label) in enumerate(ranges):
    _,scene,cond,variant=label.split('/')
    base=dict(scene=scene,condition=cond,variant=variant,call=number)
    kernels=con.execute('SELECT k.end-k.start,s.value,k.gridX,k.gridY,k.blockX,k.blockY,k.registersPerThread,k.staticSharedMemory FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName WHERE k.start>=? AND k.end<=?',(begin,end)).fetchall()
    for dt,name,gx,gy,bx,by,registers,shared in kernels:
        if 'preprocessCUDA' in name:module='preprocess'
        elif any(t in name for t in ('DepthSort','warpDepthSort','smallDepthSort')):module='sort'
        elif any(t in name for t in ('scheduleBits','executeBitTasks','executePreprocessedTasks')):module='footprint'
        elif 'compactTileBits' in name:module='compact'
        elif any(t in name for t in ('renderScalingPixels','renderExactCommonPixels','renderEarlyExitPixels')):module='pixel'
        elif 'clearGraphBits' in name:module='clear'
        elif 'updateGraphInputs' in name:module='input'
        else:module='other'
        rows.append(dict(**base,module=module,kernel=name,gpu_us=dt/1000,grid_x=gx,grid_y=gy,block_x=bx,block_y=by,
                         registers=registers,shared_bytes=shared))
    for dt,name in con.execute('SELECT a.end-a.start,s.value FROM CUPTI_ACTIVITY_KIND_RUNTIME a JOIN StringIds s ON s.id=a.nameId WHERE a.start>=? AND a.end<=?',(begin,end)).fetchall():
        apis.append(dict(**base,api=name,host_us=dt/1000))
    calls.append(dict(**base,kernels=len(kernels),gpu_us=sum(x[0] for x in kernels)/1000))
k=pd.DataFrame(rows);c=pd.DataFrame(calls)
k.to_csv(out/'KERNELS.csv',index=False);c.to_csv(out/'CALLS.csv',index=False);pd.DataFrame(apis).to_csv(out/'APIS.csv',index=False)
summary=c.groupby(['scene','condition','variant']).agg(kernels=('kernels','mean'),gpu_us=('gpu_us','mean'),calls=('call','size')).reset_index()
summary.to_csv(out/'SUMMARY.csv',index=False)
modules=k.groupby(['scene','condition','variant','module']).gpu_us.sum().reset_index()
modules=modules.merge(summary[['scene','condition','variant','calls']],on=['scene','condition','variant'])
modules['us_per_render']=modules.gpu_us/modules.calls
modules.to_csv(out/'MODULES.csv',index=False)
print(summary.to_string(index=False))
