"""以同步NVTX范围解析CUDA Graph实际节点，保留API和寄存器证据。"""
from pathlib import Path
import argparse
import sqlite3
import pandas as pd

ROOT=Path(__file__).resolve().parent
a=argparse.ArgumentParser();a.add_argument('database');a.add_argument('--out',required=True);args=a.parse_args()
con=sqlite3.connect(ROOT/args.database)
out=ROOT/args.out;out.mkdir(exist_ok=False)
ranges=con.execute("SELECT n.start,n.end,COALESCE(n.text,s.value) FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id=n.textId WHERE COALESCE(n.text,s.value) LIKE 'scaling/%' AND n.end IS NOT NULL").fetchall()
rows,calls,apis=[],[],[]
for number,(begin,end,label) in enumerate(ranges):
    parts=label.split('/')
    scene,condition,variant=parts[1:] if len(parts)==4 else ('Replica__office2__rgbd',*parts[1:])
    base=dict(scene=scene,condition=condition,variant=variant,call=number)
    kernels=con.execute('SELECT k.end-k.start,s.value,k.gridX,k.gridY,k.blockX,k.blockY,k.registersPerThread,k.staticSharedMemory FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName WHERE k.start>=? AND k.end<=?',(begin,end)).fetchall()
    for duration,name,gx,gy,bx,by,registers,shared in kernels:
        if 'preprocessCUDA' in name:module='preprocess'
        elif any(x in name for x in ('DepthSort','smallDepthSort','warpDepthSort')):module='sort'
        elif any(x in name for x in ('scheduleBits','executeBitTasks','executePreprocessedTasks')):module='footprint'
        elif 'compactTileBits' in name:module='compact_and_background'
        elif any(x in name for x in ('renderMultiPixel','renderScalingPixels','renderPackedPixels')):module='pixel'
        elif 'clearGraphBits' in name:module='clear'
        elif 'updateGraphInputs' in name:module='input_update'
        else:module='other'
        rows.append(dict(**base,module=module,kernel=name,gpu_us=duration/1000,grid_x=gx,grid_y=gy,
                         block_x=bx,block_y=by,registers=registers,shared_bytes=shared))
    for duration,name in con.execute('SELECT a.end-a.start,s.value FROM CUPTI_ACTIVITY_KIND_RUNTIME a JOIN StringIds s ON s.id=a.nameId WHERE a.start>=? AND a.end<=?',(begin,end)).fetchall():
        apis.append(dict(**base,api=name,host_us=duration/1000))
    calls.append(dict(**base,kernels=len(kernels),gpu_us=sum(k[0] for k in kernels)/1000))
k=pd.DataFrame(rows);c=pd.DataFrame(calls)
k.to_csv(out/'KERNELS.csv',index=False);c.to_csv(out/'CALLS.csv',index=False);pd.DataFrame(apis).to_csv(out/'APIS.csv',index=False)
summary=c.groupby(['scene','condition','variant']).agg(kernels=('kernels','mean'),gpu_us=('gpu_us','mean'),calls=('call','size')).reset_index()
summary.to_csv(out/'SUMMARY.csv',index=False)
modules=k.groupby(['scene','condition','variant','module']).gpu_us.sum().reset_index()
modules=modules.merge(summary[['scene','condition','variant','calls']],on=['scene','condition','variant'])
modules['us_per_render']=modules.gpu_us/modules.calls
modules.to_csv(out/'MODULES.csv',index=False)
print(summary.to_string(index=False))
