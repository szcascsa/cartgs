"""以同步NVTX区间关联真实Graph节点；不把Profile耗时当正式墙钟收益。"""
from pathlib import Path
import argparse,sqlite3,json
import pandas as pd

HERE=Path(__file__).resolve().parent
parser=argparse.ArgumentParser();parser.add_argument('name');args=parser.parse_args()
job=json.loads((HERE/'reports/jobs'/(args.name+'.json')).read_text())
assert job['state']=='complete' and job['foreign_active_samples']==0
connection=sqlite3.connect(HERE/'evaluation'/(args.name+'.sqlite'))
ranges=connection.execute("SELECT n.start,n.end,COALESCE(n.text,s.value) FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id=n.textId WHERE COALESCE(n.text,s.value) LIKE 'lfp/%' AND n.end IS NOT NULL").fetchall()
rows=[];calls=[];apis=[]
for index,(begin,end,label) in enumerate(ranges):
    _,scene,condition,variant=label.split('/');base=dict(scene=scene,condition=condition,variant=variant,call=index)
    kernels=connection.execute('SELECT k.end-k.start,s.value,k.gridX,k.gridY,k.blockX,k.blockY,k.registersPerThread,k.staticSharedMemory FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName WHERE k.start>=? AND k.end<=?',(begin,end)).fetchall()
    for duration,name,gx,gy,bx,by,registers,shared in kernels:
        if 'preprocessCUDA' in name:module='preprocess'
        elif 'DepthSort' in name:module='depth_sort'
        elif any(n in name for n in ['scheduleBits','executeBitTasks','executePreprocessedTasks','buildGaussianFootprint']):module='footprint'
        elif 'compactTileBits' in name:module='compact'
        elif 'renderMultiPixel' in name:module='pixel'
        elif 'clearGraphBits' in name:module='clear'
        elif 'updateGraphInputs' in name:module='input_update'
        else:module='other'
        rows.append(dict(**base,module=module,kernel=name,gpu_us=duration/1000,grid_x=gx,grid_y=gy,block_x=bx,block_y=by,registers=registers,shared_bytes=shared))
    for duration,name in connection.execute('SELECT a.end-a.start,s.value FROM CUPTI_ACTIVITY_KIND_RUNTIME a JOIN StringIds s ON s.id=a.nameId WHERE a.start>=? AND a.end<=?',(begin,end)).fetchall():
        apis.append(dict(**base,api=name,host_us=duration/1000))
    calls.append(dict(**base,kernels=len(kernels),gpu_us=sum(k[0] for k in kernels)/1000,nvtx_wall_us=(end-begin)/1000))
assert len(calls)==144,len(calls)
out=HERE/'reports'/args.name
pd.DataFrame(rows).to_csv(out/'KERNELS.csv',index=False);pd.DataFrame(apis).to_csv(out/'APIS.csv',index=False)
calls=pd.DataFrame(calls);calls.to_csv(out/'CALLS.csv',index=False)
summary=calls.groupby(['scene','condition','variant']).agg(kernel_launches=('kernels','mean'),gpu_kernel_us=('gpu_us','mean'),nvtx_wall_us=('nvtx_wall_us','mean'),calls=('call','size')).reset_index()
summary.to_csv(HERE/'reports/PROFILE_COMPARISON.csv',index=False)
modules=pd.DataFrame(rows).groupby(['scene','condition','variant','module']).agg(total_us=('gpu_us','sum'),kernels=('gpu_us','size')).reset_index()
modules=modules.merge(summary[['scene','condition','variant','calls']],on=['scene','condition','variant']);modules['us_per_render']=modules.total_us/modules.calls
modules.to_csv(HERE/'reports/PROFILE_MODULES.csv',index=False)
print(summary.round(4).to_string(index=False))
print(modules.pivot(index=['condition','module'],columns='variant',values='us_per_render').round(4).to_string())
