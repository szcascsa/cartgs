"""按已同步NVTX范围统计真实Graph节点、Host API及模块；保留每个调用。"""
from pathlib import Path
import argparse
import sqlite3
import pandas as pd

HERE=Path(__file__).resolve().parent


def module(name):
    if any(s in name for s in ['renderFoldedPixels','renderScalingPixels','renderTensorPixels']):return 'pixel_composite'
    if any(s in name for s in ['preprocessCUDA','preprocessGraph','preprocessFused']):return 'raster_preprocess'
    if any(s in name for s in ['DepthSort','depthKeys']):return 'depth_sort'
    if any(s in name for s in ['executePreprocessed','scheduleBits','executeBitTasks']):return 'footprint_members'
    if any(s in name for s in ['compactTileBits','compactPendingBits']):return 'member_compact'
    if any(s in name for s in ['clearGraphBits','clearPendingBits']):return 'member_clear'
    if any(s in name.lower() for s in ['compensat','publish']):return 'compensator_publish'
    if any(s in name.lower() for s in ['selector','logits']):return 'selector'
    if any(s in name.lower() for s in ['radix','select','threshold','winner']):return 'strict_selection'
    if any(s in name.lower() for s in ['candidate','frustum','anchor','view','frame','encode']):return 'candidate_view_prepare'
    if any(s in name for s in ['updateGraphInputs','updatePendingInputs']):return 'raster_input'
    return 'other'


def main():
    parser=argparse.ArgumentParser();parser.add_argument('name');args=parser.parse_args()
    out=HERE/'reports'/(args.name+'_ANALYSIS');out.mkdir(exist_ok=False)
    con=sqlite3.connect(HERE/'reports'/(args.name+'.sqlite'))
    ranges=con.execute("SELECT n.start,n.end,COALESCE(n.text,s.value) FROM NVTX_EVENTS n LEFT JOIN StringIds s ON s.id=n.textId WHERE (COALESCE(n.text,s.value) LIKE 'nine_render/%' OR COALESCE(n.text,s.value) LIKE 'nine_joint/%') AND n.end IS NOT NULL").fetchall()
    kernels=[];apis=[];calls=[]
    for index,(begin,end,label) in enumerate(ranges):
        parts=label.split('/')
        if parts[0]=='nine_render':
            _,scene,workload,condition,variant=parts;kind='frozen';scope='raster'
        else:
            _,scene,kind,condition,variant,scope=parts;workload='lod'
        base=dict(scene=scene,workload=workload,model=kind,condition=condition,variant=variant,scope=scope,call=index)
        values=con.execute('SELECT k.end-k.start,s.value,k.gridX,k.gridY,k.blockX,k.blockY,k.registersPerThread,k.staticSharedMemory FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.demangledName WHERE k.start>=? AND k.end<=?',(begin,end)).fetchall()
        for dt,name,gx,gy,bx,by,registers,shared in values:
            kernels.append(dict(**base,module=module(name),kernel=name,gpu_us=dt/1000,grid_x=gx,grid_y=gy,
                block_x=bx,block_y=by,registers=registers,shared_bytes=shared))
        runtime=con.execute('SELECT a.end-a.start,s.value FROM CUPTI_ACTIVITY_KIND_RUNTIME a JOIN StringIds s ON s.id=a.nameId WHERE a.start>=? AND a.end<=?',(begin,end)).fetchall()
        for dt,name in runtime:apis.append(dict(**base,api=name,host_us=dt/1000))
        calls.append(dict(**base,kernels=len(values),kernel_gpu_us=sum(v[0] for v in values)/1000,
            range_host_us=(end-begin)/1000,synchronize_calls=sum('Synchronize' in name for _,name in runtime),
            graph_launch_calls=sum('GraphLaunch' in name for _,name in runtime),
            memcpy_calls=sum('Memcpy' in name for _,name in runtime)))
    k=pd.DataFrame(kernels);a=pd.DataFrame(apis);c=pd.DataFrame(calls);assert len(c)>0
    for name,data in [('KERNELS',k),('APIS',a),('CALLS',c)]:data.to_csv(out/(name+'.csv.gz'),index=False,compression='gzip')
    group=['scene','workload','model','condition','variant','scope']
    summary=c.groupby(group).agg(kernel_gpu_us=('kernel_gpu_us','mean'),kernels=('kernels','mean'),
        range_host_us=('range_host_us','mean'),synchronize_calls=('synchronize_calls','mean'),
        graph_launch_calls=('graph_launch_calls','mean'),memcpy_calls=('memcpy_calls','mean'),calls=('call','size')).reset_index()
    summary.to_csv(out/'SUMMARY.csv',index=False)
    modules=k.groupby(group+['module']).gpu_us.sum().reset_index().merge(summary[group+['calls']],on=group)
    modules['gpu_us_per_call']=modules.gpu_us/modules.calls;modules.to_csv(out/'MODULES.csv',index=False)
    print('PROFILE_ANALYZED',args.name,len(c),len(k),len(a))


if __name__=='__main__':main()
