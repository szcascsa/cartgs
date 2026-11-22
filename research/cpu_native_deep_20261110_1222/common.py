"""复用旧冻结输入与oracle，所有输出和构建指向本轮目录。"""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
OLD=HERE.parent/'cpu_native_aggressive_20261110_1115'
sys.path.insert(0,str(OLD))
from cpu_study import *
from cpu_runtime import CpuRenderer as Baseline,extension as old_extension
from deep_runtime import CpuRenderer as Deep,extension
HERE=Path(__file__).resolve().parent

def native(source,variant,threads=6,diagnose=False):
    options=dict(baseline=({},Baseline),schedule=(dict(kernel_mode=0),Deep),kernel=(dict(fused_schedule=False,native_wrapper=False),Deep),
        combined=({},Deep),control=(dict(fused_schedule=False,kernel_mode=0,native_wrapper=False),Deep))
    kwargs,cls=options[variant];return cls(source['model'],source['pc'],source['reference'].noise,threads,diagnose=diagnose,**kwargs)

def request(engine,cam,condition,**kwargs):
    if condition=='full':
        flat,ids,stats,_=engine.native.prepare(cam.world_view_transform.contiguous(),cam.full_proj_transform.contiguous(),cam.camera_center.contiguous(),
            cam.image_width,cam.image_height,cam.znear,3,1.,0,False,False,False,None,None)
        return engine.packet(flat,len(ids)),ids,dict(candidate_gaussians=len(ids))
    kw=dict(view_conditioned=not condition.startswith('global'))
    if condition.startswith('local'):kw['ratio_budget']=float(condition[5:])
    elif condition.startswith('global'):kw['global_ratio']=float(condition[6:])
    else:kw['budget']=int(condition[5:])
    return engine.prepare(cam,**kw,**kwargs)
