"""单实例冻结输入；本轮以已验证Deep为基线，输出仅在本轮。"""
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
BASE=HERE.parent/'cpu_native_deep_20261110_1222';ORIGINAL=HERE.parent/'cpu_native_aggressive_20261110_1115'
sys.path[:0]=[str(ORIGINAL),str(BASE)]
from cpu_study import *
from deep_runtime import CpuRenderer as Baseline,extension as baseline_extension
from small_runtime import CpuRenderer as Small,extension
HERE=Path(__file__).resolve().parent

def native(source,variant,threads=6,diagnose=False):
    choices=dict(baseline=({},Baseline),control=(dict(small_kernel=False,ordered_mode=0),Small),small=(dict(small_kernel=True,ordered_mode=0),Small),
        serial=(dict(small_kernel=False,ordered_mode=1),Small),parallel=(dict(small_kernel=False,ordered_mode=2),Small),
        adaptive=(dict(small_kernel=False,ordered_mode=3),Small),combined=({},Small))
    kwargs,cls=choices[variant];return cls(source['model'],source['pc'],source['reference'].noise,threads,diagnose=diagnose,**kwargs)

def request(engine,cam,condition,**kwargs):
    if condition=='full':
        flat,ids,_,_=engine.native.prepare(cam.world_view_transform.contiguous(),cam.full_proj_transform.contiguous(),cam.camera_center.contiguous(),cam.image_width,cam.image_height,cam.znear,3,1.,0,False,False,False,None,None)
        return engine.packet(flat,len(ids)),ids,dict(candidate_gaussians=len(ids))
    kw=dict(view_conditioned=not condition.startswith('global'))
    if condition.startswith('local'):kw['ratio_budget']=float(condition[5:])
    elif condition.startswith('global'):kw['global_ratio']=float(condition[6:])
    else:kw['budget']=int(condition[5:])
    return engine.prepare(cam,**kw,**kwargs)
