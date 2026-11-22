"""诊断计数器验收：逐Gaussian贡献集合、图像与radii均与原接口相同。"""
from pathlib import Path
import importlib.util
import sys
import time

import count_worker as c


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def main():
    key=sys.argv[1]
    group=next(g for g in c.read(c.SOURCE)['groups'] if g['id']==key)
    batch=Path(group['trained_model'].split('/runs/')[0])
    sys.path[:0]=[str(batch/'source/python'),str(batch/'evaluation_runtime')]
    import numpy as np
    import torch
    from renderer import render_prepared
    from utils import MiniCam
    torch.set_num_threads(2)
    original=load('fr1xyz_production_counts',c.BINDING)
    fast=load('boolean_render_counts',c.ROOT/'boolean_counter_build/boolean_render_counts.so')
    selected=[r for r in group['ours'] if r['scope']=='full' or r['ratio'] in [.01,.1,.7]]
    full=next(r for r in selected if r['scope']=='full')
    plans=[(full,True)]+[(r,False) for r in selected]
    records=[]
    with torch.no_grad():
        for item,vanilla in plans:
            scope='vanilla' if vanilla else item['scope']
            output=c.ROOT/'boolean_validation'/key/f"{scope}_{item['ratio']:g}"
            output.mkdir(parents=True,exist_ok=True)
            namespace=c.setup(group,item,output,vanilla)
            for frame in [0,len(namespace['associations'])//2,len(namespace['associations'])-1]:
                tr,rg=namespace['associations'][frame]
                camera=MiniCam(namespace['width'],namespace['height'],namespace['fovx'],namespace['fovy'],
                               np.linalg.inv(namespace['poses'][tr]))
                if namespace['frustum_renderer'] is None:
                    data=namespace['render_data']
                else:data,ids,stats=namespace['frustum_renderer'].prepare(camera)
                old=c.counted(original,camera,data,namespace['background'])
                new=c.counted(fast,camera,data,namespace['background'])
                normal=render_prepared(camera,data,namespace['background'],return_radii=True)
                assert torch.equal(old[2].view(torch.int32),new[2].view(torch.int32))
                assert torch.equal(normal['render'].view(torch.int32),new[2].view(torch.int32))
                assert torch.equal(old[3],new[3]) and torch.equal(normal['radii'],new[3])
                assert torch.equal(old[5]>0,new[5]>0)
                timings=[]
                for binding in [original,fast]:
                    torch.cuda.synchronize();start=time.perf_counter()
                    for _ in range(5):c.counted(binding,camera,data,namespace['background'])
                    torch.cuda.synchronize();timings.append((time.perf_counter()-start)*1000/5)
                records.append(dict(scope=scope,ratio=item['ratio'],frame=frame,
                    gaussians=len(old[5]),contributing=int((old[5]>0).sum()),
                    bitwise_color_and_radii=True,per_gaussian_contribution_mask_equal=True,
                    original_ms=timings[0],boolean_ms=timings[1]))
                del old,new,normal,data
            del namespace
    receipt=c.ROOT/'boolean_validation'/key/'VERIFIED.json'
    c.write(receipt,dict(group=key,cases=len(records),records=records))
    print('BOOLEAN_COUNTER_VERIFIED',key,len(records),flush=True)


if __name__=='__main__':main()
