"""Nsight中仅标记热态准备与Rasterizer，保留相同输入。"""
from common import *


@torch.no_grad()
def main():
    s=scene();engines={'pytorch':reference_renderer(s)}
    engines.update({k:FixedBudgetRenderer(s.model,s.gaussians,s.config,0,'cuda_graph' if 'graph' in k else 'cuda')
                    for k in ['cuda','cuda_graph','cuda_graph_no_timing','cuda_no_view_fusion']})
    engines['cuda_no_view_fusion'].inference.fused_view=False
    cam=camera(s,396)
    for name,e in engines.items():
        for label,request in [('0.01',dict(ratio_budget=.01)),('0.7',dict(ratio_budget=.7)),('points2000',dict(budget=2000))]:
            kwargs=dict(request,view_conditioned=True)
            if name!='pytorch':kwargs['measure_time']=name!='cuda_graph_no_timing'
            for _ in range(12):
                data,_,_=e.prepare(cam,**kwargs);render_prepared(cam,data,s.background)
            torch.cuda.synchronize()
            for _ in range(10):
                with torch.cuda.nvtx.range(f'prepare/{name}/{label}'):
                    data,_,_=e.prepare(cam,**kwargs)
                with torch.cuda.nvtx.range(f'render/{name}/{label}'):
                    render_prepared(cam,data,s.background)
                torch.cuda.synchronize()


if __name__=='__main__':main()
