"""Compute Sanitizer的小型真实CUDA入口覆盖。"""
from validate import *


@torch.no_grad()
def main():
    torch.manual_seed(70)
    cam=MiniCam(97,65,1.2,1.0,np.eye(4));bg=torch.tensor([.1,.2,.3],device='cuda')
    engines=[SparseRenderer(0),SparseRenderer(1),SparseRenderer(1,True),SparseRenderer(2),SparseRenderer(3),SparseRenderer(4)]
    for P in [1,33,257]:
        xyz=torch.randn(P,3,device='cuda');xyz[:,2]=2.
        if P>33:xyz[:2,2]=-.5;xyz[3,0]=float('nan')
        q=torch.randn(P,4,device='cuda');q/=q.norm(dim=1,keepdim=True)
        packet=dict(sh_degree=3,inputs=dict(means3D=xyz,means2D=torch.zeros_like(xyz),
            opacities=torch.rand(P,1,device='cuda'),scales=torch.rand(P,3,device='cuda')*.3,
            rotations=q,dc=torch.randn(P,1,3,device='cuda'),shs=torch.randn(P,15,3,device='cuda')*.1,
            colors_precomp=None,cov3D_precomp=None))
        ref=render_prepared(cam,packet,bg,return_radii=True)
        for engine in engines:
            result=engine.render(cam,packet,bg)
            assert bits_equal(result[0],ref['render']) and bits_equal(result[1],ref['radii'])
    torch.cuda.synchronize();print('SANITIZER_SMOKE_PASS',flush=True)
    from test_boundaries import main as boundaries
    boundaries()


if __name__=='__main__':main()
