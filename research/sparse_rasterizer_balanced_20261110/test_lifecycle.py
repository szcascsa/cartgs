"""冻结推理的Stream、Workspace复用、输出独立和额外输入组合验证。"""
from validate import *


@torch.no_grad()
def main():
    checks=0;torch.manual_seed(102)
    cam=MiniCam(111,73,1.2,1.0,np.eye(4));bg=torch.tensor([.1,.4,.7],device='cuda')
    def packet(P,colors=False,degree=3):
        xyz=torch.randn(P,3,device='cuda');xyz[:,2]=2.
        quat=torch.randn(P,4,device='cuda');quat/=quat.norm(dim=1,keepdim=True)
        return dict(sh_degree=degree,inputs=dict(means3D=xyz,means2D=torch.zeros_like(xyz),
            opacities=torch.rand(P,1,device='cuda'),scales=torch.rand(P,3,device='cuda')*.2,
            rotations=quat,dc=torch.randn(P,1,3,device='cuda'),shs=None if colors else torch.randn(P,15,3,device='cuda')*.1,
            colors_precomp=torch.rand(P,3,device='cuda') if colors else None,cov3D_precomp=None))
    for mode in [0,1,2,3,4]:
        for short in [False,True] if mode<2 else [False]:
            engine=SparseRenderer(mode,short)
            keep=None
            for P in [257,33,2048,1,513,31]:
                p=packet(P,colors=P in [33,513],degree=0 if P==1 else 3)
                ref=render_prepared(cam,p,bg,return_radii=True)
                out=engine.render(cam,p,bg)
                assert bits_equal(out[0],ref['render']) and bits_equal(out[1],ref['radii'])
                if keep is not None:assert bits_equal(*keep)
                keep=(out[0],out[0].clone());checks+=1
            # 协方差预计算路径，含极细/各向异性点。
            p=packet(97);p['inputs']['cov3D_precomp']=torch.tensor([.01,0.,0.,.02,0.,.03],device='cuda').repeat(97,1)
            p['inputs']['scales']=None;p['inputs']['rotations']=None
            ref=render_prepared(cam,p,bg,return_radii=True);out=engine.render(cam,p,bg)
            assert bits_equal(out[0],ref['render']) and bits_equal(out[1],ref['radii']);checks+=1
            p=packet(257);ref=render_prepared(cam,p,bg,return_radii=True);torch.cuda.synchronize()
            stream=torch.cuda.Stream();other=SparseRenderer(mode,short)
            with torch.cuda.stream(stream):out=other.render(cam,p,bg)
            stream.synchronize()
            assert bits_equal(out[0],ref['render']) and bits_equal(out[1],ref['radii']);checks+=1
            try:other.render(cam,p,bg)
            except RuntimeError as error:assert 'one renderer per device/stream' in str(error)
            else:raise AssertionError('Cross-stream Workspace reuse was accepted')
            checks+=1
    result=dict(passed=True,checks=checks,output_independence=True,dynamic_shapes=True,
        nondefault_stream=True,cross_stream_workspace_rejected=True,precomputed_covariance=True,
        precomputed_rgb=True,sh_degrees=[0,3])
    (HERE/'reports/LIFECYCLE_VALIDATION.json').write_text(json.dumps(result,indent=2))
    print('LIFECYCLE_VALIDATION',json.dumps(result),flush=True)


if __name__=='__main__':main()
