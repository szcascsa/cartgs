"""背景内容原位变化、异步提交和旧输出独立性。"""
from render_study import *
from background_runtime import Renderer


@torch.no_grad()
def main():
    renderer=Renderer()
    cam=p.MiniCam(97,65,1.2,1.,np.eye(4))
    packet=synthetic_packet(2,'overlap')
    packet['inputs']['means3D'].zero_();packet['inputs']['means3D'][:,2]=2.
    packet['inputs']['scales'].fill_(.001);packet['inputs']['opacities'].zero_()
    bg=torch.zeros(3,device='cuda')
    values=[torch.tensor(x,device='cuda') for x in ([0.,0.,0.],[-0.,0.,0.],[.3,-.2,.1],
            [1.e-40,0.,0.],[float('nan'),.1,0.],[float('inf'),0.,0.],[0.,0.,0.])]
    references=[p.render_prepared(cam,packet,x,return_radii=True) for x in values]
    torch.cuda.synchronize()
    results=[]
    for x,reference in zip(values,references):
        bg.copy_(x)
        results.append((renderer.render(cam,packet,bg),reference))
    torch.cuda.synchronize()
    assert all(bits_equal(x[0],r['render']) and bits_equal(x[1],r['radii']) for x,r in results)
    save(HERE/'background_stage/reports/INPLACE_BACKGROUND.json',
         dict(passed=True,comparisons=len(results),same_background_pointer=True,no_intermediate_sync=True))
    print('INPLACE_BACKGROUND_PASS',len(results))


if __name__=='__main__':main()
