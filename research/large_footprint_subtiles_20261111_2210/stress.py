"""低K、大面积重叠压力输入；与真实LoD结果单独报告。"""
from footprint_study import *


@torch.no_grad()
def main():
    a=argparse.ArgumentParser();a.add_argument('--output',required=True);a.add_argument('--modes',required=True)
    args=a.parse_args();out=HERE/'reports'/args.output;out.mkdir(exist_ok=False)
    rows=[];torch.manual_seed(3407)
    for W,H in [(641,479),(1280,720)]:
        cam=p.MiniCam(W,H,1.2,1.,np.eye(4));bg=torch.tensor([.13,.2,.76],device='cuda')
        runs={MODES[k]:Renderer(k) for k in map(int,args.modes.split(','))};runs['previous_F2']=BaselineRenderer()
        for e in runs.values():e.precompile(512,cam,bg)
        for k in (1,32,128,512):
            for scale in (.03,.3,3.):
                for opacity in (.02,.5):
                    packet=synthetic_packet(k,'overlap')
                    xyz=packet['inputs']['means3D'];xyz.zero_();xyz[:,2]=2.+torch.arange(k,device='cuda')*.0001
                    packet['inputs']['scales'].fill_(scale);packet['inputs']['opacities'].fill_(opacity)
                    ref=p.render_prepared(cam,packet,bg,return_radii=True)
                    for n,e in runs.items():
                        c=compare(e,cam,packet,bg,ref);assert c['rgb_bitwise'] and c['radii_bitwise'],(W,H,k,scale,opacity,n,c)
                        for _ in range(3):e.render(cam,packet,bg)
                    for rep in range(20):
                        for n in order(list(runs),rep,reverse=(rep//len(runs))%2==1):
                            torch.cuda.synchronize();tick=time.perf_counter_ns()
                            runs[n].render(cam,packet,bg);torch.cuda.synchronize()
                            rows.append(dict(width=W,height=H,K=k,scale=scale,opacity=opacity,variant=n,repeat=rep,
                                             ms=(time.perf_counter_ns()-tick)*1e-6))
        print('FOOTPRINT_STRESS',W,H,flush=True)
    append(out/'samples.csv',rows);save(out/'DONE.json',dict(passed=True,samples=len(rows),modes=args.modes))


if __name__=='__main__':main()
