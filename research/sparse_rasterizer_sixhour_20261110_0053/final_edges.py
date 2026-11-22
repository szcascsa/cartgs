"""最终冻结入口边界：容量、空贡献、稳定排序、独立输出和Stream依赖。"""
from harness import *
from runtime import SparseRenderer
from graph_runtime import GraphRenderer


@torch.no_grad()
def packet(P,style='random',color=False,covariance=False):
    xyz=torch.randn(P,3,device='cuda');xyz[:,2]=xyz[:,2].abs()+1
    q=torch.randn(P,4,device='cuda');q/=q.norm(dim=1,keepdim=True).clamp_min(1e-6)
    scales=torch.rand(P,3,device='cuda')*.2+.002;opacity=torch.rand(P,1,device='cuda')
    if style in ['overlap','no_alpha','barely_alpha']:
        xyz.zero_();xyz[:,0]=.15;xyz[:,1]=.18;xyz[:,2]=2.
        scales.fill_(1e-5 if style!='overlap' else .2)
        opacity.fill_(.01 if style=='overlap' else (1e-8 if style=='no_alpha' else 1/255))
    if style=='equal':xyz[:,2]=2.
    if style=='outside':xyz[:,2]=-.5
    if style=='huge':scales.fill_(10.)
    if style=='thin':scales[:]=torch.tensor([1.5,.000001,.00001],device='cuda')
    if style=='nonfinite' and P>=3:xyz[0,0]=float('nan');scales[1,0]=float('inf');opacity[2,0]=float('nan')
    return dict(sh_degree=3,inputs=dict(means3D=xyz,means2D=torch.zeros_like(xyz),opacities=opacity,
        scales=None if covariance else scales,rotations=None if covariance else q,
        dc=torch.rand(P,1,3,device='cuda'),shs=None if color else torch.randn(P,15,3,device='cuda')*.1,
        colors_precomp=torch.rand(P,3,device='cuda') if color else None,
        cov3D_precomp=torch.tensor([.01,0.,0.,.02,0.,.03],device='cuda').repeat(P,1) if covariance else None))


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--sanitizer',action='store_true')
    parser.add_argument('--ordered',action='store_true');parser.add_argument('--device-clear',action='store_true');args=parser.parse_args();output=HERE/'reports'/args.output;output.mkdir(exist_ok=False);rows=[]
    c=SparseRenderer(render_variant=5,capacity=256,fused_keys=True,strided_camera=True,build_variant=1,sort_variant=2,ordered_coverage=args.ordered)
    g=GraphRenderer(plan_granularity=512,max_plans=512,ordered_coverage=args.ordered,device_clear=args.device_clear);torch.manual_seed(80191)
    sizes=[0,1,31,32,33,63,64,65,127,128,129,255,256,257,511,512,513,1023,1024,1025,4097]
    if args.sanitizer:sizes=[0,1,255,256,257,511,512,513,1025]
    def compare(label,cam,p,bg):
        ref=render_prepared(cam,p,bg,return_radii=True)
        for name,engine in [('C',c),('G',g)]:
            out=engine.render(cam,p,bg)
            row=dict(case=label,variant=name,K=len(out[1]),rgb_bitwise=bits_equal(out[0],ref['render']),radii_bitwise=bits_equal(out[1],ref['radii']))
            rows.append(row)
            if not row['rgb_bitwise'] or not row['radii_bitwise']:
                row['max_abs']=float((out[0]-ref['render']).abs().max());write_json(output/'FAILURE.json',row);raise AssertionError(row)
        return ref
    for W,H in [(97,65),(641,479)] if not args.sanitizer else [(97,65)]:
        cam=MiniCam(W,H,1.2,1.,np.eye(4));bg=torch.tensor([.13,-0.,.76],device='cuda')
        for P in sizes:
            for style in ['random','overlap','equal','outside','no_alpha','barely_alpha','huge','thin','nonfinite']:
                p=packet(P,style);compare(f'{W}x{H}/{P}/{style}',cam,p,bg)
    cam=MiniCam(111,73,1.2,1.,np.eye(4));bg=torch.tensor([.13,-0.,.76],device='cuda')
    for P in [1,129,513]:
        for color,cov in [(True,False),(False,True),(True,True)]:compare(f'precomputed/{P}/{color}/{cov}',cam,packet(P,color=color,covariance=cov),bg)
    # 非连续相机/背景/属性；原入口会contiguous，图入口保留相同值。
    p=packet(129);strided=dict(p,inputs={})
    for name,value in p['inputs'].items():
        if isinstance(value,torch.Tensor):
            buffer=value.new_empty((*value.shape,2));buffer[...,0]=value;strided['inputs'][name]=buffer[...,0]
        else:strided['inputs'][name]=value
    background=bg.new_empty(6);background[::2]=bg
    compare('strided_attributes_background',cam,strided,background[::2])
    # 多次调用不在中间同步，所有输出保持存活；覆盖池扩容/分辨率失效。
    retained=[]
    for i in range(24):
        camera=MiniCam(97+i,65,1.2,1.,np.eye(4));p=packet(31+i*25,color=True)
        ref=render_prepared(camera,p,bg,return_radii=True)
        for name,engine in [('C',c),('G',g)]:retained.append((name,engine.render(camera,p,bg),ref))
    torch.cuda.synchronize()
    for name,out,ref in retained:
        assert bits_equal(out[0],ref['render']) and bits_equal(out[1],ref['radii']);rows.append(dict(case='retained_async_outputs',variant=name,K=len(out[1]),rgb_bitwise=True,radii_bitwise=True))
    # 一个实例绑定一个Stream；两个不同Stream各自实例，通过事件显式保证输入生产依赖。
    first=torch.cuda.Stream();second=torch.cuda.Stream();ready=torch.cuda.Event();ready.record()
    p=packet(257);ready.record();ref=render_prepared(cam,p,bg,return_radii=True);torch.cuda.synchronize()
    pairs=[]
    for stream in [first,second]:
        with torch.cuda.stream(stream):
            stream.wait_event(ready)
            a=SparseRenderer(render_variant=5,capacity=256,fused_keys=True,strided_camera=True,build_variant=1,sort_variant=2,ordered_coverage=args.ordered)
            b=GraphRenderer(plan_granularity=512,ordered_coverage=args.ordered,device_clear=args.device_clear);pairs.append((stream,a,b,a.render(cam,p,bg),b.render(cam,p,bg)))
    for stream,a,b,one,two in pairs:
        stream.synchronize()
        for out in [one,two]:assert bits_equal(out[0],ref['render']) and bits_equal(out[1],ref['radii'])
        for engine in [a,b]:
            try:engine.render(cam,p,bg)
            except RuntimeError as error:assert 'per device/stream' in str(error)
            else:raise AssertionError('Unsafe cross-stream workspace accepted')
    rows.append(dict(case='two_streams_explicit_dependencies_and_reuse_rejection',variant='C+G',K=257,rgb_bitwise=True,radii_bitwise=True))
    # 容量上限必须拒绝，不制造OOM，也不创建无界Graph对象。
    safe=GraphRenderer(plan_granularity=512,max_plans=512)
    try:safe.reserve(1<<20,MiniCam(2048,2048,1.2,1.,np.eye(4)),bg)
    except RuntimeError as error:assert 'capacity limit' in str(error)
    else:raise AssertionError('Oversized workspace accepted')
    rows.append(dict(case='workspace_capacity_rejection',variant='G',K=1<<20,rgb_bitwise=True,radii_bitwise=True))
    write_json(output/'SUMMARY.json',dict(passed=True,comparisons=len(rows),rows=rows,sanitizer=args.sanitizer,
        exact_signed_zero_semantics=True,images_saved=False))
    print('FINAL_EDGES_COMPLETE',len(rows),flush=True)


if __name__=='__main__':main()
