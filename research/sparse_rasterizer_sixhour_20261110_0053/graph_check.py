"""容量图的输入更新、padding、缓存淘汰、输出独立性与Stream边界。"""
from harness import *
from graph_runtime import GraphRenderer


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--granularity',type=int,default=0)
    parser.add_argument('--graph-input-update',action='store_true');args=parser.parse_args()
    output=HERE/'reports'/args.output;output.mkdir(exist_ok=False)
    engine=GraphRenderer(plan_granularity=args.granularity,graph_input_update=args.graph_input_update);rows=[];torch.manual_seed(5543)
    def packet(P,precomputed=False,covariance=False):
        xyz=torch.randn(P,3,device='cuda');xyz[:,2]=xyz[:,2].abs()+1
        quat=torch.randn(P,4,device='cuda');quat/=quat.norm(dim=1,keepdim=True).clamp_min(1e-6)
        return dict(sh_degree=3,inputs=dict(means3D=xyz,means2D=torch.zeros_like(xyz),
            opacities=torch.rand(P,1,device='cuda'),scales=None if covariance else torch.rand(P,3,device='cuda')*.2+.001,
            rotations=None if covariance else quat,dc=torch.rand(P,1,3,device='cuda'),shs=None if precomputed else torch.randn(P,15,3,device='cuda')*.1,
            colors_precomp=torch.rand(P,3,device='cuda') if precomputed else None,
            cov3D_precomp=torch.tensor([.01,0.,0.,.02,0.,.03],device='cuda').repeat(P,1) if covariance else None))
    for P in [0,1,31,32,33,127,128,129,255,256,257,511,512,513,1023,1024,1025,4097,10000,33]:
        for style in ['random','equal','outside','nonfinite']:
            cam=MiniCam(97,65,1.2,1.,np.eye(4));bg=torch.tensor([.13,-0.,.76],device='cuda')
            p=packet(P,precomputed=P%2==0,covariance=P in [33,1025])
            if style=='equal':p['inputs']['means3D'][:,2]=2.
            if style=='outside':p['inputs']['means3D'][:,2]=-.5
            if style=='nonfinite' and P>32:
                p['inputs']['means3D'][5,0]=float('nan');p['inputs']['opacities'][6,0]=float('inf')
                if p['inputs']['scales'] is not None:p['inputs']['scales'][7,0]=float('inf')
            ref=render_prepared(cam,p,bg,return_radii=True)
            image,radii=engine.render(cam,p,bg)
            ok=bits_equal(image,ref['render']) and bits_equal(radii,ref['radii'])
            row=dict(case=f'{P}/{style}',passed=ok)
            if not ok:
                row.update(max_abs=float((image-ref['render']).abs().max()),radii_equal=bits_equal(radii,ref['radii']))
                write_json(output/'FAILURE.json',row);raise AssertionError(row)
            rows.append(row)
    # 多组输出保持存活，连续调用后再统一检查，包含池重建和计划失效。
    retained=[]
    for j in range(18):
        cam=MiniCam(97+j,65,1.2,1.,np.eye(4));p=packet(31+j*5,precomputed=True)
        bg=torch.tensor([j*.01,.2,.7],device='cuda');ref=render_prepared(cam,p,bg,return_radii=True)
        image,radii=engine.render(cam,p,bg);retained.append((image,radii,ref))
    torch.cuda.synchronize()
    for image,radii,ref in retained:assert bits_equal(image,ref['render']) and bits_equal(radii,ref['radii'])
    rows.append(dict(case='18_outputs_pool_resize',passed=True))
    # 同一分辨率预留一次完整池，再让65种容量复用，验证64计划上限及真实淘汰。
    eviction=GraphRenderer(plan_granularity=512,graph_input_update=args.graph_input_update);tiny=MiniCam(17,19,1.2,1.,np.eye(4))
    eviction.reserve(65*512,tiny,bg)
    previous=[]
    for j in range(65):
        p=packet(j*512+1,precomputed=True);ref=render_prepared(tiny,p,bg,return_radii=True)
        image,radii=eviction.render(tiny,p,bg)
        assert bits_equal(image,ref['render']) and bits_equal(radii,ref['radii'])
        if j in [0,31]:previous.append((image,image.clone()))
    assert eviction.native.stats()[3]==64 and eviction.native.stats()[2]>=1
    assert all(bits_equal(*pair) for pair in previous)
    rows.append(dict(case='65_capacities_shared_pool_64_plan_limit',passed=True))
    cam=MiniCam(111,73,1.2,1.,np.eye(4));p=packet(513);bg=torch.tensor([.13,.2,.76],device='cuda')
    ref=render_prepared(cam,p,bg,return_radii=True);torch.cuda.synchronize()
    stream=torch.cuda.Stream();other=GraphRenderer(plan_granularity=args.granularity,graph_input_update=args.graph_input_update)
    with torch.cuda.stream(stream):image,radii=other.render(cam,p,bg)
    stream.synchronize();assert bits_equal(image,ref['render']) and bits_equal(radii,ref['radii'])
    try:other.render(cam,p,bg)
    except RuntimeError as error:assert 'one graph renderer per device/stream' in str(error)
    else:raise AssertionError('Cross-stream plan reuse unexpectedly accepted')
    rows.append(dict(case='nondefault_stream_and_rejection',passed=True))
    for item in SCENES:
        if item['split']!='tune':continue
        scene,source=load_scene(item['slug'])
        for motion,frame in views_for(source):
            cam=source.camera(frame)
            for condition in ['local0.01','local0.05','local0.1','local0.03','full']:
                p,ids,stats=prepare(source,cam,frame,condition);ref=render_prepared(cam,p,source.scene.background,return_radii=True)
                image,radii=engine.render(cam,p,source.scene.background)
                row=dict(case=item['id']+'/'+str(frame)+'/'+condition,passed=bits_equal(image,ref['render']) and bits_equal(radii,ref['radii']))
                if not row['passed']:
                    row.update(max_abs=float((image-ref['render']).abs().max()),radii_equal=bits_equal(radii,ref['radii']))
                    write_json(output/'FAILURE.json',row);raise AssertionError(row)
                rows.append(row)
        del source;torch.cuda.empty_cache();print('GRAPH_PARITY',item['slug'],len(rows),flush=True)
    write_json(output/'SUMMARY.json',dict(passed=True,checks=len(rows),rows=rows,graph_stats=engine.native.stats(),
        fresh_input_values_every_request=True,output_independence=True,plan_capacity_limit=64))
    print('GRAPH_PARITY_COMPLETE',len(rows),flush=True)


if __name__=='__main__':main()
