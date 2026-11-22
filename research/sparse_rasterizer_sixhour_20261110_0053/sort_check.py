"""整数稳定排序候选：直接核对深度键/身份次序，并与原入口逐位对照图像。"""
from harness import *
from runtime import SparseRenderer


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--preprocess-fusion',action='store_true');parser.add_argument('--chunked',action='store_true')
    parser.add_argument('--gaussian-block',action='store_true');args=parser.parse_args()
    output=HERE/'reports'/args.output;output.mkdir(exist_ok=False);rows=[]
    common=dict(render_variant=5,capacity=256,fused_keys=True,strided_camera=True,build_variant=1)
    variants={name:SparseRenderer(**common,sort_variant=variant) for name,variant in
        [('CbestPair',2),('C14_radix6',3),('C15_radix7',4),('C16_blockmerge',5)]}
    if args.preprocess_fusion:
        variants={'CbestPair':variants['CbestPair'],'C17_preprocess_sort':SparseRenderer(**common,sort_variant=2,preprocess_sort=True)}
    if args.chunked:
        variants={'CbestPair':variants['CbestPair']}
        for name,render in [('C18_cache128',8),('C19_cache64',9)]:
            variants[name]=SparseRenderer(**{**common,'render_variant':render},sort_variant=2)
    if args.gaussian_block:
        variants={'CbestPair':variants['CbestPair'],'C20_gaussianBlock':SparseRenderer(**{**common,'build_variant':2},sort_variant=2)}
    def check(label,cam,packet,bg):
        ref=render_prepared(cam,packet,bg,return_radii=True)
        intermediate=None
        for name,engine in variants.items():
            out=engine.render(cam,packet,bg,True)
            if not (bits_equal(out[0],ref['render']) and bits_equal(out[1],ref['radii'])):
                failure=dict(case=label,variant=name,max_abs=float((out[0]-ref['render']).abs().max()),
                    radii_equal=bits_equal(out[1],ref['radii']),different_bytes=int((out[0].contiguous().view(torch.uint8)!=ref['render'].contiguous().view(torch.uint8)).sum()),
                    reference_radii=ref['radii'].cpu().tolist(),candidate_radii=out[1].cpu().tolist())
                write_json(output/'FAILURE.json',failure);raise AssertionError(failure)
            if len(out[1]):
                if intermediate is None:intermediate=[out[j][out[1]>0].clone() for j in [2,3,4,13]]
                else:assert all(bits_equal(out[j][out[1]>0],value) for j,value in zip([2,3,4,13],intermediate)),(label,name,'preprocess values')
                keys=out[4].cpu().numpy().view(np.uint32)
                keys=np.where(out[1].cpu().numpy()>0,keys,np.uint32(0xffffffff)).astype(np.uint32)
                expected=np.argsort(keys,kind='stable')
                assert np.array_equal(expected,out[7].cpu().numpy()),(label,name,'stable identities')
                assert np.array_equal(keys[expected],out[14].cpu().numpy().view(np.uint32)),(label,name,'keys')
            rows.append(dict(case=label,variant=name,passed=True))
    torch.manual_seed(6621);cam=MiniCam(97,65,1.2,1.,np.eye(4));bg=torch.tensor([.13,-0.,.76],device='cuda')
    for P in [0,1,31,32,33,127,128,129,255,256,257,511,512,513,1023,1024,1025,4097,65535,65536,65537]:
        for style in ['random','equal','outside']+(['overlap'] if args.chunked or args.gaussian_block else []):
            xyz=torch.randn(P,3,device='cuda');xyz[:,2]=xyz[:,2].abs()+1
            if style=='equal':xyz[:,2]=2.
            if style=='outside':xyz[:,2]=-.5
            if style=='overlap':xyz.zero_();xyz[:,2]=2.
            quat=torch.randn(P,4,device='cuda');quat/=quat.norm(dim=1,keepdim=True).clamp_min(1e-6)
            scales=torch.rand(P,3,device='cuda')*.2+.002
            if style=='overlap':scales.fill_(.2)
            if P>31 and style=='random':xyz[5,0]=float('nan');scales[6,0]=float('inf')
            packet=dict(sh_degree=3,inputs=dict(means3D=xyz,means2D=torch.zeros_like(xyz),opacities=torch.rand(P,1,device='cuda'),
                scales=scales,rotations=quat,dc=torch.rand(P,1,3,device='cuda'),shs=torch.randn(P,15,3,device='cuda')*.1,
                colors_precomp=None,cov3D_precomp=None))
            check(str(P)+'/'+style,cam,packet,bg)
    for scene in SCENES:
        if scene['split']!='tune':continue
        _,source=load_scene(scene['slug'])
        for motion,frame in views_for(source)[:2]+views_for(source)[-2:]:
            cam=source.camera(frame)
            for condition in ['local0.01','local0.05','local0.1','local0.03','global0.01','full']:
                packet,ids,stats=prepare(source,cam,frame,condition)
                check(scene['id']+'/'+str(frame)+'/'+condition,cam,packet,source.scene.background)
        del source;torch.cuda.empty_cache();print('SORT_PARITY',scene['id'],len(rows),flush=True)
    write_json(output/'RESULT.json',dict(passed=True,valid_candidates=list(variants),comparisons=len(rows),rows=rows))
    print('SORT_PARITY_COMPLETE',len(rows),flush=True)


if __name__=='__main__':main()
