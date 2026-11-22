"""新增候选的合成边界与调优输入逐位验证，不触碰保留场景的性能。"""
from harness import *
from runtime import SparseRenderer


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    output=HERE/'reports'/args.output;output.mkdir(exist_ok=False)
    variants={'C0':SparseRenderer(),'C1_shared':SparseRenderer(render_variant=1),'C2_warp':SparseRenderer(render_variant=2),
        'C3_keys':SparseRenderer(fused_keys=True),'C13_shared_keys':SparseRenderer(render_variant=1,fused_keys=True),
        'C4_stride':SparseRenderer(strided_camera=True),'C134':SparseRenderer(render_variant=1,fused_keys=True,strided_camera=True),
        'C5_tasks':SparseRenderer(build_variant=1),'C135_tasks':SparseRenderer(render_variant=1,fused_keys=True,build_variant=1),
        'C1345':SparseRenderer(render_variant=1,fused_keys=True,strided_camera=True,build_variant=1),
        'C6_sort':SparseRenderer(sort_variant=1),
        'C13456':SparseRenderer(render_variant=1,fused_keys=True,strided_camera=True,build_variant=1,sort_variant=1),
        'C7_alpha2':SparseRenderer(render_variant=3), 'C8_alpha4':SparseRenderer(render_variant=4),
        'C9_sort512':SparseRenderer(sort_variant=2),
        'C13459':SparseRenderer(render_variant=1,fused_keys=True,strided_camera=True,build_variant=1,sort_variant=2),
        'Ccap64':SparseRenderer(render_variant=1,capacity=64,fused_keys=True,strided_camera=True,build_variant=1,sort_variant=2),
        'Ccap256':SparseRenderer(render_variant=1,capacity=256,fused_keys=True,strided_camera=True,build_variant=1,sort_variant=2),
        'C10_pair':SparseRenderer(render_variant=5),'C11_quad':SparseRenderer(render_variant=6),
        'CbestPair':SparseRenderer(render_variant=5,capacity=256,fused_keys=True,strided_camera=True,build_variant=1,sort_variant=2),
        'C12_metadata':SparseRenderer(render_variant=5,capacity=256,fused_keys=True,strided_camera=True,build_variant=1,sort_variant=2,footprint_metadata=True),
        'C13_tilefused':SparseRenderer(render_variant=7,capacity=256,fused_keys=True,strided_camera=True,build_variant=1,sort_variant=2)}
    B=baseline_B();rows=[];failed={};torch.manual_seed(4107)
    def compare(label,camera,packet,bg):
        if len(packet['inputs']['means3D']):
            ref=render_prepared(camera,packet,bg,return_radii=True)
        else:ref=dict(render=torch.zeros((3,camera.image_height,camera.image_width),device='cuda'),radii=torch.zeros(0,device='cuda',dtype=torch.int32))
        for name,engine in {'B':B,**variants}.items():
            if name in failed and name!='B':continue
            image,radii=engine.render(camera,packet,bg)
            passed=bits_equal(image,ref['render']) and bits_equal(radii,ref['radii'])
            row=dict(case=label,variant=name,K=len(radii),passed=passed)
            if not passed:
                row.update(max_abs=float((image-ref['render']).abs().max()),different_bytes=int((image.contiguous().view(torch.uint8)!=ref['render'].contiguous().view(torch.uint8)).sum()))
                if name not in failed:failed[name]=row
            rows.append(row)
    for W,H in [(97,65),(641,479)]:
        camera=MiniCam(W,H,1.2,1.,np.eye(4));bg=torch.tensor([.13,-0.,.76],device='cuda')
        for P in [0,1,31,32,33,127,128,129,257,511,512,513,1023,1024,1025,4097]:
            for style in ['random','overlap','outside']:
                xyz=torch.randn(P,3,device='cuda');xyz[:,2]=xyz[:,2].abs()+1
                if style=='overlap':xyz.zero_();xyz[:,2]=2.
                if style=='outside':xyz[:,2]=-.5
                quat=torch.randn(P,4,device='cuda');quat/=quat.norm(dim=1,keepdim=True).clamp_min(1e-6)
                scales=torch.rand(P,3,device='cuda')*.2+.002
                if P>31 and style=='random':
                    scales[:2]=torch.tensor([1.5,.001,.6],device='cuda');xyz[5,0]=float('nan');scales[6,0]=float('inf')
                packet=dict(sh_degree=3,inputs=dict(means3D=xyz,means2D=torch.zeros_like(xyz),
                    opacities=torch.full((P,1),.02,device='cuda') if style=='overlap' else torch.rand(P,1,device='cuda'),
                    scales=scales,rotations=quat,dc=torch.rand(P,1,3,device='cuda'),shs=torch.randn(P,15,3,device='cuda')*.1,
                    colors_precomp=None,cov3D_precomp=None))
                compare(f'{W}x{H}/{P}/{style}',camera,packet,bg)
    for item in SCENES:
        if item['split']!='tune':continue
        scene,source=load_scene(item['slug'])
        for motion,frame in views_for(source)[:2]+views_for(source)[-2:]:
            camera=source.camera(frame)
            for condition in ['local0.01','local0.05','local0.1','local0.03','global0.01','full']:
                packet,ids,stats=prepare(source,camera,frame,condition)
                compare(scene['id']+'/'+str(frame)+'/'+condition,camera,packet,source.scene.background)
        del source;torch.cuda.empty_cache();print('SCREEN_PARITY',item['slug'],len(rows),flush=True)
    result=dict(passed='C0' not in failed,rows=rows,failed_variants=failed,baseline_B_bitwise='B' not in failed,
        valid_candidates=[k for k in variants if k not in failed],comparisons=len(rows))
    write_json(output/'RESULT.json',result);print('SCREEN_PARITY_COMPLETE',len(rows),failed,flush=True)
    assert result['passed']


if __name__=='__main__':main()
