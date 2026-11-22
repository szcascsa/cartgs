"""无需改CUDA，仅验证不同Tile短列表容量与512排序组合。"""
from harness import *
from runtime import SparseRenderer


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    output=HERE/'reports'/args.output;output.mkdir(exist_ok=False);rows=[]
    variants={'C13459':SparseRenderer(render_variant=1,fused_keys=True,strided_camera=True,build_variant=1,sort_variant=2),
        'Ccap64':SparseRenderer(render_variant=1,capacity=64,fused_keys=True,strided_camera=True,build_variant=1,sort_variant=2),
        'Ccap256':SparseRenderer(render_variant=1,capacity=256,fused_keys=True,strided_camera=True,build_variant=1,sort_variant=2)}
    torch.manual_seed(3207)
    camera=MiniCam(97,65,1.2,1.,np.eye(4));bg=torch.tensor([.13,-0.,.76],device='cuda')
    for P in [0,1,63,64,65,127,128,129,255,256,257,511,512,513,4097]:
        xyz=torch.zeros(P,3,device='cuda');xyz[:,2]=2.
        quat=torch.zeros(P,4,device='cuda');quat[:,0]=1.
        packet=dict(sh_degree=0,inputs=dict(means3D=xyz,means2D=torch.zeros_like(xyz),
            opacities=torch.full((P,1),.02,device='cuda'),scales=torch.full((P,3),.2,device='cuda'),
            rotations=quat,dc=None,shs=None,colors_precomp=torch.rand(P,3,device='cuda'),cov3D_precomp=None))
        ref=render_prepared(camera,packet,bg,return_radii=True) if P else dict(render=torch.zeros((3,65,97),device='cuda'),radii=torch.empty(0,dtype=torch.int,device='cuda'))
        for name,engine in variants.items():
            image,radii=engine.render(camera,packet,bg)
            assert bits_equal(image,ref['render']) and bits_equal(radii,ref['radii'])
            rows.append(dict(case='capacity/'+str(P),variant=name,passed=True))
    for item in SCENES:
        if item['split']!='tune':continue
        scene,source=load_scene(item['slug'])
        for motion,frame in views_for(source)[-2:]:
            camera=source.camera(frame)
            for condition in ['local0.01','local0.1','full']:
                packet,ids,stats=prepare(source,camera,frame,condition);ref=render_prepared(camera,packet,source.scene.background,return_radii=True)
                for name,engine in variants.items():
                    image,radii=engine.render(camera,packet,source.scene.background)
                    assert bits_equal(image,ref['render']) and bits_equal(radii,ref['radii'])
                    rows.append(dict(case=item['id']+'/'+str(frame)+'/'+condition,variant=name,passed=True))
        del source;torch.cuda.empty_cache()
    write_json(output/'RESULT.json',dict(passed=True,rows=rows,valid_candidates=list(variants),comparisons=len(rows)))
    print('CAPACITY_PARITY_COMPLETE',len(rows),flush=True)


if __name__=='__main__':main()
