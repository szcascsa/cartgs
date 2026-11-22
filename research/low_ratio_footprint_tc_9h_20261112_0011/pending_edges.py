"""动态GPU K与容量解耦边界；验证本轮调度没有引入额外RGB/radii误差。"""
from nine_study import *
from pending_runtime import extension as pending_extension


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--sanitizer',action='store_true');args=parser.parse_args()
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False)
    mode=json.loads((HERE/'reports/ACCEPTED_RENDER.json').read_text())['mode']
    ref=Renderer(mode);engines={}
    for native in [False,True]:engines['native' if native else 'split']=pending_extension(native).PendingRasterizer(32,True,True,12,18+mode,64)
    bg=torch.tensor([.13,-0.,.76],device='cuda');rows=[];retained=[]
    torch.manual_seed(1187)
    def check(label,cam,k,capacity,style='random'):
        packet=synthetic_packet(k,style);x=packet['inputs']
        parts=[x[name].contiguous().flatten() for name in ['means3D','opacities','scales','rotations','dc','shs']]
        parts.append(x['means3D'].new_zeros(k*3));rest=x['shs'].shape[1]*3
        flat=x['means3D'].new_empty(capacity*(17+rest))
        if k:flat[:k*(17+rest)].copy_(torch.cat(parts))
        frame=torch.zeros(96,device='cuda',dtype=torch.uint8);control=frame.view(torch.int64)
        control[:5]=torch.tensor([k,k,k,k,k],device='cuda',dtype=torch.int64)
        image,radii=ref.render(cam,packet,bg)
        for name,engine in engines.items():
            result=engine.render_pending(bg,flat,frame,capacity,rest,cam.world_view_transform,cam.full_proj_transform,
                math.tan(cam.FoVx*.5),math.tan(cam.FoVy*.5),cam.image_height,cam.image_width,packet['sh_degree'],cam.camera_center)
            result=(result[0],result[1][:k])
            row=dict(case=label,variant=name,K=k,capacity=capacity,rgb_bitwise=bits_equal(result[0],image),
                radii_bitwise=bits_equal(result[1][:k],radii))
            rows.append(row);assert row['rgb_bitwise'] and row['radii_bitwise'],row
            retained.append((result[0],result[0].clone(),result[1],result[1].clone()))
        if len(retained)>12:
            for a,b,c,d in retained:assert bits_equal(a,b) and bits_equal(c,d),'输出被后续帧覆盖'
            retained.clear()
    cam=p.MiniCam(97,65,1.2,1.,np.eye(4))
    capacities=[1,31,32,33,127,128,129,255,256,257,511,512,513,1024,1025,4097]
    if args.sanitizer:capacities=[1,32,33,256,257,512,513,1025]
    for capacity in capacities:
        for k in sorted({0,1,min(31,capacity),min(32,capacity),min(33,capacity),min(511,capacity),min(512,capacity),capacity}):
            for style in ['random','huge','thin','overlap','equal','outside','barely_alpha']:
                check(f'capacity/{capacity}/{k}/{style}',cam,k,capacity,style)
    if not args.sanitizer:
        for width,height in [(1,1),(17,19),(641,479),(4097,7)]:
            camera=p.MiniCam(width,height,1.2,1.,np.eye(4))
            for k in [0,1,257,513]:check(f'resolution/{width}/{height}/{k}',camera,k,1025,'huge')
        ready=torch.cuda.Event();ready.record()
        for stream in [torch.cuda.Stream(),torch.cuda.Stream()]:
            saved,saved_ref=engines,ref
            with torch.cuda.stream(stream):
                stream.wait_event(ready)
                ref=Renderer(mode)
                engines={name:pending_extension(name=='native').PendingRasterizer(32,True,True,12,18+mode,64) for name in saved}
                for k in [513,1,0,33]:check(f'stream/{stream.cuda_stream}/{k}',cam,k,1025,'equal')
            stream.synchronize();engines,ref=saved,saved_ref
    for a,b,c,d in retained:assert bits_equal(a,b) and bits_equal(c,d),'输出独立性失败'
    append(out/'checks.csv',rows);save(out/'SUMMARY.json',dict(passed=True,comparisons=len(rows),sanitizer=args.sanitizer,
        same_renderer_bitwise=True,capacity_and_actual_K_independent=True,no_images=True))
    print('PENDING_DYNAMIC_BOUNDARIES_COMPLETE',len(rows),flush=True)


if __name__=='__main__':main()
