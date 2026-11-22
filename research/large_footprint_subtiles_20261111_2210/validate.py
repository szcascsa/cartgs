"""逐位对照原Renderer，覆盖子块边界、细长椭圆和原足迹不变。"""
from footprint_study import *
import importlib.util


@torch.no_grad()
def main():
    a=argparse.ArgumentParser()
    a.add_argument('--output',required=True)
    a.add_argument('--modes',default='0,1,2,3,4')
    a.add_argument('--quick',action='store_true')
    args=a.parse_args()
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False)
    verify_baseline();torch.manual_seed(3407)
    runs={MODES[k]:Renderer(k) for k in map(int,args.modes.split(','))}
    failures,rows={},[]
    def check(label,cam,packet,bg):
        ref=p.render_prepared(cam,packet,bg,return_radii=True)
        for name,run in runs.items():
            if name in failures:continue
            row=dict(case=label,variant=name,K=len(ref['radii']),**compare(run,cam,packet,bg,ref))
            rows.append(row)
            if not row['rgb_bitwise'] or not row['radii_bitwise']:failures[name]=row
    sizes=[(97,65),(641,479)] if not args.quick else [(97,65)]
    bg=torch.tensor([.13,-0.,.76],device='cuda')
    for W,H in sizes:
        cam=p.MiniCam(W,H,1.2,1.,np.eye(4))
        for k in ([0,1,31,32,33,127,255,256,257,511,512,513,1025] if not args.quick else [0,1,33,257,513]):
            for style in ('random','overlap','equal','outside','no_alpha','barely_alpha','huge','thin','nonfinite'):
                check(f'{W}x{H}/{k}/{style}',cam,synthetic_packet(k,style),bg)
    cam=p.MiniCam(97,65,1.2,1.,np.eye(4))
    for opacity in (0.,-0.,1./255.,.00390625,.0078125,.015625,.0625,.125,.5,1.,1.01,2.,float('inf'),float('nan')):
        packet=synthetic_packet(33,'overlap');packet['inputs']['opacities'].fill_(opacity)
        check('opacity/'+str(opacity),cam,packet,bg)
    # 高度扁平、旋转和Scale极值，让椭圆跨越4/8像素子块边界。
    for seed in range(16 if not args.quick else 4):
        torch.manual_seed(104729+seed)
        packet=synthetic_packet(257,'random')
        packet['inputs']['scales'].mul_(packet['inputs']['scales'].new_tensor([.001,1.,100.]))
        check('anisotropic/'+str(seed),cam,packet,bg)
    for slug in TUNE:
        s,source=load_source(slug)
        for _,frame in motions(source,quick=True):
            cam=source.camera(frame)
            for cond in ('local0.01','local0.03','local0.05','local0.1','local0.3','local0.5','full'):
                packet,_,_=request(source,cam,cond)
                check(s['id']+'/'+str(frame)+'/'+cond,cam,packet,source.scene.background)
        print('SUBTILE_VALIDATION_SCENE',s['id'],len(rows),'failed',list(failures),flush=True)
        del source,packet;gc.collect();torch.cuda.empty_cache()
    append(out/'checks.csv',rows)
    save(out/'SUMMARY.json',dict(comparisons=len(rows),valid=[n for n in runs if n not in failures],failed=failures,
                               binary_sha256=sha(extension().__file__)))
    print('SUBTILE_VALIDATION_DONE',len(rows),list(failures),flush=True)


if __name__=='__main__':main()
