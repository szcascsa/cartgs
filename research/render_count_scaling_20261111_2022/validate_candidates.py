"""每个候选与原Renderer逐位比较，覆盖精度边界与真实低K大footprint。"""
from render_study import *
import argparse


@torch.no_grad()
def main():
    a=argparse.ArgumentParser()
    a.add_argument('--output',required=True)
    a.add_argument('--modes',default='1,2,3,4')
    a.add_argument('--quick',action='store_true')
    args=a.parse_args()
    out=HERE/'reports'/args.output
    out.mkdir(exist_ok=False)
    runs={MODES[k]:Renderer(k) for k in map(int,args.modes.split(','))}
    failed,rows={},[]

    def check(label,camera,packet,bg):
        ref=p.render_prepared(camera,packet,bg,return_radii=True)
        for name,run in runs.items():
            if name in failed:continue
            image,radii=run.render(camera,packet,bg)
            equal=bits_equal(image,ref['render']) and bits_equal(radii,ref['radii'])
            row=dict(case=label,variant=name,K=len(radii),rgb_bitwise=bits_equal(image,ref['render']),
                     radii_bitwise=bits_equal(radii,ref['radii']),max_error=float((image-ref['render']).abs().max()))
            rows.append(row)
            if not equal:failed[name]=row
        return ref

    torch.manual_seed(3407)
    sizes=[(97,65),(641,479)] if not args.quick else [(97,65)]
    for W,H in sizes:
        camera=p.MiniCam(W,H,1.2,1.,np.eye(4))
        bg=torch.tensor([.13,-0.,.76],device='cuda')
        for k in ([0,1,31,32,33,127,255,256,257,511,512,513,1025] if not args.quick else [0,1,33,257,513]):
            for style in ('random','overlap','equal','outside','no_alpha','barely_alpha','huge','thin','nonfinite'):
                check(f'{W}x{H}/{k}/{style}',camera,synthetic_packet(k,style),bg)
    # 针对保守Alpha拒绝分档边界、正负零和非有限opacity。
    camera=p.MiniCam(97,65,1.2,1.,np.eye(4))
    for opacity in (0.,-0.,1./255.,.0078125,.0625,.125,.5,1.,1.01,2.,float('nan'),float('inf')):
        packet=synthetic_packet(33,'overlap')
        packet['inputs']['opacities'].fill_(opacity)
        check('opacity/'+str(opacity),camera,packet,bg)
    for slug in TUNE:
        s,source=load_source(slug)
        for _,frame in motions(source,quick=True):
            camera=source.camera(frame)
            for cond in ('local0.01','local0.03','local0.05','local0.1','local0.3','local0.5','full'):
                packet,_,_=request(source,camera,cond)
                check(s['id']+'/'+cond+'/'+str(frame),camera,packet,source.scene.background)
        print('VALIDATION_SCENE',s['id'],len(rows),'failed',list(failed),flush=True)
        del source,packet
        gc.collect()
        torch.cuda.empty_cache()
    # 先构造oracle，再连续异步提交不同容量和保留历史输出。
    packets=[synthetic_packet(k,'huge') for k in (31,257,513,1)]
    refs=[p.render_prepared(camera,x,bg,return_radii=True) for x in packets]
    torch.cuda.synchronize()
    retained=[(name,run.render(camera,x,bg),ref) for x,ref in zip(packets,refs)
              for name,run in runs.items() if name not in failed]
    torch.cuda.synchronize()
    assert all(bits_equal(x[0],ref['render']) and bits_equal(x[1],ref['radii']) for _,x,ref in retained)
    append(out/'checks.csv',rows)
    save(out/'SUMMARY.json',dict(comparisons=len(rows),valid_modes=[k for k in runs if k not in failed],
                               failed=failed,binary_sha256=sha(extension().__file__),retained_outputs=len(retained)))
    print('VALIDATION_DONE',len(rows),list(failed),flush=True)


if __name__=='__main__':main()
