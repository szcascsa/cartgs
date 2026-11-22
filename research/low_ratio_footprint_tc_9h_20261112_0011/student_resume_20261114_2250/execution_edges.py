"""近似渲染边界：radii严格相同、有限性/独立输出/Stream/容量；误差如实记录。"""
from student_study import *
import os
from encoded_sign_runtime import EncodedSignRenderer
from update_clear_runtime import UpdateClearRenderer
from cooperative_sort_runtime import CooperativeSortRenderer
from cooperative_sort_runtime_v2 import CooperativeSortRenderer as CooperativeSortV2
from empty_tiles_runtime import EmptyTilesRenderer
MODES={182:"execution"}
from final_benchmark import native_modules


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--modes',required=True)
    parser.add_argument('--sanitizer',action='store_true');parser.add_argument('--candidate',choices=['clear','sort','sort2','empty'],required=True);args=parser.parse_args()
    Renderer={'clear':UpdateClearRenderer,'sort':CooperativeSortRenderer,'sort2':CooperativeSortV2,'empty':EmptyTilesRenderer}[args.candidate]
    baseline=EncodedSignRenderer(182)
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False)
    native_modules(json.loads((previous.PRIOR/'reports/FROZEN_CANDIDATE.json').read_text()))
    torch.manual_seed(58127);engines={MODES[m]:Renderer(m) for m in map(int,args.modes.split(','))}
    bg=torch.tensor([.13,-0.,.76],device='cuda');rows=[]
    def check(label,cam,packet):
        if os.environ.get('EDGE_TRACE')=='1':print('EDGE',label,flush=True)
        image_ref,radii_ref=baseline.render(cam,packet,bg)
        ref=dict(render=image_ref,radii=radii_ref)
        for name,engine in engines.items():
            image,radii=engine.render(cam,packet,bg)
            assert bits_equal(radii,ref['radii']),(label,name,'radii')
            assert torch.equal(torch.isfinite(image),torch.isfinite(ref['render'])),(label,name,'finite')
            values=(image-ref['render']).abs();values=values[torch.isfinite(values)]
            row=dict(case=label,variant=name,K=len(radii),radii_bitwise=True,rgb_bitwise=bits_equal(image,ref['render']),
                max_abs=float(values.max()) if len(values) else 0.)
            assert row['rgb_bitwise'],row
            if len(radii):
                actual=engine.native.snapshot();expected=baseline.native.snapshot()
                for j in [0,1,3]:assert bits_equal(actual[j],expected[j]),(label,name,'snapshot',j)
                for j in [4,5]:assert bits_equal(actual[j][:len(radii)],expected[j][:len(radii)]),(label,name,'stable depth',j)
            rows.append(row)
    for W,H in ([(97,65),(641,479)] if not args.sanitizer else [(97,65)]):
        cam=p.MiniCam(W,H,1.2,1.,np.eye(4))
        for k in [0,1,15,16,17,31,32,33,127,128,129,255,256,257,511,512,513,1023,1024,1025]:
            for style in ['random','overlap','equal','outside','no_alpha','barely_alpha','huge','thin','nonfinite']:
                check(f'{W}x{H}/{k}/{style}',cam,synthetic_packet(k,style))
    cam=p.MiniCam(97,65,1.2,1.,np.eye(4))
    for k in [1,129,513]:
        for color,cov in [(True,False),(False,True),(True,True)]:check(f'formats/{k}/{color}/{cov}',cam,synthetic_packet(k,color=color,covariance=cov))
    for opacity in [0.,-0.,1/255,.00390625,.5,1.,1.01,2.,float('inf'),float('nan')]:
        packet=synthetic_packet(33,'overlap');packet['inputs']['opacities'].fill_(opacity);check('opacity/'+str(opacity),cam,packet)
    cases=[]
    for index,k in enumerate([1,257,513,31,1025,4097,33,129]):
        camera=p.MiniCam(97+index,65+index%2,1.2,1.,np.eye(4));packet=synthetic_packet(k,'thin' if index%2 else 'huge')
        for name,e in engines.items():
            image,radii=e.render(camera,packet,bg);cases.append((name,camera,packet,image.clone(),radii.clone()))
    torch.cuda.synchronize();pending=[]
    for name,camera,packet,image,radii in cases:pending.append((engines[name].render(camera,packet,bg),image,radii))
    torch.cuda.synchronize()
    for result,image,radii in pending:assert bits_equal(result[0],image) and bits_equal(result[1],radii)
    packet=synthetic_packet(257,'huge');ready=torch.cuda.Event();ready.record()
    for stream in [torch.cuda.Stream(),torch.cuda.Stream()]:
        refs={name:e.render(cam,packet,bg) for name,e in engines.items()};ready.record()
        with torch.cuda.stream(stream):
            stream.wait_event(ready);local={MODES[m]:Renderer(m) for m in map(int,args.modes.split(','))}
            values={name:e.render(cam,packet,bg) for name,e in local.items()}
        stream.synchronize()
        for name,result in values.items():
            assert bits_equal(result[0],refs[name][0]) and bits_equal(result[1],refs[name][1])
            try:local[name].render(cam,packet,bg)
            except RuntimeError as error:assert 'per device/stream' in str(error)
            else:raise AssertionError('unsafe stream reuse')
    for tiles in [255,256,257]:
        camera=p.MiniCam(tiles*16,7,1.2,1.,np.eye(4));packet=synthetic_packet(1,'huge')
        packet['inputs']['means3D'].zero_();packet['inputs']['means3D'][:,2]=2.
        check('footprint_tiles/'+str(tiles),camera,packet)
    for engine in engines.values():
        try:engine.native.reserve(1<<20,2048,2048,32,bg)
        except RuntimeError as error:assert 'capacity limit' in str(error)
        else:raise AssertionError('unbounded workspace accepted')
    append(out/'checks.csv',rows)
    save(out/'SUMMARY.json',dict(passed=True,comparisons=len(rows),modes=args.modes,
        same_candidate_output_lifecycle=True,stream_dependency_checks=True,capacity_rejected=True,no_images=True))
    print('APPROX_BOUNDARIES_COMPLETE',len(rows),flush=True)


if __name__=='__main__':main()
