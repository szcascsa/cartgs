"""当前帧log缓存对Tile成员、深度稳定顺序、radii及RGB的直接比较。"""
from student_study import *
from log_runtime import LogRenderer


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False)
    rows=[]
    for spec in student_specs(tune=True):
        source,student,_=load_case(spec);network=network_variants(source,student)['fp32'][0]
        bg=source.scene.background;runs={37:Renderer(37),38:Renderer(38),**{m:LogRenderer(m) for m in range(91,95)}}
        views=motions(source,quick=True)
        cam=source.camera(views[0][1])
        for engine in runs.values():engine.precompile(source.runtime.total,cam,bg)
        for condition in ['local0.01','local0.1','local0.3','local0.5','local0.7','fixed1','full']:
            for motion,frame in views:
                cam=source.camera(frame);packet,_,_=p.request(network,cam,condition)
                references={m:runs[m].render(cam,packet,bg) for m in [37,38]}
                snapshots={m:runs[m].native.snapshot() for m in [37,38]}
                for mode in range(91,95):
                    image,radii=runs[mode].render(cam,packet,bg);ref=37 if mode==91 else 38
                    state=runs[mode].native.snapshot()
                    row=dict(scene=spec['id'],condition=condition,frame=frame,mode=mode,
                        rgb_bitwise=bits_equal(image,references[ref][0]),radii_bitwise=bits_equal(radii,references[ref][1]),
                        max_rgb_error=float((image-references[ref][0]).abs().max()),
                        same_bits=bits_equal(state[0],snapshots[ref][0]),same_directory=bits_equal(state[1],snapshots[ref][1]),
                        same_counts=bits_equal(state[3],snapshots[ref][3]),same_sorted=bits_equal(state[4],snapshots[ref][4]),
                        same_depth_keys=bits_equal(state[5],snapshots[ref][5]))
                    rows.append(row)
                    assert all(row[key] for key in ['radii_bitwise','same_bits','same_directory','same_counts','same_sorted','same_depth_keys']),row
                    assert row['max_rgb_error']<=1e-6,row
        del source,student,network,runs;gc.collect();torch.cuda.empty_cache()
    append(out/'checks.csv',rows);save(out/'DONE.json',dict(passed=True,checks=len(rows)))
    print('LOG_PARITY_COMPLETE',len(rows),flush=True)


if __name__=='__main__':main()
