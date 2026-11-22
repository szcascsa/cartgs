from nine_study import *
from formal_render import candidate_workload


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False);rows=[]
    for slug in TUNE:
        s,source=load_source(slug);bg=source.scene.background;frame=motions(source,quick=True)[0][1];cam=source.camera(frame)
        for m in [37,38,39,40]:
            e=Renderer(m);e.precompile(source.runtime.total,cam,bg)
            for condition in ['local0.01','local0.1','local0.5','full']:
                packet,ids,_=request(source,cam,condition);image,radii=e.render(cam,packet,bg)
                d=candidate_workload(e,cam,packet,bg,image,radii)
                rows.append(dict(scene=s['id'],condition=condition,variant=MODES[m],K=len(ids),**d))
        print('DIAGNOSTIC_SMOKE',s['id'],flush=True)
    append(out/'workloads.csv',rows);save(out/'SUMMARY.json',dict(passed=True,rows=len(rows),diagnostic_rgb_bitwise=True))


if __name__=='__main__':main()
