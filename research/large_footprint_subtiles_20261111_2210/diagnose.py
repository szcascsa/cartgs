"""在真实补偿输入上统计子块缩减比例，不以名义点数代替实际工作量。"""
from footprint_study import *


@torch.no_grad()
def main():
    a=argparse.ArgumentParser();a.add_argument('--output',required=True);args=a.parse_args()
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False);rows=[]
    run=Renderer(0)
    for slug in TUNE:
        s,source=load_source(slug);bg=source.scene.background
        for _,frame in motions(source,quick=True)[::3]:
            cam=source.camera(frame)
            for cond in ('local0.01','local0.05','local0.1','local0.3','full'):
                packet,ids,_=request(source,cam,cond)
                run.render(cam,packet,bg)
                for side in (4,8):
                    counters=run.native.subtile_statistics(side,True).cpu().tolist()
                    assert counters[4]==0,(s['id'],frame,cond,side,counters)
                    rows.append(dict(scene=s['id'],frame=frame,condition=cond,K=len(ids),side=side,
                                     tile_instances=counters[0],tested_regions=counters[1],kept_regions=counters[2],
                                     rejected_regions=counters[3],false_pixel_rejections=counters[4],
                                     estimated_pixel_tests=counters[5],overflow_tiles=counters[7],
                                     keep_fraction=counters[2]/max(1,counters[1])))
        print('DIAGNOSE_SCENE',s['id'],flush=True)
        del source,packet;gc.collect();torch.cuda.empty_cache()
    append(out/'workloads.csv',rows);save(out/'DONE.json',dict(rows=len(rows),passed=True))


if __name__=='__main__':main()
