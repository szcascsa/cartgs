"""完整视锥算子边界：布局/转换/补零/输出；Graph微测只用于算子归因。"""
from prep_study import *
import gc


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False);rows=[];checks=[];mod=dp.deployment_extension()
    for slug in TUNE:
        _,source=load_scene(slug);b=source.scene.model
        # 用真实冻结锚框和不同视角；不缓存Mask或上帧结果。
        student,_,_=student_for(source,slug);run=make_engine(source,student,'C');state=run._base
        for fraction in [.12,.43,.81]:
            cam=source.camera(int(fraction*len(source.scene.views)));graphs={}
            args0=(state.anchors.centers,cam.full_proj_transform,state.anchors.offset_scales,run._reference.occupied,state.eligible_counts)
            reference=mod._preparation_frustum(*args0,0)
            for mode in [0,1,2]:
                for _ in range(3):value=mod._preparation_frustum(*args0,mode)
                torch.cuda.synchronize();graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    for _ in range(16):value=mod._preparation_frustum(*args0,mode)
                graphs[mode]=graph
                checks.append(dict(scene=slug,fraction=fraction,mode=mode,
                    mask_mismatches=int((value[0]!=reference[0]).sum()),distance_bitwise=bits_equal(value[1],reference[1]),max_error=float((value[1]-reference[1]).abs().max())))
            for repeat in range(40):
                for mode in rotate([0,1,2],repeat):
                    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    start.record();graphs[mode].replay();end.record();end.synchronize()
                    rows.append(dict(scene=slug,fraction=fraction,mode=mode,repeat=repeat,ms=start.elapsed_time(end)/16,anchors=len(state.anchors.centers)))
            del graphs
        write_csv(out/'samples.csv',rows);write_csv(out/'checks.csv',checks)
        del source,student,run,state;gc.collect();torch.cuda.empty_cache()
    save(out/'SUMMARY.json',dict(passed=True,samples=len(rows),checks=len(checks),images_saved=0))


if __name__=='__main__':main()
