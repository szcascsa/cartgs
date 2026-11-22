from prep_study import *
import gc


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False);rows=[]
    for spec in [s for s in SCENES if s['dataset_kind']=='Replica']:
        _,source=load_scene(spec['slug']);student,_,_=student_for(source,spec['slug'])
        run=make_engine(source,student)
        for repeat in range(5):
            run._pipeline._preparation_options(0,0,0);torch.cuda.synchronize();tick=time.perf_counter_ns()
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            start.record();run._pipeline._preparation_options(1,1,1);end.record();end.synchronize()
            rows.append(dict(scene=spec['id'],repeat=repeat,N=run.total,anchors=run._base.anchors.centers.size(0),
                levels=len(run.knots),host_ms=(time.perf_counter_ns()-tick)/1e6,gpu_event_ms=start.elapsed_time(end),
                projection_bytes=run._pipeline._preparation_bytes(),tensor_bytes=run._pipeline._tensor_bytes()))
        write_csv(out/'samples.csv',rows);del source,student,run;gc.collect();torch.cuda.empty_cache()
    save(out/'SUMMARY.json',dict(passed=True,samples=len(rows),scope='仅构建新增静态投影及GPU工作区，不含checkpoint读取或整个模型加载；finite检查包含在内'))


if __name__=='__main__':main()
