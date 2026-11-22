"""单实例显存和冷启动；避免将14个对照同时驻留的峰值误当实际部署成本。"""
from joint_final_study import *


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--group',required=True,type=int);args=parser.parse_args()
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False);rows=[]
    config=json.loads((HERE/'reports/FROZEN_STUDENT.json').read_text())['config']
    for spec in student_specs()[args.group*2:args.group*2+2]:
        source,student,_=load_case(spec);factory=NativeCameraPreparer(p.MiniCam);bg=source.scene.background
        for name in ['G_cpp_pose','I_final']:
            torch.cuda.synchronize();gc.collect();torch.cuda.empty_cache();before=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
            start=time.perf_counter_ns()
            engine=PoseRenderer(source,student,38) if name=='G_cpp_pose' else FinalStudentRenderer(source,student,182,**config)
            cam=factory.from_source(source,motions(source,quick=True)[0][1]);engine.precompile(cam,bg)
            torch.cuda.synchronize();startup=(time.perf_counter_ns()-start)*1e-6
            for condition in ['local0.01','local0.1','local0.3','local0.5','full']:
                for _,frame in motions(source,quick=True):
                    result=engine.render_from_source(source,frame,bg,condition,factory);del result
            torch.cuda.synchronize()
            rows.append(dict(scene=spec['id'],variant=name,N=source.runtime.total,startup_ms=startup,
                live_increment_bytes=torch.cuda.memory_allocated()-before,peak_increment_bytes=torch.cuda.max_memory_allocated()-before,
                raster_workspace_bytes=engine.native.workspace_bytes(),projection_bytes=engine.network._pipeline._preparation_bytes(),
                tensor_bytes=engine.network._pipeline._tensor_bytes(),selector_prefix_bytes=engine.network._pipeline._prefix_bytes()))
            del engine,cam;gc.collect();torch.cuda.empty_cache()
        del source,student;gc.collect();torch.cuda.empty_cache()
    append(out/'memory.csv',rows);save(out/'DONE.json',dict(passed=True,rows=len(rows),single_instance=True));print('SINGLE_MEMORY_PASS',len(rows),flush=True)


if __name__=='__main__':main()
