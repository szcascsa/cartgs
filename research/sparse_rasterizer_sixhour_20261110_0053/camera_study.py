"""原位姿构建与固定内参预计算的逐位审计和调优集配对测量。"""
from harness import *
from final_benchmark import native_modules,rotate
from camera_preparation import CameraPreparer
from graph_runtime import GraphRenderer


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    output=HERE/'reports'/args.output;output.mkdir(exist_ok=False)
    config=json.loads((HERE/'reports/FROZEN_CANDIDATE.json').read_text());native_modules(config)
    factory=CameraPreparer(MiniCam);rows=[];samples=[]
    def compare(label,a,b):
        for attr in ['world_view_transform','projection_matrix','full_proj_transform','camera_center']:
            row=dict(case=label,field=attr,bitwise=bits_equal(getattr(a,attr),getattr(b,attr)))
            rows.append(row);assert row['bitwise'],row
    for i in range(48):
        W,H=[(97,65),(641,479),(1200,680)][i%3];fx=1.+(i%5)*.13;fy=.8+(i%7)*.07
        pose=np.eye(4);pose[:3,:3]=Rotation.from_euler('xyz',[.03*i,.1*i,-.03*i]).as_matrix();pose[:3,3]=[i*.01,-.04*i,1.+.03*i]
        view=np.linalg.inv(pose)
        compare(f'intrinsics_and_pose/{i}',MiniCam(W,H,fx,fy,view),factory.create(W,H,fx,fy,view))
    # 两个Stream各自重建固定输入，旧相机保持独立引用。
    retained=[]
    for stream in [torch.cuda.Stream(),torch.cuda.Stream()]:
        with torch.cuda.stream(stream):
            a=MiniCam(97,65,1.2,1.,np.eye(4));b=factory.create(97,65,1.2,1.,np.eye(4));retained.append((stream,a,b))
    for stream,a,b in retained:stream.synchronize();compare('different_stream',a,b)
    for scene in SCENES:
        if scene['split']!='tune':continue
        _,source=load_scene(scene['slug']);g=GraphRenderer(**config['graph']);factory=CameraPreparer(MiniCam)
        views=views_for(source,'tune');camera=source.camera(views[0][1]);g.precompile(source.runtime.total,camera,source.scene.background)
        for motion,frame in views:
            a=source.camera(frame);b=factory.from_source(source,frame);compare(scene['id']+'/'+str(frame),a,b)
            for condition in ['local0.01','local0.1','full']:
                packet,ids,stats=prepare(source,a,frame,condition);other,other_ids,_=prepare(source,b,frame,condition)
                image=g.render(a,packet,source.scene.background)[0];out=g.render(b,other,source.scene.background)[0]
                assert torch.equal(ids,other_ids) and bits_equal(image,out),(scene['id'],frame,condition)
        for condition in ['local0.01','local0.05','local0.1','full']:
            for repeat in range(6):
                for index in range(24):
                    frame=views[index%len(views)][1]
                    for variant in rotate(['G','F'],index+repeat):
                        torch.cuda.synchronize();tick=time.perf_counter_ns()
                        camera=source.camera(frame) if variant=='G' else factory.from_source(source,frame)
                        packet,ids,_=prepare(source,camera,frame,condition);image=g.render(camera,packet,source.scene.background)[0]
                        torch.cuda.synchronize();samples.append(dict(scene=scene['id'],condition=condition,frame=frame,
                            repeat=repeat,variant=variant,K=len(ids),wall_ms=(time.perf_counter_ns()-tick)/1e6))
            write_csv(output/'samples.csv',samples)
        del source,g,camera,packet;torch.cuda.empty_cache();print('CAMERA_STUDY',scene['id'],len(samples),flush=True)
    write_csv(output/'correctness.csv',rows);write_json(output/'SUMMARY.json',dict(passed=True,comparisons=len(rows),samples=len(samples),
        tune_only=True,cache='每实例仅一个64B投影矩阵；内参/分辨率/device/Stream/dtype变化失效；不缓存位姿、可见性、选点、Feature或图像'))
    print('CAMERA_STUDY_COMPLETE',len(samples),flush=True)


if __name__=='__main__':main()
