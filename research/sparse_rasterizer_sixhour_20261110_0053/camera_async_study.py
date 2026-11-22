"""独立探索相机小矩阵执行图；保留inverse原错误检查，不改冻结F候选。"""
from harness import *
from final_benchmark import rotate,native_modules
from camera_preparation import CameraPreparer
from graph_runtime import GraphRenderer


from camera_async_preparation import AsyncGraphCameraPreparer as GraphCameraPreparer
from camera_graph_preparation import GraphCameraPreparer as SynchronousGraphCameraPreparer


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    output=HERE/'reports'/args.output;output.mkdir(exist_ok=False);checks=[];samples=[]
    factory=GraphCameraPreparer(MiniCam)
    def compare(label,original,new):
        for name in ['world_view_transform','projection_matrix','full_proj_transform','camera_center']:
            row=dict(case=label,field=name,bitwise=bits_equal(getattr(original,name),getattr(new,name)))
            checks.append(row);assert row['bitwise'],row
    for i in range(64):
        pose=np.eye(4);pose[:3,:3]=Rotation.from_euler('xyz',[.03*i,.02*i,-.06*i]).as_matrix();pose[:3,3]=[i*.03,-i*.02,i*.01]
        view=np.linalg.inv(pose);compare('pose/'+str(i),MiniCam(641,479,1.2,1.,view),factory.create(641,479,1.2,1.,view))
    retained=[]
    for i in range(24):
        pose=np.eye(4);pose[0,3]=.031*i;retained.append((MiniCam(97,65,1.2,1.,pose),factory.create(97,65,1.2,1.,pose)))
    for a,b in retained:compare('retained_outputs',a,b)
    singular=np.eye(4);singular[1]=singular[0]
    for engine in [lambda:MiniCam(97,65,1.2,1.,singular),lambda:factory.create(97,65,1.2,1.,singular)]:
        try:engine()
        except torch.linalg.LinAlgError:checks.append(dict(case='singular_matrix',field='error',bitwise=True))
        else:raise AssertionError('Singular matrix error check removed')
    config=json.loads((HERE/'reports/FROZEN_CANDIDATE.json').read_text());native_modules(config)
    for scene in SCENES:
        if scene['split']!='tune':continue
        _,source=load_scene(scene['slug']);fixed=SynchronousGraphCameraPreparer(MiniCam);graph=GraphCameraPreparer(MiniCam);g=GraphRenderer(**config['graph'])
        views=views_for(source,'tune');camera=source.camera(views[0][1]);g.precompile(source.runtime.total,camera,source.scene.background)
        for _,frame in views:
            compare(scene['id']+'/'+str(frame),source.camera(frame),graph.from_source(source,frame))
        for condition in ['local0.01','local0.05','local0.1','full']:
            for repeat in range(6):
                for index in range(24):
                    frame=views[index%len(views)][1];outputs=[]
                    for name in rotate(['Q','async_upload'],index+repeat):
                        torch.cuda.synchronize();tick=time.perf_counter_ns()
                        camera=(fixed if name=='Q' else graph).from_source(source,frame)
                        packet,ids,_=prepare(source,camera,frame,condition);image=g.render(camera,packet,source.scene.background)[0]
                        torch.cuda.synchronize();samples.append(dict(scene=scene['id'],condition=condition,frame=frame,variant=name,
                            repeat=repeat,iteration=index,wall_ms=(time.perf_counter_ns()-tick)/1e6,K=len(ids)))
                        outputs.append((ids,image))
                    assert torch.equal(outputs[0][0],outputs[1][0]) and bits_equal(outputs[0][1],outputs[1][1])
            write_csv(output/'samples.csv',samples)
        del source,g,camera,packet;torch.cuda.empty_cache();print('ASYNC_CAMERA_STUDY',scene['id'],len(samples),flush=True)
    write_csv(output/'correctness.csv',checks);write_json(output/'SUMMARY.json',dict(passed=True,checks=len(checks),samples=len(samples),
        inverse_check_preserved=True,render_path_unchanged=True,experimental_only=True))
    print('ASYNC_CAMERA_STUDY_COMPLETE',len(samples),flush=True)


if __name__=='__main__':main()
