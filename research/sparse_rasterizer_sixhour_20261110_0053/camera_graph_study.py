"""独立探索相机小矩阵执行图；保留inverse原错误检查，不改冻结F候选。"""
from harness import *
from final_benchmark import rotate,native_modules
from camera_preparation import CameraPreparer
from graph_runtime import GraphRenderer


class GraphCameraPreparer(CameraPreparer):
    def __init__(self,camera_class):
        super().__init__(camera_class);self.plan=None;self.plan_projection=None

    @torch.no_grad()
    def create(self,width,height,fovx,fovy,world_view_transform):
        camera=self.camera_class.__new__(self.camera_class)
        camera.image_width=width;camera.image_height=height;camera.FoVx=fovx;camera.FoVy=fovy
        camera.znear=.01;camera.zfar=100.
        camera.world_view_transform=torch.tensor(world_view_transform,dtype=torch.float32).transpose(0,1).cuda()
        stream=torch.cuda.current_stream();key=(width,height,fovx,fovy,.01,100.,stream.device_index,stream.cuda_stream,torch.get_default_dtype())
        if key!=self.key:
            self.projection=self.projection_builder(znear=.01,zfar=100.,fovX=fovx,fovY=fovy).transpose(0,1).cuda()
            self.key=key;self.rebuilds+=1;self.plan=None
        camera.projection_matrix=self.projection
        if self.plan is None:
            self.view=torch.empty_strided((4,4),camera.world_view_transform.stride(),device='cuda',dtype=torch.float32)
            self.view.copy_(camera.world_view_transform)
            # 首次cuBLAS/solver初始化在捕获外，使用相同FP32算子。
            self.view.unsqueeze(0).bmm(self.projection.unsqueeze(0))
            torch.linalg.inv_ex(self.view,check_errors=False)
            self.plan=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.plan):
                self.full=self.view.unsqueeze(0).bmm(self.projection.unsqueeze(0)).squeeze(0)
                self.inverse,self.info=torch.linalg.inv_ex(self.view,check_errors=False)
        self.view.copy_(camera.world_view_transform);self.plan.replay()
        # 与Tensor.inverse一致：每帧同步检查求逆状态，不延迟或删除检查。
        torch.ops.aten._linalg_check_errors(self.info,'linalg.inv',is_matrix=True)
        camera.full_proj_transform=self.full.clone()
        camera.camera_center=self.inverse[3,:3].clone()
        return camera


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
        _,source=load_scene(scene['slug']);fixed=CameraPreparer(MiniCam);graph=GraphCameraPreparer(MiniCam);g=GraphRenderer(**config['graph'])
        views=views_for(source,'tune');camera=source.camera(views[0][1]);g.precompile(source.runtime.total,camera,source.scene.background)
        for _,frame in views:
            compare(scene['id']+'/'+str(frame),source.camera(frame),graph.from_source(source,frame))
        for condition in ['local0.01','local0.05','local0.1','full']:
            for repeat in range(6):
                for index in range(24):
                    frame=views[index%len(views)][1];outputs=[]
                    for name in rotate(['F','camera_graph'],index+repeat):
                        torch.cuda.synchronize();tick=time.perf_counter_ns()
                        camera=(fixed if name=='F' else graph).from_source(source,frame)
                        packet,ids,_=prepare(source,camera,frame,condition);image=g.render(camera,packet,source.scene.background)[0]
                        torch.cuda.synchronize();samples.append(dict(scene=scene['id'],condition=condition,frame=frame,variant=name,
                            repeat=repeat,iteration=index,wall_ms=(time.perf_counter_ns()-tick)/1e6,K=len(ids)))
                        outputs.append((ids,image))
                    assert torch.equal(outputs[0][0],outputs[1][0]) and bits_equal(outputs[0][1],outputs[1][1])
            write_csv(output/'samples.csv',samples)
        del source,g,camera,packet;torch.cuda.empty_cache();print('CAMERA_GRAPH_STUDY',scene['id'],len(samples),flush=True)
    write_csv(output/'correctness.csv',checks);write_json(output/'SUMMARY.json',dict(passed=True,checks=len(checks),samples=len(samples),
        inverse_check_preserved=True,render_path_unchanged=True,experimental_only=True))
    print('CAMERA_GRAPH_STUDY_COMPLETE',len(samples),flush=True)


if __name__=='__main__':main()
