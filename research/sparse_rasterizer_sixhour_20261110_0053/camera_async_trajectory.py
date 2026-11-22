"""Q相机准备的全轨迹审计：矩阵逐位和图像对照，图像不落盘。"""
from harness import *
from final_benchmark import native_modules,CsvSink
from camera_async_preparation import AsyncGraphCameraPreparer as GraphCameraPreparer
from graph_runtime import GraphRenderer


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    output=HERE/'reports'/args.output;output.mkdir(exist_ok=False)
    config=json.loads((HERE/'reports/FROZEN_CANDIDATE.json').read_text());native_modules(config)
    sink=CsvSink(output/'checks.csv');frames=0;renders=0;started=time.time()
    for scene in SCENES:
        _,source=load_scene(scene['slug']);factory=GraphCameraPreparer(MiniCam);g=GraphRenderer(**config['graph'])
        g.precompile(source.runtime.total,source.camera(0),source.scene.background)
        for frame in range(len(source.scene.views)):
            a=source.camera(frame);q=factory.from_source(source,frame)
            row=dict(scene=scene['id'],frame=frame)
            for field in ['world_view_transform','projection_matrix','full_proj_transform','camera_center']:
                row[field]=bits_equal(getattr(a,field),getattr(q,field))
            assert all(row[field] for field in ['world_view_transform','projection_matrix','full_proj_transform','camera_center']),row
            sink.add(row);frames+=1
            # 渲染器已经全轨迹逐位验收；相机路径另外在每25帧及末帧核对完整选择/图像。
            if frame%25==0 or frame==len(source.scene.views)-1:
                for condition in ['full','local0.01','local0.05','local0.1']:
                    packet,ids,_=prepare(source,a,frame,condition);other,other_ids,_=prepare(source,q,frame,condition)
                    reference=render_prepared(a,packet,source.scene.background,return_radii=True)
                    image,radii=g.render(q,other,source.scene.background)
                    assert torch.equal(ids,other_ids) and bits_equal(image,reference['render']) and bits_equal(radii,reference['radii'])
                    renders+=1
            if (frame+1)%500==0:
                sink.flush();print('ASYNC_CAMERA_TRAJECTORY',scene['id'],frame+1,frames,flush=True)
        assert factory.rebuilds==1 and g.native.stats()[1:3]==[0,0]
        del source,g,factory,a,q,packet,other,reference,image;torch.cuda.empty_cache()
    sink.close();write_json(output/'SUMMARY.json',dict(passed=True,scenes=len(SCENES),full_trajectory_camera_frames=frames,
        camera_tensor_comparisons=4*frames,paired_selection_rgb_radii_inputs=renders,render_check_stride=25,
        elapsed_s=time.time()-started,no_render_images_saved=True))
    print('ASYNC_CAMERA_TRAJECTORY_COMPLETE',frames,renders,flush=True)


if __name__=='__main__':main()
