"""独立验收动态相机执行图：非刚体矩阵、异常、Stream和输出生命周期。"""
from harness import *
from camera_async_preparation import AsyncGraphCameraPreparer as GraphCameraPreparer


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    output=HERE/'reports'/args.output;output.mkdir(exist_ok=False);rows=[];factory=GraphCameraPreparer(MiniCam)
    generator=np.random.default_rng(887219)
    def compare(label,a,b):
        for name in ['world_view_transform','projection_matrix','full_proj_transform','camera_center']:
            row=dict(case=label,field=name,bitwise=bits_equal(getattr(a,name),getattr(b,name)))
            rows.append(row);assert row['bitwise'],row
    retained=[]
    for index in range(192):
        W,H=[(97,65),(641,479),(1200,680)][(index//16)%3];fx=1.+(index//32)*.13;fy=.8+(index//24)*.07
        view=np.eye(4)
        if index%3==0:view+=generator.normal(0,.2,(4,4))
        else:
            view[:3,:3]=Rotation.from_euler('xyz',generator.normal(0,2.,3)).as_matrix()
            view[:3,3]=generator.normal(0,100.,3)
        if index%7==0:view[1]*=1e-4
        a=MiniCam(W,H,fx,fy,view);b=factory.create(W,H,fx,fy,view)
        compare('dynamic_matrix/'+str(index),a,b);retained.append((a,b))
    for index,(a,b) in enumerate(retained):compare('retained_output/'+str(index),a,b)
    for index in range(8):
        singular=np.eye(4);singular[index%4]=0
        messages=[]
        for fn in [lambda:MiniCam(97,65,1.2,1.,singular),lambda:factory.create(97,65,1.2,1.,singular)]:
            try:fn()
            except torch.linalg.LinAlgError as error:messages.append(str(error))
            else:raise AssertionError('Singular inverse check missing')
        # 原算子报错类型、失败对角位置与消息一致。
        assert messages[0]==messages[1],messages
        rows.append(dict(case='singular_error/'+str(index),field='error_message',bitwise=True))
    # 同一个factory跨Stream时销毁旧计划并重建；每个输出独立，不依赖下一帧。
    for index in range(12):
        stream=torch.cuda.Stream();view=np.eye(4);view[0,3]=index*.02
        with torch.cuda.stream(stream):a=MiniCam(97,65,1.2,1.,view);b=factory.create(97,65,1.2,1.,view)
        stream.synchronize();compare('stream_rebuild/'+str(index),a,b)
    write_csv(output/'checks.csv',rows);write_json(output/'SUMMARY.json',dict(passed=True,comparisons=len(rows),
        singular_errors_identical=True,same_FP32_BMM_and_inverse=True,stream_and_intrinsics_invalidation=True,
        output_values_independent=True,images_saved=False))
    print('ASYNC_CAMERA_EDGES_COMPLETE',len(rows),flush=True)


if __name__=='__main__':main()
