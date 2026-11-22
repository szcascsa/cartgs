"""真实checkpoint公开入口，不依赖私有开关组装；与冻结D逐值对照。"""
from prep_study import *
from native_camera import NativeCameraPreparer


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    out=HERE/'reports'/args.output;out.mkdir(exist_ok=False)
    spec,source=load_scene(TUNE[0]);student,path,_=student_for(source,TUNE[0])
    public=dp.DeploymentRenderer.from_checkpoint(spec['checkpoint'],student_checkpoint=path,
        tensor_selector=2,tensor_compensator=2,tensor_warps=4,tensor_compensator_warps=2,
        preparation_frustum=1,preparation_projection=1,device_schedule=1)
    reference=make_engine(source,student,frustum=1,projection=1,device=1)
    camera=NativeCameraPreparer(MiniCam).from_source(source,int(.43*len(source.scene.views)))
    checks=[]
    for ratio in [.01,.03,.1,.2,.3,.4,.5,.7]:
        a,ia,sa=public.prepare(camera,ratio_budget=ratio);b,ib,sb=reference.prepare(camera,ratio_budget=ratio)
        assert torch.equal(ia,ib) and all(bits_equal(v,b['inputs'][k]) for k,v in a['inputs'].items() if isinstance(v,torch.Tensor))
        checks.append(dict(ratio=ratio,K=len(ia),bitwise=True))
    write_csv(out/'checks.csv',checks);save(out/'SUMMARY.json',dict(passed=True,checks=len(checks),public_checkpoint_entry=True,
        tensor_config=public.tensor_config,preparation_config=public.preparation_config,
        binary_sha256=sha(public._ext.__file__),tf32_matmul_enabled=torch.backends.cuda.matmul.allow_tf32))
    print('PUBLIC_PREPARATION_API_PASS',len(checks),flush=True)


if __name__=='__main__':main()
