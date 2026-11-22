"""读取真实冻结权重，CPU FP32 推理微测；不调用CUDA、不训练、不改生产。"""
from pathlib import Path
import argparse,csv,hashlib,json,math,os,sys,time,types,statistics,subprocess
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
FROZEN=Path('/home/prmi/zwj/experiments/compensator_distillation_replica_all_20261108')
PYTHON=FROZEN/'source/python'
sys.path[:0]=[str(PYTHON),'/home/prmi/zwj/experiments/global_view_joint_lod_20261105/evaluation/python/diff-gaussian-rasterization']
import numpy as np
import torch
from selector import load_selector
from gaussian_model import GaussianModel
from compensation_student import load_student,decode_student
from anchor_budget import FixedBudgetRenderer
from budget_viewer import read_config
from utils import getProjectionMatrix,focal2fov


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2))
def csv_save(path,rows):
    if not rows:return
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def camera(record):
    pose=np.eye(4);pose[:3,:3]=np.asarray(record['rotation']);pose[:3,3]=record['position']
    w,h=record['width'],record['height'];fx,fy=focal2fov(record['fx'],w),focal2fov(record['fy'],h)
    view=torch.tensor(np.linalg.inv(pose),dtype=torch.float32).transpose(0,1)
    projection=getProjectionMatrix(.01,100.,fx,fy).transpose(0,1)
    return types.SimpleNamespace(image_width=w,image_height=h,FoVx=fx,FoVy=fy,znear=.01,zfar=100.,
        world_view_transform=view,projection_matrix=projection,full_proj_transform=view.unsqueeze(0).bmm(projection.unsqueeze(0)).squeeze(0),
        camera_center=view.inverse()[3,:3])


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--quick',action='store_true')
    args=parser.parse_args();out=HERE/'reports'/args.output;out.mkdir(exist_ok=False)
    torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.manual_seed(586123)
    scenes=json.loads((FROZEN/'reports/SCENES.json').read_text());scene=next(s for s in scenes if s['slug']=='Replica__office2__rgbd')
    checkpoint=Path(scene['checkpoint']);point_cloud,=list((checkpoint/'point_cloud').glob('iteration_*/point_cloud.ply'))
    student_path=FROZEN/'models'/scene['slug']/'continuous_calibration/best.pt'
    protected=[checkpoint/'selector.pt',checkpoint/'selector_config.yaml',point_cloud,student_path]+list(PYTHON.glob('*.py'))
    before={str(p):sha(p) for p in protected}
    model=load_selector(str(checkpoint/'selector.pt'),'cpu').eval().requires_grad_(False)
    student,payload=load_student(student_path,'cpu');student.eval().requires_grad_(False)
    pc=GaussianModel(3);pc.load_ply(str(point_cloud),'cpu')
    for p in [pc._xyz,pc._scaling,pc._rotation,pc._opacity,pc._features_dc,pc._features_rest]:p.requires_grad_(False)
    assert all(p.device.type=='cpu' and p.dtype==torch.float32 for p in model.parameters())
    assert all(p.device.type=='cpu' and p.dtype==torch.float32 for p in student.parameters())
    spatial=model.spatial_compensation
    assert model.local_emd[0].in_features==19 and spatial.encode.in_features==51 and int(getattr(spatial,'geometry_head_schema',0))==0
    records=json.loads((checkpoint/'cameras.json').read_text());cams=[camera(records[int(f*(len(records)-1))]) for f in [.19,.51,.83]]
    codes=[model.joint_view_code(cam) for cam in cams]
    N=len(pc.get_xyz);source_order=torch.randperm(N)
    selector_parameters=sum(p.numel() for module in [model.local_emd,model.soft_net] for p in module.parameters())
    parameters=dict(selector=selector_parameters,teacher=sum(p.numel() for p in spatial.parameters()),student=sum(p.numel() for p in student.parameters()))
    environment=dict(cpu=subprocess.check_output(['lscpu'],text=True),torch=torch.__version__,python=sys.version,
        torch_config=torch.__config__.show(),affinity=sorted(os.sched_getaffinity(0)),interop_threads=torch.get_num_interop_threads(),
        scene=scene,checkpoint=str(checkpoint),student=str(student_path),parameters=parameters,
        CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],cpu_before=subprocess.check_output(['ps','-eo','pid,comm,pcpu','--sort=-pcpu'],text=True).splitlines()[:9])
    save(out/'ENVIRONMENT.json',environment);save(out/'PROTECTED_BEFORE.json',before)
    samples=[];checks=[];pipeline=[]
    ratio=.1
    lo,hi=spatial.ratio_encoding_values;t=2*(math.log(ratio)-math.log(lo))/(math.log(hi)-math.log(lo))-1
    ratio_code=torch.tensor([[t,math.sin(math.pi*t/2),math.cos(math.pi*t/2)]],dtype=torch.float32)
    def teacher_core(inputs,reference,condition):
        encoded=spatial.encode(inputs)+spatial.ratio_projection(ratio_code)
        if condition is not None:encoded=encoded+spatial.view_projection(condition)
        hidden=torch.relu(spatial.body(torch.relu(encoded)))
        parts=[head(hidden) for head in spatial.heads.values()]
        if spatial.bounded_xyz:parts[0]=parts[0]/torch.sqrt(1+parts[0].square().sum(1,keepdim=True)/4)
        parts[0]=parts[0]*reference
        return torch.cat(parts,1)
    def make_case(B,index):
        rows=source_order.roll(index*701)[:B];owners=model.anchors.rows[rows]
        xyz=pc.get_xyz[rows];scale=pc.get_scaling[rows];dc=pc.get_features_dc[rows];opacity=pc.get_opacity[rows]
        code=codes[index][owners];reference=model.anchors.offset_scales[owners];center=model.anchors.centers[owners]
        _,_,_,selector_input=model.forward_selection_features(xyz,scale,dc,opacity,torch.tensor([[ratio]]),
            source_rows=rows,view_code=code,return_logits=True,return_input=True)
        feature=model.anchors.compensation_lookup(rows,ratio)
        offset=((torch.asinh((xyz-center)/reference)-spatial.mean[3:6])/spatial.stddev[3:6]).clamp(-15,15)
        inputs=torch.cat((offset,feature),1)
        comp_view=model.joint_branch_view_code(codes[index],compensation=True)[owners]
        condition=torch.cat((comp_view,comp_view.new_ones((B,1))),1)
        truth=spatial(xyz,pc._scaling[rows],pc.get_rotation[rows],dc,pc._opacity[rows],center,reference,ratio,
            log_scale_input=True,anchor_feature=feature,view_condition=condition)
        actual=teacher_core(inputs,reference,condition);assert torch.equal(actual,truth)
        score=torch.softmax(model.soft_net(model.local_emd(selector_input))/1.,dim=1)[:,1]
        original=model.forward_selection_features(xyz,scale,dc,opacity,torch.tensor([[ratio]]),source_rows=rows,view_code=code)[0]
        assert torch.equal(score,original)
        return dict(selector_input=selector_input,inputs=inputs,reference=reference,condition=condition)
    sizes=[1,100,500,1000,5000,10000,40000] if not args.quick else [100,1000]
    cases={B:[make_case(B,index) for index in range(3)] for B in sizes}
    def call(name,case):
        if name=='selector':return torch.softmax(model.soft_net(model.local_emd(case['selector_input']))/1.,dim=1)[:,1]
        if name=='teacher':return teacher_core(case['inputs'],case['reference'],case['condition'])
        return student(case['inputs'],ratio_code,case['reference'],case['condition'])
    for threads in ([1,2,6] if not args.quick else [1]):
        torch.set_num_threads(threads)
        for round_index in ([1,2] if not args.quick else [1]):
            for B in sizes if round_index==1 else list(reversed(sizes)):
                for name in ['selector','teacher','student']:
                    for index in range(8):value=call(name,cases[B][index%3])
                    assert torch.isfinite(value).all()
                for repeat in range(60 if not args.quick else 8):
                    order=['selector','teacher','student'];shift=(repeat+round_index)%3;order=order[shift:]+order[:shift]
                    if repeat%2:order=order[::-1]
                    for name in order:
                        case=cases[B][repeat%3];tick=time.perf_counter_ns();value=call(name,case);elapsed=(time.perf_counter_ns()-tick)/1e6
                        samples.append(dict(threads=threads,round=round_index,points=B,variant=name,repeat=repeat,ms=elapsed))
                print('CPU_MLP',threads,round_index,B,flush=True)
                csv_save(out/'mlp_samples.csv',samples)
            # 真实CPU准备：名义Anchor候选、真实视角、严格K、网络、补偿、输出打包；不渲染。
            runtime=FixedBudgetRenderer(model,pc,read_config(checkpoint/'selector_config.yaml'),0,'pytorch')
            for requested_ratio in [.01,.1,.7] if not args.quick else [.01]:
                for cam in cams:
                    for _ in range(2):packet,ids,stats=runtime.prepare(cam,ratio_budget=requested_ratio,view_conditioned=True,measure_time=False)
                    for repeat in range(12 if not args.quick else 2):
                        tick=time.perf_counter_ns();packet,ids,stats=runtime.prepare(cam,ratio_budget=requested_ratio,view_conditioned=True,measure_time=False)
                        elapsed=(time.perf_counter_ns()-tick)/1e6
                        assert packet['inputs']['means3D'].device.type=='cpu'
                        pipeline.append(dict(threads=threads,round=round_index,ratio=requested_ratio,repeat=repeat,
                            candidate_N=stats['candidate_gaussians'],K=len(ids),ms=elapsed))
                csv_save(out/'cpu_prepare_samples.csv',pipeline)
    after={name:sha(name) for name in before};assert before==after
    assert not torch.cuda.is_initialized(),'CPU benchmark initialized CUDA'
    save(out/'PROTECTED_AFTER.json',after)
    save(out/'SUMMARY.json',dict(passed=True,parameters=parameters,MLP_samples=len(samples),prepare_samples=len(pipeline),
        device='cpu',precision='FP32',cuda_initialized=False,source_unchanged=True,checkpoint_unchanged=True,
        MLP_scope='预组装真实输入：线性层、ReLU、ratio/view投影、Head、cat、XYZ参考尺度；Selector含softmax',
        prepare_scope='CPU名义候选+视角+Selector+严格K+Feature查询+Compensator+解码+打包；Teacher；不含相机构造、I/O、渲染和CPU/GPU搬运',
        note='Selector按候选数运行，Compensator按选中数运行；同B表为组件规模对照，不能直接当作同一帧相加'))
    print('CPU_BENCHMARK_COMPLETE',parameters,len(samples),len(pipeline),flush=True)


if __name__=='__main__':main()
