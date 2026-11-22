"""训练完成后的独立CPU AVX2部署，全部属性准备在CPU；不改变训练入口。"""
from pathlib import Path
from functools import lru_cache
import copy,os,torch

HERE=Path(__file__).resolve().parent

@lru_cache(None)
def extension():
    from torch.utils.cpp_extension import load
    os.environ['PATH']='/home/prmi/anaconda3/envs/zwj/bin:'+os.environ.get('PATH','')
    os.environ.setdefault('MAX_JOBS','1')
    (HERE/'build').mkdir(exist_ok=True)
    return load(name='mycartgs_cpu_student_fast',sources=[str(HERE/'source/cpu_student_fast.cpp')],
        extra_cflags=['-O3','-mavx2','-mfma','-fno-math-errno','-pthread','-I/usr/local/cuda/include'],
        extra_ldflags=['-pthread','-L/usr/local/cuda/lib64','-lcudart','-lc10_cuda','-ltorch_cuda','-lmvec','-lm'],
        build_directory=str(HERE/'build'),verbose=True)


class CpuRenderer:
    @classmethod
    @torch.no_grad()
    def from_gpu_reference(cls,reference,**kwargs):
        """离线导出时保留现有GPU噪声身份；返回后运行全部在CPU。"""
        model=copy.deepcopy(reference.model).cpu().eval().requires_grad_(False)
        pc=copy.copy(reference.gaussians)
        for name in ['_xyz','_scaling','_rotation','_opacity','_features_dc','_features_rest']:
            setattr(pc,name,getattr(reference.gaussians,name).detach().cpu().clone())
        student=kwargs.get('student')
        if student is not None:kwargs['student']=copy.deepcopy(student).cpu()
        return cls(model,pc,reference.noise.detach().cpu(),**kwargs)

    @classmethod
    @torch.no_grad()
    def from_checkpoint(cls,checkpoint,noise_path,**kwargs):
        """CPU独立启动；noise_path是离线保存的原GPU固定键，避免换设备重采样。"""
        from gaussian_model import GaussianModel
        from selector import load_selector
        checkpoint=Path(checkpoint);model=load_selector(str(checkpoint/'selector.pt'),'cpu')
        pc=GaussianModel(3);paths=list((checkpoint/'point_cloud').glob('iteration_*/point_cloud.ply'))
        if len(paths)!=1:raise ValueError('Expected one frozen point cloud')
        pc.load_ply(str(paths[0]),'cpu');noise=torch.load(noise_path,map_location='cpu',weights_only=True)
        if isinstance(noise,dict):noise=noise['noise']
        return cls(model,pc,noise,**kwargs)

    @torch.no_grad()
    def __init__(self,model,pc,noise,threads=6,student=None,selector_prefix=True,factored=True,diagnose=False,
                 fused_schedule=True,kernel_mode=1,native_wrapper=True,small_kernel=True,ordered_mode=3,selection_mode=1,comp_batch=6,student_kernel=0,student_decode=0,student_context=0):
        from selector import slot_eligibility
        if pc.get_xyz.device.type!='cpu' or next(model.parameters()).device.type!='cpu':raise ValueError('Load map/network directly on CPU')
        a=model.anchors;s=model.spatial_compensation;schema=int(getattr(model,'joint_view_conditioning_schema',0))
        if getattr(a,'compensation_residual_mode',0)!=1 or s.ratio_feature_width!=48 or not s.offset_only or s.bounded_xyz:
            raise ValueError('CPU deployment requires unbounded Residual48')
        owners=a.rows;reference=a.offset_scales[owners];raw_offset=(pc.get_xyz-a.centers[owners])/reference
        offset=((torch.asinh(raw_offset)-s.mean[3:6])/s.stddev[3:6]).clamp(-15,15)
        physical=torch.cat((raw_offset,pc.get_scaling,pc.get_features_dc.squeeze(1),pc.get_opacity),1)
        state=[a.centers,a.offset_scales,owners,slot_eligibility(model,pc.get_opacity),pc.get_xyz,pc._scaling,
            pc.get_rotation,pc._opacity,pc.get_features_dc,pc.get_features_rest,physical,offset,noise]
        selector=[model.local_emd[0].weight,model.local_emd[0].bias,model.local_emd[2].weight,model.local_emd[2].bias,
            model.soft_net.weight,model.soft_net.bias]
        comp=s if student is None else student
        if student is None:
            params=[s.encode.weight,s.encode.bias,s.body.weight,s.body.bias,s.ratio_projection.weight]
            for head in s.heads.values():params.extend((head[0].weight,head[0].bias,head[2].weight,head[2].bias))
        else:params=list(student.deployment_parameters())
        tables=[getattr(a,'compensation_features_'+str(i)) for i in range(len(a.compensation_values))]
        vw=comp.view_projection.weight if schema else pc.get_xyz.new_empty(0)
        self.native=extension().CpuPipeline(state,selector,params,tables,vw,list(a.compensation_values),list(s.ratio_encoding_values),
            schema,float(model.selector_view_scene_scale) if schema else 1.,hasattr(s,'softplus_scale_schema'),threads,
            selector_prefix,factored,diagnose,fused_schedule,kernel_mode,small_kernel,ordered_mode,selection_mode,comp_batch,student_kernel,student_decode,student_context)
        self.native_wrapper=native_wrapper
        self.rest=pc.get_features_rest.shape[1:];self.sh_degree=pc.active_sh_degree;self.total=len(pc.get_xyz)
        self.width=17+pc.get_features_rest[0].numel()

    def packet(self,flat,n):
        shapes=[(n,3),(n,1),(n,3),(n,4),(n,1,3),(n,)+tuple(self.rest),(n,3)]
        inputs={};start=0
        for key,shape in zip(['means3D','opacities','scales','rotations','dc','shs','means2D'],shapes):
            size=1
            for dim in shape:size*=dim
            inputs[key]=flat[start:start+size].view(shape);start+=size
        assert start==flat.numel()
        inputs.update(colors_precomp=None,cov3D_precomp=None)
        return dict(sh_degree=self.sh_degree,inputs=inputs)

    @torch.no_grad()
    def prepare(self,camera,budget=None,*,ratio_budget=None,global_ratio=None,view_conditioned=True,compensation=True,
                timing=False,output=None,id_output=None,flat_output=False):
        if sum(v is not None for v in [budget,ratio_budget,global_ratio])!=1:raise ValueError('Use one CPU budget request')
        kind=1 if budget is not None else 2 if global_ratio is not None else 0
        ratio=global_ratio if global_ratio is not None else ratio_budget or 0.
        if self.native_wrapper and not flat_output:
            return self.native.prepare_packet(camera,kind,float(ratio),int(budget or 0),view_conditioned,compensation,timing,output,id_output,self.sh_degree)
        flat,ids,stats,times=self.native.prepare(camera.world_view_transform.contiguous(),camera.full_proj_transform.contiguous(),
            camera.camera_center.contiguous(),camera.image_width,camera.image_height,camera.znear,kind,float(ratio),int(budget or 0),
            view_conditioned,compensation,timing,output,id_output)
        info=dict(zip(['total_gaussians','anchors','visible_anchors','candidate_gaussians','eligible_candidates',
            'requested_budget','active_gaussians','ratio','workspace_bytes','threads'],stats))
        if timing:info['stage_ms']=times
        return (flat if flat_output else self.packet(flat,len(ids))),ids,info


if __name__=='__main__':extension()
