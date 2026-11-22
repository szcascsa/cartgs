"""从受保护v31源码生成独立Tensor Core后端；不修改原项目。"""
from pathlib import Path
HERE=Path(__file__).resolve().parent;SRC=HERE/'source'
def edit(name,old,new,count=1):
    p=SRC/name;s=p.read_text(encoding='utf-8');assert s.count(old)==count,(name,old,s.count(old));p.write_text(s.replace(old,new),encoding='utf-8')
edit('deployment_runtime_kernels.cu','#include <torch/extension.h>','#include <torch/extension.h>\n#include <mma.h>\n#include <cuda_fp16.h>')
edit('deployment_runtime_kernels.cu','\n}\n\ntorch::Tensor deployment_view(', '\n#include "tensor_networks.cuh"\n}\n\n'+(HERE/'host_wrappers.txt').read_text()+'\ntorch::Tensor deployment_view(')
edit('deployment_runtime_kernels.cu','const torch::Tensor& prefix,const torch::Tensor& visible) {','const torch::Tensor& prefix,const torch::Tensor& visible,\n    const std::vector<torch::Tensor>& tensor_weights,int64_t selector_mode,int64_t compensation_mode,int64_t tensor_warps) {')
edit('deployment_runtime_kernels.cu','  if(prefix.defined()) {\n  if(selector_tiled)', '''  if(selector_mode) {
    auto launch=[&](auto mode,auto warps){
      constexpr int Mode=decltype(mode)::value,Warps=decltype(warps)::value;
      if(selector[0].size(0)==19)selectorTensor<Mode,Warps,19><<<(n+Warps*16-1)/(Warps*16),Warps*32,0,stream>>>(n,state[6].data_ptr<float>(),rows,state[3].data_ptr<int64_t>(),buffers[2].data_ptr<float>(),state[9].data_ptr<float>(),sw,0.f,local,schema==2||schema==3,buffers[3].data_ptr<float>(),buffers[4].data_ptr<float>(),status,frame,prefix.defined()?prefix.data_ptr<float>():nullptr,tensor_weights[0].data_ptr());
      else selectorTensor<Mode,Warps,11><<<(n+Warps*16-1)/(Warps*16),Warps*32,0,stream>>>(n,state[6].data_ptr<float>(),rows,state[3].data_ptr<int64_t>(),buffers[2].data_ptr<float>(),state[9].data_ptr<float>(),sw,0.f,local,false,buffers[3].data_ptr<float>(),buffers[4].data_ptr<float>(),status,frame,prefix.defined()?prefix.data_ptr<float>():nullptr,tensor_weights[0].data_ptr());
    };
    dispatchTensor(selector_mode,tensor_warps,launch);
  } else if(prefix.defined()) {
  if(selector_tiled)''')
edit('deployment_runtime_kernels.cu','      if(compensation.size()==13) {','''      if(compensation_mode) {
        TORCH_CHECK(compensation.size()==13,"Tensor Compensation requires the frozen 32/32 student");
        TensorWeights tw{};for(int i=1;i<4;++i)tw.p[i]=tensor_weights[i].data_ptr();
        auto launch=[&](auto mode,auto warps){constexpr int Mode=decltype(mode)::value,Warps=decltype(warps)::value;
          compensationTensor<Mode,Warps,Direct><<<(capacity+Warps*16-1)/(Warps*16),Warps*32,0,stream>>>(capacity,state[7].data_ptr<float>(),buffers[5].data_ptr<int64_t>(),state[3].data_ptr<int64_t>(),state[8].data_ptr<float>(),t,c,cw,buffers[2].data_ptr<float>(),local?view_weight.data_ptr<float>():nullptr,local&&schema,schema==2||schema==4,m,softplus,buffers[6].data_ptr<float>(),nullptr,status,frame,destination,tw);
        };dispatchTensor(compensation_mode,tensor_warps,launch);
      } else if(compensation.size()==13) {''')
edit('deployment_prepare.cpp','const torch::Tensor&,const torch::Tensor&);\nstd::vector<torch::Tensor> deployment_fused_candidates','const torch::Tensor&,const torch::Tensor&,const std::vector<torch::Tensor>&,int64_t,int64_t,int64_t);\nstd::vector<torch::Tensor> deployment_tensor_pack(const std::vector<torch::Tensor>&,const std::vector<torch::Tensor>&,int64_t,int64_t);\nstd::vector<torch::Tensor> deployment_fused_candidates')
edit('deployment_prepare.cpp','  bool visible_view_=false;','  bool visible_view_=false;\n  int64_t tensor_selector_=0,tensor_compensator_=0,tensor_warps_=4;\n  std::vector<torch::Tensor> tensor_weights_;')
edit('deployment_prepare.cpp','visible_view_?plan.visible:torch::Tensor());','visible_view_?plan.visible:torch::Tensor(),tensor_weights_,tensor_selector_,tensor_compensator_,tensor_warps_);')
edit('deployment_prepare.cpp','visible_view_?visible:torch::Tensor());','visible_view_?visible:torch::Tensor(),tensor_weights_,tensor_selector_,tensor_compensator_,tensor_warps_);')
edit('deployment_prepare.cpp','  int64_t prefix_bytes() const', '''  void tensor_options(int64_t selector_mode,int64_t compensator_mode,int64_t warps) {
    TORCH_CHECK(selector_mode>=0&&selector_mode<=2&&compensator_mode>=0&&compensator_mode<=2,"Tensor mode: 0=CUDA,1=TF32,2=FP16");
    TORCH_CHECK(warps==1||warps==2||warps==4,"Tensor warps must be 1,2 or 4");
    c10::cuda::CUDAGuard guard(state_[0].device());graphs_.clear();
    tensor_weights_=deployment_tensor_pack(selector_,compensator_,selector_mode,compensator_mode);
    tensor_selector_=selector_mode;tensor_compensator_=compensator_mode;tensor_warps_=warps;
  }
  int64_t tensor_bytes() const {int64_t n=0;for(const auto& x:tensor_weights_)if(x.defined())n+=x.nbytes();return n;}
  int64_t prefix_bytes() const''')
edit('deployment_prepare.cpp','    stats["selector_tiled"]=result.selector_tiled;','    stats["selector_tiled"]=result.selector_tiled;\n    stats["tensor_selector"]=result.device_counts?tensor_selector_:0;stats["tensor_compensator"]=result.device_counts&&compensation?tensor_compensator_:0;\n    stats["tensor_warps"]=tensor_warps_;stats["tensor_weight_bytes"]=tensor_bytes();')
edit('deployment_prepare.cpp','    .def("_prefix_bytes",','    .def("_tensor_options",&DeploymentPipeline::tensor_options)\n    .def("_tensor_bytes",&DeploymentPipeline::tensor_bytes)\n    .def("_prefix_bytes",')
# pybind already module_local; distinct type names also avoid symbol/identity ambiguity.
p=SRC/'deployment_prepare.cpp';p.write_text(p.read_text().replace('DeploymentPipeline','TensorDeploymentPipeline'),encoding='utf-8')
edit('deployment_runtime.py',"name='mycartgs_deployment_fp32_v31'","name='mycartgs_tensor_deployment_v1'")
edit('deployment_runtime.py',"extra_cflags=['-O3'], extra_cuda_cflags=['-O3'], verbose=False)","extra_cflags=['-O3'], extra_cuda_cflags=['-O3','-lineinfo'], build_directory=str(root.parent/'build'), verbose=True)")
edit('deployment_runtime.py','small_compensation_tile=True,student=None):','small_compensation_tile=True,student=None,tensor_selector=1,tensor_compensator=1,tensor_warps=4):')
edit('deployment_runtime.py','small_compensation_tile=True,student_checkpoint=None):','small_compensation_tile=True,student_checkpoint=None,tensor_selector=1,tensor_compensator=1,tensor_warps=4):')
edit('deployment_runtime.py','        self._pipeline._static_options(selector_static_prefix,visible_view,small_compensation_tile)','        self._pipeline._static_options(selector_static_prefix,visible_view,small_compensation_tile)\n        self._pipeline._tensor_options(tensor_selector,tensor_compensator,tensor_warps)')
edit('deployment_runtime.py','            student=student)','            student=student,tensor_selector=tensor_selector,tensor_compensator=tensor_compensator,tensor_warps=tensor_warps)')
print('TC_SOURCE_INTEGRATED')
