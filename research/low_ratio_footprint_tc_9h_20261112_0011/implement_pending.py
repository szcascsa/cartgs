"""第二阶段候选：GPU准备→渲染无中间Host回读；有界容量、必要finite Device gate。"""
from pathlib import Path
import json
import shutil

HERE=Path(__file__).resolve().parent
PREP=HERE.parent/'tensor_core_inference_20261110_2200/preparation_phase/source'


def main():
    state=json.loads((HERE/'reports/RENDER_CONTROLLER.json').read_text())
    assert state['state']=='complete','正式渲染复测结束后才编译，避免CPU编译污染'
    target=HERE/'joint_source';shutil.copytree(PREP,target);shutil.copytree(HERE/'source',target/'raster')
    (HERE/'joint_build').mkdir()
    path=target/'deployment_prepare.cpp';s=path.read_text()
    anchor='  int64_t schema_,sh_degree_,total_,width_;'
    fields='''
  bool pending_requested_=false,pending_inflight_=false;
  cudaStream_t pending_stream_=nullptr;
  torch::Tensor pending_attributes_,pending_ids_,pending_frame_;
  int64_t pending_capacity_=0;
'''
    assert anchor in s;s=s.replace(anchor,anchor+fields)
    graph_anchor='''    plan->execution->replay_frozen();
    readback(plan->buffers[0],host_control_);'''
    graph_extra='''    plan->execution->replay_frozen();
    if(pending_requested_){
      pending_attributes_=attributes;pending_ids_=selected;pending_frame_=plan->buffers[0];pending_capacity_=capacity;
      return Prepared{};
    }
    readback(plan->buffers[0],host_control_);'''
    assert graph_anchor in s;s=s.replace(graph_anchor,graph_extra)
    anchor='''    readback(buffers_[0],host_control_);'''
    assert s.count(anchor)==1;s=s.replace(anchor,'''    if(pending_requested_){
      pending_attributes_=buffers_[6];pending_ids_=buffers_[5];pending_frame_=buffers_[0];pending_capacity_=capacity;
      return Prepared{};
    }
'''+anchor)
    anchor='''    // Route before Graph validation/device guards to retain native request cost.'''
    s=s.replace(anchor,'''    TORCH_CHECK(!pending_inflight_,"Finish pending frame before reusing this pipeline");
'''+anchor)
    anchor='''  torch::Tensor debug_control() const'''
    methods='''  std::vector<torch::Tensor> begin_pending(const torch::Tensor& vm,const torch::Tensor& pm,const torch::Tensor& camera,
      int64_t width,int64_t height,double near,int64_t budget,double ratio,bool use_ratio,bool local,bool compensation){
    TORCH_CHECK(device_counts_,"Pending rendering requires existing GPU count path");
    TORCH_CHECK(!pending_requested_&&!pending_inflight_,"Finish pending frame before reusing this pipeline");
    pending_requested_=true;
    try {prepare_native(vm,pm,camera,width,height,near,budget,ratio,use_ratio,local,compensation);}
    catch(...){pending_requested_=false;throw;}
    pending_requested_=false;pending_inflight_=true;pending_stream_=c10::cuda::getCurrentCUDAStream().stream();
    return {pending_attributes_,pending_ids_,pending_frame_};
  }
  pybind11::tuple finish_pending(){
    namespace py=pybind11;
    TORCH_CHECK(pending_inflight_,"No pending frame");
    c10::cuda::CUDAGuard guard(state_[0].device());
    TORCH_CHECK(c10::cuda::getCurrentCUDAStream().stream()==pending_stream_,"Pending frame belongs to its submission stream");
    {py::gil_scoped_release release;readback(pending_frame_,host_control_);}
    const auto frame=*reinterpret_cast<const DeploymentFrame*>(host_control_.data_ptr<uint8_t>());
    pending_inflight_=false;
    TORCH_CHECK(frame.status!=1,"Nonfinite deployment Selector logits");
    TORCH_CHECK(!frame.status,"Nonfinite deployment compensated attributes");
    TORCH_CHECK(frame.k>=0&&frame.k<=pending_capacity_,"Pending publication capacity");
    py::dict stats;
    stats["candidate_gaussians"]=frame.n;stats["visible_anchors"]=frame.visible;stats["eligible_candidates"]=frame.available;
    stats["requested_budget"]=frame.budget;stats["active_gaussians"]=frame.k;stats["ratio"]=frame.ratio;
    stats["device_counts"]=true;stats["cache_hit"]=false;stats["gpu_render_gate"]=true;
    return py::make_tuple(pending_ids_.slice(0,0,frame.k),stats);
  }
'''
    assert anchor in s;s=s.replace(anchor,methods+anchor)
    anchor='''    .def("prepare",&TensorDeploymentPipeline::prepare)'''
    s=s.replace(anchor,anchor+'''
    .def("_begin_pending",&TensorDeploymentPipeline::begin_pending,py::call_guard<py::gil_scoped_release>())
    .def("_finish_pending",&TensorDeploymentPipeline::finish_pending)''')
    path.write_text(s)
    path=target/'raster/graph_main.cu';s=path.read_text().replace('NineHourRasterizer','PendingNineRasterizer')
    s=s.replace('#include "graph_kernels.cuh"','#include "graph_kernels.cuh"\n#include "../deployment_device_control.h"')
    anchor='using Tensor=torch::Tensor;'
    updater='''
__global__ void updatePendingInputs(GraphInputs input,const float* flat,const DeploymentFrame* frame,int rest,GraphInputs* destination){
  const int cap=input.P;const int k=(frame->status||frame->k<0||frame->k>cap)?0:int(frame->k);
  input.P=k;input.xyz=flat;input.opacity=flat+3ll*k;input.scales=flat+4ll*k;input.rotations=flat+7ll*k;
  input.dc=flat+11ll*k;input.sh=flat+14ll*k;input.M=rest/3;
  *destination=input;
}
'''
    s=s.replace(anchor,anchor+updater)
    begin=s.index('  std::vector<Tensor> render(const Tensor& bg')
    end=s.index('  std::vector<Tensor> diagnose()const',begin)
    original=s[begin:end]
    method=original
    signature=method[:method.index('    TORCH_CHECK(!torch::GradMode')]
    method=method.replace(signature,'''  std::vector<Tensor> render_pending(const Tensor& bg,const Tensor& flat,const Tensor& frame,int64_t capacity_bound,int64_t rest,
      const Tensor& view,const Tensor& proj,double tanx,double tany,int64_t H,int64_t W,int64_t degree,const Tensor& camera){
    const Tensor& xyz=flat;const int variant=5,list_capacity=256,plan_granularity=32;const bool input_in_graph=false;const double modifier=1.;
''')
    method=method.replace('TORCH_CHECK(xyz.is_cuda()&&xyz.dim()==2&&xyz.size(1)==3&&H>0&&W>0,"Graph inputs");',
        'TORCH_CHECK(flat.is_cuda()&&flat.dim()==1&&flat.is_contiguous()&&H>0&&W>0&&capacity_bound>=0&&rest>=0&&flat.numel()==capacity_bound*(17+rest),"Pending flat attributes");\n    TORCH_CHECK(frame.is_cuda()&&frame.device()==flat.device()&&frame.scalar_type()==at::kByte&&frame.numel()==sizeof(DeploymentFrame),"Pending GPU count layout");')
    method=method.replace('for(const auto& t:{bg,xyz,colors,opacity,scales,rotation,covariance,view,proj,dc,sh,camera})','for(const auto& t:{bg,flat,view,proj,camera})')
    method=method.replace('const int P=xyz.size(0);','const int P=capacity_bound;')
    method=method.replace('if(!P)return {out,output_radii};','if(!P)return {bg.view({3,1,1}).expand({3,H,W}).clone(),output_radii};')
    begin=method.index('    auto x=xyz.contiguous()');end=method.index('    if(!parameters.defined())',begin)
    method=method[:begin]+method[end:]
    method=method.replace('inputs.M=sh.numel()?sh.size(1):0','inputs.M=rest/3')
    begin=method.index('    inputs.xyz=x.data_ptr<float>();');end=method.index('    inputs.view=view.data_ptr<float>();',begin)
    method=method[:begin]+'''    inputs.covariance=nullptr;inputs.colors=nullptr;
'''+method[end:]
    begin=method.index('    if(input_in_graph) {');end=method.index('    C10_CUDA_CHECK(cudaGraphLaunch',begin)
    method=method[:begin]+'''    updatePendingInputs<<<1,1,0,stream>>>(inputs,flat.data_ptr<float>(),reinterpret_cast<const DeploymentFrame*>(frame.data_ptr()),rest,reinterpret_cast<GraphInputs*>(parameters.data_ptr()));
'''+method[end:]
    position=s.index('  std::vector<Tensor> diagnose()const');s=s[:position]+method+s[position:]
    s=s.replace('PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {','void register_pending_raster(pybind11::module_& m) {')
    s=s.replace('>(m,"FootprintRasterizer")','>(m,"PendingRasterizer")')
    anchor='.def("render",&PendingNineRasterizer::render,pybind11::call_guard<pybind11::gil_scoped_release>())'
    assert anchor in s;s=s.replace(anchor,anchor+'\n      .def("render_pending",&PendingNineRasterizer::render_pending,pybind11::call_guard<pybind11::gil_scoped_release>())')
    path.write_text(s)
    path=target/'deployment_runtime_bindings.cpp';s=path.read_text()
    anchor='PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {'
    s=s.replace(anchor,'void register_pending_raster(pybind11::module_&);\n'+anchor+'\n  register_pending_raster(module);');path.write_text(s)
    save=HERE/'reports/PENDING_IMPLEMENTATION.json'
    save.write_text(json.dumps(dict(implemented=True,scope='独立联合推理候选，未验收',finite='GPU status在Raster前将非法帧P设0；返回前仍执行原finite异常检查',
        capacity='按请求ratio×总地图N分配上界，实际K只在GPU消费，可能增加padding成本，须实测',
        ownership='每实例串行pending；输出独立分配；finish后才可发下一帧；Stream绑定',time=__import__('time').time()),indent=2,ensure_ascii=False))
    print('PENDING_JOINT_SOURCE_CREATED')


if __name__=='__main__':main()
