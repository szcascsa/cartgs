#include <torch/extension.h>
#include <pybind11/numpy.h>
#include <ATen/Functions.h>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/core/InferenceMode.h>
#include <memory>
#include <chrono>
#include <cmath>
#include <climits>
#include "deployment_device_control.h"
#include "pending_raster_api.h"

torch::Tensor frustum_planes(const torch::Tensor&);
torch::Tensor frustum_test(const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,const torch::Tensor&);
std::vector<torch::Tensor> preparation_candidates(const torch::Tensor&,const torch::Tensor&,
    const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,
    const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,int64_t);
void deployment_finish_counts(const torch::Tensor&,const torch::Tensor&);
std::vector<torch::Tensor> visible_candidates(const torch::Tensor&,const torch::Tensor&,
    const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,const torch::Tensor&);
void deployment_frame_controls(const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,
    const DeploymentKnots&,int64_t,double,bool);
void deployment_device_forward(const std::vector<torch::Tensor>&,const std::vector<torch::Tensor>&,
    const std::vector<torch::Tensor>&,const std::vector<torch::Tensor>&,const std::vector<torch::Tensor>&,
    const std::vector<torch::Tensor>&,const torch::Tensor&,const std::vector<torch::Tensor>&,
    const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,double,double,int64_t,bool,bool,bool,int64_t,
    bool,const std::vector<torch::Tensor>&,bool,bool,const torch::Tensor&,bool,bool,
    const torch::Tensor&,const torch::Tensor&,const std::vector<torch::Tensor>&,int64_t,int64_t,int64_t,int64_t,bool);
std::vector<torch::Tensor> deployment_tensor_pack(const std::vector<torch::Tensor>&,const std::vector<torch::Tensor>&,const torch::Tensor&,int64_t,int64_t);
std::vector<torch::Tensor> deployment_fused_candidates(const torch::Tensor&,const torch::Tensor&,
    const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,
    const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,const torch::Tensor&);
void deployment_frame_controls_fused(const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,
    const DeploymentKnots&,const torch::Tensor&,const torch::Tensor&,int64_t,double,bool);
void deployment_frame_controls_request(const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,
    const DeploymentKnots&,const torch::Tensor&);
void deployment_upload_request(const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,int64_t,
    double,double,double,bool,const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,const torch::Tensor&);

torch::Tensor deployment_radix_select(const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,
    int64_t,const std::vector<torch::Tensor>&,const torch::Tensor&);
std::vector<torch::Tensor> deployment_compensate_impl(const torch::Tensor&,const torch::Tensor&,
    const torch::Tensor&,const torch::Tensor&,const std::vector<torch::Tensor>&,
    const std::vector<torch::Tensor>&,const std::vector<torch::Tensor>&,const torch::Tensor&,
    const torch::Tensor&,const std::vector<double>&,const std::vector<double>&,bool,int64_t,bool,const torch::Tensor&,int64_t);

torch::Tensor deployment_select_checked(const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,
    int64_t,const torch::Tensor&,const std::vector<torch::Tensor>&,const torch::Tensor&);
torch::Tensor publish_selected(const std::vector<torch::Tensor>&,const torch::Tensor&,
    const torch::Tensor&,const torch::Tensor&,bool);
torch::Tensor deployment_view(const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,
    const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,double,double);
std::vector<torch::Tensor> deployment_selector(const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,
    const torch::Tensor&,const torch::Tensor&,const std::vector<torch::Tensor>&,double,bool,int64_t);
std::vector<torch::Tensor> deployment_selector_impl(const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,
    const torch::Tensor&,const torch::Tensor&,const std::vector<torch::Tensor>&,double,bool,int64_t,bool,const torch::Tensor& = torch::Tensor());
std::vector<torch::Tensor> deployment_compensate_checked(const torch::Tensor&,const torch::Tensor&,
    const torch::Tensor&,const torch::Tensor&,const std::vector<torch::Tensor>&,
    const std::vector<torch::Tensor>&,const std::vector<torch::Tensor>&,const torch::Tensor&,
    const torch::Tensor&,const std::vector<double>&,const std::vector<double>&,bool,int64_t,bool,const torch::Tensor&);

torch::Tensor deployment_selector_prefix(const torch::Tensor&,const torch::Tensor&);
torch::Tensor deployment_view_masked(const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,
    const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,double,double,const torch::Tensor&);

std::vector<torch::Tensor> deployment_tensor_selector(const torch::Tensor& physical,const torch::Tensor& rows,
    const torch::Tensor& owners,const torch::Tensor& view,const torch::Tensor& noise,
    const std::vector<torch::Tensor>& weights,double ratio,bool local,int64_t schema,
    const torch::Tensor& prefix,const std::vector<torch::Tensor>& packed,int64_t mode,int64_t warps);

std::vector<torch::Tensor> deployment_tensor_compensate(const torch::Tensor& offset,const torch::Tensor& rows,
    const torch::Tensor& owners,const torch::Tensor& reference,const std::vector<torch::Tensor>& tables,
    const std::vector<torch::Tensor>& weights,const std::vector<torch::Tensor>& attributes,
    const torch::Tensor& view,const torch::Tensor& view_weight,const std::vector<double>& residual_weights,
    const std::vector<double>& ratio,bool local,int64_t schema,bool softplus,
    const std::vector<torch::Tensor>& packed,int64_t mode,int64_t warps,const torch::Tensor& shared_status);

namespace {
// Preparation uses fixed Gumbel keys and has no RNG nodes. Keep LibTorch's
// graph allocator ownership, and avoid its two seed/offset fills per replay.
struct FrozenPreparationGraph : at::cuda::CUDAGraph {
  void replay_frozen() {
    TORCH_CHECK(wholegraph_increment_==0,"Deployment Graph unexpectedly captured RNG");
    C10_CUDA_CHECK(cudaGraphLaunch(graph_exec_,c10::cuda::getCurrentCUDAStream()));
  }
};

struct Prepared {
  std::vector<torch::Tensor> parts;
  torch::Tensor ids;
  int64_t n,visible,available,budget,k;
  double ratio,ms;
  bool has_ratio;
  bool device_counts=false;
  bool shrinking_radix=false;
  int64_t compensation_tile=0;
  bool selector_tiled=false;
  bool graph_replay=false;
  int64_t graph_count=0;
};
}

// Per-frame scheduling is native C++; no Python callback or result cache is used.
std::vector<torch::Tensor> preparation_camera_parts(const torch::Tensor&,const torch::Tensor&,const torch::Tensor&);

void preparation_camera_into(const double*,const torch::Tensor&,const torch::Tensor&,const torch::Tensor&);

class TensorDeploymentPipeline {
  bool small_radix_finish_=false;
  torch::Tensor pose_host_,pose_device_;
  std::vector<torch::Tensor> pose_parts_;
  cudaStream_t pose_stream_=nullptr;

  std::vector<torch::Tensor> state_,selector_,compensator_,tables_,master_,base_,buffers_,sort_,radix_;
  torch::Tensor view_weight_,rows_,counts_,candidate_workspace_,host_control_,empty_,host_counts_,host_status_,select_workspace_;
  DeploymentKnots control_knots_;
  std::vector<double> knots_,bounds_;
  std::vector<std::vector<int64_t>> shapes_;
  int64_t schema_,sh_degree_,total_,width_;
  bool pending_requested_=false,pending_inflight_=false;
  cudaStream_t pending_stream_=nullptr;
  torch::Tensor pending_attributes_,pending_ids_,pending_frame_;
  int64_t pending_capacity_=0;

  bool softplus_,device_counts_;
  bool shrinking_radix_=true;
  int64_t compensation_tile_=16;
  bool selector_tiled_=true;
  bool saturated_selection_=true;
  bool graph_enabled_=false;
  bool unroll_compensation_=false;
  bool fused_selection_=false,fused_candidates_=false;
  torch::Tensor anchor_partials_,selector_prefix_;
  bool visible_view_=false;
  int64_t tensor_selector_=0,tensor_compensator_=0,tensor_warps_=4,tensor_comp_warps_=4;
  std::vector<torch::Tensor> tensor_weights_;
  bool small_compensation_tile_=false;
  int64_t preparation_frustum_=0,preparation_projection_=0,preparation_device_=0;
  struct GraphPlan {
    int64_t capacity;
    bool local,compensation;
    int64_t tile;
    torch::Tensor camera,request,planes,distance,visible;
    std::vector<torch::Tensor> buffers;
    std::unique_ptr<FrozenPreparationGraph> execution;
  };
  std::vector<std::unique_ptr<GraphPlan>> graphs_;

  int64_t compensation_tile_for(int64_t count,bool original_tiled) const {
    if(original_tiled)return compensation_tile_;
    if(small_compensation_tile_ && unroll_compensation_ && compensation_tile_==16 && count>=768)return 8208;
    return 0;
  }

  void readback(const torch::Tensor& from,const torch::Tensor& to) {
    const auto stream=c10::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cudaMemcpyAsync(to.data_ptr(),from.data_ptr(),from.nbytes(),cudaMemcpyDeviceToHost,stream));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  }

  void graph_work(GraphPlan& plan) {
    auto vm=plan.camera.slice(0,0,16).view({4,4});
    auto pm=plan.camera.slice(0,16,32).view({4,4});
    auto center=plan.camera.slice(0,32,35);
    if(!preparation_frustum_){plan.planes=frustum_planes(pm);
      plan.distance=at::matmul(state_[0],plan.planes.slice(1,0,3).t());}
    if(fused_candidates_ || preparation_frustum_) {
      auto candidates=preparation_frustum_?
          preparation_candidates(state_[0],pm,state_[1],state_[2],state_[3],state_[4],rows_,counts_,candidate_workspace_,anchor_partials_,preparation_frustum_):
          deployment_fused_candidates(plan.distance,plan.planes,state_[1],state_[2],state_[3],state_[4],rows_,counts_,candidate_workspace_,anchor_partials_);
      plan.visible=candidates[0];candidate_workspace_=candidates[1];
      deployment_frame_controls_fused(counts_,plan.buffers[0],plan.buffers[7],control_knots_,anchor_partials_,plan.request,0,0.,false);
    } else {
      plan.visible=frustum_test(plan.distance,plan.planes,state_[1],state_[2]);
      auto candidates=visible_candidates(plan.visible,state_[3],state_[4],rows_,counts_,candidate_workspace_);
      candidate_workspace_=candidates[2];
      deployment_frame_controls_request(counts_,plan.buffers[0],plan.buffers[7],control_knots_,plan.request);
    }
    deployment_device_forward(state_,selector_,compensator_,tables_,master_,base_,view_weight_,plan.buffers,
        vm,pm,center,1.,1.,schema_,softplus_,plan.local,plan.compensation,
        plan.tile,shrinking_radix_ && total_>=4096,radix_,selector_tiled_,
        saturated_selection_,plan.request,unroll_compensation_,fused_selection_,selector_prefix_,
        visible_view_?plan.visible:torch::Tensor(),tensor_weights_,tensor_selector_,tensor_compensator_,tensor_warps_,tensor_comp_warps_,small_radix_finish_);
  }

  Prepared prepare_graph(const torch::Tensor& vm,const torch::Tensor& pm,const torch::Tensor& center,
      int64_t width,int64_t height,double near,int64_t budget,double ratio,bool use_ratio,bool local,bool compensation) {
    c10::cuda::CUDAGuard guard(state_[0].device());
    TORCH_CHECK(!torch::GradMode::is_enabled() && width>0 && height>0 && near>0 && budget>=0 &&
        (!use_ratio || (std::isfinite(ratio) && ratio>=0 && ratio<=1)),"Invalid deployment Graph request");
    TORCH_CHECK(!local || schema_,"Checkpoint has no view-conditioned branch");
    TORCH_CHECK(vm.device()==state_[0].device() && pm.device()==vm.device() && center.device()==vm.device() &&
        vm.scalar_type()==at::kFloat && pm.scalar_type()==at::kFloat && center.scalar_type()==at::kFloat &&
        vm.sizes()==at::IntArrayRef({4,4}) && pm.sizes()==vm.sizes() && center.numel()==3 && center.dim()==1,"Graph camera layout");
    const auto start=std::chrono::steady_clock::now();
    const int64_t capacity=use_ratio?int64_t(std::floor(ratio*total_)):std::min(budget,total_);
    auto selected=torch::empty({capacity},rows_.options());
    auto attributes=torch::empty({capacity*width_},state_[0].options());
    int64_t bucket=std::min(total_,int64_t(1024));
    for(int64_t limit:{int64_t(2048),int64_t(4096),int64_t(8192),int64_t(16384),int64_t(32768)}) {
      if(capacity>bucket)bucket=std::min(total_,limit);
    }
    if(capacity>bucket)bucket=total_;
    const int64_t tile=compensation?compensation_tile_for(capacity,bucket>1024):0;
    GraphPlan* plan=nullptr;
    for(auto& candidate:graphs_) {
      if(candidate->capacity==bucket && candidate->local==local && candidate->compensation==compensation && candidate->tile==tile) {
        plan=candidate.get();
        break;
      }
    }
    if(!plan) {
      c10::InferenceMode normal_tensors(false);
      at::NoGradGuard no_grad;
      auto created=std::make_unique<GraphPlan>();
      created->capacity=bucket;
      created->local=local;
      created->compensation=compensation;
      created->tile=tile;
      created->camera=torch::empty({35},state_[0].options());
      created->request=torch::empty({sizeof(DeploymentRequest)},state_[0].options().dtype(at::kByte));
      created->buffers=buffers_;
      created->buffers[5]=torch::empty({bucket},rows_.options());
      created->buffers[6]=empty_;
      const auto caller=c10::cuda::getCurrentCUDAStream();
      const auto capture=c10::cuda::getStreamFromPool(false,state_[0].get_device());
      cudaEvent_t ready;
      C10_CUDA_CHECK(cudaEventCreateWithFlags(&ready,cudaEventDisableTiming));
      C10_CUDA_CHECK(cudaEventRecord(ready,caller));
      C10_CUDA_CHECK(cudaStreamWaitEvent(capture,ready,0));
      C10_CUDA_CHECK(cudaEventDestroy(ready));
      {
        c10::cuda::CUDAStreamGuard stream_guard(capture);
        deployment_upload_request(vm,pm,center,budget,ratio,double(width)/height,near,use_ratio,
            created->camera,created->request,attributes,selected);
        graph_work(*created);
        C10_CUDA_CHECK(cudaStreamSynchronize(capture));
        created->execution=std::make_unique<FrozenPreparationGraph>();
        created->execution->capture_begin({0,0},cudaStreamCaptureModeThreadLocal);
        graph_work(*created);
        created->execution->capture_end();
      }
      plan=created.get();
      graphs_.push_back(std::move(created));
    }
    deployment_upload_request(vm,pm,center,budget,ratio,double(width)/height,near,use_ratio,
        plan->camera,plan->request,attributes,selected);
    plan->execution->replay_frozen();
    if(pending_requested_){
      pending_attributes_=attributes;pending_ids_=selected;pending_frame_=plan->buffers[0];pending_capacity_=capacity;
      return Prepared{};
    }
    readback(plan->buffers[0],host_control_);
    const auto& frame=*reinterpret_cast<const DeploymentFrame*>(host_control_.data_ptr<uint8_t>());
    TORCH_CHECK(frame.status!=1,"Nonfinite deployment Selector logits");
    TORCH_CHECK(!frame.status,"Nonfinite deployment compensated attributes");
    TORCH_CHECK(frame.k<=capacity,"Graph publication capacity");
    Prepared output;
    output.device_counts=true;
    output.graph_replay=true;
    output.graph_count=graphs_.size();
    output.n=frame.n;output.visible=frame.visible;output.available=frame.available;
    output.budget=frame.budget;output.k=frame.k;output.ratio=frame.ratio;output.has_ratio=frame.has_ratio;
    output.shrinking_radix=frame.k && shrinking_radix_ && total_>=4096;
    output.selector_tiled=frame.k && selector_tiled_;
    output.compensation_tile=compensation?tile:0;
    output.ids=selected.slice(0,0,frame.k);
    int64_t offset=0;
    for(auto shape:shapes_) {
      shape[0]=frame.k;
      int64_t length=1;
      for(auto dim:shape)length*=dim;
      output.parts.push_back(attributes.slice(0,offset,offset+length).view(shape));
      offset+=length;
    }
    output.ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    return output;
  }
public:
  TensorDeploymentPipeline(std::vector<torch::Tensor> state,std::vector<torch::Tensor> selector,
      std::vector<torch::Tensor> compensator,std::vector<torch::Tensor> tables,
      std::vector<torch::Tensor> master,std::vector<torch::Tensor> base,torch::Tensor view_weight,
      std::vector<double> knots,std::vector<double> bounds,int64_t schema,int64_t degree,bool softplus):
      state_(std::move(state)),selector_(std::move(selector)),compensator_(std::move(compensator)),
      tables_(std::move(tables)),master_(std::move(master)),base_(std::move(base)),view_weight_(view_weight),
      knots_(knots),bounds_(bounds),schema_(schema),sh_degree_(degree),softplus_(softplus) {
    TORCH_CHECK(state_.size()==13 && bounds.size()==2 && knots.size()==tables_.size() && knots.size()>=2 && knots.size()<=6,"Frozen pipeline state");
    total_=state_[3].numel();TORCH_CHECK(total_<=INT_MAX,"Deployment map capacity");
    device_counts_=total_<=65536;
    control_knots_={};control_knots_.count=knots.size();
    for(size_t j=0;j<knots.size();++j)control_knots_.values[j]=knots[j];
    control_knots_.log_lo=std::log(bounds[0]);control_knots_.log_span=std::log(bounds[1])-control_knots_.log_lo;
    c10::cuda::CUDAGuard guard(state_[3].device());
    rows_=torch::empty_like(state_[3]);counts_=torch::empty({3},state_[3].options());
    anchor_partials_=torch::empty({2*((state_[0].size(0)+255)/256)},state_[3].options());
    candidate_workspace_=torch::empty({0},state_[3].options().dtype(at::kByte));
    empty_=torch::empty({0},state_[0].options());
    const auto floats=state_[0].options(),ints=state_[3].options().dtype(at::kInt);
    radix_={torch::empty({total_},ints),torch::empty({total_},ints),torch::empty({264},ints),
        torch::empty({2*((total_+255)/256)},ints)};
    {
      host_control_=torch::empty({sizeof(DeploymentFrame)},torch::TensorOptions().dtype(at::kByte).device(at::kCPU).pinned_memory(true));
      const int64_t blocks=(total_+255)/256;
      buffers_={torch::empty({sizeof(DeploymentFrame)},floats.dtype(at::kByte)),rows_,
          torch::empty({schema_?state_[0].size(0):0,7},floats),torch::empty({total_},floats),
          torch::empty({total_,2},floats),torch::empty({0},rows_.options()),torch::empty({0},floats),
          torch::empty({2+2*blocks},ints)};
    }
    if(!device_counts_ || total_>16384) {
      host_counts_=torch::empty({3},torch::TensorOptions().dtype(at::kLong).device(at::kCPU).pinned_memory(true));
      host_status_=torch::empty({1},torch::TensorOptions().dtype(at::kInt).device(at::kCPU).pinned_memory(true));
      select_workspace_=torch::empty({2+2*((int64_t(16384)+255)/256)},ints);
      for(auto dtype:{at::kFloat,at::kFloat,at::kLong,at::kLong})sort_.push_back(torch::empty({total_},floats.dtype(dtype)));
      sort_.push_back(torch::empty({32*total_+4096},floats.dtype(at::kByte)));
    }
    width_=17;int64_t rest=1;for(int j=1;j<base_[5].dim();++j)rest*=base_[5].size(j);width_+=rest;
    // Validate immutable layouts once using the original checked entry points.
    auto check_rows=rows_.slice(0,0,0);
    auto checked=deployment_selector(state_[6],check_rows,state_[3],empty_,state_[9],selector_,knots[0],false,schema_);
    deployment_compensate_checked(state_[7],check_rows,state_[3],state_[8],tables_,compensator_,master_,empty_,
        view_weight_,std::vector<double>(knots.size(),0.),{0.,0.,1.},false,schema_,softplus_,checked[2]);
    TORCH_CHECK(!schema_ || view_weight_.sizes()==at::IntArrayRef({8,compensator_.size()==13?32:64}),"View projection layout");
    for(int j=0;j<6;++j)shapes_.push_back(base_[j].sizes().vec());
    shapes_.push_back({total_,3});
  }

  Prepared prepare_device(const torch::Tensor& vm,const torch::Tensor& pm,const torch::Tensor& camera,
      int64_t width,int64_t height,double near,int64_t requested_budget,double requested_ratio,
      bool use_ratio,bool local,bool compensation) {
    TORCH_CHECK(!torch::GradMode::is_enabled(),"Deployment pipeline is inference-only");
    TORCH_CHECK(width>0 && height>0 && near>0 && requested_budget>=0 &&
        (!use_ratio || (std::isfinite(requested_ratio) && requested_ratio>=0 && requested_ratio<=1)),"Invalid deployment request");
    TORCH_CHECK(!local || schema_,"Checkpoint has no view-conditioned branch");
    c10::cuda::CUDAGuard guard(state_[0].device());
    const auto start=std::chrono::steady_clock::now();
    torch::Tensor planes,distance;
    if(!preparation_frustum_){planes=frustum_planes(pm);
      distance=at::matmul(state_[0],planes.slice(1,0,3).t());}
    torch::Tensor visible;
    if(fused_candidates_ || preparation_frustum_) {
      auto candidates=preparation_frustum_?
          preparation_candidates(state_[0],pm,state_[1],state_[2],state_[3],state_[4],rows_,counts_,candidate_workspace_,anchor_partials_,preparation_frustum_):
          deployment_fused_candidates(distance,planes,state_[1],state_[2],state_[3],state_[4],rows_,counts_,candidate_workspace_,anchor_partials_);
      candidate_workspace_=candidates[1];visible=candidates[0];
    } else {
      visible=frustum_test(distance,planes,state_[1],state_[2]);
      auto candidates=visible_candidates(visible,state_[3],state_[4],rows_,counts_,candidate_workspace_);
      candidate_workspace_=candidates[2];
    }
    const int64_t capacity=use_ratio?int64_t(std::floor(requested_ratio*total_)):std::min(requested_budget,total_);
    buffers_[5]=torch::empty({capacity},rows_.options());
    buffers_[6]=torch::empty({capacity*width_},state_[0].options());
    if(fused_candidates_ || preparation_frustum_)deployment_frame_controls_fused(counts_,buffers_[0],buffers_[7],control_knots_,anchor_partials_,
        torch::Tensor(),requested_budget,requested_ratio,use_ratio);
    else deployment_frame_controls(counts_,buffers_[0],buffers_[7],control_knots_,requested_budget,requested_ratio,use_ratio);
    if(capacity) {
      auto center=(local && schema_!=1)?camera.contiguous():camera;
      TORCH_CHECK(!local || schema_==1 || (camera.device()==state_[0].device() && camera.scalar_type()==at::kFloat &&
          camera.numel()==3 && vm.device()==camera.device() && vm.scalar_type()==at::kFloat &&
          vm.sizes()==at::IntArrayRef({4,4})),"Frozen camera layout");
      deployment_device_forward(state_,selector_,compensator_,tables_,master_,base_,view_weight_,buffers_,
          vm,pm,center,double(width)/height,near,schema_,softplus_,local,compensation,
          compensation_tile_for(capacity,capacity>=1024),shrinking_radix_ && total_>=4096,radix_,selector_tiled_,
          saturated_selection_,torch::Tensor(),unroll_compensation_,fused_selection_,selector_prefix_,
          visible_view_?visible:torch::Tensor(),tensor_weights_,tensor_selector_,tensor_compensator_,tensor_warps_,tensor_comp_warps_,small_radix_finish_);
    }
    if(pending_requested_){
      pending_attributes_=buffers_[6];pending_ids_=buffers_[5];pending_frame_=buffers_[0];pending_capacity_=capacity;
      return Prepared{};
    }
    readback(buffers_[0],host_control_);
    const auto& frame=*reinterpret_cast<const DeploymentFrame*>(host_control_.data_ptr<uint8_t>());
    TORCH_CHECK(frame.status!=1,"Nonfinite deployment Selector logits");
    TORCH_CHECK(!frame.status,"Nonfinite deployment compensated attributes");
    Prepared output;output.device_counts=true;output.n=frame.n;output.visible=frame.visible;output.available=frame.available;
    output.graph_count=graphs_.size();
    output.shrinking_radix=frame.k && shrinking_radix_ && total_>=4096;
    output.compensation_tile=compensation?compensation_tile_for(capacity,capacity>=1024):0;
    output.selector_tiled=selector_tiled_ && frame.k;
    output.budget=frame.budget;output.k=frame.k;output.ratio=frame.ratio;output.has_ratio=frame.has_ratio;
    TORCH_CHECK(output.k<=capacity,"Deployment output capacity");
    output.ids=buffers_[5].slice(0,0,output.k);auto flat=buffers_[6];
    int64_t offset=0;
    for(auto shape:shapes_) {
      shape[0]=output.k;int64_t length=1;for(auto dim:shape)length*=dim;
      output.parts.push_back(flat.slice(0,offset,offset+length).view(shape));offset+=length;
    }
    output.ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    return output;
  }

  Prepared prepare_host(const torch::Tensor& vm,const torch::Tensor& pm,const torch::Tensor& camera,
      int64_t width,int64_t height,double near,int64_t requested_budget,double requested_ratio,
      bool use_ratio,bool local,bool compensation) {
    TORCH_CHECK(!torch::GradMode::is_enabled(),"Deployment pipeline is inference-only");
    TORCH_CHECK(width>0 && height>0 && near>0 && requested_budget>=0 &&
        (!use_ratio || (std::isfinite(requested_ratio) && requested_ratio>=0 && requested_ratio<=1)),"Invalid deployment request");
    TORCH_CHECK(!local || schema_,"Checkpoint has no view-conditioned branch");
    c10::cuda::CUDAGuard guard(state_[0].device());
    const auto start=std::chrono::steady_clock::now();
    torch::Tensor visible;
    if(preparation_frustum_){
      auto candidates=preparation_candidates(state_[0],pm,state_[1],state_[2],state_[3],state_[4],rows_,counts_,candidate_workspace_,anchor_partials_,preparation_frustum_);
      visible=candidates[0];candidate_workspace_=candidates[1];deployment_finish_counts(anchor_partials_,counts_);
    }else{
      auto planes=frustum_planes(pm);auto distance=at::matmul(state_[0],planes.slice(1,0,3).t());
      visible=frustum_test(distance,planes,state_[1],state_[2]);
      auto candidates=visible_candidates(visible,state_[3],state_[4],rows_,counts_,candidate_workspace_);
      candidate_workspace_=candidates[2];
    }
    readback(counts_,host_counts_);
    const auto counts=host_counts_.data_ptr<int64_t>();
    Prepared output;output.n=counts[0];output.visible=counts[1];output.available=counts[2];
    output.budget=use_ratio?int64_t(std::floor(requested_ratio*output.n)):requested_budget;
    output.has_ratio=output.n && (use_ratio?requested_ratio>0:requested_budget>0);
    output.ratio=output.has_ratio?(use_ratio?requested_ratio:std::min(1.,double(requested_budget)/output.n)):0.;
    output.k=std::min(output.budget,output.available);
    auto rows=rows_.slice(0,0,output.n);torch::Tensor flat;
    if(output.k) {
      auto code=empty_;
      if(local && schema_!=1) {
        if(visible_view_)code=deployment_view_masked(state_[0],state_[10],state_[11],camera.contiguous(),vm,pm,
            state_[12],double(width)/height,near,visible);
        else code=deployment_view(state_[0],state_[10],state_[11],camera.contiguous(),vm,pm,
            state_[12],double(width)/height,near);
      }
      auto scores=tensor_selector_?deployment_tensor_selector(state_[6],rows,state_[3],code,state_[9],selector_,output.ratio,local,schema_,selector_prefix_,tensor_weights_,tensor_selector_,tensor_warps_):
          deployment_selector_impl(state_[6],rows,state_[3],code,state_[9],selector_,output.ratio,local,schema_,selector_tiled_,selector_prefix_);
      output.selector_tiled=selector_tiled_;
      if(!compensation) {
        readback(scores[2],host_status_);
        TORCH_CHECK(!host_status_.data_ptr<int>()[0],"Nonfinite deployment Selector logits");
      }
      output.shrinking_radix=shrinking_radix_ && output.n>=4096;
      output.ids=output.shrinking_radix?
          deployment_radix_select(scores[0],rows,state_[5],output.k,radix_,scores[2]):
          deployment_select_checked(scores[0],rows,state_[5],output.k,select_workspace_,sort_,scores[2]);
      if(compensation) {
        std::vector<double> weights(knots_.size(),-1.);weights[0]=1.;
        for(size_t j=1;j<knots_.size();++j) {
          if(output.ratio<=knots_[j-1]+1e-8)break;
          weights[j]=std::min(1.,(output.ratio-knots_[j-1])/(knots_[j]-knots_[j-1]));
        }
        const double t=2*(std::log(output.ratio)-std::log(bounds_[0]))/(std::log(bounds_[1])-std::log(bounds_[0]))-1;
        std::vector<double> ratio={t,std::sin(M_PI*t/2),std::cos(M_PI*t/2)};
        output.compensation_tile=compensation_tile_for(output.k,output.k>=1024);
        auto result=tensor_compensator_?deployment_tensor_compensate(state_[7],output.ids,state_[3],state_[8],tables_,compensator_,master_,code,
            view_weight_,weights,ratio,local&&schema_,schema_,softplus_,tensor_weights_,tensor_compensator_,tensor_comp_warps_,scores[2]):
            deployment_compensate_impl(state_[7],output.ids,state_[3],state_[8],tables_,compensator_,master_,code,
            view_weight_,weights,ratio,local&&schema_,schema_,softplus_,scores[2],output.compensation_tile);
        readback(result[2],host_status_);
        TORCH_CHECK(host_status_.data_ptr<int>()[0]!=1,"Nonfinite deployment Selector logits");
        TORCH_CHECK(!host_status_.data_ptr<int>()[0],"Nonfinite deployment compensated attributes");
        flat=result[0];
      } else flat=publish_selected(base_,output.ids,empty_,empty_,true);
    } else {
      output.ids=rows.slice(0,0,0);
      flat=publish_selected(base_,output.ids,empty_,empty_,true);
    }
    int64_t offset=0;
    for(auto shape:shapes_) {
      shape[0]=output.k;int64_t length=1;for(auto dim:shape)length*=dim;
      output.parts.push_back(flat.slice(0,offset,offset+length).view(shape));offset+=length;
    }
    output.ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    return output;
  }

  Prepared prepare_native(const torch::Tensor& vm,const torch::Tensor& pm,const torch::Tensor& camera,
      int64_t width,int64_t height,double near,int64_t budget,double ratio,bool use_ratio,bool local,bool compensation) {
    TORCH_CHECK(!pending_inflight_,"Finish pending frame before reusing this pipeline");
    // Route before Graph validation/device guards to retain native request cost.
    const bool graph_capacity=!compensation || (use_ratio?
        (std::isfinite(ratio) && ratio*double(total_)<32769.):std::min(budget,total_)<=32768);
    if(graph_enabled_ && graph_capacity && device_counts_ && (total_<=16384 || shrinking_radix_) && total_)
      return prepare_graph(vm,pm,camera,width,height,near,budget,ratio,use_ratio,local,compensation);
    if(device_counts_ && (total_<=16384 || shrinking_radix_))return prepare_device(vm,pm,camera,width,height,near,budget,ratio,use_ratio,local,compensation);
    return prepare_host(vm,pm,camera,width,height,near,budget,ratio,use_ratio,local,compensation);
  }

  std::vector<torch::Tensor> begin_pending(const torch::Tensor& vm,const torch::Tensor& pm,const torch::Tensor& camera,
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
  pybind11::tuple render_pending_current(pybind11::capsule handle,const torch::Tensor& bg,
      const torch::Tensor& vm,const torch::Tensor& pm,const torch::Tensor& camera,
      int64_t width,int64_t height,double near,int64_t budget,double ratio,bool use_ratio,
      bool local,bool compensation,int64_t rest,double tanx,double tany){
    namespace py=pybind11;
    void* raster=handle.get_pointer();
    TORCH_CHECK(std::string(handle.name())=="mycartgs.student-tile-clear.shared.v1","Pending renderer handle schema");
    std::vector<torch::Tensor> output;
    {
      py::gil_scoped_release release;
      auto pending=begin_pending(vm,pm,camera,width,height,near,budget,ratio,use_ratio,local,compensation);
      output=render_pending_shared(raster,bg,pending[0],pending[2],pending[1].numel(),rest,
          vm,pm,tanx,tany,height,width,sh_degree_,camera);
    }
    auto finished=finish_pending();
    auto ids=finished[0].cast<torch::Tensor>();
    return py::make_tuple(output[0],output[1].slice(0,0,ids.numel()),ids,finished[1]);
  }
  pybind11::tuple render_pose_current(pybind11::capsule handle,const torch::Tensor& bg,
      pybind11::array_t<double,pybind11::array::c_style|pybind11::array::forcecast> pose,
      const torch::Tensor& projection,const torch::Tensor& device_reference,
      int64_t width,int64_t height,int64_t budget,double ratio,bool use_ratio,int64_t rest,double tanx,double tany){
    namespace py=pybind11;
    TORCH_CHECK(pose.ndim()==2&&pose.shape(0)==4&&pose.shape(1)==4,"Current CPU camera pose must be 4x4");
    auto input=torch::from_blob(const_cast<double*>(pose.data()),{4,4},torch::TensorOptions().dtype(at::kDouble));
    std::vector<torch::Tensor> camera;
    {
      py::gil_scoped_release release;
      camera=preparation_camera_parts(input,projection,device_reference);
    }
    return render_pending_current(handle,bg,camera[0],camera[1],camera[2],width,height,.01,
        budget,ratio,use_ratio,true,true,rest,tanx,tany);
  }
  pybind11::tuple render_pose_reused(pybind11::capsule handle,const torch::Tensor& bg,
      pybind11::array_t<double,pybind11::array::c_style|pybind11::array::forcecast> pose,
      const torch::Tensor& projection,const torch::Tensor& device_reference,
      int64_t width,int64_t height,int64_t budget,double ratio,bool use_ratio,int64_t rest,double tanx,double tany){
    namespace py=pybind11;
    TORCH_CHECK(pose.ndim()==2&&pose.shape(0)==4&&pose.shape(1)==4,"Current CPU camera pose must be 4x4");
    TORCH_CHECK(!pending_requested_&&!pending_inflight_,"Finish pending frame before reusing camera staging");
    c10::cuda::CUDAGuard guard(state_[0].device());
    auto stream=c10::cuda::getCurrentCUDAStream();
    TORCH_CHECK(!pose_host_.defined()||pose_stream_==stream.stream(),"Use one reused camera pipeline per stream");
    if(!pose_host_.defined()){
      pose_host_=torch::empty({35},torch::TensorOptions().dtype(at::kFloat).device(at::kCPU).pinned_memory(true));
      pose_device_=torch::empty({35},state_[0].options());pose_stream_=stream.stream();
      pose_parts_={pose_device_.slice(0,0,16).view({4,4}),pose_device_.slice(0,16,32).view({4,4}),pose_device_.slice(0,32,35)};
    }
    {py::gil_scoped_release release;preparation_camera_into(pose.data(),projection,pose_host_,pose_device_);}
    return render_pending_current(handle,bg,pose_parts_[0],pose_parts_[1],pose_parts_[2],width,height,.01,
        budget,ratio,use_ratio,true,true,rest,tanx,tany);
  }
  void small_radix_finish(bool enabled){graphs_.clear();small_radix_finish_=enabled;}
  torch::Tensor debug_control() const {TORCH_CHECK(device_counts_,"Map uses the retained C++ count path");return buffers_[0].clone();}
  void tuning(bool radix,int64_t tile,bool selector_tiled) {
    TORCH_CHECK(tile==0 || tile==4 || tile==8 || tile==16,"Deployment tile tuning");
    graphs_.clear();
    shrinking_radix_=radix;compensation_tile_=tile;selector_tiled_=selector_tiled;
  }

  void optimization_controls(bool saturated_selection) {
    graphs_.clear();
    saturated_selection_=saturated_selection;
  }

  void graph_enabled(bool enabled) {
    graph_enabled_=enabled;
    if(!enabled)graphs_.clear();
  }

  void kernel_options(bool unroll_compensation,bool fused_selection,bool fused_candidates) {
    graphs_.clear();unroll_compensation_=unroll_compensation;
    fused_selection_=fused_selection;fused_candidates_=fused_candidates;
  }

  void static_options(bool prefix,bool visible_view,bool small_tile) {
    TORCH_CHECK(prefix||!tensor_selector_,"Disable Tensor Selector before disabling frozen prefix");
    c10::cuda::CUDAGuard guard(state_[0].device());
    graphs_.clear();
    if(prefix && !selector_prefix_.defined())selector_prefix_=deployment_selector_prefix(state_[6],selector_[0]);
    if(!prefix)selector_prefix_=torch::Tensor();
    visible_view_=visible_view;small_compensation_tile_=small_tile;
  }

  void tensor_options(int64_t selector_mode,int64_t compensator_mode,int64_t warps,int64_t comp_warps=0) {
    if(!comp_warps)comp_warps=warps;
    TORCH_CHECK(comp_warps==1||comp_warps==2||comp_warps==4,"Compensator Tensor warps must be 1,2 or 4");
    TORCH_CHECK(selector_mode>=0&&selector_mode<=2&&compensator_mode>=0&&compensator_mode<=9,"Tensor mode: 0=CUDA,1=TF32,2=FP16");
    TORCH_CHECK(warps==1||warps==2||warps==4,"Tensor warps must be 1,2 or 4");
    c10::cuda::CUDAGuard guard(state_[0].device());graphs_.clear();
    TORCH_CHECK(!selector_mode||selector_prefix_.defined(),"Tensor Selector requires frozen static prefix");
    tensor_weights_=deployment_tensor_pack(selector_,compensator_,view_weight_,selector_mode,compensator_mode);
    tensor_selector_=selector_mode;tensor_compensator_=compensator_mode;tensor_warps_=warps;tensor_comp_warps_=comp_warps;
    if(preparation_projection_ && tensor_compensator_)
      preparation_options(preparation_frustum_,preparation_projection_,preparation_device_);
  }
  int64_t tensor_bytes() const {int64_t n=0;for(const auto& x:tensor_weights_)if(x.defined())n+=x.nbytes();return n;}
  int64_t prefix_bytes() const {return selector_prefix_.defined()?selector_prefix_.nbytes():0;}

  void preparation_options(int64_t frustum,int64_t projection,int64_t device) {
    TORCH_CHECK(frustum>=0&&frustum<=1&&projection>=0&&projection<=1&&device>=0&&device<=1,"Preparation option range");
    TORCH_CHECK(!projection||tensor_compensator_,"Projected input requires Tensor Compensation");
    c10::cuda::CUDAGuard guard(state_[0].device());graphs_.clear();
    preparation_frustum_=frustum;preparation_projection_=projection;preparation_device_=device;
    device_counts_=total_<=65536 || device;
    tensor_weights_.resize(projection?8:5);
    if(projection){
      TORCH_CHECK(compensator_.size()==13,"Projected input requires frozen 32/32 student");
      auto feature_weight=compensator_[0].slice(0,3,51);
      auto base_projection=at::matmul(tables_[0],feature_weight);
      tensor_weights_[5]=(at::matmul(state_[7],compensator_[0].slice(0,0,3))+
          base_projection.index_select(0,state_[3])).contiguous();
      std::vector<torch::Tensor> residuals;
      for(size_t j=1;j<tables_.size();++j)residuals.push_back(at::matmul(tables_[j],feature_weight));
      tensor_weights_[6]=torch::stack(residuals).contiguous();
      tensor_weights_[7]=torch::empty({state_[0].size(0),32},state_[0].options());
      TORCH_CHECK(at::isfinite(tensor_weights_[5]).all().item<bool>()&&
          at::isfinite(tensor_weights_[6]).all().item<bool>(),"Nonfinite projected frozen input");
    }
  }
  std::vector<torch::Tensor> tensor_state() const {return tensor_weights_;}
  int64_t preparation_bytes() const {
    int64_t bytes=0;for(size_t j=5;j<tensor_weights_.size();++j)bytes+=tensor_weights_[j].nbytes();return bytes;
  }

  pybind11::tuple prepare(const torch::Tensor& vm,const torch::Tensor& pm,const torch::Tensor& camera,
      int64_t width,int64_t height,double near,int64_t budget,double ratio,bool use_ratio,bool local,bool compensation) {
    namespace py=pybind11;
    Prepared result;
    {py::gil_scoped_release release;result=prepare_native(vm,pm,camera,width,height,near,budget,ratio,use_ratio,local,compensation);}
    py::dict inputs,packet,stats;
    const char* names[] = {"means3D","opacities","scales","rotations","dc","shs","means2D"};
    for(int j=0;j<7;++j)inputs[names[j]]=result.parts[j];
    inputs["colors_precomp"]=py::none();inputs["cov3D_precomp"]=py::none();
    packet["sh_degree"]=sh_degree_;packet["inputs"]=inputs;
    stats["total_gaussians"]=total_;stats["visible_anchors"]=result.visible;stats["candidate_gaussians"]=result.n;
    stats["eligible_candidates"]=result.available;stats["requested_budget"]=result.budget;
    stats["effective_budget"]=result.k;stats["active_gaussians"]=result.k;
    stats["ratio"]=result.has_ratio?py::cast(result.ratio):py::none();
    stats["budget_shortfall"]=std::max(int64_t(0),result.budget-result.k);stats["inference_backend"]="deployment_cpp";
    stats["device_counts"]=result.device_counts;stats["prepare_ms"]=result.ms;stats["cache_hit"]=false;stats["view_conditioned"]=local;stats["compensation_enabled"]=compensation;
    stats["shrinking_radix"]=result.shrinking_radix;stats["compensation_tile"]=result.compensation_tile;
    stats["selector_tiled"]=result.selector_tiled;
    stats["graph_replay"]=result.graph_replay;
    stats["graph_count"]=result.graph_count;
    stats["unroll_compensation"]=result.compensation_tile==8208 ||
        (result.device_counts && unroll_compensation_ && result.compensation_tile==16);
    stats["fused_selection"]=result.device_counts && fused_selection_ && result.shrinking_radix && total_>=16384;
    stats["fused_candidates"]=result.device_counts && fused_candidates_;
    return py::make_tuple(packet,result.ids,stats);
  }
};

void bind_deployment_prepare(pybind11::module_& module) {
  namespace py=pybind11;
  py::class_<TensorDeploymentPipeline>(module,"Pipeline",py::module_local())
    .def(py::init<std::vector<torch::Tensor>,std::vector<torch::Tensor>,std::vector<torch::Tensor>,
        std::vector<torch::Tensor>,std::vector<torch::Tensor>,std::vector<torch::Tensor>,torch::Tensor,
        std::vector<double>,std::vector<double>,int64_t,int64_t,bool>())
    .def("prepare",&TensorDeploymentPipeline::prepare)
    .def("_begin_pending",&TensorDeploymentPipeline::begin_pending,py::call_guard<py::gil_scoped_release>())
    .def("_finish_pending",&TensorDeploymentPipeline::finish_pending)
    .def("_render_pending_current",&TensorDeploymentPipeline::render_pending_current)
    .def("_render_pose_current",&TensorDeploymentPipeline::render_pose_current)
    .def("_render_pose_reused",&TensorDeploymentPipeline::render_pose_reused)
    .def("_tuning",&TensorDeploymentPipeline::tuning,py::arg("radix"),py::arg("tile"),py::arg("selector_tiled")=true)
    .def("_optimization_controls",&TensorDeploymentPipeline::optimization_controls)
    .def("_graph_enabled",&TensorDeploymentPipeline::graph_enabled)
    .def("_kernel_options",&TensorDeploymentPipeline::kernel_options)
    .def("_static_options",&TensorDeploymentPipeline::static_options)
    .def("_tensor_options",&TensorDeploymentPipeline::tensor_options,py::arg("selector"),py::arg("compensator"),py::arg("warps"),py::arg("comp_warps")=0)
    .def("_tensor_bytes",&TensorDeploymentPipeline::tensor_bytes)
    .def("_preparation_options",&TensorDeploymentPipeline::preparation_options)
    .def("_preparation_bytes",&TensorDeploymentPipeline::preparation_bytes)
    .def("_tensor_state",&TensorDeploymentPipeline::tensor_state)
    .def("_prefix_bytes",&TensorDeploymentPipeline::prefix_bytes)
    .def("_control",&TensorDeploymentPipeline::debug_control)
    .def("_small_radix_finish",&TensorDeploymentPipeline::small_radix_finish);
}
