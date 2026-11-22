#include <torch/extension.h>
#include <ATen/Functions.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <chrono>
#include <cmath>

torch::Tensor frustum_planes(const torch::Tensor&);
torch::Tensor frustum_test(const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,const torch::Tensor&);
std::vector<torch::Tensor> visible_candidates(const torch::Tensor&,const torch::Tensor&,
    const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,const torch::Tensor&);
torch::Tensor deployment_select_checked(const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,
    int64_t,const torch::Tensor&,const std::vector<torch::Tensor>&,const torch::Tensor&);
torch::Tensor publish_selected(const std::vector<torch::Tensor>&,const torch::Tensor&,
    const torch::Tensor&,const torch::Tensor&,bool);
torch::Tensor deployment_view(const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,
    const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,double,double);
std::vector<torch::Tensor> deployment_selector(const torch::Tensor&,const torch::Tensor&,const torch::Tensor&,
    const torch::Tensor&,const torch::Tensor&,const std::vector<torch::Tensor>&,double,bool,int64_t);
std::vector<torch::Tensor> deployment_compensate_checked(const torch::Tensor&,const torch::Tensor&,
    const torch::Tensor&,const torch::Tensor&,const std::vector<torch::Tensor>&,
    const std::vector<torch::Tensor>&,const std::vector<torch::Tensor>&,const torch::Tensor&,
    const torch::Tensor&,const std::vector<double>&,const std::vector<double>&,bool,int64_t,bool,const torch::Tensor&);

namespace {
struct Prepared {
  std::vector<torch::Tensor> parts;
  torch::Tensor ids;
  int64_t n,visible,available,budget,k;
  double ratio,ms;
  bool has_ratio;
};
}

// Per-frame scheduling is native C++; no Python callback or result cache is used.
class DeploymentPipeline {
  std::vector<torch::Tensor> state_,selector_,compensator_,tables_,master_,base_,sort_;
  torch::Tensor view_weight_,rows_,counts_,candidate_workspace_,host_counts_,host_status_,empty_,select_workspace_;
  std::vector<double> knots_,bounds_;
  std::vector<std::vector<int64_t>> shapes_;
  int64_t schema_,sh_degree_,total_;
  bool softplus_;

  void readback(const torch::Tensor& from,const torch::Tensor& to) {
    const auto stream=c10::cuda::getCurrentCUDAStream();
    C10_CUDA_CHECK(cudaMemcpyAsync(to.data_ptr(),from.data_ptr(),from.nbytes(),cudaMemcpyDeviceToHost,stream));
    C10_CUDA_CHECK(cudaStreamSynchronize(stream));
  }
public:
  DeploymentPipeline(std::vector<torch::Tensor> state,std::vector<torch::Tensor> selector,
      std::vector<torch::Tensor> compensator,std::vector<torch::Tensor> tables,
      std::vector<torch::Tensor> master,std::vector<torch::Tensor> base,torch::Tensor view_weight,
      std::vector<double> knots,std::vector<double> bounds,int64_t schema,int64_t degree,bool softplus):
      state_(std::move(state)),selector_(std::move(selector)),compensator_(std::move(compensator)),
      tables_(std::move(tables)),master_(std::move(master)),base_(std::move(base)),view_weight_(view_weight),
      knots_(std::move(knots)),bounds_(std::move(bounds)),schema_(schema),sh_degree_(degree),softplus_(softplus) {
    TORCH_CHECK(state_.size()==13 && bounds_.size()==2 && knots_.size()==tables_.size(),"Frozen pipeline state");
    total_=state_[3].numel();
    c10::cuda::CUDAGuard guard(state_[3].device());
    rows_=torch::empty_like(state_[3]);counts_=torch::empty({3},state_[3].options());
    candidate_workspace_=torch::empty({0},state_[3].options().dtype(at::kByte));
    empty_=torch::empty({0},state_[0].options());
    host_counts_=torch::empty({3},torch::TensorOptions().dtype(at::kLong).device(at::kCPU).pinned_memory(true));
    host_status_=torch::empty({1},torch::TensorOptions().dtype(at::kInt).device(at::kCPU).pinned_memory(true));
    select_workspace_=torch::empty({2+2*((std::min(total_,int64_t(16384))+255)/256)},state_[0].options().dtype(at::kInt));
    if(total_>16384) {
      for(auto dtype:{at::kFloat,at::kFloat,at::kLong,at::kLong})sort_.push_back(torch::empty({total_},state_[0].options().dtype(dtype)));
      sort_.push_back(torch::empty({32*total_+4096},state_[0].options().dtype(at::kByte)));
    }
    for(int j=0;j<6;++j)shapes_.push_back(base_[j].sizes().vec());
    shapes_.push_back({total_,3});
  }

  Prepared prepare_native(const torch::Tensor& vm,const torch::Tensor& pm,const torch::Tensor& camera,
      int64_t width,int64_t height,double near,int64_t requested_budget,double requested_ratio,
      bool use_ratio,bool local,bool compensation) {
    TORCH_CHECK(!torch::GradMode::is_enabled(),"Deployment pipeline is inference-only");
    TORCH_CHECK(width>0 && height>0 && near>0 && requested_budget>=0 &&
        (!use_ratio || (std::isfinite(requested_ratio) && requested_ratio>=0 && requested_ratio<=1)),"Invalid deployment request");
    TORCH_CHECK(!local || schema_,"Checkpoint has no view-conditioned branch");
    c10::cuda::CUDAGuard guard(state_[0].device());
    const auto start=std::chrono::steady_clock::now();
    auto planes=frustum_planes(pm);
    auto distance=at::matmul(state_[0],planes.slice(1,0,3).t());
    auto visible=frustum_test(distance,planes,state_[1],state_[2]);
    auto candidates=visible_candidates(visible,state_[3],state_[4],rows_,counts_,candidate_workspace_);
    candidate_workspace_=candidates[2];readback(counts_,host_counts_);
    const auto counts=host_counts_.data_ptr<int64_t>();
    Prepared output;output.n=counts[0];output.visible=counts[1];output.available=counts[2];
    output.budget=use_ratio?int64_t(std::floor(requested_ratio*output.n)):requested_budget;
    output.has_ratio=output.n && (use_ratio?requested_ratio>0:requested_budget>0);
    output.ratio=output.has_ratio?(use_ratio?requested_ratio:std::min(1.,double(requested_budget)/output.n)):0.;
    output.k=std::min(output.budget,output.available);
    auto rows=rows_.slice(0,0,output.n);torch::Tensor flat;
    if(output.k) {
      auto code=empty_;
      if(local && schema_!=1)code=deployment_view(state_[0],state_[10],state_[11],camera.contiguous(),vm,pm,
          state_[12],double(width)/height,near);
      auto scores=deployment_selector(state_[6],rows,state_[3],code,state_[9],selector_,output.ratio,local,schema_);
      if(!compensation) {
        readback(scores[2],host_status_);
        TORCH_CHECK(!host_status_.data_ptr<int>()[0],"Nonfinite deployment Selector logits");
      }
      output.ids=deployment_select_checked(scores[0],rows,state_[5],output.k,select_workspace_,sort_,scores[2]);
      if(compensation) {
        std::vector<double> weights(knots_.size(),-1.);weights[0]=1.;
        for(size_t j=1;j<knots_.size();++j) {
          if(output.ratio<=knots_[j-1]+1e-8)break;
          weights[j]=std::min(1.,(output.ratio-knots_[j-1])/(knots_[j]-knots_[j-1]));
        }
        const double t=2*(std::log(output.ratio)-std::log(bounds_[0]))/(std::log(bounds_[1])-std::log(bounds_[0]))-1;
        std::vector<double> ratio={t,std::sin(M_PI*t/2),std::cos(M_PI*t/2)};
        auto result=deployment_compensate_checked(state_[7],output.ids,state_[3],state_[8],tables_,compensator_,master_,code,
            view_weight_,weights,ratio,local&&schema_,schema_,softplus_,scores[2]);
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
    stats["prepare_ms"]=result.ms;stats["cache_hit"]=false;stats["view_conditioned"]=local;stats["compensation_enabled"]=compensation;
    return py::make_tuple(packet,result.ids,stats);
  }
};

void bind_deployment_prepare(pybind11::module_& module) {
  namespace py=pybind11;
  py::class_<DeploymentPipeline>(module,"Pipeline",py::module_local())
    .def(py::init<std::vector<torch::Tensor>,std::vector<torch::Tensor>,std::vector<torch::Tensor>,
        std::vector<torch::Tensor>,std::vector<torch::Tensor>,std::vector<torch::Tensor>,torch::Tensor,
        std::vector<double>,std::vector<double>,int64_t,int64_t,bool>())
    .def("prepare",&DeploymentPipeline::prepare);
}
