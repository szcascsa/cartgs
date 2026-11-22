#pragma once
#include <torch/torch.h>
#ifdef CARTGS_FUSED_MAPPING_CUDA
#include <nvToolsExt.h>
#include <cstdlib>
#endif

namespace mapping_cuda {
torch::Tensor anchor128Pack(const torch::Tensor& xyz,const torch::Tensor& center,
    const torch::Tensor& reference,const torch::Tensor& feature,const torch::Tensor& tanh_gate,
    const torch::Tensor& mean,const torch::Tensor& stddev);
std::vector<torch::Tensor> anchor128PackForward(const std::vector<torch::Tensor>& a);
std::vector<torch::Tensor> anchor128PackBackward(const std::vector<torch::Tensor>& a,
    const torch::Tensor& saved,const torch::Tensor& gradient);
torch::Tensor maturityLossFused(const torch::Tensor& weights,const torch::Tensor& maturity,
    const torch::Tensor& eligible,float delta);
std::vector<torch::Tensor> maturityPrepare(const torch::Tensor& weights,const torch::Tensor& maturity,
    const torch::Tensor& eligible);
std::vector<torch::Tensor> maturityFinish(const std::vector<torch::Tensor>& sums,float delta);
torch::Tensor maturityLossVjp(const torch::Tensor& maturity,const torch::Tensor& eligible,
    const torch::Tensor& saved,const torch::Tensor& gradient);
torch::Tensor fusedL1Loss(const torch::Tensor& x,const torch::Tensor& y);
std::vector<torch::Tensor> l1Prepare(const torch::Tensor& x,const torch::Tensor& y);
std::vector<torch::Tensor> l1Vjp(const torch::Tensor& difference,const torch::Tensor& gradient,bool target_gradient);
void addZeroTfStatistics(torch::Tensor& sums,const torch::TensorOptions& options,
    int64_t selected,int64_t candidates,int64_t gi_count,int64_t selector_candidates,int64_t population);
bool l1FusionEnabled();
void setL1FusionEnabled(bool value);
class L1FusionScope {
  bool previous_;
 public:
  explicit L1FusionScope(bool value):previous_(l1FusionEnabled()){setL1FusionEnabled(value);}
  ~L1FusionScope(){setL1FusionEnabled(previous_);}
};
// Only labels host launch intervals; no CUDA synchronization or timing events.
class FusionRange {
  bool active_=false;
 public:
  explicit FusionRange(const char* name) {
#ifdef CARTGS_FUSED_MAPPING_CUDA
    static const bool enabled=std::getenv("CARTGS_FUSION_PROFILE")!=nullptr;
    active_=enabled;if(active_)nvtxRangePushA(name);
#endif
  }
  ~FusionRange(){
#ifdef CARTGS_FUSED_MAPPING_CUDA
    if(active_)nvtxRangePop();
#endif
  }
};
}
