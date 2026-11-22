#pragma once
#include <torch/torch.h>

// Ordered PL sample log probability. O(N+K) storage/work, FP64, no safety clamp.
struct PLProbability {torch::Tensor log_prob, denominator, weights;};
inline PLProbability plLogProbability(const torch::Tensor& z,const torch::Tensor& order){
  TORCH_CHECK(z.dim()==1&&z.scalar_type()==torch::kFloat64,"PL requires vector FP64 logits");
  TORCH_CHECK(order.dim()==1&&order.numel()>0&&order.numel()<=z.numel(),"Invalid ordered sample");
  TORCH_CHECK(torch::isfinite(z).all().item<bool>(),"PL_NONFINITE_LOGIT");
  auto shift=z.max().detach();auto w=(z-shift).exp();
  TORCH_CHECK(torch::isfinite(w).all().item<bool>()&&(w>0).all().item<bool>(),"PL_WEIGHT_OVERFLOW_OR_UNDERFLOW");
  auto selected=w.index_select(0,order);
  auto previous=torch::cat({torch::zeros({1},w.options()),selected.cumsum(0).slice(0,0,order.numel()-1)});
  auto denominator=w.sum()-previous;
  TORCH_CHECK(torch::isfinite(denominator).all().item<bool>()&&(denominator>0).all().item<bool>(),"PL_NONPOSITIVE_DENOMINATOR");
  auto log_prob=(z.index_select(0,order)-shift-denominator.log()).sum();
  TORCH_CHECK(torch::isfinite(log_prob).item<bool>(),"PL_NONFINITE_LOG_PROB");
  return {log_prob,denominator,w};
}
inline std::tuple<torch::Tensor,torch::Tensor> sampleFreshGumbelTopK(const torch::Tensor& z,int64_t K){
  torch::NoGradGuard guard;
  auto u=torch::rand_like(z);
  TORCH_CHECK((u>0).all().item<bool>()&&(u<1).all().item<bool>(),"GUMBEL_UNIFORM_ENDPOINT");
  auto g=-(-u.log()).log();auto order=std::get<1>((z.detach()+g).topk(K,0,true,true));
  TORCH_CHECK(order.numel()==K,"GUMBEL_NOT_EXACT_K");
  return {order,g};
}
