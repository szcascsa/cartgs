#pragma once
#include <torch/torch.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>

namespace mixed_ratio {
struct Config {
  bool enabled=false;
  int seed=0;
  std::array<int,3> concentration{8,12,20};
  float eta_min=.3f, eta_max=.8f;
  void validate() const {
    if(seed<0 || !std::isfinite(eta_min) || !std::isfinite(eta_max) ||
        eta_min<=0 || eta_max<eta_min || eta_max>1 ||
        std::any_of(concentration.begin(),concentration.end(),[](int n){return n<=0||n>1000;}))
      throw std::invalid_argument("Invalid MixedRatio seed, concentration or sigma range");
  }
};

// Independent of Torch's Gumbel RNG; seed + committed LoD step determines a plan.
struct Random {
  std::uint64_t state;
  Random(std::uint64_t seed,std::uint64_t sample)
      : state(seed^((sample+1)*0xd1b54a32d192ed03ULL)) {}
  double uniform() {
    auto z=(state+=0x9e3779b97f4a7c15ULL);
    z=(z^(z>>30))*0xbf58476d1ce4e5b9ULL;
    z=(z^(z>>27))*0x94d049bb133111ebULL;
    z^=z>>31;
    return (double(z>>12)+.5)/4503599627370496.;
  }
};
struct Draw {
  std::array<double,3> shares{}, quantiles{};
  double sigma=0;
};
inline Draw draw(const Config& config,double target,double low,double high,std::uint64_t sample) {
  config.validate();
  if(!std::isfinite(target)||!(0<low&&low<high&&high<=1)||target<low||target>high)
    throw std::invalid_argument("MixedRatio target outside training endpoints");
  Random random(config.seed,sample);
  Draw result;
  double total=0;
  for(int i=0;i<3;++i) {
    for(int j=0;j<config.concentration[i];++j)result.shares[i]-=std::log(random.uniform());
    total+=result.shares[i];
  }
  for(auto& p:result.shares)p/=total;
  result.sigma=(config.eta_min+(config.eta_max-config.eta_min)*random.uniform())*
      std::min(target-low,high-target);
  for(auto& q:result.quantiles)q=random.uniform();
  std::sort(result.quantiles.begin(),result.quantiles.end(),std::greater<double>());
  return result;
}
// Stable interval mass, including both tails; not a clamped normal sample.
inline double normalMass(double a,double b) {
  const double root=std::sqrt(2.);
  if(a>=0)return .5*(std::erfc(a/root)-std::erfc(b/root));
  if(b<=0)return .5*(std::erfc(-b/root)-std::erfc(-a/root));
  return 1.-.5*std::erfc(b/root)-.5*std::erfc(-a/root);
}
inline double quantile(double mu,double sigma,double q,double low,double high) {
  double left=low,right=high;
  const double a=(low-mu)/sigma, mass=normalMass(a,(high-mu)/sigma);
  for(int j=0;j<56;++j) {
    const double mid=(left+right)*.5;
    if(normalMass(a,(mid-mu)/sigma)<q*mass)left=mid;else right=mid;
  }
  return (left+right)*.5;
}
inline std::array<double,3> ratios(const Draw& draw,const std::array<int64_t,3>& counts,
    double target,double low,double high) {
  std::array<double,3> result{target,target,target};
  const double n=double(counts[0])+counts[1]+counts[2];
  if(draw.sigma<=0||!n)return result;
  double left=low-32*draw.sigma,right=high+32*draw.sigma;
  for(int j=0;j<64;++j) {
    const double mu=(left+right)*.5;
    double mean=0;
    for(int i=0;i<3;++i) {
      result[i]=quantile(mu,draw.sigma,draw.quantiles[i],low,high);
      mean+=counts[i]/n*result[i];
    }
    if(mean<target)left=mu;else right=mu;
  }
  return result;
}

inline torch::Tensor projectedLongestSide(const torch::Tensor& centers,const torch::Tensor& scales,
    const torch::Tensor& projection,const torch::Tensor& view,int height,int width,float near_plane) {
  torch::NoGradGuard guard;
  const auto signs=torch::tensor({{-1.f,-1.f,-1.f},{-1.f,-1.f,1.f},{-1.f,1.f,-1.f},{-1.f,1.f,1.f},
      {1.f,-1.f,-1.f},{1.f,-1.f,1.f},{1.f,1.f,-1.f},{1.f,1.f,1.f}},centers.options());
  auto corners=centers.unsqueeze(1)+scales.unsqueeze(1)*.5*signs;
  auto homogeneous=torch::cat({corners,torch::ones_like(corners.slice(2,0,1))},2);
  auto clip=homogeneous.matmul(projection),camera=homogeneous.matmul(view);
  auto near=std::get<0>(camera.select(2,2).min(1))<=near_plane;
  auto behind=std::get<0>(camera.select(2,2).max(1))<=near_plane;
  auto ndc=clip.slice(2,0,2)/clip.slice(2,3,4).clamp_min(std::numeric_limits<float>::min());
  auto extent=std::get<0>(ndc.max(1))-std::get<0>(ndc.min(1));
  auto pixels=extent*torch::tensor({float(width),float(height)},centers.options())*.5;
  auto size=std::get<0>(pixels.max(1))/height;
  // All Anchors remain candidates. Fully behind-near cells rank last; only
  // cells crossing the near plane need conservative infinite projected size.
  size=torch::where(near,torch::full_like(size,std::numeric_limits<float>::infinity()),size);
  return torch::where(behind,torch::zeros_like(size),size);
}

inline torch::Tensor levels(const torch::Tensor& metric,const torch::Tensor& counts,
    const torch::Tensor& uids,const std::array<double,3>& shares) {
  torch::NoGradGuard guard;
  auto order=at::argsort(uids,true,0,false);
  order=order.index_select(0,at::argsort(metric.index_select(0,order),true,0,true));
  auto weight=counts.index_select(0,order).to(torch::kFloat64);
  auto position=(weight.cumsum(0)-.5*weight)/weight.sum().clamp_min(1);
  auto level=(position>=shares[0]).to(torch::kLong)+(position>=shares[0]+shares[1]).to(torch::kLong);
  return torch::empty_like(order).index_copy_(0,order,level);
}
} // namespace mixed_ratio
