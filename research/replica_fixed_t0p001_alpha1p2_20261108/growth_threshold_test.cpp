#include "include/gaussian_model.h"
#include <cmath>
#include <iostream>
#include <limits>

int main(int argc,char** argv) {
  const auto device=argc>1&&std::string(argv[1])=="cuda"?torch::kCUDA:torch::kCPU;
  const auto options=torch::TensorOptions().device(device).dtype(torch::kFloat32);
  GaussianModel policy(3);
  TORCH_CHECK(policy.anchorGrowthThreshold()==0 && policy.anchorGrowthAlpha()==2 && !policy.anchorGrowthManual());
  for(auto settings:std::vector<std::array<double,2>>{{-.1,1.2},{.001,1.},{.001,0.},
      {std::numeric_limits<double>::quiet_NaN(),1.2},{.001,std::numeric_limits<double>::infinity()}}) {
    bool rejected=false;
    try { policy.setAnchorGrowthPolicy(settings[0],settings[1]); } catch(const c10::Error&) { rejected=true; }
    TORCH_CHECK(rejected,"Invalid growth policy accepted");
  }
  policy.setAnchorGrowthPolicy(.001,1.2);
  TORCH_CHECK(policy.anchorGrowthThreshold()==.001 && policy.anchorGrowthAlpha()==1.2 && policy.anchorGrowthManual());
  const double t=policy.anchorGrowthThreshold(),alpha=policy.anchorGrowthAlpha();
  auto check=[&](int64_t level,double g,int64_t expected,bool strict,bool observed=true) {
    improvements::selector::AnchorBank bank(.1,0);
    bank->enableExplicitBlocks();bank->enableVariableMembers();bank->to(device);
    auto observations=torch::tensor({{.011f,.012f,.013f}},options);
    auto initial=bank->observeBlocks(observations,torch::full_like(observations,.4));
    auto xyz=initial.xyz;
    if(level==2) {
      auto owner=torch::nonzero(bank->levels()==1).reshape({-1})[0].item<int64_t>();
      auto row=torch::nonzero(bank->rows()==owner).reshape({-1})[0].item<int64_t>();
      xyz[row][0]=.411f;
      auto gradient=torch::zeros({xyz.size(0),1},options.dtype(torch::kFloat64));gradient[row]=1.;
      auto grown=bank->growBlocks(xyz,initial.color,gradient,torch::ones_like(gradient),t,alpha,{},false,true);
      xyz=torch::cat({xyz,grown.xyz});initial.color=torch::cat({initial.color,grown.color});
    }
    auto owner=torch::nonzero(bank->levels()==level).reshape({-1})[0].item<int64_t>();
    auto row=torch::nonzero(bank->rows()==owner).reshape({-1})[0].item<int64_t>();
    xyz[row][0]=.711f;
    auto gradient=torch::zeros({xyz.size(0),1},options.dtype(torch::kFloat64));gradient[row]=g;
    auto counts=torch::ones_like(gradient);if(!observed)counts[row]=0;
    auto grown=bank->growBlocks(xyz,initial.color,gradient,counts,t,alpha,{},false,strict);
    if(expected<0) { TORCH_CHECK(grown.xyz.numel()==0,"Unexpected growth at level ",level," gradient ",g); }
    else { TORCH_CHECK(grown.xyz.size(0)==1 && bank->levels()[-1].item<int64_t>()==expected,
                      "Incorrect growth level for source ",level," gradient ",g); }
    bank->assertBlocks(xyz.size(0)+grown.xyz.size(0));
  };
  for(int64_t level=0;level<=2;++level) {
    const double low=t*std::pow(alpha,level),high=low*alpha;
    check(level,std::nextafter(low,0.),-1,true);
    check(level,low,level,true);
    check(level,std::nextafter(high,0.),level,true);
    check(level,high,level,true);
    check(level,std::nextafter(high,std::numeric_limits<double>::infinity()),level==2?-1:level+1,true);
    check(level,high,level==2?-1:level+1,false);
    check(level,std::numeric_limits<double>::quiet_NaN(),-1,true);
    check(level,std::numeric_limits<double>::infinity(),-1,true);
    check(level,high*2,-1,true,false);
  }
  policy.setAnchorGrowthPolicy(0,2);
  TORCH_CHECK(policy.anchorGrowthThreshold()==0 && policy.anchorGrowthAlpha()==2 && !policy.anchorGrowthManual());
  std::cout<<"PASS actual Anchor growth: t=0.001 alpha=1.2; closed upper; L0/L1/L2; legacy upper; invalid and unobserved\n";
}
