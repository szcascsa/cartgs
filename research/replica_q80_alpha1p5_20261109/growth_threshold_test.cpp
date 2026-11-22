#include "include/gaussian_model.h"
#include <cmath>
#include <iostream>
#include <limits>

struct CalibrationModel : GaussianModel {
  using GaussianModel::GaussianModel;
  using GaussianModel::anchor_growth_gradient_accum_;
  using GaussianModel::anchor_growth_denom_;
};

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
  for(double quantile:{-.01,1.01,std::numeric_limits<double>::quiet_NaN()}) {
    bool rejected=false;
    try { policy.setAnchorGrowthPolicy(0,1.5,quantile,true); } catch(const c10::Error&) { rejected=true; }
    TORCH_CHECK(rejected,"Invalid quantile accepted");
  }
  policy.setAnchorGrowthPolicy(0,1.5,.8,true);
  TORCH_CHECK(policy.anchorGrowthThreshold()==0 && policy.anchorGrowthAlpha()==1.5 && !policy.anchorGrowthManual() &&
              policy.anchorGrowthQuantile()==.8 && policy.anchorGrowthStrictUpper());
  const double t=.001,alpha=policy.anchorGrowthAlpha();
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
  TORCH_CHECK(policy.anchorGrowthThreshold()==0 && policy.anchorGrowthAlpha()==2 && !policy.anchorGrowthManual() &&
              policy.anchorGrowthQuantile()==.9 && !policy.anchorGrowthStrictUpper());
  // Exercise the actual Full-gradient calibration path, including window reset and freeze.
  for(double quantile:{.8,.9}) {
    CalibrationModel model(3);model.device_type_=device;
    improvements::selector::AnchorBank bank(.1,0);
    bank->enableExplicitBlocks();bank->enableVariableMembers();bank->to(device);
    model.enableExplicitAnchors(bank,nullptr);model.setAnchorTopologyPolicy(1,false);
    model.setAnchorGrowthPolicy(0,1.5,quantile,true);
    std::map<point3D_id_t,Point3D> observed;
    for(int i=0;i<16;++i) {
      Point3D point;point.xyz_=Eigen::Vector3d(.011+.2*i,.012,.013);point.color_=Eigen::Vector3f(.4,.4,.4);
      observed[i]=point;
    }
    model.createFromPcd(observed,1);GaussianOptimizationParams optimization;model.trainingSetup(optimization);
    bank->topologyRound().fill_(1);
    model.densifyAndPrune(100.,.005,1.,0);
    TORCH_CHECK(model.anchorGrowthThreshold()==0,"Empty window calibrated threshold");
    auto root=torch::nonzero(bank->levels().index_select(0,bank->rows())==0).reshape({-1});
    TORCH_CHECK(root.numel()==16);
    auto accum=torch::full_like(model.anchor_growth_gradient_accum_,.2);
    auto counts=torch::full_like(model.anchor_growth_denom_,2);
    std::vector<float> valid;
    for(int i=0;i<11;++i) { valid.push_back(float(i+1)*.0001f);accum[root[i]]=2*valid.back(); }
    accum[root[11]]=std::numeric_limits<float>::quiet_NaN();
    accum[root[12]]=std::numeric_limits<float>::infinity();accum[root[13]]=-1;accum[root[14]]=0;
    accum[root[15]]=1;counts[root[15]]=0;
    auto values=torch::tensor(valid,options);
    const double expected=torch::quantile(values,quantile).item<double>();
    model.anchor_growth_gradient_accum_.copy_(accum);model.anchor_growth_denom_.copy_(counts);
    bank->topologyRound().fill_(1);model.densifyAndPrune(100.,.005,1.,0);
    TORCH_CHECK(model.anchorGrowthThreshold()==expected,"Actual q80/q90 calibration differs from valid L0 quantile");
    TORCH_CHECK(model.anchor_growth_denom_.sum().item<double>()==0,"Anchor window did not reset");
    model.anchor_growth_gradient_accum_.fill_(100);model.anchor_growth_denom_.fill_(1);
    bank->topologyRound().fill_(1);model.densifyAndPrune(100.,.005,1.,0);
    TORCH_CHECK(model.anchorGrowthThreshold()==expected,"Calibrated threshold changed in a later window");
    bank->assertBlocks(model.xyz_.size(0));
  }
  std::cout<<"PASS actual Anchor q80/q90 calibration; first valid positive finite L0 window; frozen threshold; alpha1.5; L0/L1/L2 boundaries\n";
}
