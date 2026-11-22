#include "include/gaussian_model.h"
#include "include/general_utils.h"
#include "improvements/selector/gumbel_network.h"
#include <cmath>
#include <iostream>

namespace {
torch::Tensor noise(int64_t count,torch::Device device) {
  auto z=torch::normal(0.f,1.f,{count,3}).to(device);
  z.index_put_({z.abs()>3.f},0.f);
  return z;
}

void mathTest(GaussianModel& map,const torch::TensorOptions& options) {
  torch::NoGradGuard guard;
  map.scaling_=torch::tensor({{.2f,.2f,.2f},{2.f,2.f,2.f},{.1f,.7f,3.f}},options).log();
  map.rotation_=torch::tensor({{1.f,0.f,0.f,0.f},{1.f,0.f,0.f,0.f},{.8f,.2f,-.3f,.45f}},options);
  auto rotation=map.rotation_.clone(),R=general_utils::build_rotation(rotation);
  auto L=map.getScalingRotationActivation(),cov=map.getCovarianceActivation();
  auto previous_L=R.matmul(torch::diag_embed(map.getScalingActivation()));
  TORCH_CHECK(torch::equal(L,previous_L)&&torch::equal(cov,previous_L.matmul(previous_L.transpose(1,2))),
      "Covariance interface changed");
  TORCH_CHECK(torch::allclose(map.getScalingRotationActivation(2),2*L),"Length scaling must be linear");
  const double phi=std::exp(-4.5)/std::sqrt(2*std::acos(-1.));
  const double retained_variance=1-2*(3*phi+.5*std::erfc(3/std::sqrt(2.)));
  torch::manual_seed(2026);
  for(int i=0;i<3;++i) {
    auto z=noise(200000,L.device());
    auto delta=z.matmul(L[i].t()).to(torch::kFloat64);
    auto empirical=delta.t().matmul(delta)/delta.size(0);
    auto expected=retained_variance*cov[i].to(torch::kFloat64);
    double error=((empirical-expected).norm()/expected.norm()).item<double>();
    TORCH_CHECK(error<.025,"Sample covariance differs from Gaussian covariance: ",error);
    auto old_delta=z.matmul(cov[i].t()).to(torch::kFloat64);
    auto old_cov=old_delta.t().matmul(old_delta)/old_delta.size(0);
    TORCH_CHECK(((old_cov-expected).norm()/expected.norm()).item<double>()>.3,
        "Fixture failed to distinguish squared-scale sampling");
    std::cout<<"covariance_case="<<i<<" relative_error="<<error<<'\n';
  }
  map.scaling_=torch::empty({0,3},options);map.rotation_=torch::empty({0,4},options);
  TORCH_CHECK(map.getScalingRotationActivation().sizes()==torch::IntArrayRef({0,3,3}),"Empty factor shape");
}

void initializeLegacy(GaussianModel& map,const torch::TensorOptions& options) {
  map.xyz_=torch::tensor({{.1f,.2f,2.f}},options).requires_grad_();
  map.features_dc_=torch::full({1,1,3},.3,options).requires_grad_();
  map.features_rest_=torch::zeros({1,3,3},options).requires_grad_();
  map.opacity_=torch::zeros({1,1},options).requires_grad_();
  map.scaling_=torch::tensor({{.2f,.4f,.6f}},options).log().requires_grad_();
  map.rotation_=torch::tensor({{.8f,.2f,-.3f,.45f}},options).requires_grad_();
  map.max_radii2D_=torch::zeros({1},options);
  map.exist_since_iter_=torch::full({1},5,options.dtype(torch::kInt32));
  map.protection_.reset(1,options);
  map.Tensor_vec_xyz_={map.xyz_};map.Tensor_vec_feature_dc_={map.features_dc_};
  map.Tensor_vec_feature_rest_={map.features_rest_};map.Tensor_vec_opacity_={map.opacity_};
  map.Tensor_vec_scaling_={map.scaling_};map.Tensor_vec_rotation_={map.rotation_};
  GaussianOptimizationParams optimizer;map.trainingSetup(optimizer);
}

void splitTest(GaussianModelParams params,const torch::TensorOptions& options,bool variable) {
  GaussianModel map(params);
  auto selector=improvements::selector::GumbelNetwork(.1,true,false,true);selector->to(options.device());
  auto bank=selector->anchors();bank->enableVariableMembers();bank->enableSlotDeactivation();
  torch::optim::Adam feature_optimizer(selector->parameters(),torch::optim::AdamOptions(.001));
  if(variable) {
    map.enableExplicitAnchors(bank,&feature_optimizer);map.setAnchorTopologyPolicy(true,false);
    std::map<point3D_id_t,Point3D> observed;Point3D p;
    p.xyz_=Eigen::Vector3d(.01,.01,2.01);p.color_=Eigen::Vector3f(.4,.4,.4);observed[0]=p;
    map.createFromPcd(observed,1);GaussianOptimizationParams optimizer;map.trainingSetup(optimizer);
    torch::NoGradGuard guard;
    map.scaling_.copy_(torch::tensor({{.2f,.4f,.6f}},options).log());
    map.rotation_.copy_(torch::tensor({{.8f,.2f,-.3f,.45f}},options));
  } else initializeLegacy(map,options);
  // Prime Gaussian and Feature Adam states before the topology transaction.
  auto loss=map.xyz_.sum()+map.features_dc_.sum()+map.features_rest_.sum()+map.opacity_.sum()+map.scaling_.sum()+map.rotation_.sum();
  loss.backward();map.optimizer_->step();map.optimizer_->zero_grad(true);
  bank->features().sum().backward();feature_optimizer.step();feature_optimizer.zero_grad(true);
  torch::NoGradGuard guard;
  auto parent=map.xyz_.clone(),scale=map.getScalingActivation().clone();
  auto dc=map.features_dc_.clone(),alpha=map.opacity_.clone();
  auto feature=bank->features().clone(),uids=bank->blockUIDs().clone();
  auto L=map.getScalingRotationActivation();
  auto owners=variable?bank->rows().clone():torch::Tensor{};
  torch::manual_seed(2026);
  auto expected=parent.repeat({2,1})+L.repeat({2,1,1}).matmul(noise(2*parent.size(0),L.device()).unsqueeze(-1)).squeeze(-1);
  torch::manual_seed(2026);
  if(variable) {
    map.xyz_gradient_accum_.fill_(10);map.denom_.fill_(1);
    map.densifyAndPrune(1,.02,1,0,500);
  } else {
    auto gradients=torch::ones({1,1},options);map.densifyAndSplit(gradients,.5,1,2);
  }
  TORCH_CHECK(torch::allclose(map.xyz_,expected,2e-5,2e-6),
      variable?"Variable split uses covariance instead of scale factor":"Legacy split uses covariance instead of scale factor");
  TORCH_CHECK(torch::allclose(map.getScalingActivation(),scale.repeat({2,1})/1.6),"Child scale rule changed");
  TORCH_CHECK(torch::equal(map.features_dc_,dc.repeat({2,1,1}))&&torch::equal(map.opacity_,alpha.repeat({2,1})),"Inherited attributes changed");
  TORCH_CHECK(map.optimizer_->param_groups()[0].params()[0].sizes()==map.xyz_.sizes(),"Gaussian optimizer lost rows");
  if(variable) {
    bank->assertBlocks(2*parent.size(0));
    TORCH_CHECK(bank->size()==uids.numel()&&bank->memberCounts().eq(2).all().item<bool>()&&
        torch::equal(bank->rows(),owners.repeat({2})),"Child ownership/capacity changed");
    TORCH_CHECK(torch::equal(feature,bank->features())&&torch::equal(uids,bank->blockUIDs()),"Split changed Anchor identity/Feature");
    TORCH_CHECK(feature_optimizer.state().count(bank->features().unsafeGetTensorImpl())==1,"Feature Adam state lost");
  }
  std::cout<<(variable?"variable_split":"legacy_split")<<" passed\n";
}
}

int main(int argc,char** argv) {
  try {
    bool cuda=argc>1&&std::string(argv[1])=="cuda";
    if(cuda&&!torch::cuda::is_available())return 77;
    auto device=cuda?torch::kCUDA:torch::kCPU;
    auto options=torch::TensorOptions().dtype(torch::kFloat32).device(device);
    GaussianModelParams params;params.sh_degree_=1;params.data_device_=cuda?"cuda":"cpu";
    std::string mode=argc>2?argv[2]:"all";
    if(mode=="all"||mode=="math"){GaussianModel map(params);mathTest(map,options);}
    if(mode=="all"||mode=="legacy_split")splitTest(params,options,false);
    if(mode=="all"||mode=="variable_split")splitTest(params,options,true);
    std::cout<<"SPLIT_SAMPLING_PASSED "<<(cuda?"cuda":"cpu")<<'\n';return 0;
  } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
