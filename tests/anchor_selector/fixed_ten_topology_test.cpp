#include "include/gaussian_model.h"
#include "improvements/selector/gumbel_network.h"
#include <iostream>

int main(int argc,char** argv) try {
  const torch::Device device=argc>1&&std::string(argv[1])=="cuda"?torch::kCUDA:torch::kCPU;
  for(int mode:{3,4}) {
    auto selector=improvements::selector::GumbelNetwork(.1,true,false,true,std::vector<double>{},0);selector->to(device);
    auto bank=selector->anchors();bank->enableVariableMembers();
    selector->enableSpatialCompensation(false,false,0,{.01,.05,.1,.2,.3,.4},true,false,true,false,48,1);
    torch::optim::Adam feature_optimizer(selector->parameters(),torch::optim::AdamOptions(.001));
    GaussianModelParams params;params.sh_degree_=1;params.data_device_=device==torch::kCUDA?"cuda":"cpu";
    GaussianModel map(params);map.enableExplicitAnchors(bank,&feature_optimizer);map.setAnchorTopologyPolicy(mode,true);
    std::map<point3D_id_t,Point3D> observed;
    Point3D point;point.xyz_=Eigen::Vector3d(.01,.01,2.01);point.color_=Eigen::Vector3f(.4,.4,.4);observed[0]=point;
    map.createFromPcd(observed,1);GaussianOptimizationParams optimization;map.trainingSetup(optimization);
    const auto n=map.xyz_.size(0);auto owners=bank->rows().clone();
    TORCH_CHECK(n==bank->size()*10&&bank->slotDeactivation()==(mode==4),"Fixed-ten map initialization failed");
    {
      torch::NoGradGuard guard;map.xyz_gradient_accum_.fill_(100);map.denom_.fill_(1);map.opacity_.fill_(-20);map.max_radii2D_.fill_(100);
      map.densifyAndPrune(1,.02,1,1,5);map.densifyAndPrune(1,.02,1,1,5);
    }
    TORCH_CHECK(bank->topologyRound().item<int64_t>()==2&&map.xyz_.size(0)==n&&torch::equal(owners,bank->rows()),
        "Fixed-ten topology ran Gaussian clone/split/quality pruning");
    TORCH_CHECK(map.gaussianPrunedMembers()==0&&map.xyz_gradient_accum_.count_nonzero().item<int64_t>()==0,"Gaussian statistics were not bounded");
    // A persistent low-opacity member remains stored; only mode 4 excludes it from LoD eligibility.
    {
      torch::NoGradGuard guard;map.opacity_.fill_(std::log(.1/.9));map.opacity_[0]=-20;
    }
    auto all=torch::arange(n,bank->rows().options());for(int i=0;i<5;++i)map.addAnchorPruningStats(all);
    {torch::NoGradGuard guard;map.densifyAndPrune(1e6,.02,1,0,5);}
    auto eligible=selector->maskInactiveSlots(torch::ones({n},map.xyz_.options()),map.getOpacityActivation());
    TORCH_CHECK(map.xyz_.size(0)==n&&eligible[0].item<float>()==(mode==3?1.f:0.f),"Slot ablation physically deleted members or used wrong mode");
    // The first and second consecutive events both execute Anchor growth.
    auto gradients=torch::zeros_like(map.xyz_).set_requires_grad(true);
    for(int event=0;event<2;++event) {
      {
        torch::NoGradGuard guard;map.xyz_[0][0]=.411f+.2f*event;
      }
      if(gradients.grad().defined())gradients.mutable_grad()=torch::Tensor();
      auto weights=torch::zeros_like(gradients);weights[0][0]=1;
      (gradients*weights).sum().backward();map.addDensificationStats(gradients,all);
      const auto before=bank->size();
      {torch::NoGradGuard guard;map.densifyAndPrune(1e6,.02,1,0,5);}
      TORCH_CHECK(bank->size()>before&&(bank->memberCounts()==10).all().item<bool>(),"A consecutive Anchor event failed to grow ten-point blocks");
      all=torch::arange(map.xyz_.size(0),bank->rows().options());gradients=torch::zeros_like(map.xyz_).set_requires_grad(true);
    }
    const auto dead=bank->size()-1;auto dead_rows=torch::nonzero(bank->rows()==dead).reshape({-1});
    {torch::NoGradGuard guard;map.opacity_.index_fill_(0,dead_rows,-20);}
    for(int i=0;i<5;++i)map.addAnchorPruningStats(all);
    const auto before=map.xyz_.size(0);
    {torch::NoGradGuard guard;map.densifyAndPrune(1e6,.02,1,0,5);}
    TORCH_CHECK(map.xyz_.size(0)==before-10&&map.anchorPrunedMembers()==10,"Whole-Anchor pruning did not remove exactly ten members");
    auto mask=torch::zeros({map.xyz_.size(0)},bank->rows().options().dtype(torch::kBool));mask[0]=true;
    bool rejected=false;try{map.prunePoints(mask);}catch(const c10::Error&){rejected=true;}
    TORCH_CHECK(rejected&&map.xyz_.size(0)==before-10,"Model accepted partial Anchor deletion");
    auto loss=torch::zeros({},map.xyz_.options());
    for(auto& group:map.optimizer_->param_groups())for(auto& value:group.params())loss=loss+value.sum();
    loss.backward();map.optimizer_->step();map.optimizer_->zero_grad(true);
    loss=torch::zeros({},map.xyz_.options());for(auto value:selector->parameters())loss=loss+value.sum();
    loss.backward();feature_optimizer.step();feature_optimizer.zero_grad(true);
    auto source=torch::nonzero(bank->rows()==0).reshape({-1});
    auto target=torch::nonzero((bank->levels()==0)&(bank->blockUIDs()!=bank->blockUIDs()[0])).reshape({-1})[0].item<int64_t>();
    auto shift=bank->centers()[target]-bank->centers()[0];bank->beginBlockGeometry();
    {
      torch::NoGradGuard guard;map.xyz_.index_put_({source},map.xyz_.index_select(0,source)+shift);
      GaussianTransformResult correction;correction.indices=source;correction.population_count=map.xyz_.size(0);
      correction.transform_4x4=torch::eye(4,map.xyz_.options());correction.transform_4x4[3].slice(0,0,3).copy_(shift);
      correction.changed_attributes=toBits(GaussianAttrFlag::XYZ);bank->transportBlocks(correction);
    }
    map.finishAnchorGeometry();auto keep=bank->rows()!=0;auto alive=bank->blockUIDs()!=bank->blockUIDs()[0];
    std::vector<torch::Tensor> parameters,moments,feature_values,feature_moments;
    for(auto& group:map.optimizer_->param_groups())for(auto& value:group.params()) {
      parameters.push_back(value.detach().index({keep}).clone());
      moments.push_back(static_cast<torch::optim::AdamParamState&>(*map.optimizer_->state().at(value.unsafeGetTensorImpl())).exp_avg().index({keep}).clone());
    }
    for(auto value:bank->featureTables()) {
      if(!value.numel())continue;
      feature_values.push_back(value.detach().index({alive}).clone());
      feature_moments.push_back(static_cast<torch::optim::AdamParamState&>(*feature_optimizer.state().at(value.unsafeGetTensorImpl())).exp_avg().index({alive}).clone());
    }
    map.migrateLocalAnchors();size_t j=0;
    for(auto& group:map.optimizer_->param_groups())for(auto& value:group.params()) {
      auto& state=static_cast<torch::optim::AdamParamState&>(*map.optimizer_->state().at(value.unsafeGetTensorImpl()));
      TORCH_CHECK(torch::equal(value,parameters[j])&&torch::equal(state.exp_avg(),moments[j]),"Model collision lost surviving Gaussian parameters/Adam");++j;
    }
    j=0;for(auto value:bank->featureTables()) {
      if(!value.numel())continue;
      auto& state=static_cast<torch::optim::AdamParamState&>(*feature_optimizer.state().at(value.unsafeGetTensorImpl()));
      TORCH_CHECK(torch::equal(value,feature_values[j])&&torch::equal(state.exp_avg(),feature_moments[j]),"Model collision lost Residual48 Feature/Adam");++j;
    }
    auto frozen=selector->cloneFrozen();TORCH_CHECK(frozen->anchors()->fixedTenMode()==mode,"Selector teacher lost fixed-ten mode");
    const auto checkpoint="fixed_ten_selector_"+std::to_string(mode)+".pt";
    torch::serialize::OutputArchive archive;selector->save(archive);archive.save_to(checkpoint);
    auto loaded=improvements::selector::GumbelNetwork(.1,true,false,true,std::vector<double>{},0);loaded->to(device);
    torch::serialize::InputArchive input;input.load_from(checkpoint,device);loaded->load(input);
    TORCH_CHECK(loaded->anchors()->fixedTenMode()==mode&&torch::equal(loaded->anchors()->rows(),bank->rows()),"Residual48 checkpoint lost fixed-ten layout");
    bank->assertBlocks(map.xyz_.size(0));
  }
  std::cout<<"PASS fixed-ten topology: modes 3/4, consecutive growth, no Gaussian topology, slot eligibility, whole pruning, teacher/checkpoint\n";
} catch(const std::exception& error) {
  std::cerr<<error.what()<<'\n';return 1;
}
