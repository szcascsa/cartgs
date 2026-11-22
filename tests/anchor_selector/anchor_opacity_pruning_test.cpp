#include "include/gaussian_model.h"
#include "improvements/selector/gumbel_network.h"
#include <iostream>

int main(int argc,char** argv) {
  const bool cuda=argc>1&&std::string(argv[1])=="cuda";
  GaussianModelParams params;params.data_device_=cuda?"cuda":"cpu";params.sh_degree_=1;
  const auto device=cuda?torch::kCUDA:torch::kCPU;
  const auto opts=torch::TensorOptions().dtype(torch::kFloat32).device(device);
  auto selector=improvements::selector::GumbelNetwork(.1,true,false,true);selector->to(device);
  torch::optim::Adam feature_optimizer(selector->parameters(),torch::optim::AdamOptions(.001));
  GaussianModel map(params);map.enableExplicitAnchors(selector->anchors(),&feature_optimizer);
  std::map<point3D_id_t,Point3D> observed;
  for(int i=0;i<3;++i) {
    Point3D p;p.xyz_=Eigen::Vector3d(.011+i*.2,.01,2.01);p.color_=Eigen::Vector3f(.4,.4,.4);observed[i]=p;
  }
  map.createFromPcd(observed,1);
  GaussianOptimizationParams optimization;map.trainingSetup(optimization);
  auto bank=selector->anchors();const auto initial=bank->size();
  TORCH_CHECK(initial>=6 && map.anchorVisits().numel()==initial,"Initial statistics layout");
  auto set_opacity=[&](int64_t block,float value) {
    torch::NoGradGuard guard;
    map.opacity_.slice(0,block*10,block*10+10).fill_(std::log(value/(1-value)));
  };
  set_opacity(0,.0001f);set_opacity(1,.001f);set_opacity(2,.0001f);
  // An effective anchor with eight permanently sub-cutoff slots must survive.
  set_opacity(5,.0001f);
  {torch::NoGradGuard guard;map.opacity_.slice(0,58,60).fill_(std::log(.2/.8));}
  auto all=torch::arange(map.xyz_.size(0),opts.dtype(torch::kLong));
  auto regular=all.index({(all/10).floor()!=2});
  map.addAnchorPruningStats(all);map.addAnchorPruningStats(all);
  set_opacity(4,.001f);
  map.addAnchorPruningStats(regular);map.addAnchorPruningStats(regular);
  TORCH_CHECK(map.anchorVisits()[0].item<int64_t>()==4 && map.anchorVisits()[2].item<int64_t>()==2,
              "Ten visible members must count as one visit");
  auto accum=map.anchorOpacityAccum().clone(), visits=map.anchorVisits().clone();
  TORCH_CHECK(!map.anchorPruningMask(5).any().item<bool>(),"Strict count gate failed at exactly 0.8*interval");
  TORCH_CHECK(torch::equal(accum,map.anchorOpacityAccum())&&torch::equal(visits,map.anchorVisits()),
              "Insufficient visits were reset");
  TORCH_CHECK(map.anchorSlotBelowCutoff()[2][0].item<int64_t>()==2&&
      map.anchorSlotBelowCutoff()[4][0].item<int64_t>()==2,"Slot history included unobserved zeros or lost temporary state");
  // Current opacity cannot replace the time average: a useful early view survives.
  set_opacity(3,.0001f);
  map.addAnchorPruningStats(regular);
  accum=map.anchorOpacityAccum().clone();visits=map.anchorVisits().clone();
  map.saveAnchorPruningState("anchor_history_before_check.pt",5);
  auto expected=(visits>4)&(accum<.005*visits.to(torch::kFloat64));
  auto mask=map.anchorPruningMask(5);
  TORCH_CHECK(torch::equal(mask,expected.repeat_interleave(10)),"Octree-GS formula mismatch");
  TORCH_CHECK(mask.slice(0,0,10).all().item<bool>()&&!mask.slice(0,10).any().item<bool>(),
              "Wrong anchor deleted: slot sum, temporal mean, or visit gate");
  TORCH_CHECK((map.anchorSlotBelowCutoff()[5]==0).all().item<bool>(),"Mature slot history was not reset with its Anchor");
  TORCH_CHECK(map.anchorVisits()[0].item<int64_t>()==0&&map.anchorVisits()[2].item<int64_t>()==2,
              "Only sufficiently visited anchors should reset");
  // Exact equality at the opacity threshold must be retained (strict '<').
  {torch::NoGradGuard guard;
    map.anchorVisits()[3].fill_(5);map.anchorOpacityAccum()[3].fill_(.005*5);
    map.anchorSlotOpacityAccum()[3].fill_(.005*5/10);
  }
  TORCH_CHECK(!map.anchorPruningMask(5).any().item<bool>(),"Opacity equality was pruned");
  // Prime both Adam owners and verify surviving state after actual group pruning.
  bank->features().sum().backward();feature_optimizer.step();feature_optimizer.zero_grad(true);
  map.optimizer_->add_param_group(torch::optim::OptimizerParamGroup(std::vector<torch::Tensor>{bank->features()}));
  auto loss=bank->features().sum();
  for(int group=0;group<6;++group)
    for(const auto& parameter:map.optimizer_->param_groups()[group].params())loss=loss+parameter.sum();
  loss.backward();
  map.optimizer_->step();map.optimizer_->zero_grad(true);
  const auto uid=bank->blockUIDs().clone();
  auto xyz_moment=static_cast<torch::optim::AdamParamState&>(
      *map.optimizer_->state().at(map.xyz_.unsafeGetTensorImpl())).exp_avg().clone();
  auto feature_moment=static_cast<torch::optim::AdamParamState&>(
      *feature_optimizer.state().at(bank->features().unsafeGetTensorImpl())).exp_avg().clone();
  auto joint_feature_moment=static_cast<torch::optim::AdamParamState&>(
      *map.optimizer_->state().at(bank->features().unsafeGetTensorImpl())).exp_avg().clone();
  auto saved_accum=map.anchorOpacityAccum().clone(),saved_visits=map.anchorVisits().clone();
  map.prunePoints(mask);
  bank->assertBlocks(map.xyz_.size(0));
  TORCH_CHECK(torch::equal(bank->blockUIDs(),uid.slice(0,1))&&map.xyz_.size(0)==(initial-1)*10,"Whole group/UID pruning");
  TORCH_CHECK(torch::equal(map.anchorOpacityAccum(),saved_accum.slice(0,1))&&
      torch::equal(map.anchorVisits(),saved_visits.slice(0,1)),"Statistics were not pruned with UID");
  TORCH_CHECK(torch::equal(static_cast<torch::optim::AdamParamState&>(
      *map.optimizer_->state().at(map.xyz_.unsafeGetTensorImpl())).exp_avg(),xyz_moment.slice(0,10)),"Gaussian Adam migration");
  TORCH_CHECK(torch::equal(static_cast<torch::optim::AdamParamState&>(
      *feature_optimizer.state().at(bank->features().unsafeGetTensorImpl())).exp_avg(),feature_moment.slice(0,1)),"Feature Adam migration");
  TORCH_CHECK(torch::equal(static_cast<torch::optim::AdamParamState&>(
      *map.optimizer_->state().at(bank->features().unsafeGetTensorImpl())).exp_avg(),joint_feature_moment.slice(0,1)),"Joint Feature Adam migration");
  saved_accum=map.anchorOpacityAccum().clone();saved_visits=map.anchorVisits().clone();
  auto xyz=torch::tensor({{1.01f,.01f,2.01f}},opts),rgb=torch::full_like(xyz,.4);
  const auto old=bank->size();map.increasePcd(xyz,rgb,100);
  TORCH_CHECK(bank->size()==old+2&&torch::equal(map.anchorVisits().slice(0,0,old),saved_visits)&&
      map.anchorVisits().slice(0,old).sum().item<int64_t>()==0&&
      torch::equal(map.anchorOpacityAccum().slice(0,0,old),saved_accum)&&
      map.anchorOpacityAccum().slice(0,old).sum().item<double>()==0,"Append did not preserve old/add zero statistics");
  TORCH_CHECK(map.anchorSlotOpacityAccum().slice(0,old).count_nonzero().item<int64_t>()==0&&
      map.anchorSlotBelowCutoff().slice(0,old).count_nonzero().item<int64_t>()==0,"Newborn slot history must be empty");
  map.saveAnchorPruningState("anchor_history_newborn.pt",500);
  // One never-seen, entirely transparent newborn must survive the real densify path.
  {torch::NoGradGuard guard;map.opacity_.slice(0,old*10).fill_(-20);}
  const auto count=map.xyz_.size(0);
  map.densifyAndPrune(1e6,.02,1,20,500);
  TORCH_CHECK(map.xyz_.size(0)==count,"Legacy opacity/size predicate deleted unobserved newborns");
  set_opacity(0,.0001f);
  auto first=torch::arange(10,opts.dtype(torch::kLong));
  for(int i=0;i<5;++i)map.addAnchorPruningStats(first);
  map.densifyAndPrune(1e6,.02,1,20,5);
  TORCH_CHECK(map.xyz_.size(0)==count-10,"Real densifyAndPrune did not apply the sufficient-visit Anchor rule");
  saved_accum=map.anchorOpacityAccum().clone();saved_visits=map.anchorVisits().clone();
  map.applyScaledTransformation(1.01f);map.finishAnchorGeometry();
  TORCH_CHECK(torch::equal(saved_accum,map.anchorOpacityAccum())&&torch::equal(saved_visits,map.anchorVisits()),
              "Geometry transport changed UID-owned pruning evidence");
  torch::serialize::OutputArchive fixture;
  fixture.write("opacity_accum",accum,true);fixture.write("anchor_visits",visits,true);
  fixture.write("prune_mask",mask,true);fixture.save_to("anchor_pruning_fixture.pt");
  std::cout<<"PASS anchor opacity pruning: official formula, strict gate, partial windows, slot histories diagnostic only, sum, temporal mean, Full stats, append/prune, two Adam owners, geometry\n";
}
