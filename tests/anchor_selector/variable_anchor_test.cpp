#include "include/gaussian_model.h"
#include "improvements/selector/gumbel_network.h"
#ifdef CARTGS_FUSED_MAPPING_CUDA
#include "include/mapping_cuda_optimizer.h"
#endif
#include <iostream>

int main(int argc,char** argv) {
  const bool cuda=argc>1&&std::string(argv[1])=="cuda";
  auto device=cuda?torch::kCUDA:torch::kCPU;
  auto options=torch::TensorOptions().dtype(torch::kFloat32).device(device);
  auto selector=improvements::selector::GumbelNetwork(.1,true,false,true);selector->to(device);
  const bool fused=argc>2&&std::string(argv[2])=="fused";
  selector->setFusedInputPack(fused);
  auto takeStep=[&](torch::optim::Adam& optimizer) {
#ifdef CARTGS_FUSED_MAPPING_CUDA
    if(fused){TORCH_CHECK(mapping_cuda::tryFusedDenseAdam(optimizer),"Unsupported fused lifecycle layout");return;}
#endif
    optimizer.step();
  };
  auto bank=selector->anchors();bank->enableVariableMembers();bank->enableSlotDeactivation();bank->setFusedFullStats(fused);
#ifdef CARTGS_TEST_PRIVATE_COMPENSATION
  selector->enableCompensation();selector->enablePrivateCompensation(true,{.01,.05,.1,.15});
#endif
  torch::optim::Adam feature_optimizer(selector->parameters(),torch::optim::AdamOptions(.001));
  GaussianModelParams params;params.sh_degree_=1;params.data_device_=cuda?"cuda":"cpu";
  GaussianModel map(params);map.enableExplicitAnchors(bank,&feature_optimizer);
#ifdef CARTGS_TEST_PRIVATE_COMPENSATION
  map.enablePrivateCompensation(selector->privateCompensation());
#endif
  std::map<point3D_id_t,Point3D> observed;
  int id=0;
  for(int group=0;group<3;++group)for(int k=0;k<std::vector<int>{1,3,12}[group];++k) {
    Point3D p;p.xyz_=Eigen::Vector3d(.01+.2*group+.001*k,.01,2.01);p.color_=Eigen::Vector3f(.4,.4,.4);observed[id++]=p;
  }
  map.createFromPcd(observed,1);GaussianOptimizationParams opt;map.trainingSetup(opt);
  bank->assertBlocks(map.xyz_.size(0));
  TORCH_CHECK(bank->size()==6 && map.xyz_.size(0)==28 && bank->memberCounts().max().item<int64_t>()==10,
      "Real observation count/capacity initialization failed");
  auto xyz_cpu=map.xyz_.detach().cpu();
  for(int64_t i=0;i<xyz_cpu.size(0);++i) {
    bool found=false;for(const auto& v:observed) {
      auto expected=torch::tensor({float(v.second.xyz_(0)),float(v.second.xyz_(1)),float(v.second.xyz_(2))});
      found|=torch::equal(expected,xyz_cpu[i]);
    }
    TORCH_CHECK(found,"Initialization invented a non-observed position");
  }
  TORCH_CHECK(!bank->features().count_nonzero().item<int64_t>(),"Initial Features must be zero");
  const auto original_n=map.xyz_.size(0);
  {torch::NoGradGuard guard;auto x=map.xyz_.detach().clone(),c=torch::full_like(x,.4);map.increasePcd(x,c,1);}
  TORCH_CHECK(map.xyz_.size(0)==original_n,"Repeated occupied observation inserted duplicate members");
  // Prime both Feature Adam owners and all six Gaussian groups.
  auto feature_loss=bank->features().sum();
  std::vector<torch::Tensor> auxiliary{bank->features()};
#ifdef CARTGS_TEST_PRIVATE_COMPENSATION
  auto private_bank=selector->privateCompensation();
  TORCH_CHECK(private_bank->size()==map.xyz_.size(0),"Initial map lost private rows");
  for(auto table:private_bank->tables()){feature_loss=feature_loss+table.sum();auxiliary.push_back(table);}
#endif
  feature_loss.backward();takeStep(feature_optimizer);feature_optimizer.zero_grad(true);
  map.optimizer_->add_param_group(torch::optim::OptimizerParamGroup(auxiliary));
  auto loss=bank->features().sum();for(int group=0;group<6;++group)loss=loss+map.optimizer_->param_groups()[group].params()[0].sum();
#ifdef CARTGS_TEST_PRIVATE_COMPENSATION
  for(auto table:private_bank->tables())loss=loss+table.sum();
#endif
  loss.backward();takeStep(*map.optimizer_);map.optimizer_->zero_grad(true);
  auto full_owner=torch::nonzero(bank->memberCounts()==10).reshape({-1})[0].item<int64_t>();
  auto full_rows=torch::nonzero(bank->rows()==full_owner).reshape({-1});auto dead=full_rows[0].item<int64_t>();
  auto all=torch::arange(map.xyz_.size(0),options.dtype(torch::kLong));
  {torch::NoGradGuard guard;map.opacity_.fill_(std::log(.1/.9));map.opacity_[dead].fill_(-12);}
  for(int i=0;i<5;++i)map.addAnchorPruningStats(all);
  const auto visits=map.anchorVisits().clone(),uids=bank->blockUIDs().clone(),features=bank->features().clone();
  auto opacity_moment=static_cast<torch::optim::AdamParamState&>(*map.optimizer_->state().at(map.opacity_.unsafeGetTensorImpl())).exp_avg().clone();
  {
    torch::NoGradGuard guard;
    map.densifyAndPrune(1e6,.02,1,0,5);
  }
  TORCH_CHECK(bank->topologyRound().item<int64_t>()==1 && bank->memberCounts()[full_owner].item<int64_t>()==9,
      "Full Anchor must allow per-Gaussian pruning");
  TORCH_CHECK(bank->size()==uids.numel()&&torch::equal(uids,bank->blockUIDs())&&torch::equal(features,bank->features()),
      "Partial prune changed Anchor identity/Feature");
  auto keep=all!=dead;
  TORCH_CHECK(torch::equal(static_cast<torch::optim::AdamParamState&>(*map.optimizer_->state().at(map.opacity_.unsafeGetTensorImpl())).exp_avg(),
      opacity_moment.index({keep})),"Partial prune corrupted Gaussian Adam state");
  TORCH_CHECK(map.anchorVisits()[full_owner].item<int64_t>()==0,"Changed membership kept a mixed pruning window");
  auto unchanged=torch::arange(bank->size(),all.options())!=full_owner;
  TORCH_CHECK(torch::equal(map.anchorVisits().index({unchanged}),visits.index({unchanged})),"Unchanged Anchor lost its window");
  // Anchor round cannot run Gaussian clone/split, even with high Gaussian gradients.
  const auto n=map.xyz_.size(0);
  {torch::NoGradGuard guard;map.xyz_gradient_accum_.fill_(100);map.denom_.fill_(1);map.densifyAndPrune(1,.02,1,0,500);}
  TORCH_CHECK(bank->topologyRound().item<int64_t>()==2&&map.xyz_.size(0)==n,"Alternation ran Gaussian densify on Anchor round");
  // Gaussian statistics survived the Anchor round; clones fill only available capacity.
  TORCH_CHECK(map.xyz_gradient_accum_.max().item<float>()==100,"Anchor round erased Gaussian gradient window");
  {torch::NoGradGuard guard;map.scaling_.fill_(-7);map.densifyAndPrune(1,.02,1,0,500);}
  bank->assertBlocks(map.xyz_.size(0));
  TORCH_CHECK(bank->memberCounts()[full_owner].item<int64_t>()==10 && bank->size()==uids.numel(),"Clone capacity/owner failed");
  // Make room in a full Anchor and force split rather than clone.
  {torch::NoGradGuard guard;
    bank->topologyRound().fill_(4);
    auto members=torch::nonzero(bank->rows()==full_owner).reshape({-1});
    map.opacity_[members[0]].fill_(-12);map.scaling_.index_fill_(0,members,-2);
    map.xyz_gradient_accum_.zero_();map.xyz_gradient_accum_.index_fill_(0,members,10);map.denom_.fill_(1);
    map.densifyAndPrune(1,.02,1,0,500);
  }
  TORCH_CHECK(bank->memberCounts()[full_owner].item<int64_t>()==10,"Split net capacity exceeded ten or lost owner");
  // Empty Anchor removal must migrate both Adam owners by surviving UID.
  auto feature_moment=static_cast<torch::optim::AdamParamState&>(*feature_optimizer.state().at(bank->features().unsafeGetTensorImpl())).exp_avg().clone();
  auto old_uids=bank->blockUIDs().clone();
  {torch::NoGradGuard guard;bank->topologyRound().fill_(6);map.opacity_.index_fill_(0,torch::nonzero(bank->rows()==0).reshape({-1}),-12);
    map.densifyAndPrune(1e6,.02,1,0,500);}
  TORCH_CHECK(torch::equal(bank->blockUIDs(),old_uids.slice(0,1)),"Empty Anchor was not removed");
  TORCH_CHECK(torch::equal(static_cast<torch::optim::AdamParamState&>(*feature_optimizer.state().at(bank->features().unsafeGetTensorImpl())).exp_avg(),
      feature_moment.slice(0,1)),"Empty Anchor removal corrupted Feature Adam");
  // A retained Anchor still supports the original window prune and slot eligibility.
  map.anchorPruningMask(1);
  {torch::NoGradGuard guard;map.opacity_.fill_(std::log(.1/.9));map.opacity_[0].fill_(-12);}
  all=torch::arange(map.xyz_.size(0),all.options());
  for(int i=0;i<5;++i)map.addAnchorPruningStats(all);
  TORCH_CHECK(!map.anchorPruningMask(5).any().item<bool>() && bank->inactiveSlots().sum().item<int64_t>()==1,
      "Variable slot window must deactivate one member without deleting its effective Anchor");
  // Geometry transport must use actual member count, including mixed correction groups.
  auto owner0=torch::nonzero(bank->rows()==0).reshape({-1});
  auto count=owner0.numel();auto center=bank->centers()[0].clone();
  bank->beginBlockGeometry();GaussianTransformResult correction;correction.indices=owner0.slice(0,0,1);
  correction.population_count=map.xyz_.size(0);correction.scale=1;correction.changed_attributes=toBits(GaussianAttrFlag::XYZ);
  correction.transform_4x4=torch::eye(4,options);correction.transform_4x4[3][0]=.3;
  bank->transportBlocks(correction);bank->finishBlockGeometry();
  TORCH_CHECK(torch::allclose(bank->centers()[0],center+torch::tensor({.3f/float(count),0.f,0.f},options),1e-5,1e-6),
      "Geometry used fixed ten denominator");
  auto frozen=selector->cloneFrozen();TORCH_CHECK(torch::equal(frozen->anchors()->rows(),bank->rows()),"Teacher lost variable ownership");
  bank->assertBlocks(map.xyz_.size(0));
  torch::serialize::OutputArchive archive;selector->save(archive);archive.save_to("variable_selector.pt");
  auto loaded=improvements::selector::GumbelNetwork(.1,true,false,true);loaded->to(device);
  torch::serialize::InputArchive input;input.load_from("variable_selector.pt",device);loaded->load(input);
  loaded->anchors()->assertBlocks(map.xyz_.size(0));
#ifdef CARTGS_TEST_PRIVATE_COMPENSATION
  TORCH_CHECK(private_bank->size()==map.xyz_.size(0) && loaded->privateCompensation()->size()==map.xyz_.size(0),
              "Actual Gaussian clone/split/prune lost private rows");
  for(auto table:private_bank->tables())for(auto* optimizer:{map.optimizer_.get(),&feature_optimizer}) {
    auto found=optimizer->state().find(table.unsafeGetTensorImpl());
    TORCH_CHECK(found!=optimizer->state().end(),"Topology lost private optimizer alias");
    auto& state=static_cast<torch::optim::AdamParamState&>(*found->second);
    TORCH_CHECK(state.exp_avg().sizes()==table.sizes() && state.exp_avg_sq().sizes()==table.sizes(),
                "Actual topology lost Adam row alignment");
  }
#endif
  TORCH_CHECK(torch::equal(loaded->anchors()->rows(),bank->rows())&&torch::equal(loaded->anchors()->topologyRound(),bank->topologyRound()),"Variable checkpoint round trip");
  torch::serialize::OutputArchive fixture;fixture.write("xyz",map.xyz_);fixture.write("scale",map.getScalingActivation());
  fixture.write("dc",map.features_dc_);fixture.write("opacity",map.getOpacityActivation());fixture.write("uniform",map.protection_.cached_uniform);
  int j=0;for(float ratio:{.01f,.05f,.1f,.15f}) {
    auto p=selector->forwardAnchorProbability(map.xyz_,ratio,1,{},map.getScalingActivation(),map.features_dc_,map.getOpacityActivation());
    fixture.write("p"+std::to_string(j++),selector->maskInactiveSlots(p,map.getOpacityActivation()));
  }
  fixture.save_to("variable_fixture.pt");map.saveAnchorPruningState("variable_pruning.pt",500);
  // Variable-sized same/fine growth: max inheritance, zero fine Feature, dual Adam.
  auto growth=improvements::selector::GumbelNetwork(.1,true,false,true);growth->to(device);
  auto gb=growth->anchors();gb->enableVariableMembers();gb->setFusedFullStats(fused);
  torch::optim::Adam first(growth->parameters(),torch::optim::AdamOptions(.001));
  torch::optim::Adam second(std::vector<torch::Tensor>{gb->features()},torch::optim::AdamOptions(.001));
  auto gx=torch::tensor({{.01f,.01f,.01f},{.21f,.01f,.01f}},options),gc=torch::full_like(gx,.4);
  auto seeds=gb->observeBlocks(gx,gc,{&first,&second});gx=seeds.xyz;gc=seeds.color;
  {torch::NoGradGuard guard;for(int64_t j=0;j<gb->size();++j)gb->features()[j].fill_(-float(j+1));}
  gb->features().sum().backward();takeStep(first);takeStep(second);first.zero_grad(true);second.zero_grad(true);
  auto saved=gb->features().detach().clone();
  auto moment=static_cast<torch::optim::AdamParamState&>(*first.state().at(gb->features().unsafeGetTensorImpl())).exp_avg().clone();
  gx[0][0]=.411;gx[1][0]=.419;gx[2][0]=.71;
  auto gg=torch::tensor({1.5f,1.5f,5.f,0.f},options).reshape({-1,1}),gd=torch::ones_like(gg);
  auto grown=gb->growBlocks(gx,gc,gg,gd,1,2,{&first,&second});
  TORCH_CHECK(gb->size()==6 && grown.xyz.size(0)==3 && torch::equal(gb->features()[4],saved[0]) &&
      !gb->features()[5].count_nonzero().item<int64_t>(),"Variable same/fine growth or Feature inheritance failed");
  for(auto* optimizer:{&first,&second}) {
    auto& state=static_cast<torch::optim::AdamParamState&>(*optimizer->state().at(gb->features().unsafeGetTensorImpl()));
    TORCH_CHECK(torch::equal(state.exp_avg().slice(0,0,4),moment) && !state.exp_avg().slice(0,4).count_nonzero().item<int64_t>() &&
        !state.exp_avg_sq().slice(0,4).count_nonzero().item<int64_t>(),"Variable growth corrupted Feature Adam");
  }
  gb->assertBlocks(gx.size(0)+grown.xyz.size(0));
  if(fused) {
    map.optimizer_->zero_grad(true);auto final_loss=map.xyz_.sum()+map.features_dc_.sum()+map.features_rest_.sum()+
        map.opacity_.sum()+map.scaling_.sum()+map.rotation_.sum()+bank->features().sum();
#ifdef CARTGS_TEST_PRIVATE_COMPENSATION
    final_loss=final_loss+private_bank->forward(torch::arange(map.xyz_.size(0),bank->rows().options()),.05).sum();
#endif
    final_loss.backward();takeStep(*map.optimizer_);map.optimizer_->zero_grad(true);
    bank->assertBlocks(map.xyz_.size(0));
  }
  std::cout<<"PASS variable anchors: real seeds, capacity, partial/empty prune, clone/split net budget, alternating windows, Adam, geometry, teacher, checkpoint\n";
  return 0;
}
