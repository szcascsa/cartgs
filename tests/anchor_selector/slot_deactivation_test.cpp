#include "include/gaussian_model.h"
#include "include/selector_handoff.h"
#include "improvements/selector/gumbel_network.h"
#include <iostream>

int main(int argc,char** argv) {
  const bool cuda=argc>1 && std::string(argv[1])=="cuda";
  auto device=cuda?torch::kCUDA:torch::kCPU;
  auto opts=torch::TensorOptions().dtype(torch::kFloat32).device(device);
  auto selector=improvements::selector::GumbelNetwork(.1,true,false,true);selector->to(device);
  auto bank=selector->anchors();bank->enableSlotDeactivation();
  torch::optim::Adam feature_optimizer(selector->parameters(),torch::optim::AdamOptions(.001));
  GaussianModelParams params;params.data_device_=cuda?"cuda":"cpu";params.sh_degree_=1;
  GaussianModel map(params);map.enableExplicitAnchors(bank,&feature_optimizer);
  std::map<point3D_id_t,Point3D> observed;
  for(int i=0;i<3;++i) {Point3D p;p.xyz_=Eigen::Vector3d(.01+i*.2,.01,2.01);p.color_=Eigen::Vector3f(.4,.4,.4);observed[i]=p;}
  map.createFromPcd(observed,1);GaussianOptimizationParams optimization;map.trainingSetup(optimization);
  const auto n=map.xyz_.size(0),m=bank->size();
  {torch::NoGradGuard guard;map.opacity_.fill_(std::log(.1/.9));map.opacity_.slice(0,0,8).fill_(-12);map.opacity_.slice(0,20,30).fill_(-12);}
  auto all=torch::arange(n,opts.dtype(torch::kLong));
  auto regular=all.index({(all/10).floor()!=2});
  map.addAnchorPruningStats(all);
  {torch::NoGradGuard guard;map.opacity_.slice(0,10,20).fill_(-12);}
  for(int i=0;i<3;++i)map.addAnchorPruningStats(regular);
  map.anchorPruningMask(5);
  TORCH_CHECK(!bank->inactiveSlots().any().item<bool>(),"Insufficient visits must not deactivate");
  map.addAnchorPruningStats(regular);map.anchorPruningMask(5);
  TORCH_CHECK(bank->inactiveSlots().sum().item<int64_t>()==8 && bank->inactiveSlots()[0].slice(0,0,8).all().item<bool>(),
      "Persistent/temporary/unobserved slot distinction failed");
  auto alpha=map.getOpacityActivation();
  auto p=torch::full({n},.2f,opts.requires_grad(true));
  auto probability=selector->maskInactiveSlots(p,alpha);
  auto uniform=torch::zeros_like(p);
  auto hard=sampleSelectorST(probability,uniform);
  TORCH_CHECK(hard.sum().item<int64_t>()==n-8,"Cached sampling retained inactive slots");
  auto loss=selectorRatioLoss(probability,.05);
  TORCH_CHECK(torch::allclose(loss,torch::abs(probability.sum()-.05*n)/(.05*n)),"Budget was renormalized to eligible count");
  loss.backward();
  TORCH_CHECK(p.grad().slice(0,0,8).count_nonzero().item<int64_t>()==0 &&
      p.grad().slice(0,8).abs().sum().item<float>()>0,"Masked budget gradients are incorrect");
  // Trial candidates are pure; the real boundary clears old evidence on recovery.
  auto recovered_alpha=alpha.detach().clone();recovered_alpha[0].fill_(1.f/255.f);
  TORCH_CHECK(selector->maskInactiveSlots(p,recovered_alpha)[0].item<float>()>0 && bank->inactiveSlots()[0][0].item<bool>(),
      "Cutoff equality must be eligible; trial query must not mutate state");
  bank->refreshSlotActivity(recovered_alpha);
  auto visits=torch::full({m},5,opts.dtype(torch::kLong));auto below=torch::full({m,10},5,visits.options());
  auto enough=torch::ones({m},opts.dtype(torch::kBool));
  bank->closeSlotWindow(enough,visits,below,alpha);
  TORCH_CHECK(!bank->inactiveSlots()[0][0].item<bool>(),"Stale pre-recovery evidence reactivated the mask");
  // Only a fresh complete window may qualify it again.
  bank->closeSlotWindow(enough,visits,below,alpha);
  TORCH_CHECK(bank->inactiveSlots()[0][0].item<bool>(),"Fresh complete window did not requalify a low slot");
  auto frozen=selector->cloneFrozen();
  TORCH_CHECK(torch::equal(frozen->anchors()->inactiveSlots(),bank->inactiveSlots()),"Teacher snapshot lost slot state");
  auto before=bank->inactiveSlots().clone();
  auto prune=torch::zeros({n},opts.dtype(torch::kBool));prune.slice(0,10,20).fill_(true);
  {torch::NoGradGuard guard;map.prunePoints(prune);}
  auto keep=torch::arange(m,all.options()).index({torch::arange(m,all.options())!=1});
  TORCH_CHECK(torch::equal(bank->inactiveSlots(),before.index_select(0,keep)),"Whole-block pruning lost slot ownership");
  auto xyz=torch::tensor({{2.01f,.01f,2.01f}},opts),rgb=torch::full_like(xyz,.4);
  auto old=bank->size();{torch::NoGradGuard guard;map.increasePcd(xyz,rgb,100);}
  TORCH_CHECK(bank->size()==old+2&&!bank->inactiveSlots().slice(0,old).any().item<bool>(),"New slots started inactive");
  // The geometry optimizer replacement expects Adam state from a real update.
  auto gaussian_loss=map.xyz_.sum()+map.scaling_.sum()+map.rotation_.sum();
  gaussian_loss.backward();map.optimizer_->step();map.optimizer_->zero_grad(true);
  before=bank->inactiveSlots().clone();map.applyScaledTransformation(1.01f);map.finishAnchorGeometry();
  TORCH_CHECK(torch::equal(before,bank->inactiveSlots()),"Geometry changed UID-owned eligibility");
  torch::serialize::OutputArchive archive;selector->save(archive);archive.save_to("slot_selector.pt");
  auto loaded=improvements::selector::GumbelNetwork(.1,true,false,true);loaded->to(device);
  torch::serialize::InputArchive input;input.load_from("slot_selector.pt",device);loaded->load(input);
  TORCH_CHECK(torch::equal(loaded->anchors()->inactiveSlots(),bank->inactiveSlots()),"Native checkpoint lost slot state");
  torch::serialize::OutputArchive fixture;
  fixture.write("xyz",map.xyz_);fixture.write("scale",map.getScalingActivation());fixture.write("dc",map.features_dc_);
  fixture.write("opacity",map.getOpacityActivation());fixture.write("uniform",map.protection_.cached_uniform);
  for(int j=0;j<4;++j) {
    const float ratio=std::vector<float>{.01f,.05f,.1f,.15f}[j];
    auto raw=selector->forwardAnchorProbability(map.xyz_,ratio,1,{},map.getScalingActivation(),map.features_dc_,map.getOpacityActivation());
    auto masked=selector->maskInactiveSlots(raw,map.getOpacityActivation());
    fixture.write("p"+std::to_string(j),masked);fixture.write("mask"+std::to_string(j),map.protection_.cached_uniform<masked);
  }
  fixture.save_to("slot_fixture.pt");
  std::cout<<"PASS slot deactivation: strict windows, temporary lows, cached sampling, total budget, gradients, recovery, fresh evidence, UID lifecycle, geometry, teacher/checkpoint\n";
}
