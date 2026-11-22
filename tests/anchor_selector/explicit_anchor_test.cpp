#include "improvements/selector/gumbel_network.h"
#include <iostream>

int main(int argc,char** argv) {
  torch::manual_seed(2026);
  auto device=argc>1&&std::string(argv[1])=="cuda"?torch::kCUDA:torch::kCPU;
  auto options=torch::TensorOptions().dtype(torch::kFloat32).device(device);
  auto model=improvements::selector::GumbelNetwork(.1,true,false,true);model->to(device);
  auto bank=model->anchors();
  auto observed=torch::tensor({{.011f,.012f,.013f},{.031f,.032f,.033f},{-.11f,.02f,.03f}},options);
  auto colors=torch::full_like(observed,.4);
  auto seeds=bank->observeBlocks(observed,colors);
  auto xyz=seeds.xyz, scale=seeds.sigma,dc=seeds.color.unsqueeze(1),opacity=torch::full({xyz.size(0),1},.067,options);
  bank->assertBlocks(xyz.size(0));
  TORCH_CHECK(bank->size()==4&&xyz.size(0)==40,"Multiscale initialization/dedup");
  TORCH_CHECK(bank->features().count_nonzero().item<int64_t>()==0,"Initial features must be zero");
  for(int64_t a=0;a<bank->size();++a)
    TORCH_CHECK((xyz.slice(0,a*10,a*10+10)-xyz[a*10]).abs().sum().item<double>()>0,"Sparse seeds were identical");
  TORCH_CHECK(bank->observeBlocks(observed,colors).xyz.size(0)==0,"Duplicate observation appended blocks");
  torch::optim::Adam first(model->parameters(),torch::optim::AdamOptions(.001).amsgrad(true));
  torch::optim::Adam second(model->parameters(),torch::optim::AdamOptions(.001));
  model->forwardAnchorProbability(xyz,.05,1,{},scale,dc,opacity).sum().backward();
  TORCH_CHECK(bank->features().grad().abs().sum().item<double>()>0,"Feature gradient missing");
  first.step();second.step();first.zero_grad(true);second.zero_grad(true);
  {torch::NoGradGuard guard;bank->features().fill_(-2);bank->features()[1].fill_(-1);}
  const auto old=bank->features().detach().clone(),old_ids=bank->blockUIDs().clone();
  std::vector<torch::Tensor> moments;
  for(auto* owner:{&first,&second})moments.push_back(static_cast<torch::optim::AdamParamState&>(
      *owner->state().at(bank->features().unsafeGetTensorImpl())).exp_avg().clone());
  auto changed=xyz.clone();changed[0].copy_(torch::tensor({.21f,.01f,.01f},options));
  changed[1].copy_(torch::tensor({.31f,.01f,.01f},options));
  changed[10].copy_(torch::tensor({.21f,.01f,.01f},options));
  auto grad=torch::zeros({xyz.size(0),1},options),denom=torch::ones_like(grad);
  grad[0]=1.5;grad[1]=3;grad[10]=1.5;
  auto grown=bank->growBlocks(changed,dc.squeeze(1),grad,denom,1,2,{&first,&second});
  TORCH_CHECK(grown.xyz.size(0)>=20,"Same-level and fine-level growth not triggered");
  TORCH_CHECK(torch::equal(bank->features().slice(0,0,old.size(0)),old),"Growth changed old Feature");
  TORCH_CHECK(torch::equal(bank->blockUIDs().slice(0,0,old_ids.numel()),old_ids),"Growth changed UID");
  for(size_t i=0;i<2;++i) {
    auto* owner=i==0?&first:&second;
    auto& state=static_cast<torch::optim::AdamParamState&>(*owner->state().at(bank->features().unsafeGetTensorImpl()));
    TORCH_CHECK(torch::equal(state.exp_avg().slice(0,0,old.size(0)),moments[i])&&
        state.exp_avg().slice(0,old.size(0)).count_nonzero().item<int64_t>()==0&&
        state.exp_avg_sq().slice(0,old.size(0)).count_nonzero().item<int64_t>()==0,"Adam migration");
  }
  TORCH_CHECK((bank->levels()==2).any().item<bool>() || (bank->levels()==1).sum().item<int64_t>()>2,"Coarse occupancy blocked fine growth");
  bool negative_inherited=false,zero_fine=false;
  for(int64_t j=old.size(0);j<bank->size();++j) {
    negative_inherited|=(bank->features()[j]<0).all().item<bool>();
    zero_fine|=bank->features()[j].count_nonzero().item<int64_t>()==0;
  }
  TORCH_CHECK(negative_inherited&&zero_fine,"Same max aggregation/fine zero initialization");
  TORCH_CHECK(torch::equal(bank->features()[old.size(0)],old[1]),"Per-channel max failed for all-negative sources");
  xyz=torch::cat({changed,grown.xyz});scale=torch::cat({scale,grown.sigma});
  dc=torch::cat({dc,grown.color.unsqueeze(1)});opacity=torch::full({xyz.size(0),1},.067,options);
  auto l1=torch::nonzero(bank->levels()==1).reshape({-1})[0].item<int64_t>()*10;
  xyz[l1].copy_(torch::tensor({.71f,.01f,.01f},options));
  grad=torch::zeros({xyz.size(0),1},options);denom=torch::ones_like(grad);grad[l1]=5;
  auto finest=bank->growBlocks(xyz,dc.squeeze(1),grad,denom,1,2,{&first,&second});
  TORCH_CHECK(finest.xyz.size(0)==10&&(bank->levels()==2).any().item<bool>(),"L2 growth failed");
  xyz=torch::cat({xyz,finest.xyz});scale=torch::cat({scale,finest.sigma});dc=torch::cat({dc,finest.color.unsqueeze(1)});
  opacity=torch::full({xyz.size(0),1},.067,options);
  grad=torch::zeros({xyz.size(0),1},options);denom=torch::ones_like(grad);grad[xyz.size(0)-1]=100;
  TORCH_CHECK(bank->growBlocks(xyz,dc.squeeze(1),grad,denom,1,2,{&first,&second}).xyz.size(0)==0,"Exceeded max L2");
  model->forwardAnchorProbability(xyz,.1,1,{},scale,dc,opacity).sum().backward();
  first.step();first.zero_grad(true);
  auto before=model->forwardAnchorProbability(xyz,.05,1,{},scale,dc,opacity).detach();
  auto frozen=improvements::selector::GumbelNetwork(model->cloneFrozen());
  TORCH_CHECK(torch::equal(before,frozen->forwardAnchorProbability(xyz,.05,1,{},scale,dc,opacity)),"Frozen clone layout");
  auto mask=torch::zeros({xyz.size(0)},options.dtype(torch::kBool));mask[0]=true;
  TORCH_CHECK(bank->wholeBlockPrune(mask).sum().item<int64_t>()==0,"Single member deleted whole block");
  mask.slice(0,0,10).fill_(true);auto removed=bank->wholeBlockPrune(mask);
  bank->pruneBlocks(~removed,{&first,&second});
  xyz=xyz.index({~removed});scale=scale.index({~removed});dc=dc.index({~removed});opacity=opacity.index({~removed});
  bank->assertBlocks(xyz.size(0));
  const auto uids=bank->blockUIDs().clone(),features=bank->features().detach().clone();
  const auto refs=bank->centers().clone();
  auto same_level=torch::nonzero(bank->levels()==bank->levels()[0]).reshape({-1});
  const auto collision_target=same_level[1].item<int64_t>();
  bank->beginBlockGeometry();
  GaussianTransformResult correction;correction.indices=torch::arange(10,options.dtype(torch::kLong));
  correction.transform_4x4=torch::eye(4,options);correction.rotation_wxyz=torch::tensor({1.f,0.f,0.f,0.f},options);
  correction.scale=1;correction.changed_attributes=toBits(GaussianAttrFlag::XYZ);
  const auto offset_before=bank->inputs(xyz).second.slice(0,0,10).clone();
  auto shift=refs[collision_target]-refs[0];
  correction.transform_4x4[3].slice(0,0,3).copy_(shift);
  xyz.slice(0,0,10).add_(shift);
  bank->transportBlocks(correction);bank->finishBlockGeometry();
  TORCH_CHECK(torch::allclose(bank->inputs(xyz).second.slice(0,0,10),offset_before,1e-4,1e-5),"Block transport lost offset relationship");
  TORCH_CHECK(torch::allclose(bank->centers()[0],bank->centers()[collision_target])&&
      torch::equal(bank->named_buffers()["keys"][0],bank->named_buffers()["keys"][collision_target]),"Same-level collision fixture");
  TORCH_CHECK(torch::equal(uids,bank->blockUIDs())&&torch::equal(features,bank->features()),"Collision merged blocks or Feature");
  bank->assertBlocks(xyz.size(0));
  const auto mixed_reference=bank->centers()[0].clone();
  bank->beginBlockGeometry();
  correction.indices=torch::arange(3,options.dtype(torch::kLong));
  correction.transform_4x4=torch::eye(4,options);correction.transform_4x4[3][0]=.2;
  bank->transportBlocks(correction);
  correction.indices=torch::arange(3,10,options.dtype(torch::kLong));correction.transform_4x4[3][0]=-.1;
  bank->transportBlocks(correction);bank->finishBlockGeometry();
  TORCH_CHECK(torch::allclose(bank->centers()[0],mixed_reference+torch::tensor({-.01f,0.f,0.f},options),1e-5,1e-6),
              "Mixed correction groups did not transport one stable block reference");
  auto probe=bank->centers().slice(0,0,1).clone(),rgb=torch::full_like(probe,.4);
  auto n=bank->size();bank->observeBlocks(probe,rgb,{&first,&second});
  TORCH_CHECK(bank->size()<=n+1,"Occupied collision bucket did not deduplicate");
  // Save/load the actual layout; used by the Python/native parity gate.
  torch::serialize::OutputArchive archive;model->save(archive);archive.save_to("explicit_selector.pt");
  auto positions=bank->centers().index_select(0,bank->rows())+.13*bank->named_buffers()["offset_scales"].index_select(0,bank->rows());
  auto scales=torch::full_like(positions,.01),sh=torch::full_like(positions,.2),op=torch::full({positions.size(0),1},.1,options);
  torch::serialize::OutputArchive fixture;
  fixture.write("xyz",positions);fixture.write("scale",scales);fixture.write("dc",sh);fixture.write("opacity",op);
  int index=0;for(float ratio:{.01f,.05f,.1f,.15f})
    fixture.write("p"+std::to_string(index++),model->forwardAnchorProbability(positions,ratio,1,{},scales,sh,op).detach());
  fixture.save_to("explicit_fixture.pt");
  auto reloaded=improvements::selector::GumbelNetwork(.1,true,false,true);reloaded->to(device);
  torch::serialize::InputArchive input;input.load_from("explicit_selector.pt",device);reloaded->load(input);
  reloaded->anchors()->assertBlocks(bank->rows().numel());
  TORCH_CHECK(torch::equal(reloaded->anchors()->blockUIDs(),bank->blockUIDs()),"UID round trip");
  std::cout<<"PASS explicit anchors: ten slots, dedup, growing, features, two Adam owners, prune, collision, checkpoint\n";
}
