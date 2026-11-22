#include "improvements/selector/gumbel_network.h"
#include <iostream>
#include <sstream>

using improvements::selector::GumbelNetwork;
int main(int argc, char** argv) {
  const auto device = argc > 1 && std::string(argv[1]) == "cuda" ? torch::kCUDA : torch::kCPU;
  torch::manual_seed(2026);
  auto opt = torch::TensorOptions().device(device).dtype(torch::kFloat32);
  auto xyz = torch::cat({torch::rand({500,3},opt)*.099 + .0001,
                        torch::full({150,3},.011,opt), -torch::rand({300,3},opt)*.099-.0001});
  auto frozen_xyz = xyz.clone();
  const bool global_xyz = argc > 2 && std::string(argv[2]) == "global_xyz";
  auto model = GumbelNetwork(.1, true, global_xyz); model->to(device);
  model->anchors()->reset(xyz);
  model->anchors()->enableAdaptive();
  auto bank = model->anchors();
  auto scale = torch::full_like(xyz,.02), dc = torch::randn_like(xyz), opacity = torch::full({xyz.size(0),1},.5,opt);
  torch::optim::Adam first(model->parameters(), torch::optim::AdamOptions(.001).amsgrad(true));
  torch::optim::Adam second(model->parameters(), torch::optim::AdamOptions(.001));
  model->forwardAnchorProbability(xyz,.05,1,{},scale,dc,opacity).sum().backward();
  if (global_xyz)
    TORCH_CHECK(model->named_parameters()["local_emd.0.weight"].grad().slice(1,19,22).abs().sum().item<double>()>0,
                "Missing Global XYZ input weight gradient");
  first.step(); second.step(); first.zero_grad(true); second.zero_grad(true);
  const auto old_rows = bank->rows().clone(), old_features = bank->features().detach().clone();
  const auto offset = bank->inputs(xyz).second.clone();
  struct State { torch::Tensor avg, sq, max; int64_t step; };
  std::vector<State> states;
  for (auto* owner : {&first,&second}) {
    auto& s = static_cast<torch::optim::AdamParamState&>(*owner->state().at(bank->features().unsafeGetTensorImpl()));
    states.push_back({s.exp_avg().clone(),s.exp_avg_sq().clone(),s.max_exp_avg_sq().defined()?s.max_exp_avg_sq().clone():torch::Tensor{},s.step()});
  }
  auto before = model->forwardProbabilities(xyz,{},scale,{.01,.05,.1,.15},1,{},dc,opacity);
  const auto logitsFor = [&](float r) {
    const auto input=bank->inputs(xyz);
    auto local = torch::cat({input.first,input.second},1);
    if (global_xyz) local=torch::cat({local,xyz},1);
    const auto joined=torch::cat({local,scale,dc,opacity,torch::full({xyz.size(0),1},r,opt).log()},1);
    return model->named_children()["soft_net"]->as<torch::nn::Linear>()->forward(
        model->named_children()["local_emd"]->as<torch::nn::Sequential>()->forward(joined));
  };
  std::vector<torch::Tensor> old_logits;
  for (float r : {.01f,.05f,.1f,.15f})
    old_logits.push_back(logitsFor(r).detach());
  TORCH_CHECK(bank->refine(xyz,{&first,&second}) > 0,"No refinement");
  auto after = model->forwardProbabilities(xyz,{},scale,{.01,.05,.1,.15},1,{},dc,opacity);
  double maximum_error = 0;
  double maximum_logit_error = 0;
  for (int i=0;i<4;++i) {
    maximum_error = std::max(maximum_error,(after[i]-before[i]).abs().max().item<double>());
    if (!global_xyz) TORCH_CHECK(torch::allclose(after[i],before[i],1e-6,1e-7),"Split score discontinuity");
  }
  const std::vector<float> ratios{.01f,.05f,.1f,.15f};
  for(int i=0;i<4;++i) {
    auto logits=logitsFor(ratios[i]);
    maximum_logit_error=std::max(maximum_logit_error,(logits-old_logits[i]).abs().max().item<double>());
    if (!global_xyz) TORCH_CHECK(torch::allclose(logits,old_logits[i],1e-6,1e-7),"Split logit discontinuity");
  }
  if (global_xyz) {
    auto sizes=.1/torch::pow(2.,bank->levels().to(torch::kFloat32));
    auto keys=bank->named_buffers()["keys"].to(torch::kFloat32);
    auto centers=(keys+.5)*sizes.unsqueeze(1);
    auto expected=(xyz-centers.index_select(0,bank->rows()))/sizes.index_select(0,bank->rows()).unsqueeze(1);
    TORCH_CHECK(torch::equal(bank->centers(),centers) && torch::allclose(bank->inputs(xyz).second,expected,1e-6,1e-7),"Child local center/scale formula");
    TORCH_CHECK(!torch::equal(offset,bank->inputs(xyz).second) && maximum_error>0,"New policy must recompute local offsets");
    TORCH_CHECK(model->named_parameters()["local_emd.0.weight"].size(1)==30,"Global XYZ input width");
    // Move geometry and its reference together: local offset stays constant,
    // so any score change is due to the added world XYZ input.
    const auto old_center=bank->centers().clone();
    const auto local_before=bank->inputs(xyz).second.clone();
    auto p=model->forwardAnchorProbability(xyz,.05,1,{},scale,dc,opacity);
    const auto shift=torch::tensor({.4f,.2f,-.3f},opt);
    {torch::NoGradGuard guard;bank->centers().add_(shift);}
    TORCH_CHECK(torch::allclose(local_before,bank->inputs(xyz+shift).second,1e-5,1e-5),"Global XYZ isolation fixture");
    auto varied=model->forwardAnchorProbability(xyz+shift,.05,1,{},scale,dc,opacity);
    {torch::NoGradGuard guard;bank->centers().copy_(old_center);}
    TORCH_CHECK(!torch::allclose(p,varied,1e-6,1e-7),"Global XYZ did not affect scores");
  } else TORCH_CHECK(torch::equal(offset,bank->inputs(xyz).second),"Split changed original offset reference");
  TORCH_CHECK(torch::equal(old_features.index_select(0,old_rows),bank->inputs(xyz).first),"Feature inheritance");
  for (int j=0;j<2;++j) {
    auto* owner=j?&second:&first;
    auto& s=static_cast<torch::optim::AdamParamState&>(*owner->state().at(bank->features().unsafeGetTensorImpl()));
    TORCH_CHECK(s.step()==states[j].step && torch::equal(s.exp_avg().index_select(0,bank->rows()),states[j].avg.index_select(0,old_rows)) &&
      torch::equal(s.exp_avg_sq().index_select(0,bank->rows()),states[j].sq.index_select(0,old_rows)),"Adam inheritance");
    if(states[j].max.defined()) TORCH_CHECK(torch::equal(s.max_exp_avg_sq().index_select(0,bank->rows()),states[j].max.index_select(0,old_rows)),"AMSGrad inheritance");
    bool alias=false; for(const auto& p:owner->parameters()) alias|=p.is_same(bank->features());
    TORCH_CHECK(alias,"Adam alias not replaced");
  }
  auto counts=torch::bincount(bank->rows(),{},bank->size());
  TORCH_CHECK(bank->levels().max().item<int64_t>()==2 &&
    !((counts>128)&(bank->levels()<2)).any().item<bool>(),"Depth/occupancy cap");
  TORCH_CHECK(((counts>128)&(bank->levels()==2)).any().item<bool>(),"L2 overflow fixture");
  auto frozen=model->cloneFrozen();
  TORCH_CHECK(frozen->anchors()->adaptive() && torch::equal(after[1],frozen->forwardAnchorProbability(xyz,.05,1,{},scale,dc,opacity)),"Adaptive teacher clone");
  after[1].square().mean().backward(); first.step(); first.zero_grad(true); second.zero_grad(true);
  TORCH_CHECK(bank->features().isfinite().all().item<bool>(),"Independent child optimization");
  // Retain only one member, then append a new clone into the same L2 leaf.
  auto survivors=torch::tensor({500},opt.dtype(torch::kLong));
  auto old_level=bank->levels().index_select(0,bank->rows().index_select(0,survivors));
  bank->selectRows(survivors,{&first,&second});
  auto kept=xyz.index_select(0,survivors);
  bank->append(kept,{&first,&second});
  TORCH_CHECK(torch::equal(bank->levels().index_select(0,bank->rows()),old_level.repeat({2})),"Prune/append collapsed tree");
  auto moved=kept.repeat({2,1})+torch::tensor({.2f,0.f,0.f},opt);
  bank->reassociate(moved,{&first,&second}); bank->refine(moved,{&first,&second});
  bank->reassociate(kept.repeat({2,1}),{&first,&second});
  TORCH_CHECK((bank->levels().index_select(0,bank->rows())==2).all().item<bool>(),"Geometry return collapsed tree");
  model->forwardAnchorProbability(kept.repeat({2,1}),.1,1,{},scale.slice(0,0,2),dc.slice(0,0,2),opacity.slice(0,0,2)).sum().backward();
  first.step(); first.zero_grad(true);
  std::stringstream bytes;
  torch::serialize::OutputArchive archive; model->save(archive); archive.save_to(bytes);
  auto restored=GumbelNetwork(.1,true,global_xyz); restored->to(device);
  torch::serialize::InputArchive loaded; loaded.load_from(bytes); restored->load(loaded);
  TORCH_CHECK(restored->anchors()->adaptive() && torch::equal(bank->rows(),restored->anchors()->rows()) &&
    torch::equal(bank->features(),restored->anchors()->features()),"Adaptive archive roundtrip");
  TORCH_CHECK(torch::equal(xyz,frozen_xyz),"Octree modified explicit geometry");
  auto sparse=improvements::selector::AnchorBank(.1); sparse->to(device); sparse->enableAdaptive();
  auto identical=torch::full({129,3},.011,opt);
  sparse->reset(identical.slice(0,0,128)); sparse->refine(identical.slice(0,0,128));
  TORCH_CHECK(sparse->size()==1,"128 rows must not split");
  sparse->append(identical.slice(0,128)); sparse->refine(identical);
  TORCH_CHECK(sparse->size()==3,"Created empty octree siblings");
  sparse->append(torch::full({1,3},.09,opt));
  TORCH_CHECK(sparse->size()==4 && sparse->levels().index_select(0,sparse->rows()).select(0,129).item<int64_t>()==1,
              "Append did not descend existing split path/create only occupied sibling");
  torch::serialize::OutputArchive exported; frozen->save(exported); exported.save_to("octree_selector_test.pt");
  torch::serialize::OutputArchive probe; probe.write("xyz",xyz);probe.write("scale",scale);probe.write("sh_dc",dc);probe.write("opacity",opacity);
  for(int i=0;i<4;++i)probe.write("probability_"+std::to_string(i),after[i].detach());
  probe.save_to("octree_selector_probe.pt");
  std::cout<<"{\"passed\":true,\"max_probability_error\":"<<maximum_error<<",\"max_logit_error\":"<<maximum_logit_error<<",\"population\":"<<xyz.size(0)<<",\"max_level\":2,\"two_adam_owners\":true,\"l2_overflow_retained\":true}\n";
}
