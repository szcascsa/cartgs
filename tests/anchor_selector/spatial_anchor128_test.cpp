#include "improvements/selector/gumbel_network.h"
#include <iostream>
using improvements::selector::GumbelNetwork;
int main(int argc,char** argv) {
  torch::manual_seed(2026);
  const torch::Device device=argc>1&&std::string(argv[1])=="cpu"?torch::kCPU:torch::kCUDA;
  const std::string path=argc>2?argv[2]:"anchor128_selector.pt";
  const bool shared_selector=argc>3&&std::string(argv[3])=="shared128";
  const bool bounded_xyz=shared_selector||(argc>3&&std::string(argv[3])=="bound2");
  const bool offset_only=bounded_xyz||(argc>3&&std::string(argv[3])=="offset");
  const std::vector<double> ratios{.01,.05,.1,.2,.3,.4},queries{.01,.025,.05,.075,.1,.15,.2,.25,.3,.35,.4};
  auto net=GumbelNetwork(.1,true,false,true,std::vector<double>{},0);net->to(device);
  auto bank=net->anchors();bank->enableVariableMembers();bank->enableSlotDeactivation();
  net->enableSpatialCompensation(false,false,0,ratios,true,true,offset_only,bounded_xyz);auto spatial=net->spatialCompensation();
  TORCH_CHECK(spatial->sharedAnchor128()&&!spatial->ratio_net&&!spatial->early_feature,"Wrong Anchor128 layout");
  int64_t count=0;for(auto p:spatial->parameters())count+=p.numel();TORCH_CHECK(count==(offset_only?34702:35726),"Wrong shared parameter count");
  if(shared_selector)net->enableSelectorSpatialFeatures();
  torch::optim::Adam optimizer(net->parameters(),torch::optim::AdamOptions(.001));auto opts=torch::TensorOptions().device(device);
  auto observed=torch::tensor({{.02f,.03f,2.02f},{.04f,.03f,2.03f},{.22f,.03f,2.02f}},opts);
  auto seeds=bank->observeBlocks(observed,torch::full_like(observed,.4),{&optimizer});
  auto xyz=seeds.xyz,scale=seeds.sigma.log(),dc=seeds.color.unsqueeze(1),logits=torch::zeros({xyz.size(0),1},opts);
  auto q=torch::zeros({xyz.size(0),4},opts);q.select(1,0).fill_(1);
  TORCH_CHECK(bank->spatialFeaturesTable().size(1)==128&&!bank->compensationFeatures(),"Feature table layout mismatch");
  auto run=[&](GumbelNetwork& model,double r,const torch::Tensor& position) {
    auto a=model->anchors();auto owners=a->rows();return model->spatialCompensation()->forward(position,scale,q,dc,logits,a->centers().index_select(0,owners),a->offsetScales().index_select(0,owners),r,true,a->spatialFeaturesTable().index_select(0,owners));
  };
  spatial->initialize(xyz,scale,q,dc,logits,bank->centers().index_select(0,bank->rows()),bank->offsetScales().index_select(0,bank->rows()),true);
  TORCH_CHECK(run(net,.1,xyz).count_nonzero().item<int64_t>()==0,"Step0 must be identity");
  if(bounded_xyz) {
    auto& bias=spatial->heads->at<torch::nn::SequentialImpl>("xyz").at<torch::nn::LinearImpl>(2).bias;
    auto reference=bank->offsetScales().index_select(0,bank->rows());
    optimizer.zero_grad(true);run(net,.1,xyz).slice(1,0,3).sum().backward();
    TORCH_CHECK(torch::allclose(bias.grad(),reference.sum().expand({3}),1e-5,1e-6),"Bound must preserve the unit derivative at zero");
    for(const auto magnitude:{.01f,1.f,10.f,10000.f}) {
      {torch::NoGradGuard guard;bias.copy_(torch::tensor({magnitude,2*magnitude,-3*magnitude},opts));}
      auto out=run(net,.1,xyz).slice(1,0,3)/reference;
      TORCH_CHECK(out.isfinite().all().item<bool>()&&out.norm(2,1).max().item<float>()<=2.000002f,"XYZ vector norm exceeded 2 Anchor units");
      TORCH_CHECK(torch::allclose(out.slice(1,1,2),2*out.slice(1,0,1))&&torch::allclose(out.slice(1,2,3),-3*out.slice(1,0,1)),"Radial bound changed direction");
    }
    {torch::NoGradGuard guard;bias.zero_();}optimizer.zero_grad(true);
  }
  for(int step=0;step<24;++step) {
    optimizer.zero_grad(true);auto out=run(net,ratios[step%6],xyz);(out-.03).square().mean().backward();
    TORCH_CHECK(!net->named_parameters()["local_emd.0.weight"].grad().defined(),"Compensation leaked into Selector");optimizer.step();
  }
  optimizer.zero_grad(true);auto position=xyz.detach().clone().requires_grad_(true);run(net,.15,position).square().mean().backward();
  TORCH_CHECK(position.grad().isfinite().all().item<bool>()&&position.grad().abs().sum().item<float>()>0,"Master XYZ gradient missing");
  TORCH_CHECK(spatial->ratio_gate->at<torch::nn::LinearImpl>(2).weight.grad().abs().sum().item<float>()>0,"Ratio gate gradient missing");
  if(offset_only) {
    auto owners=bank->rows(),centers=bank->centers().index_select(0,owners),reference=bank->offsetScales().index_select(0,owners);
    auto out=run(net,.15,xyz);
    TORCH_CHECK(torch::equal(out,spatial->forward(xyz,scale+2,q.roll({1},{1}),dc+3,logits-4,centers,reference,.15,true,bank->spatialFeaturesTable().index_select(0,owners))),"Removed attributes still affect compensation");
    TORCH_CHECK(torch::allclose(out,spatial->forward(xyz+1,scale,q,dc,logits,centers+1,reference,.15,true,bank->spatialFeaturesTable().index_select(0,owners)),1e-5,1e-6),"Global XYZ leaked into compensation");
  }
  if(shared_selector) {
    for(int step=0;step<12;++step) {
      optimizer.zero_grad(true);
      auto z=net->forwardLogits(xyz,q,scale.exp(),.1,{},dc,logits.sigmoid());
      (z.select(1,1)-torch::linspace(-.3,.4,xyz.size(0),opts)).square().mean().backward();optimizer.step();
    }
    optimizer.zero_grad(true);net->forwardLogits(xyz,q,scale.exp(),.1,{},dc,logits.sigmoid()).square().sum().backward();
    TORCH_CHECK(bank->spatialFeaturesTable().grad().abs().sum().item<float>()>0,"Selector cannot update the shared Feature table");
    TORCH_CHECK(!spatial->encode->weight.grad().defined(),"Selector loss leaked into Compensation encoder");
  }
  auto teacher=GumbelNetwork(net->cloneFrozen());TORCH_CHECK(teacher->spatialCompensation()->sharedAnchor128(),"Teacher lost gate");
  TORCH_CHECK(torch::equal(run(net,.15,xyz),run(teacher,.15,xyz)),"Teacher output mismatch");
  if(shared_selector)TORCH_CHECK(torch::equal(net->forwardLogits(xyz,q,scale.exp(),.15,{},dc,logits.sigmoid()),teacher->forwardLogits(xyz,q,scale.exp(),.15,{},dc,logits.sigmoid())),"Teacher lost shared Selector projection");
  torch::serialize::OutputArchive saved;net->save(saved);saved.save_to(path);
  auto loaded=GumbelNetwork(.1,true,false,true,std::vector<double>{},0);loaded->to(device);torch::serialize::InputArchive archive;archive.load_from(path,device);loaded->load(archive);
  std::vector<torch::Tensor> outputs;
  for(auto r:queries){auto out=run(net,r,xyz);TORCH_CHECK(torch::equal(out,run(loaded,r,xyz)),"Checkpoint mismatch");outputs.push_back(out.detach());}
  torch::serialize::OutputArchive fixture;fixture.write("xyz",xyz);fixture.write("scale",scale);fixture.write("rotation",q);fixture.write("dc",dc);fixture.write("logits",logits);fixture.write("ratios",torch::tensor(queries,torch::kFloat64));fixture.write("outputs",torch::stack(outputs));fixture.save_to(path+".fixture");
  if(shared_selector) {
    std::vector<torch::Tensor> z;
    for(auto r:queries) {auto v=net->forwardLogits(xyz,q,scale.exp(),r,{},dc,logits.sigmoid());TORCH_CHECK(torch::equal(v,loaded->forwardLogits(xyz,q,scale.exp(),r,{},dc,logits.sigmoid())),"Shared Selector checkpoint mismatch");z.push_back(v.detach());}
    torch::serialize::OutputArchive selection;selection.write("outputs",torch::stack(z));selection.save_to(path+".selection");
  }
  optimizer.zero_grad(true);
  const auto old_size=bank->size();auto before=bank->spatialFeaturesTable().detach().clone();auto uids=bank->blockUIDs().cpu().clone();
  auto& state=static_cast<torch::optim::AdamParamState&>(*optimizer.state().at(bank->spatialFeaturesTable().unsafeGetTensorImpl()));
  auto moment=state.exp_avg().clone(),moment2=state.exp_avg_sq().clone();const auto clock=state.step();
  bank->observeBlocks(torch::tensor({{.61f,.02f,2.03f}},opts),torch::full({1,3},.5,opts),{&optimizer});
  auto table=bank->spatialFeaturesTable();auto& appended=static_cast<torch::optim::AdamParamState&>(*optimizer.state().at(table.unsafeGetTensorImpl()));
  TORCH_CHECK(table.size(0)>old_size&&torch::equal(table.slice(0,0,old_size),before)&&table.slice(0,old_size).count_nonzero().item<int64_t>()>0,"Append Feature initialization incorrect");
  TORCH_CHECK(torch::equal(appended.exp_avg().slice(0,0,old_size),moment)&&appended.exp_avg().slice(0,old_size).count_nonzero().item<int64_t>()==0&&appended.exp_avg_sq().slice(0,old_size).count_nonzero().item<int64_t>()==0&&appended.step()==clock,"Append Adam migration incorrect");
  auto keep=torch::nonzero(bank->rows()!=0).reshape({-1});bank->pruneVariableMembers(keep,{&optimizer});
  auto alive=bank->blockUIDs().cpu();auto& pruned=static_cast<torch::optim::AdamParamState&>(*optimizer.state().at(bank->spatialFeaturesTable().unsafeGetTensorImpl()));
  for(int64_t row=0;row<alive.numel();++row)for(int64_t old=0;old<uids.numel();++old)if(alive[row].item<int64_t>()==uids[old].item<int64_t>())
    TORCH_CHECK(torch::equal(bank->spatialFeaturesTable()[row],before[old])&&torch::equal(pruned.exp_avg()[row],moment[old])&&torch::equal(pruned.exp_avg_sq()[row],moment2[old]),"Prune UID/Adam mismatch");
  optimizer.zero_grad(true);bank->spatialFeaturesTable().square().sum().backward();optimizer.step();
  std::cout<<"ANCHOR128_PASS live_gradient gate selector_isolation teacher checkpoint append_prune_uid_adam_next_step offset_only="<<offset_only<<" bounded_xyz="<<bounded_xyz<<" shared_selector="<<shared_selector<<"\n";
}
