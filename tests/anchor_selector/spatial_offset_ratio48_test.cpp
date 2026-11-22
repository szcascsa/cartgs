#include "improvements/selector/gumbel_network.h"
#include <iostream>

using improvements::selector::GumbelNetwork;

int main(int argc,char** argv) {
  torch::manual_seed(2026);
  const torch::Device device=argc>1&&std::string(argv[1])=="cpu"?torch::kCPU:torch::kCUDA;
  const std::string path=argc>2?argv[2]:"offset_ratio48_selector.pt";
  const std::vector<double> knots{.01,.05,.1,.2,.3,.4},queries{.005,.01,.025,.05,.075,.1,.15,.2,.25,.3,.35,.4,.6};
  auto net=GumbelNetwork(.1,true,false,true,std::vector<double>{},0);net->to(device);
  auto bank=net->anchors();bank->enableVariableMembers();bank->enableSlotDeactivation();
  net->enableSpatialCompensation(false,false,0,knots,true,false,true,false,48);
  auto spatial=net->spatialCompensation();spatial->enablePhysicalScaleAdditive();
  TORCH_CHECK(spatial->offsetOnlyInput()&&spatial->ratioFeatureWidth()==48&&spatial->ratioCodeInput()&&
      !spatial->ratio_gate&&!spatial->ratio_net&&!bank->spatialFeatures(),"Wrong ratio48 layout");
  int64_t count=0;for(auto p:spatial->parameters())count+=p.numel();
  TORCH_CHECK(count==25230&&spatial->encode->weight.size(1)==51,"Wrong 54D network parameter count");
  torch::optim::Adam optimizer(net->parameters(),torch::optim::AdamOptions(.001));
  auto opts=torch::TensorOptions().device(device);
  auto observed=torch::tensor({{.02f,.03f,2.02f},{.04f,.03f,2.03f},{.22f,.03f,2.02f}},opts);
  auto seeds=bank->observeBlocks(observed,torch::full_like(observed,.4),{&optimizer});
  auto xyz=seeds.xyz,scale=seeds.sigma.log(),dc=seeds.color.unsqueeze(1),logits=torch::zeros({xyz.size(0),1},opts);
  auto q=torch::zeros({xyz.size(0),4},opts);q.select(1,0).fill_(1);
  TORCH_CHECK(bank->compensationFeatureTables().size()==6,"Expected six ratio Feature tables");
  for(auto table:bank->compensationFeatureTables())
    TORCH_CHECK(table.is_leaf()&&table.requires_grad()&&table.size(1)==48&&
        torch::equal(table,bank->compensationFeatureTables()[0])&&table.count_nonzero().item<int64_t>()>0,
        "Initial Feature must be random and identical across ratios");
  auto run=[&](GumbelNetwork& model,double r,const torch::Tensor& position) {
    auto a=model->anchors();auto owners=a->rows();
    return model->spatialCompensation()->forward(position,scale,q,dc,logits,a->centers().index_select(0,owners),
        a->offsetScales().index_select(0,owners),r,true,a->compensationInput({},r));
  };
  auto owners=bank->rows();
  spatial->initialize(xyz,scale,q,dc,logits,bank->centers().index_select(0,owners),bank->offsetScales().index_select(0,owners),true);
  TORCH_CHECK(run(net,.1,xyz).count_nonzero().item<int64_t>()==0,"Step0 must be identity");
  for(int step=0;step<24;++step) {
    auto tables=bank->compensationFeatureTables();std::vector<torch::Tensor> before;
    for(auto table:tables)before.push_back(table.detach().clone());
    const int active=step%6;optimizer.zero_grad(true);
    (run(net,knots[active],xyz)-.03).square().mean().backward();
    for(int j=0;j<6;++j)TORCH_CHECK(tables[j].grad().defined()==(j==active),"Inactive ratio gradient leaked");
    TORCH_CHECK(!net->named_parameters()["local_emd.0.weight"].grad().defined(),"Compensation leaked into Selector");
    optimizer.step();
    for(int j=0;j<6;++j)if(j!=active)TORCH_CHECK(torch::equal(tables[j],before[j]),"Inactive ratio Adam moved Feature");
  }
  for(auto table:bank->compensationFeatureTables())
    TORCH_CHECK(static_cast<torch::optim::AdamParamState&>(*optimizer.state().at(table.unsafeGetTensorImpl())).step()==4,"Ratio Adam clock mismatch");
  optimizer.zero_grad(true);auto interpolated=bank->compensationInput({},.075);
  TORCH_CHECK(torch::allclose(interpolated,.5*bank->compensationInput({},.05)+.5*bank->compensationInput({},.1)),"Raw-ratio interpolation mismatch");
  TORCH_CHECK(torch::equal(bank->compensationInput({},.005),bank->compensationInput({},.01))&&
      torch::equal(bank->compensationInput({},.6),bank->compensationInput({},.4)),"Endpoint clamp mismatch");
  interpolated.sum().backward();
  for(int j=0;j<6;++j)TORCH_CHECK(bank->compensationFeatureTables()[j].grad().defined()==(j==1||j==2),"Interpolation gradient did not reach exactly two tables");
  optimizer.zero_grad(true);auto position=xyz.detach().clone().requires_grad_(true);
  run(net,.15,position).square().mean().backward();
  TORCH_CHECK(position.grad().isfinite().all().item<bool>()&&position.grad().abs().sum().item<float>()>0,"Master XYZ gradient missing");
  TORCH_CHECK(spatial->ratio_projection->weight.grad().abs().sum().item<float>()>0,"Continuous ratio gradient missing");
  auto expected=run(net,.15,xyz);
  auto shifted=spatial->forward(xyz,scale+2,q.roll({1},{1}),dc+3,logits-4,
      bank->centers().index_select(0,owners),bank->offsetScales().index_select(0,owners),.15,true,bank->compensationInput({},.15));
  TORCH_CHECK(torch::equal(expected,shifted),"Removed physical attributes still enter compensation");
  auto teacher=GumbelNetwork(net->cloneFrozen());TORCH_CHECK(torch::equal(expected,run(teacher,.15,xyz)),"Teacher output mismatch");
  torch::serialize::OutputArchive saved;net->save(saved);saved.save_to(path);
  auto loaded=GumbelNetwork(.1,true,false,true,std::vector<double>{},0);loaded->to(device);
  torch::serialize::InputArchive archive;archive.load_from(path,device);loaded->load(archive);
  TORCH_CHECK(loaded->spatialCompensation()->ratioFeatureWidth()==48&&loaded->spatialCompensation()->physicalScaleAdditive(),"Checkpoint schema lost ratio48 or physical Scale");
  std::vector<torch::Tensor> outputs;
  for(auto r:queries) {auto out=run(net,r,xyz);TORCH_CHECK(torch::equal(out,run(loaded,r,xyz)),"Checkpoint output mismatch");outputs.push_back(out.detach());}
  torch::serialize::OutputArchive fixture;
  fixture.write("xyz",xyz);fixture.write("scale",scale);fixture.write("rotation",q);fixture.write("dc",dc);fixture.write("logits",logits);
  fixture.write("ratios",torch::tensor(queries,torch::kFloat64));fixture.write("outputs",torch::stack(outputs));fixture.save_to(path+".fixture");
  optimizer.zero_grad(true);auto ids=bank->blockUIDs().cpu().clone();const auto old_size=bank->size();
  std::vector<torch::Tensor> values,moments;
  for(auto table:bank->compensationFeatureTables()) {
    values.push_back(table.detach().clone());moments.push_back(static_cast<torch::optim::AdamParamState&>(*optimizer.state().at(table.unsafeGetTensorImpl())).exp_avg().clone());
  }
  bank->observeBlocks(torch::tensor({{.61f,.02f,2.03f}},opts),torch::full({1,3},.5,opts),{&optimizer});
  TORCH_CHECK(bank->size()>old_size,"Append did not create Anchor");
  auto tables=bank->compensationFeatureTables();
  for(int j=0;j<6;++j) {
    auto& state=static_cast<torch::optim::AdamParamState&>(*optimizer.state().at(tables[j].unsafeGetTensorImpl()));
    TORCH_CHECK(torch::equal(tables[j].slice(0,0,old_size),values[j])&&torch::equal(state.exp_avg().slice(0,0,old_size),moments[j])&&
        state.exp_avg().slice(0,old_size).count_nonzero().item<int64_t>()==0&&
        torch::equal(tables[j].slice(0,old_size),tables[0].slice(0,old_size)),"Append Feature/Adam state mismatch");
  }
  auto keep=torch::nonzero(bank->rows()!=0).reshape({-1});bank->pruneVariableMembers(keep,{&optimizer});
  auto alive=bank->blockUIDs().cpu();
  for(int j=0;j<6;++j)for(int64_t row=0;row<alive.numel();++row)for(int64_t old=0;old<ids.numel();++old)
    if(alive[row].item<int64_t>()==ids[old].item<int64_t>()) {
      auto table=bank->compensationFeatureTables()[j];auto& state=static_cast<torch::optim::AdamParamState&>(*optimizer.state().at(table.unsafeGetTensorImpl()));
      TORCH_CHECK(torch::equal(table[row],values[j][old])&&torch::equal(state.exp_avg()[row],moments[j][old]),"Prune UID/Adam mismatch");
    }
  optimizer.zero_grad(true);bank->compensationInput({},.05).square().sum().backward();optimizer.step();
  std::cout<<"OFFSET_RATIO48_PASS identity interpolation gradients inactive_adam master selector_isolation teacher checkpoint append_prune_uid_adam\n";
}
