#include "improvements/selector/gumbel_network.h"
#include <iostream>
using improvements::selector::GumbelNetwork;
int main(int argc,char** argv) {
  torch::manual_seed(2026);
  const torch::Device device=argc>1&&std::string(argv[1])=="cpu"?torch::kCPU:torch::kCUDA;
  const std::string path=argc>2?argv[2]:"ratio54_selector.pt";
  const std::vector<double> knots{.01,.05,.1,.2,.3,.5},queries{.01,.025,.05,.075,.1,.15,.2,.25,.3,.4,.5};
  auto net=GumbelNetwork(.1,true,false,true,std::vector<double>{},0);net->to(device);
  auto bank=net->anchors();bank->enableVariableMembers();bank->enableSlotDeactivation();
  net->enableSpatialCompensation(false,false,0,knots,true);auto spatial=net->spatialCompensation();
  TORCH_CHECK(spatial->ratioCodeInput()&&!spatial->ratio_net&&!spatial->early_feature,"Wrong Ratio54 layout");
  int64_t count=0;for(auto p:spatial->parameters())count+=p.numel();TORCH_CHECK(count==25230,"Wrong shared parameter count");
  torch::optim::Adam optimizer(net->parameters(),torch::optim::AdamOptions(.001));auto opts=torch::TensorOptions().device(device);
  auto observed=torch::tensor({{.02f,.03f,2.02f},{.04f,.03f,2.03f},{.22f,.03f,2.02f}},opts);
  auto seeds=bank->observeBlocks(observed,torch::full_like(observed,.4),{&optimizer});
  auto xyz=seeds.xyz,scale=seeds.sigma.log(),dc=seeds.color.unsqueeze(1),logits=torch::zeros({xyz.size(0),1},opts);
  auto q=torch::zeros({xyz.size(0),4},opts);q.select(1,0).fill_(1);auto owners=bank->rows();
  spatial->initialize(xyz,scale,q,dc,logits,bank->centers().index_select(0,owners),bank->offsetScales().index_select(0,owners),true);
  auto run=[&](GumbelNetwork& model,double ratio,const torch::Tensor& position) {
    auto a=model->anchors();return model->spatialCompensation()->forward(position,scale,q,dc,logits,a->centers().index_select(0,a->rows()),a->offsetScales().index_select(0,a->rows()),ratio,true,a->compensationInput({},ratio));
  };
  TORCH_CHECK(run(net,.1,xyz).count_nonzero().item<int64_t>()==0,"Step0 must be identity");
  for(int step=0;step<24;++step) {
    const auto j=step%6;optimizer.zero_grad(true);auto out=run(net,knots[j],xyz);(out-.03).square().mean().backward();
    for(int k=0;k<6;++k)if(k!=j)TORCH_CHECK(!bank->compensationFeatureTables()[k].grad().defined(),"Inactive ratio gradient leaked");
    TORCH_CHECK(!net->named_parameters()["local_emd.0.weight"].grad().defined(),"Selector gradient leaked");optimizer.step();
  }
  for(auto table:bank->compensationFeatureTables())TORCH_CHECK(static_cast<torch::optim::AdamParamState&>(*optimizer.state().at(table.unsafeGetTensorImpl())).step()==4,"Ratio Adam clock mismatch");
  optimizer.zero_grad(true);auto position=xyz.detach().clone().requires_grad_(true);run(net,.15,position).square().mean().backward();
  TORCH_CHECK(position.grad().defined()&&position.grad().isfinite().all().item<bool>()&&position.grad().abs().sum().item<float>()>0,"Master XYZ gradient missing");
  TORCH_CHECK(spatial->ratio_projection->weight.grad().defined()&&spatial->ratio_projection->weight.grad().abs().sum().item<float>()>0,"Ratio projection gradient missing");
  auto teacher=GumbelNetwork(net->cloneFrozen());TORCH_CHECK(teacher->spatialCompensation()->ratioCodeInput(),"Teacher lost ratio code");
  TORCH_CHECK(torch::equal(run(net,.15,xyz),run(teacher,.15,xyz)),"Teacher output mismatch");
  torch::serialize::OutputArchive saved;net->save(saved);saved.save_to(path);
  auto loaded=GumbelNetwork(.1,true,false,true,std::vector<double>{},0);loaded->to(device);torch::serialize::InputArchive archive;archive.load_from(path,device);loaded->load(archive);
  std::vector<torch::Tensor> outputs;
  for(auto ratio:queries){auto expected=run(net,ratio,xyz);auto actual=run(loaded,ratio,xyz);TORCH_CHECK(torch::equal(expected,actual),"Checkpoint mismatch");outputs.push_back(actual.detach());}
  torch::serialize::OutputArchive fixture;fixture.write("xyz",xyz);fixture.write("scale",scale);fixture.write("rotation",q);fixture.write("dc",dc);fixture.write("logits",logits);fixture.write("ratios",torch::tensor(queries,torch::kFloat64));fixture.write("outputs",torch::stack(outputs));fixture.save_to(path+".fixture");
  std::cout<<"RATIO54_PASS selected_gradient six_ratio_adam_isolation master_gradient teacher checkpoint\n";
}
