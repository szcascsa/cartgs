#include "improvements/selector/gumbel_network.h"
#include <iostream>
using improvements::selector::GumbelNetwork;
int main(int argc,char** argv) {
  torch::manual_seed(2026);
  const torch::Device device=argc>1&&std::string(argv[1])=="cpu"?torch::kCPU:torch::kCUDA;
  const std::string path=argc>2?argv[2]:"ratio32_selector.pt";
  const std::vector<double> knots{.1,.2,.4,.6,.8},queries{.1,.15,.2,.3,.4,.5,.6,.7,.8};
  auto net=GumbelNetwork(.1,true,false,true,std::vector<double>{},0);net->to(device);
  auto bank=net->anchors();bank->enableVariableMembers();bank->enableSlotDeactivation();
  net->enableSpatialCompensation(false,false,0,knots);auto spatial=net->spatialCompensation();
  TORCH_CHECK(spatial->encode->weight.size(1)==51&&!spatial->ratio_net&&!spatial->early_feature&&!bank->spatialFeatures(),"Wrong input/FiLM/Feature layout");
  int64_t count=0;for(auto p:spatial->parameters())count+=p.numel();TORCH_CHECK(count==25038,"Wrong shared parameter count");
  torch::optim::Adam optimizer(net->parameters(),torch::optim::AdamOptions(.001));auto opts=torch::TensorOptions().device(device);
  auto observed=torch::tensor({{.02f,.03f,2.02f},{.04f,.03f,2.03f},{.22f,.03f,2.02f},{.44f,.03f,2.03f}},opts);
  auto seeds=bank->observeBlocks(observed,torch::full_like(observed,.4),{&optimizer});
  auto xyz=seeds.xyz,scale=seeds.sigma.log(),dc=seeds.color.unsqueeze(1),logits=torch::zeros({xyz.size(0),1},opts);
  auto q=torch::zeros({xyz.size(0),4},opts);q.select(1,0).fill_(1);
  auto owners=bank->rows();spatial->initialize(xyz,scale,q,dc,logits,bank->centers().index_select(0,owners),bank->offsetScales().index_select(0,owners),true);
  auto ids=torch::tensor({0,2},bank->rows().options());
  auto run=[&](double ratio,const torch::Tensor& rows) {
    auto a=bank->rows().index_select(0,rows);
    return spatial->forward(xyz.index_select(0,rows),scale.index_select(0,rows),q.index_select(0,rows),dc.index_select(0,rows),logits.index_select(0,rows),
        bank->centers().index_select(0,a),bank->offsetScales().index_select(0,a),ratio,true,bank->compensationInput(rows,ratio));
  };
  TORCH_CHECK(run(.2,ids).count_nonzero().item<int64_t>()==0,"Step0 must be identity");
  auto selection=net->forwardProbability(xyz,{},scale.exp(),.2,1,{},dc,logits.sigmoid()).detach();
  for(int step=0;step<4;++step) {
    optimizer.zero_grad(true);auto out=run(.2,ids);(out-.03).square().mean().backward();
    for(int i:{0,2,3,4})TORCH_CHECK(!bank->compensationFeatureTables()[i].grad().defined(),"Inactive ratio gradient leaked");
    TORCH_CHECK(!net->named_parameters()["local_emd.0.weight"].grad().defined(),"Compensation leaked into Selector");
    if(step>1)TORCH_CHECK(bank->compensationFeatureTables()[1].grad().abs().sum().item<float>()>0,"Feature gradient missing");
    optimizer.step();
  }
  for(int i:{0,2,3,4})TORCH_CHECK(!optimizer.state().count(bank->compensationFeatureTables()[i].unsafeGetTensorImpl()),"Inactive Adam clock advanced");
  TORCH_CHECK(torch::equal(selection,net->forwardProbability(xyz,{},scale.exp(),.2,1,{},dc,logits.sigmoid()).detach()),"Feature changed Selector output");
  auto middle=bank->compensationInput(ids,.3);
  TORCH_CHECK(torch::allclose(middle,.5*(bank->compensationInput(ids,.2)+bank->compensationInput(ids,.4))),"Latent interpolation mismatch");
  TORCH_CHECK(torch::equal(bank->compensationInput(ids,.05),bank->compensationInput(ids,.1))&&torch::equal(bank->compensationInput(ids,.9),bank->compensationInput(ids,.8)),"Endpoint clamp mismatch");
  const auto old=bank->size();auto moment=static_cast<torch::optim::AdamParamState&>(*optimizer.state().at(bank->compensationFeatureTables()[1].unsafeGetTensorImpl())).exp_avg().clone();
  auto extra=bank->observeBlocks(torch::tensor({{.82f,.03f,2.02f}},opts),torch::full({1,3},.4,opts),{&optimizer});
  xyz=torch::cat({xyz,extra.xyz});scale=torch::cat({scale,extra.sigma.log()});dc=torch::cat({dc,extra.color.unsqueeze(1)});q=torch::zeros({xyz.size(0),4},opts);q.select(1,0).fill_(1);logits=torch::zeros({xyz.size(0),1},opts);
  auto& state=static_cast<torch::optim::AdamParamState&>(*optimizer.state().at(bank->compensationFeatureTables()[1].unsafeGetTensorImpl()));
  TORCH_CHECK(torch::equal(state.exp_avg().slice(0,0,old),moment)&&state.exp_avg().slice(0,old).count_nonzero().item<int64_t>()==0,"Append Adam state mismatch");
  for(auto t:bank->compensationFeatureTables())TORCH_CHECK(t.slice(0,old).count_nonzero().item<int64_t>()==0,"New Feature not zero");
  const auto source=bank->rows()[0].item<int64_t>(),target=bank->rows()[2].item<int64_t>(),members=bank->memberCounts()[target].item<int64_t>();
  {torch::NoGradGuard guard;for(auto t:bank->compensationFeatureTables()){t[source].fill_(2);t[target].fill_(8);}xyz[0][0].add_(.2);}
  auto migration=bank->migrateLocalMembers(xyz,torch::tensor({0},bank->rows().options()),{&optimizer});TORCH_CHECK(migration.moved==1&&!migration.remove.any().item<bool>(),"Local migration failed");
  for(auto t:bank->compensationFeatureTables())TORCH_CHECK(torch::allclose(t[target],torch::full_like(t[target],(8.*members+2)/(members+1))),"Ratio32 weighted migration failed");
  auto keep=bank->rows()!=source;bank->pruneVariableMembers(keep,{&optimizer});xyz=xyz.index({keep});scale=scale.index({keep});q=q.index({keep});dc=dc.index({keep});logits=logits.index({keep});bank->assertBlocks(xyz.size(0));
  auto teacher=GumbelNetwork(net->cloneFrozen());TORCH_CHECK(teacher->spatialCompensation()->ratioFeatureWidth()==32&&teacher->anchors()->compensationFeatureWidth()==32,"Teacher schema mismatch");
  torch::serialize::OutputArchive saved;net->save(saved);saved.save_to(path);
  auto loaded=GumbelNetwork(.1,true,false,true,std::vector<double>{},0);loaded->to(device);torch::serialize::InputArchive archive;archive.load_from(path,device);loaded->load(archive);
  std::vector<torch::Tensor> outputs;auto all=torch::arange(xyz.size(0),bank->rows().options());
  for(auto ratio:queries) {
    auto expected=run(ratio,all);auto a=loaded->anchors();auto owner=a->rows();
    auto actual=loaded->spatialCompensation()->forward(xyz,scale,q,dc,logits,a->centers().index_select(0,owner),a->offsetScales().index_select(0,owner),ratio,true,a->compensationInput(all,ratio));
    TORCH_CHECK(torch::equal(expected,actual),"Checkpoint forward mismatch");outputs.push_back(actual.detach());
  }
  torch::serialize::OutputArchive fixture;fixture.write("xyz",xyz);fixture.write("scale",scale);fixture.write("rotation",q);fixture.write("dc",dc);fixture.write("logits",logits);fixture.write("ratios",torch::tensor(queries,torch::kFloat64));fixture.write("outputs",torch::stack(outputs));fixture.save_to(path+".fixture");
  std::cout<<"RATIO32_PASS input51 zero_init selected_grad ratio_adam_isolation interpolation append prune weighted_migration teacher checkpoint\n";
}
