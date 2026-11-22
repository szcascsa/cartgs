#include "improvements/selector/gumbel_network.h"
#include <iostream>

using improvements::selector::GumbelNetwork;

int main(int argc,char** argv) {
  const int64_t width=argc>1 && std::string(argv[1])=="e50"?50:8;
  const std::string prefix=width==50?"s00_e50":"s00";
  torch::manual_seed(2026);
  const std::vector<double> knots{.01,.05,.1,.15};
  const std::vector<float> ratios{.01f,.025f,.05f,.075f,.1f,.125f,.15f};
  auto net=GumbelNetwork(.1,true,false,true,std::vector<double>{},0);
  auto bank=net->anchors();bank->enableVariableMembers();bank->enableSlotDeactivation();
  net->enableCompensation();net->enableAnchorRatioCompensation(knots,width);
  TORCH_CHECK(net->named_parameters()["local_emd.0.weight"].size(1)==11,"Wrong Selector input");
  TORCH_CHECK(net->compensationHiddenLayers()==2 && net->compensationOutputWidth()==14,"Wrong compensation depth/output");
  TORCH_CHECK(net->named_parameters()["compensation.0.weight"].size(1)==32+width,"Wrong compensation input");
  TORCH_CHECK(!bank->features().requires_grad() && bank->features().size(1)==0,"Selector latent must be absent");
  torch::optim::Adam optimizer(net->parameters(),torch::optim::AdamOptions(.001).eps(1e-15));
  auto observed=torch::tensor({{.02f,.03f,2.02f},{.04f,.03f,2.03f},{.22f,.03f,2.02f},{.44f,.03f,2.03f}});
  auto seeds=bank->observeBlocks(observed,torch::full_like(observed,.4),{&optimizer});
  auto xyz=seeds.xyz,scale=seeds.sigma,dc=seeds.color.unsqueeze(1),alpha=torch::full({xyz.size(0),1},.2);
  auto ids=torch::tensor({0,2},torch::kLong);
  for(float r:ratios) {
    auto f=net->forwardSelectionFeatures(xyz,r,1,{},scale,dc,alpha);
    TORCH_CHECK(torch::equal(f.probability,torch::full_like(f.probability,.5)),"Initial probability mismatch");
    TORCH_CHECK(net->compensationResidual(f.shared.index_select(0,ids),ids,r).count_nonzero().item<int64_t>()==0,"Initial compensation not zero");
  }
  for(int step=0;step<5;++step) {
    optimizer.zero_grad(true);
    auto f=net->forwardSelectionFeatures(xyz,.05,1,{},scale,dc,alpha);
    auto residual=net->compensationResidual(f.shared.index_select(0,ids),ids,.05);
    auto loss=(residual-.03).square().mean()+.01*(f.probability.mean()-.05).square();
    loss.backward();
    TORCH_CHECK(net->named_parameters()["compensation.4.weight"].grad().abs().sum().item<float>()>0,"Output gradient missing");
    TORCH_CHECK(!bank->features().grad().defined(),"Zero-width Selector latent received a gradient");
    for(int i:{0,2,3})TORCH_CHECK(!bank->compensationFeatureTables()[i].grad().defined(),"Inactive ratio gradient leaked");
    if(step>1)TORCH_CHECK(bank->compensationFeatureTables()[1].grad().abs().sum().item<float>()>0,"Private latent gradient missing");
    optimizer.step();
  }
  for(int i:{0,2,3})TORCH_CHECK(optimizer.state().find(bank->compensationFeatureTables()[i].unsafeGetTensorImpl())==optimizer.state().end(),"Inactive ratio Adam clock advanced");
  auto old=bank->size();
  auto moment=static_cast<torch::optim::AdamParamState&>(*optimizer.state().at(bank->compensationFeatureTables()[1].unsafeGetTensorImpl())).exp_avg().clone();
  auto extra=bank->observeBlocks(torch::tensor({{.82f,.03f,2.02f}}),torch::tensor({{.4f,.4f,.4f}}),{&optimizer});
  xyz=torch::cat({xyz,extra.xyz});scale=torch::cat({scale,extra.sigma});dc=torch::cat({dc,extra.color.unsqueeze(1)});alpha=torch::full({xyz.size(0),1},.2);
  auto& state=static_cast<torch::optim::AdamParamState&>(*optimizer.state().at(bank->compensationFeatureTables()[1].unsafeGetTensorImpl()));
  TORCH_CHECK(torch::equal(state.exp_avg().slice(0,0,old),moment),"Append damaged Adam state");
  TORCH_CHECK(state.exp_avg().slice(0,old).count_nonzero().item<int64_t>()==0,"New rows inherited Adam moments");
  // All four ratio tables take part in the existing population-weighted migration.
  const auto source=bank->rows()[0].item<int64_t>();
  auto keys=bank->named_buffers()["keys"];
  auto key=keys[source].clone();key[0].add_(2);
  auto targets=torch::nonzero((keys==key).all(1)&(bank->levels()==bank->levels()[source])).reshape({-1});
  TORCH_CHECK(targets.numel()==1,"Fixture destination missing");
  const auto target=targets[0].item<int64_t>(),count=bank->memberCounts()[target].item<int64_t>();
  {torch::NoGradGuard guard;
    for(auto t:bank->compensationFeatureTables()){t[source].fill_(2);t[target].fill_(8);}
    xyz[0][0].add_(.2);
  }
  auto migrated=bank->migrateLocalMembers(xyz,torch::tensor({0},torch::kLong),{&optimizer});
  TORCH_CHECK(migrated.moved==1 && migrated.remove.sum().item<int64_t>()==0,"Migration failed");
  for(auto t:bank->compensationFeatureTables())TORCH_CHECK(torch::allclose(t[target],torch::full_like(t[target],(8.*count+2)/(count+1))),"Latent8 migration weight mismatch");
  auto keep=bank->rows()!=source;
  bank->pruneVariableMembers(keep,{&optimizer});xyz=xyz.index({keep});scale=scale.index({keep});dc=dc.index({keep});alpha=alpha.index({keep});
  bank->assertBlocks(xyz.size(0));
  TORCH_CHECK(bank->features().sizes()==torch::IntArrayRef({bank->size(),0}),"Zero-width row mapping lost topology");
  auto frozen=GumbelNetwork(net->cloneFrozen());
  auto reference=net->forwardLogCompensationResidual({xyz,scale.log(),torch::zeros({xyz.size(0),4})},.075,{},dc,alpha);
  auto teacher=frozen->forwardLogCompensationResidual({xyz,scale.log(),torch::zeros({xyz.size(0),4})},.075,{},dc,alpha);
  TORCH_CHECK(torch::equal(reference,teacher),"Frozen teacher head mismatch");
  torch::serialize::OutputArchive saved;net->save(saved);saved.save_to(prefix+"_selector.pt");
  auto loaded=GumbelNetwork(.1,true,false,true);
  torch::serialize::InputArchive archive;archive.load_from(prefix+"_selector.pt");loaded->load(archive);
  TORCH_CHECK(loaded->anchors()->features().size(1)==0 && loaded->compensationHiddenLayers()==2,"Checkpoint layout lost");
  torch::serialize::OutputArchive fixture;
  fixture.write("xyz",xyz);fixture.write("scale",scale);fixture.write("dc",dc);fixture.write("alpha",alpha);fixture.write("ids",ids);
  std::vector<torch::Tensor> probabilities,shared,residuals;
  for(float r:ratios) {
    auto a=net->forwardSelectionFeatures(xyz,r,1,{},scale,dc,alpha),b=loaded->forwardSelectionFeatures(xyz,r,1,{},scale,dc,alpha);
    TORCH_CHECK(torch::equal(a.probability,b.probability),"Checkpoint probability mismatch");
    probabilities.push_back(b.probability);shared.push_back(b.shared);
    residuals.push_back(loaded->compensationResidual(b.shared.index_select(0,ids),ids,r));
  }
  fixture.write("probabilities",torch::stack(probabilities));fixture.write("shared",torch::stack(shared));fixture.write("residuals",torch::stack(residuals));fixture.save_to(prefix+"_fixture.pt");
  // Existing legacy heads and R3 remain separately selectable.
  auto legacy=GumbelNetwork(.1,true,false,true);legacy->anchors()->enableVariableMembers();legacy->enableCompensation();
  legacy->enablePrivateCompensation(true,knots);
  TORCH_CHECK(legacy->privateCompensation()->independent() && legacy->compensationHiddenLayers()==1,"R3 compatibility lost");
  legacy->anchors()->observeBlocks(observed,torch::full_like(observed,.4));
  legacy->privateCompensation()->append(legacy->anchors()->rows().numel(),{},{});
  torch::serialize::OutputArchive r3;legacy->save(r3);r3.save_to("r3_reference_selector.pt");
  auto loaded_r3=GumbelNetwork(.1,true,false,true);
  torch::serialize::InputArchive r3_input;r3_input.load_from("r3_reference_selector.pt");loaded_r3->load(r3_input);
  TORCH_CHECK(loaded_r3->privateCompensation()->independent() && loaded_r3->compensationHiddenLayers()==1,"R3 checkpoint compatibility lost");
  std::cout<<"PASS S00: shape/init/gradients/ratio Adam isolation/append/prune/migration/teacher/checkpoint; R3 retained\n";
}
