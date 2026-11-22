#include "improvements/selector/gumbel_network.h"
#include <iostream>

using improvements::selector::GumbelNetwork;
using Key=std::array<int64_t,4>;

int main(int argc,char** argv) {
  torch::manual_seed(2026);
  const torch::Device device=argc>1&&std::string(argv[1])=="cuda"?torch::kCUDA:torch::kCPU;
  const std::vector<double> knots{.01,.05,.1,.15};
  GumbelNetwork model(.1,true,false,true,knots);model->to(device);model->enableCompensation();
  auto bank=model->anchors();bank->enableVariableMembers();bank->enableSlotDeactivation();
  torch::optim::Adam optimizer(model->parameters(),torch::optim::AdamOptions(.001));
  auto opts=torch::TensorOptions().device(device);
  auto observed=torch::tensor({{.01f,.02f,.03f},{.02f,.02f,.03f},{.03f,.02f,.03f},
      {.04f,.02f,.03f},{.05f,.02f,.03f},{.21f,.02f,.03f},{.22f,.02f,.03f},
      {.23f,.02f,.03f},{.24f,.02f,.03f},{.25f,.02f,.03f}},opts);
  auto seeds=bank->insertVariableBlocks({{{0,0,0,0},{0,1,2,3,4}},{{0,2,0,0},{5,6,7,8,9}}},
      observed,torch::full_like(observed,.4),false,{&optimizer});
  auto xyz=seeds.xyz,scale=seeds.sigma,dc=seeds.color.unsqueeze(1),opacity=torch::full({10,1},.2,opts);
  TORCH_CHECK(bank->featureTables().size()==5 && bank->features().size(1)==8,"Not five independent tables");
  for(auto table:bank->featureTables())TORCH_CHECK(table.is_leaf() && table.count_nonzero().item<int64_t>()==0,"Nonzero initial Feature");
  for(int epoch=0;epoch<3;++epoch)for(size_t active=0;active<knots.size();++active) {
    optimizer.zero_grad(true);
    auto before=bank->featureTables();std::vector<torch::Tensor> values,moments;std::vector<int64_t> steps;
    for(auto table:before) {
      values.push_back(table.detach().clone());
      auto found=optimizer.state().find(table.unsafeGetTensorImpl());
      if(found==optimizer.state().end()){moments.push_back({});steps.push_back(0);}
      else {auto& state=static_cast<torch::optim::AdamParamState&>(*found->second);
        moments.push_back(torch::stack({state.exp_avg().clone(),state.exp_avg_sq().clone()}));steps.push_back(state.step());}
    }
    auto encoded=model->forwardSelectionFeatures(xyz,knots[active],1,{},scale,dc,opacity);
    auto residual=model->compensationResidual(encoded.shared.slice(0,0,2));
    TORCH_CHECK(residual.size(0)==2 && residual.size(1)==14,"Selected-only compensation changed");
    auto loss=(encoded.probability-.15-.03*active).square().mean()+(residual-.02).square().mean();loss.backward();
    for(size_t i=0;i<before.size();++i) {
      if(i==0 || i==active+1) {
        TORCH_CHECK(before[i].grad().defined() && torch::isfinite(before[i].grad()).all().item<bool>() &&
            before[i].grad().abs().sum().item<float>()>0,"Missing active Feature gradient");
      } else TORCH_CHECK(!before[i].grad().defined(),"Inactive ratio received a gradient");
    }
    optimizer.step();
    for(size_t i=1;i<before.size();++i)if(i!=active+1) {
      TORCH_CHECK(torch::equal(before[i],values[i]),"Inactive ratio drifted under Adam momentum");
      auto found=optimizer.state().find(before[i].unsafeGetTensorImpl());
      if(steps[i]==0){TORCH_CHECK(found==optimizer.state().end(),"Inactive ratio acquired Adam history");}
      else {auto& state=static_cast<torch::optim::AdamParamState&>(*found->second);
        TORCH_CHECK(state.step()==steps[i] && torch::equal(moments[i],torch::stack({state.exp_avg(),state.exp_avg_sq()})),
                    "Inactive ratio Adam history advanced");}
    }
  }
  optimizer.zero_grad(true);
  // Same-layer max inheritance applies independently to all five tables; finer-level Features start at zero.
  const auto snapshot=bank->packedFeatures().detach();
  bank->insertVariableBlocks({{{0,4,0,0},{0,5}}},xyz,dc.squeeze(1),true,{&optimizer});
  TORCH_CHECK(torch::equal(bank->packedFeatures()[2],std::get<0>(snapshot.max(0))),"Same-layer max inheritance incomplete");
  bank->insertVariableBlocks({{{1,12,0,0},{0}}},xyz,dc.squeeze(1),true,{&optimizer});
  TORCH_CHECK(bank->packedFeatures()[3].count_nonzero().item<int64_t>()==0,"Fine-layer Feature not zero");
  bank->pruneVariableMembers(torch::arange(10,bank->rows().options()),{&optimizer});
  TORCH_CHECK(bank->size()==2 && torch::equal(bank->packedFeatures(),snapshot),"Prune did not preserve all Feature tables");
  {torch::NoGradGuard guard;int j=0;for(auto table:bank->featureTables()) {
    table[0].fill_(2+10*j);table[1].fill_(8+10*j);++j;}}
  auto moving=torch::arange(5,bank->rows().options());xyz.index_put_({moving,0},xyz.index({moving,0})+.2);
  auto migration=bank->migrateLocalMembers(xyz,moving,{&optimizer});
  bank->pruneVariableMembers(~migration.remove,{&optimizer});
  TORCH_CHECK(migration.moved==5 && migration.remove.count_nonzero().item<int64_t>()==0 && bank->size()==1,"Migration capacity changed");
  int index=0;for(auto table:bank->featureTables()) {
    TORCH_CHECK(torch::allclose(table,torch::full_like(table,5+10*index++)),"Weighted Feature migration incomplete");
    auto& state=static_cast<torch::optim::AdamParamState&>(*optimizer.state().at(table.unsafeGetTensorImpl()));
    TORCH_CHECK(state.exp_avg().count_nonzero().item<int64_t>()==0 && state.exp_avg_sq().count_nonzero().item<int64_t>()==0,
                "Mixed Feature retained Adam moments");
  }
  auto queried=bank->inputs(xyz,{},.03).first;
  TORCH_CHECK(torch::allclose(queried.slice(1,8),torch::full_like(queried.slice(1,8),20)),"Raw-ratio interpolation failed");
  TORCH_CHECK(torch::equal(bank->inputs(xyz,{},.005).first,bank->inputs(xyz,{},.01).first) &&
      torch::equal(bank->inputs(xyz,{},.5).first,bank->inputs(xyz,{},.15).first),"Endpoint policy failed");
  // Compact lookup and selected-only head must retain the same encoded rows.
  auto all=model->forwardSelectionFeatures(xyz,.075,1,{},scale,dc,opacity);
  auto ids=torch::tensor({0,3,8},bank->rows().options());
  auto compact=model->forwardSelectionFeatures(xyz.index_select(0,ids),.075,1,ids,scale.index_select(0,ids),dc.index_select(0,ids),opacity.index_select(0,ids));
  TORCH_CHECK(torch::allclose(compact.shared,all.shared.index_select(0,ids),1e-5,1e-6),"Compact feature lookup mismatch");
  auto teacher=GumbelNetwork(model->cloneFrozen());
  TORCH_CHECK(torch::equal(teacher->anchors()->packedFeatures(),bank->packedFeatures()),"Teacher omitted ratio Features");
  torch::serialize::OutputArchive saved;model->save(saved);saved.save_to("ratio_selector.pt");
  GumbelNetwork loaded(.1,true,false,true);loaded->to(device);
  torch::serialize::InputArchive input;input.load_from("ratio_selector.pt",device);loaded->load(input);
  TORCH_CHECK(loaded->anchors()->ratioFeatures() && torch::equal(loaded->anchors()->packedFeatures(),bank->packedFeatures()),"New checkpoint round trip failed");
  torch::serialize::OutputArchive fixture;
  fixture.write("xyz",xyz);fixture.write("scale",scale);fixture.write("dc",dc);fixture.write("opacity",opacity);
  const std::vector<float> ratios{.005f,.01f,.03f,.05f,.075f,.1f,.125f,.15f,.2f};
  fixture.write("ratios",torch::tensor(ratios));
  std::vector<torch::Tensor> probability,shared,residual;
  for(float ratio:ratios) {
    auto result=loaded->forwardSelectionFeatures(xyz,ratio,1,{},scale,dc,opacity);
    TORCH_CHECK(torch::equal(result.probability,model->forwardAnchorProbability(xyz,ratio,1,{},scale,dc,opacity)),"Native save/load or forward mismatch");
    probability.push_back(result.probability);shared.push_back(result.shared);
    residual.push_back(loaded->compensationResidual(result.shared.slice(0,0,3)));
  }
  fixture.write("probability",torch::stack(probability));fixture.write("shared",torch::stack(shared));fixture.write("residual",torch::stack(residual));
  fixture.save_to("ratio_fixture.pt");
  // Explicit old 27D checkpoints remain loadable, with no automatic lossy conversion.
  GumbelNetwork legacy(.1,true,false,true);legacy->to(device);legacy->enableCompensation();
  legacy->anchors()->enableVariableMembers();legacy->anchors()->observeBlocks(observed,torch::full_like(observed,.4));
  torch::serialize::OutputArchive old;legacy->save(old);old.save_to("ratio_legacy_selector.pt");
  GumbelNetwork old_loaded(.1,true,false,true);old_loaded->to(device);
  torch::serialize::InputArchive old_input;old_input.load_from("ratio_legacy_selector.pt",device);old_loaded->load(old_input);
  TORCH_CHECK(!old_loaded->anchors()->ratioFeatures(),"Legacy checkpoint changed schema");
  std::cout<<"PASS: five independent leaves; active-only gradients/Adam; growth/prune/migration; raw-ratio interpolation/endpoints; compact lookup; teacher; old/new checkpoints; selected-only compensation\n";
}
