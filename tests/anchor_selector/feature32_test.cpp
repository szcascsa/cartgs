#include "improvements/selector/gumbel_network.h"
#include <iostream>

int main(int argc,char** argv) {
  const torch::Device device=argc>1&&std::string(argv[1])=="cuda"?torch::kCUDA:torch::kCPU;
  torch::manual_seed(2026);
  using improvements::selector::GumbelNetwork;
  GumbelNetwork model(.1,true,false,true,std::vector<double>{},32);model->to(device);model->enableCompensation();
  model->anchors()->enableVariableMembers();
  auto opts=torch::TensorOptions().device(device);
  auto xyz=torch::tensor({{.01f,.02f,2.f},{.04f,.02f,2.f},{.31f,.02f,2.f}},opts);
  auto seeds=model->anchors()->observeBlocks(xyz,torch::full_like(xyz,.4));xyz=seeds.xyz;
  auto scale=seeds.sigma,dc=seeds.color.unsqueeze(1),alpha=torch::full({xyz.size(0),1},.2,opts);
  TORCH_CHECK(model->anchors()->features().size(1)==32 && model->anchors()->featureTables().size()==1 &&
      !model->anchors()->ratioFeatures() && model->named_parameters()["local_emd.0.weight"].size(1)==43,"Wrong Feature/input layout");
  torch::optim::Adam optimizer(model->parameters(),torch::optim::AdamOptions(.001));
  for(int step=0;step<3;++step) {
    optimizer.zero_grad(true);
    auto f=model->forwardSelectionFeatures(xyz,.05,1,{},scale,dc,alpha);
    auto residual=model->compensationResidual(f.shared.slice(0,0,2));
    auto loss=(f.probability-.1).square().mean()+(residual-.02).square().mean();loss.backward();
    auto gradient=model->anchors()->features().grad();
    TORCH_CHECK(gradient.sizes()==model->anchors()->features().sizes() && torch::isfinite(gradient).all().item<bool>() &&
        gradient.abs().sum().item<float>()>0,"Feature32 gradient missing");
    optimizer.step();
  }
  auto teacher=GumbelNetwork(model->cloneFrozen());
  TORCH_CHECK(teacher->anchors()->features().size(1)==32 &&
      torch::equal(teacher->anchors()->features(),model->anchors()->features()),"Frozen teacher width/values changed");
  torch::serialize::OutputArchive saved;model->save(saved);saved.save_to("feature32_selector.pt");
  GumbelNetwork loaded(.1,true,false,true);loaded->to(device);
  torch::serialize::InputArchive input;input.load_from("feature32_selector.pt",device);loaded->load(input);
  torch::serialize::OutputArchive fixture;
  fixture.write("xyz",xyz);fixture.write("scale",scale);fixture.write("dc",dc);fixture.write("alpha",alpha);
  std::vector<torch::Tensor> probabilities,shared,residuals;
  for(float ratio:{.01f,.05f,.1f,.15f}) {
    auto f=loaded->forwardSelectionFeatures(xyz,ratio,1,{},scale,dc,alpha);
    TORCH_CHECK(torch::equal(f.probability,model->forwardAnchorProbability(xyz,ratio,1,{},scale,dc,alpha)),"Feature32 checkpoint mismatch");
    probabilities.push_back(f.probability);shared.push_back(f.shared);residuals.push_back(loaded->compensationResidual(f.shared.slice(0,0,2)));
  }
  fixture.write("probabilities",torch::stack(probabilities));fixture.write("shared",torch::stack(shared));fixture.write("residuals",torch::stack(residuals));
  fixture.save_to("feature32_fixture.pt");
  std::cout<<"PASS: single 32D Feature, 43D input, finite gradients/Adam, selected-only decoder, teacher and checkpoint\n";
}
