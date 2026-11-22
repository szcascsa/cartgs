#include "improvements/selector/gumbel_network.h"
#include "include/real_lod_schedule.h"
#include <iostream>
#include <filesystem>

using namespace improvements::selector;
int main(int argc,char** argv) {
  const bool cuda=argc>1&&std::string(argv[1])=="cuda";
  if(cuda&&!torch::cuda::is_available())return 77;
  torch::set_num_threads(2);torch::manual_seed(7);at::globalContext().setAllowTF32CuBLAS(false);
  auto o=torch::TensorOptions().dtype(torch::kFloat32).device(cuda?torch::kCUDA:torch::kCPU);
  auto net=GumbelNetwork(.1,true,false,true,std::vector<double>{},0);net->to(o.device());
  auto rng=at::globalContext().defaultGenerator(torch::Device(torch::kCPU)).get_state().clone();
  auto weights=net->named_parameters();std::vector<torch::Tensor> copies;
  for(auto p:weights)copies.push_back(p.value().detach().clone());
  net->enableSelectorViewResidual();
  TORCH_CHECK(rng.equal(at::globalContext().defaultGenerator(torch::Device(torch::kCPU)).get_state()),"Branch initialization changed RNG");
  size_t j=0;for(auto p:weights)TORCH_CHECK(p.value().equal(copies[j++]),"Branch initialization changed global weights");
  int64_t count=0;for(auto p:net->selectorViewResidual()->parameters())count+=p.numel();
  TORCH_CHECK(count==657,"View residual parameter count");
  auto centers=torch::tensor({{0.f,0.f,2.f},{0.f,0.f,-2.f},{0.f,0.f,0.f},{1.f,2.f,4.f}},o).requires_grad_(true);
  auto scales=torch::full({4,1},.1,o),camera=torch::zeros({3},o),matrix=torch::eye(4,o);
  net->selectorViewResidual()->initialize(centers);
  auto code=anchorViewCode(centers,scales,camera,matrix,matrix,480,640,.01,net->selectorViewResidual()->scene_scale);
  TORCH_CHECK(code.sizes()==torch::IntArrayRef({4,7})&&torch::isfinite(code).all().item<bool>()&&!code.requires_grad(),"Finite detached view7");
  TORCH_CHECK(code[1][6].item<float>()==0&&code[2][6].item<float>()>0,"Behind/near-plane projection semantics");
  auto other=anchorViewCode(centers,scales,torch::ones({3},o),matrix,matrix,480,640,.01,net->selectorViewResidual()->scene_scale);
  TORCH_CHECK(!other.equal(code),"Camera movement did not change descriptor");
  auto empty=anchorViewCode(centers.slice(0,0,0),scales.slice(0,0,0),camera,matrix,matrix,480,640,.01,net->selectorViewResidual()->scene_scale);
  TORCH_CHECK(empty.sizes()==torch::IntArrayRef({0,7}),"Empty Anchor view code");
  auto shared=torch::randn({4,32},o).requires_grad_(true),logits=torch::randn({4,2},o).requires_grad_(true);
  GumbelNetworkImpl::SelectionFeatures global{{},shared,logits};
  auto actual=net->applyViewResidual(global,code);
  TORCH_CHECK(actual.logits.equal(logits),"Zero residual changed global logits");
  actual.logits.select(1,1).square().sum().backward();
  TORCH_CHECK(!shared.grad().defined()&&!logits.grad().defined()&&!centers.grad().defined(),"View branch leaked gradient");
  TORCH_CHECK(net->selectorViewResidual()->output->weight.grad().abs().sum().item<float>()>0,"Zero initialized output cannot learn");
  torch::optim::Adam optimizer(net->selectorViewResidual()->parameters(),torch::optim::AdamOptions(.01));optimizer.step();optimizer.zero_grad(true);
  auto learned=net->applyViewResidual(global,code),changed=net->applyViewResidual(global,other);
  TORCH_CHECK(!learned.logits.equal(changed.logits),"Learned residual ignores view");
  learned.logits.sum().backward();
  TORCH_CHECK(net->selectorViewResidual()->hidden->weight.grad().abs().sum().item<float>()>0,"Hidden view branch cannot learn after first step");
  const auto file=std::filesystem::path(argc>2?argv[2]:"/tmp/cartgs_view_residual.pt");
  torch::serialize::OutputArchive archive;net->save(archive);archive.save_to(file.string());
  auto restored=GumbelNetwork(.1,true,false,true,std::vector<double>{},0);restored->to(o.device());
  torch::serialize::InputArchive saved;saved.load_from(file.string(),o.device());restored->load(saved);
  TORCH_CHECK(restored->selectorViewResidual()&&restored->applyViewResidual(global,code).logits.equal(learned.logits),"View checkpoint roundtrip");
  auto frozen=restored->cloneFrozen();
  for(auto p:frozen->parameters())TORCH_CHECK(!p.requires_grad(),"Frozen teacher remains trainable");
  TORCH_CHECK(frozen->applyViewResidual(global,code).logits.equal(learned.logits),"Frozen view clone changed output");
  RealLodSchedule schedule;schedule.enabled=true;schedule.online_steps=24;schedule.endOnline();schedule.beginLod();
  int global_steps=0,view_steps=0;
  while(schedule.optimizingLod()) {
    TORCH_CHECK(schedule.ratioIndex(6)==schedule.lod_steps%6,"View schedule changed ratio rotation");
    (schedule.viewResidualRound(6)?view_steps:global_steps)++;schedule.completeLod();
  }
  TORCH_CHECK(global_steps==12&&view_steps==12,"Global/view rounds must alternate after full ratio rotation");
  torch::serialize::OutputArchive fixture;fixture.write("centers",centers.detach(),true);fixture.write("scales",scales,true);
  fixture.write("camera",camera,true);fixture.write("view",matrix,true);fixture.write("projection",matrix,true);
  fixture.write("code",code,true);fixture.write("shared",shared.detach(),true);fixture.write("logits",logits.detach(),true);
  fixture.write("view_logits",learned.logits.detach(),true);fixture.save_to(file.string()+".fixture.pt");
  std::cout<<"PASS ViewResidual "<<(cuda?"CUDA":"CPU")<<": 657 params, RNG/global identity, finite view7, zero init, isolated VJP, Adam, camera response, checkpoint/frozen, 6+6 schedule\n";
}
