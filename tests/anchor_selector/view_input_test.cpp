#include "improvements/selector/gumbel_network.h"
#include "include/flexgs_importance.h"
#include "include/selector_handoff.h"
#include <filesystem>
#include <iostream>
using namespace improvements::selector;

int main(int argc,char** argv) {
  bool cuda=argc>1&&std::string(argv[1])=="cuda";
  if(cuda&&!torch::cuda::is_available())return 77;
  torch::set_num_threads(2);torch::manual_seed(7);at::globalContext().setAllowTF32CuBLAS(false);
  auto o=torch::TensorOptions().dtype(torch::kFloat32).device(cuda?torch::kCUDA:torch::kCPU);
  auto current=torch::tensor({2.f,2.f,0.f,0.f,0.f,9.f},o).requires_grad_(true);
  auto all_views=torch::tensor({1.f,3.f,5.f,5.f,1.f,9.f},o).requires_grad_(true);
  auto uids=torch::tensor({9,8,7,6,5,4},o.dtype(torch::kLong));
  auto eligible=torch::tensor({true,true,true,true,true,false},o.dtype(torch::kBool));
  auto y=budgetAlignedImportanceTarget(current,all_views,uids,eligible,.5);
  TORCH_CHECK(y.equal(torch::tensor({1.f,1.f,0.f,1.f,0.f,0.f},o))&&!y.requires_grad(),"GI primary/global/UID exact-K ordering");
  auto permutation=torch::tensor({5,3,1,0,4,2},uids.options());
  auto permuted=budgetAlignedImportanceTarget(current.index_select(0,permutation),all_views.index_select(0,permutation),
      uids.index_select(0,permutation),eligible.index_select(0,permutation),.5);
  TORCH_CHECK(permuted.equal(y.index_select(0,permutation)),"GI tie-break changed after row reorder");
  TORCH_CHECK(budgetAlignedImportanceTarget(current,all_views,uids,eligible,1.).sum().item<int>()==5,"GI eligible budget shortfall");
  TORCH_CHECK(budgetAlignedImportanceTarget(current,all_views,uids,eligible,0.).sum().item<int>()==0,"Zero GI budget");
  auto exact_two=budgetAlignedImportanceTargetK(current,all_views,uids,eligible,2);
  TORCH_CHECK(exact_two.equal(torch::tensor({1.f,1.f,0.f,0.f,0.f,0.f},o)),"Visible-population explicit GI budget");
  auto budget_mask=torch::tensor({0.f,1.f,1.f,0.f},o).requires_grad_(true);
  auto budget_loss=selectorGumbelBudgetLoss(budget_mask,1,true);
  TORCH_CHECK(budget_loss.item<float>()==1,"Explicit target K normalization");
  budget_loss.backward();TORCH_CHECK(budget_mask.grad().equal(torch::ones_like(budget_mask)),"Explicit budget VJP");
  TORCH_CHECK(selectorGumbelBudgetLoss(torch::empty({0},o),0,true).item<float>()==0&&
      budgetAlignedImportanceTargetK(current,all_views,uids,eligible,0).sum().item<float>()==0,"Empty visible budget");
  auto zero=torch::zeros_like(current);
  TORCH_CHECK(budgetAlignedImportanceTarget(zero,zero,uids,eligible,.5).equal(torch::tensor({0.f,0.f,1.f,1.f,1.f,0.f},o)),"All-zero GI must fill K by UID");
  auto logits=torch::tensor({{0.f,-100.f},{0.f,100.f},{1.f,1.f},{0.f,0.f},{-100.f,100.f},{100.f,-100.f}},o).requires_grad_(true);
  auto bce=budgetAlignedImportanceLoss(logits,y,eligible,1.);bce.backward();
  TORCH_CHECK(torch::isfinite(bce).item<bool>()&&torch::isfinite(logits.grad()).all().item<bool>(),"Finite saturated BCE");
  TORCH_CHECK(logits.grad()[5].abs().sum().item<float>()==0&&!current.grad().defined()&&!all_views.grad().defined(),"BCE teacher/eligibility gradient isolation");
  auto empty_labels=budgetAlignedImportanceTarget(current.slice(0,0,0),all_views.slice(0,0,0),uids.slice(0,0,0),eligible.slice(0,0,0),.1);
  TORCH_CHECK(empty_labels.numel()==0,"Empty exact-K teacher");
  auto empty_logits=torch::empty({0,2},o).requires_grad_(true);
  auto empty_loss=budgetAlignedImportanceLoss(empty_logits,empty_labels,eligible.slice(0,0,0),1.);
  TORCH_CHECK(empty_loss.item<float>()==0,"Empty BCE must be zero");
  auto net=GumbelNetwork(.1,true,false,true,std::vector<double>{},0);net->to(o.device());
  {torch::NoGradGuard guard;net->named_parameters()["soft_net.weight"].normal_(0,.1);}
  auto old=net->named_parameters();std::map<std::string,torch::Tensor> before;
  for(auto p:old)before[p.key()]=p.value().detach().clone();
  auto rng=at::globalContext().defaultGenerator(torch::Device(torch::kCPU)).get_state().clone();
  net->enableSelectorViewInput();
  TORCH_CHECK(rng.equal(at::globalContext().defaultGenerator(torch::Device(torch::kCPU)).get_state()),"Shared input changed RNG");
  auto parameters=net->named_parameters();int64_t count=0;
  for(auto p:parameters) {
    if(p.key()=="local_emd.0.weight") {
      TORCH_CHECK(p.value().slice(1,0,11).equal(before[p.key()])&&p.value().slice(1,11,19).abs().sum().item<float>()==0,"First-layer zero expansion");
    } else TORCH_CHECK(p.value().equal(before[p.key()]),"Expansion changed old parameter");
    count+=p.value().numel();
  }
  TORCH_CHECK(count==1762&&!net->selectorViewResidual(),"19D Selector must have 1762 parameters and no residual");
  auto forward=[&](const torch::Tensor& input) {
    auto h=torch::relu(torch::nn::functional::linear(input,parameters["local_emd.0.weight"],parameters["local_emd.0.bias"]));
    auto h1=torch::relu(torch::nn::functional::linear(h,parameters["local_emd.2.weight"],parameters["local_emd.2.bias"]));
    return std::vector<torch::Tensor>{torch::nn::functional::linear(h1,parameters["soft_net.weight"],parameters["soft_net.bias"]),h};
  };
  for(int n:{0,1,1031,2048}) {
    auto x=torch::randn({n,11},o).requires_grad_(true),code=torch::randn({n,7},o).requires_grad_(true);
    auto global=net->packSelectorViewInput(x),view=net->packSelectorViewInput(x,code);
    TORCH_CHECK(global.size(1)==19&&global.slice(1,11,19).abs().sum().item<float>()==0,"Global mode must pack eight zeros");
    TORCH_CHECK(view.slice(1,11,18).equal(code)&&view.select(1,18).sum().item<float>()==n,"View mode code and explicit flag");
    auto reference=forward(view);
    if(cuda) {
      auto fast=mapping_cuda::flexgsMLP(view,parameters["local_emd.0.weight"],parameters["local_emd.0.bias"],
          parameters["local_emd.2.weight"],parameters["local_emd.2.bias"],parameters["soft_net.weight"],parameters["soft_net.bias"]);
      TORCH_CHECK(torch::allclose(fast[0],reference[0],3e-5,3e-6),"MLP19 fused forward");
      auto g=torch::randn_like(reference[0]);std::vector<torch::Tensor> leaves{x};
      for(auto p:net->parameters())if(p.requires_grad())leaves.push_back(p);
      auto a=torch::autograd::grad({reference[0]},leaves,{g},true),b=torch::autograd::grad({fast[0]},leaves,{g},true);
      for(size_t j=0;j<a.size();++j)TORCH_CHECK(torch::allclose(a[j],b[j],3e-4,3e-5),"MLP19 VJP at ",j," size ",n);
      if(n==2048) {
        auto original=mapping_cuda::flexgsMLP(x,before["local_emd.0.weight"],before["local_emd.0.bias"],
            before["local_emd.2.weight"],before["local_emd.2.bias"],before["soft_net.weight"],before["soft_net.bias"]);
        TORCH_CHECK(original[0].equal(fast[0])&&original[1].equal(fast[1]),"Zero expansion must retain tiled FP32 output bitwise");
      }
    }
    reference[0].sum().backward();
    TORCH_CHECK(!code.grad().defined()&&x.grad().defined(),"View geometry must detach while original input gradients remain");
    if(n)TORCH_CHECK(parameters["local_emd.0.weight"].grad().slice(1,11,19).abs().sum().item<float>()>0,"Direct view columns have no gradient");
    net->zero_grad(true);
  }
  auto geometry=torch::tensor({{0.f,0.f,2.f},{1.f,1.f,3.f}},o);
  net->initializeSelectorViewInput(geometry);
  auto scale=net->selectorViewSceneScale().clone();net->initializeSelectorViewInput(geometry*100);
  TORCH_CHECK(scale.equal(net->selectorViewSceneScale()),"Scene normalization changed after initialization");
  auto x=torch::randn({9,11},o),code=torch::randn({9,7},o);
  torch::optim::Adam optimizer(net->parameters(),torch::optim::AdamOptions(.01));
  for(int mode=0;mode<2;++mode) {
    auto input=net->packSelectorViewInput(x,mode?code:torch::Tensor{});
    forward(input)[0].square().sum().backward();optimizer.step();optimizer.zero_grad(true);
  }
  for(auto p:net->parameters())if(p.numel()) {
    auto it=optimizer.state().find(p.unsafeGetTensorImpl());TORCH_CHECK(it!=optimizer.state().end(),"Shared state missing");
    TORCH_CHECK(static_cast<const torch::optim::AdamParamState&>(*it->second).step()==2,"Shared Adam must advance in both modes");
  }
  auto global=forward(net->packSelectorViewInput(x))[0],view=forward(net->packSelectorViewInput(x,code))[0];
  TORCH_CHECK(!global.equal(view),"Shared Selector cannot distinguish modes");
  const auto path=std::filesystem::path(argc>2?argv[2]:"/tmp/cartgs_view_input.pt");
  torch::serialize::OutputArchive saved;net->save(saved);saved.save_to(path.string());
  auto restored=GumbelNetwork(.1,true,false,true,std::vector<double>{},0);restored->to(o.device());
  torch::serialize::InputArchive archive;archive.load_from(path.string(),o.device());restored->load(archive);
  TORCH_CHECK(restored->selectorViewInput()&&restored->selectorViewSceneScale().equal(scale),"Shared schema/normalization restore");
  auto frozen=restored->cloneFrozen();for(auto p:frozen->parameters())TORCH_CHECK(!p.requires_grad(),"Frozen shared input gradients");
  for(auto p:restored->named_parameters())TORCH_CHECK(p.value().equal(frozen->named_parameters()[p.key()]),"Frozen shared input weight mismatch");
  torch::serialize::OutputArchive fixture;fixture.write("input11",x,true);fixture.write("view7",code,true);
  fixture.write("global_logits",global,true);fixture.write("view_logits",view,true);fixture.save_to(path.string()+".fixture.pt");
  std::cout<<"PASS ViewInput "<<(cuda?"CUDA":"CPU")<<": 1762 params, mode0/1, zero expansion/RNG, fused forward/VJP, Master input gradients, camera detach, shared Adam, schema/Frozen, exact-K GI ties/reorder/empty, deterministic BCE\n";
}
