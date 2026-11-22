#include "improvements/selector/gumbel_network.h"
#include <filesystem>
#include <iostream>

using improvements::selector::GumbelNetwork;
using improvements::selector::AnchorContextResidual;

std::vector<torch::Tensor> reference(const std::vector<torch::Tensor>& a,int64_t anchors) {
  auto h=torch::relu(torch::nn::functional::linear(a[0],a[1],a[2]));
  AnchorContextResidual pool;pool->to(h.device());
  auto c=pool->context(h,a[8],a[9],anchors);
  auto next=torch::relu(torch::nn::functional::linear(h,a[3],a[4])+c.mm(a[7].t()).index_select(0,a[8]));
  return {torch::nn::functional::linear(next,a[5],a[6]),h,c};
}

int main(int argc,char** argv) {
  const bool cuda=argc>1 && std::string(argv[1])=="cuda";
  if(cuda && !torch::cuda::is_available())return 77;
  torch::manual_seed(61);torch::set_num_threads(2);
  auto o=torch::TensorOptions().dtype(torch::kFloat32).device(cuda?torch::kCUDA:torch::kCPU);
  auto ids=o.dtype(torch::kLong);
  AnchorContextResidual pool;pool->to(o.device());
  TORCH_CHECK(pool->projection->weight.numel()==544 && pool->projection->weight.abs().sum().item<float>()==0,"544 zero-initialized shared parameters");
  auto owners=torch::tensor({6,1,1,4,4,4,2,2,2,2,5,5,5,5,5,5,5,5,5,5},ids);
  const int64_t n=owners.numel(),anchors=8;
  auto eligible=(torch::arange(n,ids).remainder(3)!=1)&(owners!=4);
  auto h=torch::rand({n,32},o).requires_grad_(true);
  auto context=pool->context(h,owners,eligible,anchors);
  for(int64_t a=0;a<anchors;++a) {
    auto rows=torch::nonzero((owners==a)&eligible).flatten();
    if(rows.numel()) {
      auto value=h.index_select(0,rows).slice(1,0,8);
      TORCH_CHECK(torch::allclose(context[a].slice(0,0,8),value.mean(0),1e-5,1e-6),"Context mean oracle");
      TORCH_CHECK(torch::allclose(context[a].slice(0,8,16),std::get<0>(value.max(0)),1e-5,1e-6),"Context max oracle");
      TORCH_CHECK(std::abs(context[a][16].item<float>()-rows.numel()/10.)<1e-6,"Eligible context count");
    } else TORCH_CHECK(context[a].abs().sum().item<float>()==0,"Empty/all-inactive Anchor context");
  }
  auto polluted=torch::where(eligible.unsqueeze(1),h,torch::full_like(h,1000));
  TORCH_CHECK(torch::allclose(pool->context(polluted,owners,eligible,anchors),context,1e-5,1e-6),"Inactive context pollution");
  auto pg=torch::autograd::grad({context.sum()},{h})[0];
  TORCH_CHECK(pg.index({~eligible}).abs().sum().item<float>()==0,"Inactive context VJP");
  auto net=GumbelNetwork(.1,true,false,true,std::vector<double>{},0);net->to(o.device());net->anchors()->enableVariableMembers();
  auto before=net->named_parameters();std::vector<torch::Tensor> copied;for(const auto& p:before)copied.push_back(p.value().clone());
  net->enableSelectorContextResidual();int64_t parameters=0;for(auto p:net->parameters())parameters+=p.numel();
  TORCH_CHECK(parameters==2050,"2050 shared Selector parameter count");
  for(size_t i=0;i<before.size();++i)TORCH_CHECK(before[i].value().equal(copied[i]),"Attaching residual changed old weights");
  const auto path=std::filesystem::path(argc>2?argv[2]:"/tmp/cartgs_anchor_context_residual.pt");
  torch::serialize::OutputArchive saved;net->save(saved);saved.save_to(path.string());
  auto restored=GumbelNetwork(.1,true,false,true,std::vector<double>{},0);restored->to(o.device());
  torch::serialize::InputArchive archive;archive.load_from(path.string(),o.device());restored->load(archive);
  TORCH_CHECK(restored->selectorContextResidual()&&!restored->selectorSetEncoder(),"Independent context checkpoint schema");
  auto frozen=restored->cloneFrozen();TORCH_CHECK(frozen->selectorContextResidual(),"Frozen context schema");
  for(const auto& p:frozen->named_parameters())TORCH_CHECK(!p.value().requires_grad()&&p.value().equal(net->named_parameters()[p.key()]),"Frozen context weights");

  std::vector<torch::Tensor> a{torch::randn({n,11},o).requires_grad_(true),
      torch::randn({32,11},o).mul(.1).requires_grad_(true),torch::randn({32},o).mul(.1).requires_grad_(true),
      torch::randn({32,32},o).mul(.1).requires_grad_(true),torch::randn({32},o).mul(.1).requires_grad_(true),
      torch::randn({2,32},o).mul(.1).requires_grad_(true),torch::randn({2},o).mul(.1).requires_grad_(true),
      torch::randn({32,17},o).mul(.03).requires_grad_(true),owners,eligible};
  auto expected=reference(a,anchors);
  auto gz=torch::randn_like(expected[0]),gh=torch::randn_like(expected[1]);
  std::vector<torch::Tensor> leaves(a.begin(),a.begin()+8);
  auto rg=torch::autograd::grad({expected[0],expected[1]},leaves,{gz,gh});
  auto actual=cuda?mapping_cuda::anchorContextMLP(a,anchors):reference(a,anchors);
  auto ag=torch::autograd::grad({actual[0],actual[1]},leaves,{gz,gh});
  double forward_error=(actual[0]-expected[0]).abs().max().item<double>(),gradient_error=0;
  TORCH_CHECK(torch::allclose(actual[0],expected[0],3e-5,3e-6),"Context MLP forward");
  for(int j=0;j<8;++j) {
    TORCH_CHECK(torch::allclose(ag[j],rg[j],2e-4,3e-5),"Context MLP VJP index ",j);
    gradient_error=std::max(gradient_error,(ag[j]-rg[j]).abs().max().item<double>());
  }
  torch::serialize::OutputArchive fixture;
  for(int j=0;j<10;++j)fixture.write("input_"+std::to_string(j),a[j].detach(),true);
  fixture.write("logits",actual[0].detach(),true);fixture.write("shared",actual[1].detach(),true);
  fixture.write("reference_logits",expected[0].detach(),true);fixture.write("gz",gz,true);fixture.write("gh",gh,true);
  for(int j=0;j<8;++j)fixture.write("gradient_"+std::to_string(j),ag[j],true);
  fixture.save_to(path.string()+".fixture.pt");
  if(cuda) {
    for(int count:{0,1,1031,20000}) {
      auto b=a;b[0]=torch::randn({count,11},o).requires_grad_(true);
      b[8]=torch::arange(count,ids).floor_divide(4);b[9]=torch::ones({count},o.dtype(torch::kBool));
      b[7]=torch::zeros_like(a[7]).requires_grad_(true);const int m=(count+3)/4;
      auto control=reference(b,m),r=mapping_cuda::anchorContextMLP(b,m);
      TORCH_CHECK(torch::isfinite(r[0]).all().item<bool>() && torch::allclose(r[0],control[0],3e-5,3e-6),"Context size boundary ",count);
      if(count>=1024) {
        auto old=mapping_cuda::flexgsMLP(b[0],b[1],b[2],b[3],b[4],b[5],b[6]);
        TORCH_CHECK(r[0].equal(old[0])&&r[1].equal(old[1]),"Zero residual must preserve fused 11D output bitwise");
        auto target=torch::randn_like(old[0]),extra=torch::randn_like(old[1]);
        std::vector<torch::Tensor> base_parameters(b.begin(),b.begin()+7);
        auto old_gradients=torch::autograd::grad({old[0],old[1]},base_parameters,{target,extra},true);
        auto new_gradients=torch::autograd::grad({r[0],r[1]},base_parameters,{target,extra},true);
        for(int j=0;j<7;++j)TORCH_CHECK(old_gradients[j].equal(new_gradients[j]),"Zero residual base VJP changed at index ",j);
      }
      auto target=torch::randn_like(r[0]);std::vector<torch::Tensor> bl(b.begin(),b.begin()+8);
      auto gradients=torch::autograd::grad({r[0]},bl,{target});
      for(auto g:gradients)TORCH_CHECK(torch::isfinite(g).all().item<bool>(),"Finite size boundary VJP");
      if(count)TORCH_CHECK(gradients[7].abs().sum().item<float>()>0,"Zero projection has no learning signal");
      auto inactive=b;inactive[9]=torch::zeros_like(b[9]);
      auto none=mapping_cuda::anchorContextMLP(inactive,m);
      auto ng=torch::autograd::grad({none[0]},bl,{target});
      TORCH_CHECK(ng[7].abs().sum().item<float>()==0,"All-inactive projection gradient");
    }
    // All equal hidden channels exercise amax's equal-share VJP, including zero ties.
    auto b=a;b[1]=torch::zeros_like(a[1]).requires_grad_(true);b[2]=torch::ones_like(a[2]).requires_grad_(true);
    auto control=reference(b,anchors),fast=mapping_cuda::anchorContextMLP(b,anchors);
    std::vector<torch::Tensor> bl(b.begin(),b.begin()+8);
    auto cg=torch::autograd::grad({control[0]},bl,{gz}),fg=torch::autograd::grad({fast[0]},bl,{gz});
    for(int j=0;j<8;++j)TORCH_CHECK(torch::allclose(cg[j],fg[j],2e-4,3e-5),"Tied max VJP ",j);
  }
  std::cout<<"PASS ContextResidual "<<(cuda?"CUDA":"CPU")<<" 2050 params, eligible mean8/max8/count, base weight identity, projection VJP, schema/frozen, size0/1/1031/20000, zero-initialized fused bitwise, max ties; forward_max_abs="
      <<forward_error<<" vjp_max_abs="<<gradient_error<<'\n';
}
