#include "improvements/selector/gumbel_network.h"
#include <algorithm>
#include <filesystem>
#include <iostream>

using improvements::selector::AnchorSetEncoder;
using improvements::selector::GumbelNetwork;

int main(int argc,char** argv) {
  const bool cuda=argc>1 && std::string(argv[1])=="cuda";
  if(cuda && !torch::cuda::is_available())return 77;
  torch::manual_seed(13);torch::set_num_threads(2);
  auto o=torch::TensorOptions().dtype(torch::kFloat32).device(cuda?torch::kCUDA:torch::kCPU);
  auto ids=o.dtype(torch::kLong);
  auto owners=torch::tensor({6,1,1,4,4,4,2,2,2,2,5,5,5,5,5,5,5,5,5,5},ids);
  const int64_t n=owners.numel(),a=8;
  auto xyz=(torch::arange(n*3,o).reshape({n,3})*.01).requires_grad_(true);
  auto scale=torch::full({n,3},.08,o).requires_grad_(true);
  auto dc=torch::rand({n,1,3},o).requires_grad_(true),op=torch::rand({n,1},o).requires_grad_(true);
  auto ref=torch::full({a,1},.1,o),ratio=torch::full({1,1},.05,o);
  auto centers=torch::rand({a,3},o),eligible=torch::ones({n},o.dtype(torch::kBool));
  AnchorSetEncoder encoder;encoder->to(o.device());
  auto out=encoder->forward(xyz,scale,dc,op,owners,ref,ratio,centers,eligible);
  auto repeated=encoder->forward(xyz,scale,dc,op,owners,ref,ratio,centers,eligible);
  const auto repeat_error=(out-repeated).abs().max().item<float>();
  TORCH_CHECK(out.sizes()==torch::IntArrayRef({n,50}) && torch::isfinite(out).all().item<bool>(),"Set input shape/finite");
  for(int64_t k:{1,2,4,5,6}) {
    auto rows=torch::nonzero(owners==k).flatten(),v=out.index_select(0,rows);
    auto h=v.slice(1,0,16);
    TORCH_CHECK(torch::allclose(v.slice(1,16,32),h.mean(0).expand({rows.numel(),16}),1e-5,1e-6),"Mean pooling");
    TORCH_CHECK(torch::allclose(v.slice(1,32,48),std::get<0>(h.max(0)).expand({rows.numel(),16}),1e-5,1e-6),"Max pooling");
    TORCH_CHECK(torch::allclose(v.select(1,48),torch::full({rows.numel()},rows.numel()/10.,o)),"Count encoding");
  }
  TORCH_CHECK(torch::allclose(out.select(1,49),ratio.log().expand({n,1}).reshape({n})),"Original log ratio");
  auto permutation=torch::randperm(n,ids);
  auto permuted=encoder->forward(xyz.index_select(0,permutation),scale.index_select(0,permutation),
      dc.index_select(0,permutation),op.index_select(0,permutation),owners.index_select(0,permutation),ref,ratio,centers,eligible.index_select(0,permutation));
  TORCH_CHECK(torch::allclose(permuted,out.index_select(0,permutation),1e-5,1e-6),"Member permutation equivariance");
  auto shift=torch::tensor({2.f,-1.f,.5f},o);
  auto translated=encoder->forward(xyz+shift,scale,dc,op,owners,ref,ratio,centers+shift,eligible);
  TORCH_CHECK(torch::allclose(translated,out,1e-4,2e-5),"Nominal cell and member translation invariance");
  auto changed=dc.detach().clone();changed[2].add_(2);
  auto neighbor=encoder->forward(xyz,scale,changed,op,owners,ref,ratio,centers,eligible);
  TORCH_CHECK(!torch::allclose(neighbor[1].slice(0,16,48),out[1].slice(0,16,48)),"Neighbor appearance must affect context");
  const auto independent_error=(neighbor.index({owners!=1})-out.index({owners!=1})).abs().max().item<float>();
  TORCH_CHECK(torch::allclose(neighbor.index({owners!=1}),out.index({owners!=1}),1e-5,2e-6),"Independent Anchors must not interact");
  auto anchor_order=torch::tensor({3,6,1,7,4,2,5,0},ids),inverse=torch::argsort(anchor_order);
  auto reordered=encoder->forward(xyz,scale,dc,op,inverse.index_select(0,owners),ref.index_select(0,anchor_order),ratio,centers.index_select(0,anchor_order),eligible);
  const auto reorder_error=(reordered-out).abs().max().item<float>();
  TORCH_CHECK(torch::allclose(reordered,out,1e-5,2e-6),"Anchor compact/reorder identity");
  std::cout<<"POOL_ROUNDING repeat_max_abs="<<repeat_error<<" independent_max_abs="<<independent_error
      <<" reorder_max_abs="<<reorder_error<<std::endl;
  auto empty=encoder->forward(xyz.slice(0,0,0),scale.slice(0,0,0),dc.slice(0,0,0),op.slice(0,0,0),owners.slice(0,0,0),ref,ratio,centers,eligible.slice(0,0,0));
  TORCH_CHECK(empty.sizes()==torch::IntArrayRef({0,50}),"Empty member set");
  out.square().mean().backward();
  for(auto p:{xyz,scale,dc,op})TORCH_CHECK(p.grad().defined() && torch::isfinite(p.grad()).all().item<bool>(),"Master input gradients");
  double norm=0;for(auto p:encoder->parameters())norm+=p.grad().abs().sum().item<double>();
  TORCH_CHECK(norm>0,"Shared Encoder gradients");

  auto valid=(torch::arange(n,ids).remainder(3)!=1) & (owners!=4);
  auto masked=encoder->forward(xyz,scale,dc,op,owners,ref,ratio,centers,valid);
  auto polluted=encoder->forward(torch::where(valid.unsqueeze(1),xyz,xyz+1000),
      torch::where(valid.unsqueeze(1),scale,torch::full_like(scale,1000)),
      torch::where(valid.reshape({n,1,1}),dc,torch::full_like(dc,1000)),op,owners,ref,ratio,centers,valid);
  TORCH_CHECK(torch::allclose(masked,polluted,1e-5,1e-6),"Inactive members changed context");
  TORCH_CHECK(masked.index({owners==4}).slice(1,0,49).abs().sum().item<float>()==0,"All-inactive Anchor context");
  auto masked_grad=torch::autograd::grad({masked.square().mean()},{xyz,scale,dc,op});
  for(const auto& g:masked_grad)
    TORCH_CHECK(torch::isfinite(g).all().item<bool>() && g.index({~valid}).abs().sum().item<float>()==0,"Inactive member input VJP");
  auto none=encoder->forward(xyz,scale,dc,op,owners,ref,ratio,centers,torch::zeros_like(valid));
  TORCH_CHECK(torch::isfinite(none).all().item<bool>() && none.slice(1,0,49).abs().sum().item<float>()==0,"All-inactive population");
  for(const auto& g:torch::autograd::grad({none.square().mean()},{xyz,scale,dc,op}))
    TORCH_CHECK(torch::isfinite(g).all().item<bool>() && g.abs().sum().item<float>()==0,"All-inactive VJP");
  AnchorSetEncoder singleton;singleton->to(o.device());
  {torch::NoGradGuard guard;for(auto p:singleton->parameters())p.zero_();
    singleton->network->at<torch::nn::LinearImpl>(0).weight[0][0].fill_(1);
    singleton->network->at<torch::nn::LinearImpl>(2).weight[0][0].fill_(1);}
  auto one_x=torch::tensor({{.01f,0.f,0.f},{.04f,0.f,0.f}},o).requires_grad_(true);
  auto one=singleton->forward(one_x,scale.slice(0,0,2),dc.slice(0,0,2),op.slice(0,0,2),
      torch::tensor({0,1},ids),ref,ratio,torch::zeros_like(centers),eligible.slice(0,0,2));
  TORCH_CHECK(one[0][0].item<float>()!=one[1][0].item<float>(),"Singleton position lost");
  one[0][0].backward();TORCH_CHECK(one_x.grad()[0][0].item<float>()!=0,"Singleton position VJP lost");

  // Exercise registered schema, frozen clone and native checkpoint roundtrip.
  auto net=GumbelNetwork(.1,true,false,true,std::vector<double>{},0);net->to(o.device());
  net->anchors()->enableVariableMembers();net->enableSelectorSetEncoder();
  int64_t parameters=0;for(auto p:net->parameters())parameters+=p.numel();
  TORCH_CHECK(parameters==3202,"Shared Selector parameter count");
  auto path=std::filesystem::path(argc>2?argv[2]:"/tmp/cartgs_anchor_set_encoder.pt");
  torch::serialize::OutputArchive archive;net->save(archive);archive.save_to(path.string());
  torch::serialize::OutputArchive enc;encoder->save(enc);enc.save_to(path.string()+".encoder.pt");
  torch::serialize::OutputArchive fixture;
  fixture.write("xyz",xyz);fixture.write("scale",scale);fixture.write("dc",dc);fixture.write("opacity",op);
  fixture.write("owners",owners,true);fixture.write("ref",ref,true);fixture.write("ratio",ratio,true);
  fixture.write("centers",centers,true);fixture.write("eligible",eligible,true);
  fixture.write("masked_output",masked.detach(),true);fixture.write("mixed_eligible",valid,true);
  for(size_t i=0;i<masked_grad.size();++i)fixture.write("masked_grad_"+std::to_string(i),masked_grad[i],true);
  fixture.write("output",out.detach(),true);fixture.write("gx",xyz.grad(),true);fixture.write("gs",scale.grad(),true);
  fixture.write("gc",dc.grad(),true);fixture.write("go",op.grad(),true);
  for(const auto& p:encoder->named_parameters()) {
    auto name=p.key();std::replace(name.begin(),name.end(),'.','_');fixture.write("grad_"+name,p.value().grad(),true);
  }
  fixture.save_to(path.string()+".fixture.pt");
  auto restored=GumbelNetwork(.1,true,false,true,std::vector<double>{},0);restored->to(o.device());
  torch::serialize::InputArchive saved;saved.load_from(path.string(),o.device());restored->load(saved);
  TORCH_CHECK(restored->selectorSetEncoder(),"Checkpoint Set Encoder schema");
  TORCH_CHECK(restored->selectorSetEncoderSchema()==2,"Nominal/eligible checkpoint schema");
  auto original=net->named_parameters(),loaded=restored->named_parameters();
  for(const auto& p:original)TORCH_CHECK(p.value().equal(loaded[p.key()]),"Checkpoint parameter identity");
  auto frozen=net->cloneFrozen();TORCH_CHECK(frozen->selectorSetEncoder(),"Frozen teacher Set Encoder");
  TORCH_CHECK(frozen->selectorSetEncoderSchema()==2,"Frozen nominal/eligible schema");
  for(const auto& p:frozen->named_parameters())TORCH_CHECK(!p.value().requires_grad() && p.value().equal(original[p.key()]),"Frozen clone identity");
  if(argc>3) {
    auto legacy=GumbelNetwork(.1,true,false,true,std::vector<double>{},0);legacy->to(o.device());
    torch::serialize::InputArchive old;old.load_from(argv[3],o.device());legacy->load(old);
    TORCH_CHECK(legacy->selectorSetEncoderSchema()==1,"Historical checkpoint silently upgraded");
    auto teacher=legacy->cloneFrozen();TORCH_CHECK(teacher->selectorSetEncoderSchema()==1,"Historical frozen schema changed");
    auto old_parameters=legacy->named_parameters();
    for(const auto& p:teacher->named_parameters())
      TORCH_CHECK(!p.value().requires_grad() && p.value().equal(old_parameters[p.key()]),"Historical frozen weights changed");
  }
  std::cout<<"PASS Set Encoder: nominal positions, slot eligibility, singleton VJP, inactive isolation, all-inactive finite VJP, variable 1/2/3/4/10, mean/max/count, permutation/translation, neighbor context, reorder, empty, gradients, 3202 params, checkpoint/frozen clone "
      <<(cuda?"CUDA":"CPU")<<'\n';
}
