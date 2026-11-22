#include "improvements/selector/gumbel_network.h"
#include <filesystem>
#include <iostream>

using improvements::selector::AnchorSetEncoder;
using improvements::selector::GumbelNetwork;

int main(int argc,char** argv) {
  const bool cuda=argc>1 && std::string(argv[1])=="cuda";
  if(cuda && !torch::cuda::is_available())return 77;
  torch::manual_seed(53);torch::set_num_threads(2);
  auto o=torch::TensorOptions().dtype(torch::kFloat32).device(cuda?torch::kCUDA:torch::kCPU);
  auto ids=o.dtype(torch::kLong),bo=o.dtype(torch::kBool);
  auto owners=torch::tensor({6,1,1,4,4,4,2,2,2,2,5,5,5,5,5,5,5,5,5,5},ids);
  const int64_t n=owners.numel(),a=8;
  auto centers=torch::rand({a,3},o),ref=torch::full({a,1},.1,o),ratio=torch::full({1,1},.05,o);
  auto xyz=(centers.index_select(0,owners)+torch::randn({n,3},o)*.02).detach().requires_grad_(true);
  auto scale=torch::rand({n,3},o).requires_grad_(true),dc=torch::rand({n,1,3},o).requires_grad_(true);
  auto op=torch::rand({n,1},o).requires_grad_(true);
  auto eligible=(torch::arange(n,ids).remainder(3)!=1)&(owners!=4);
  AnchorSetEncoder encoder(3);encoder->to(o.device());
  TORCH_CHECK(encoder->parameters().empty() && !encoder->network,"Statistics cannot contain a learned Encoder");
  auto out=encoder->forward(xyz,scale,dc,op,owners,ref,ratio,centers,eligible);
  TORCH_CHECK(out.sizes()==torch::IntArrayRef({n,21}) && torch::isfinite(out).all().item<bool>(),"Statistics shape/finite");
  for(int64_t k:{1,2,4,5,6}) {
    auto rows=torch::nonzero(owners==k).flatten(),valid=torch::nonzero((owners==k)&eligible).flatten();
    auto expected=torch::zeros({10},o);
    if(valid.numel()) {
      auto q=(xyz.index_select(0,valid)-centers[k])/ref[k];auto mu=q.mean(0),d=q-mu;
      auto cov=d.t().mm(d)/valid.numel();
      expected.slice(0,0,3).copy_(mu);
      expected.slice(0,3,9).copy_(torch::stack({cov[0][0],cov[0][1],cov[0][2],cov[1][1],cov[1][2],cov[2][2]}));
      expected[9].fill_(valid.numel()/10.);
    }
    TORCH_CHECK(torch::allclose(out.index_select(0,rows).slice(1,11,21),expected.expand({rows.numel(),10}),1e-5,1e-6),"Independent covariance oracle");
  }
  auto polluted=encoder->forward(torch::where(eligible.unsqueeze(1),xyz,torch::full_like(xyz,1000)),
      torch::where(eligible.unsqueeze(1),scale,torch::full_like(scale,1000)),
      torch::where(eligible.reshape({n,1,1}),dc,torch::full_like(dc,1000)),op,owners,ref,ratio,centers,eligible);
  TORCH_CHECK(torch::allclose(out,polluted,1e-5,1e-6),"Inactive attribute pollution");
  auto perm=torch::randperm(n,ids);
  auto permuted=encoder->forward(xyz.index_select(0,perm),scale.index_select(0,perm),dc.index_select(0,perm),op.index_select(0,perm),
      owners.index_select(0,perm),ref,ratio,centers,eligible.index_select(0,perm));
  TORCH_CHECK(torch::allclose(permuted,out.index_select(0,perm),1e-5,1e-6),"Statistics member permutation");
  auto g=torch::randn_like(out);auto leaves=torch::autograd::variable_list{xyz,scale,dc,op};
  auto gradients=torch::autograd::grad({out},leaves,{g});
  for(auto v:gradients)TORCH_CHECK(torch::isfinite(v).all().item<bool>() && v.index({~eligible}).abs().sum().item<float>()==0,"Statistics VJP isolation");
  double max_forward=0,max_vjp=0;
  auto fused=encoder->forward(xyz,scale,dc,op,owners,ref,ratio,centers,eligible,true);
  TORCH_CHECK(torch::allclose(fused,out,2e-5,2e-6),"CUDA statistics forward/reference");
  auto fg=torch::autograd::grad({fused},leaves,{g});
  max_forward=(fused-out).abs().max().item<double>();
  for(int j=0;j<4;++j){TORCH_CHECK(torch::allclose(fg[j],gradients[j],1e-4,2e-5),"CUDA statistics VJP/reference");
    max_vjp=std::max(max_vjp,(fg[j]-gradients[j]).abs().max().item<double>());}
  auto repeated=encoder->forward(xyz,scale,dc,op,owners,ref,ratio,centers,eligible,true);
  TORCH_CHECK(fused.equal(repeated),"CUDA grouping arrival order changed output");
  for(bool accelerated:{false,true}) {
    auto none=encoder->forward(xyz,scale,dc,op,owners,ref,ratio,centers,torch::zeros_like(eligible),accelerated);
    TORCH_CHECK(torch::isfinite(none).all().item<bool>() && none.slice(1,11,21).abs().sum().item<float>()==0,"All-inactive statistics");
    for(auto v:torch::autograd::grad({none},leaves,{g}))TORCH_CHECK(torch::isfinite(v).all().item<bool>() && v.abs().sum().item<float>()==0,"All-inactive statistics VJP");
    auto empty=encoder->forward(xyz.slice(0,0,0),scale.slice(0,0,0),dc.slice(0,0,0),op.slice(0,0,0),owners.slice(0,0,0),
        ref.slice(0,0,0),ratio,centers.slice(0,0,0),eligible.slice(0,0,0),accelerated);
    TORCH_CHECK(empty.sizes()==torch::IntArrayRef({0,21}),"Empty Statistics21 population");
    torch::autograd::grad({empty.sum()},leaves,{},false,false,true);
  }
  auto net=GumbelNetwork(.1,true,false,true,std::vector<double>{},0);net->to(o.device());
  net->anchors()->enableVariableMembers();net->enableSelectorSetEncoder(3);
  int64_t parameters=0;for(auto p:net->parameters())parameters+=p.numel();
  TORCH_CHECK(parameters==1826,"Statistics21 parameter count");
  auto path=std::filesystem::path(argc>2?argv[2]:"/tmp/cartgs_anchor_statistics21.pt");
  torch::serialize::OutputArchive saved;net->save(saved);saved.save_to(path.string());
  auto restored=GumbelNetwork(.1,true,false,true,std::vector<double>{},0);restored->to(o.device());
  torch::serialize::InputArchive archive;archive.load_from(path.string(),o.device());restored->load(archive);
  TORCH_CHECK(restored->selectorSetEncoderSchema()==3 && restored->named_parameters().size()==net->named_parameters().size(),"Statistics checkpoint schema");
  auto teacher=restored->cloneFrozen();TORCH_CHECK(teacher->selectorSetEncoderSchema()==3,"Statistics frozen schema");
  for(const auto& p:teacher->named_parameters())TORCH_CHECK(!p.value().requires_grad()&&p.value().equal(net->named_parameters()[p.key()]),"Statistics frozen weights");
  torch::serialize::OutputArchive fixture;
  fixture.write("xyz",xyz,true);fixture.write("scale",scale,true);fixture.write("dc",dc,true);fixture.write("opacity",op,true);
  fixture.write("owners",owners,true);fixture.write("centers",centers,true);fixture.write("ref",ref,true);fixture.write("ratio",ratio,true);
  fixture.write("eligible",eligible,true);fixture.write("output",out.detach(),true);fixture.write("fused_output",fused.detach(),true);fixture.write("vjp",g,true);
  for(int j=0;j<4;++j){fixture.write("grad_"+std::to_string(j),gradients[j],true);fixture.write("fused_grad_"+std::to_string(j),fg[j],true);}
  fixture.save_to(path.string()+".fixture.pt");
  if(cuda) {
    for(int64_t count:{0,1,1031,20000}) {
      auto input=torch::randn({count,21},o).requires_grad_(true);
      auto w0=torch::randn({32,21},o).mul(.02).requires_grad_(true),b0=torch::randn({32},o).mul(.02).requires_grad_(true);
      auto w1=torch::randn({32,32},o).mul(.02).requires_grad_(true),b1=torch::randn({32},o).mul(.02).requires_grad_(true);
      auto w2=torch::randn({2,32},o).mul(.02).requires_grad_(true),b2=torch::randn({2},o).mul(.02).requires_grad_(true);
      auto h=torch::relu(torch::nn::functional::linear(input,w0,b0));
      auto z=torch::nn::functional::linear(torch::relu(torch::nn::functional::linear(h,w1,b1)),w2,b2);
      auto accelerated=mapping_cuda::flexgsMLP(input,w0,b0,w1,b1,w2,b2);
      auto gz=torch::randn_like(z),gh=torch::randn_like(h);
      TORCH_CHECK(torch::allclose(z,accelerated[0],2e-5,2e-6) && torch::allclose(h,accelerated[1],2e-5,2e-6),"21D tiled MLP forward");
      auto p=torch::autograd::variable_list{input,w0,b0,w1,b1,w2,b2};
      auto expected=torch::autograd::grad({z,h},p,{gz,gh}),actual=torch::autograd::grad({accelerated[0],accelerated[1]},p,{gz,gh});
      double error=0;
      for(size_t j=0;j<p.size();++j) {
        TORCH_CHECK(torch::allclose(actual[j],expected[j],2e-4,2e-4),"21D tiled MLP VJP");
        if(actual[j].numel())error=std::max(error,(actual[j]-expected[j]).abs().max().item<double>());
      }
      std::cout<<"MLP21_PARITY N="<<count<<" forward_max_abs="<<(z.numel()?(z-accelerated[0]).abs().max().item<double>():0.)
          <<" vjp_max_abs="<<error<<'\n';
    }
  }
  std::cout<<"PASS Statistics21 "<<(cuda?"CUDA":"CPU")<<" mean/covariance/count oracle, eligibility, singleton/mixed occupancy, empty/all-inactive, VJP, 1826 parameters, schema/frozen, MLP21; max_forward="
      <<max_forward<<" max_vjp="<<max_vjp<<'\n';
}
