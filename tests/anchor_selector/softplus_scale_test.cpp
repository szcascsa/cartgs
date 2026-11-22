#include "improvements/selector/gumbel_network.h"
#include "include/anchor_selector_cuda.h"
#include <filesystem>
#include <iostream>

using T=torch::Tensor;
using List=std::vector<T>;
using improvements::selector::GumbelNetwork;
using improvements::selector::SpatialCompensation;

GumbelNetwork network(torch::Device device) {
  auto net=GumbelNetwork(.1,true,false,true,std::vector<double>{},0);
  net->to(device);net->anchors()->enableVariableMembers();
  net->enableSpatialCompensation(false,false,0,{.01,.05,.1,.2,.3,.4},true,false,true,false,48,1);
  net->enableSelectorContextResidual();
  return net;
}

void schema(torch::Device device,const std::filesystem::path& root) {
  auto net=network(device);auto field=net->spatialCompensation();
  const auto rng=at::globalContext().defaultGenerator(torch::kCPU).get_state();
  int64_t parameters=0;for(auto p:net->parameters())parameters+=p.numel();
  field->enableSoftplusScale();field->enableSoftplusScale();
  TORCH_CHECK(rng.equal(at::globalContext().defaultGenerator(torch::kCPU).get_state()),"Schema consumed RNG");
  int64_t after=0;for(auto p:net->parameters())after+=p.numel();
  TORCH_CHECK(after==parameters,"Scale schema added parameters");
  const auto path=root/(device.is_cuda()?"softplus_cuda.pt":"softplus_cpu.pt");
  torch::save(net,path.string());auto loaded=network(device);torch::load(loaded,path.string());
  TORCH_CHECK(loaded->spatialCompensation()->softplusScale(),"Softplus schema lost on load");
  auto frozen=loaded->cloneFrozen();TORCH_CHECK(frozen->spatialCompensation()->softplusScale(),"Frozen schema lost");
  for(const auto& p:net->named_parameters()) {
    TORCH_CHECK(p.value().equal(loaded->named_parameters()[p.key()]),"Checkpoint weights changed");
    TORCH_CHECK(p.value().equal(frozen->named_parameters()[p.key()])&&!frozen->named_parameters()[p.key()].requires_grad(),"Frozen weights changed");
  }
  bool rejected=false;try{field->enablePhysicalScaleAdditive();}catch(const c10::Error&){rejected=true;}
  TORCH_CHECK(rejected,"Conflicting Scale modes accepted");
  for(int mode:{0,1}) {
    auto old=network(device);if(mode)old->spatialCompensation()->enablePhysicalScaleAdditive();
    auto file=root/(std::string(mode?"additive_":"log_")+(device.is_cuda()?"cuda.pt":"cpu.pt"));
    torch::save(old,file.string());auto restore=network(device);torch::load(restore,file.string());
    TORCH_CHECK(!restore->spatialCompensation()->softplusScale()&&restore->spatialCompensation()->physicalScaleAdditive()==bool(mode),"Old checkpoint reinterpreted");
    auto incompatible=network(device);incompatible->spatialCompensation()->enableSoftplusScale();
    rejected=false;try{torch::load(incompatible,file.string());}catch(const c10::Error&){rejected=true;}
    TORCH_CHECK(rejected,"Old checkpoint accepted as Softplus");
  }
}

int main(int argc,char** argv) {
  const bool cuda=argc>1&&std::string(argv[1])=="cuda";
  if(cuda&&!torch::cuda::is_available())return 77;
  torch::manual_seed(192);torch::set_num_threads(2);
  const torch::Device device=cuda?torch::kCUDA:torch::kCPU;
  const std::filesystem::path root=argc>2?argv[2]:"/tmp/cartgs_softplus_scale_test";
  std::filesystem::create_directories(root);schema(device,root);
  auto o=torch::TensorOptions().dtype(torch::kFloat32).device(device);
  SpatialCompensation field(false,false,0,48,true,.01,.4,false,true,false);field->to(device);field->enableSoftplusScale();
  auto ell=torch::linspace(-20,8,301,o).unsqueeze(1).expand({301,3}).contiguous().requires_grad_(true);
  auto zero=torch::zeros_like(ell).requires_grad_(true);
  TORCH_CHECK(torch::allclose(field->decodeScale(ell,zero),ell.exp(),1e-7,0),"Step0 changed Scale");
  auto grad=torch::autograd::grad({field->decodeScale(ell,zero).sum()},{zero})[0];
  TORCH_CHECK((grad>0).all().item<bool>(),"Zero Head has no effective Scale gradient");
  auto low=field->decodeScale(ell,torch::full_like(ell,-2)),high=field->decodeScale(ell,torch::full_like(ell,2));
  TORCH_CHECK((low>0).all().item<bool>()&&(low<ell.exp()).all().item<bool>()&&(high>ell.exp()).all().item<bool>(),"Scale cannot shrink/grow positively");
  auto stress=torch::tensor({-1000.f,-200.f,-100.f,-20.f,0.f,20.f,100.f,1000.f},o).unsqueeze(1).expand({8,3}).contiguous().requires_grad_(true);
  auto base=torch::zeros_like(stress).requires_grad_(true);
  auto positive=field->decodeScale(base,stress);
  TORCH_CHECK(torch::isfinite(positive).all().item<bool>()&&(positive>0).all().item<bool>(),"Stress Scale not positive/finite");
  auto sg=torch::autograd::grad({positive.sum()},{base,stress});
  for(auto g:sg)TORCH_CHECK(torch::isfinite(g).all().item<bool>(),"Stress VJP not finite");
  torch::serialize::OutputArchive fixture;fixture.write("ell",base.detach(),true);fixture.write("u",stress.detach(),true);
  fixture.write("scale",positive.detach(),true);fixture.write("master_gradient",sg[0],true);fixture.write("head_gradient",sg[1],true);
  fixture.save_to((root/(cuda?"fixture_cuda.pt":"fixture_cpu.pt")).string());
  if(cuda)for(int64_t n:{0,1,17,257}) {
    auto q=torch::randn({n,4},o);q=q/q.norm(2,1,true);
    List a{torch::randn({n,3},o),torch::randn({n,3},o)-3,q,torch::randn({n,1},o),torch::randn({n,1,3},o),torch::randn({n,14},o)*.1,torch::rand({n},o)};
    if(n){a[5][0][3]=-1000;a[5][0][4]=1000;a[1][0][2]=-120;}
    List r;for(auto& v:a){v=v.clone().requires_grad_(true);r.push_back(v.detach().clone().requires_grad_(true));}
    auto out=mapping_cuda::decode14SoftplusScale(a[0],a[1],a[2],a[3],a[4],a[5],a[6]);
    auto old=mapping_cuda::decode14LogScale(r[0],r[1],r[2],r[3],r[4],r[5],r[6]);
    List wanted{old[0],field->decodeScale(r[1],r[5].slice(1,3,6)),old[2],old[3],old[4]},upstream;
    for(int j=0;j<5;++j){TORCH_CHECK(torch::allclose(out[j],wanted[j],2e-5,1e-6),"CUDA forward mismatch ",j);upstream.push_back(torch::randn_like(out[j]));}
    auto ag=torch::autograd::grad(out,a,upstream),rg=torch::autograd::grad(wanted,r,upstream);
    for(int j=0;j<7;++j)TORCH_CHECK(torch::isfinite(ag[j]).all().item<bool>()&&torch::allclose(ag[j],rg[j],5e-4,3e-6),"CUDA VJP mismatch ",j);
    if(n) {
      TORCH_CHECK((out[1]>0).all().item<bool>(),"CUDA nonpositive Scale");
      auto only=mapping_cuda::decode14SoftplusScale(a[0],a[1],a[2],a[3],a[4],a[5],a[6]);
      auto single=torch::autograd::grad({only[1].sum()},{a[1],a[5]});
      for(auto g:single)TORCH_CHECK(torch::isfinite(g).all().item<bool>(),"Partial-output VJP");
      auto invalid=a;invalid[5]=a[5].detach().clone();invalid[5][0][3]=NAN;
      auto nan=mapping_cuda::decode14SoftplusScale(invalid[0],invalid[1],invalid[2],invalid[3],invalid[4],invalid[5],invalid[6]);
      TORCH_CHECK(torch::isnan(nan[1][0][0]).item<bool>(),"NaN hidden by floor");
    }
    std::cout<<"CUDA_SOFTPLUS_PASS N="<<n<<std::endl;
  }
  std::cout<<"SOFTPLUS_SCALE_PASS "<<(cuda?"CUDA":"CPU")<<" zero-init, positive, shrink/grow, stress VJP, schema, frozen, legacy compatibility"<<std::endl;
}
