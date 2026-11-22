#include "improvements/selector/gumbel_network.h"
#include <filesystem>
#include <iostream>

using namespace improvements::selector;
bool bitwise(const torch::Tensor& a,const torch::Tensor& b) {
  return a.sizes()==b.sizes()&&a.contiguous().reshape({-1}).view(torch::kUInt8).equal(b.contiguous().reshape({-1}).view(torch::kUInt8));
}

void verify(torch::Device device,const std::filesystem::path& directory) {
  auto o=torch::TensorOptions().dtype(torch::kFloat32).device(device);
  torch::manual_seed(17);
  auto full=GumbelNetwork(.1,true,false,true,std::vector<double>{},0);full->to(device);
  full->anchors()->enableVariableMembers();
  full->enableSpatialCompensation(false,false,0,{.01,.05,.1,.3,.5,.7},true,false,true,false,48,1);
  full->spatialCompensation()->enableSoftplusScale();full->enableJointViewConditioning(2);
  auto x=torch::randn({31,3},o),center=torch::zeros_like(x),reference=torch::ones({31,1},o);
  auto scale=torch::zeros_like(x),rotation=torch::zeros({31,4},o);rotation.select(1,0).fill_(1);
  auto dc=torch::zeros({31,1,3},o),opacity=torch::zeros({31,1},o);
  full->spatialCompensation()->initialize(x,scale,rotation,dc,opacity,center,reference,true);
  {torch::NoGradGuard guard;
    for(auto p:full->spatialCompensation()->named_parameters())p.value().normal_(0,.03);
  }
  auto compact=full->cloneFrozen();for(auto p:compact->parameters())p.requires_grad_(true);
  auto rng=at::globalContext().defaultGenerator(torch::Device(torch::kCPU)).get_state().clone();
  int64_t before=0,after=0;for(auto p:compact->parameters())before+=p.numel();
  compact->spatialCompensation()->configureGeometryHeads(1);
  for(auto p:compact->parameters())after+=p.numel();
  TORCH_CHECK(before-after==4420&&rng.equal(at::globalContext().defaultGenerator(torch::Device(torch::kCPU)).get_state()),"Removal changed capacity/RNG unexpectedly");
  for(auto p:compact->named_parameters())TORCH_CHECK(p.key().find("heads.other.")==std::string::npos,"Other parameter retained");
  for(int n:{0,1,31,500,4096})for(double ratio:{.01,.03,.05,.1,.2,.3,.5,.7})for(bool local:{false,true}) {
    auto input=torch::randn({n,3},o),c=torch::zeros_like(input),s=torch::ones({n,1},o),f=torch::randn({n,48},o);
    auto q=torch::zeros({n,4},o);q.select(1,0).fill_(1);
    auto condition=local?torch::randn({n,8},o):torch::Tensor{};
    auto run=[&](std::shared_ptr<GumbelNetworkImpl> net){return net->spatialCompensation()->forward(input,input,q,torch::zeros({n,1,3},o),torch::zeros({n,1},o),c,s,ratio,true,f,{},condition);};
    torch::NoGradGuard guard;auto a=run(full.ptr()),b=run(compact);
    TORCH_CHECK(bitwise(a.slice(1,0,10),b.slice(1,0,10))&&b.slice(1,10,14).count_nonzero().item<int64_t>()==0,"Output geometry changed or Other not zero");
  }
  auto feature=torch::randn({31,48},o),condition=torch::randn({31,8},o),vjp=torch::randn({31,14},o);vjp.slice(1,10,14).zero_();
  std::vector<std::string> names;
  for(auto p:full->spatialCompensation()->named_parameters())if(p.key().rfind("heads.other.",0)!=0)names.push_back(p.key());
  auto derivatives=[&](std::shared_ptr<GumbelNetworkImpl> net) {
    auto input=x.detach().clone().requires_grad_(true),c=center.detach().clone().requires_grad_(true);
    auto ref=reference.detach().clone().requires_grad_(true),f=feature.detach().clone().requires_grad_(true);
    auto out=net->spatialCompensation()->forward(input,scale,rotation,dc,opacity,c,ref,.05,true,f,{},condition);
    std::vector<torch::Tensor> parameters{input,c,ref,f};
    auto named=net->spatialCompensation()->named_parameters();
    for(const auto& name:names)parameters.push_back(named[name]);
    return torch::autograd::grad({out},parameters,{vjp});
  };
  auto a=derivatives(full.ptr()),b=derivatives(compact);TORCH_CHECK(a.size()==b.size(),"Gradient size mismatch");
  for(size_t i=0;i<a.size();++i)TORCH_CHECK(bitwise(a[i],b[i]),"Geometry/shared gradient mismatch at ",i<4?std::to_string(i):names[i-4]);
  std::filesystem::create_directories(directory);torch::serialize::OutputArchive saved;compact->save(saved);saved.save_to((directory/"selector.pt").string());
  auto restored=GumbelNetwork(.1,true,false,true,std::vector<double>{},0);restored->to(device);
  torch::serialize::InputArchive archive;archive.load_from((directory/"selector.pt").string(),device);restored->load(archive);
  auto frozen=restored->cloneFrozen();TORCH_CHECK(restored->spatialCompensation()->geometryHeadVariant()==1&&frozen->spatialCompensation()->geometryHeadVariant()==1,"Restore schema mismatch");
  for(auto p:compact->named_parameters())TORCH_CHECK(bitwise(p.value(),restored->named_parameters()[p.key()])&&bitwise(p.value(),frozen->named_parameters()[p.key()]),"Restore/frozen weight mismatch");
  bool rejected=false;try{restored->spatialCompensation()->configureGeometryHeads(2);}catch(const c10::Error&){rejected=true;}
  TORCH_CHECK(rejected,"Cannot reinterpret no-Other as merged MLP");
  torch::serialize::OutputArchive fixture;
  for(auto item:std::vector<std::pair<std::string,torch::Tensor>>{{"xyz",x},{"scale",scale},{"rotation",rotation},{"dc",dc},
      {"opacity",opacity},{"center",center},{"reference",reference},{"feature",feature},{"condition",condition},
      {"local_output",compact->spatialCompensation()->forward(x,scale,rotation,dc,opacity,center,reference,.05,true,feature,{},condition)},
      {"global_output",compact->spatialCompensation()->forward(x,scale,rotation,dc,opacity,center,reference,.05,true,feature)}})
    fixture.write(item.first,item.second.detach(),true);
  fixture.save_to((directory/"fixture.pt").string());
  std::cout<<"PASS Residual48 no-Other "<<device<<": 80 shapes/ratio/view cases, geometry/shared VJP bitwise, 4420 parameters removed, RNG, checkpoint/frozen\n";
}

int main(int argc,char**argv) {
  torch::set_num_threads(2);at::globalContext().setAllowTF32CuBLAS(false);
  std::filesystem::path out=argc>1?argv[1]:"/tmp/residual48_no_other";
  verify(torch::kCPU,out/"cpu");if(torch::cuda::is_available())verify(torch::kCUDA,out/"cuda");return 0;
}
