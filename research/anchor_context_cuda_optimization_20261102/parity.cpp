#include "improvements/selector/gumbel_network.h"
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <iostream>

using T = torch::Tensor;
using V = std::vector<T>;
int comparisons=0;
void same(const T& a,const T& b,const std::string& name) {
  TORCH_CHECK(a.sizes()==b.sizes() && a.scalar_type()==b.scalar_type(),name," layout");
  auto aa=a.contiguous(),bb=b.contiguous();
  if(aa.scalar_type()==torch::kFloat32){aa=aa.view(torch::kInt32);bb=bb.view(torch::kInt32);}
  TORCH_CHECK(aa.equal(bb),"BITWISE mismatch ",name," count=",(aa!=bb).sum().item<int64_t>());
  ++comparisons;
}
V run(const V& a,int64_t anchors,int mode) {
  return mode?mapping_cuda::anchorContextMLPOptimized(a,anchors,mode):mapping_cuda::anchorContextMLP(a,anchors);
}
V fixture(int n,int occupancy,int gap,torch::TensorOptions o,int& anchors) {
  const int groups=(n+occupancy-1)/occupancy;anchors=groups*gap+3;
  auto owners=torch::arange(n,o.dtype(torch::kLong)).floor_divide(occupancy)*gap+1;
  if(n)owners=owners.index_select(0,torch::randperm(n,owners.options()));
  auto valid=torch::rand({n},o)>.25;
  return {torch::randn({n,11},o).requires_grad_(true),
      (torch::randn({32,11},o)*.15).requires_grad_(true),(torch::randn({32},o)*.1).requires_grad_(true),
      (torch::randn({32,32},o)*.15).requires_grad_(true),(torch::randn({32},o)*.1).requires_grad_(true),
      (torch::randn({2,32},o)*.15).requires_grad_(true),(torch::randn({2},o)*.1).requires_grad_(true),
      (torch::randn({32,17},o)*.1).requires_grad_(true),owners,valid};
}
void check(const V& a,int anchors,const std::string& name) {
  auto base=run(a,anchors,0);V leaves(a.begin(),a.begin()+8);
  auto gz=torch::randn_like(base[0]),gh=torch::randn_like(base[1]);
  for(int mode:{1,3,5,7}) {
    auto fast=run(a,anchors,mode);same(base[0],fast[0],name+" logits");same(base[1],fast[1],name+" hidden");
    for(int which:{0,1,2}) {
      auto bg=which==0?torch::autograd::grad({base[0]},leaves,{gz},true):
          (which==1?torch::autograd::grad({base[1]},leaves,{gh},true,false,true):
              torch::autograd::grad({base[0],base[1]},leaves,{gz,gh},true));
      auto fg=which==0?torch::autograd::grad({fast[0]},leaves,{gz},true):
          (which==1?torch::autograd::grad({fast[1]},leaves,{gh},true,false,true):
              torch::autograd::grad({fast[0],fast[1]},leaves,{gz,gh},true));
      for(int j=0;j<8;++j){TORCH_CHECK(bg[j].defined()==fg[j].defined(),name," undefined VJP");
        if(bg[j].defined())same(bg[j],fg[j],name+" VJP "+std::to_string(mode)+"/"+std::to_string(which)+"/"+std::to_string(j));}
    }
  }
}
V clone(const V& a) {V out;for(auto t:a)out.push_back(t.detach().clone().requires_grad_(t.requires_grad()));return out;}
void adamReplay(torch::TensorOptions o,int mode) {
  int anchors;auto base=fixture(2049,4,3,o,anchors),fast=clone(base);
  V bp(base.begin(),base.begin()+8),fp(fast.begin(),fast.begin()+8);
  torch::optim::Adam bo(bp,torch::optim::AdamOptions(.001)),fo(fp,torch::optim::AdamOptions(.001));
  auto maturity=torch::rand({2049},o),extra=torch::randn({2049,32},o),gz=torch::randn({2049,2},o);
  for(int step=0;step<24;++step) {
    auto exp=torch::empty({2049,2},o).exponential_();bo.zero_grad();fo.zero_grad();
    V outputs[2];
    for(int branch=0;branch<2;++branch) {
      auto& a=branch?fast:base;auto& p=branch?fp:bp;
      auto out=run(a,anchors,branch?mode:0);
      auto selected=mapping_cuda::flexgsSampleCompactWithWeights(out[0],exp,a[9],1.);
      outputs[branch]={out[0],out[1],selected[0],selected[1],selected[2]};
      auto weights=selected[3].select(1,1),allowed=a[9].to(torch::kFloat32);
      auto selected_m=(weights*maturity*allowed).sum()/((weights*allowed).sum()+1e-8);
      auto mloss=torch::relu((maturity*allowed).sum()/allowed.sum()+.05-selected_m).square();
      V private_p(p.begin()+1,p.end());
      auto mg=torch::autograd::grad({mloss},private_p,{},true,false,true);
      for(int j=0;j<7;++j)if(mg[j].defined())private_p[j].mutable_grad()=mg[j].detach().clone();
      auto photo=(out[0]*gz).sum()*.001+(out[1]*extra).sum()*.0001;
      auto budget=(selected[1].sum()-2049*.05).abs()/(2049*.05);
      (photo+budget*.01).backward();
    }
    for(int j=0;j<5;++j)same(outputs[0][j],outputs[1][j],"Replay output/selection");
    for(int j=0;j<8;++j)same(bp[j].grad(),fp[j].grad(),"Replay dual VJP accumulation");
    bo.step();fo.step();
    for(int j=0;j<8;++j){same(bp[j],fp[j],"Adam parameter");
      auto& bs=static_cast<torch::optim::AdamParamState&>(*bo.state().at(bp[j].unsafeGetTensorImpl()));
      auto& fs=static_cast<torch::optim::AdamParamState&>(*fo.state().at(fp[j].unsafeGetTensorImpl()));
      TORCH_CHECK(bs.step()==fs.step(),"Adam step");same(bs.exp_avg(),fs.exp_avg(),"Adam first moment");same(bs.exp_avg_sq(),fs.exp_avg_sq(),"Adam second moment");}
  }
}
int main() {
  if(!torch::cuda::is_available())return 77;
  at::globalContext().setAllowTF32CuBLAS(false);torch::manual_seed(91);torch::set_num_threads(2);
  auto o=torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  for(int n:{0,1,7,1031,10000,50000,100000})for(int occupancy:{1,4,10}) {
    int anchors;auto a=fixture(n,occupancy,n<2000?3:1,o,anchors);
    check(a,anchors,"size "+std::to_string(n)+" occ "+std::to_string(occupancy));
  }
  int anchors;auto a=fixture(1001,10,9,o,anchors);
  for(int variant=0;variant<5;++variant) {
    auto b=clone(a);
    {torch::NoGradGuard guard;
      if(variant==0)b[9].zero_();
      if(variant==1){b[1].zero_();b[2].fill_(1);}
      if(variant==2){b[1].zero_();b[2].zero_();}
      if(variant==3)b[7].zero_();
      if(variant==4){b[0][0][0]=std::numeric_limits<float>::quiet_NaN();b[0][1][1]=std::numeric_limits<float>::infinity();}}
    check(b,anchors,"inactive/ties/zero/nonfinite "+std::to_string(variant));
  }
  // Two live graphs must own independent caches. Simulate prune/reorder/owners migration.
  auto keep=torch::randperm(1001,o.dtype(torch::kLong)).slice(0,0,501);auto b=clone(a);
  b[0]=b[0].index_select(0,keep).detach().requires_grad_(true);b[8]=b[8].index_select(0,keep);b[9]=b[9].index_select(0,keep);
  auto older=run(a,anchors,3),newer=run(b,anchors,3);
  V al(a.begin(),a.begin()+8),bl(b.begin(),b.begin()+8);
  auto bg=torch::autograd::grad({run(a,anchors,0)[0].sum()},al);
  auto ag=torch::autograd::grad({older[0].sum()},al);
  for(int j=0;j<8;++j)same(bg[j],ag[j],"Two live graphs");
  check(b,anchors,"Prune/reorder");b[8]=(b[8]+1).remainder(anchors);check(b,anchors,"Anchor migration");
  {c10::cuda::CUDAStreamGuard stream(c10::cuda::getStreamFromPool());
    int stream_anchors;auto streamed=fixture(1031,4,2,o,stream_anchors);check(streamed,stream_anchors,"Nondefault CUDA stream");}
  {int clone_anchors;auto c=fixture(17,4,2,o,clone_anchors);
    for(int j:{0,8,9})c[j]=torch::cat({c[j],c[j].slice(0,16,17)},0).detach().requires_grad_(c[j].requires_grad());
    check(c,clone_anchors,"Clone member");}
  // Empty table in addition to n=0 with nonempty Anchor table.
  auto empty=fixture(0,4,1,o,anchors);check(empty,0,"Empty scene");
  for(int mode:{1,3,5,7})adamReplay(o,mode);
  std::cout<<"PASS cached ContextResidual: "<<comparisons<<" bitwise comparisons; logits/hidden/input+parameter VJP, ties, inactive, sparse Anchors, nonfinite, graph lifetime, topology and 24-step dual-VJP Adam per mode\n";
}
