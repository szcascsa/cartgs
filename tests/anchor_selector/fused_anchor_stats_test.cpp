#include "include/anchor_stats_cuda.h"
#include "improvements/selector/gumbel_network.h"
#include <cuda_runtime_api.h>
#include <nvToolsExt.h>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <iostream>
namespace fs=std::filesystem;using T=torch::Tensor;using L=std::vector<T>;
using Net=improvements::selector::GumbelNetwork;using Bank=improvements::selector::AnchorBank;
L initial(Bank& b){return {torch::zeros({b->size()},b->features().options().dtype(torch::kFloat64)),torch::zeros({b->size()},b->rows().options()),
    torch::zeros({b->rows().numel()},b->features().options().dtype(torch::kFloat64)),torch::zeros_like(b->rows()),torch::zeros_like(b->rows())};}
void legacy(Bank& b,const T& visible,const T& opacity,L& s){
  b->assertBlocks(b->rows().numel());auto mark=torch::zeros({b->size()},opacity.options().dtype(torch::kBool));mark.index_fill_(0,b->rows().index_select(0,visible),true);
  auto values=opacity.reshape({-1}).to(torch::kFloat64);b->refreshSlotActivity(opacity);auto active=mark.index_select(0,b->rows());
  auto observed=torch::where(active,values,torch::zeros_like(values));s[0].index_add_(0,b->rows(),observed);s[1]+=mark.to(torch::kLong);
  s[2]+=observed;s[3]+=((values<.02)&active).to(torch::kLong);s[4]+=((values<1./255)&active).to(torch::kLong);
}
void fused(Bank& b,const T& visible,const T& opacity,L& s){b->assertBlocks(b->rows().numel());mapping_cuda::anchorFullStats(b->rows(),b->validatedMembers(),b->memberCounts(),visible,opacity,s,b->inactiveSlots(),b->recoveredSlots());}
void compare(std::ofstream& log,const std::string& name,const T& x,const T& y,bool exact){
  auto d=(x.to(torch::kFloat64)-y.to(torch::kFloat64)).abs();bool ok=exact?torch::equal(x,y):torch::allclose(x,y,2e-14,1e-12);
  int64_t idx=d.numel()?d.reshape({-1}).argmax().item<int64_t>():-1;
  log<<name<<','<<exact<<','<<(d.numel()?d.max().item<double>():0)<<','<<(d.numel()?(d/x.to(torch::kFloat64).abs().clamp_min(1e-300)).max().item<double>():0)<<','
    <<(d.numel()?d.square().mean().sqrt().item<double>():0)<<','<<idx<<','<<ok<<'\n';TORCH_CHECK(ok,"Anchor stats parity ",name);
}
int main(int argc,char** argv){try{
  TORCH_CHECK(argc==4,"fused_anchor_stats_test checkpoint fixture out");torch::set_num_threads(2);torch::manual_seed(2026);fs::path cp=argv[1],out=argv[3];fs::create_directories(out);
  Net net(.1,true,false,true);net->to(torch::kCUDA);torch::serialize::InputArchive ar;ar.load_from((cp/"selector.pt").string(),torch::kCUDA);net->load(ar);
  auto na=net->cloneFrozen(),nb=net->cloneFrozen();auto a=na->anchors(),b=nb->anchors();a->setFusedFullStats(false);b->setFusedFullStats(true);
  torch::serialize::InputArchive fixture;fixture.load_from(argv[2],torch::kCUDA);T radii,logits;fixture.read("radii",radii);fixture.read("parameter_before_3",logits);
  auto vis=torch::nonzero(radii>0).reshape({-1});auto opacity=logits.sigmoid();const auto n=a->rows().numel();
  auto sa=initial(a),sb=initial(b);std::ofstream log(out/"anchor_parity.csv");log<<"name,exact,max_abs,max_rel,rms,worst_index,passed\n";
  std::ofstream timing(out/"anchor_timing.csv");timing<<"repeat,implementation,stage,host_ms,cuda_ms,N,A\n";
  for(int t=0;t<1000;++t){T visible=t%11==0?vis.slice(0,0,0):(t%13==0?torch::cat({vis,vis}):vis);
    auto alpha=t%7==0?opacity*.001:opacity;legacy(a,visible,alpha,sa);fused(b,visible,alpha,sb);
    if(t%47==0||t==999){for(int i=0;i<5;++i)compare(log,"window"+std::to_string(t)+"/"+std::to_string(i),sa[i],sb[i],i!=0&&i!=2);
      compare(log,"inactive",a->inactiveSlots(),b->inactiveSlots(),true);compare(log,"recovered",a->recoveredSlots(),b->recoveredSlots(),true);
      auto enough=sa[1]>400,prune_a=enough&(sa[0]<.005*sa[1].to(torch::kFloat64)),prune_b=(sb[1]>400)&(sb[0]<.005*sb[1].to(torch::kFloat64));compare(log,"prune",prune_a,prune_b,true);
      a->closeSlotWindow(enough,sa[1],sa[4],alpha);b->closeSlotWindow(sb[1]>400,sb[1],sb[4],alpha);compare(log,"closed_slots",a->inactiveSlots(),b->inactiveSlots(),true);}
  }
  // Window boundary with FP64 accumulated state immediately adjacent to threshold.
  sa=initial(a);sb=initial(b);sa[1].fill_(401);sb[1].fill_(401);
  auto target=.005*sa[1].to(torch::kFloat64);sa[0].copy_(torch::nextafter(target,torch::zeros_like(target)));sb[0].copy_(sa[0]);
  legacy(a,vis,opacity,sa);fused(b,vis,opacity,sb);compare(log,"threshold_prune",sa[0]<.005*sa[1].to(torch::kFloat64),sb[0]<.005*sb[1].to(torch::kFloat64),true);
  // Every invalid-data category must report failure; no illegal owner is dereferenced.
  const auto originals=L{b->rows().clone(),b->memberCounts().clone(),b->levels().clone(),b->centers().clone(),b->offsetScales().clone()};
  for(int kind=0;kind<6;++kind){torch::NoGradGuard guard;
    if(kind==0)b->rows()[0].fill_(-1);if(kind==1)b->memberCounts()[0].fill_(11);if(kind==2)b->memberCounts()[0].add_(1);
    if(kind==3)b->levels()[0].fill_(3);if(kind==4)b->centers()[0][0].fill_(std::numeric_limits<float>::quiet_NaN());if(kind==5)b->offsetScales()[0].fill_(0);
    bool old=false,now=false;try{b->setFusedFullStats(false);b->assertBlocks(n);}catch(...){old=true;}
    try{b->setFusedFullStats(true);b->assertBlocks(n);}catch(...){now=true;}
    TORCH_CHECK(old&&now,"Invalid layout accepted ",kind);
    b->rows().copy_(originals[0]);b->memberCounts().copy_(originals[1]);b->levels().copy_(originals[2]);b->centers().copy_(originals[3]);b->offsetScales().copy_(originals[4]);
  }
  cudaEvent_t begin,end;cudaEventCreate(&begin);cudaEventCreate(&end);
  for(int t=-10;t<100;++t)for(int order=0;order<2;++order)for(const std::string stage:{"validate","full_stats"}){
    bool use=(t+order)%2==0;auto& bank=use?b:a;auto& s=use?sb:sa;torch::cuda::synchronize();nvtxRangePushA((std::string(use?"fused/":"legacy/")+stage).c_str());
    cudaEventRecord(begin);auto start=std::chrono::steady_clock::now();if(stage=="validate")bank->assertBlocks(n);else if(use)fused(bank,vis,opacity,s);else legacy(bank,vis,opacity,s);
    double host=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();cudaEventRecord(end);cudaEventSynchronize(end);float elapsed;cudaEventElapsedTime(&elapsed,begin,end);nvtxRangePop();
    if(t>=0)timing<<t<<','<<(use?"fused":"legacy")<<','<<stage<<','<<host<<','<<elapsed<<','<<n<<','<<bank->size()<<'\n';
  }
  // Fresh identical topology event, clones excluded from timing.
  for(int rep=0;rep<25;++rep)for(bool partial:{false,true})for(bool use:{false,true}) {
    auto model=net->cloneFrozen();auto bank=model->anchors();bank->setFusedFullStats(use);auto keep=torch::ones_like(bank->rows(),torch::kBool);
    if(partial){auto owner=torch::nonzero(bank->memberCounts()>1).reshape({-1})[0];auto ids=torch::nonzero(bank->rows()==owner).reshape({-1});keep[ids[0]].fill_(false);}
    auto before_uid=bank->blockUIDs().clone();auto before_keys=bank->named_buffers()["keys"].clone();
    torch::cuda::synchronize();cudaEventRecord(begin);auto start=std::chrono::steady_clock::now();
    nvtxRangePushA(use?"fused/topology_prune":"legacy/topology_prune");bank->pruneVariableMembers(keep,{});nvtxRangePop();
    double host=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();cudaEventRecord(end);cudaEventSynchronize(end);float ms;cudaEventElapsedTime(&ms,begin,end);
    timing<<rep<<','<<(use?"fused":"legacy")<<','<<(partial?"topology_partial":"topology_noop")<<','<<host<<','<<ms<<','<<n<<','<<bank->size()<<'\n';
    TORCH_CHECK(torch::equal(before_uid,bank->blockUIDs())&&torch::equal(before_keys,bank->named_buffers()["keys"]),"Prune changed spatial keys/UID without empty Anchor");
    TORCH_CHECK(torch::equal(bank->rows(),net->anchors()->rows().index({keep})),"Prune owner mismatch");
  }
  cudaEventDestroy(begin);cudaEventDestroy(end);std::cout<<"PASS stats windows1000,FP64,visits,prune,slots,all validation errors,paired timing\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
