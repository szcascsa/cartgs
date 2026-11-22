#include "include/anchor_si.h"
#include "include/gaussian_rasterizer.h"
#include "cuda_rasterizer/anchor_si_capture.h"
#include <fstream>
#include <iostream>

int main(int argc,char** argv) { try {
  TORCH_CHECK(argc==3,"anchor_si_test checkpoint output");torch::set_num_threads(2);torch::manual_seed(2026);
  namespace fs=std::filesystem;fs::create_directories(argv[2]);auto opts=torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
  std::ofstream csv(fs::path(argv[2])/"parity.csv");csv<<"name,max_abs,rms\n";
  auto compare=[&](const char* name,const torch::Tensor& a,const torch::Tensor& b){
    auto d=(a-b).abs();csv<<name<<','<<(d.numel()?d.max().item<double>():0)<<','<<(d.numel()?d.square().mean().sqrt().item<double>():0)<<'\n';
    TORCH_CHECK(torch::allclose(a,b,2e-5,2e-6),name);};
  auto net=improvements::selector::GumbelNetwork(.1,true,false,true);net->to(torch::kCUDA);
  torch::serialize::InputArchive archive;archive.load_from((fs::path(argv[1])/"selector.pt").string(),torch::kCUDA);net->load(archive);
  auto bank=net->anchors();bank->setFusedFullStats(true);bank->assertBlocks(bank->rows().numel());
  auto owner=bank->rows(),counts=bank->memberCounts(),members=bank->validatedMembers();const auto n=owner.numel(),a=counts.numel();
  auto opacity=torch::rand({n,2},opts).select(1,0),inactive=torch::rand({n},opts)>.7;
  opacity.slice(0,0,3).fill_(1.f/255.f);
  auto value=torch::rand({n,2},opts).select(1,1).detach().requires_grad_(true);
  auto eligible=~(inactive&(opacity<1.f/255.f));
  auto expected_count=torch::zeros({a},opts).index_add(0,owner,eligible.to(torch::kFloat32));
  auto reference=torch::zeros({a},opts).index_add(0,owner,value*eligible)/expected_count.clamp_min(1);
  auto output=anchor_si::reduce(value,owner,members,counts,opacity,inactive);
  TORCH_CHECK(torch::equal(expected_count,output[1]),"Eligible counts must be exact");compare("mean",reference,output[0]);
  auto weight=torch::randn({a},opts);auto ga=torch::autograd::grad({reference},{value},{weight},true)[0];
  auto gb=torch::autograd::grad({output[0]},{value},{weight},true)[0];compare("VJP",ga,gb);
  auto empty=anchor_si::reduce(torch::empty({0},opts),torch::empty({0},owner.options()),torch::empty({0,10},owner.options()),torch::empty({0},owner.options()),torch::empty({0},opts),{});
  TORCH_CHECK(!empty[0].numel()&&!empty[1].numel(),"Empty reduction");
  auto visits=torch::full({a},4,owner.options()),last=torch::zeros_like(visits),ema=torch::rand({a},opts)+1e-5;
  auto ratios=torch::tensor({.01f,.05f,.1f,.15f},opts);
  auto target=anchor_si::targets(ema,visits,last,expected_count,ratios,0,n,4,128,1);
  for(int k=0;k<4;++k)TORCH_CHECK(std::abs((target[k]*expected_count).sum().item<double>()-ratios[k].item<double>()*n)<.02,"Member-weighted budget calibration");
  auto expired=anchor_si::targets(ema,visits,last,expected_count,ratios,129,n,4,128,1);
  TORCH_CHECK(torch::equal(expired,-torch::ones_like(expired)),"Unobserved/expired must not become negative labels");
  anchor_si::State state;auto ones=torch::ones({n},opts),alpha=torch::ones({n,1},opts)*.5;
  anchor_si::Config config;config.enabled=true;state.configure(config);
  for(int i=0;i<16;++i)if(state.captureNextFull())state.observe(net,ones,alpha,100,1,{.01,.05,.1,.15});
  TORCH_CHECK(torch::equal(state.visits(),torch::full_like(state.visits(),4)),"Observation exposure count");
  state.observe(net,torch::zeros_like(ones),alpha,100,1,{.01,.05,.1,.15});
  TORCH_CHECK(torch::equal(state.visits(),torch::full_like(state.visits(),4)),"Unobserved update counted as opacity zero");
  auto xyz=bank->centers().index_select(0,owner).detach().requires_grad_(true);
  auto scale=torch::full({n,3},.01,opts).requires_grad_(true),dc=torch::full({n,1,3},.2,opts).requires_grad_(true);
  alpha.requires_grad_(true);auto p=net->forwardSelectionFeatures(xyz,.05,1,{},scale,dc,alpha).probability;
  auto loss=state.loss(net,p,alpha,1,1);TORCH_CHECK(loss.defined()&&loss.requires_grad(),"SI loss unavailable");
  state.accumulateHeadGradients(loss,net);
  int received=0;for(auto& item:net->named_parameters()) {
    bool head=item.key().rfind("local_emd.2.",0)==0 || item.key().rfind("soft_net.",0)==0;
    TORCH_CHECK(head==item.value().grad().defined(),"SI gradient escaped private Selector head: ",item.key());
    if(head)received+=item.value().grad().abs().sum().item<double>()>0;
  }
  TORCH_CHECK(received&& !xyz.grad().defined()&&!scale.grad().defined()&&!dc.grad().defined()&&!alpha.grad().defined(),"SI touched Master attributes");
  torch::optim::Adam gradient_owner(net->parameters(),torch::optim::AdamOptions(.001));
  for(int repeat=0;repeat<2;++repeat) {
    gradient_owner.zero_grad(true);
    state.accumulateHeadGradients(state.loss(net,p,alpha,1,1),net);
  }
  gradient_owner.zero_grad(true);
  {
    auto owner_saved=bank->rows().clone(),uid_saved=net->privateCompensation()->uids().clone(),inactive_saved=bank->inactiveSlots().clone();
    auto a10=torch::nonzero(counts==10)[0].item<int64_t>(),row=members[a10][0].item<int64_t>();
    auto alpha_reorder=torch::full({n,1},.5,opts);alpha_reorder[row].fill_(.001);bank->inactiveSlots()[row].fill_(true);
    anchor_si::Config cfg;cfg.enabled=true;cfg.minimum_visits=1;anchor_si::State reorder(cfg);
    reorder.observe(net,ones,alpha_reorder,100,1,{.01,.05,.1,.15});auto before=reorder.visits().clone();
    auto permutation=torch::arange(n,owner.options()).flip({0});
    bank->rows().copy_(owner_saved.index_select(0,permutation));net->privateCompensation()->uids().copy_(uid_saved.index_select(0,permutation));
    bank->inactiveSlots().copy_(bank->inactiveSlots().clone().index_select(0,permutation));bank->assertBlocks(n);
    reorder.observe(net,ones,alpha_reorder.index_select(0,permutation),100,2,{.01,.05,.1,.15});
    TORCH_CHECK(torch::equal(reorder.visits(),before+1),"Pure row reorder changed eligible identities or reset SI visits");
    bank->rows().copy_(owner_saved);net->privateCompensation()->uids().copy_(uid_saved);bank->inactiveSlots().copy_(inactive_saved);bank->assertBlocks(n);
  }
  auto old_ema=state.ema().clone();state.sync(net,2);compare("stable_UID_remap",old_ema,state.ema());
  // Same population/counts, changed member identities must invalidate only affected Anchors.
  auto uids=net->privateCompensation()->uids();auto saved=uids.clone();int64_t i=0,j=n-1;
  auto t=uids[i].clone();uids[i].copy_(uids[j]);uids[j].copy_(t);state.sync(net,3);
  auto invalid=(state.visits()==0);TORCH_CHECK(invalid[owner[i]].item<bool>()&&invalid[owner[j]].item<bool>(),"Changed identities retained stale SI");uids.copy_(saved);
  state.save(fs::path(argv[2])/"state.pt");state.invalidate();state.sync(net,4);TORCH_CHECK(!state.visits().count_nonzero().item<int64_t>(),"Geometry invalidation");
  // Actual compositing test: white colors on black => each RGB channel equals sum(alpha*T).
  auto x=torch::tensor({{0.f,0.f,2.f},{.01f,0.f,2.1f}},opts),s=torch::full({2,3},.12,opts),q=torch::tensor({{1.f,0.f,0.f,0.f},{1.f,0.f,0.f,0.f}},opts);
  auto op=torch::full({2,1},.4,opts),colors=torch::ones({2,3},opts),bg=torch::zeros({3},opts),view=torch::eye(4,opts),proj=view.clone(),cam=torch::zeros({3},opts),e=torch::empty({0},opts);
  GaussianRasterizationSettings settings(7,7,.6,.6,bg,1,view,proj,0,cam,false,false,false);
  for(auto tensor:{x,s,q,op,colors})tensor.requires_grad_(true);
  auto render=[&](bool capture){anchor_si::CaptureScope scope(capture);auto xy=torch::zeros_like(x);return rasterizeGaussians(x,xy,e,e,colors,op,s,q,e,settings);};
  auto plain=render(false),captured=render(true);TORCH_CHECK(torch::equal(plain[0],captured[0])&&torch::equal(plain[1],captured[1]),"Capture changed RGB/radii");
  compare("alpha_T_equals_white_RGB",captured[2].sum().reshape({1}),captured[0][0].sum().reshape({1}));TORCH_CHECK(captured[3].numel()==0,"Unused pixel counts allocated");
  TORCH_CHECK(captured[2].sum().item<double>()>0,"Empty analytic render cannot validate contribution");
  auto plain_grad=torch::autograd::grad({plain[0].sum()},{x,s,q,op,colors});
  auto captured_grad=torch::autograd::grad({captured[0].sum()},{x,s,q,op,colors});
  for(size_t k=0;k<plain_grad.size();++k)compare(("render_VJP_"+std::to_string(k)).c_str(),plain_grad[k],captured_grad[k]);
  TORCH_CHECK(!anchor_si::captureEnabled(),"Capture scope leaked");
  std::cout<<"PASS Anchor SI mean/VJP,strides,inactive,budget,age,head-only gradients,UID,geometry,checkpoint,alpha*T RGB parity\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;} }
