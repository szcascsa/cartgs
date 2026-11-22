#include <torch/torch.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <nvToolsExt.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <ATen/Context.h>
#include <fstream>
#include <iostream>
#include <chrono>
#include <numeric>
#include <json/json.h>
#include "improvements/selector/gumbel_network.h"
#include "include/elastic_entry.h"
#include "include/gaussian_renderer.h"
#include "include/transform_field.h"
#include "include/gi_teacher.h"
#include "include/loss_utils.h"
#include "include/constrained_adam.h"

using improvements::selector::GumbelNetwork;
namespace {
torch::Tensor read(const std::string& path) {
  std::ifstream f(path,std::ios::binary);int64_t n=0,w=0;f.read((char*)&n,8);f.read((char*)&w,8);
  TORCH_CHECK(f && n>0 && w>0,"Invalid fixture ",path);auto t=torch::empty({n,w},torch::kFloat32);
  f.read((char*)t.data_ptr<float>(),n*w*4);TORCH_CHECK(f,"Truncated fixture ",path);return t.cuda();
}
void exact(const std::string& name,const torch::Tensor& a,const torch::Tensor& b) {
  TORCH_CHECK(a.defined()==b.defined(),name," gradient presence");
  if(a.defined())TORCH_CHECK(torch::equal(a,b),name," differs; max_abs=",(a-b).abs().max().item<double>());
}
struct Fixture {
  std::string dir;
  std::string model_file;
  torch::Tensor data,metadata,maturity,uniform,eligible;
  GaussianRenderInput full;
  std::vector<std::shared_ptr<GaussianKeyframe>> cameras;
  std::vector<torch::Tensor> gt;
  GaussianPipelineParams pipe;
  torch::Tensor background,override_color;
  TransformField tf{nullptr};
  float entry=0;
  Fixture(const std::string& path):dir(path) {
    model_file=dir+"/current_checkpoint/selector.pt";
    data=read(dir+"/current_input.bin");metadata=read(dir+"/current_metadata.bin");
    maturity=metadata.select(1,0).contiguous();uniform=metadata.select(1,2).contiguous();
    std::ifstream(dir+"/current_spec.txt")>>entry;
    eligible=torch::nonzero(maturity>=entry).flatten();
    const int n=data.size(0);
    full={data.slice(1,0,3).contiguous(),data.slice(1,10,11).sigmoid(),data.slice(1,7,10).exp(),
      torch::nn::functional::normalize(data.slice(1,3,7),torch::nn::functional::NormalizeFuncOptions().dim(1)),
      data.slice(1,11,14).contiguous().reshape({n,1,3}),
      data.slice(1,14).contiguous().reshape({n,3,-1}).transpose(1,2).contiguous(),3,3};
    background=torch::zeros({3},data.options());
    for(int i=0;i<3;++i) {
      auto c=std::make_shared<GaussianKeyframe>(i);float values[37];
      std::ifstream f(dir+"/current_camera_"+std::to_string(i)+".bin",std::ios::binary);f.read((char*)values,sizeof(values));TORCH_CHECK(f,"camera fixture");
      c->world_view_transform_=torch::from_blob(values,{4,4}).clone().cuda();
      c->full_proj_transform_=torch::from_blob(values+16,{4,4}).clone().cuda();
      c->camera_center_=torch::from_blob(values+32,{3}).clone().cuda();c->FoVx_=values[35];c->FoVy_=values[36];cameras.push_back(c);
      gt.push_back(read(dir+"/current_gt_"+std::to_string(i)+".bin").reshape({3,240,320}));
    }
    Json::Value state;std::ifstream(dir+"/current_checkpoint/transform_state.json")>>state;
    if(state["initialized"].asBool()) {
      TransformFieldConfig cfg;cfg.backend="cuda";cfg.hidden_dim=state["hidden_dim"].asInt();
      cfg.hexplane.feature_dim=state["feature_dim"].asInt();cfg.hexplane.spatial_resolution=state["spatial_resolution"].asInt();
      cfg.hexplane.ratio_resolution=state["ratio_resolution"].asInt();cfg.hexplane.multires.clear();
      for(auto v:state["multires"])cfg.hexplane.multires.push_back(v.asInt());
      cfg.elastic_entry_maturity=entry;cfg.maturity_compute_epsilon=state.get("maturity_compute_epsilon",0).asFloat();
      std::vector<float> lo,hi;for(auto v:state["aabb_min"])lo.push_back(v.asFloat());for(auto v:state["aabb_max"])hi.push_back(v.asFloat());
      tf=TransformField(torch::tensor(lo),torch::tensor(hi),cfg);torch::serialize::InputArchive archive;
      archive.load_from(dir+"/current_checkpoint/transform_field.pt",torch::kCPU);tf->load(archive);tf->to(torch::kCUDA);
      for(auto p:tf->parameters())p.requires_grad_(false);
    }
  }
  GumbelNetwork model(const std::string& backend) {
    GumbelNetwork m;torch::serialize::InputArchive archive;archive.load_from(model_file,torch::kCPU);
    m->load(archive);m->to(torch::kCUDA);m->setBackend(backend);return m;
  }
  torch::Tensor probability(GumbelNetwork& m,float ratio) {
    return elasticSelectorProbability(maturity,maturity,entry,ratio,eligible,[&](const torch::Tensor& ids){
      return m->forwardProbability(ids.defined()?full.xyz.index_select(0,ids):full.xyz,
          ids.defined()?full.rotation.index_select(0,ids):full.rotation,
          ids.defined()?full.scaling.index_select(0,ids):full.scaling,ratio,1.0);
    });
  }
  GaussianRenderInput compact(const torch::Tensor& mask,float ratio,bool transform) {
    auto ids=torch::nonzero(mask.detach()>0).flatten();auto out=full.indexSelect(ids);
    if(transform && tf && ids.numel()) {
      torch::NoGradGuard ng;auto m=maturity.index_select(0,ids);auto local=torch::nonzero(tf->candidateMask(out.xyz,m)).flatten();
      if(local.numel()) {
        auto t=tf->applyResidual(out.xyz.index_select(0,local),out.scaling.index_select(0,local),out.rotation.index_select(0,local),ratio,
            elasticMaturity(m.index_select(0,local),entry));
        out.xyz=out.xyz.index_copy(0,local,t.xyz);out.scaling=out.scaling.index_copy(0,local,t.scaling);out.rotation=out.rotation.index_copy(0,local,t.rotation);
      }
    }
    out.opacity=out.opacity*mask.index_select(0,ids).unsqueeze(1);return out;
  }
  torch::Tensor image(const GaussianRenderInput& input,int frame) {
    return GaussianRenderer::render(cameras[frame],240,320,input,pipe,background,override_color).image;
  }
};
void states(GumbelNetwork& a,GumbelNetwork& b,torch::optim::Adam& oa,torch::optim::Adam& ob) {
  auto pa=a->parameters(),pb=b->parameters();for(size_t i=0;i<pa.size();++i) {
    exact("parameter",pa[i],pb[i]);exact("gradient",pa[i].grad(),pb[i].grad());
    const auto& sa=static_cast<const torch::optim::AdamParamState&>(*oa.state().at(pa[i].unsafeGetTensorImpl()));
    const auto& sb=static_cast<const torch::optim::AdamParamState&>(*ob.state().at(pb[i].unsafeGetTensorImpl()));
    TORCH_CHECK(sa.step()==sb.step(),"Adam step differs");exact("m",sa.exp_avg(),sb.exp_avg());exact("v",sa.exp_avg_sq(),sb.exp_avg_sq());
  }
}
void write(const std::string& path,torch::Tensor t) {
  t=t.detach().cpu().contiguous();std::ofstream f(path,std::ios::binary);int64_t n=t.numel();f.write((char*)&n,8);f.write((char*)t.data_ptr<float>(),n*4);
}
std::vector<torch::Tensor> gaussianLeaves(const torch::Tensor& d) {
  const int n=d.size(0);std::vector<torch::Tensor> r{d.slice(1,0,3),d.slice(1,11,14).reshape({n,1,3}),
    d.slice(1,14).reshape({n,3,-1}).transpose(1,2),d.slice(1,10,11),d.slice(1,7,10),d.slice(1,3,7)};
  for(auto& p:r)p=p.detach().contiguous().clone().set_requires_grad(true);return r;
}
GaussianRenderInput activatedGaussian(const std::vector<torch::Tensor>& r) {
  return {r[0],r[3].sigmoid(),r[4].exp(),
    torch::nn::functional::normalize(r[5],torch::nn::functional::NormalizeFuncOptions().dim(1)),r[1],r[2],3,3};
}
void replay(Fixture& f,int steps) {
  auto a=f.model("legacy"),b=f.model("cuda");
  torch::optim::Adam oa(a->parameters(),torch::optim::AdamOptions(.001).eps(1e-15));
  torch::optim::Adam ob(b->parameters(),torch::optim::AdamOptions(.001).eps(1e-15));
  torch::Tensor importance;
  {torch::NoGradGuard ng;importance=GaussianRenderer::render(f.cameras[0],240,320,f.full,f.pipe,f.background,f.override_color,1,false,true,RasterizeMode::Inference).frame_importance;}
  std::ofstream record(f.dir+"/fixed_map_replay.csv");record<<"step,ratio,frame,N,selected,loss,all_exact\n";
  std::ofstream selected_trace(f.dir+"/fixed_map_selected_ids.bin",std::ios::binary);
  for(int step=1;step<=steps;++step) {
    const float ratio=std::vector<float>{.01f,.05f,.1f,.15f}[(step-1)%4];const int frame=(step-1)%3;
    auto pa=f.probability(a,ratio),pb=f.probability(b,ratio);exact("current probability",pa,pb);
    auto ha=sampleSelectorST(pa,f.uniform),hb=sampleSelectorST(pb,f.uniform);exact("current hard",ha,hb);
    auto ids=torch::nonzero(ha.detach()>0).flatten();exact("current compact",ids,torch::nonzero(hb.detach()>0).flatten());
    auto cpu_ids=ids.cpu().contiguous();int64_t trace_step=step,trace_count=ids.numel();
    selected_trace.write((char*)&trace_step,8);selected_trace.write((char*)&trace_count,8);
    selected_trace.write((char*)cpu_ids.data_ptr<int64_t>(),trace_count*8);
    auto compact=f.compact(ha,ratio,true);
    // One real renderer VJP is shared by both network graphs. This removes
    // rasterizer atomic-reduction noise from the controlled backend experiment.
    auto alpha=compact.opacity.detach().clone().set_requires_grad(true);compact.opacity=alpha;
    auto img=f.image(compact,frame);
    auto photo=.8*(img-f.gt[frame]).abs().mean()+.2*(1-loss_utils::fast_ssim(img,f.gt[frame]));
    auto upstream=torch::autograd::grad({photo},{alpha})[0].detach();
    const auto teacher=GITeacherBuilder().build(importance,torch::isfinite(importance),importance>0,f.maturity,ratio,1,1,f.entry);
    auto loss=[&](const torch::Tensor& p,const torch::Tensor& h){return
        (f.full.opacity.index_select(0,ids)*h.index_select(0,ids).unsqueeze(1)*upstream).sum()+
        .01*selectorRatioLoss(p,ratio)+teacher.loss(h,false);};
    auto la=loss(pa,ha),lb=loss(pb,hb);exact("fixed-map loss",la,lb);la.backward();lb.backward();
    auto ap=a->parameters(),bp=b->parameters();for(size_t i=0;i<ap.size();++i)exact("pre-step gradient",ap[i].grad(),bp[i].grad());
    oa.step();tf_cuda::tfCudaAdamDirect(ob,{0});states(a,b,oa,ob);oa.zero_grad(true);ob.zero_grad(true);
    record<<step<<','<<ratio<<','<<frame<<','<<f.data.size(0)<<','<<ids.numel()<<','<<photo.item<double>()<<",1\n";record.flush();
    if(step==1||step==10||step==100||step==1000)std::cout<<"fixed-map step="<<step<<" exact=1\n"<<std::flush;
  }
  {torch::NoGradGuard ng;for(float ratio:{.01f,.05f,.1f,.15f}) {
    auto ha=sampleSelectorST(f.probability(a,ratio),f.uniform),hb=sampleSelectorST(f.probability(b,ratio),f.uniform);
    exact("final selected IDs",ha,hb);
    for(int frame=0;frame<3;++frame) {
      auto ia=f.image(f.compact(ha,ratio,true),frame),ib=f.image(f.compact(hb,ratio,true),frame);exact("final image",ia,ib);
      const auto stem=f.dir+"/fixed_map_ratio"+std::to_string(ratio)+"_frame"+std::to_string(frame);
      write(stem+"_legacy.bin",ia);write(stem+"_cuda.bin",ib);write(stem+"_gt.bin",f.gt[frame]);
    }
  }}
  std::cout<<"fixed-map selector-only replay complete; frozen Gaussian/TF, shared real renderer VJP\n";
}
void benchmark(Fixture& f,const std::string& filename="caller_benchmark.csv") {
  std::ofstream out(f.dir+"/"+filename);out<<"backend,stage,metric,mean_ms,p50_ms,p95_ms\n";
  std::ofstream memory(f.dir+"/"+filename+".memory.csv");memory<<"backend,M,stage,allocated_before,allocated_peak,incremental_peak,reserved_peak\n";
  auto fixed_logits=torch::randn({f.data.size(0),2},f.data.options()).set_requires_grad(true);
  auto generator=at::globalContext().defaultGenerator(torch::Device(torch::kCUDA));auto gumbel_rng=generator.get_state();
  const auto frozen_full=f.full;
  for(const std::string backend:{"legacy","cuda"}) {
    f.full=frozen_full;auto raw=gaussianLeaves(f.data);
    auto m=f.model(backend);torch::optim::Adam optimizer(m->parameters(),torch::optim::AdamOptions(.001).eps(1e-15));
    std::vector<torch::Tensor> initial;for(auto p:m->parameters())initial.push_back(p.detach().clone());
    auto fixed_probability=f.probability(m,.05f).detach();
    for(const std::string stage:{"current_caller_structural","adam","guard_preview_commit","guard_preview_discard","selection_mask_gather","legacy_gumbel_st"}) {
      if(stage=="legacy_gumbel_st")generator.set_state(gumbel_rng);
      const auto before=c10::cuda::CUDACachingAllocator::getDeviceStats(0).allocated_bytes[0].current;
      c10::cuda::CUDACachingAllocator::resetPeakStats(0);
      cudaEvent_t a,b;cudaEventCreate(&a);cudaEventCreate(&b);std::vector<double> event,wall;
      for(int i=0;i<600;++i) {
        if(stage=="adam"||stage.find("guard_preview")==0)for(auto p:m->parameters())p.mutable_grad()=torch::full_like(p,1e-4);
        cudaDeviceSynchronize();auto start=std::chrono::steady_clock::now();cudaEventRecord(a,c10::cuda::getCurrentCUDAStream());
        if(stage=="current_caller_structural") {
          f.full=activatedGaussian(raw);
          auto p=f.probability(m,.05f),h=sampleSelectorST(p,f.uniform);auto compact=f.compact(h,.05f,false);
          auto loss=compact.opacity.mean()+.01*selectorRatioLoss(p,.05f);loss.backward();
          for(int j:{0,3,4,5})TORCH_CHECK(raw[j].grad().defined(),"Caller must retain Gaussian input and opacity gradients");
        }
        if(stage=="legacy_gumbel_st") {
          auto noise=-torch::empty_like(fixed_logits).exponential_().log();auto soft=torch::softmax((fixed_logits+noise)/1.0,1);
          auto hard=torch::zeros_like(soft).scatter_(1,soft.argmax(1,true),1);auto st=hard-soft.detach()+soft;
        }
        else if(stage=="selection_mask_gather") {auto h=sampleSelectorST(fixed_probability,f.uniform);auto compact=f.compact(h,.05f,false);}
        else if(stage.find("guard_preview")==0) {
          AdamStepPreview preview(optimizer,backend=="cuda"?std::vector<std::size_t>{0}:std::vector<std::size_t>{});auto candidate=preview.candidate(preview.delta(),1);
          if(stage=="guard_preview_commit")preview.commit(candidate);
        } else if(backend=="cuda")tf_cuda::tfCudaAdamDirect(optimizer,{0});else optimizer.step();
        if(stage!="legacy_gumbel_st")optimizer.zero_grad(true);
        for(auto& p:raw)p.mutable_grad()=torch::Tensor();
        cudaEventRecord(b,c10::cuda::getCurrentCUDAStream());cudaEventSynchronize(b);
        if(i>=100){float ms=0;cudaEventElapsedTime(&ms,a,b);event.push_back(ms);wall.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());}
        {torch::NoGradGuard ng;auto p=m->parameters();for(size_t j=0;j<p.size();++j)p[j].copy_(initial[j]);}
      }
      cudaEventDestroy(a);cudaEventDestroy(b);
      for(auto sample:std::vector<std::pair<std::string,std::vector<double>>>{{"event",event},{"wall",wall}}) {
        auto v=sample.second;auto mean=std::accumulate(v.begin(),v.end(),0.)/v.size();std::sort(v.begin(),v.end());
        out<<backend<<','<<stage<<','<<sample.first<<','<<mean<<','<<v[250]<<','<<v[474]<<'\n';out.flush();
      }
      const auto stats=c10::cuda::CUDACachingAllocator::getDeviceStats(0);
      memory<<backend<<','<<f.data.size(0)<<','<<stage<<','<<before<<','<<stats.allocated_bytes[0].peak<<','
            <<stats.allocated_bytes[0].peak-before<<','<<stats.reserved_bytes[0].peak<<'\n';memory.flush();
    }
  }
}
void benchmarkGrid(Fixture& f,const std::string& fixture,const std::string& checkpoint) {
  auto data=read(fixture+"/input.bin"),meta=read(fixture+"/metadata.bin");int64_t mapping=0;std::ifstream(fixture+"/spec.txt")>>mapping;
  data=data.index_select(0,torch::nonzero((mapping-meta.select(1,0)>=100)&(meta.select(1,1)>=80)&(meta.select(1,2)>=4)).flatten());
  f.model_file=checkpoint;f.entry=0;torch::manual_seed(123);auto uniform=torch::rand({data.size(0)},data.options());
  for(int n:{1000,5000,10000,20000,50000,100000,200000,219969}) {
    TORCH_CHECK(n<=data.size(0),"Caller grid requires the real audit population");f.data=data.slice(0,0,n);
    f.maturity=torch::ones({n},data.options());f.uniform=uniform.slice(0,0,n);f.eligible=torch::arange(n,data.options().dtype(torch::kInt64));
    f.full={f.data.slice(1,0,3).contiguous(),f.data.slice(1,10,11).sigmoid(),f.data.slice(1,7,10).exp(),
      torch::nn::functional::normalize(f.data.slice(1,3,7),torch::nn::functional::NormalizeFuncOptions().dim(1)),
      f.data.slice(1,11,14).contiguous().reshape({n,1,3}),f.data.slice(1,14).contiguous().reshape({n,3,-1}).transpose(1,2).contiguous(),3,3};
    benchmark(f,"grid_caller_"+std::to_string(n)+".csv");std::cout<<"caller grid M="<<n<<" complete\n"<<std::flush;
  }
}
void profile(Fixture& f,const std::string& backend) {
  auto m=f.model(backend);torch::optim::Adam optimizer(m->parameters(),torch::optim::AdamOptions(.001).eps(1e-15));
  auto raw=gaussianLeaves(f.data);
  for(int i=0;i<110;++i) {
    if(i==100){cudaDeviceSynchronize();cudaProfilerStart();}
    nvtxRangePushA("selector_caller");
    nvtxRangePushA("caller_preparation");f.full=activatedGaussian(raw);nvtxRangePop();
    nvtxRangePushA("probability_network");auto p=f.probability(m,.05f);nvtxRangePop();
    nvtxRangePushA("selection_mask_gather");auto h=sampleSelectorST(p,f.uniform);auto compact=f.compact(h,.05f,false);nvtxRangePop();
    nvtxRangePushA("loss");auto loss=compact.opacity.mean()+.01*selectorRatioLoss(p,.05f);nvtxRangePop();
    nvtxRangePushA("caller_backward");loss.backward();nvtxRangePop();
    nvtxRangePushA("adam");if(backend=="cuda")tf_cuda::tfCudaAdamDirect(optimizer,{0});else optimizer.step();optimizer.zero_grad(true);nvtxRangePop();
    for(auto& p:raw)p.mutable_grad()=torch::Tensor();
    nvtxRangePop();
  }
  cudaDeviceSynchronize();cudaProfilerStop();
}
// Deterministic joint replay: reuse only the unchanged renderer/TF VJP to
// isolate their atomic-reduction noise. Selector-to-Gaussian VJPs are computed
// independently, and both complete Gaussian/Selector/TF optimizers advance.
void jointReplay(Fixture& f,int steps) {
  auto a=f.model("legacy"),b=f.model("cuda");
  auto make_raw=[&](){const auto& d=f.data;const int n=d.size(0);std::vector<torch::Tensor> r{
    d.slice(1,0,3),d.slice(1,11,14).reshape({n,1,3}),d.slice(1,14).reshape({n,3,-1}).transpose(1,2),
    d.slice(1,10,11),d.slice(1,7,10),d.slice(1,3,7)};
    for(auto& p:r)p=p.detach().contiguous().clone().set_requires_grad(true);return r;};
  auto ra=make_raw(),rb=make_raw();
  TransformField ta=f.tf,tb(ta->cloneFrozen());
  for(auto field:{ta,tb})for(auto p:field->parameters())p.requires_grad_(true);
  auto make_optimizer=[&](std::vector<torch::Tensor>& raw,GumbelNetwork& net,TransformField& tf){
    auto o=std::make_unique<torch::optim::Adam>(raw,torch::optim::AdamOptions(1e-5).eps(1e-15));
    o->add_param_group(net->parameters());o->param_groups()[1].options().set_lr(.001);
    o->add_param_group(tf->parameters());o->param_groups()[2].options().set_lr(1e-5);return o;};
  auto oa=make_optimizer(ra,a,ta),ob=make_optimizer(rb,b,tb);
  auto activated=[&](const std::vector<torch::Tensor>& r){return GaussianRenderInput{r[0],r[3].sigmoid(),r[4].exp(),
    torch::nn::functional::normalize(r[5],torch::nn::functional::NormalizeFuncOptions().dim(1)),r[1],r[2],3,3};};
  auto compact=[&](GaussianRenderInput full,const torch::Tensor& hard,float ratio,TransformField& tf){
    auto ids=torch::nonzero(hard.detach()>0).flatten();auto out=full.indexSelect(ids);
    auto maturity=f.maturity.index_select(0,ids),local=torch::nonzero(tf->candidateMask(out.xyz,maturity)).flatten();
    if(local.numel()){
      auto t=tf->applyResidual(out.xyz.index_select(0,local),out.scaling.index_select(0,local),out.rotation.index_select(0,local),ratio,
                              elasticMaturity(maturity.index_select(0,local),f.entry));
      out.xyz=out.xyz.index_copy(0,local,t.xyz);out.scaling=out.scaling.index_copy(0,local,t.scaling);out.rotation=out.rotation.index_copy(0,local,t.rotation);
    }
    out.opacity=out.opacity*hard.index_select(0,ids).unsqueeze(1);return out;
  };
  auto all=[&](const std::vector<torch::Tensor>& raw,GumbelNetwork& net,TransformField& tf){
    auto p=raw,n=net->parameters(),t=tf->parameters();p.insert(p.end(),n.begin(),n.end());p.insert(p.end(),t.begin(),t.end());return p;};
  auto pa=all(ra,a,ta),pb=all(rb,b,tb);
  auto compare_all=[&](){for(size_t i=0;i<pa.size();++i){
    exact("joint parameter",pa[i],pb[i]);exact("joint gradient",pa[i].grad(),pb[i].grad());
    auto x=oa->state().find(pa[i].unsafeGetTensorImpl()),y=ob->state().find(pb[i].unsafeGetTensorImpl());
    TORCH_CHECK((x==oa->state().end())==(y==ob->state().end()),"joint Adam presence");if(x==oa->state().end())continue;
    const auto& sx=static_cast<const torch::optim::AdamParamState&>(*x->second);const auto& sy=static_cast<const torch::optim::AdamParamState&>(*y->second);
    TORCH_CHECK(sx.step()==sy.step(),"joint Adam step");exact("joint m",sx.exp_avg(),sy.exp_avg());exact("joint v",sx.exp_avg_sq(),sy.exp_avg_sq());
  }};
  std::ofstream record(f.dir+"/joint_fixed_map_replay.csv");record<<"step,ratio,frame,N,selected,photo_loss,all_exact\n";
  std::ofstream trace(f.dir+"/joint_fixed_map_selected_ids.bin",std::ios::binary);
  for(int step=1;step<=steps;++step){
    const float ratio=std::vector<float>{.01f,.05f,.1f,.15f}[(step-1)%4];const int frame=(step-1)%3;
    auto fa=activated(ra),fb=activated(rb);f.full=fa;auto p=f.probability(a,ratio);f.full=fb;auto q=f.probability(b,ratio);exact("joint probability",p,q);
    auto ha=sampleSelectorST(p,f.uniform),hb=sampleSelectorST(q,f.uniform);exact("joint ST",ha,hb);
    auto ids=torch::nonzero(ha.detach()>0).flatten();exact("joint IDs",ids,torch::nonzero(hb.detach()>0).flatten());
    auto cpu=ids.cpu().contiguous();int64_t s=step,count=ids.numel();trace.write((char*)&s,8);trace.write((char*)&count,8);trace.write((char*)cpu.data_ptr<int64_t>(),count*8);
    auto input=compact(fa,ha.detach(),ratio,ta);auto image=f.image(input,frame);
    auto photo=.8*(image-f.gt[frame]).abs().mean()+.2*(1-loss_utils::fast_ssim(image,f.gt[frame]));
    auto external_targets=ra,t=ta->parameters();external_targets.insert(external_targets.end(),t.begin(),t.end());external_targets.push_back(input.opacity);
    auto external=torch::autograd::grad({photo},external_targets,{},true,false,true);
    torch::Tensor importance;{torch::NoGradGuard ng;importance=GaussianRenderer::render(f.cameras[frame],240,320,fa,f.pipe,f.background,f.override_color,1,false,true,RasterizeMode::Inference).frame_importance;}
    auto teacher=GITeacherBuilder().build(importance,torch::isfinite(importance),importance>0,f.maturity,ratio,1,1,f.entry);
    auto loss=[&](const torch::Tensor& probability,const torch::Tensor& hard,const GaussianRenderInput& full){return
      (full.opacity.index_select(0,ids).detach()*hard.index_select(0,ids).unsqueeze(1)*external.back().detach()).sum()+
      .01*selectorRatioLoss(probability,ratio)+teacher.loss(hard,false);};
    auto selector_a=ra,selector_b=rb;auto sa=a->parameters(),sb=b->parameters();selector_a.insert(selector_a.end(),sa.begin(),sa.end());selector_b.insert(selector_b.end(),sb.begin(),sb.end());
    auto ga=torch::autograd::grad({loss(p,ha,fa)},selector_a,{},false,false,true);
    auto gb=torch::autograd::grad({loss(q,hb,fb)},selector_b,{},false,false,true);
    for(size_t i=0;i<pa.size();++i){torch::Tensor da,db;
      if(i<6){da=ga[i];db=gb[i];if(external[i].defined()){da=da.defined()?da+external[i]:external[i];db=db.defined()?db+external[i]:external[i];}}
      else if(i<24){da=ga[i];db=gb[i];}else da=db=external[i-18];
      exact("joint independently computed Selector/input VJP",da,db);
      pa[i].mutable_grad()=da.defined()?da.detach().clone():torch::Tensor();
      pb[i].mutable_grad()=db.defined()?db.detach().clone():torch::Tensor();
    }
    oa->step();tf_cuda::tfCudaAdamMixedStep(*ob,{1,2});compare_all();oa->zero_grad(true);ob->zero_grad(true);
    record<<step<<','<<ratio<<','<<frame<<','<<f.data.size(0)<<','<<ids.numel()<<','<<photo.item<double>()<<",1\n";record.flush();
    if(step==1||step==10||step==100||step==1000)std::cout<<"joint fixed-map step="<<step<<" exact=1\n"<<std::flush;
  }
  {torch::NoGradGuard ng;for(float ratio:{.01f,.05f,.1f,.15f}){
    f.full=activated(ra);auto ha=sampleSelectorST(f.probability(a,ratio),f.uniform);auto fa=f.full;
    f.full=activated(rb);auto hb=sampleSelectorST(f.probability(b,ratio),f.uniform);auto fb=f.full;exact("joint final ST",ha,hb);
    for(int frame=0;frame<3;++frame){auto ia=f.image(compact(fa,ha,ratio,ta),frame),ib=f.image(compact(fb,hb,ratio,tb),frame);exact("joint final image",ia,ib);
      auto stem=f.dir+"/joint_fixed_map_ratio"+std::to_string(ratio)+"_frame"+std::to_string(frame);write(stem+"_legacy.bin",ia);write(stem+"_cuda.bin",ib);write(stem+"_gt.bin",f.gt[frame]);}
  }}
  std::cout<<"joint Gaussian/Selector/TF deterministic VJP replay complete\n";
}
void rendererGradientParity(Fixture& f) {
  std::ofstream out(f.dir+"/renderer_gradient_parity.csv");
  out<<"ratio,frame,comparison,tensor,field,max_abs,rms,p99,worst_global_gaussian_id,unequal_elements,passed\n";
  int failures=0;
  for(float ratio:{.01f,.05f,.1f,.15f})for(int frame=0;frame<3;++frame){
    torch::Tensor importance;{torch::NoGradGuard ng;importance=GaussianRenderer::render(f.cameras[frame],240,320,f.full,f.pipe,f.background,f.override_color,1,false,true,RasterizeMode::Inference).frame_importance;}
    auto teacher=GITeacherBuilder().build(importance,torch::isfinite(importance),importance>0,f.maturity,ratio,1,1,f.entry);
    std::vector<std::string> names;
    auto evaluate=[&](const std::string& backend){
      auto net=f.model(backend);auto raw=gaussianLeaves(f.data);TransformField tf(f.tf->cloneFrozen());
      for(auto p:tf->parameters())p.requires_grad_(true);
      auto full=activatedGaussian(raw);
      auto full_image=f.image(full,frame);auto full_loss=.8*(full_image-f.gt[frame]).abs().mean()+.2*(1-loss_utils::fast_ssim(full_image,f.gt[frame]));
      // Real mode0 accumulates Full first, then LOD into the same .grad slots.
      full_loss.backward();std::vector<torch::Tensor> before_selector{full_image.detach().clone(),full_loss.detach().clone()};
      for(auto p:raw)before_selector.push_back(p.grad().defined()?p.grad().detach().clone():torch::Tensor());
      full=activatedGaussian(raw);auto saved_full=f.full;f.full=full;auto probability=f.probability(net,ratio);f.full=saved_full;
      auto hard=sampleSelectorST(probability,f.uniform),ids=torch::nonzero(hard.detach()>0).flatten();auto compact=full.indexSelect(ids);
      auto maturity=f.maturity.index_select(0,ids),local=torch::nonzero(tf->candidateMask(compact.xyz,maturity)).flatten();
      if(local.numel()){
        auto t=tf->applyResidual(compact.xyz.index_select(0,local),compact.scaling.index_select(0,local),compact.rotation.index_select(0,local),ratio,elasticMaturity(maturity.index_select(0,local),f.entry));
        compact.xyz=compact.xyz.index_copy(0,local,t.xyz);compact.scaling=compact.scaling.index_copy(0,local,t.scaling);compact.rotation=compact.rotation.index_copy(0,local,t.rotation);
      }
      compact.opacity=compact.opacity*hard.index_select(0,ids).unsqueeze(1);auto image=f.image(compact,frame);
      auto loss=.8*(image-f.gt[frame]).abs().mean()+.2*(1-loss_utils::fast_ssim(image,f.gt[frame]))+.01*selectorRatioLoss(probability,ratio)+teacher.loss(hard,false);
      loss.backward();auto params=raw,n=net->parameters(),t=tf->parameters();params.insert(params.end(),n.begin(),n.end());params.insert(params.end(),t.begin(),t.end());
      std::vector<torch::Tensor> result{probability.detach(),hard.detach(),ids};
      for(auto p:params)result.push_back(p.grad().defined()?p.grad().detach().clone():torch::Tensor());
      result.insert(result.end(),before_selector.begin(),before_selector.end());result.push_back(image.detach().clone());result.push_back(loss.detach().clone());
      names={"probability","hard","compact_ids"};
      for(const std::string k:{"xyz","sh_dc","sh_rest","opacity","log_scale","quaternion"})names.push_back("gradient/gaussian/"+k);
      for(const auto& p:net->named_parameters())names.push_back("gradient/selector/"+p.key());
      for(const auto& p:tf->named_parameters())names.push_back("gradient/tf/"+p.key());
      names.push_back("before_selector/full_image");names.push_back("before_selector/full_loss");
      for(const std::string k:{"xyz","sh_dc","sh_rest","opacity","log_scale","quaternion"})names.push_back("before_selector/gradient/gaussian/"+k);
      names.push_back("lod_image");names.push_back("lod_total_loss");return result;
    };
    auto legacy=evaluate("legacy"),repeat=evaluate("legacy"),cuda=evaluate("cuda");
    for(const auto& comparison:std::vector<std::pair<std::string,std::vector<torch::Tensor>>>{{"legacy_repeat",repeat},{"cuda",cuda}}){
      for(size_t i=0;i<legacy.size();++i){
        const auto& a=legacy[i];const auto& b=comparison.second[i];bool passed=a.defined()==b.defined();double maximum=0,rms=0,p99=0;int64_t unequal=0,worst=-1;
        if(a.defined()&&b.defined()&&a.sizes()!=b.sizes()){passed=false;unequal=std::max(a.numel(),b.numel());}
        else if(a.defined()&&b.defined()){
          auto error=(a-b).abs().to(torch::kFloat32).flatten();maximum=error.numel()?error.max().item<double>():0;
          rms=error.numel()?error.square().mean().sqrt().item<double>():0;p99=error.numel()?torch::quantile(error,.99).item<double>():0;
          unequal=(a!=b).sum().item<int64_t>();if(maximum && a.dim()>0 && a.size(0)==f.data.size(0))worst=error.argmax().item<int64_t>()/(a.numel()/a.size(0));
          passed=i==1||i==2?torch::equal(a,b):torch::allclose(a,b,2e-4,2e-6);
        }
        failures+=!passed;out<<ratio<<','<<frame<<','<<comparison.first<<','<<i<<','<<names[i]<<','<<maximum<<','<<rms<<','<<p99<<','<<worst<<','<<unequal<<','<<passed<<'\n';out.flush();
      }
    }
    std::cout<<"independent renderer gradient ratio="<<ratio<<" frame="<<frame<<" failures="<<failures<<'\n'<<std::flush;
  }
  TORCH_CHECK(!failures,"Independent renderer gradient parity failed; initial tolerances unchanged");
}
}  // namespace
int main(int argc,char** argv) {
  try {TORCH_CHECK(argc>=3,"caller_replay FIXTURE benchmark|grid|replay|profile|joint [arguments]");torch::set_num_threads(1);torch::manual_seed(123);
    Fixture f(argv[1]);if(std::string(argv[2])=="benchmark")benchmark(f);
    else if(std::string(argv[2])=="grid"){TORCH_CHECK(argc==5,"grid AUDIT_FIXTURE CHECKPOINT");benchmarkGrid(f,argv[3],argv[4]);}
    else if(std::string(argv[2])=="profile"){TORCH_CHECK(argc==4,"profile requires legacy|cuda");profile(f,argv[3]);}
    else if(std::string(argv[2])=="joint")jointReplay(f,argc>3?std::stoi(argv[3]):1000);
    else if(std::string(argv[2])=="renderer_parity")rendererGradientParity(f);
    else replay(f,argc>3?std::stoi(argv[3]):1000);return 0;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
