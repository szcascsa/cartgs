// Independent full renderer graphs; the old coupled VJP test remains separate.
#define main previous_caller_main
#include "../cuda_selector_implementation_20261015/caller_replay.cpp"
#undef main

namespace {
torch::Tensor flat(const std::vector<torch::Tensor>& tensors) {
  std::vector<torch::Tensor> parts;
  for(auto t:tensors)parts.push_back(t.detach().flatten());
  return torch::cat(parts).cpu();
}
torch::Tensor optimizerField(torch::optim::Adam& optimizer,const std::vector<torch::Tensor>& params,int field) {
  std::vector<torch::Tensor> parts;
  for(auto p:params) {
    auto it=optimizer.state().find(p.unsafeGetTensorImpl());
    if(field==0)parts.push_back(p.grad().defined()?p.grad():torch::zeros_like(p));
    else if(it==optimizer.state().end())parts.push_back(torch::zeros_like(p));
    else {const auto& s=static_cast<const torch::optim::AdamParamState&>(*it->second);
      parts.push_back(field==1?s.exp_avg():s.exp_avg_sq());}
  }
  return flat(parts);
}
void bytes(std::ostream& stream,const torch::Tensor& tensor) {
  stream.write((char*)tensor.data_ptr(),tensor.nbytes());
}
void compare(std::ofstream& out,int step,const std::string& field,const torch::Tensor& a,const torch::Tensor& b) {
  auto x=a.to(torch::kFloat64),y=b.to(torch::kFloat64),d=y-x;
  double norm=x.norm().item<double>(), other=y.norm().item<double>();
  double cosine=norm==0||other==0?(torch::equal(x,y)?1:0):(x*y).sum().item<double>()/(norm*other);
  out<<step<<','<<field<<','<<d.abs().max().item<double>()<<','<<d.square().mean().sqrt().item<double>()
     <<','<<d.norm().item<double>()/std::max(norm,1e-30)<<','<<cosine<<'\n';
}
void independent(Fixture& f,const std::string& backend,const std::string& label,int steps) {
  torch::manual_seed(123);
  auto net=f.model(backend);auto raw=gaussianLeaves(f.data);
  TransformField tf(f.tf->cloneFrozen());for(auto p:tf->parameters())p.requires_grad_(true);
  torch::optim::Adam optimizer(raw,torch::optim::AdamOptions(1e-5).eps(1e-15));
  auto parameters=net->parameters();optimizer.add_param_group(parameters);optimizer.param_groups()[1].options().set_lr(.001);
  optimizer.add_param_group(tf->parameters());optimizer.param_groups()[2].options().set_lr(1e-5);
  const bool reference=label=="legacy";
  const auto ref_path=f.dir+"/independent_reference.bin";
  std::ofstream reference_out;if(reference)reference_out.open(ref_path,std::ios::binary);
  std::ifstream reference_in;if(!reference)reference_in.open(ref_path,std::ios::binary);
  std::ofstream rows(f.dir+"/"+label+"_steps.csv"),parity(f.dir+"/"+label+"_parity.csv");
  rows<<"step,ratio,frame,selected,loss,probability_mean,probability_max,mismatch_ratio,training_ms,selector_forward_ms\n";
  parity<<"step,field,max_abs,rms,relative_l2,cosine\n";
  auto compact=[&](GaussianRenderInput full,const torch::Tensor& hard,float ratio){
    auto ids=torch::nonzero(hard.detach()>0).flatten();auto out=full.indexSelect(ids);
    auto maturity=f.maturity.index_select(0,ids),local=torch::nonzero(tf->candidateMask(out.xyz,maturity)).flatten();
    if(local.numel()) {
      auto t=tf->applyResidual(out.xyz.index_select(0,local),out.scaling.index_select(0,local),out.rotation.index_select(0,local),ratio,
                              elasticMaturity(maturity.index_select(0,local),f.entry));
      out.xyz=out.xyz.index_copy(0,local,t.xyz);out.scaling=out.scaling.index_copy(0,local,t.scaling);out.rotation=out.rotation.index_copy(0,local,t.rotation);
    }
    out.opacity=out.opacity*hard.index_select(0,ids).unsqueeze(1);return out;
  };
  for(int step=1;step<=steps;++step) {
    const float ratio=std::vector<float>{.01f,.05f,.1f,.15f}[(step-1)%4];const int frame=(step-1)%3;
    cudaDeviceSynchronize();auto begin=std::chrono::steady_clock::now();
    auto full=activatedGaussian(raw);
    auto full_image=f.image(full,frame);
    auto full_loss=.8*(full_image-f.gt[frame]).abs().mean()+.2*(1-loss_utils::fast_ssim(full_image,f.gt[frame]));
    full_loss.backward();full=activatedGaussian(raw);f.full=full;
    auto selector_begin=std::chrono::steady_clock::now();auto probability=f.probability(net,ratio);
    cudaDeviceSynchronize();double selector_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-selector_begin).count();
    auto hard=sampleSelectorST(probability,f.uniform);auto input=compact(full,hard,ratio);
    auto image=f.image(input,frame);
    torch::Tensor importance;{torch::NoGradGuard ng;importance=GaussianRenderer::render(f.cameras[frame],240,320,full,f.pipe,f.background,f.override_color,1,false,true,RasterizeMode::Inference).frame_importance;}
    auto teacher=GITeacherBuilder().build(importance,torch::isfinite(importance),importance>0,f.maturity,ratio,1,1,f.entry);
    auto loss=.8*(image-f.gt[frame]).abs().mean()+.2*(1-loss_utils::fast_ssim(image,f.gt[frame]))+
      .01*selectorRatioLoss(probability,ratio)+teacher.loss(hard,false);
    loss.backward();if(backend=="cuda")tf_cuda::tfCudaAdamMixedStep(optimizer,{1,2});else optimizer.step();
    cudaDeviceSynchronize();double ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count();
    // Diagnostics are outside the synchronized training wall measurement.
    auto mask=(hard.detach()>0).to(torch::kCPU).to(torch::kUInt8).contiguous();
    std::vector<torch::Tensor> values{flat(parameters),optimizerField(optimizer,parameters,0),
      optimizerField(optimizer,parameters,1),optimizerField(optimizer,parameters,2)};
    double mismatch=0;
    if(reference){bytes(reference_out,mask);for(auto t:values)bytes(reference_out,t);}
    else {
      auto expected=torch::empty_like(mask);reference_in.read((char*)expected.data_ptr(),expected.nbytes());
      mismatch=(mask!=expected).to(torch::kFloat64).mean().item<double>();
      for(int i=0;i<4;++i){auto v=torch::empty_like(values[i]);reference_in.read((char*)v.data_ptr(),v.nbytes());compare(parity,step,std::vector<std::string>{"parameter","gradient","adam_m","adam_v"}[i],v,values[i]);}
      TORCH_CHECK(reference_in,"Truncated independent reference");
    }
    rows<<std::setprecision(12)<<step<<','<<ratio<<','<<frame<<','<<mask.sum().item<int64_t>()<<','<<loss.item<double>()
      <<','<<probability.mean().item<double>()<<','<<probability.max().item<double>()<<','<<mismatch<<','<<ms<<','<<selector_ms<<'\n';
    optimizer.zero_grad(true);
    if(step%100==0){rows.flush();parity.flush();std::cout<<label<<" step="<<step<<" mismatch="<<mismatch<<std::endl;}
  }
  {torch::NoGradGuard ng;for(float ratio:{.01f,.05f,.1f,.15f}){
    f.full=activatedGaussian(raw);auto hard=sampleSelectorST(f.probability(net,ratio),f.uniform);auto input=compact(f.full,hard,ratio);
    for(int frame=0;frame<3;++frame){const auto stem=f.dir+"/independent_ratio"+std::to_string(ratio)+"_frame"+std::to_string(frame);
      write(stem+"_"+label+".bin",f.image(input,frame));if(reference)write(stem+"_gt.bin",f.gt[frame]);}
  }}
  std::vector<torch::Tensor> final=raw,t=tf->parameters();final.insert(final.end(),t.begin(),t.end());
  for(size_t i=0;i<final.size();++i){auto path=f.dir+"/independent_state_"+std::to_string(i);
    if(reference)write(path+".bin",final[i]);else {std::ifstream file(path+".bin",std::ios::binary);int64_t size;file.read((char*)&size,8);
      auto value=torch::empty({size});file.read((char*)value.data_ptr(),value.nbytes());compare(parity,steps,"final_gaussian_tf_"+std::to_string(i),value,final[i].detach().cpu().flatten());}}
}
}
int main(int argc,char** argv) {
  try {TORCH_CHECK(argc==5,"fixed_map FIXTURE legacy|cuda LABEL STEPS");torch::set_num_threads(1);
    at::globalContext().setAllowTF32CuBLAS(false);Fixture fixture(argv[1]);independent(fixture,argv[2],argv[3],std::stoi(argv[4]));
  }catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}return 0;
}
