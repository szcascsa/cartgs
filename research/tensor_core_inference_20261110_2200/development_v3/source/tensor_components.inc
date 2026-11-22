std::vector<torch::Tensor> deployment_tensor_selector(const torch::Tensor& physical,const torch::Tensor& rows,
    const torch::Tensor& owners,const torch::Tensor& view,const torch::Tensor& noise,
    const std::vector<torch::Tensor>& weights,double ratio,bool local,int64_t schema,
    const torch::Tensor& prefix,const std::vector<torch::Tensor>& packed,int64_t mode,int64_t warps){
  TORCH_CHECK(!torch::GradMode::is_enabled()&&weights.size()==6&&packed.size()==5&&ratio>0&&ratio<=1,"Tensor Selector inputs");
  TORCH_CHECK((mode==1||mode==2)&&(warps==1||warps==2||warps==4),"Tensor mode/block");
  c10::cuda::CUDAGuard guard(physical.device());check(rows,physical,at::kLong);check(owners,physical,at::kLong);check(view,physical);check(noise,physical);
  check(packed[0],physical,mode==1?at::kFloat:at::kHalf);TORCH_CHECK(packed[0].numel()==1024,"Tensor Selector weights");
  Weights w{};for(int j=0;j<6;++j){check(weights[j],physical);w.p[j]=weights[j].data_ptr<float>();}
  auto n=rows.numel();auto scores=torch::empty({n},physical.options()),logits=torch::empty({n,2},physical.options()),status=torch::zeros({1},rows.options().dtype(at::kInt));
  if(n){auto launch=[&](auto mode,auto warps){constexpr int Mode=decltype(mode)::value,Warps=decltype(warps)::value;
    auto stream=c10::cuda::getCurrentCUDAStream();
    if(weights[0].size(0)==19)selectorTensor<Mode,Warps,19><<<(n+Warps*16-1)/(Warps*16),128,0,stream>>>(n,physical.data_ptr<float>(),rows.data_ptr<int64_t>(),owners.data_ptr<int64_t>(),view.data_ptr<float>(),noise.data_ptr<float>(),w,logf(float(ratio)),local,schema==2||schema==3,scores.data_ptr<float>(),logits.data_ptr<float>(),status.data_ptr<int>(),nullptr,prefix.defined()&&prefix.numel()?prefix.data_ptr<float>():nullptr,packed[0].data_ptr(),packed[4].data_ptr());
    else selectorTensor<Mode,Warps,11><<<(n+Warps*16-1)/(Warps*16),128,0,stream>>>(n,physical.data_ptr<float>(),rows.data_ptr<int64_t>(),owners.data_ptr<int64_t>(),view.data_ptr<float>(),noise.data_ptr<float>(),w,logf(float(ratio)),local,false,scores.data_ptr<float>(),logits.data_ptr<float>(),status.data_ptr<int>(),nullptr,prefix.defined()&&prefix.numel()?prefix.data_ptr<float>():nullptr,packed[0].data_ptr(),packed[4].data_ptr());
  };dispatchTensor(mode,warps,launch);}
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {scores,logits,status};
}

std::vector<torch::Tensor> deployment_tensor_compensate(const torch::Tensor& offset,const torch::Tensor& rows,
    const torch::Tensor& owners,const torch::Tensor& reference,const std::vector<torch::Tensor>& tables,
    const std::vector<torch::Tensor>& weights,const std::vector<torch::Tensor>& attributes,
    const torch::Tensor& view,const torch::Tensor& view_weight,const std::vector<double>& residual_weights,
    const std::vector<double>& ratio,bool local,int64_t schema,bool softplus,
    const std::vector<torch::Tensor>& packed,int64_t mode,int64_t warps){
  TORCH_CHECK(!torch::GradMode::is_enabled()&&weights.size()==13&&attributes.size()==6&&packed.size()==5&&ratio.size()==3&&tables.size()==residual_weights.size()&&tables.size()<=6,"Tensor Compensation inputs");
  TORCH_CHECK((mode==1||mode==2)&&(warps==1||warps==2||warps==4),"Tensor mode/block");
  c10::cuda::CUDAGuard guard(offset.device());check(rows,offset,at::kLong);check(owners,offset,at::kLong);check(reference,offset);check(view,offset);
  Weights w{};Tables t{};Controls c{};Master m{};TensorWeights tw{};
  for(int j=0;j<13;++j){check(weights[j],offset);w.p[j]=weights[j].data_ptr<float>();}
  for(int j=1;j<4;++j){check(packed[j],offset,mode==1?at::kFloat:at::kHalf);tw.p[j]=packed[j].data_ptr();}
  t.count=tables.size();for(int j=0;j<t.count;++j){check(tables[j],offset);t.p[j]=tables[j].data_ptr<float>();c.weights[j]=residual_weights[j];}
  for(int j=0;j<3;++j)c.ratio[j]=ratio[j];for(int j=0;j<6;++j){check(attributes[j],offset);m.p[j]=attributes[j].data_ptr<float>();}
  if(local)check(view_weight,offset);m.rest=attributes[5].numel()/std::max(int64_t(1),owners.numel());
  auto n=rows.numel();auto output=torch::empty({n*(17+m.rest)},offset.options()),residual=torch::empty({n,14},offset.options()),status=torch::zeros({1},rows.options().dtype(at::kInt));
  if(n){auto launch=[&](auto mode,auto warps){constexpr int Mode=decltype(mode)::value,Warps=decltype(warps)::value;
    compensationTensor<Mode,Warps,false><<<(n+Warps*16-1)/(Warps*16),Warps*128,0,c10::cuda::getCurrentCUDAStream()>>>(n,offset.data_ptr<float>(),rows.data_ptr<int64_t>(),owners.data_ptr<int64_t>(),reference.data_ptr<float>(),t,c,w,view.data_ptr<float>(),local?view_weight.data_ptr<float>():nullptr,local,schema==2||schema==4,m,softplus,output.data_ptr<float>(),residual.data_ptr<float>(),status.data_ptr<int>(),nullptr,nullptr,tw);
  };dispatchTensor(mode,warps,launch);}
  C10_CUDA_KERNEL_LAUNCH_CHECK();return {output,residual,status};
}
