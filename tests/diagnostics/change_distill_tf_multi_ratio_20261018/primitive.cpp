// 隔离数值/shape 验证，不参与 native timing。
int main(int argc,char** argv) {
  torch::set_num_threads(1);
  Snapshot snapshot(argv[1]); auto field=student(snapshot.tf);
  torch::NoGradGuard guard;
  Json::Value result(Json::arrayValue);
  Json::Value timings(Json::arrayValue);
  auto compare=[&](const torch::Tensor& a,const torch::Tensor& b,const std::string& name,int n,int r) {
    auto d=(a-b).abs();Json::Value row;
    row["name"]=name;row["batch"]=n;row["ratios"]=r;
    row["max_abs"]=d.numel()?d.max().item<double>():0;
    row["mean_abs"]=d.numel()?d.mean().item<double>():0;
    row["bitwise"]=torch::equal(a,b);row["allclose"]=torch::allclose(a,b,2e-4,2e-5);
    result.append(row);TORCH_CHECK(row["allclose"].asBool(),name," B=",n," R=",r);
  };
  for(int n:{0,1,7,5107,8192})for(int count:{1,2,4}) {
    auto state=snapshot.after.select(torch::arange(n,snapshot.mature.options().dtype(torch::kLong)));
    std::vector<float> ratios(snapshot.ratios.begin(),snapshot.ratios.begin()+count);
    auto many=field->applyLogResidualMany(state,ratios);
#ifdef CD_HAS_FORWARD_MANY
    auto residuals=field->forwardMany(state.xyz,ratios);
#else
    std::vector<TransformResidual> residuals;
    for(auto r:ratios)residuals.push_back(field->forward(state.xyz,r));
#endif
    for(int r=0;r<count;++r) {
      auto original=field->applyLogResidual(state,ratios[r]);
      auto old=field->forward(state.xyz,ratios[r]);
      compare(original.xyz,many[r].xyz,"xyz",n,count);
      compare(original.log_scaling,many[r].log_scaling,"log_scale",n,count);
      compare(original.rotation,many[r].rotation,"quaternion",n,count);
      compare(old.delta_xyz,residuals[r].delta_xyz,"residual_xyz",n,count);
      compare(old.delta_log_scale,residuals[r].delta_log_scale,"residual_scale",n,count);
      compare(old.delta_rotation_raw,residuals[r].delta_rotation_raw,"residual_rotation",n,count);
    }
    for(int method=0;method<4;++method) {
      auto call=[&] {
        if(method==0)for(auto r:ratios)field->forward(state.xyz,r);
        if(method==1) {
#ifdef CD_HAS_FORWARD_MANY
          field->forwardMany(state.xyz,ratios);
#else
          for(auto r:ratios)field->forward(state.xyz,r);
#endif
        }
        if(method==2)for(auto r:ratios)field->applyLogResidual(state,r);
        if(method==3)field->applyLogResidualMany(state,ratios);
      };
      for(int i=0;i<3;++i)call();
      cdperf::check(cudaDeviceSynchronize());
      cudaEvent_t start,end;cudaEventCreate(&start);cudaEventCreate(&end);
      cudaEventRecord(start,c10::cuda::getCurrentCUDAStream());auto t=cdperf::Clock::now();
      for(int i=0;i<10;++i)call();
      cudaEventRecord(end,c10::cuda::getCurrentCUDAStream());cudaEventSynchronize(end);
      float ms;cudaEventElapsedTime(&ms,start,end);cudaEventDestroy(start);cudaEventDestroy(end);
      Json::Value row;row["method"]=method;row["batch"]=n;row["ratio_count"]=count;
      row["cpu_us_per_call"]=std::chrono::duration<double,std::micro>(cdperf::Clock::now()-t).count()/10;
      row["cuda_span_us_per_call"]=ms*100;timings.append(row);
    }
    if(n) {
      state.xyz=field->aabbMax().reshape({1,3}).expand({n,3}).clone()+10;
      auto outside=field->applyLogResidualMany(state,ratios);
      for(auto& x:outside) {
        compare(state.xyz,x.xyz,"outside_xyz",n,count);
        compare(state.log_scaling,x.log_scaling,"outside_scale",n,count);
        compare(state.rotation,x.rotation,"outside_rotation",n,count);
      }
    }
  }
  TORCH_CHECK(field->applyLogResidualMany(snapshot.after,{}).empty(),"empty ratios");
  writeJson(argv[2],result);writeJson(std::string(argv[2])+".timing.json",timings);
  std::cout<<"PRIMITIVE PASS "<<result.size()<<std::endl;
}
