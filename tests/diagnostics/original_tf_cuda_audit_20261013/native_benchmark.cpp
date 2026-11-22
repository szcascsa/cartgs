#include <torch/torch.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <nvToolsExt.h>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>
#include "include/transform_field.h"
#include "include/constrained_adam.h"

// Diagnostic only: invokes the unchanged production TF and AdamStepPreview.
struct Event {
  cudaEvent_t e;
  Event() { TORCH_CHECK(cudaEventCreate(&e) == cudaSuccess, "event create"); }
  ~Event() { cudaEventDestroy(e); }
  void record() { TORCH_CHECK(cudaEventRecord(e, c10::cuda::getCurrentCUDAStream()) == cudaSuccess, "event record"); }
};
double elapsed(Event& a, Event& b) {
  float ms; TORCH_CHECK(cudaEventElapsedTime(&ms, a.e, b.e) == cudaSuccess, "event elapsed"); return ms;
}
void emit(std::ofstream& out, const std::string& mode, int64_t n,
          const std::string& stage, std::vector<double> v) {
  double mean = std::accumulate(v.begin(), v.end(), 0.0) / v.size();
  std::sort(v.begin(), v.end());
  out << mode << ',' << n << ',' << stage << ',' << mean << ','
      << v[v.size()/2] << ',' << v[static_cast<size_t>(0.95*(v.size()-1))] << '\n';
  out.flush();
}
int main(int argc, char** argv) {
  TORCH_CHECK(argc == 4 || argc == 5, "native_benchmark CHECKPOINT DIAGNOSTICS INPUT_BIN [isolated|profile]");
  torch::set_num_threads(1);
  std::string checkpoint = argv[1], dest = argv[2];
  std::ifstream spec(dest + "/spec.txt");
  TransformFieldConfig cfg; int levels;
  spec >> cfg.hexplane.spatial_resolution >> cfg.hexplane.ratio_resolution
       >> cfg.hexplane.feature_dim >> cfg.hidden_dim >> levels;
  cfg.hexplane.multires.resize(levels);
  for (auto& m : cfg.hexplane.multires) spec >> m;
  std::vector<float> lo(3), hi(3);
  for (auto& v : lo) spec >> v;
  for (auto& v : hi) spec >> v;
  std::ifstream binary(argv[3], std::ios::binary); int64_t rows;
  binary.read(reinterpret_cast<char*>(&rows), sizeof(rows));
  std::vector<float> raw(rows*10);
  binary.read(reinterpret_cast<char*>(raw.data()), raw.size()*sizeof(float));
  TORCH_CHECK(binary && rows > 0, "invalid representative input");
  auto cpu = torch::from_blob(raw.data(), {rows,10}, torch::kFloat32).clone();
  auto input = cpu.to(torch::kCUDA);
  auto make_field = [&]() {
    TransformField field(torch::tensor(lo), torch::tensor(hi), cfg);
    torch::serialize::InputArchive archive;
    archive.load_from(checkpoint + "/transform_field.pt", torch::Device(torch::kCPU));
    field->load(archive); field->to(torch::kCUDA); return field;
  };
  if(argc==5 && std::string(argv[4])=="profile") {
    auto field=make_field();
    auto x=input.slice(1,0,3).contiguous().detach().set_requires_grad(true);
    auto s=input.slice(1,3,6).contiguous().detach().set_requires_grad(true);
    auto q=input.slice(1,6,10).contiguous().detach().set_requires_grad(true);
    torch::optim::Adam opt(field->mlpParameters(),torch::optim::AdamOptions(0.00016).eps(1e-15));
    opt.add_param_group(field->gridParameters());opt.param_groups()[1].options().set_lr(0.0016);
    for(int i=0;i<110;++i) {
      if(i==100) {
        TORCH_CHECK(cudaDeviceSynchronize()==cudaSuccess,"profile prior sync");
        TORCH_CHECK(cudaProfilerStart()==cudaSuccess,"profile start");
      }
      nvtxRangePushA("iteration");
      nvtxRangePushA("zero_grad");
      opt.zero_grad(true);x.mutable_grad()=torch::Tensor();s.mutable_grad()=torch::Tensor();q.mutable_grad()=torch::Tensor();
      nvtxRangePop();nvtxRangePushA("forward");
      auto y=field->applyResidual(x,s,q,0.05f);
      nvtxRangePop();nvtxRangePushA("loss");
      auto loss=(y.xyz-x.detach()).mean()+(y.scaling/s.detach()).mean()+y.rotation.mean();
      nvtxRangePop();nvtxRangePushA("regularization");
      loss=loss+field->regularizationLoss();
      nvtxRangePop();nvtxRangePushA("backward");
      loss.backward();
      nvtxRangePop();nvtxRangePushA("optimizer");
      opt.step();
      nvtxRangePop();nvtxRangePushA("synchronize");
      TORCH_CHECK(cudaStreamSynchronize(c10::cuda::getCurrentCUDAStream())==cudaSuccess,"profile end sync");
      nvtxRangePop();nvtxRangePop();
    }
    TORCH_CHECK(cudaProfilerStop()==cudaSuccess,"profile stop");
    std::cout<<"native profile N="<<rows<<" warmup=100 capture=10 complete"<<std::endl;
    return 0;
  }
  if(argc==5) {
    std::ofstream csv(dest+"/native_optimizer_isolated.csv");
    csv<<"mode,N,stage,mean_ms,p50_ms,p95_ms\n";
    for(const auto& mode:{std::string("adam"),std::string("preview"),std::string("zero_grad")}) {
      auto field=make_field();
      torch::optim::Adam opt(field->mlpParameters(),torch::optim::AdamOptions(0.00016).eps(1e-15));
      opt.add_param_group(field->gridParameters());opt.param_groups()[1].options().set_lr(0.0016);
      std::vector<torch::Tensor> grads;
      for(auto& p:field->parameters()) {auto g=torch::full_like(p,1e-5);p.mutable_grad()=g;grads.push_back(g);}
      Event a,b;std::vector<double> gpu,wall;
      for(int i=0;i<600;++i) {
        if(mode=="zero_grad") {int j=0;for(auto& p:field->parameters())p.mutable_grad()=grads[j++];}
        TORCH_CHECK(cudaDeviceSynchronize()==cudaSuccess,"isolated prior sync");
        auto start=std::chrono::steady_clock::now();a.record();
        if(mode=="zero_grad")opt.zero_grad(true);
        else if(mode=="preview") {AdamStepPreview preview(opt);preview.commit(preview.candidate(preview.delta(),1.0));}
        else opt.step();
        b.record();TORCH_CHECK(cudaEventSynchronize(b.e)==cudaSuccess,"isolated end sync");
        if(i>=100) {gpu.push_back(elapsed(a,b));wall.push_back(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count());}
      }
      emit(csv,mode,0,"cuda_event",gpu);emit(csv,mode,0,"wall",wall);
    }
    return 0;
  }
  // C++/Python parity archive, before any updates.
  {
    auto field = make_field();
    auto x = input.slice(0,0,std::min(rows,int64_t{257})).slice(1,0,3).contiguous().detach().set_requires_grad(true);
    auto s = input.slice(0,0,x.size(0)).slice(1,3,6).contiguous().detach().set_requires_grad(true);
    auto q = input.slice(0,0,x.size(0)).slice(1,6,10).contiguous().detach().set_requires_grad(true);
    auto y = field->applyResidual(x,s,q,0.05f);
    (y.xyz.square().mean()+y.scaling.square().mean()+y.rotation.square().mean()).backward();
    torch::serialize::OutputArchive a;
    a.write("xyz",y.xyz.detach().cpu()); a.write("scaling",y.scaling.detach().cpu());
    a.write("rotation",y.rotation.detach().cpu()); a.write("xyz_grad",x.grad().cpu());
    a.write("scale_grad",s.grad().cpu()); a.write("rotation_grad",q.grad().cpu());
    int i=0; for(const auto& p:field->parameters()) a.write("grad_"+std::to_string(i++),p.grad().cpu());
    a.save_to(dest+"/native_parity.pt");
    torch::serialize::OutputArchive opt_archive;
    torch::optim::Adam shadow(field->mlpParameters(), torch::optim::AdamOptions(0.00016).eps(1e-15));
    shadow.add_param_group(field->gridParameters());
    torch::load(shadow,checkpoint+"/transform_optimizer.pt");
    std::ofstream states(dest+"/checkpoint_optimizer.txt");
    states << "groups=" << shadow.param_groups().size() << " states=" << shadow.state().size() << '\n';
    for(const auto& item:shadow.state()) {
      const auto& v=static_cast<const torch::optim::AdamParamState&>(*item.second);
      states << "step=" << v.step() << " m_numel=" << v.exp_avg().numel() << " dtype=" << v.exp_avg().scalar_type() << '\n';
    }
  }
  std::ofstream out(dest+"/native_benchmark.csv");
  out << "mode,N,stage,mean_ms,p50_ms,p95_ms\n";
  std::ofstream raw_out(dest+"/native_samples.csv");
  raw_out << "mode,N,iteration,forward_ms,loss_ms,backward_ms,zero_ms,adam_ms,total_ms,wall_ms\n";
  std::ofstream validity(dest+"/native_validity.csv");
  validity<<"mode,N,parameters_outputs_gradients_states_finite\n";
  for(const auto& mode: {std::string("core_adam"),std::string("regularized_adam"),std::string("regularized_preview")}) {
    for(int64_t n: {int64_t{1000},int64_t{5000},int64_t{10000},int64_t{20000},int64_t{50000}}) {
      auto field=make_field();
      auto indices=torch::arange(n,torch::TensorOptions().device(torch::kCUDA).dtype(torch::kLong)).remainder(rows);
      auto values=input.index_select(0,indices);
      auto x=values.slice(1,0,3).contiguous().detach().set_requires_grad(true);
      auto s=values.slice(1,3,6).contiguous().detach().set_requires_grad(true);
      auto q=values.slice(1,6,10).contiguous().detach().set_requires_grad(true);
      torch::optim::Adam opt(field->mlpParameters(),torch::optim::AdamOptions(0.00016).eps(1e-15));
      opt.add_param_group(field->gridParameters()); opt.param_groups()[1].options().set_lr(0.0016);
      // Fixed nonzero VJP exercises the TF backward without timing a renderer.
      std::vector<std::vector<double>> samples(7);
      Event e[7];
      for(int iteration=0;iteration<600;++iteration) {
        auto start=std::chrono::steady_clock::now();
        e[0].record(); opt.zero_grad(true); x.mutable_grad()=torch::Tensor(); s.mutable_grad()=torch::Tensor(); q.mutable_grad()=torch::Tensor(); e[1].record();
        auto y=field->applyResidual(x,s,q,0.05f); e[2].record();
        auto loss=(y.xyz-x.detach()).mean()+(y.scaling/s.detach()).mean()+y.rotation.mean();
        if(mode!="core_adam") loss=loss+field->regularizationLoss(); e[3].record();
        loss.backward(); e[4].record();
        if(mode=="regularized_preview") { AdamStepPreview preview(opt); preview.commit(preview.candidate(preview.delta(),1.0)); }
        else opt.step(); e[5].record();
        TORCH_CHECK(cudaEventSynchronize(e[5].e)==cudaSuccess,"measurement sync");
        double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        if(iteration>=100) {
          double z=elapsed(e[0],e[1]),f=elapsed(e[1],e[2]),l=elapsed(e[2],e[3]),b=elapsed(e[3],e[4]),a=elapsed(e[4],e[5]),t=elapsed(e[0],e[5]);
          std::vector<double> row={f,l,b,z,a,t,wall};
          for(size_t j=0;j<row.size();++j)samples[j].push_back(row[j]);
          raw_out<<mode<<','<<n<<','<<iteration-100<<','<<f<<','<<l<<','<<b<<','<<z<<','<<a<<','<<t<<','<<wall<<'\n';
        }
      }
      std::vector<std::string> stages={"forward","loss_regularization","backward","zero_grad","optimizer","total","wall"};
      {
        torch::NoGradGuard no_grad;
        auto check=field->applyResidual(x,s,q,0.05f);
        std::vector<torch::Tensor> tests={torch::isfinite(check.xyz).all(),torch::isfinite(check.scaling).all(),torch::isfinite(check.rotation).all()};
        for(const auto& p:field->parameters()) {tests.push_back(torch::isfinite(p).all());if(p.grad().defined())tests.push_back(torch::isfinite(p.grad()).all());}
        for(const auto& p:{x,s,q})if(p.grad().defined())tests.push_back(torch::isfinite(p.grad()).all());
        for(const auto& item:opt.state()) {const auto& st=static_cast<const torch::optim::AdamParamState&>(*item.second);tests.push_back(torch::isfinite(st.exp_avg()).all());tests.push_back(torch::isfinite(st.exp_avg_sq()).all());}
        bool finite=torch::stack(tests).all().item<bool>();validity<<mode<<','<<n<<','<<finite<<'\n';validity.flush();TORCH_CHECK(finite,"nonfinite diagnostic run",mode,n);
      }
      for(size_t j=0;j<stages.size();++j)emit(out,mode,n,stages[j],samples[j]);
      std::cout<<mode<<" N="<<n<<" complete"<<std::endl;
    }
  }
}
