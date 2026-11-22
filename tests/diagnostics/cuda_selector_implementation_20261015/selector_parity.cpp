#include <torch/torch.h>
#include <ATen/Context.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "include/model_snapshot.h"
// Diagnostic-only access; the legacy production implementation is the oracle.
#define private public
#include "improvements/selector/gumbel_network.h"
#undef private
#ifdef SELECTOR_CUDA_CANDIDATES
#include "include/selector_cuda.h"
#endif

using improvements::selector::GumbelNetwork;
namespace {
std::ofstream records;
int failures = 0;
std::string candidate = "legacy";
torch::Tensor global_ids;
#ifdef SELECTOR_CUDA_CANDIDATES
selector_cuda::Candidate cudaCandidate() {
  if(candidate=="warp")return selector_cuda::Candidate::Warp;
  if(candidate=="cublas_strict")return selector_cuda::Candidate::CublasStrict;
  return selector_cuda::Candidate::Cublas;
}
#endif
torch::Tensor readBinary(const std::string& path) {
  std::ifstream f(path,std::ios::binary); int64_t n=0,w=0;
  f.read(reinterpret_cast<char*>(&n),8);f.read(reinterpret_cast<char*>(&w),8);
  TORCH_CHECK(f && n>0 && w>0,"Invalid fixture ",path);
  auto t=torch::empty({n,w},torch::kFloat32);
  f.read(reinterpret_cast<char*>(t.data_ptr<float>()),n*w*4);
  TORCH_CHECK(f,"Truncated fixture ",path);return t;
}
void compare(const std::string& name, const torch::Tensor& actual,
             const torch::Tensor& expected, bool exact = false) {
  bool pass = actual.defined() == expected.defined();
  double maximum = 0, rms = 0, p99 = 0;
  int64_t worst = -1, changed = 0;
  if (pass && actual.defined()) {
    pass = actual.sizes() == expected.sizes();
    if (pass && actual.numel()) {
      auto a = actual.detach().to(torch::kCPU).to(torch::kFloat64);
      auto e = expected.detach().to(torch::kCPU).to(torch::kFloat64);
      auto delta = (a - e).abs().reshape({-1});
      maximum = delta.max().item<double>();
      rms = delta.square().mean().sqrt().item<double>();
      p99 = std::get<0>(delta.sort())[int64_t(.99 * (delta.numel() - 1))].item<double>();
      worst = delta.argmax().item<int64_t>();
      if (actual.dim() && actual.size(0) == global_ids.numel())
        worst = global_ids[worst / (actual.numel() / actual.size(0))].item<int64_t>();
      else worst = -1;
      changed = (a != e).sum().item<int64_t>();
      pass = torch::isfinite(a).all().item<bool>() && torch::isfinite(e).all().item<bool>() &&
          (exact ? torch::equal(actual, expected) : torch::allclose(actual, expected, 2e-4, 2e-6));
    }
  }
  records << name << ',' << exact << ',' << maximum << ',' << rms << ',' << p99
          << ',' << worst << ',' << changed << ',' << pass << '\n';
  records.flush();
  if (!pass) { ++failures; std::cerr << "FAIL " << name << " max_abs=" << maximum << '\n'; }
}
std::vector<torch::Tensor> legacyStages(GumbelNetwork& model,
                                       const std::vector<torch::Tensor>& in) {
  std::vector<torch::Tensor> stages, h;
  std::vector<torch::nn::Sequential> branches{model->pos_emd_, model->rotation_emd_,
                                            model->scale_emd_, model->time_emd_};
  for (int b = 0; b < 4; ++b) {
    auto first = branches[b]->ptr<torch::nn::LinearImpl>(0)->forward(in[b]).relu();
    auto second = branches[b]->ptr<torch::nn::LinearImpl>(2)->forward(first).relu();
    stages.push_back(first); stages.push_back(second);
    h.push_back(b == 3 && second.size(0) == 1 ? second.expand({in[0].size(0), 32}) : second);
  }
  stages.push_back(model->soft_net_->forward(torch::cat(h, 1)));
  return stages;
}
torch::Tensor logits(GumbelNetwork& model, const std::vector<torch::Tensor>& in, bool use_candidate) {
#ifdef SELECTOR_CUDA_CANDIDATES
  if (use_candidate && candidate != "legacy")
    return selector_cuda::network(in, model->parameters(), cudaCandidate());
#endif
  return model->logitsFor(in[0], in[1], in[2], in[3]);
}
torch::Tensor st(const torch::Tensor& z, const torch::Tensor& noise, double tau = 1.0) {
  auto soft = ((z + noise) / tau).softmax(-1);
  auto hard = torch::zeros_like(z).scatter_(-1, std::get<1>(soft.max(-1, true)), 1.0);
  return (hard - soft.detach() + soft).select(1, 1);
}
torch::Tensor uniformST(const torch::Tensor& z, const torch::Tensor& uniform) {
  auto d = z.select(1, 1) - z.select(1, 0);
  auto u = uniform.clamp(1e-7, 1.0 - 1e-7);
  auto soft = (d - (u.log() - (1.0-u).log())).sigmoid();
  return (uniform < d.sigmoid()).to(soft.scalar_type()) - soft.detach() + soft;
}
std::vector<torch::Tensor> leaves(const std::vector<torch::Tensor>& in) {
  std::vector<torch::Tensor> out;
  for (auto t : in) out.push_back(t.detach().clone().set_requires_grad(true));
  return out;
}
torch::Tensor objective(const std::string& kind, const torch::Tensor& z,
                        const torch::Tensor& noise, const torch::Tensor& opacity,
                        const torch::Tensor& uniform) {
  if (kind == "vjp") return (z * noise).sum();
  auto p = z.softmax(-1).select(1, 1);
  if (kind == "cd") return (p - .3).abs().mean();
  if (kind == "dsd") return torch::nn::functional::binary_cross_entropy_with_logits(
      z.select(1, 1)-z.select(1, 0), torch::full_like(p, .3));
  auto h = kind == "guard_uniform" ? uniformST(z, uniform) : st(z, noise);
  auto ratio = .01 * (p.mean() - .05).abs();
  if (kind == "ratio") return ratio;
  auto supervised = torch::arange(z.size(0), z.options().dtype(torch::kLong)).remainder(3) == 0;
  auto gi = (h.index({supervised}) - .4).abs().mean();
  if (kind == "gi") return gi;
  auto ids = torch::nonzero(h.detach() > 0).flatten();
  // Structural opacity VJP, not a renderer/TF quality claim. Empty selection
  // remains connected, so autograd still exercises a defined zero gradient.
  auto photo = (h.index_select(0, ids) * opacity.index_select(0, ids)).sum() /
      std::max<int64_t>(1, ids.numel());
  return kind == "combined" ? photo + ratio + gi : photo;
}
void gradients(GumbelNetwork& model, const std::vector<torch::Tensor>& input,
               const torch::Tensor& noise, const std::string& prefix) {
  auto opacity = torch::full({input[0].size(0)}, .7, input[0].options()).set_requires_grad(true);
  auto uniform = torch::rand_like(opacity);
  for (std::string kind : {"vjp", "ratio", "gi", "photo", "combined", "detach", "guard_uniform", "cd", "dsd", "activation_chain"}) {
    auto in = leaves(input);
    if (kind == "activation_chain") in[2]=input[2].log().detach().set_requires_grad(true);
    if (kind == "detach") for (int i=0; i<3; ++i) in[i] = in[i].detach();
    auto args = in;
    args.erase(std::remove_if(args.begin(), args.end(), [](const torch::Tensor& t){return !t.requires_grad();}), args.end());
    args.push_back(opacity);
    auto parameters = model->parameters(); args.insert(args.end(), parameters.begin(), parameters.end());
    auto evaluate=[&](bool use_candidate) {
      auto network_input=in;
      if(kind=="activation_chain") {
        network_input[1]=torch::nn::functional::normalize(in[1],torch::nn::functional::NormalizeFuncOptions().dim(1));
        network_input[2]=in[2].exp();
      }
      return objective(kind,logits(model,network_input,use_candidate),noise,opacity,uniform);
    };
    auto loss = evaluate(false);
    auto a = torch::autograd::grad({loss}, args, {}, false, false, true);
    // Same graph inputs and upstream values, independent forward execution.
    auto repeated = evaluate(false);
    auto repeat = torch::autograd::grad({repeated}, args, {}, false, false, true);
    auto b_loss = evaluate(true);
    auto b = torch::autograd::grad({b_loss}, args, {}, false, false, true);
    compare(prefix+"/"+kind+"/loss", b_loss, loss);
    for (size_t i=0; i<args.size(); ++i) {
      compare(prefix+"/"+kind+"/legacy_repeat/"+std::to_string(i), repeat[i], a[i]);
      compare(prefix+"/"+kind+"/gradient/"+std::to_string(i), b[i], a[i]);
    }
  }
}
void copyParameters(GumbelNetwork& to, GumbelNetwork& from) {
  torch::NoGradGuard ng; auto a=to->parameters(), b=from->parameters();
  for (size_t i=0; i<a.size(); ++i) a[i].copy_(b[i]);
}
void training(GumbelNetwork& source, const std::vector<torch::Tensor>& input, int steps) {
  GumbelNetwork a, b; a->to(input[0].device()); b->to(input[0].device());
  copyParameters(a,source); copyParameters(b,source);
  torch::optim::Adam oa(a->parameters(), torch::optim::AdamOptions(.001));
  torch::optim::Adam ob(b->parameters(), torch::optim::AdamOptions(.001));
  auto pa=a->parameters(), pb=b->parameters();
  auto opacity=torch::full({input[0].size(0)}, .7, input[0].options());
  auto u=torch::rand_like(opacity);
  std::vector<int64_t> expected_steps(pa.size(),0);
  for (int step=1; step<=steps; ++step) {
    const int before_failures=failures;
    auto in=input; in[3]=torch::full_like(input[3], std::vector<float>{.01f,.05f,.1f,.15f}[(step-1)%4]);
    auto noise=-torch::empty({in[0].size(0),2},in[0].options()).exponential_().log();
    auto za=logits(a,in,false), zb=logits(b,in,true);
    const std::string label="step_"+std::to_string(step);
    compare(label+"/hard",st(zb,noise),st(za,noise),true);
    auto la=objective("combined",za,noise,opacity,u), lb=objective("combined",zb,noise,opacity,u);
    compare(label+"/loss",lb,la); compare(label+"/logits",zb,za);
    la.backward();lb.backward();
    // Explicitly distinguish undefined gradient from an allocated zero tensor.
    if (step%7==0) { pa[0].mutable_grad()=torch::Tensor();pb[0].mutable_grad()=torch::Tensor(); }
    if (step%11==0) { pa[1].mutable_grad()=torch::zeros_like(pa[1]);pb[1].mutable_grad()=torch::zeros_like(pb[1]); }
    for(size_t i=0;i<pa.size();++i) {
      compare(label+"/grad/"+std::to_string(i),pb[i].grad(),pa[i].grad());
      expected_steps[i]+=pa[i].grad().defined()?1:0;
    }
    oa.step();ob.step();oa.zero_grad(true);ob.zero_grad(true);
    for(size_t i=0;i<pa.size();++i) {
      const auto name=label+"/parameter_"+std::to_string(i);
      compare(name,pb[i],pa[i]);
      auto sa=oa.state().find(pa[i].unsafeGetTensorImpl()), sb=ob.state().find(pb[i].unsafeGetTensorImpl());
      TORCH_CHECK((sa==oa.state().end())==(sb==ob.state().end()),"Adam state presence");
      if(sa==oa.state().end())continue;
      const auto& ma=static_cast<const torch::optim::AdamParamState&>(*sa->second);
      const auto& mb=static_cast<const torch::optim::AdamParamState&>(*sb->second);
      TORCH_CHECK(ma.step()==expected_steps[i],"Undefined/zero Adam gradient step semantics changed");
      TORCH_CHECK(a->parameters()[i].unsafeGetTensorImpl()==pa[i].unsafeGetTensorImpl() &&
                  b->parameters()[i].unsafeGetTensorImpl()==pb[i].unsafeGetTensorImpl(),"Parameter identity changed");
      compare(name+"/m",mb.exp_avg(),ma.exp_avg());compare(name+"/v",mb.exp_avg_sq(),ma.exp_avg_sq());
      compare(name+"/step",torch::tensor(mb.step()),torch::tensor(ma.step()),true);
    }
    if (failures>before_failures) { std::cerr<<"Training parity stopped at first failing step "<<step<<'\n'; return; }
  }
}
}  // namespace

int main(int argc,char** argv) {
  try {
    std::string out="parity.csv", device="cuda", checkpoint, fixture;
    int steps=100, rows=257;
    for(int i=1;i<argc;i+=2) {
      TORCH_CHECK(i+1<argc,"Missing option value"); std::string key=argv[i],value=argv[i+1];
      if(key=="--out")out=value;else if(key=="--device")device=value;
      else if(key=="--fixture")fixture=value;
      else if(key=="--checkpoint")checkpoint=value;else if(key=="--candidate")candidate=value;
      else if(key=="--steps")steps=std::stoi(value);else if(key=="--rows")rows=std::stoi(value);
      else TORCH_CHECK(false,"Unknown option ",key);
    }
    TORCH_CHECK(candidate=="legacy"||candidate=="cublas"||candidate=="warp"||candidate=="cublas_strict","Invalid candidate");
#ifndef SELECTOR_CUDA_CANDIDATES
    TORCH_CHECK(candidate=="legacy","Build with SELECTOR_CUDA_CANDIDATES for CUDA candidates");
#endif
    TORCH_CHECK(rows>0 && steps>=0,"Invalid test size");
    torch::set_num_threads(1);torch::manual_seed(123);
    records.open(out);TORCH_CHECK(records,"Cannot create output ",out);
    std::cout<<"Harness revision: near_tie+relu_zero+activation_chain+adam_defined_zero\n"<<std::flush;
    records<<"check,exact,max_abs,rms,p99,worst_global_gaussian_id,unequal_elements,passed\n";
    auto options=torch::TensorOptions().dtype(torch::kFloat32).device(torch::Device(device));
    GumbelNetwork model;
    TORCH_CHECK(model->backend()=="legacy","Default Selector backend changed");
    model->setBackend("legacy");
    for(const std::string backend:{"invalid","cuda"}) {
      bool rejected=false;try{model->setBackend(backend);}catch(const c10::Error&){rejected=true;}
      TORCH_CHECK(rejected,"Unaccepted production backend must be rejected");
    }
    if(!checkpoint.empty()){torch::serialize::InputArchive a;a.load_from(checkpoint,torch::kCPU);model->load(a);}
    model->to(torch::Device(device));
    TORCH_CHECK(model->parameters().size()==18,"Parameter tensor identity/count changed");
    int64_t count=0;for(auto p:model->parameters())count+=p.numel();TORCH_CHECK(count==4962,"Network architecture changed");
    torch::Tensor actual_data,actual_ids;
    if(!fixture.empty()) {
      actual_data=readBinary(fixture+"/input.bin");auto metadata=readBinary(fixture+"/metadata.bin");
      int64_t mapping=0;std::ifstream(fixture+"/spec.txt")>>mapping;
      TORCH_CHECK(actual_data.size(0)==metadata.size(0) && metadata.size(1)>=4 && mapping>0,"Fixture metadata mismatch");
      // Frozen audit Mature population is only an input fixture for the MLP;
      // it does not replace current production maturity/elastic-entry logic.
      actual_ids=torch::nonzero((mapping-metadata.select(1,0)>=100)&
          (metadata.select(1,1)>=80)&(metadata.select(1,2)>=4)).flatten();
      TORCH_CHECK(actual_ids.numel()>=rows,"Requested rows exceed frozen Mature population");
    }
    for (int n : {0,1,31,rows}) {
      global_ids=torch::arange(n,torch::kLong)*3+7;
      std::vector<torch::Tensor> input{torch::randn({n,3},options),torch::randn({n,4},options),
        torch::randn({n,3},options).exp(),torch::full({1,1},.05,options)};
      input[1]=torch::nn::functional::normalize(input[1],torch::nn::functional::NormalizeFuncOptions().dim(1));
      if(actual_data.defined()) {
        global_ids=actual_ids.slice(0,0,n);auto data=actual_data.index_select(0,global_ids).to(options.device());
        input[0]=data.slice(1,0,3).contiguous();
        input[1]=torch::nn::functional::normalize(data.slice(1,3,7),torch::nn::functional::NormalizeFuncOptions().dim(1));
        input[2]=data.slice(1,7,10).exp();
      }
      for(bool per_row:{false,true}) {
        if(per_row)input[3]=torch::full({n,1},.05,options);
        const auto label="n"+std::to_string(n)+(per_row?"/row_ratio":"/scalar_ratio");
        auto stages=legacyStages(model,input), b_stages=legacyStages(model,input);
#ifdef SELECTOR_CUDA_CANDIDATES
        if(candidate!="legacy")b_stages=selector_cuda::forward(input,model->parameters(),cudaCandidate());
#endif
        for(int i=0;i<9;++i)compare(label+"/stage_"+std::to_string(i),b_stages[i],stages[i]);
        auto za=logits(model,input,false),zb=logits(model,input,true);
        compare(label+"/logits",zb,za);compare(label+"/probability",zb.softmax(-1),za.softmax(-1));
        if(!n)continue;
        auto noise=-torch::empty_like(za).exponential_().log();
        for(double tau:{.1,1.,3.}) {
          auto ha=st(za,noise,tau),hb=st(zb,noise,tau);
          compare(label+"/soft_tau"+std::to_string(tau),((zb+noise)/tau).softmax(-1),((za+noise)/tau).softmax(-1));
          compare(label+"/hard_tau"+std::to_string(tau),hb,ha,true);
          auto ia=torch::nonzero(ha.detach()>0).flatten(),ib=torch::nonzero(hb.detach()>0).flatten();
          compare(label+"/compact",global_ids.index_select(0,ib.cpu()),global_ids.index_select(0,ia.cpu()),true);
        }
        // An explicit tie fixture catches a changed FP32 accumulation order
        // that allclose plus ordinary random samples can otherwise conceal.
        auto tie_noise=-za.detach();
        compare(label+"/near_tie/hard",st(zb,tie_noise),st(za,tie_noise),true);
        auto tie_a=torch::nonzero(st(za,tie_noise).detach()>0).flatten().cpu();
        auto tie_b=torch::nonzero(st(zb,tie_noise).detach()>0).flatten().cpu();
        compare(label+"/near_tie/compact",global_ids.index_select(0,tie_b),global_ids.index_select(0,tie_a),true);
        compare(label+"/near_tie/count",torch::tensor(tie_b.numel()),torch::tensor(tie_a.numel()),true);
        // Reuse the actual legacy RNG expression. No custom random generator.
        auto generator=at::globalContext().defaultGenerator(input[0].device());
        auto before=generator.get_state().clone();
        auto na=-torch::empty_like(za).exponential_().log();auto next_a=torch::rand({17},options);
        auto after=generator.get_state().clone();generator.set_state(before);
        auto production_z=logits(model,input,true);
        auto nb=-torch::empty_like(production_z).exponential_().log();auto next_b=torch::rand({17},options);
        compare(label+"/rng_noise",nb,na,true);compare(label+"/rng_next",next_b,next_a,true);
        compare(label+"/rng_state",generator.get_state(),after,true);
        compare(label+"/rng_hard",st(production_z,nb),st(za,na),true);
        if(n==rows)gradients(model,input,noise,label);
      }
    }
    // Fixed sequence starts from the same parameters, frames, ratios and noise.
    const int n=std::min(rows,257);global_ids=torch::arange(n,torch::kLong)*3+7;
    std::vector<torch::Tensor> training_input{torch::randn({n,3},options),torch::randn({n,4},options),
      torch::randn({n,3},options).exp(),torch::full({1,1},.05,options)};
    if(actual_data.defined()) {
      global_ids=actual_ids.slice(0,0,n);auto data=actual_data.index_select(0,global_ids).to(options.device());
      training_input[0]=data.slice(1,0,3).contiguous();
      training_input[1]=torch::nn::functional::normalize(data.slice(1,3,7),torch::nn::functional::NormalizeFuncOptions().dim(1));
      training_input[2]=data.slice(1,7,10).exp();
    }
    training(model,training_input,steps);
    // ReLU's exact-zero derivative must agree with threshold_backward.
    GumbelNetwork zero_model;zero_model->to(torch::Device(device));copyParameters(zero_model,model);
    {torch::NoGradGuard ng;auto p=zero_model->parameters();p[0][0].zero_();p[1][0].zero_();}
    global_ids=torch::arange(31,torch::kLong)*3+7;
    gradients(zero_model,{torch::randn({31,3},options),torch::randn({31,4},options),
        torch::randn({31,3},options).exp(),torch::full({1,1},.05,options)},
        torch::randn({31,2},options),"relu_zero");
    std::cout<<"candidate="<<candidate<<" device="<<device<<" steps="<<steps<<" failures="<<failures<<'\n';
    return failures?1:0;
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 2;}
}
