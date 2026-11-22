#pragma once
#include "improvements/selector/gumbel_network.h"
#include "include/mapping_cuda_optimizer.h"
#include "include/residual48_fusion.h"
#include <torch/script.h>
#include <nvToolsExt.h>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <sstream>

using improvements::selector::GumbelNetwork;
inline bool bits(const torch::Tensor& a,const torch::Tensor& b) {
  if(a.defined()!=b.defined())return false;if(!a.defined())return true;
  return a.sizes()==b.sizes()&&torch::equal(a.contiguous().view(torch::kInt32),b.contiguous().view(torch::kInt32));
}
struct Difference {int64_t different=0,total=0;double absolute=0,relative_l2=0;bool equal=true;};
inline Difference difference(const torch::Tensor& a,const torch::Tensor& b) {
  Difference d;d.equal=bits(a,b);if(!a.defined()||!b.defined()) {if(!d.equal)d.different=-1;return d;}
  d.total=a.numel();if(d.equal||!d.total)return d;
  d.different=(a.contiguous().view(torch::kInt32)!=b.contiguous().view(torch::kInt32)).sum().item<int64_t>();
  auto delta=(a-b).to(torch::kFloat64);d.absolute=delta.abs().max().item<double>();
  d.relative_l2=delta.norm().item<double>()/std::max(1e-30,a.to(torch::kFloat64).norm().item<double>());return d;
}
inline void log_difference(std::ofstream& file,const std::string& prefix,const std::string& node,const torch::Tensor& a,const torch::Tensor& b) {
  auto d=difference(a,b);file<<prefix<<','<<node<<','<<d.equal<<','<<d.total<<','<<d.different<<','<<d.absolute<<','<<d.relative_l2<<'\n';
}
inline std::vector<double> weights(double ratio) {
  std::vector<double> knots{.01,.05,.1,.2,.3,.4},w{1.};
  for(int j=1;j<6;++j){if(ratio<=knots[j-1]+1e-8)break;w.push_back(std::min(1.,(ratio-knots[j-1])/(knots[j]-knots[j-1])));}return w;
}
inline torch::Tensor legacy_pack(const torch::Tensor& x,const torch::Tensor& c,const torch::Tensor& r,
    const torch::Tensor& mean,const torch::Tensor& stddev,const std::vector<torch::Tensor>& f,const std::vector<double>& w) {
  auto z=f[0];for(size_t j=1;j<f.size();++j)z=z+w[j]*f[j];
  return torch::cat({((torch::asinh((x-c)/r)-mean.slice(0,3,6))/stddev.slice(0,3,6)).clamp(-15,15),z},1);
}
struct Tap {
  torch::Tensor value;int64_t stride0=0,stride1=0;std::string node;
  void take(const torch::Tensor& g) {stride0=g.stride(0);stride1=g.dim()>1?g.stride(1):0;value=g.detach().clone();}
};
struct Capture {
  torch::Tensor output,loss,input,master_gradient;Tap input_vjp;
  std::vector<torch::Tensor> gather_values,leaf_gradients,network_gradients;
  std::vector<Tap> gather_vjp,scatter_output;
};
struct Scene {
  GumbelNetwork net;torch::jit::Module fixture;std::vector<torch::Tensor> tables;torch::Tensor xyz;
  Scene(const std::string& path,const std::string& checkpoint):net(.1,true,false,true,std::vector<double>{},0),fixture(torch::jit::load(path+"/fixture.pt")) {
    fixture.to(torch::kCUDA);net->to(torch::kCUDA);torch::serialize::InputArchive ar;ar.load_from(checkpoint,torch::Device(torch::kCUDA));net->load(ar);
    net->spatialCompensation()->setRatioCodeCache(true);tables=net->anchors()->compensationFeatureTables();xyz=fixture.attr("xyz").toTensor();
  }
  torch::Tensor ids(double ratio,int j) {
    if(j<6)return fixture.attr("ids_"+std::to_string(j)).toTensor();
    return fixture.attr("ids_5").toTensor().slice(0,0,int64_t(std::ceil(xyz.size(0)*ratio)));
  }
  Capture capture(const torch::Tensor& ids,double ratio,const torch::Tensor& up,bool fused,bool model,bool hooks=true,bool original_api=false) {
    net->zero_grad(true);auto anchor=net->anchors();auto m=net->spatialCompensation();Capture result;
    const auto w=anchor->compensationResidualWeights(ratio);auto owners=anchor->rows().index_select(0,ids);
    auto x=xyz.detach().clone().requires_grad_(true),position=x.index_select(0,ids),center=anchor->centers().index_select(0,owners),reference=anchor->offsetScales().index_select(0,owners);
    std::vector<torch::Tensor> parameters,gathered;for(auto t:tables)parameters.push_back(original_api?t:t.detach().clone().requires_grad_(true));
    result.gather_vjp.resize(w.size());result.scatter_output.resize(w.size());
    for(size_t j=0;j<w.size();++j) {
      gathered.push_back(parameters[j].index_select(0,owners));result.gather_values.push_back(gathered.back().detach());
      if(hooks&&!original_api)gathered.back().register_hook([&result,j](torch::Tensor g){result.gather_vjp[j].take(g);return g;});
    }
    if(hooks)for(size_t j=0;j<w.size();++j)parameters[j].register_hook([&result,j](torch::Tensor g){result.scatter_output[j].take(g);return g;});
    torch::Tensor input,output;
    if(original_api)output=m->forward(position,{},{},{},{},center,reference,ratio,true,anchor->compensationInput(ids,ratio));
    else {
      input=fused?mapping_cuda::residual48TrainingPack(position,center,reference,m->mean,m->stddev,gathered,w):legacy_pack(position,center,reference,m->mean,m->stddev,gathered,w);
      result.input=input.detach();if(hooks)input.register_hook([&result](torch::Tensor g){result.input_vjp.take(g);return g;});
      output=model?m->forward(position,{},{},{},{},center,reference,ratio,true,{},input):input;
    }
    auto loss=(output*up).sum();loss.backward();
    result.output=output.detach();result.loss=loss.detach();result.master_gradient=x.grad().detach().clone();
    for(auto t:parameters)result.leaf_gradients.push_back(t.grad().defined()?t.grad().detach().clone():torch::Tensor());
    for(auto t:m->parameters())result.network_gradients.push_back(t.grad().defined()?t.grad().detach().clone():torch::Tensor());
    return result;
  }
};
inline void compare_capture(std::ofstream& file,const std::string& prefix,const Capture& a,const Capture& b,bool intermediates=true) {
  log_difference(file,prefix,"output",a.output,b.output);log_difference(file,prefix,"loss",a.loss,b.loss);
  log_difference(file,prefix,"Master_XYZ_VJP",a.master_gradient,b.master_gradient);
  if(intermediates){log_difference(file,prefix,"packed_forward",a.input,b.input);log_difference(file,prefix,"packed_input_VJP",a.input_vjp.value,b.input_vjp.value);}
  for(size_t j=0;j<a.leaf_gradients.size();++j) {
    if(intermediates&&j<a.gather_vjp.size()) {
      log_difference(file,prefix,"gather_value_"+std::to_string(j),a.gather_values[j],b.gather_values[j]);
      log_difference(file,prefix,"gather_input_VJP_"+std::to_string(j),a.gather_vjp[j].value,b.gather_vjp[j].value);
      log_difference(file,prefix,"scatter_output_"+std::to_string(j),a.scatter_output[j].value,b.scatter_output[j].value);
    }
    log_difference(file,prefix,"leaf_gradient_"+std::to_string(j),a.leaf_gradients[j],b.leaf_gradients[j]);
  }
  for(size_t j=0;j<a.network_gradients.size();++j)log_difference(file,prefix,"network_gradient_"+std::to_string(j),a.network_gradients[j],b.network_gradients[j]);
}
inline std::string key(int deterministic,int seed,double ratio,const std::string& branch,int rep) {
  std::ostringstream s;s<<deterministic<<','<<seed<<','<<ratio<<','<<branch<<','<<rep;return s.str();
}
