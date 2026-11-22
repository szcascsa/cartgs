#include "include/anchor_selector_cuda.h"
#include "include/spatial_compensation_cuda.h"
#include "improvements/selector/spatial_compensation.h"
#include <torch/script.h>
#include <json/json.h>
#include <filesystem>
#include <iostream>

using T=torch::Tensor;using List=std::vector<T>;
using Field=improvements::selector::SpatialCompensation;
int64_t bad(const T& t){return t.defined()?(~torch::isfinite(t)).sum().item<int64_t>():0;}
void compare(const std::string& name,const T& a,const T& b,double atol=2e-5,double rtol=2e-4){
  TORCH_CHECK(a.defined()==b.defined(),name," grad=None");if(!a.defined())return;
  TORCH_CHECK(a.sizes()==b.sizes()&&a.scalar_type()==b.scalar_type()&&a.device()==b.device(),name," layout");
  TORCH_CHECK(!bad(a)&&!bad(b),name," nonfinite");if(!a.numel())return;
  auto diff=(a.to(torch::kFloat64)-b.to(torch::kFloat64)).abs();auto failures=diff>atol+rtol*b.to(torch::kFloat64).abs();
  Json::Value row;row["name"]=name;row["max_abs"]=diff.max().item<double>();row["rms"]=diff.square().mean().sqrt().item<double>();
  row["max_rel"]= (diff/b.abs().clamp_min(1e-30)).max().item<double>();row["outside_tolerance"]=Json::Int64(failures.sum().item<int64_t>());
  Json::StreamWriterBuilder w;w["indentation"]="";std::cout<<Json::writeString(w,row)<<std::endl;
  TORCH_CHECK(!failures.any().item<bool>(),name," parity");
}
List leaves(const List& a){List b;for(const auto& t:a)b.push_back(t.detach().clone().requires_grad_(true));return b;}
List decode(const List& a,bool log_input){return log_input?mapping_cuda::decode14LogScale(a[0],a[1],a[2],a[3],a[4],a[5],a[6]):mapping_cuda::decode14(a[0],a[1].exp(),a[2],a[3],a[4],a[5],a[6]);}
List inputs(int64_t n){auto o=torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);
  return {torch::randn({n,3},o),torch::randn({n,3},o)*2-5,torch::nn::functional::normalize(torch::randn({n,4},o)),
    torch::randn({n,1},o),torch::randn({n,1,3},o),torch::randn({n,14},o)*.1,torch::ones({n},o)};}
void normalDecode(const List& input,const std::string& name,bool unused=false){
  auto a=leaves(input),b=leaves(input);auto x=decode(a,false),y=decode(b,true);List upstream;
  for(int j=0;j<5;++j){compare(name+"/out"+std::to_string(j),y[j],x[j]);upstream.push_back(torch::randn_like(x[j])*.001);}
  if(unused){x={x[0]};y={y[0]};upstream={upstream[0]};}
  auto gx=torch::autograd::grad(x,a,upstream,false,false,true),gy=torch::autograd::grad(y,b,upstream,false,false,true);
  for(int j=0;j<7;++j)compare(name+"/grad"+std::to_string(j),gy[j],gx[j]);
}
void realFailure(const std::filesystem::path& file){
  auto m=torch::jit::load(file.string(),torch::kCUDA);
  auto get=[&](const char* s){return m.attr(s).toTensor().detach().clone();};
  List a{get("xyz"),get("log_scale"),get("rotation"),get("opacity_logits"),get("dc"),get("delta"),torch::ones({1},get("xyz").options())};
  auto old=leaves(a),fresh=leaves(a);auto x=decode(old,false),y=decode(fresh,true);
  TORCH_CHECK(bad(x[1])>0&&!bad(y[1]),"Real failure does not exercise intermediate overflow");
  auto zero=torch::zeros_like(y[1]);auto go=torch::autograd::grad({x[1]},old,{zero},false,false,true);
  auto gn=torch::autograd::grad({y[1]},fresh,{zero},true,false,true);
  TORCH_CHECK(bad(go[1])&&bad(go[5])&&!bad(gn[1])&&!bad(gn[5]),"Real zero-VJP regression");
  auto reference=(a[1].to(torch::kFloat64)+a[5].slice(1,3,6).to(torch::kFloat64)).exp().to(torch::kFloat32);
  compare("real/fp64_scale",y[1],reference,0,1e-5);
  auto small=torch::full_like(y[1],1e-38);auto grad=torch::autograd::grad({y[1]},fresh,{small},false,false,true);
  compare("real/log_scale_vjp",grad[1],small*reference,1e-6,1e-5);
  compare("real/delta_scale_vjp",grad[5].slice(1,3,6),small*reference,1e-6,1e-5);
  std::cout<<"REAL_INTERMEDIATE_OVERFLOW_FIXED=1\n";
}
void cancellation(){
  auto a=inputs(3);a[1]=torch::tensor({{-200.f,-110.f,80.f},{-20.f,-15.f,-10.f},{0.f,10.f,-100.f}},a[1].options());
  a[5].slice(1,3,6).copy_(torch::tensor({{200.f,110.f,-80.f},{19.f,14.f,9.f},{1.f,-9.f,99.f}},a[1].options()));
  a=leaves(a);auto y=decode(a,true);auto expected=(a[1]+a[5].slice(1,3,6)).exp();compare("cancellation/scale",y[1],expected,0,0);
  auto g=torch::autograd::grad({y[1]},a,{torch::ones_like(y[1])},false,false,true);
  compare("cancellation/master",g[1],expected,0,0);compare("cancellation/delta",g[5].slice(1,3,6),expected,0,0);
}
void spatial(const T& logs,const std::string& name){
  auto o=logs.options();const auto n=logs.size(0);
  List a{torch::randn({n,3},o),logs,torch::nn::functional::normalize(torch::randn({n,4},o)),torch::randn({n,1,3},o),
      torch::randn({n,1},o),torch::zeros({n,3},o),torch::ones({n,1},o)};
  auto x=leaves(a),y=leaves(a);auto wc=torch::zeros({3},o),ws=torch::ones({3},o),mean=torch::zeros({19},o),sd=torch::ones({19},o);
  Field field;field->to(torch::kCUDA);field->setFused(false);field->ready_buffer.fill_(1);field->syncReady();
  auto expected=field->input55(x[0],x[1],x[2],x[3],x[4],x[5],x[6],true);
  auto actual=mapping_cuda::spatialLiveLogScaleInput55({y[0],y[1],y[2],y[3],y[4],y[5],y[6],wc,ws,mean,sd});
  compare(name+"/input55",actual,expected);auto u=torch::randn_like(actual)*.001;
  auto gx=torch::autograd::grad({expected},x,{u},false,false,true),gy=torch::autograd::grad({actual},y,{u},false,false,true);
  for(int j=0;j<7;++j)compare(name+"/vjp"+std::to_string(j),gy[j],gx[j]);
  // Independent FP64 direct physical formula, including the existing trace floor.
  auto xx=leaves(a);auto xd=xx[1].to(torch::kFloat64),qd=xx[2].to(torch::kFloat64);
  auto direct=improvements::selector::SpatialCompensationImpl::physical(xx[0].to(torch::kFloat64),xd.exp(),qd,xx[3].to(torch::kFloat64),xx[4].to(torch::kFloat64),xx[5].to(torch::kFloat64),xx[6].to(torch::kFloat64),wc.to(torch::kFloat64),ws.to(torch::kFloat64));
  auto stable=improvements::selector::SpatialCompensationImpl::physical(xx[0].to(torch::kFloat64),xd,qd,xx[3].to(torch::kFloat64),xx[4].to(torch::kFloat64),xx[5].to(torch::kFloat64),xx[6].to(torch::kFloat64),wc.to(torch::kFloat64),ws.to(torch::kFloat64),true);
  compare(name+"/fp64_physical",stable,direct,1e-12,1e-11);
}
void fieldParity(){
  auto a=inputs(127);Field legacy,stable;legacy->to(torch::kCUDA);stable->to(torch::kCUDA);
  {torch::NoGradGuard guard;for(auto& p:legacy->parameters())p.normal_(0,.02);auto x=legacy->parameters(),y=stable->parameters();for(size_t i=0;i<x.size();++i)y[i].copy_(x[i]);}
  legacy->initialize(a[0],a[1].exp(),a[2],a[4],a[3],torch::zeros_like(a[0]),torch::ones({127,1},a[0].options()));
  {torch::NoGradGuard guard;auto x=legacy->buffers(),y=stable->buffers();for(size_t i=0;i<x.size();++i)y[i].copy_(x[i]);}stable->syncReady();
  for(double r:{.1,.2,.4,.6,.8}){
    auto x=leaves(a),y=leaves(a);auto c=torch::zeros_like(a[0]),ref=torch::ones({127,1},a[0].options());
    auto old=legacy->forward(x[0],x[1].exp(),x[2],x[4],x[3],c,ref,r),fresh=stable->forward(y[0],y[1],y[2],y[4],y[3],c,ref,r,true);
    compare("field/forward_"+std::to_string(r),fresh,old);auto g=torch::randn_like(old)*.001;
    List px{x[0],x[1],x[2],x[3],x[4]},py{y[0],y[1],y[2],y[3],y[4]};auto lp=legacy->parameters(),sp=stable->parameters();px.insert(px.end(),lp.begin(),lp.end());py.insert(py.end(),sp.begin(),sp.end());
    auto gx=torch::autograd::grad({old},px,{g},false,false,true),gy=torch::autograd::grad({fresh},py,{g},false,false,true);
    for(size_t i=0;i<gx.size();++i)compare("field/vjp_"+std::to_string(r)+"_"+std::to_string(i),gy[i],gx[i]);
  }
}
int main(int argc,char** argv){try{
  TORCH_CHECK(argc==2,"scale_chain_test real_renderer_origin_input.pt");torch::set_num_threads(2);torch::manual_seed(20261028);
  for(int n:{0,1,17,4096}){normalDecode(inputs(n),"normal_"+std::to_string(n));normalDecode(inputs(n),"unused_"+std::to_string(n),true);}
  auto values=inputs(17);for(auto& v:values)if(v.dim()>1){auto x=torch::empty({v.size(0),v.numel()/v.size(0)*2},v.options());v=x.slice(1,0,x.size(1),2).reshape(v.sizes()).copy_(v);}normalDecode(values,"noncontiguous");
  auto n=inputs(8);auto ids=torch::tensor({1,1,3,1,7},torch::TensorOptions().device(torch::kCUDA).dtype(torch::kLong));auto a=leaves(n),b=leaves(n);List x,y;for(int i=0;i<7;++i){x.push_back(a[i].index_select(0,ids));y.push_back(b[i].index_select(0,ids));}
  auto ox=decode(x,false),oy=decode(y,true);List up;for(int j=0;j<5;++j)up.push_back(torch::randn_like(ox[j])*.001);auto gx=torch::autograd::grad(ox,a,up,false,false,true),gy=torch::autograd::grad(oy,b,up,false,false,true);for(int j=0;j<7;++j)compare("repeated_ids/grad"+std::to_string(j),gy[j],gx[j]);
  realFailure(argv[1]);cancellation();auto o=n[1].options();spatial(torch::randn({127,3},o)*3-5,"spatial_normal");
  spatial(torch::tensor({{-200.f,-199.f,-201.f},{50.f,49.f,48.f},{100.f,0.f,-100.f},{-14.f,-15.f,-16.f}},o),"spatial_edges");fieldParity();
  std::cout<<"SCALE_CHAIN_TEST_PASS\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}
