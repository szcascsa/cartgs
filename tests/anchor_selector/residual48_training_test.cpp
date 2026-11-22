#include "include/residual48_fusion.h"
#include <iostream>
#include <limits>

bool bits(const torch::Tensor& a,const torch::Tensor& b) {
  if(a.defined()!=b.defined())return false;if(!a.defined())return true;
  return a.sizes()==b.sizes()&&torch::equal(a.contiguous().view(torch::kInt32),b.contiguous().view(torch::kInt32));
}
void same(const torch::Tensor& a,const torch::Tensor& b,const std::string& label) {
  if(!bits(a,b)) {
    if(a.defined()&&b.defined()&&a.sizes()==b.sizes()&&a.numel())std::cerr<<label<<" max_abs="<<(a-b).abs().max().item<float>()<<" different="
      <<(a.contiguous().view(torch::kInt32)!=b.contiguous().view(torch::kInt32)).sum().item<int64_t>()<<std::endl;
    TORCH_CHECK(false,label);
  }
}
std::vector<double> weights(double ratio) {
  std::vector<double> knots{.01,.05,.1,.2,.3,.4},w{1.};
  for(int j=1;j<6;++j) {if(ratio<=knots[j-1]+1e-8)break;w.push_back(std::min(1.,(ratio-knots[j-1])/(knots[j]-knots[j-1])));}
  return w;
}
torch::Tensor legacy(const torch::Tensor& x,const torch::Tensor& c,const torch::Tensor& r,
    const torch::Tensor& mean,const torch::Tensor& stddev,const std::vector<torch::Tensor>& f,const std::vector<double>& w) {
  auto z=f[0];for(size_t j=1;j<f.size();++j)z=z+w[j]*f[j];
  return torch::cat({((torch::asinh((x-c)/r)-mean.slice(0,3,6))/stddev.slice(0,3,6)).clamp(-15,15),z},1);
}
int main(int argc,char** argv) {
  torch::set_num_threads(2);torch::manual_seed(2026);at::globalContext().setAllowTF32CuBLAS(false);
  auto o=torch::TensorOptions().device(torch::kCUDA).dtype(torch::kFloat32);int checks=0;
  const std::vector<double> ratios{.005,.01,.025,.05,.075,.1,.15,.2,.25,.3,.35,.4,.6};
  for(int64_t n:{0,1,17,127,627,3129,25023})for(double ratio:ratios) {
    auto w=weights(ratio);auto ids=torch::arange(n,o.dtype(torch::kInt64))%97;
    auto xa=torch::randn({n,3},o).requires_grad_(true),xb=xa.detach().clone().requires_grad_(true);
    auto c=torch::randn({n,3},o),r=(torch::rand({n,1},o)+.01),mean=torch::randn({19},o),stddev=torch::rand({19},o)+.1;
    std::vector<torch::Tensor> ta,tb,fa,fb;
    for(size_t j=0;j<w.size();++j) {
      ta.push_back(torch::randn({97,48},o).requires_grad_(true));tb.push_back(ta.back().detach().clone().requires_grad_(true));
      fa.push_back(ta.back().index_select(0,ids));fb.push_back(tb.back().index_select(0,ids));
    }
    for(auto& f:fa)f=f.detach().requires_grad_(true);for(auto& f:fb)f=f.detach().requires_grad_(true);auto ya=legacy(xa,c,r,mean,stddev,fa,w),yb=mapping_cuda::residual48TrainingPack(xb,c,r,mean,stddev,fb,w);
    same(ya,yb,"pack forward n="+std::to_string(n)+" r="+std::to_string(ratio));
    auto storage=torch::randn({n,102},o);auto up=storage.slice(1,0,102,2);
    ya.backward(up);yb.backward(up);same(xa.grad(),xb.grad(),"pack XYZ VJP");
    for(size_t j=0;j<w.size();++j)same(fa[j].grad(),fb[j].grad(),"gathered Feature VJP n="+std::to_string(n)+" table="+std::to_string(j));
    ++checks;
  }
  // Clamp boundaries, signed zero, very thin scales, infinities/NaNs, and geometry-gradient fallback.
  for(int geometry=0;geometry<2;++geometry)for(int extreme=0;extreme<3;++extreme) {
    auto xa=torch::tensor({{-0.f,0.f,1.f},{-1e20f,1e20f,-1.f},{.1f,.2f,.3f}},o).requires_grad_(true);
    if(extreme==1){torch::NoGradGuard guard;xa[1][0]=std::numeric_limits<float>::infinity();xa[1][1]=std::numeric_limits<float>::quiet_NaN();}
    auto xb=xa.detach().clone().requires_grad_(true),ca=torch::zeros_like(xa).requires_grad_(geometry),cb=ca.detach().clone().requires_grad_(geometry);
    auto ra=torch::full({3,1},extreme==2?1e-30:1.,o).requires_grad_(geometry),rb=ra.detach().clone().requires_grad_(geometry);
    auto ma=torch::zeros({19},o).requires_grad_(geometry),mb=ma.detach().clone().requires_grad_(geometry);
    auto sa=torch::ones({19},o).requires_grad_(geometry),sb=sa.detach().clone().requires_grad_(geometry);
    auto f=torch::randn({3,48},o).requires_grad_(true),g=f.detach().clone().requires_grad_(true);
    auto a=legacy(xa,ca,ra,ma,sa,{f},{1.}),b=mapping_cuda::residual48TrainingPack(xb,cb,rb,mb,sb,{g},{1.});same(a,b,"extreme forward");
    auto up=torch::randn_like(a);a.backward(up);b.backward(up);
    for(auto pair:std::vector<std::pair<torch::Tensor,torch::Tensor>>{{xa,xb},{ca,cb},{ra,rb},{ma,mb},{sa,sb},{f,g}})same(pair.first.grad(),pair.second.grad(),"extreme/geometry VJP");
    ++checks;
  }
  std::cout<<"PACK_FORWARD_VJP_BITWISE_PASS checks="<<checks<<std::endl;
}