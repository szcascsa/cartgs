#include "calibrated_selection.h"
#include <iostream>

using T=torch::Tensor;

int main(){
  auto options=torch::TensorOptions().dtype(torch::kFloat64);
  double maximum_fd_error=0;
  for(auto device:{torch::Device(torch::kCPU),torch::Device(torch::kCUDA)}){
    auto s=torch::tensor({-3.,-1.,-.2,.4,1.3,2.7},options.device(device)).set_requires_grad(true);
    auto h=torch::tensor({0.,-1.4,0.,.2,-.7,0.},s.options());
    auto out=calibratedSelectionProbability(s,2);
    auto original=(s.detach()-out.second).sigmoid();
    TORCH_CHECK(out.first.detach().equal(original),"FORWARD_CHANGED");
    (out.first*h).sum().backward();auto grad=s.grad();
    auto d=original*(1-original),g=d*h,expected=g-d*g.sum()/d.sum();
    TORCH_CHECK(torch::allclose(grad,expected,1e-12,1e-14),"WRONG_CALIBRATION_VJP");
    TORCH_CHECK(std::abs(grad.sum().item<double>())<1e-14,"COMMON_GRADIENT");
    TORCH_CHECK(grad.index({0}).item<double>()!=0,"UNSELECTED_BUDGET_GRADIENT_MISSING");
    for(int i=0;i<s.numel();++i){
      auto delta=torch::zeros_like(s);delta.index_put_({i},1e-4);
      double plus=(calibratedSelectionProbability(s.detach()+delta,2).first*h).sum().item<double>();
      double minus=(calibratedSelectionProbability(s.detach()-delta,2).first*h).sum().item<double>();
      maximum_fd_error=std::max(maximum_fd_error,std::abs((plus-minus)/2e-4-grad[i].item<double>()));
    }
    auto c=torch::zeros({},s.options()).set_requires_grad(true);
    auto shifted=calibratedSelectionProbability(s.detach()+c,2).first;
    (shifted*h).sum().backward();
    TORCH_CHECK(std::abs(c.grad().item<double>())<1e-14,"SHIFT_SCALAR_GRADIENT");
    auto r=s.detach().clone().set_requires_grad(true);
    auto bias=torch::zeros({},s.options()).set_requires_grad(true);
    auto raw=r+bias;auto a=raw-raw.mean();
    (calibratedSelectionProbability(a,2).first*h).sum().backward();
    TORCH_CHECK(std::abs(bias.grad().item<double>())<1e-14,"CENTERED_BIAS_GRADIENT");
    TORCH_CHECK(std::abs(r.grad().sum().item<double>())<1e-14,"CENTERED_RAW_GRADIENT_SUM");
    auto shift100=calibratedSelectionProbability(s.detach()+100,2).first;
    TORCH_CHECK((shift100-original).abs().max().item<double>()<1e-10,"SHIFT_FORWARD");
    auto uniform=torch::tensor({.2,.3,.1,.4,.5,.6},s.options());
    TORCH_CHECK((shift100>uniform).equal(original>uniform),"SHIFT_SAMPLE_CHANGED");
    // 原反例中修复后的gradient descent应降低重新校准的连续局部目标。
    auto x=torch::tensor({-.15302792457320802,3.3528943521874783,-2.913402581974696,
        2.2468514949034466,-1.7847830786937395,-4.9209283350681385},s.options()).set_requires_grad(true);
    auto upstream=torch::tensor({-.3664626131509324,-.6557429454021607,-2.223064866960999,
        -.3523733298804491,-1.1973025371979495,-.9238232598755329},s.options());
    auto objective=(calibratedSelectionProbability(x,2).first*upstream).sum();objective.backward();
    auto improved=(calibratedSelectionProbability(x.detach()-1e-3*x.grad(),2).first*upstream).sum();
    TORCH_CHECK(improved.item<double>()<objective.item<double>(),"REGRESSION_ASCENT_DIRECTION");
  }
  TORCH_CHECK(maximum_fd_error<1e-6,"FINITE_DIFFERENCE_MISMATCH");
  std::cout<<"{\"all_passed\":true,\"devices\":[\"CPU\",\"CUDA\"],\"forward_exact\":true,\"common_gradient_zero\":true,\"FD_max_error\":"<<maximum_fd_error<<"}"<<std::endl;
}
