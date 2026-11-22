#include "calibrated_selection.h"
#include <iostream>
int main(){
  double error=0;
  for(auto device:{torch::Device(torch::kCPU),torch::Device(torch::kCUDA)}){
    auto opt=torch::TensorOptions().dtype(torch::kFloat64).device(device);
    auto occupancy=torch::tensor({2.,1.,3.},opt),weights=occupancy/6.;
    auto cell=torch::tensor({0,0,1,2,2,2},torch::TensorOptions().dtype(torch::kInt64).device(device));
    auto r=torch::tensor({-.3,.7,1.1},opt).set_requires_grad(true),c=torch::zeros({},opt).set_requires_grad(true);
    auto raw=r+c,a=raw-(raw*weights).sum(),ga=a.index_select(0,cell);
    TORCH_CHECK(std::abs(ga.mean().item<double>())<1e-14,"WEIGHTED_MEAN_FAILED");
    TORCH_CHECK(ga[0].item<double>()==ga[1].item<double>()&&ga[3].item<double>()==ga[5].item<double>(),"NOT_SHARED");
    auto base=torch::tensor({-.2,.8,.4,-1.,.9,1.3},opt),h=torch::tensor({-.2,.1,.8,-.3,1.1,.5},opt);
    auto p=calibratedSelectionProbability(base+ga,2).first;auto loss=(p*h).sum();loss.backward();
    TORCH_CHECK(std::abs(c.grad().item<double>())<1e-14&&std::abs(r.grad().sum().item<double>())<1e-14,"COMMON_DIRECTION_GRADIENT");
    auto gradient=r.grad().clone();
    auto objective=[&](torch::Tensor input){auto ac=input-(input*weights).sum();return (calibratedSelectionProbability(base+ac.index_select(0,cell),2).first*h).sum().item<double>();};
    for(int j=0;j<3;++j){auto d=torch::zeros_like(r);d.index_put_({j},1e-4);double fd=(objective(r.detach()+d)-objective(r.detach()-d))/2e-4;error=std::max(error,std::abs(fd-gradient[j].item<double>()));}
  }
  TORCH_CHECK(error<1e-6,"WEIGHTED_CENTER_VJP_FD");
  std::cout<<"{\"all_passed\":true,\"devices\":[\"CPU\",\"CUDA\"],\"weighted_centering_FD_max_abs\":"<<error<<"}"<<std::endl;
}
