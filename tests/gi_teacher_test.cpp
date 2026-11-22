#include "include/gi_teacher.h"
#include <stdexcept>

void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
int main(int argc, char**) {
  const auto options=torch::TensorOptions().dtype(torch::kFloat32).device(argc>1?torch::kCUDA:torch::kCPU);
  const auto scores=torch::tensor({.1f,.2f,.3f,.4f,.9f},options);
  const auto valid=torch::tensor({true,true,true,false,true},options.dtype(torch::kBool));
  const auto observed=torch::tensor({true,true,true,true,false},valid.options());
  auto maturity=torch::tensor({0.f,.5f,1.f,1.f,1.f},options).requires_grad_(true);
  const auto teacher=GITeacherBuilder().build(scores,valid,observed,maturity,.5,7,11);
  check(teacher.supervised_k==3 && teacher.target_k==1, "budget must use valid observed population");
  check(teacher.supervised_mask.equal(valid & observed), "eligibility must not depend on maturity");
  check(teacher.weight.equal(torch::tensor({0.f,.5f,1.f,0.f,0.f},options)), "continuous GI weights");
  check(teacher.label.equal(torch::tensor({0.f,0.f,1.f,0.f,0.f},options)), "global ratio percentile");
  auto prediction=torch::full({5},.25,options).requires_grad_(true);
  auto loss=teacher.loss(prediction,false); loss.backward();
  check(std::abs(loss.item<float>()-(.5*.25+.75)/1.5)<1e-6, "weighted normalized GI loss");
  check(prediction.grad()[0].item<float>()==0 && prediction.grad()[3].item<float>()==0 &&
      prediction.grad()[4].item<float>()==0, "zero/invalid/unobserved rows must receive no GI gradient");
  check(std::abs(prediction.grad()[1].abs().item<float>()/prediction.grad()[2].abs().item<float>()-.5)<1e-6,
      "m=.5 must receive half GI gradient in the same population");
  check(!maturity.grad().defined(), "GI maturity must be detached");
  const auto zero=GITeacherBuilder().build(scores,valid,observed,torch::zeros_like(maturity),.01,8);
  check(zero.loss(prediction,false).item<float>()==0, "zero weight population must be finite zero loss");
  const auto all=torch::ones({100},valid.options());
  for(float ratio:{.01f,.05f,.1f,.15f}) {
    const auto ranked=GITeacherBuilder().build(torch::arange(100,options),all,all,torch::zeros({100},options),ratio,9);
    check(ranked.supervised_k==100 && ranked.target_k==static_cast<int64_t>(std::floor(double(ratio)*100)),
        "ranking population must include valid observed m=0 rows");
  }
  return 0;
}
