#include <torch/extension.h>
#include "cuda_rasterizer/rasterize_points.h"
#include "cuda_rasterizer/anchor_si_capture.h"
#include "include/local_importance.h"
#include "include/loss_utils.h"

std::vector<torch::Tensor> renderFixed(const std::vector<torch::Tensor>& a,int height,int width,
    float tanx,float tany,int degree,int statistic,int mode) {
  TORCH_CHECK(a.size()==12&&statistic>=0&&statistic<=2&&(mode==0||mode==1),"Invalid frozen render arguments");
  torch::NoGradGuard guard;
  anchor_si::CaptureScope alpha(statistic==2);
  auto result=RasterizeGaussiansCUDAWithImportance(a[0],a[1],a[2],a[3],a[4],a[5],1.f,a[6],a[7],a[8],
      tanx,tany,height,width,a[9],a[10],degree,a[11],false,false,statistic!=0,
      mode==0?RasterizeMode::Training:RasterizeMode::Inference);
  return {std::get<2>(result),std::get<3>(result),std::get<4>(result),std::get<5>(result)};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("render",&renderFixed);
  m.def("teacher",&localImportanceTarget);
  m.def("photo",[](torch::Tensor image,torch::Tensor target) {
    torch::NoGradGuard guard;
    return .8f*loss_utils::l1_loss(image,target)+.2f*(1.f-loss_utils::fast_ssim(image,target));
  });
}
