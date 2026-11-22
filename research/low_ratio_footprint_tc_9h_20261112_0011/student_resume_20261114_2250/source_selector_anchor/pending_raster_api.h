#pragma once
#include <torch/extension.h>
std::vector<torch::Tensor> render_pending_shared(void* handle,const torch::Tensor& bg,
    const torch::Tensor& flat,const torch::Tensor& frame,int64_t capacity,int64_t rest,
    const torch::Tensor& view,const torch::Tensor& proj,double tanx,double tany,
    int64_t H,int64_t W,int64_t degree,const torch::Tensor& camera);
