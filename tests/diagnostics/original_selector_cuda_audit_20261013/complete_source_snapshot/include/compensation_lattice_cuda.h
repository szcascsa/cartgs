#pragma once
#include <torch/types.h>
torch::Tensor compensationLatticeCUDA(const torch::Tensor& xyz, const torch::Tensor& coarse,
    const torch::Tensor& lo, const torch::Tensor& hi, const torch::Tensor& bound,
    int resolution, int lower, int upper, float beta, float scale_bound, bool outside_identity,
    const torch::Tensor& fine, const torch::Tensor& tile_min, const torch::Tensor& tile_max,
    const torch::Tensor& cell_tiles, int fine_resolution);
void reportCompensationCudaTiming();
void compensationDemandCUDA(const torch::Tensor& xyz,const torch::Tensor& gx,const torch::Tensor& gs,
    const torch::Tensor& lo,const torch::Tensor& hi,int cells_per_axis,
    const torch::Tensor& demand,const torch::Tensor& updates,float beta,float scale_weight);
