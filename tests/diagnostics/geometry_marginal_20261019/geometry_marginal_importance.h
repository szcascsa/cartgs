#pragma once
#include <torch/torch.h>
#include <filesystem>
#include "geometry_marginal_ranker.h"
// Published score/mask rows follow the exact model append/prune callbacks.
// Geometry caches live only inside a refresh, never across mapping updates.
class GeometryMarginalImportance {
 public:
  void reset(int64_t n,const torch::TensorOptions&options);
  void append(int64_t n);
  void prune(const torch::Tensor&keep);
  void refresh(const torch::Tensor&xyz,const torch::Tensor&scale,const torch::Tensor&raw,
      const torch::Tensor&valid,uint64_t iteration,uint64_t gi_version,uint64_t topology,
      const std::filesystem::path&directory,bool compute,bool shadow,bool diagnostics);
  void assertAligned(int64_t n)const;
  bool ready()const{return ready_;}
  const torch::Tensor&score()const{return score_;}
  const torch::Tensor&valid()const{return valid_;}
  uint64_t version()const{return gi_version_;}
  uint64_t publishedIteration()const{return published_iteration_;}
 private:
  torch::Tensor score_,valid_;
  double h_=0;
  bool ready_=false;
  uint64_t gi_version_=0,published_iteration_=0;
};
