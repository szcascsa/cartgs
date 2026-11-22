#pragma once
#include <torch/types.h>

// 与SGI共用一次光栅化，但保留独立alpha*T口径。Scope只在发射端使用。
namespace rendering_contribution {
class CaptureScope {
 public:
  CaptureScope(bool enabled,int stride=1,int64_t phase=0);
  ~CaptureScope();
  torch::Tensor sums;
  int stride,offset_x,offset_y;
 private:
  CaptureScope* previous_;
};
CaptureScope* current();
}
