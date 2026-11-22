#pragma once
#include <cstdint>

// One request, written on the current CUDA stream and read once after preparation.
// No view/ratio result is retained for reuse by a later request.
struct DeploymentFrame {
  int64_t n,visible,available,budget,k;
  double ratio;
  float log_ratio,weights[6],ratio_code[3];
  int status,has_ratio;
};
static_assert(sizeof(DeploymentFrame)==96,"Deployment control layout");

struct DeploymentKnots {
  double values[6],log_lo,log_span;
  int count;
};
