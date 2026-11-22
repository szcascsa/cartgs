#pragma once
#include <torch/torch.h>

namespace mapping_cuda {
// Returns false before any mutation when the complete step needs legacy Adam.
bool tryFusedDenseAdam(torch::optim::Adam& optimizer);
}
