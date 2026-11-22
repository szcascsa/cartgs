#pragma once

#include <cstdint>

// Normal mapping only: CD/DSD/replay must never advance these clocks.
struct MappingUpdateClock {
  std::uint64_t rejected_updates = 0;
  std::uint64_t tf_updates = 0;

  bool guardReady(bool tf_initialized, int warmup_steps) const {
    return tf_initialized && tf_updates >= static_cast<std::uint64_t>(warmup_steps);
  }
  void record(bool committed, bool tf_trained) {
    if (!committed) ++rejected_updates;
    else if (tf_trained) ++tf_updates;
  }
  std::uint64_t gaussianLRStep(std::uint64_t iteration) const {
    return iteration > rejected_updates ? iteration - rejected_updates : 0;
  }
};
