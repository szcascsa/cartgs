#pragma once

#include <cstdint>

// A due task waits for permission to run; execution still uses the real mapping clock.
struct DeferredMappingTask {
  bool pending = false;

  void observe(std::uint64_t mapping, std::uint64_t interval) {
    if (interval > 0 && mapping > 0 && mapping % interval == 0)
      pending = true;
  }
  bool take(bool can_run) {
    if (!can_run || !pending) return false;
    pending = false;
    return true;
  }
};
