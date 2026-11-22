#include "include/mapping_profiler.h"

#include <stdexcept>

int main() {
  std::ostringstream captured;
  auto* original = std::cout.rdbuf(captured.rdbuf());
  {
    MappingProfiler profiler;
    profiler.begin(false);
    profiler.cancel();
    for (int i = 1; i <= 1000; ++i) {
      profiler.begin(false);
      profiler.setPhase(i > 500);
      { auto stage = profiler.scope(MappingProfiler::Operations); }
      // Nested begin used by trainForOneIteration must not reset this cycle.
      profiler.begin(false);
      { auto stage = profiler.scope(MappingProfiler::Historical); }
      { auto stage = profiler.scope(MappingProfiler::GaussianOptimizer); }
      { auto stage = profiler.scope(MappingProfiler::LodOptimizer); }
      profiler.end(i);
    }
    profiler.report(1000);
  }
  std::cout.rdbuf(original);
  const auto log = captured.str();
  if (log.find("[Mapping PERF] iter=1000 cycles=1000") == std::string::npos ||
      log.find("wall_ms(avg/p95/max)=") == std::string::npos ||
      log.find("historical:") == std::string::npos ||
      log.find("sampled_cuda_elapsed_ms=") == std::string::npos ||
      log.find("phase=online cycles=500") == std::string::npos ||
      log.find("phase=tail cycles=500") == std::string::npos ||
      log.find("gaussian_optimizer:") == std::string::npos ||
      log.find("lod_optimizer:") == std::string::npos)
    throw std::runtime_error("Mapping profiler omitted full-cycle/stage metrics");
  std::ostringstream quiet;
  original = std::cout.rdbuf(quiet.rdbuf());
  {
    MappingProfiler profiler;
    profiler.enabled = false;
    for (int i = 0; i < 1000; ++i) {
      profiler.begin(true);
      { auto stage = profiler.scope(MappingProfiler::Operations); }
      profiler.end(i);
    }
    profiler.report(1000);
  }
  std::cout.rdbuf(original);
  if (!quiet.str().empty())
    throw std::runtime_error("Disabled Mapping profiler still produced output");
}
