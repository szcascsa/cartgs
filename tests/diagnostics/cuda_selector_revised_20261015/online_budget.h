#pragma once
// Test-only timing and cycle-boundary stop controls. No training math changes.
#include "include/gaussian_mapper.h"
#include <cstdlib>
#include <chrono>
#include <utility>

namespace selector_revised {
using Clock=std::chrono::steady_clock;
struct State {
  int seed=std::getenv("SELECTOR_VALIDATION_SEED")?std::stoi(std::getenv("SELECTOR_VALIDATION_SEED")):0;
  double budget_ms=std::getenv("SELECTOR_VALIDATION_MAPPING_SECONDS")?1000*std::stod(std::getenv("SELECTOR_VALIDATION_MAPPING_SECONDS")):0;
  double mapping_ms=0,selector_forward_ms=0,mapper_thread_ms=0;
  uint64_t cycles=0,commits=0,selector_calls=0;
  bool time_budget_reached=false;
};
inline State state;
template<class... Args> torch::Tensor probability(improvements::selector::GumbelNetwork& network,Args&&... args) {
  auto begin=Clock::now();auto result=network->forwardProbability(std::forward<Args>(args)...);
  state.selector_forward_ms+=std::chrono::duration<double,std::milli>(Clock::now()-begin).count();
  ++state.selector_calls;return result;
}
inline void recordStep(GaussianMapper& mapper,Clock::time_point begin,uint64_t cycles,uint64_t rejected) {
  state.mapping_ms+=std::chrono::duration<double,std::milli>(Clock::now()-begin).count();
  state.cycles=cycles;
  state.commits=cycles-rejected;
  if(state.budget_ms>0 && state.mapping_ms>=state.budget_ms){state.time_budget_reached=true;mapper.signalStop();}
}
inline void run(GaussianMapper& mapper) {
  auto begin=Clock::now();mapper.run();
  state.mapper_thread_ms=std::chrono::duration<double,std::milli>(Clock::now()-begin).count();
}
inline void report() {
  std::cout<<std::fixed<<std::setprecision(6)<<"[SELECTOR REVISED] algorithm_seed="<<state.seed
    <<" mapping_cycles="<<state.cycles<<" committed_updates="<<state.commits
    <<" mapping_host_work_ms="<<state.mapping_ms<<" mapper_thread_wall_ms="<<state.mapper_thread_ms
    <<" selector_forward_host_ms="<<state.selector_forward_ms<<" selector_forward_calls="<<state.selector_calls
    <<" mapping_budget_ms="<<state.budget_ms<<" time_budget_reached="<<state.time_budget_reached
    <<" overshoot_ms="<<(state.time_budget_reached?state.mapping_ms-state.budget_ms:0)
    <<" selector_timing_scope=network_forward_host_only"<<std::endl;
}
}
