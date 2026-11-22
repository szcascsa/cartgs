#include "include/ultrafast_rotation.h"
#include "include/ultra_light_schedule.h"
#include <iostream>
#include <stdexcept>

void check(bool value, const char* message) {
  if (!value) throw std::runtime_error(message);
}

int main() {
  UltrafastRotation rotation;
  const std::array<std::size_t, 12> expected = {0,1,2,3,4,0,1,2,3,4,0,1};
  for (auto phase : expected) {
    check(rotation.phase() == phase, "Wrong ratio order");
    check(rotation.full() == (phase == 4), "Wrong Full phase");
    check(rotation.ratio() == (phase == 4 ? 1.0f : UltrafastRotation::ratios[phase]), "Wrong ratio");
    // Auxiliary work and KF selection do not complete a slot.
    check(rotation.phase() == phase, "Auxiliary work consumed a slot");
    rotation.complete(phase + 1.0, 0.5, true, phase != 4, phase != 4);
    check(rotation.balanced(), "Unbalanced phase counts");
  }
  check(rotation.step == 12 && rotation.full_render_count == 2 &&
        rotation.lod_render_count == 10, "Wrong render counts");
  check(rotation.p95() == 5.0, "Wrong nearest-rank P95");
  UltrafastRotation quiet;
  for (auto phase : expected)
    quiet.complete(phase + 1.0, .5, true, phase != 4, phase != 4, false);
  check(quiet.counts == rotation.counts && quiet.step == rotation.step &&
        quiet.full_render_count == rotation.full_render_count &&
        quiet.lod_render_count == rotation.lod_render_count &&
        quiet.selector_updates == rotation.selector_updates && quiet.tf_updates == rotation.tf_updates &&
        quiet.elapsed.empty(), "Disabling timing changed training counters or retained timing samples");

  rotation.endOnline(12);
  rotation.endOnline(999);
  check(rotation.online_end_mapping_iteration == 12 && rotation.tailSteps() == 0,
        "Online boundary changed");
  const auto online_selector = rotation.online_end_selector_updates;
  const auto online_tf = rotation.online_end_tf_updates;
  const auto online_counts = rotation.counts;
  for (int i = 0; i < 7; ++i) {
    check(rotation.phase() == (2+i)%5, "Tail did not continue the online five-slot cursor");
    const bool full = rotation.full();
    rotation.completeRotatingTail(1.0, 0.5, true, !full, !full, !full);
    check(rotation.balanced(), "Tail changed ratio rotation");
  }
  check(rotation.tailSteps() == 7 && rotation.online_end_rotation_step == 12 &&
        rotation.online_end_selector_updates == online_selector && rotation.online_end_tf_updates == online_tf &&
        rotation.counts == online_counts && rotation.full_render_count == 3 && rotation.lod_render_count == 16 &&
        rotation.tail_full_renders == 1 && rotation.tail_lod_renders == 6,
        "Tail moved the frozen online boundary/counters");
  for (std::uint64_t boundary = 0; boundary < 5; ++boundary) {
    UltrafastRotation partial;
    for (std::uint64_t i = 0; i < boundary; ++i) partial.complete(0,0,true,true,false,false);
    partial.endOnline(boundary);
    const auto first = boundary;
    check(partial.tailRatioIndex() == first,"Incorrect tail entry from a partial online cycle");
    for (std::size_t i = 0; i < 10; ++i) {
      check(partial.phase() == (first+i)%5,"Tail ratio order changed");
      const bool full = partial.full();
      partial.completeRotatingTail(0,0,true,!full,false,false,false);
    }
    for (auto n : partial.tail_ratio_counts) check(n == 2,"Tail ratios not balanced");
    for (auto n : partial.tail_phase_counts) check(n == 2,"Tail five-slot counts not balanced");
    check(partial.balanced() && partial.elapsed.empty() && partial.tail_lod_renders == 8 &&
          partial.tail_full_renders == 2 && partial.full_render_count + partial.lod_render_count == partial.step,
          "Tail bookkeeping changed online balance or disabled timing");
  }

  // A deadline falling on LOD is executed at the next Full, on the original clock.
  DeferredMappingTask topology;
  for (std::uint64_t iteration = 1; iteration <= 5; ++iteration) {
    topology.observe(iteration, 3);
    check(topology.take(iteration == 5) == (iteration == 5), "Topology escaped Full phase");
  }
  std::cout << "PASS: ratio rotation, counters, P95, online boundary, tail continuation, topology deadline\n";
}
