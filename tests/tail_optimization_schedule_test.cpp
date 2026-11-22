#include "include/tail_optimization_schedule.h"
#include <initializer_list>
#include <stdexcept>

namespace {
void check(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
void automaticTail() {
  for (const auto iteration : {0u, 1u, 7u, 800u, 16001u}) {
    TailOptimizationSchedule tail;
    tail.enabled = true;
    check(tail.begin(iteration, 100), "Tail did not begin");
    check(tail.target_steps == iteration / 2, "Automatic tail is not floor(iteration/2)");
    check(!tail.begin(iteration + 1000, 101) && tail.start_mapping == 100 &&
          tail.start_iteration == iteration, "Shutdown snapshot changed after the first tail mapping");
    if (tail.target_steps > 0)
      check(!tail.complete(100 + tail.target_steps - 1), "Tail completed one mapping early");
    check(tail.complete(100 + tail.target_steps), "Tail did not stop at its exact target");
  }
}
void fixedTailAndClocks() {
  TailOptimizationSchedule tail;
  tail.enabled = true;
  tail.steps = 7;
  tail.begin(9999, 101);
  for (std::uint64_t mapping = 101; mapping < 108; ++mapping) {
    check(!tail.complete(mapping), "Fixed tail ended early");
    check(tail.completed(mapping) == mapping - 101, "Tail length used an enhanced-step clock");
  }
  check(tail.complete(108) && tail.completed(108) == 7, "Fixed length retained the 400-step floor");
  TailOptimizationSchedule zero;
  zero.enabled = true;
  zero.steps = 0;
  zero.begin(9999, 101);
  check(zero.complete(101), "Zero-step override performed tail training");
}
void limitsAndDefaults() {
  TailOptimizationSchedule tail;
  check(!tail.begin(800, 800), "Disabled mode started an overridden tail");
  check(tail.mappingLimit(16400, true) == 16400, "Disabled mode changed the Mapping cap");
  tail.enabled = true;
  check(tail.mappingLimit(0, true) == 0 && tail.mappingLimit(1, true) == 0,
        "Forced tail retained the online Mapping cap");
  check(tail.mappingLimit(16400, false) == 16400, "SLAM tail mode changed COLMAP limits");
  check(!tail.allowsOptimization(51, 50) && tail.allowsOptimization(49, 50) &&
        tail.allowsOptimization(51, -1), "Online Gaussian optimization cap changed");
  tail.begin(51, 51);
  check(tail.allowsOptimization(52, 50), "Forced tail ran without optimizer updates");
  tail.steps = -2;
  bool rejected = false;
  try { tail.validate(); } catch (const std::invalid_argument&) { rejected = true; }
  check(rejected, "Invalid negative fixed length was accepted");
}
}  // namespace
int main() {
  automaticTail();
  fixedTailAndClocks();
  limitsAndDefaults();
}
