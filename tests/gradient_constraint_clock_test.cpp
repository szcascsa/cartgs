#include "include/gradient_constraint_clock.h"
#include <stdexcept>

void check(bool ok) { if (!ok) throw std::runtime_error("Mapping update clock regression"); }
int main() {
  MappingUpdateClock clock;
  check(!clock.guardReady(false, 0));
  for (int i = 0; i < 999; ++i) clock.record(true, true);
  check(!clock.guardReady(true, 1000));
  clock.record(true, false);  // Gaussian-only update is not TF training.
  clock.record(false, true); // CANCEL/NOOP is not a successful TF update.
  check(clock.tf_updates == 999 && clock.rejected_updates == 1);
  check(clock.gaussianLRStep(1001) == 1000);
  clock.record(true, true);
  check(clock.guardReady(true, 1000) && clock.tf_updates == 1000);
  check(!clock.guardReady(false, 1000));
  clock.record(false, false);
  check(clock.tf_updates == 1000 && clock.gaussianLRStep(1002) == 1000);
  // CD/replay/mapping topology operations never call record().
  check(clock.guardReady(true, 1000));
  clock.tf_updates = 0; // Only creating a new TF resets its warmup clock.
  check(!clock.guardReady(true, 1000));
  MappingUpdateClock tail;
  tail.record(true,false,false);  // Full commits.
  tail.record(false,true,false);  // LoD rejected, with no Mapping rejection yet.
  tail.record(true,false);        // Full makes the Mapping successful.
  check(tail.rejected_updates == 0 && tail.tf_updates == 0 && tail.gaussianLRStep(1) == 1);
  tail.record(false,false,false);
  tail.record(false,true,false);
  tail.record(false,false);       // Both stages rejected: count exactly once.
  check(tail.rejected_updates == 1 && tail.tf_updates == 0 && tail.gaussianLRStep(2) == 1);
  tail.record(true,false,false);
  tail.record(true,true,false);
  tail.record(true,false);
  check(tail.rejected_updates == 1 && tail.tf_updates == 1 && tail.gaussianLRStep(3) == 2);
}
