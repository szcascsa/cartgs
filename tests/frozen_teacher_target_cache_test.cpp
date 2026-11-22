#include "include/frozen_teacher_target_cache.h"

#include <stdexcept>

namespace {
void check(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
}

int main() {
  auto input = torch::arange(8, torch::TensorOptions().dtype(torch::kFloat32))
                   .reshape({4, 2}).set_requires_grad(true);
  const auto ids = torch::TensorOptions().dtype(torch::kLong);
  FrozenTeacherTargetCache cache(4, input.options());
  int64_t evaluated = 0;
  auto teacher = [&](const torch::Tensor& missing) {
    evaluated += missing.numel();
    return input.index_select(0, missing) * 2.0;
  };
  auto first = cache.get(torch::tensor({2, 0}, ids), teacher);
  check(evaluated == 2 && !first.requires_grad(), "Targets must be frozen");
  auto second = cache.get(torch::tensor({0, 3, 2}, ids), teacher);
  check(evaluated == 3, "Repeated targets were recomputed");
  check(torch::equal(second, input.detach().index_select(
      0, torch::tensor({0, 3, 2}, ids)) * 2.0), "Shuffled cache lookup mismatch");

  // Mutating a returned batch must not alter cached targets.
  first.zero_();
  auto repeated = cache.get(torch::tensor({2, 0}, ids), teacher);
  check(evaluated == 3 && torch::equal(repeated, input.detach().index_select(
      0, torch::tensor({2, 0}, ids)) * 2.0), "Cache aliases returned batches");

  auto duplicates = torch::tensor({2, 0, 2, 3}, ids);
  const auto direct = cache.getCached(duplicates);
  const auto generic = cache.get(duplicates, teacher);
  check(torch::equal(direct, generic) && evaluated == 3,
        "Proven cache hits differ for repeated IDs");
  auto fast = cache.get(duplicates, [&](const torch::Tensor&) -> torch::Tensor {
    throw std::runtime_error("Fully cached path evaluated teacher");
  }, true);
  check(torch::equal(fast, direct), "Explicit coverage contract mismatch");
  auto partial = cache.get(torch::tensor({1, 2, 1}, ids), teacher);
  check(evaluated == 5 && torch::equal(partial, input.detach().index_select(
      0, torch::tensor({1, 2, 1}, ids)) * 2.0), "Duplicate lazy misses mismatch");

  // Cached teacher targets must preserve the uncached Student loss and gradient.
  auto student = torch::ones_like(second).set_requires_grad(true);
  auto cached_loss = (student - second).abs().mean();
  cached_loss.backward();
  auto expected_student = torch::ones_like(second).set_requires_grad(true);
  auto expected_loss = (expected_student - input.detach().index_select(
      0, torch::tensor({0, 3, 2}, ids)) * 2.0).abs().mean();
  expected_loss.backward();
  check(torch::allclose(cached_loss, expected_loss) &&
            torch::allclose(student.grad(), expected_student.grad()) &&
            !input.grad().defined(), "Caching changed gradient routing");

  // Each ratio/event owns a fresh cache. A new input must not inherit old targets.
  FrozenTeacherTargetCache next_event(4, input.options());
  auto changed = next_event.get(torch::tensor({2, 0}, ids),
      [&](const torch::Tensor& missing) {
        return input.index_select(0, missing) * 3.0;
      });
  check(torch::equal(changed, input.detach().index_select(
      0, torch::tensor({2, 0}, ids)) * 3.0), "New event reused old targets");
  return 0;
}
