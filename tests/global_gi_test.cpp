#include "include/global_gi.h"
#include "include/gi_teacher.h"
#include <limits>
#include <stdexcept>

namespace {
void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
void equal(const torch::Tensor& actual, std::initializer_list<float> expected) {
  check(torch::allclose(actual, torch::tensor(std::vector<float>(expected))), "GI contribution or aggregate is misaligned");
}
GlobalGIConfig config() {
  GlobalGIConfig result;
  result.mode = "streaming"; result.view_ratio = 1; result.performance_logging = false;
  return result;
}
void incrementalAndValidity() {
  GlobalGIManager gi(torch::kCPU, config());
  gi.resetForGaussianCount(3);
  int renders = 0;
  std::map<uint64_t, torch::Tensor> contributions{
      {1, torch::tensor({1.f, 2.f, 3.f})}, {2, torch::tensor({2.f, 1.f, 4.f})},
      {3, torch::tensor({3.f, 3.f, 3.f})}};
  auto render = [&](uint64_t id) { ++renders; return contributions.at(id); };
  gi.updateBankBatch({{{1, {1}}}, {}}, render); equal(gi.importance(), {1, 2, 3});
  check(gi.validMask().all().item<bool>(), "Initial complete bank estimate is invalid");
  gi.updateBank({{1, {1}}, {2, {2}}}, render); equal(gi.importance(), {1.5f, 1.5f, 3.5f});
  check(renders == 2, "An existing view was re-rendered on bank addition");
  gi.updateBank({{2, {2}}, {3, {3}}}, render); equal(gi.importance(), {2.5f, 2, 3.5f});
  check(renders == 3, "Replacement re-rendered the surviving bank view");
  gi.updateBank({{3, {3}}}, render); equal(gi.importance(), {3, 3, 3});
  check(renders == 3, "Removal invoked the contribution renderer");
  gi.updateBankBatch({}, render);
  check(renders == 3, "Ordinary mapping refreshed a cached view");
  gi.appendGaussians(1, GITopologyKind::Append);
  check(gi.validMask().equal(torch::tensor({true, true, true, false})), "Append inherited old GI validity");
  contributions[3] = torch::tensor({30.f, 30.f, 30.f, 30.f});
  contributions[2] = torch::tensor({2.f, 1.f, 4.f, 99.f});
  gi.updateBank({{2, {2}}, {3, {3}}}, render);
  check(!gi.validMask()[3].item<bool>(), "A partial new-view estimate resurrected an invalid newborn");
  auto mature_all = torch::ones({4}, torch::kBool);
  auto partial_teacher = GITeacherBuilder().build(gi.importance(), gi.validMask(),
      mature_all, torch::ones({4}), 1.0f, gi.version(), gi.topologyVersion());
  check(partial_teacher.target_k == 3 && partial_teacher.supervised_k == 3,
        "Insufficient valid GI changed the global selection budget");
  auto prediction = torch::full({4}, 0.5f).requires_grad_(true);
  (prediction.index({partial_teacher.supervised_mask}) -
      partial_teacher.label.index({partial_teacher.supervised_mask})).abs().sum().backward();
  check(prediction.grad()[3].item<float>() == 0, "An invalid newborn received GI supervision");
  check(!gi.refreshIfDue(999, render) && renders == 4, "GI refreshed before iteration 1000");
  check(gi.refreshIfDue(1000, render) && renders == 6, "Iteration 1000 did not re-render the entire bank");
  equal(gi.importance(), {16, 15.5f, 17, 64.5f});
  check(gi.validMask().all().item<bool>(), "Full refresh failed to validate newborns");
  check(!gi.refreshIfDue(1000, render) && !gi.refreshIfDue(1999, render), "Full refresh schedule repeated");
  check(gi.refreshIfDue(2000, render) && renders == 8, "Iteration 2000 failed to refresh");
  auto mask = torch::tensor({true, true, true, false});
  auto mature = torch::ones({4}, torch::kBool);
  auto teacher = GITeacherBuilder().build(gi.importance(), gi.validMask(), mask,
      torch::ones({4}), 0.75f, gi.version(), gi.topologyVersion());
  check(teacher.target_k == 2 && teacher.supervised_k == 3 && teacher.supervised_mask.equal(mask),
        "Invalid GI narrowed the global budget or supervised unknown rows");
  // Replacing the map requires a fresh bootstrap even if bank IDs stay identical.
  gi.resetForGaussianCount(4);
  gi.updateBankBatch({}, render);
  check(gi.validMask().all().item<bool>() && gi.views().size() == 2, "Map reset retained an empty/stale bank cache");
}
void topologySequence() {
  GlobalGIManager gi(torch::kCPU, config());
  gi.resetForGaussianCount(4);
  int renders = 0;
  auto render = [&](uint64_t id) {
    ++renders;
    return id == 1 ? torch::tensor({1.f, 2.f, 3.f, 4.f}) : torch::tensor({4.f, 3.f, 2.f, 1.f});
  };
  gi.updateBank({{1, {1}}, {2, {2}}}, render);
  gi.assertAligned(torch::zeros({4, 3}));
  gi.appendGaussians(1, GITopologyKind::Append);
  gi.assertAligned(torch::zeros({5, 3}));
  equal(gi.views().at(1).contribution, {1, 2, 3, 4, 0});
  gi.pruneGaussians(torch::tensor({true, false, true, true, true}));
  gi.assertAligned(torch::zeros({4, 3}));
  equal(gi.views().at(1).contribution, {1, 3, 4, 0});
  equal(gi.views().at(2).contribution, {4, 2, 1, 0});
  gi.appendGaussians(1, GITopologyKind::Clone);
  gi.assertAligned(torch::zeros({5, 3}));
  equal(gi.views().at(1).contribution, {1, 3, 4, 0, 0});
  gi.appendGaussians(2, GITopologyKind::Split);
  gi.assertAligned(torch::zeros({7, 3}));
  equal(gi.views().at(1).contribution, {1, 3, 4, 0, 0, 0, 0});
  // GaussianModel removes the split parent only after appending its children.
  gi.pruneGaussians(torch::tensor({false, true, true, true, true, true, true}));
  gi.assertAligned(torch::zeros({6, 3}));
  equal(gi.views().at(1).contribution, {3, 4, 0, 0, 0, 0});
  gi.pruneGaussians(torch::tensor({true, false, true, true, true, true}));
  gi.assertAligned(torch::zeros({5, 3}));
  equal(gi.views().at(1).contribution, {3, 0, 0, 0, 0});
  equal(gi.views().at(2).contribution, {2, 0, 0, 0, 0});
  equal(gi.importance(), {2.5f, 0, 0, 0, 0});
  check(gi.validMask().equal(torch::tensor({true, false, false, false, false})), "Continuous topology changed old/new identity");
  check(renders == 2, "Topology synchronization re-rendered historical views");
  // Pruning everything and appending to an empty map must stay aligned.
  gi.pruneGaussians(torch::zeros({5}, torch::kBool));
  gi.assertAligned(torch::zeros({0, 3}));
  gi.appendGaussians(1, GITopologyKind::Append);
  equal(gi.importance(), {0});
  check(!gi.validMask()[0].item<bool>(), "Newborn in an empty map became valid");
}
void nonfiniteAndEmpty() {
  GlobalGIManager gi(torch::kCPU, config());
  gi.resetForGaussianCount(2);
  auto bad = [](uint64_t) { return torch::tensor({std::numeric_limits<float>::quiet_NaN(), 0.f}); };
  gi.updateBank({{1, {}}}, bad);
  check(gi.validMask().equal(torch::tensor({false, true})), "Unknown/non-finite GI became a supervised zero");
  gi.refreshIfDue(1000, bad);
  check(gi.validMask().equal(torch::tensor({false, true})), "Full refresh validated non-finite GI");
  gi.updateBank({}, bad);
  check(!gi.validMask().any().item<bool>(), "Empty bank retained GI guidance");
  auto mature = torch::ones({2}, torch::kBool);
  auto empty_teacher = GITeacherBuilder().build(gi.importance(), gi.validMask(), mature,
      torch::ones({2}), 1.0f, gi.version(), gi.topologyVersion());
  check(empty_teacher.supervised_k == 0 && empty_teacher.target_k == 0,
        "Empty valid Mature set cancelled the normal Selector budget");
  gi.resetForGaussianCount(0);
  gi.updateBank({{1, {}}}, [](uint64_t) { return torch::empty({0}); });
  gi.assertAligned(torch::zeros({0, 3}));
}
void deferredRefreshUsesMappingClock() {
  GlobalGIManager gi(torch::kCPU, config());
  gi.resetForGaussianCount(1);
  int renders = 0;
  auto render = [&](uint64_t) { ++renders; return torch::ones({1}); };
  gi.updateBank({{1, {1}}}, render);
  check(!gi.refreshIfDue(1001, render), "Ordinary non-due mapping refreshed GI");
  check(gi.refreshIfDue(1001, render, true) && renders == 2,
        "Deferred mapping refresh was rejected");
  gi.appendGaussians(1, GITopologyKind::Clone);
  gi.pruneGaussians(torch::tensor({false, true}));
  check(!gi.validMask().any().item<bool>(), "Basic-step topology inherited validity");
  check(gi.refreshIfDue(1019, render, true) && gi.validMask().all().item<bool>(),
        "Shutdown refresh failed to validate new topology");
  check(gi.refreshIfDue(2000, render), "Deferred refresh shifted the original 2000 deadline");
}
}
int main() { incrementalAndValidity(); topologySequence(); nonfiniteAndEmpty(); deferredRefreshUsesMappingClock(); }
