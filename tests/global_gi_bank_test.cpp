#include "include/global_gi_bank.h"
#include "global_gi_bank_reference.h"
#include <chrono>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>

namespace {
void check(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
void quotasAndSampling() {
  GlobalGIConfig config;
  GlobalGIViewBank bank(config), repeat(config);
  std::vector<GlobalGIKeyframe> pool;
  for (uint64_t id = 1; id <= 250; ++id) {
    pool.push_back({id, {id, id + 1000, id}});
    bank.update(pool);
    auto shuffled = pool;
    std::reverse(shuffled.begin(), shuffled.end());
    repeat.update(shuffled);
    const size_t k = (3 * pool.size() + 9) / 10;
    const size_t anchors = (4 * k + 4) / 5;
    check(bank.members().size() == k && bank.target() == k, "30% quota is incorrect");
    check(bank.anchors().size() == anchors && bank.representatives().size() == k - anchors,
          "80/20 composition is incorrect");
    check(bank.members() == repeat.members(), "Input order changed deterministic sampling");
    std::vector<std::pair<uint64_t, uint64_t>> priorities;
    for (const auto& frame : pool)
      if (!bank.anchors().count(frame.id)) priorities.emplace_back(bank.priority(frame.id), frame.id);
    std::sort(priorities.begin(), priorities.end());
    for (size_t i = 0; i < k - anchors; ++i)
      check(bank.representatives().count(priorities[i].second), "Representative is not Bottom-K");
    check(!bank.update(pool), "Unchanged history changed bank membership");
  }
  check(bank.anchors().count(1), "An old useful anchor expired with time");
  // Anchor removal, changed associations and quota shrink must remain reproducible.
  pool.erase(pool.begin(), pool.begin() + 80);
  for (auto& frame : pool) frame.mappoints = {frame.id % 9};
  bank.update(pool); repeat.update(pool);
  check(bank.members() == repeat.members(), "Deletion/association changes broke reproducibility");
  for (auto id : bank.members()) check(id > 80, "Deleted keyframe remains in bank");
  bank.update({});
  check(bank.members().empty() && bank.anchors().empty(), "Empty history retained bank members");
}
void greedyAndReplacement() {
  GlobalGIViewBank bank(GlobalGIConfig{});
  bank.update({{1, {1, 2, 3, 4}}, {2, {3, 4, 5, 6}}, {3, {7, 8, 9}},
               {4, {1}}, {5, {1}}, {6, {1}}, {7, {1}}, {8, {1}}, {9, {1}}, {10, {1}}});
  check(bank.anchors() == std::set<uint64_t>({1, 2, 3}), "Greedy maximum coverage chose wrong anchors");
  check(bank.coverageRatio() == 1.0, "Coverage count includes duplicates or misses points");
  GlobalGIViewBank streaming(GlobalGIConfig{});
  std::vector<GlobalGIKeyframe> pool{{1, {1, 2, 3}}, {2, {4, 5}},
                                    {3, {1}}, {4, {1}}, {5, {1}}};
  streaming.update(pool);
  pool.push_back({6, {6, 7, 8, 9}});
  streaming.update(pool);
  check(streaming.anchors() == std::set<uint64_t>({1, 6}), "Full anchor bank did not choose best strict-gain replacement");
  // No strict gain: a changed candidate with equal coverage cannot displace an anchor.
  pool[1].mappoints = {6, 7, 8, 9};
  streaming.update(pool);
  check(streaming.anchors() == std::set<uint64_t>({1, 6}), "Equal coverage replaced a stable anchor");
}
void coverageAccounting() {
  GlobalGIViewBank bank(GlobalGIConfig{});
  std::mt19937 rng(7);
  for (int iteration = 0; iteration < 400; ++iteration) {
    std::vector<GlobalGIKeyframe> pool;
    std::set<uint64_t> all, covered;
    for (uint64_t id = 1; id <= 40; ++id) {
      if (rng() % 5 == 0) continue;
      GlobalGIKeyframe frame{id, {}};
      for (int p = 0; p < 15; ++p) frame.mappoints.push_back(rng() % 90);
      all.insert(frame.mappoints.begin(), frame.mappoints.end());
      pool.push_back(std::move(frame));
    }
    bank.update(pool);
    for (const auto& frame : pool)
      if (bank.anchors().count(frame.id)) covered.insert(frame.mappoints.begin(), frame.mappoints.end());
    check(std::abs(bank.coverageRatio() - double(covered.size()) / all.size()) < 1e-12,
          "Incremental coverage counters differ from independent set-union reference");
    check(bank.members().size() == (3 * pool.size() + 9) / 10, "Random update broke bank capacity");
  }
  GlobalGIConfig cap; cap.max_views = 3;
  GlobalGIViewBank limited(cap);
  std::vector<GlobalGIKeyframe> pool;
  for (uint64_t id = 0; id < 100; ++id) pool.push_back({id, {id}});
  limited.update(pool);
  check(limited.members().size() == 3, "Explicit experimental capacity cap was ignored");
}
void checkEquivalent(const GlobalGIViewBank& bank, const ReferenceGlobalGIViewBank& reference) {
  check(bank.members() == reference.members(), "Optimized bank members differ from original policy");
  check(bank.anchors() == reference.anchors(), "Optimized anchors differ from original policy");
  check(bank.representatives() == reference.representatives(), "Optimized Bottom-K differs from original policy");
  check(bank.target() == reference.target() && bank.totalKeyframes() == reference.totalKeyframes(),
        "Optimized bank quota differs from original policy");
  check(bank.coverageRatio() == reference.coverageRatio(), "Optimized coverage differs from original policy");
}
void differentialPolicy() {
  for (unsigned scenario = 0; scenario < 8; ++scenario) {
    GlobalGIConfig config;
    config.view_ratio = scenario < 4 ? 0.3 : 0.7;
    config.coverage_anchor_fraction = std::vector<double>{0.0, 0.25, 0.8, 1.0}[scenario % 4];
    config.max_views = scenario == 5 ? 3 : 0;
    config.sampling_seed = scenario * 977;
    GlobalGIViewBank bank(config);
    ReferenceGlobalGIViewBank reference(config);
    std::mt19937 rng(100 + scenario);
    for (unsigned iteration = 0; iteration < 250; ++iteration) {
      std::vector<GlobalGIKeyframe> pool;
      if (iteration % 83 != 0) {
        for (uint64_t slot = 0; slot < 32; ++slot) {
          if (rng() % 5 == 0) continue;
          const uint64_t id = slot == 31 ? std::numeric_limits<uint64_t>::max() : slot;
          GlobalGIKeyframe frame{id, {}};
          const unsigned points = rng() % 30;
          for (unsigned p = 0; p < points; ++p) frame.mappoints.push_back(rng() % 83);
          if (!frame.mappoints.empty()) frame.mappoints.push_back(frame.mappoints.front());
          pool.push_back(std::move(frame));
        }
      }
      std::shuffle(pool.begin(), pool.end(), rng);
      check(bank.update(pool) == reference.update(pool), "Optimized membership-change flag differs");
      checkEquivalent(bank, reference);
      std::reverse(pool.begin(), pool.end());
      check(!bank.update(pool) && !reference.update(pool), "Unchanged input triggered a bank update");
      checkEquivalent(bank, reference);
    }
  }
  std::cout << "Differential policy: 2000 changed/empty snapshots and 2000 unchanged snapshots passed\n";
}
void benchmarkPolicy() {
  using Clock = std::chrono::steady_clock;
  std::mt19937 rng(20261010);
  std::vector<GlobalGIKeyframe> initial;
  for (uint64_t id = 0; id < 243; ++id) {
    GlobalGIKeyframe frame{id, {}};
    for (unsigned p = 0; p < 1000; ++p) frame.mappoints.push_back(rng() % 20000);
    initial.push_back(std::move(frame));
  }
  for (const std::string mode : {"anchor_changes", "candidate_changes", "unchanged"}) {
    std::vector<double> old_times, new_times;
    for (unsigned repeat = 0; repeat < 3; ++repeat) {
      GlobalGIViewBank bank(GlobalGIConfig{});
      ReferenceGlobalGIViewBank reference(GlobalGIConfig{});
      auto pool = initial;
      bank.update(pool); reference.update(pool);
      checkEquivalent(bank, reference);
      for (unsigned step = 0; step < 10; ++step) {
        if (mode == "anchor_changes") {
          for (auto& frame : pool) frame.mappoints[step] = 20000 + step * 243 + frame.id;
        } else if (mode == "candidate_changes") {
          auto candidate = std::find_if(pool.begin(), pool.end(), [&](const GlobalGIKeyframe& frame) {
            return !reference.anchors().count(frame.id);
          });
          candidate->mappoints[step] = 30000 + step;
        }
        auto old_start = Clock::now();
        bool old_changed, new_changed;
        double old_ms, new_ms;
        // Alternate execution order to avoid always giving one implementation warmer data.
        if ((repeat + step) % 2 == 0) {
          old_changed = reference.update(pool);
          old_ms = std::chrono::duration<double, std::milli>(Clock::now() - old_start).count();
          const auto new_start = Clock::now();
          new_changed = bank.update(pool);
          new_ms = std::chrono::duration<double, std::milli>(Clock::now() - new_start).count();
        } else {
          new_changed = bank.update(pool);
          new_ms = std::chrono::duration<double, std::milli>(Clock::now() - old_start).count();
          const auto reference_start = Clock::now();
          old_changed = reference.update(pool);
          old_ms = std::chrono::duration<double, std::milli>(Clock::now() - reference_start).count();
        }
        check(old_changed == new_changed, "Benchmark membership-change flag differs");
        checkEquivalent(bank, reference);
        old_times.push_back(old_ms); new_times.push_back(new_ms);
      }
    }
    const auto mean = [](const std::vector<double>& times) {
      double sum = 0; for (double time : times) sum += time; return sum / times.size();
    };
    const auto p95 = [](std::vector<double> times) {
      std::sort(times.begin(), times.end());
      return times[static_cast<size_t>(std::ceil(times.size() * 0.95)) - 1];
    };
    std::cout << std::fixed << std::setprecision(4)
              << "Bank CPU benchmark mode=" << mode << " keyframes=243 points_per_frame=1000 samples="
              << old_times.size() << " old_mean_ms=" << mean(old_times) << " new_mean_ms=" << mean(new_times)
              << " old_p95_ms=" << p95(old_times) << " new_p95_ms=" << p95(new_times)
              << " speedup=" << mean(old_times) / mean(new_times) << " parity=passed\n";
  }
}
}
int main(int argc, char** argv) {
  if (argc == 2 && std::string(argv[1]) == "--benchmark") { benchmarkPolicy(); return 0; }
  quotasAndSampling(); greedyAndReplacement(); coverageAccounting(); differentialPolicy();
  std::cout << "GlobalGI bank: quota, greedy, strict replacement, Bottom-K, history and coverage tests passed\n";
}
