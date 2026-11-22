#pragma once
#include <cstdint>
#include <cstddef>
#include <stdexcept>

// Mapping counts exclude Full importance scans, evaluation renders and Replay.
struct RealLodSchedule {
  enum class Phase { OnlineFull, TailFull, Lod, Complete };
  bool enabled = false;
  Phase phase = Phase::OnlineFull;
  std::uint64_t online_steps = 0, tail_steps = 0, lod_steps = 0, target_steps = 0;
  void completeFull() {
    if (phase == Phase::OnlineFull) ++online_steps;
    else if (phase == Phase::TailFull) ++tail_steps;
    else throw std::logic_error("Full update outside RealLoD Full phases");
  }
  void endOnline() { if (phase == Phase::OnlineFull) phase = Phase::TailFull; }
  void beginLod() {
    if (phase != Phase::TailFull) throw std::logic_error("RealLoD must finish online and Full tail first");
    target_steps = online_steps;
    phase = target_steps ? Phase::Lod : Phase::Complete;
  }
  std::size_t ratioIndex(std::size_t count) const {
    if (!count) throw std::invalid_argument("RealLoD needs training ratios");
    return lod_steps % count;
  }
  bool mixedRound(std::size_t count) const {
    if (!count) throw std::invalid_argument("MixedRatio needs training ratios");
    return (lod_steps / count) % 2 == 1;
  }
  bool viewResidualRound(std::size_t count) const {
    if (!count) throw std::invalid_argument("View residual needs training ratios");
    return (lod_steps / count) % 2 == 1;
  }
  void completeLod() {
    if (phase != Phase::Lod) throw std::logic_error("LoD update outside RealLoD LoD phase");
    if (++lod_steps == target_steps) phase = Phase::Complete;
  }
  bool optimizingLod() const { return enabled && phase == Phase::Lod; }
  const char* name() const {
    switch (phase) {
      case Phase::OnlineFull: return "online_full";
      case Phase::TailFull: return "tail_full";
      case Phase::Lod: return "lod";
      default: return "complete";
    }
  }
};
