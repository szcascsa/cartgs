#pragma once

#include <atomic>
#include <cassert>
#include <cstdint>
#include <shared_mutex>

namespace ORB_SLAM3 {
// Association writers share this barrier. A GI snapshot excludes all writers,
// including nested bad/replace transactions, but never takes mMutexMapUpdate.
class GIAssociationState {
 public:
  class Write {
   public:
    Write();
    ~Write();
    Write(const Write&) = delete;
    Write& operator=(const Write&) = delete;
  };
  class Read {
   public:
    Read();
    ~Read();
    Read(const Read&) = delete;
    Read& operator=(const Read&) = delete;
  };
  class TrackingScope {
   public:
    TrackingScope() : previous_(tracking_) { tracking_ = true; }
    ~TrackingScope() { tracking_ = previous_; }
   private:
    bool previous_;
  };
  static uint64_t writerWaitNs() { return writer_wait_ns_.load(std::memory_order_relaxed); }
  static uint64_t trackingWaitNs() { return tracking_wait_ns_.load(std::memory_order_relaxed); }
  static uint64_t badRevision() { return bad_revision_.load(std::memory_order_relaxed); }
  static void invalidateBadPoints() { bad_revision_.fetch_add(1, std::memory_order_relaxed); }
  static uint64_t newGeneration() { return generation_.fetch_add(1, std::memory_order_relaxed); }
  static void assertReading() { assert(read_depth_ != 0); }
 private:
  static std::shared_timed_mutex mutex_;
  static std::atomic<uint64_t> bad_revision_, generation_;
  static std::atomic<uint64_t> writer_wait_ns_, tracking_wait_ns_;
  static thread_local unsigned write_depth_, read_depth_;
  static thread_local bool tracking_;
};
}  // namespace ORB_SLAM3
