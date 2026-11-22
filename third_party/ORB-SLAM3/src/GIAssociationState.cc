#include "GIAssociationState.h"
#include <chrono>

namespace ORB_SLAM3 {
std::shared_timed_mutex GIAssociationState::mutex_;
std::atomic<uint64_t> GIAssociationState::bad_revision_{0};
std::atomic<uint64_t> GIAssociationState::generation_{1};
std::atomic<uint64_t> GIAssociationState::writer_wait_ns_{0}, GIAssociationState::tracking_wait_ns_{0};
thread_local unsigned GIAssociationState::write_depth_ = 0;
thread_local unsigned GIAssociationState::read_depth_ = 0;
thread_local bool GIAssociationState::tracking_ = false;

GIAssociationState::Write::Write() {
  assert(read_depth_ == 0);
  if (write_depth_++ == 0) {
    const auto start = std::chrono::steady_clock::now();
    mutex_.lock_shared();
    const auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start).count();
    writer_wait_ns_.fetch_add(ns, std::memory_order_relaxed);
    if (tracking_) tracking_wait_ns_.fetch_add(ns, std::memory_order_relaxed);
  }
}
GIAssociationState::Write::~Write() {
  if (--write_depth_ == 0) mutex_.unlock_shared();
}
GIAssociationState::Read::Read() {
  assert(write_depth_ == 0);
  if (read_depth_++ == 0) mutex_.lock();
}
GIAssociationState::Read::~Read() {
  if (--read_depth_ == 0) mutex_.unlock();
}
}  // namespace ORB_SLAM3
