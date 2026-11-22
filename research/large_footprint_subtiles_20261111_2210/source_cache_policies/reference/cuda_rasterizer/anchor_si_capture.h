#pragma once

// Host-side scope is sampled at launch, never read by a device/worker thread.
namespace anchor_si {
bool captureEnabled();
class CaptureScope {
 public:
  explicit CaptureScope(bool enabled);
  ~CaptureScope();
 private:
  bool previous_;
};
}
