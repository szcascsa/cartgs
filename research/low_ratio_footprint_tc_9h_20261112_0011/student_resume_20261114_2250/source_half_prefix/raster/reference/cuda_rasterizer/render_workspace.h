#pragma once
#include <cstdint>
// Thread-local, opt-in. No change to rasterizer stream or saved-state ownership.
namespace renderer_workspace {
struct Stats { int64_t hits=0, misses=0, growths=0, retained_bytes=0; };
bool enabled();
void setEnabled(bool value);
Stats stats();
void clearIdle();
class Scope {
  bool previous_;
 public:
  explicit Scope(bool value):previous_(enabled()){setEnabled(value);}
  ~Scope(){setEnabled(previous_);}
};
}
