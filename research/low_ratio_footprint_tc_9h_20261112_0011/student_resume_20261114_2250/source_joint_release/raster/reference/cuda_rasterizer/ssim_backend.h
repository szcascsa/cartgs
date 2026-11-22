#pragma once
#include <vector>
namespace ssim_backend {
bool optimized();
bool cached();
void setCached(bool value);
void setOptimized(bool value);
// numRegs, staticSharedBytes, maxThreadsPerBlock, residentBlocksPerSM.
std::vector<int> attributes(bool optimized);
class Scope {
  bool previous_;
 public:
  explicit Scope(bool value):previous_(optimized()){setOptimized(value);}
  ~Scope(){setOptimized(previous_);}
};
class CacheScope {
  bool previous_;
 public:
  explicit CacheScope(bool value):previous_(cached()){setCached(value);}
  ~CacheScope(){setCached(previous_);}
};
}
