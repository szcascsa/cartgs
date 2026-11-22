#pragma once
#include <array>
#include <chrono>
#include <memory>
#include <ostream>
#include <vector>

class ProtectionProfiler {
 public:
  enum Stage { MaturityUpdate, MaturitySnapshot, SelectorAllRows, Total,
               TFForward, TFRegularization, TFBackward, TFOptimizer, MaturityDiagnostics, Count };
  class Scope {
   public:
    Scope(ProtectionProfiler* owner, Stage stage, bool cuda);
    ~Scope();
    Scope(const Scope&) = delete;
    Scope& operator=(const Scope&) = delete;
   private:
    ProtectionProfiler* owner_;
    Stage stage_;
    std::chrono::steady_clock::time_point start_;
    void* events_ = nullptr;
  };
  ProtectionProfiler();
  ~ProtectionProfiler();
  void beginIteration();
  void endIteration();
  void report(std::ostream& out, double metadata_mb);
  void reportSelector(std::ostream& out, float ratio);
 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
