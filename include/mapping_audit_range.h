#pragma once
#ifdef CARTGS_FUSED_MAPPING_CUDA
#include <nvToolsExt.h>
#include <cstdlib>
class MappingAuditRange {
 public:
  explicit MappingAuditRange(const char* name) : enabled_(enabled()) { if(enabled_)nvtxRangePushA(name); }
  ~MappingAuditRange(){if(enabled_)nvtxRangePop();}
 private:
  static bool enabled(){static const bool value=std::getenv("CARTGS_CUDA_PHASE2_TRACE")!=nullptr;return value;}
  bool enabled_;
};
#else
class MappingAuditRange { public: explicit MappingAuditRange(const char*) {} };
#endif
