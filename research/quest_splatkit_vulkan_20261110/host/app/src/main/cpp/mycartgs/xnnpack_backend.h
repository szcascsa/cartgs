#pragma once
#include "inference_backend.h"
#include <memory>

namespace mycartgs {
class XnnpackBackend final : public InferenceBackend {
public:
    XnnpackBackend();
    ~XnnpackBackend() override;
    void Initialize(uint32_t threads = 2);
    void SetThreadCount(uint32_t threads);
    void Bind(const SceneAsset& scene) override;
    const InferenceStatus& Status() const override { return status_; }
    void RunSelector(uint32_t rows,const float* input19,const float* noise2,float* probability2) override;
    void RunStudent(uint32_t rows,const float* input62,float* residual14) override;
    void SetOptimizationFlags(uint32_t flags);
    uint32_t OptimizationFlags() const { return flags_; }
    float* SelectorInput(uint32_t rows) override;
    float* StudentInput(uint32_t rows) override;
    void ParallelFor(size_t range,size_t tile,RangeTask task,void* context) const override;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    InferenceStatus status_;
    uint32_t flags_=CpuOptimizationsDefault;
};
}  // namespace mycartgs
