#pragma once
#include "inference_backend.h"
#include <memory>

namespace mycartgs {
class ReferenceXnnpackBackend final : public InferenceBackend {
public:
    ReferenceXnnpackBackend();
    ~ReferenceXnnpackBackend() override;
    void Initialize(uint32_t threads = 2);
    void Bind(const SceneAsset& scene) override;
    const InferenceStatus& Status() const override { return status_; }
    void RunSelector(uint32_t rows,const float* input19,const float* noise2,float* probability2) override;
    void RunStudent(uint32_t rows,const float* input62,float* residual14) override;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    InferenceStatus status_;
};
}  // namespace mycartgs
