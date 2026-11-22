#pragma once
#include "inference_backend.h"
#include <cstdint>
#include <memory>
#include <string>

namespace mycartgs {
class SceneAsset;
using QnnState = InferenceState;
using QnnStatus = InferenceStatus;

class QnnBackend : public InferenceBackend {
public:
    QnnBackend();
    ~QnnBackend();
    QnnStatus Initialize(const std::string& nativeLibraryDirectory, bool signedPd = false);
    void Bind(const SceneAsset& scene);
    bool Ready() const { return status_.state == QnnState::Ready; }
    const QnnStatus& Status() const { return status_; }
    // Fixed 1024-row HTP graphs; final incomplete batch is padded and trimmed, never changes K.
    void RunSelector(uint32_t rows, const float* input19, const float* noise2, float* probability2);
    void RunStudent(uint32_t rows, const float* input62, float* residual14);
    void Close();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    QnnStatus status_;
};
}  // namespace mycartgs
