#pragma once
#include "gaussian_frame.h"
#include "decode.h"
#include "scene_asset.h"
#include "inference_backend.h"
#include <array>
#include <cstdint>
#include <vector>

namespace mycartgs {
class ReferenceGaussianFrame {
    friend struct CpuProbe;
public:
    void Bind(const SceneAsset& scene);
    // Full bypass is an explicit user mode. It never runs an alternative MLP.
    void Prepare(const FrameCamera& camera,InferenceBackend& backend,float ratio,uint32_t fixedK,bool full);
    void PreparePointBenchmark(uint32_t k);
    void Project(const FrameCamera& camera,GaussianEye& output) const;
    const FrameStatistics& Statistics() const { return stats_; }
    const std::vector<uint32_t>& SelectedRows() const { return selected_; }
    const std::vector<uint32_t>& CandidateRows() const { return candidates_; }
    const std::vector<DecodedAttribute>& Decoded() const { return decoded_; }
private:
    const SceneAsset* scene_=nullptr;
    std::vector<MasterAttribute> master_;
    std::vector<uint32_t> candidates_,selected_;
    std::vector<uint32_t> benchmarkOrder_;
    std::vector<DecodedAttribute> decoded_;
    std::vector<float> views_,selector_,noise_,probability_,student_,residual_;
    std::vector<float> scores_;
    std::vector<uint64_t> uids_;
    std::vector<uint8_t> eligible_;
    FrameStatistics stats_;
};
std::array<float,7> ReferenceAnchorViewCode(const float* center,float scale,const FrameCamera& camera,float sceneScale);
bool ReferenceAnchorVisible(const float* center,float scale,const FrameCamera& camera);
}  // namespace mycartgs
