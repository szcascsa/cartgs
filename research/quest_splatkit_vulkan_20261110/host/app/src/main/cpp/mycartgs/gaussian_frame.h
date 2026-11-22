#pragma once
#include "decode.h"
#include "scene_asset.h"
#include "inference_backend.h"
#include "strict_topk.h"
#include <array>
#include <cstdint>
#include <vector>

namespace mycartgs {
struct FrameCamera {
    // Row-major world-to-camera, OpenCV axes (right, down, forward).
    std::array<float,16> view{}, projection{};
    std::array<float,3> center{};
    uint32_t width=0,height=0;
    float fx=0,fy=0,cx=0,cy=0,nearPlane=.01f,farPlane=100.f;
};
struct ProjectedSplat { std::array<float,4> centerRadius{}, conicAlpha{}, color{}; };
struct TileRange { uint32_t begin=0,end=0; };
struct GaussianEye {
    uint32_t width=0,height=0;
    std::vector<ProjectedSplat> splats;
    std::vector<TileRange> ranges;
    std::vector<uint32_t> members;
    uint32_t projected=0,alphaPoints=0;
};
struct FrameStatistics {
    uint32_t candidates=0,eligible=0,k=0;
    double prepareMs=0,selectorMs=0,topkMs=0,studentMs=0,decodeMs=0;
    float maxXyz=0,maxScaleRatio=1;
};

class GaussianFrame {
    friend struct CpuProbe;
public:
    void SetOptimizationFlags(uint32_t flags);
    uint32_t OptimizationFlags() const { return flags_; }
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
    std::vector<uint8_t> visible_,anchorUsed_;
    std::vector<uint32_t> activeAnchors_,availableByRow_;
    std::vector<float> anchorFeature_;
    std::vector<std::array<float,2>> decodeDiagnostics_;
    std::vector<ScoreKey> keyWorkspace_;
    std::vector<uint32_t> topkWorkspace_;
    uint32_t flags_=CpuOptimizationsDefault;
    FrameStatistics stats_;
};
std::array<float,7> AnchorViewCode(const float* center,float scale,const FrameCamera& camera,float sceneScale);
bool AnchorVisible(const float* center,float scale,const FrameCamera& camera);
}  // namespace mycartgs
