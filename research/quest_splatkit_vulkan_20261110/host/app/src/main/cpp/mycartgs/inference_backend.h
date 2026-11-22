#pragma once
#include <cstdint>
#include <cstddef>
#include <string>
#include <type_traits>

namespace mycartgs {
class SceneAsset;
enum CpuOptimization : uint32_t {
    CpuParallelPrepare=1, CpuSharedAnchorFeature=2, CpuParallelDecode=4,
    CpuReusableTopK=8, CpuDirectInput=16, CpuStableBatchPlans=32,
    CpuHistogramTopK=64, CpuLargeBatch=128, CpuOptimizationsAll=255,
    CpuOptimizationsDefault=CpuParallelPrepare|CpuParallelDecode|CpuReusableTopK|CpuDirectInput|CpuHistogramTopK|CpuLargeBatch
};
enum class InferenceState { Uninitialized, LibrariesMissing, DeviceError, GraphError, Ready };
struct InferenceStatus {
    InferenceState state = InferenceState::Uninitialized;
    std::string message;
    std::string execution = "NOT_EXECUTED";
    uint32_t soc = 0, arch = 0;
    uint64_t selectorCalls = 0, studentCalls = 0;
    uint64_t errorCode = 0;
    int rpcErrno = 0, secureRpcErrno = 0;
    bool transportAccessDenied = false;
    uint32_t threads = 0;
    uint64_t workspaceBytes = 0;
};
class InferenceBackend {
public:
    virtual ~InferenceBackend() = default;
    virtual void Bind(const SceneAsset& scene) = 0;
    virtual const InferenceStatus& Status() const = 0;
    bool Ready() const { return Status().state == InferenceState::Ready; }
    virtual void RunSelector(uint32_t rows,const float* input19,const float* noise2,float* probability2) = 0;
    virtual void RunStudent(uint32_t rows,const float* input62,float* residual14) = 0;
    virtual float* SelectorInput(uint32_t) { return nullptr; }
    virtual float* StudentInput(uint32_t) { return nullptr; }
    using RangeTask=void (*)(void*,size_t,size_t);
    virtual void ParallelFor(size_t range,size_t,RangeTask task,void* context) const {
        if(range)task(context,0,range);
    }
};
template<class Function>void CpuParallelFor(const InferenceBackend& backend,size_t range,size_t tile,Function&& function) {
    using Task=std::remove_reference_t<Function>;
    backend.ParallelFor(range,tile,[](void* context,size_t begin,size_t count){
        (*static_cast<Task*>(context))(begin,count);
    },const_cast<void*>(static_cast<const void*>(&function)));
}
}  // namespace mycartgs
