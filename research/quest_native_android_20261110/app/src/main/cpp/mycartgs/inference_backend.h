#pragma once
#include <cstdint>
#include <string>

namespace mycartgs {
class SceneAsset;
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
};
}  // namespace mycartgs
