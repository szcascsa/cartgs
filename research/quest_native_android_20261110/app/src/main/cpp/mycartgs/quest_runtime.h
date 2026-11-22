#pragma once

#include "inference_backend.h"
#include "scene_asset.h"
#include "scene_menu.h"
#include "gaussian_frame.h"
#include "point_benchmark.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <future>
#include <memory>
#include <string>

struct AAssetManager;

namespace mycartgs {

enum class BudgetMode { Ratio, FixedK, Full };

struct RuntimeControls {
    BudgetMode mode = BudgetMode::Ratio;
    float ratio = 0.1f;
    uint32_t fixedK = 1000;
};

struct RuntimeTelemetry {
    double assetMs = 0;
    double prepareMs = 0;
    double selectorMs = 0;
    double topkMs = 0;
    double studentMs = 0;
    double decodeMs = 0;
    double renderMs = 0;
    double projectMs = 0, cpuTotalMs = 0;
    uint32_t inputN = 0;
    uint32_t requestedK = 0;
    uint32_t selectedK = 0;
    uint64_t frame = 0;
};

class QuestRuntime {
public:
    void Initialize(AAssetManager* assets, const std::string& nativeLibraryDirectory, const std::string& outputDirectory = {});
    void Shutdown();
    void UpdateHeadPose(const std::array<float, 3>& position, const std::array<float, 4>& orientation);
    void UpdateTrigger(uint32_t hand, float value);
    void UpdatePointer(uint32_t hand, const UiPose& pose, bool tracked);
    void SetFocused(bool focused) {
        if(focused&&!focused_&&benchmark_.Running())benchmarkResumeWarmup_=PointBenchmark::kWarmup;
        focused_ = focused;
    }
    void OpenSceneMenu();
    void BeginFrame();
    void ValidationLaunch(bool fullPreview, bool runLod = false, bool runPointBenchmark = false);

    const SceneMenu& Menu() const { return menu_; }
    const InferenceStatus& Network() const;
    const RuntimeControls& Controls() const { return controls_; }
    const RuntimeTelemetry& Telemetry() const { return telemetry_; }
    std::string StatusLine() const;
    AAssetManager* Assets() const { return assets_; }
    bool PrepareEye(const UiPose& eye, const std::array<float,4>& fov, uint32_t width, uint32_t height);
    void RecordDrawTiming(double milliseconds);
    const GaussianEye& Eye() const { return eye_; }
    bool PreparePointBenchmark(const UiPose& eye,const std::array<float,4>& fov,uint32_t width,uint32_t height);
    void RecordPointBenchmarkTiming(double uploadMs,double gpuMs,double submitWaitMs);
    void PointBenchmarkError(const std::string& error);

private:
    struct LoadResult {
        std::unique_ptr<SceneAsset> scene;
        std::string error;
        std::unique_ptr<InferenceBackend> backend;
        double milliseconds = 0;
    };
    void LoadSelected();
    FrameCamera EyeCamera(const UiPose& eye,const std::array<float,4>& fov,uint32_t width,uint32_t height);
    void StartPointBenchmark();
    void FinishPointBenchmark(bool cancelled);
    void UpdatePointBenchmarkMenu();
    std::unique_ptr<SceneAsset> scene_;
    AAssetManager* assets_ = nullptr;
    std::vector<SceneDescriptor> catalog_;
    std::future<LoadResult> loadJob_;
    std::unique_ptr<InferenceBackend> backend_;
    GaussianFrame frame_;
    GaussianEye eye_;
    SceneDescriptor activeDescriptor_;
    std::array<float,9> mapFromAppRotation_{};
    std::array<float,3> mapFromAppTranslation_{};
    bool calibrated_ = false;
    bool rendering_ = false;
    bool menuVisible_ = true;
    bool validationPreview_ = false;
    bool validationLod_ = false;
    PointBenchmark benchmark_;
    FrameCamera benchmarkCamera_;
    PointBenchmarkMetrics benchmarkMetrics_;
    uint32_t benchmarkPreparedStage_ = UINT32_MAX;
    uint32_t benchmarkResumeWarmup_ = 0;
    uint64_t benchmarkFrame_ = 0;
    bool benchmarkCameraReady_ = false,benchmarkAfterLoad_ = false;
    std::string benchmarkDirectory_;
    SceneMenu menu_;
    MenuContent content_;
    RuntimeControls controls_;
    RuntimeTelemetry telemetry_;
    UiPose head_;
    std::array<UiController, 2> controllers_{};
    bool focused_ = false;
    uint64_t renderedEyes_ = 0;
};

QuestRuntime& Runtime();
void InitializeRuntime(AAssetManager* assets, const std::string& nativeLibraryDirectory, const std::string& outputDirectory = {});
void ShutdownRuntime();
void RuntimeHeadPose(const float position[3], const float orientation[4]);
void RuntimeTrigger(uint32_t hand, float value);
void RuntimePointer(uint32_t hand, const float position[3], const float orientation[4], bool tracked);
void RuntimeFocused(bool focused);
void RuntimeOpenSceneMenu();
void RuntimeBeginFrame();
std::string RuntimeStatusLine();

}  // namespace mycartgs
