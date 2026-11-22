#include "quest_runtime.h"
#include "xnnpack_backend.h"

#include <android/log.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sys/stat.h>
#include <sstream>
#include <stdexcept>

namespace mycartgs {

namespace {
std::array<float,9> Rotation(const std::array<float,4>& q) {
    const float x=q[0],y=q[1],z=q[2],w=q[3];
    return {1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),
            2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)};
}
}

const InferenceStatus& QuestRuntime::Network() const {
    static const InferenceStatus unloaded{InferenceState::Uninitialized,"Load a scene for XNNPACK ARM CPU","NOT_EXECUTED"};
    return backend_?backend_->Status():unloaded;
}

FrameCamera QuestRuntime::EyeCamera(const UiPose& eye,const std::array<float,4>& fov,uint32_t width,uint32_t height) {
        if(!calibrated_) {
            const auto initial=Rotation(head_.orientation);const auto& source=activeDescriptor_.viewRotation;
            for(uint32_t r=0;r<3;++r)for(uint32_t c=0;c<3;++c){float v=0;
                for(uint32_t j=0;j<3;++j)v+=source[r*3+j]*(j==0?1.f:-1.f)*initial[c*3+j];
                mapFromAppRotation_[r*3+c]=v;}
            for(uint32_t r=0;r<3;++r){float v=activeDescriptor_.viewPosition[r];
                for(uint32_t c=0;c<3;++c)v-=mapFromAppRotation_[r*3+c]*head_.position[c];mapFromAppTranslation_[r]=v;}
            calibrated_=true;
        }
        FrameCamera camera;camera.width=width;camera.height=height;
        const auto eyeRotation=Rotation(eye.orientation);
        std::array<float,9> c2w{};
        for(uint32_t r=0;r<3;++r){camera.center[r]=mapFromAppTranslation_[r];
            for(uint32_t c=0;c<3;++c){camera.center[r]+=mapFromAppRotation_[r*3+c]*eye.position[c];
                for(uint32_t j=0;j<3;++j)c2w[r*3+c]+=mapFromAppRotation_[r*3+j]*eyeRotation[j*3+c]*(c==0?1.f:-1.f);}}
        for(uint32_t r=0;r<3;++r){for(uint32_t c=0;c<3;++c)camera.view[r*4+c]=c2w[c*3+r];
            for(uint32_t c=0;c<3;++c)camera.view[r*4+3]-=camera.view[r*4+c]*camera.center[c];}camera.view[15]=1;
        const float left=std::tan(fov[0]),right=std::tan(fov[1]),up=std::tan(fov[2]),down=std::tan(fov[3]);
        camera.fx=width/(right-left);camera.fy=height/(up-down);
        camera.cx=-left*camera.fx-.5f;camera.cy=up*camera.fy-.5f;
        std::array<float,16> projection{};projection[0]=2.f/(right-left);projection[2]=-(right+left)/(right-left);
        projection[5]=2.f/(up-down);projection[6]=(up+down)/(up-down);
        projection[10]=camera.farPlane/(camera.farPlane-camera.nearPlane);
        projection[11]=-camera.farPlane*camera.nearPlane/(camera.farPlane-camera.nearPlane);projection[14]=1;
        for(uint32_t r=0;r<4;++r)for(uint32_t c=0;c<4;++c)for(uint32_t j=0;j<4;++j)
            camera.projection[r*4+c]+=projection[r*4+j]*camera.view[j*4+c];
        return camera;
}

bool QuestRuntime::PrepareEye(const UiPose& eye,const std::array<float,4>& fov,uint32_t width,uint32_t height) {
    if(!rendering_ || !scene_ || menuVisible_) return false;
    const auto cpuStart=std::chrono::steady_clock::now();
    try {
        const FrameCamera camera=EyeCamera(eye,fov,width,height);
        // Fresh input preparation for each predicted eye pose. No previous-view mask/output cache.
        frame_.Prepare(camera,*backend_,controls_.ratio,controls_.mode==BudgetMode::FixedK?controls_.fixedK:0,
                        controls_.mode==BudgetMode::Full);
        const auto projectStart=std::chrono::steady_clock::now();frame_.Project(camera,eye_);
        const auto cpuEnd=std::chrono::steady_clock::now();
        telemetry_.projectMs=std::chrono::duration<double,std::milli>(cpuEnd-projectStart).count();
        telemetry_.cpuTotalMs=std::chrono::duration<double,std::milli>(cpuEnd-cpuStart).count();
        const auto& stats=frame_.Statistics();
        telemetry_.inputN=stats.candidates;telemetry_.selectedK=stats.k;
        telemetry_.requestedK=controls_.mode==BudgetMode::Full?stats.candidates:
            controls_.mode==BudgetMode::FixedK?std::min(stats.candidates,controls_.fixedK):
            uint32_t(std::floor(double(stats.candidates)*controls_.ratio));
        telemetry_.prepareMs=stats.prepareMs;telemetry_.selectorMs=stats.selectorMs;telemetry_.topkMs=stats.topkMs;
        telemetry_.studentMs=stats.studentMs;telemetry_.decodeMs=stats.decodeMs;
        if(telemetry_.selectedK && (renderedEyes_++<2 || renderedEyes_%300==0))__android_log_print(ANDROID_LOG_INFO,"MyCartGS",
            "GAUSSIAN_FRAME mode=%s K=%u projected=%u tiles=%zu instances=%zu selector_ms=%.3f student_ms=%.3f exec=%s",
            controls_.mode==BudgetMode::Full?"FULL_PREVIEW":"XNNPACK_LOD",stats.k,eye_.projected,eye_.ranges.size(),eye_.members.size(),
            stats.selectorMs,stats.studentMs,Network().execution.c_str());
        return true;
    } catch(const std::exception& error) {
        content_.message=error.what();content_.inferenceReady=false;rendering_=false;menuVisible_=true;
        content_.page=MenuPage::Details;menu_.Recenter();
        __android_log_print(ANDROID_LOG_ERROR,"MyCartGS","Gaussian frame: %s",error.what());
        return false;
    }
}

void QuestRuntime::RecordDrawTiming(double milliseconds) {
    telemetry_.renderMs=milliseconds;
    char text[120];
    std::snprintf(text,sizeof(text),"SEL %.2f  COMP %.2f  CPU %.1f  DRAW %.1f MS/EYE",telemetry_.selectorMs,
                  telemetry_.studentMs,telemetry_.cpuTotalMs,milliseconds);
    content_.timing=text;
    if(renderedEyes_<=2 || renderedEyes_%300==0)__android_log_print(ANDROID_LOG_INFO,"MyCartGS",
        "EYE_TIMING N=%u K=%u prepare_ms=%.3f selector_ms=%.3f topk_ms=%.3f student_ms=%.3f decode_ms=%.3f project_ms=%.3f cpu_total_ms=%.3f draw_submit_wait_ms=%.3f total_ms=%.3f execution=%s",
        telemetry_.inputN,telemetry_.selectedK,telemetry_.prepareMs,telemetry_.selectorMs,telemetry_.topkMs,
        telemetry_.studentMs,telemetry_.decodeMs,telemetry_.projectMs,telemetry_.cpuTotalMs,milliseconds,
        telemetry_.cpuTotalMs+milliseconds,Network().execution.c_str());
}

void QuestRuntime::StartPointBenchmark() {
    if(!scene_)return;
    benchmark_.Start(scene_->Identity().gaussianCount);
    benchmarkPreparedStage_=UINT32_MAX;benchmarkFrame_=0;benchmarkCameraReady_=false;
    benchmarkResumeWarmup_=0;
    content_.page=MenuPage::PointBenchmark;content_.benchmarkRunning=true;
    content_.benchmarkRows.clear();content_.benchmarkExport.clear();
    content_.message="ASCENDING THEN DESCENDING K. MASTER SUBSETS, NOT LOD FPS.";
    rendering_=false;menuVisible_=true;
    __android_log_print(ANDROID_LOG_INFO,"MyCartGS","POINT_BENCHMARK_START scene=%s N=%u rounds=%u warmup=%u measured=%u",
        scene_->Identity().slug.c_str(),scene_->Identity().gaussianCount,PointBenchmark::kRounds,PointBenchmark::kWarmup,PointBenchmark::kMeasured);
}

bool QuestRuntime::PreparePointBenchmark(const UiPose& eye,const std::array<float,4>& fov,uint32_t width,uint32_t height) {
    if(!benchmark_.Running() || !scene_ || !focused_ || benchmarkFrame_==telemetry_.frame)return false;
    benchmarkFrame_=telemetry_.frame; // Exactly one offscreen sample per XR frame, not one per eye.
    try {
        if(!benchmarkCameraReady_){benchmarkCamera_=EyeCamera(eye,fov,width,height);benchmarkCameraReady_=true;}
        if(benchmarkPreparedStage_!=benchmark_.Stage()) {
            const auto start=std::chrono::steady_clock::now();
            frame_.PreparePointBenchmark(benchmark_.K());frame_.Project(benchmarkCamera_,eye_);
            benchmarkMetrics_={};benchmarkMetrics_.projected=eye_.projected;
            benchmarkMetrics_.instances=eye_.members.size();benchmarkMetrics_.width=eye_.width;benchmarkMetrics_.height=eye_.height;
            for(const auto& range:eye_.ranges)benchmarkMetrics_.activeTiles+=range.end>range.begin;
            benchmarkMetrics_.prepareMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            benchmarkPreparedStage_=benchmark_.Stage();
        }
        return true;
    } catch(const std::exception& error){PointBenchmarkError(error.what());return false;}
}

void QuestRuntime::RecordPointBenchmarkTiming(double uploadMs,double gpuMs,double submitWaitMs) {
    if(!benchmark_.Running())return;
    if(benchmarkResumeWarmup_){--benchmarkResumeWarmup_;return;}
    benchmarkMetrics_.uploadMs=uploadMs;benchmarkMetrics_.gpuMs=gpuMs;benchmarkMetrics_.submitWaitMs=submitWaitMs;
    benchmark_.Record(benchmarkMetrics_);
    if(!benchmark_.Running())FinishPointBenchmark(false);
}

void QuestRuntime::UpdatePointBenchmarkMenu() {
    content_.benchmarkRunning=benchmark_.Running();
    if(benchmark_.Running()) {
        char progress[140];std::snprintf(progress,sizeof(progress),"%sROUND %u/2  CASE %u/%u  K %u  SAMPLE %u/18",
            focused_?"":"PAUSED / WEAR HEADSET. ",benchmark_.Round()+1,benchmark_.Stage()+1,benchmark_.TotalCases(),
            benchmark_.K(),benchmark_.Iteration()+1);content_.benchmarkProgress=progress;
    }
    const auto summaries=benchmark_.Summaries();double full=0;
    for(const auto& s:summaries)if(scene_&&s.k==scene_->Identity().gaussianCount)full=s.medianMs;
    content_.benchmarkRows.clear();
    for(const auto& s:summaries) {
        char line[110],speed[20];
        if(full>0&&s.medianMs>0)std::snprintf(speed,sizeof(speed),"%.2fX",full/s.medianMs);
        else std::snprintf(speed,sizeof(speed),"--");
        std::snprintf(line,sizeof(line),"%-8u %-10u %7.3f   %7.3f   %s",s.k,s.projected,s.medianMs,s.p95Ms,speed);
        content_.benchmarkRows.push_back(line);
    }
}

void QuestRuntime::FinishPointBenchmark(bool cancelled) {
    benchmark_.Cancel();content_.benchmarkRunning=false;
    content_.benchmarkProgress=cancelled?"CANCELLED / PARTIAL RESULTS":"COMPLETE / 30 MEASURED SAMPLES PER K";
    UpdatePointBenchmarkMenu();
    if(benchmark_.Samples().empty())return;
    try {
        if(benchmarkDirectory_.empty())throw std::runtime_error("Android output directory unavailable");
        const std::string directory=benchmarkDirectory_+"/benchmarks";mkdir(directory.c_str(),0700);
        const auto stamp=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        const std::string name="pointcount_"+std::to_string(stamp)+".csv",path=directory+"/"+name;
        std::ofstream out(path);out.precision(10);
        out<<"# scene="<<scene_->Identity().slug<<"\n# status="<<(cancelled?"cancelled":"complete")
           <<"\n# source=original_master_attributes_uid_hash_nested_subsets_no_selector_no_compensation"
           <<"\n# gpu=offscreen_raster_compute_and_barriers_only_no_present_no_menu"
           <<"\n# prepare_once_ms=repeated_case_metadata_not_per_sample_cpu_work"
           <<"\n# camera_fx_fy_cx_cy="<<benchmarkCamera_.fx<<','<<benchmarkCamera_.fy<<','<<benchmarkCamera_.cx<<','<<benchmarkCamera_.cy
           <<"\n# camera_view=";
        for(float x:benchmarkCamera_.view)out<<x<<',';
        out<<'\n'<<benchmark_.Csv();out.close();if(!out)throw std::runtime_error("Cannot write point benchmark CSV");
        content_.benchmarkExport="CSV SAVED: "+name;
        __android_log_print(ANDROID_LOG_INFO,"MyCartGS","POINT_BENCHMARK_EXPORT status=%s samples=%zu path=%s",
            cancelled?"cancelled":"complete",benchmark_.Samples().size(),path.c_str());
        for(const auto& s:benchmark_.Summaries())__android_log_print(ANDROID_LOG_INFO,"MyCartGS",
            "POINT_BENCHMARK_RESULT K=%u projected=%u samples=%u gpu_median_ms=%.4f gpu_p95_ms=%.4f",s.k,s.projected,s.samples,s.medianMs,s.p95Ms);
    } catch(const std::exception& error) {
        content_.benchmarkExport=std::string("CSV ERROR: ")+error.what();
        __android_log_print(ANDROID_LOG_ERROR,"MyCartGS","POINT_BENCHMARK_EXPORT_ERROR %s",error.what());
    }
}

void QuestRuntime::PointBenchmarkError(const std::string& error) {
    FinishPointBenchmark(true);content_.benchmarkProgress="BENCHMARK FAILED";
    content_.message=error;
    __android_log_print(ANDROID_LOG_ERROR,"MyCartGS","POINT_BENCHMARK_ERROR %s",error.c_str());
}

void QuestRuntime::Initialize(AAssetManager* assets, const std::string&, const std::string& outputDirectory) {
    Shutdown();
    assets_ = assets;
    content_ = MenuContent{};
    telemetry_ = RuntimeTelemetry{};
    renderedEyes_ = 0;
    menu_ = SceneMenu{};
    head_ = UiPose{};
    controllers_ = {};
    controls_ = RuntimeControls{};
    focused_ = false;
    rendering_ = false; menuVisible_ = true; calibrated_ = false;
    validationPreview_ = validationLod_ = false;
    benchmark_=PointBenchmark{};benchmarkAfterLoad_=false;benchmarkDirectory_=outputDirectory;
    try {
        catalog_ = LoadSceneCatalog(assets);
        for (const auto& scene : catalog_)
            content_.scenes.push_back({scene.label, scene.gaussianCount, scene.anchorCount});
    } catch (const std::exception& error) {
        catalog_.clear();
        content_.page = MenuPage::Error;
        content_.message = error.what();
        __android_log_print(ANDROID_LOG_ERROR, "MyCartGS", "Scene catalog: %s", error.what());
    }
}

void QuestRuntime::Shutdown() {
    benchmark_.Cancel();
    // Keep the Activity AssetManager alive until the only bounded loader task has exited.
    if (loadJob_.valid()) loadJob_.wait();
    loadJob_ = std::future<LoadResult>{};
    scene_.reset();
    backend_.reset();
    assets_ = nullptr;
}

void QuestRuntime::LoadSelected() {
    if (loadJob_.valid() || content_.selected >= catalog_.size()) return;
    const SceneDescriptor descriptor = catalog_[content_.selected];
    content_.page = MenuPage::Loading;
    content_.loadedLabel = descriptor.label;
    content_.message.clear();
    scene_.reset();  // At most one scene allocation, not an unbounded per-scene cache.
    backend_.reset(); rendering_=false; menuVisible_=true; calibrated_=false;
    telemetry_.inputN = telemetry_.selectedK = telemetry_.requestedK = 0;
    const auto assets = assets_;
    activeDescriptor_ = descriptor;
    loadJob_ = std::async(std::launch::async, [assets, descriptor]() {
        LoadResult result;
        const auto start = std::chrono::steady_clock::now();
        try {
            auto scene = std::make_unique<SceneAsset>();
            scene->Load(assets, descriptor);
            result.scene = std::move(scene);
            auto backend = std::make_unique<XnnpackBackend>();
            auto* cpu=backend.get();
            result.backend = std::move(backend);
            cpu->Initialize(2);
            try { result.backend->Bind(*result.scene); }
            catch (const std::exception& error) { result.error = error.what(); }
        } catch (const std::exception& error) { result.error = error.what(); }
        catch (...) { result.error = "Unknown scene loading error"; }
        result.milliseconds = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now()-start).count();
        return result;
    });
}

void QuestRuntime::ValidationLaunch(bool fullPreview, bool runLod, bool runPointBenchmark) {
    validationPreview_=fullPreview;
    validationLod_=runLod;
    benchmarkAfterLoad_=runPointBenchmark;
    LoadSelected();
}

void QuestRuntime::UpdateHeadPose(const std::array<float, 3>& position, const std::array<float, 4>& orientation) {
    head_ = {position, orientation};
}

void QuestRuntime::UpdateTrigger(uint32_t hand, float value) {
    if (hand < controllers_.size()) controllers_[hand].trigger = std::isfinite(value) ? value : 0;
}

void QuestRuntime::UpdatePointer(uint32_t hand, const UiPose& pose, bool tracked) {
    if (hand < controllers_.size()) { controllers_[hand].aim = pose; controllers_[hand].tracked = tracked; }
}

void QuestRuntime::OpenSceneMenu() {
    menuVisible_=true; rendering_=false;
    if(content_.page==MenuPage::PointBenchmark) {
        if(benchmark_.Running())FinishPointBenchmark(true);
        menu_.Recenter();return;
    }
    if (content_.page != MenuPage::Loading) content_.page = scene_?MenuPage::Details:MenuPage::Catalog;
    menu_.Recenter();
}

void QuestRuntime::BeginFrame() {
    ++telemetry_.frame;
    if (loadJob_.valid() && loadJob_.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
        auto result = loadJob_.get();
        telemetry_.assetMs = result.milliseconds;
        content_.loadMs = float(result.milliseconds);
        scene_ = std::move(result.scene);
        backend_ = std::move(result.backend);
        if (!scene_) {
            benchmarkAfterLoad_=false;
            content_.page = MenuPage::Error;
            content_.message = result.error;
            __android_log_print(ANDROID_LOG_ERROR, "MyCartGS", "Scene load failed: %s", result.error.c_str());
        } else {
            telemetry_.inputN = content_.loadedGaussians = scene_->Identity().gaussianCount;
            content_.loadedAnchors = scene_->Identity().anchorCount;
            content_.page = MenuPage::Details;
            frame_.Bind(*scene_);
            content_.message = result.error.empty()?Network().message:result.error;
            content_.inferenceReady = backend_ && backend_->Ready();
            if(validationPreview_) {
                controls_.mode=BudgetMode::Full;rendering_=true;menuVisible_=false;calibrated_=false;
                __android_log_print(ANDROID_LOG_INFO,"MyCartGS","VALIDATION_LAUNCH FULL_PREVIEW (explicit Intent extra)");
            } else if(validationLod_ && content_.inferenceReady) {
                controls_.mode=BudgetMode::Ratio;rendering_=true;menuVisible_=false;calibrated_=false;
                __android_log_print(ANDROID_LOG_INFO,"MyCartGS","VALIDATION_LAUNCH XNNPACK_LOD ratio=%.2f",controls_.ratio);
            }
            __android_log_print(ANDROID_LOG_INFO, "MyCartGS", "Scene verified: %s N=%u A=%u load_ms=%.2f",
                scene_->Identity().slug.c_str(), content_.loadedGaussians, content_.loadedAnchors, result.milliseconds);
            if(benchmarkAfterLoad_){benchmarkAfterLoad_=false;StartPointBenchmark();}
        }
    }

    if (!menuVisible_) return;
    const MenuAction action = menu_.Update(head_, controllers_, content_, focused_);
    switch (action.command) {
        case MenuCommand::Select: if (action.value < catalog_.size()) content_.selected = action.value; break;
        case MenuCommand::Load: LoadSelected(); break;
        case MenuCommand::Previous: if (content_.listPage > 0) --content_.listPage; break;
        case MenuCommand::Next:
            if ((content_.listPage+1)*kScenesPerPage < catalog_.size()) ++content_.listPage;
            break;
        case MenuCommand::Back:
            if(content_.page==MenuPage::PointBenchmark) {
                if(benchmark_.Running())FinishPointBenchmark(true);
                else content_.page=MenuPage::Details;
            } else content_.page=MenuPage::Catalog;
            rendering_=false;break;
        case MenuCommand::PointBenchmark:
            validationPreview_=validationLod_=false;
            if(content_.page==MenuPage::Catalog){benchmarkAfterLoad_=true;LoadSelected();}
            else if(scene_&&!benchmark_.Running())StartPointBenchmark();
            break;
        case MenuCommand::Preview:
            controls_.mode=BudgetMode::Full; rendering_=true; menuVisible_=false; calibrated_=false; break;
        case MenuCommand::RunLod:
            if(backend_ && backend_->Ready()) {
                if(controls_.mode==BudgetMode::Full) controls_.mode=BudgetMode::Ratio;
                rendering_=true; menuVisible_=false; calibrated_=false;
            }
            break;
        case MenuCommand::Ratio:
            controls_.mode = BudgetMode::Ratio; controls_.ratio = float(action.value)*.01f; break;
        case MenuCommand::FixedK:
            controls_.mode = BudgetMode::FixedK; controls_.fixedK = action.value; break;
        case MenuCommand::None: break;
    }
    const uint32_t n = telemetry_.inputN;
    telemetry_.requestedK = controls_.mode == BudgetMode::Ratio
                               ? std::min(n, uint32_t(std::floor(double(n)*controls_.ratio)))
                               : std::min(n, controls_.fixedK);
    content_.fixedBudget = controls_.mode == BudgetMode::FixedK;
    content_.budget = controls_.fixedK;
    content_.ratioPercent = uint32_t(std::lround(controls_.ratio*100));
    if(content_.page==MenuPage::PointBenchmark)UpdatePointBenchmarkMenu();
    if(menuVisible_) menu_.Build(content_);
    else { menu_ = SceneMenu{}; }
}

std::string QuestRuntime::StatusLine() const {
    std::ostringstream stream;
    stream << "page=" << int(content_.page) << " scenes=" << catalog_.size() << " N=" << telemetry_.inputN
           << " requestedK=" << telemetry_.requestedK << " selectedK=" << telemetry_.selectedK
           << " backend=" << Network().message << " execution=" << Network().execution;
    return stream.str();
}

QuestRuntime& Runtime() { static QuestRuntime runtime; return runtime; }
void InitializeRuntime(AAssetManager* assets,const std::string& directory,const std::string& outputDirectory) { Runtime().Initialize(assets,directory,outputDirectory); }
void ShutdownRuntime() { Runtime().Shutdown(); }
void RuntimeHeadPose(const float position[3], const float orientation[4]) {
    Runtime().UpdateHeadPose({position[0],position[1],position[2]},
                             {orientation[0],orientation[1],orientation[2],orientation[3]});
}
void RuntimeTrigger(uint32_t hand, float value) { Runtime().UpdateTrigger(hand,value); }
void RuntimePointer(uint32_t hand, const float position[3], const float orientation[4], bool tracked) {
    Runtime().UpdatePointer(hand, {{position[0],position[1],position[2]},
                                  {orientation[0],orientation[1],orientation[2],orientation[3]}}, tracked);
}
void RuntimeFocused(bool focused) { Runtime().SetFocused(focused); }
void RuntimeOpenSceneMenu() { Runtime().OpenSceneMenu(); }
void RuntimeBeginFrame() { Runtime().BeginFrame(); }
std::string RuntimeStatusLine() { return Runtime().StatusLine(); }

}  // namespace mycartgs
