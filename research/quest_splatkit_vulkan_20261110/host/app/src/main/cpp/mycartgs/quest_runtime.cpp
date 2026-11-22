#include "quest_runtime.h"
#include "xnnpack_backend.h"
#include "splatkit_gaussians.h"
#include "sha256.h"

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

bool QuestRuntime::PrepareLeftEye(const UiPose& eye,const std::array<float,4>& fov,uint32_t width,uint32_t height) {
    if(cpuBenchmark_.Running()){TickCpuBenchmark(eye,fov,width,height);return false;}
    if(!rendering_ || !scene_ || menuVisible_) return false;
    if(preparedThisFrame_ && preparedFrame_==telemetry_.frame)return true;
    const auto cpuStart=std::chrono::steady_clock::now();
    try {
        const FrameCamera camera=EyeCamera(eye,fov,width,height);
        const auto selectorBefore=Network().selectorCalls,studentBefore=Network().studentCalls;
        // Only the current frame's located LEFT eye defines selection and compensation.
        frame_.Prepare(camera,*backend_,controls_.ratio,controls_.mode==BudgetMode::FixedK?controls_.fixedK:0,
                        controls_.mode==BudgetMode::Full);
        renderCamera_=camera;
        const bool full=controls_.mode==BudgetMode::Full;
        if(!full||!previousFull_)++renderRevision_;
        previousFull_=full;
        const auto projectStart=std::chrono::steady_clock::now();
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
        telemetry_.selectorCalls=uint32_t(Network().selectorCalls-selectorBefore);
        telemetry_.studentCalls=uint32_t(Network().studentCalls-studentBefore);
        if(telemetry_.selectorCalls>1||telemetry_.studentCalls>1)throw std::runtime_error("More than one network inference in a shared stereo frame");
        preparedSelectorCalls_=Network().selectorCalls;preparedStudentCalls_=Network().studentCalls;
        preparedFrame_=telemetry_.frame;preparedThisFrame_=true;
        if(telemetry_.frame<8||telemetry_.frame%120==0){
            const auto& rows=frame_.SelectedRows();const auto& values=frame_.Decoded();
            preparedRowsHash_=Sha256Hex(reinterpret_cast<const uint8_t*>(rows.data()),rows.size()*sizeof(uint32_t));
            preparedAttributesHash_=Sha256Hex(reinterpret_cast<const uint8_t*>(values.data()),values.size()*sizeof(DecodedAttribute));
            __android_log_print(ANDROID_LOG_INFO,"SplatKitStereo",
                "LEFT_INFERENCE frame=%llu revision=%llu mode=%s N=%u K=%u selector_calls=%u student_calls=%u cpu_ms=%.3f rows=%s attributes=%s",
                (unsigned long long)telemetry_.frame,(unsigned long long)renderRevision_,controls_.mode==BudgetMode::Full?"FULL":"LOD",stats.candidates,stats.k,
                telemetry_.selectorCalls,telemetry_.studentCalls,telemetry_.cpuTotalMs,preparedRowsHash_.c_str(),preparedAttributesHash_.c_str());
        }
        return true;
    } catch(const std::exception& error) {
        preparedThisFrame_=false;content_.message=error.what();content_.inferenceReady=false;rendering_=false;menuVisible_=true;
        content_.page=MenuPage::Details;menu_.Recenter();
        __android_log_print(ANDROID_LOG_ERROR,"MyCartGS","Gaussian frame: %s",error.what());
        return false;
    }
}

bool QuestRuntime::PrepareEye(const UiPose& eye,const std::array<float,4>& fov,uint32_t width,uint32_t height) {
    if(!rendering_||!scene_||menuVisible_||!preparedThisFrame_||preparedFrame_!=telemetry_.frame)return false;
    if(Network().selectorCalls!=preparedSelectorCalls_||Network().studentCalls!=preparedStudentCalls_)
        throw std::runtime_error("Network ran between the left-eye inference and stereo rendering");
    // Both eyes share immutable selected rows and decoded attributes; only the renderer camera changes.
    renderCamera_=EyeCamera(eye,fov,width,height);
    if(telemetry_.frame<8||telemetry_.frame%120==0){
        const auto& rows=frame_.SelectedRows();const auto& values=frame_.Decoded();
        if(preparedRowsHash_!=Sha256Hex(reinterpret_cast<const uint8_t*>(rows.data()),rows.size()*sizeof(uint32_t))||
           preparedAttributesHash_!=Sha256Hex(reinterpret_cast<const uint8_t*>(values.data()),values.size()*sizeof(DecodedAttribute)))
            throw std::runtime_error("Shared stereo Gaussian output changed before rendering");
        __android_log_print(ANDROID_LOG_INFO,"SplatKitStereo",
            "EYE_INPUT frame=%llu eye=%u revision=%llu K=%zu camera=%.6f,%.6f,%.6f rows=%s attributes=%s",
            (unsigned long long)telemetry_.frame,telemetry_.renderedStereoEyes,(unsigned long long)renderRevision_,rows.size(),
            renderCamera_.center[0],renderCamera_.center[1],renderCamera_.center[2],preparedRowsHash_.c_str(),preparedAttributesHash_.c_str());
    }
    return true;
}

void QuestRuntime::RecordDrawTiming(double milliseconds,const SplatKitMetrics& gpu) {
    telemetry_.renderMs=milliseconds;
    if(telemetry_.renderedStereoEyes==0)telemetry_.leftGpuMs=gpu.pipelineMs;
    else if(telemetry_.renderedStereoEyes==1)telemetry_.rightGpuMs=gpu.pipelineMs;
    else throw std::runtime_error("Unexpected extra eye in shared stereo frame");
    ++telemetry_.renderedStereoEyes;++renderedEyes_;
    char text[120];
    std::snprintf(text,sizeof(text),"LEFT CPU %.1f/FRAME GPU L/R %.2f/%.2f MS K %u",telemetry_.cpuTotalMs,telemetry_.leftGpuMs,telemetry_.rightGpuMs,gpu.input);
    content_.timing=text;
    if(telemetry_.renderedStereoEyes==2&&(telemetry_.frame<8||telemetry_.frame%120==0))__android_log_print(ANDROID_LOG_INFO,"MyCartGS",
        "FRAME_TIMING inference_eye=LEFT N=%u K=%u prepare_ms=%.3f selector_ms=%.3f topk_ms=%.3f student_ms=%.3f decode_ms=%.3f project_ms=%.3f cpu_once_ms=%.3f right_upload_submit_wait_ms=%.3f gpu_left_ms=%.3f gpu_right_ms=%.3f execution=%s",
        telemetry_.inputN,telemetry_.selectedK,telemetry_.prepareMs,telemetry_.selectorMs,telemetry_.topkMs,
        telemetry_.studentMs,telemetry_.decodeMs,telemetry_.projectMs,telemetry_.cpuTotalMs,milliseconds,
        telemetry_.leftGpuMs,telemetry_.rightGpuMs,Network().execution.c_str());
    if(renderedEyes_<=2 || renderedEyes_%300==0)__android_log_print(ANDROID_LOG_INFO,"SplatKitBench",
        "EYE_GPU frame=%llu eye=%u revision=%llu upload_serial=%llu input=%u drawn=%u cull_ms=%.4f sort_ms=%.4f render_pass_with_UI_ms=%.4f pipeline_with_UI_ms=%.4f",
        (unsigned long long)telemetry_.frame,telemetry_.renderedStereoEyes-1,(unsigned long long)renderRevision_,
        (unsigned long long)gpu.uploadSerial,gpu.input,gpu.drawn,gpu.cullMs,gpu.sortMs,gpu.rasterMs,gpu.pipelineMs);
    if(telemetry_.renderedStereoEyes==2&&(telemetry_.frame<8||telemetry_.frame%120==0))__android_log_print(ANDROID_LOG_INFO,"SplatKitStereo",
        "STEREO_COMPLETE frame=%llu revision=%llu K=%u selector_calls=%u student_calls=%u cpu_once_ms=%.3f gpu_left_ms=%.4f gpu_right_ms=%.4f gpu_sum_ms=%.4f",
        (unsigned long long)telemetry_.frame,(unsigned long long)renderRevision_,telemetry_.selectedK,telemetry_.selectorCalls,telemetry_.studentCalls,
        telemetry_.cpuTotalMs,telemetry_.leftGpuMs,telemetry_.rightGpuMs,telemetry_.leftGpuMs+telemetry_.rightGpuMs);
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

void QuestRuntime::StartCpuBenchmark() {
    if(!scene_||!backend_||!backend_->Ready())return;
    benchmark_.Cancel();cpuBenchmark_.Start();cpuCameraReady_=false;cpuBenchmarkFrame_=UINT64_MAX;cpuResumeWarmup_=0;
    content_.page=MenuPage::CpuBenchmark;content_.benchmarkRunning=true;content_.benchmarkRows.clear();content_.benchmarkExport.clear();
    content_.message="LEFT CAMERA FROZEN. NO GAUSSIAN RENDER OR GPU UPLOAD.";
    rendering_=false;menuVisible_=true;preparedThisFrame_=false;
    __android_log_print(ANDROID_LOG_INFO,"SplatKitCpuBench","START cases=%u warmup=%u measured=%u source=LEFT scope=local_nominal_anchor",
        cpuBenchmark_.Cases(),CpuBenchmark::kWarmup,CpuBenchmark::kMeasured);
}

void QuestRuntime::TickCpuBenchmark(const UiPose& left,const std::array<float,4>& fov,uint32_t width,uint32_t height) {
    if(!cpuBenchmark_.Running()||!scene_||!focused_||cpuBenchmarkFrame_==telemetry_.frame)return;
    cpuBenchmarkFrame_=telemetry_.frame;
    try {
        if(!cpuCameraReady_){cpuBenchmarkCamera_=EyeCamera(left,fov,width,height);cpuCameraReady_=true;}
        const auto budget=cpuBenchmark_.Budget();const auto before=Network();const auto timer=std::chrono::steady_clock::now();
        frame_.Prepare(cpuBenchmarkCamera_,*backend_,budget.ratio,budget.fixedK,false);
        const double total=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-timer).count();
        // Resume warmups run the real network but are kept out of the timed sample set.
        if(cpuResumeWarmup_){--cpuResumeWarmup_;return;}
        cpuBenchmark_.Record(frame_.Statistics(),total,uint32_t(Network().selectorCalls-before.selectorCalls),uint32_t(Network().studentCalls-before.studentCalls));
        if(!cpuBenchmark_.Running())FinishCpuBenchmark(false);
    }catch(const std::exception& error){FinishCpuBenchmark(true);content_.benchmarkProgress="CPU BENCHMARK FAILED";content_.message=error.what();
        __android_log_print(ANDROID_LOG_ERROR,"SplatKitCpuBench","ERROR %s",error.what());}
}

void QuestRuntime::UpdateCpuBenchmarkMenu() {
    if(cpuBenchmark_.Running()){
        char progress[120];std::snprintf(progress,sizeof(progress),"%sCASE %u/%u  %s %u/35",
            focused_?"":"PAUSED / WEAR HEADSET. ",cpuBenchmark_.Stage()+1,cpuBenchmark_.Cases(),
            cpuBenchmark_.Iteration()<CpuBenchmark::kWarmup?"WARMUP":"MEASURE",cpuBenchmark_.Iteration()+1);
        content_.benchmarkProgress=progress;
    }
    content_.benchmarkRows=cpuBenchmark_.Rows();
}

void QuestRuntime::FinishCpuBenchmark(bool cancelled) {
    cpuBenchmark_.Cancel();content_.benchmarkRunning=false;content_.benchmarkProgress=cancelled?"CANCELLED / PARTIAL RESULTS":"COMPLETE / 30 CPU SAMPLES PER BUDGET";
    UpdateCpuBenchmarkMenu();if(cpuBenchmark_.Samples().empty())return;
    try {
        if(benchmarkDirectory_.empty())throw std::runtime_error("CPU report directory unavailable");
        const auto directory=benchmarkDirectory_+"/benchmarks";mkdir(directory.c_str(),0755);chmod(directory.c_str(),0755);
        const auto stamp=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        const auto name="cpu_inference_"+std::to_string(stamp)+".csv",path=directory+"/"+name;std::ofstream out(path);out.precision(10);
        out<<"# status="<<(cancelled?"cancelled":"complete")<<"\n# backend=XNNPACK_ARM_CPU_FP32_"<<Network().threads<<"_threads_Selector19_32_32_2_Student62_32_32_14"
            <<"\n# cpu_optimization_flags="<<frame_.OptimizationFlags()<<"\n# backend_workspace_bytes="<<Network().workspaceBytes
            <<"\n# inference_eye=LEFT_fixed_camera_local_nominal_anchor_candidates"
            <<"\n# cpu_scope=candidates_features_selector_topk_student_decode_no_Gaussian_render_no_GPU_upload_no_model_load"
            <<"\n# scene="<<scene_->Identity().slug<<"\n# teacher_map_uid_fingerprint="<<scene_->Identity().teacherFingerprint
            <<"\n# student_checkpoint_sha256="<<scene_->Identity().studentCheckpointSha256<<"\n# camera_view=";
        for(float value:cpuBenchmarkCamera_.view)out<<value<<',';out<<"\n# camera_projection=";
        for(float value:cpuBenchmarkCamera_.projection)out<<value<<',';out<<'\n'<<cpuBenchmark_.Csv();out.close();if(!out)throw std::runtime_error("CPU benchmark CSV write failed");
        content_.benchmarkExport="CSV SAVED: "+name;
        __android_log_print(ANDROID_LOG_INFO,"SplatKitCpuBench","EXPORT status=%s samples=%zu path=%s",cancelled?"cancelled":"complete",cpuBenchmark_.Samples().size(),path.c_str());
        for(const auto& line:cpuBenchmark_.Rows())__android_log_print(ANDROID_LOG_INFO,"SplatKitCpuBench","RESULT %s",line.c_str());
    }catch(const std::exception& error){content_.benchmarkExport="CSV ERROR: "+std::string(error.what());__android_log_print(ANDROID_LOG_ERROR,"SplatKitCpuBench","EXPORT_ERROR %s",error.what());}
}

bool QuestRuntime::PreparePointBenchmark(const UiPose& eye,const std::array<float,4>& fov,uint32_t width,uint32_t height) {
    if(!benchmark_.Running() || !scene_ || !focused_ || benchmarkFrame_==telemetry_.frame)return false;
    benchmarkFrame_=telemetry_.frame; // Exactly one offscreen sample per XR frame, not one per eye.
    try {
        if(!benchmarkCameraReady_){benchmarkCamera_=EyeCamera(eye,fov,width,height);benchmarkCameraReady_=true;}
        if(benchmarkPreparedStage_!=benchmark_.Stage()) {
            const auto start=std::chrono::steady_clock::now();
            frame_.PreparePointBenchmark(benchmark_.K());++renderRevision_;previousFull_=false;
            benchmarkMetrics_={};benchmarkMetrics_.width=benchmarkCamera_.width;benchmarkMetrics_.height=benchmarkCamera_.height;
            benchmarkMetrics_.prepareMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            benchmarkPreparedStage_=benchmark_.Stage();
        }
        return true;
    } catch(const std::exception& error){PointBenchmarkError(error.what());return false;}
}

void QuestRuntime::RecordPointBenchmarkTiming(double uploadMs,double gpuMs,double submitWaitMs,
                                             double cullMs,double sortMs,double rasterMs,uint32_t drawn) {
    if(!benchmark_.Running())return;
    if(benchmarkResumeWarmup_){--benchmarkResumeWarmup_;return;}
    benchmarkMetrics_.uploadMs=uploadMs;benchmarkMetrics_.gpuMs=gpuMs;benchmarkMetrics_.submitWaitMs=submitWaitMs;
    benchmarkMetrics_.cullMs=cullMs;benchmarkMetrics_.sortMs=sortMs;benchmarkMetrics_.rasterMs=rasterMs;benchmarkMetrics_.projected=drawn;
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
        const std::string directory=benchmarkDirectory_+"/benchmarks";mkdir(directory.c_str(),0755);chmod(directory.c_str(),0755);
        const auto stamp=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
        const std::string name="pointcount_"+std::to_string(stamp)+".csv",path=directory+"/"+name;
        std::ofstream out(path);out.precision(10);
        out<<"# scene="<<scene_->Identity().slug<<"\n# status="<<(cancelled?"cancelled":"complete")
           <<"\n# source=original_master_attributes_uid_hash_nested_subsets_no_selector_no_compensation"
           <<"\n# renderer=SplatKit_Vulkan_8a25c57c6c4f62cf682e4c6159cfcae048f7fefe"
           <<"\n# gpu=offscreen_cull_projection_stable_sort_indirect_raster_and_barriers_no_present_no_menu"
           <<"\n# policy=flat_world_no_hierarchy_SH3_RGBA8_cov_SH_half_sort16_subpixel1_render_scale1"
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
    preparedFrame_=UINT64_MAX;preparedThisFrame_=false;
    menu_ = SceneMenu{};
    head_ = UiPose{};
    controllers_ = {};
    controls_ = RuntimeControls{};
    focused_ = false;
    rendering_ = false; menuVisible_ = true; calibrated_ = false;
    validationPreview_ = validationLod_ = false;
    benchmark_=PointBenchmark{};benchmarkAfterLoad_=false;benchmarkDirectory_=outputDirectory;
    cpuBenchmark_=CpuBenchmark{};cpuAfterLoad_=false;cpuCameraReady_=false;
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
    cpuBenchmark_.Cancel();
    // Keep the Activity AssetManager alive until the only bounded loader task has exited.
    if (loadJob_.valid()) loadJob_.wait();
    loadJob_ = std::future<LoadResult>{};
    scene_.reset();
    backend_.reset();
    assets_ = nullptr;
    preparedThisFrame_=false;preparedFrame_=UINT64_MAX;
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
            cpu->Initialize(4);
            try { result.backend->Bind(*result.scene); }
            catch (const std::exception& error) { result.error = error.what(); }
        } catch (const std::exception& error) { result.error = error.what(); }
        catch (...) { result.error = "Unknown scene loading error"; }
        result.milliseconds = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now()-start).count();
        return result;
    });
}

void QuestRuntime::ValidationLaunch(bool fullPreview, bool runLod, bool runPointBenchmark,bool runCpuBenchmark) {
    validationPreview_=fullPreview;
    validationLod_=runLod;
    benchmarkAfterLoad_=runPointBenchmark;
    cpuAfterLoad_=runCpuBenchmark;
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
    if(content_.page==MenuPage::CpuBenchmark){if(cpuBenchmark_.Running())FinishCpuBenchmark(true);menu_.Recenter();return;}
    if (content_.page != MenuPage::Loading) content_.page = scene_?MenuPage::Details:MenuPage::Catalog;
    menu_.Recenter();
}

void QuestRuntime::BeginFrame() {
    ++telemetry_.frame;
    preparedThisFrame_=false;telemetry_.renderedStereoEyes=0;telemetry_.selectorCalls=telemetry_.studentCalls=0;
    telemetry_.leftGpuMs=telemetry_.rightGpuMs=0;
    if (loadJob_.valid() && loadJob_.wait_for(std::chrono::milliseconds(0)) == std::future_status::ready) {
        auto result = loadJob_.get();
        telemetry_.assetMs = result.milliseconds;
        content_.loadMs = float(result.milliseconds);
        scene_ = std::move(result.scene);
        backend_ = std::move(result.backend);
        if (!scene_) {
            benchmarkAfterLoad_=false;cpuAfterLoad_=false;
            content_.page = MenuPage::Error;
            content_.message = result.error;
            __android_log_print(ANDROID_LOG_ERROR, "MyCartGS", "Scene load failed: %s", result.error.c_str());
        } else {
            telemetry_.inputN = content_.loadedGaussians = scene_->Identity().gaussianCount;
            content_.loadedAnchors = scene_->Identity().anchorCount;
            content_.page = MenuPage::Details;
            frame_.Bind(*scene_);
            previousFull_=false;++renderRevision_;
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
            if(cpuAfterLoad_){cpuAfterLoad_=false;StartCpuBenchmark();}
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
            if(content_.page==MenuPage::CpuBenchmark){if(cpuBenchmark_.Running())FinishCpuBenchmark(true);else content_.page=MenuPage::Details;rendering_=false;break;}
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
        case MenuCommand::CpuBenchmark:
            validationPreview_=validationLod_=false;
            if(content_.page==MenuPage::Catalog){cpuAfterLoad_=true;LoadSelected();}
            else if(scene_&&!cpuBenchmark_.Running())StartCpuBenchmark();
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
    if(content_.page==MenuPage::CpuBenchmark)UpdateCpuBenchmarkMenu();
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
