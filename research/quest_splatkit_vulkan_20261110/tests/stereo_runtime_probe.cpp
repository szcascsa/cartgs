#include "quest_runtime.h"
#include "splatkit_gaussians.h"
#include "xnnpack_backend.h"
#include "gaussian_frame_reference.h"
#include "xnnpack_backend_reference.h"
#include "rendering/vulkan/device/VulkanContext.h"
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace mycartgs {
struct StereoRuntimeProbe {
    static void Bind(QuestRuntime& runtime,std::unique_ptr<SceneAsset> scene) {
        runtime.scene_=std::move(scene);auto backend=std::make_unique<XnnpackBackend>();backend->Initialize(4);backend->Bind(*runtime.scene_);
        runtime.backend_=std::move(backend);runtime.frame_.Bind(*runtime.scene_);runtime.rendering_=true;runtime.menuVisible_=false;
        runtime.calibrated_=true;runtime.mapFromAppRotation_={1,0,0,0,1,0,0,0,1};runtime.mapFromAppTranslation_={0,0,0};
    }
    static void Controls(QuestRuntime& runtime,BudgetMode mode,float ratio,uint32_t k) {runtime.controls_={mode,ratio,k};}
    static void StartCpu(QuestRuntime& runtime,const std::string& output){runtime.benchmarkDirectory_=output;runtime.focused_=true;runtime.StartCpuBenchmark();}
    static bool CpuRunning(const QuestRuntime& runtime){return runtime.cpuBenchmark_.Running();}
    static const CpuBenchmark& Cpu(const QuestRuntime& runtime){return runtime.cpuBenchmark_;}
};
}
namespace {
void Require(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
template<class T>bool Same(const std::vector<T>& a,const std::vector<T>& b){return a.size()==b.size()&&(!a.size()||std::memcmp(a.data(),b.data(),a.size()*sizeof(T))==0);}
void VkCheck(VkResult r){if(r!=VK_SUCCESS)throw std::runtime_error("GPU probe VkResult="+std::to_string(r));}
struct ImageDifference { uint64_t different=0;uint32_t maximum=0;double squared=0; };
ImageDifference CompareImages(const std::vector<uint8_t>& a,const std::vector<uint8_t>& b){
    Require(a.size()==b.size(),"Image shape mismatch");ImageDifference d;
    for(size_t i=0;i<a.size();++i){const int v=int(a[i])-int(b[i]);d.different+=v!=0;d.maximum=std::max(d.maximum,uint32_t(std::abs(v)));d.squared+=double(v)*v;}
    return d;
}
}
int main(int argc,char** argv){
    try {
        using namespace mycartgs;
        if(argc!=3)throw std::runtime_error("usage: stereo_runtime_probe scene.mcgs output.csv");
        SceneDescriptor d;d.slug="Replica__office2__rgbd";d.gaussianCount=43197;d.anchorCount=16723;
        d.assetSha256="1d735b0871ff94cc6acbf039ad014e0d1490ca4e209caef92648a0b9e53a6685";
        d.teacherFingerprint="9e2d413fa7864127465394fa22bb3da4e05d912d4c9e0822a4e928933aa36804";
        d.studentCheckpointSha256="fd85285c6bd5871f4b41c85690597ccadc965d5a5ec6f474a983e4e09f547a75";
        auto scene=std::make_unique<SceneAsset>();scene->LoadFile(argv[1],d);QuestRuntime runtime;StereoRuntimeProbe::Bind(runtime,std::move(scene));
        ReferenceXnnpackBackend referenceBackend;referenceBackend.Initialize(2);referenceBackend.Bind(runtime.Scene());ReferenceGaussianFrame reference;reference.Bind(runtime.Scene());
        const UiPose left{{-.829486f,-.842684f,1.765204f},{.37828892f,-.39012796f,.8369403f,.065027215f}};
        UiPose right=left;right.position[0]+=.044f;right.position[1]+=.012f;right.position[2]-=.043f;
        const std::array<float,4> leftFov{-.94247776f,.6981317f,.7679449f,-.9599311f},rightFov{-.6981317f,.94247776f,.7679449f,-.9599311f};
        auto contextResult=splatkit::VulkanContext::create();Require(bool(contextResult),"Vulkan probe context failed");auto context=std::move(contextResult.value());
        VkCommandPool pool;VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};pi.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;pi.queueFamilyIndex=context->queueFamily();VkCheck(vkCreateCommandPool(context->device(),&pi,nullptr,&pool));
        VkCommandBuffer command;VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};ai.commandPool=pool;ai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;ai.commandBufferCount=1;VkCheck(vkAllocateCommandBuffers(context->device(),&ai,&command));
        SplatKitGaussians renderer;renderer.Initialize(context->instance(),context->physicalDevice(),context->device(),context->queue(),context->queueFamily());
        auto draw=[&](){const auto begin=std::chrono::steady_clock::now();
            renderer.Upload(runtime.Scene(),runtime.Frame().SelectedRows(),runtime.Frame().Decoded(),runtime.RenderRevision());VkCheck(vkResetCommandBuffer(command,0));
            VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};VkCheck(vkBeginCommandBuffer(command,&bi));renderer.Offscreen(command,runtime.Camera());VkCheck(vkEndCommandBuffer(command));
            VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};submit.commandBufferCount=1;submit.pCommandBuffers=&command;VkCheck(vkQueueSubmit(context->queue(),1,&submit,VK_NULL_HANDLE));VkCheck(vkQueueWaitIdle(context->queue()));renderer.Collect();
            const auto gpu=renderer.Metrics();runtime.RecordDrawTiming(std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-begin).count(),gpu);return gpu;};
        const uint32_t width=1680,height=1760;std::ofstream csv(argv[2]);csv<<"case,frame,N,K,selector_calls,student_calls,cpu_once_ms,left_bitwise,right_shared,render_revision_shared,gpu_left_ms,gpu_right_ms,drawn_left,drawn_right,one_upload,left_nonblack,right_nonblack,rgba_left_bitwise,rgba_right_bitwise,rgba_no_upload_repeat_bitwise,left_changed_bytes,left_max_byte,left_squared,right_changed_bytes,right_max_byte,right_squared,control_changed_bytes,control_max_byte,control_squared\n";
        uint32_t cases=0;
        for(int pass=0;pass<2;++pass)for(const auto ratio:{.01f,.1f,.3f,.5f}){
            const uint32_t budgets[]={100,1000,5000,100000};const uint32_t budget=budgets[cases%4];
            const auto mode=pass?BudgetMode::FixedK:BudgetMode::Ratio;StereoRuntimeProbe::Controls(runtime,mode,ratio,budget);
            runtime.BeginFrame();Require(!runtime.PrepareEye(right,rightFov,width,height),"A prior frame's output leaked before left inference");
            const auto before=runtime.Network();Require(runtime.PrepareLeftEye(left,leftFov,width,height),"Left inference failed");
            Require(runtime.PrepareEye(left,leftFov,width,height),"Left render camera failed");const auto camera=runtime.Camera();
            const auto rows=runtime.Frame().SelectedRows();const auto decoded=runtime.Frame().Decoded();const auto revision=runtime.RenderRevision();
            reference.Prepare(camera,referenceBackend,ratio,pass?budget:0,false);
            Require(rows==reference.SelectedRows()&&Same(decoded,reference.Decoded()),"Shared output differs from original left-eye inference");
            const auto after=runtime.Network();Require(after.selectorCalls-before.selectorCalls==1&&after.studentCalls-before.studentCalls==1,"Expected one Selector and one Student call");
            Require(runtime.PrepareLeftEye(left,leftFov,width,height),"Same-frame left request failed");
            Require(runtime.Network().selectorCalls==after.selectorCalls&&runtime.Network().studentCalls==after.studentCalls,"Duplicate left request reran the networks");
            const auto gpuLeft=draw();std::string leftHash,rightHash;std::vector<uint8_t> leftPixels,rightPixels;const auto leftImage=renderer.ImageDiagnostics(&leftHash,&leftPixels);
            Require(runtime.PrepareEye(right,rightFov,width,height),"Right render camera failed");
            Require(runtime.Camera().center!=camera.center&&runtime.Camera().projection!=camera.projection,"Right camera reused left-eye pose or FOV");
            Require(rows==runtime.Frame().SelectedRows()&&Same(decoded,runtime.Frame().Decoded())&&runtime.RenderRevision()==revision,"Right-eye Gaussian output or revision changed");
            Require(runtime.Network().selectorCalls==after.selectorCalls&&runtime.Network().studentCalls==after.studentCalls,"Right rendering reran a network");
            const auto gpuRight=draw();const auto rightImage=renderer.ImageDiagnostics(&rightHash,&rightPixels);
            Require(gpuLeft.uploadSerial==gpuRight.uploadSerial,"Right eye uploaded a second Gaussian buffer");
            Require(leftImage[0]>0&&rightImage[0]>0&&leftImage!=rightImage,"Stereo views are blank or use the same camera");
            Require(runtime.Telemetry().renderedStereoEyes==2&&runtime.Telemetry().leftGpuMs==gpuLeft.pipelineMs&&runtime.Telemetry().rightGpuMs==gpuRight.pipelineMs,"Per-eye GPU telemetry changed");
            const auto originalImage=[&](const FrameCamera& view,bool upload,std::vector<uint8_t>& pixels){
                if(upload)renderer.Upload(runtime.Scene(),reference.SelectedRows(),reference.Decoded(),revision+100000);
                VkCheck(vkResetCommandBuffer(command,0));VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};VkCheck(vkBeginCommandBuffer(command,&bi));
                renderer.Offscreen(command,view);VkCheck(vkEndCommandBuffer(command));VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};submit.commandBufferCount=1;submit.pCommandBuffers=&command;
                VkCheck(vkQueueSubmit(context->queue(),1,&submit,VK_NULL_HANDLE));VkCheck(vkQueueWaitIdle(context->queue()));renderer.Collect();std::string hash;renderer.ImageDiagnostics(&hash,&pixels);return hash;};
            std::vector<uint8_t> repeated,originalLeft,originalRight,originalRepeat;
            const auto repeatHash=originalImage(camera,false,repeated);
            const auto originalLeftHash=originalImage(camera,true,originalLeft);
            originalImage(camera,false,originalRepeat);
            const auto originalRightHash=originalImage(runtime.Camera(),true,originalRight);
            const auto dl=CompareImages(leftPixels,originalLeft),dr=CompareImages(rightPixels,originalRight),dc=CompareImages(originalLeft,originalRepeat);
            csv<<(pass?"fixed":"ratio")<<ratio<<','<<runtime.Telemetry().frame<<','<<runtime.Telemetry().inputN<<','<<rows.size()<<",1,1,"<<runtime.Telemetry().cpuTotalMs<<",1,1,1,"<<gpuLeft.pipelineMs<<','<<gpuRight.pipelineMs<<','<<gpuLeft.drawn<<','<<gpuRight.drawn<<",1,"<<leftImage[0]<<','<<rightImage[0]<<','<<(leftHash==originalLeftHash)<<','<<(rightHash==originalRightHash)<<','<<(leftHash==repeatHash)
                <<','<<dl.different<<','<<dl.maximum<<','<<dl.squared<<','<<dr.different<<','<<dr.maximum<<','<<dr.squared<<','<<dc.different<<','<<dc.maximum<<','<<dc.squared<<'\n';++cases;
        }
        runtime.BeginFrame();StereoRuntimeProbe::Controls(runtime,BudgetMode::Full,.1f,1000);const auto beforeFull=runtime.Network();
        Require(runtime.PrepareLeftEye(left,leftFov,width,height),"Full preparation failed");Require(runtime.Frame().SelectedRows().size()==43197,"Full topology changed");
        Require(runtime.Network().selectorCalls==beforeFull.selectorCalls&&runtime.Network().studentCalls==beforeFull.studentCalls,"Full ran a network");
        Require(runtime.PrepareEye(right,rightFov,width,height),"Full right camera failed");
        runtime.BeginFrame();StereoRuntimeProbe::Controls(runtime,BudgetMode::Ratio,.1f,1000);UiPose empty=left;empty.position={100000,100000,100000};
        const auto beforeEmpty=runtime.Network();Require(runtime.PrepareLeftEye(empty,leftFov,width,height),"Empty candidate frame failed");
        Require(runtime.Frame().SelectedRows().empty(),"Empty candidates retained old Gaussians");Require(runtime.PrepareEye(right,rightFov,width,height),"Empty right frame failed");
        Require(runtime.Network().selectorCalls==beforeEmpty.selectorCalls&&runtime.Network().studentCalls==beforeEmpty.studentCalls,"Empty frame ran a network");
        runtime.OpenSceneMenu();runtime.BeginFrame();Require(!runtime.PrepareLeftEye(left,leftFov,width,height)&&!runtime.PrepareEye(right,rightFov,width,height),"Menu consumed old stereo outputs");
        const std::string file=argv[2],directory=file.substr(0,file.find_last_of('/'));StereoRuntimeProbe::StartCpu(runtime,directory);
        uint32_t iterations=0;
        while(StereoRuntimeProbe::CpuRunning(runtime)){
            runtime.BeginFrame();Require(!runtime.PrepareLeftEye(left,leftFov,width,height),"CPU test enabled Gaussian rendering");
            const auto calls=runtime.Network();runtime.PrepareLeftEye(left,leftFov,width,height);
            Require(runtime.Network().selectorCalls==calls.selectorCalls&&runtime.Network().studentCalls==calls.studentCalls,"CPU test ran twice in one XR frame");
            Require(!runtime.PrepareEye(right,rightFov,width,height),"CPU test ran right-eye inference");Require(++iterations<=315,"CPU test schedule unbounded");
        }
        const auto& cpu=StereoRuntimeProbe::Cpu(runtime);Require(cpu.Samples().size()==315&&cpu.Rows().size()==9,"CPU sweep count failed");
        for(const auto& sample:cpu.Samples())Require(sample.selectorCalls==1&&sample.studentCalls==1,"CPU sample network count changed");
        SceneMenu menu;MenuContent content;content.scenes.push_back({"REPLICA / OFFICE2 / RGB-D",43197,16723});menu.Build(content);
        bool cpuButton=false;for(const auto& button:menu.Buttons())cpuButton|=button.action.command==MenuCommand::CpuBenchmark&&button.enabled;
        Require(cpuButton,"CPU benchmark missing from catalog");content.page=MenuPage::Details;content.inferenceReady=false;menu.Build(content);
        for(const auto& button:menu.Buttons())if(button.action.command==MenuCommand::CpuBenchmark)Require(!button.enabled,"CPU readiness gate missing");
        renderer.Close();vkDestroyCommandPool(context->device(),pool,nullptr);
        std::cout<<"STEREO_RUNTIME_PASS cases="<<cases<<" left_original_bitwise right_shared fresh_each_frame duplicate_guard empty_full_menu camera_and_FOV_independent real_GPU_both_eyes one_upload RGBA_repeat_controls_recorded CPU_sweep315_samples_menu_gate"<<std::endl;
    }catch(const std::exception& e){std::cerr<<"STEREO_RUNTIME_FAIL "<<e.what()<<std::endl;return 1;}
}
