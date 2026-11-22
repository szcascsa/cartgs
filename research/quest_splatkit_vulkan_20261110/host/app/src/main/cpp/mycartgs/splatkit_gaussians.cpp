#include "splatkit_gaussians.h"
#include "sha256.h"
#include "rendering/vulkan/device/VulkanContext.h"
#include "rendering/vulkan/frame/VulkanCameraUniforms.h"
#include "rendering/vulkan/frame/VulkanFrameCompute.h"
#include "rendering/vulkan/presentation/VulkanRenderTarget.h"
#include "rendering/vulkan/raster/VulkanSplatPipeline.h"
#include "rendering/vulkan/world/VulkanWorld.h"
#include "splatkit/Log.h"
#include "point_benchmark.h"
#include <android/log.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <optional>
#include <stdexcept>

namespace mycartgs {
namespace {
void Check(VkResult r,const char* stage){if(r!=VK_SUCCESS)throw std::runtime_error(std::string(stage)+" VkResult="+std::to_string(r));}
template<class T>T Value(splat::Result<T>&& value){if(!value)throw std::runtime_error(value.error().message);return std::move(value.value());}
void Sink(splatkit::LogLevel level,const char* message){__android_log_print(level==splatkit::LogLevel::error?ANDROID_LOG_ERROR:level==splatkit::LogLevel::warn?ANDROID_LOG_WARN:ANDROID_LOG_INFO,"SplatKit", "%s",message);}
}
struct SplatKitGaussians::Impl {
    std::unique_ptr<splatkit::VulkanContext> ctx;
    std::unique_ptr<splatkit::VulkanCameraUniforms> cameras;
    std::optional<splatkit::VulkanFrameCompute::Pipelines> pipelines;
    std::unique_ptr<splatkit::VulkanWorld> world;
    std::unique_ptr<splatkit::VulkanFrameCompute> compute;
    std::unique_ptr<splatkit::VulkanRenderTarget> target;
    std::unique_ptr<splatkit::VulkanSplatPipeline> screenRaster,offscreenRaster;
    splatkit::VulkanSplatPipeline* raster=nullptr;
    VkRenderPass screenPass=VK_NULL_HANDLE;
    VkQueryPool queries=VK_NULL_HANDLE;
    uint32_t timestampBits=0;
    float period=0;
    uint64_t revision=UINT64_MAX,serial=0;
    const SceneAsset* boundScene=nullptr;
    bool uploaded=false,encoded=false;
    VkExtent2D extent{};
    splatkit::SplatRenderer::Frame frame;
    splatkit::SplatRenderer::Range range;
    std::optional<splatkit::VulkanFrameCompute::Draw> gpuDraw;
    SplatKitMetrics metrics;
    ~Impl(){if(ctx){ctx->waitIdle();screenRaster.reset();offscreenRaster.reset();target.reset();compute.reset();world.reset();cameras.reset();pipelines.reset();if(queries)vkDestroyQueryPool(ctx->device(),queries,nullptr);}}
    void Camera(const FrameCamera& c) {
        extent={c.width,c.height};frame={};frame.orderSource=splatkit::SplatRenderer::OrderSource::gpu;
        // Map coordinates stay untouched. Convert the CV camera axes into SplatKit's RUB view.
        for(int r=0;r<4;++r)for(int col=0;col<4;++col)frame.view.at(r,col)=c.view[r*4+col]*((r==1||r==2)?-1.f:1.f);
        // Original asymmetric FOV including the original half-pixel convention.
        frame.proj={};frame.proj.at(0,0)=2*c.fx/c.width;frame.proj.at(1,1)=2*c.fy/c.height;
        frame.proj.at(0,2)=1-2*(c.cx+.5f)/c.width;
        frame.proj.at(1,2)=2*(c.cy+.5f)/c.height-1;
        frame.proj.at(2,2)=-c.farPlane/(c.farPlane-c.nearPlane);
        frame.proj.at(2,3)=-c.farPlane*c.nearPlane/(c.farPlane-c.nearPlane);frame.proj.at(3,2)=-1;
        frame.cameraPosition={c.center[0],c.center[1],c.center[2]};frame.shDegree=3;
        range={0,metrics.input};frame.ranges=&range;frame.rangeCount=metrics.input?1:0;
        // These are the upstream flat-render defaults: 1px subpixel cutoff and 16bit sort.
        if(!cameras->update(0,frame.view,frame.proj,frame.cameraPosition,extent,false,0,splatkit::RenderPolicy{}.subpixelThreshold))
            throw std::runtime_error("SplatKit rejected camera projection");
    }
};
SplatKitGaussians::SplatKitGaussians()=default;
SplatKitGaussians::~SplatKitGaussians()=default;
void SplatKitGaussians::Initialize(VkInstance instance,VkPhysicalDevice physical,VkDevice device,VkQueue queue,uint32_t family){
    if(impl_)return;
    auto state=std::make_unique<Impl>();splatkit::setLogSink(Sink);
    state->ctx=Value(splatkit::VulkanContext::adopt(instance,physical,device,queue,family));
    if(!splatkit::VulkanFrameCompute::supported(*state->ctx))throw std::runtime_error("SplatKit GPU cull/radix capability missing; CPU fallback prohibited");
    state->cameras=Value(splatkit::VulkanCameraUniforms::create(*state->ctx));
    state->pipelines=splatkit::VulkanFrameCompute::Pipelines::compile(*state->ctx);
    if(!state->pipelines)throw std::runtime_error("Cannot compile upstream SplatKit compute pipelines");
    uint32_t count=0;vkGetPhysicalDeviceQueueFamilyProperties(physical,&count,nullptr);std::vector<VkQueueFamilyProperties> queues(count);
    vkGetPhysicalDeviceQueueFamilyProperties(physical,&count,queues.data());state->timestampBits=queues.at(family).timestampValidBits;
    VkPhysicalDeviceProperties props;vkGetPhysicalDeviceProperties(physical,&props);state->period=props.limits.timestampPeriod;
    if(!state->timestampBits||!props.limits.timestampComputeAndGraphics)throw std::runtime_error("Actual GPU timestamp support required for this baseline");
    VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};info.queryType=VK_QUERY_TYPE_TIMESTAMP;info.queryCount=3;
    Check(vkCreateQueryPool(device,&info,nullptr,&state->queries),"SplatKit timing pool");
    impl_=std::move(state);
    __android_log_print(ANDROID_LOG_INFO,"SplatKitBench","RENDERER_READY source=8a25c57c6c4f62cf682e4c6159cfcae048f7fefe sort=16 subpixel=1 sh=3 render_scale=1 internal_lod=off");
}
void SplatKitGaussians::Upload(const SceneAsset& scene,const std::vector<uint32_t>& selected,const std::vector<DecodedAttribute>& decoded,uint64_t revision){
    if(!impl_||selected.size()!=decoded.size())throw std::runtime_error("Invalid SplatKit upload");auto& s=*impl_;
    if(s.uploaded&&s.boundScene==&scene&&s.revision==revision)return;
    const auto start=std::chrono::steady_clock::now();const uint32_t n=uint32_t(selected.size());
    splat::SplatCloud cloud;cloud.shDegree=n?3:0;cloud.positions.resize(n*3);cloud.covariances.resize(n*6);cloud.colors.resize(n*3);cloud.alphas.resize(n);cloud.sh.resize(n*45);
    const auto* rest=reinterpret_cast<const float*>(scene.Section("sh_rest").data);
    for(uint32_t i=0;i<n;++i){if(selected[i]>=scene.Identity().gaussianCount)throw std::runtime_error("SplatKit selected row out of bounds");const auto& a=decoded[i];
        std::copy(a.xyz.begin(),a.xyz.end(),cloud.positions.begin()+i*3);cloud.alphas[i]=a.opacity;
        for(int c=0;c<3;++c)cloud.colors[i*3+c]=std::clamp(.5f+.28209479177387814f*a.dc[c],0.f,1.f);
        const auto& q=a.rotation;const float w=q[0],x=q[1],y=q[2],z=q[3];
        const float r[9]={1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y),2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x),2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)};
        const int rr[]={0,0,0,1,1,2},cc[]={0,1,2,1,2,2};
        for(int j=0;j<6;++j){float v=0;for(int k=0;k<3;++k)v+=r[rr[j]*3+k]*a.scale[k]*a.scale[k]*r[cc[j]*3+k];cloud.covariances[i*6+j]=v;}
        std::copy_n(rest+selected[i]*45,45,cloud.sh.data()+i*45);
    }
    // World shape changes happen outside timed dispatch. Pipelines remain compiled and reused.
    s.ctx->waitIdle();
    if(!s.world||s.world->count!=n){s.compute.reset();s.world=splatkit::VulkanWorld::upload(*s.ctx,cloud,3,false);
        if(!s.world)throw std::runtime_error("SplatKit world upload failed");
        s.compute=Value(splatkit::VulkanFrameCompute::create(*s.ctx,*s.pipelines,n));}
    else if(n&&!s.world->uploadTile(0,cloud))throw std::runtime_error("SplatKit attribute update failed");
    s.revision=revision;s.boundScene=&scene;s.uploaded=true;s.metrics.input=n;
    ++s.metrics.uploadSerial;
    s.metrics.uploadMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
}
void SplatKitGaussians::Prepare(const FrameCamera& camera,VkRenderPass pass){auto& s=*impl_;s.Camera(camera);
    if(!s.screenRaster||s.screenPass!=pass){s.screenRaster=Value(splatkit::VulkanSplatPipeline::create(*s.ctx,pass,*s.cameras));s.screenPass=pass;}
    s.raster=s.screenRaster.get();}
void SplatKitGaussians::Compute(VkCommandBuffer command){auto& s=*impl_;
    vkCmdResetQueryPool(command,s.queries,0,3);vkCmdWriteTimestamp(command,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,s.queries,0);
    s.gpuDraw=s.compute->encode(command,0,s.cameras->buffer(0),*s.world->splats,*s.world->sh,s.world->shDegree,s.frame);
    if(!s.gpuDraw)throw std::runtime_error("SplatKit GPU frame encoding failed");
    s.raster->bindProjected(0,s.gpuDraw->projected,s.gpuDraw->order,s.gpuDraw->capacity);s.encoded=true;
    // On tile GPUs a timestamp *inside* the pass can run after tile shading. Bracket the
    // complete pass, including clear/store, after all compute work has completed.
    vkCmdWriteTimestamp(command,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,s.queries,1);
}
void SplatKitGaussians::Draw(VkCommandBuffer command){auto& s=*impl_;
    const VkViewport viewport{0,float(s.extent.height),float(s.extent.width),-float(s.extent.height),0,1};const VkRect2D scissor{{0,0},s.extent};
    vkCmdSetViewport(command,0,1,&viewport);vkCmdSetScissor(command,0,1,&scissor);
    s.raster->drawProjected(command,0,s.gpuDraw->arguments);
}
void SplatKitGaussians::End(VkCommandBuffer command){auto& s=*impl_;vkCmdWriteTimestamp(command,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,s.queries,2);}
void SplatKitGaussians::Offscreen(VkCommandBuffer command,const FrameCamera& camera){auto& s=*impl_;
    if(!s.target||s.target->extent().width!=camera.width||s.target->extent().height!=camera.height){s.offscreenRaster.reset();s.target=Value(splatkit::VulkanRenderTarget::create(*s.ctx,VK_FORMAT_R8G8B8A8_UNORM,{camera.width,camera.height}));
        s.offscreenRaster=Value(splatkit::VulkanSplatPipeline::create(*s.ctx,s.target->renderPass(),*s.cameras));}
    s.Camera(camera);s.raster=s.offscreenRaster.get();Compute(command);
    const VkClearValue clear{{{0,0,0,1}}};VkRenderPassBeginInfo pass{VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO};pass.renderPass=s.target->renderPass();pass.framebuffer=s.target->framebuffer();
    pass.renderArea.extent=s.extent;pass.clearValueCount=1;pass.pClearValues=&clear;vkCmdBeginRenderPass(command,&pass,VK_SUBPASS_CONTENTS_INLINE);Draw(command);vkCmdEndRenderPass(command);End(command);
}
void SplatKitGaussians::Collect(){auto& s=*impl_;if(!s.encoded)return;s.compute->submitted(0,++s.serial);s.compute->collectCompleted(s.serial);
    const auto& stats=s.compute->stats();if(stats.status)throw std::runtime_error("SplatKit GPU overflow/status="+std::to_string(stats.status));
    uint64_t ticks[3];Check(vkGetQueryPoolResults(s.ctx->device(),s.queries,0,3,sizeof(ticks),ticks,sizeof(uint64_t),VK_QUERY_RESULT_64_BIT),"SplatKit query result");
    s.metrics.drawn=stats.drawn;s.metrics.cullMs=stats.cullMillis;s.metrics.sortMs=stats.sortMillis;
    s.metrics.rasterMs=TimestampTicks(ticks[1],ticks[2],s.timestampBits)*double(s.period)/1e6;
    s.metrics.pipelineMs=TimestampTicks(ticks[0],ticks[2],s.timestampBits)*double(s.period)/1e6;s.encoded=false;
}
const SplatKitMetrics& SplatKitGaussians::Metrics()const{return impl_->metrics;}
std::array<uint64_t,4> SplatKitGaussians::ImageDiagnostics(std::string* rgbaSha256,std::vector<uint8_t>* outputRgba)const{
    const auto& s=*impl_;if(!s.target)throw std::runtime_error("No offscreen image for diagnostics");
    const auto bytes=VkDeviceSize(s.extent.width)*s.extent.height*4;
    auto pixels=splatkit::VulkanBuffer::hostVisible(*s.ctx,bytes,VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    if(!pixels||!pixels->mapped())throw std::runtime_error("Cannot allocate diagnostic readback");
    VkCommandPool pool;VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};pi.queueFamilyIndex=s.ctx->queueFamily();
    Check(vkCreateCommandPool(s.ctx->device(),&pi,nullptr,&pool),"diagnostic pool");
    VkCommandBuffer command;VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};ai.commandPool=pool;ai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;ai.commandBufferCount=1;
    Check(vkAllocateCommandBuffers(s.ctx->device(),&ai,&command),"diagnostic command");
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};Check(vkBeginCommandBuffer(command,&bi),"diagnostic begin");
    VkMemoryBarrier written{VK_STRUCTURE_TYPE_MEMORY_BARRIER};written.srcAccessMask=VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT;written.dstAccessMask=VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(command,VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,0,1,&written,0,nullptr,0,nullptr);
    VkBufferImageCopy copy{};copy.imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,0,1};copy.imageExtent={s.extent.width,s.extent.height,1};
    vkCmdCopyImageToBuffer(command,s.target->image(),VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,pixels->handle(),1,&copy);
    VkMemoryBarrier copied{VK_STRUCTURE_TYPE_MEMORY_BARRIER};copied.srcAccessMask=VK_ACCESS_TRANSFER_WRITE_BIT;copied.dstAccessMask=VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(command,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_HOST_BIT,0,1,&copied,0,nullptr,0,nullptr);
    Check(vkEndCommandBuffer(command),"diagnostic end");VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};submit.commandBufferCount=1;submit.pCommandBuffers=&command;
    Check(vkQueueSubmit(s.ctx->queue(),1,&submit,VK_NULL_HANDLE),"diagnostic submit");Check(vkQueueWaitIdle(s.ctx->queue()),"diagnostic wait");pixels->invalidate(0,bytes);
    std::array<uint64_t,4> result{};const auto* rgba=static_cast<const uint8_t*>(pixels->mapped());
    for(size_t i=0;i<bytes;i+=4){result[0]+=(rgba[i]|rgba[i+1]|rgba[i+2])!=0;for(int c=0;c<3;++c)result[c+1]+=rgba[i+c];}
    if(rgbaSha256)*rgbaSha256=Sha256Hex(rgba,size_t(bytes));
    if(outputRgba)outputRgba->assign(rgba,rgba+size_t(bytes));
    vkDestroyCommandPool(s.ctx->device(),pool,nullptr);return result;
}
void SplatKitGaussians::Close(){impl_.reset();}
}
