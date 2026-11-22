#include "splatkit_gaussians.h"
#include "rendering/vulkan/device/VulkanContext.h"
#include "sha256.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <numeric>

namespace {
using Clock=std::chrono::steady_clock;
double Ms(Clock::time_point t){return std::chrono::duration<double,std::milli>(Clock::now()-t).count();}
void Check(VkResult r,const char* name){if(r!=VK_SUCCESS)throw std::runtime_error(std::string(name)+" vk="+std::to_string(r));}
mycartgs::FrameCamera Camera(uint32_t width,uint32_t height){
    mycartgs::FrameCamera c;c.width=width;c.height=height;c.fx=600.f*width/1200;c.fy=600.f*height/680;c.cx=(width-1)*.5f;c.cy=(height-1)*.5f;
    c.center={-.8294860125f,-.8426836729f,1.7652033567f};
    const float r[9]={-.7053378820f,.4040099382f,-.5824726224f,-.1863143742f,.6871432066f,.7022258043f,.6839483976f,.6038295031f,-.4093953669f};
    for(int i=0;i<3;++i){for(int j=0;j<3;++j)c.view[i*4+j]=r[j*3+i];for(int j=0;j<3;++j)c.view[i*4+3]-=c.view[i*4+j]*c.center[j];}c.view[15]=1;return c;
}
}
int main(int argc,char** argv){
    try{
        if(argc!=5)throw std::runtime_error("usage: splatkit_point_probe scene.mcgs output.csv width height");
        mycartgs::SceneDescriptor d;d.slug="Replica__office2__rgbd";d.gaussianCount=43197;d.anchorCount=16723;
        d.assetSha256="1d735b0871ff94cc6acbf039ad014e0d1490ca4e209caef92648a0b9e53a6685";
        d.teacherFingerprint="9e2d413fa7864127465394fa22bb3da4e05d912d4c9e0822a4e928933aa36804";
        d.studentCheckpointSha256="fd85285c6bd5871f4b41c85690597ccadc965d5a5ec6f474a983e4e09f547a75";
        mycartgs::SceneAsset scene;scene.LoadFile(argv[1],d);mycartgs::GaussianFrame frame;frame.Bind(scene);
        auto result=splatkit::VulkanContext::create();if(!result)throw std::runtime_error(result.error().message);auto owner=std::move(result.value());
        std::cout<<"GPU "<<owner->deviceDescription()<<" CAPS "<<owner->capabilities()<<std::endl;
        VkCommandPool pool;VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};pi.queueFamilyIndex=owner->queueFamily();pi.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;Check(vkCreateCommandPool(owner->device(),&pi,nullptr,&pool),"pool");
        VkCommandBuffer command;VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};ai.commandPool=pool;ai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;ai.commandBufferCount=1;Check(vkAllocateCommandBuffers(owner->device(),&ai,&command),"command");
        VkFence fence;VkFenceCreateInfo fi{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};Check(vkCreateFence(owner->device(),&fi,nullptr,&fence),"fence");
        mycartgs::SplatKitGaussians renderer;renderer.Initialize(owner->instance(),owner->physicalDevice(),owner->device(),owner->queue(),owner->queueFamily());
        const auto camera=Camera(uint32_t(std::stoul(argv[3])),uint32_t(std::stoul(argv[4])));
        std::ofstream csv(argv[2]);csv.precision(10);
        csv<<"# source=SplatKit_8a25c57c6c4f62cf682e4c6159cfcae048f7fefe_original_Master_UID_hash_subsets_no_network\n"
           <<"# policy=SH3_RGBA8_cov_SH_half_sort16_subpixel1_renderScale1_no_hierarchy\n"
           <<"round,k,sample,warmup,drawn,width,height,prepare_once_ms,upload_once_ms,cull_gpu_ms,sort_gpu_ms,raster_gpu_ms,total_gpu_ms,record_ms,submit_wait_ms,full_call_ms,selected_uid_sha256\n";
        std::vector<uint32_t> budgets{0,1000,2000,3000,4000,5000,10000,15000,43197};uint64_t revision=0;
        for(int round=1;round<=2;++round){auto order=budgets;if(round==2)std::reverse(order.begin(),order.end());
            for(uint32_t k:order){const auto prepare=Clock::now();frame.PreparePointBenchmark(k);const double prepareMs=Ms(prepare);
                renderer.Upload(scene,frame.SelectedRows(),frame.Decoded(),++revision);
                std::vector<uint64_t> uids;const auto* all=reinterpret_cast<const uint64_t*>(scene.Section("uids").data);for(uint32_t row:frame.SelectedRows())uids.push_back(all[row]);
                const auto hash=mycartgs::Sha256Hex(reinterpret_cast<const uint8_t*>(uids.data()),uids.size()*sizeof(uint64_t));
                for(int i=0;i<115;++i){const auto begin=Clock::now();Check(vkResetCommandBuffer(command,0),"reset");Check(vkResetFences(owner->device(),1,&fence),"reset fence");
                    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};bi.flags=VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;Check(vkBeginCommandBuffer(command,&bi),"begin");
                    renderer.Offscreen(command,camera);Check(vkEndCommandBuffer(command),"end");const double recordMs=Ms(begin);
                    VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};submit.commandBufferCount=1;submit.pCommandBuffers=&command;auto timer=Clock::now();
                    Check(vkQueueSubmit(owner->queue(),1,&submit,fence),"submit");Check(vkWaitForFences(owner->device(),1,&fence,VK_TRUE,UINT64_MAX),"wait");const double waitMs=Ms(timer);
                    renderer.Collect();const auto& m=renderer.Metrics();if(m.drawn>k)throw std::runtime_error("GPU drawn count exceeds input K");
                    if(!std::isfinite(m.pipelineMs)||m.pipelineMs<0||m.rasterMs<0)throw std::runtime_error("Invalid real GPU timing");
                    csv<<round<<','<<k<<','<<i<<','<<(i<15)<<','<<m.drawn<<','<<camera.width<<','<<camera.height<<','<<prepareMs<<','<<m.uploadMs<<','<<m.cullMs<<','<<m.sortMs<<','<<m.rasterMs<<','<<m.pipelineMs<<','<<recordMs<<','<<waitMs<<','<<Ms(begin)<<','<<hash<<'\n';
                }
                const auto image=renderer.ImageDiagnostics();if((k==0&&image[0]!=0)||(k>0&&image[0]==0))throw std::runtime_error("Offscreen image coverage check failed");
                std::cout<<"CASE_PASS round="<<round<<" K="<<k<<" drawn="<<renderer.Metrics().drawn<<" nonblack="<<image[0]<<" rgb_sum="<<image[1]<<','<<image[2]<<','<<image[3]<<std::endl;
            }
        }
        renderer.Close();vkDestroyFence(owner->device(),fence,nullptr);vkDestroyCommandPool(owner->device(),pool,nullptr);
        std::cout<<"SPLATKIT_POINT_PROBE_PASS real_GPU_timestamps stable_UID_subsets 18_cases 1800_measured_samples"<<std::endl;
    }catch(const std::exception& e){std::cerr<<"SPLATKIT_POINT_PROBE_FAIL "<<e.what()<<std::endl;return 1;}
}
