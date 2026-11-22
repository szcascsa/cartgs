#include "gaussian_frame.h"
#include "qnn_backend.h"
#include "probe_backend_io.h"
#include "vulkan_gaussians.h"
#include <vulkan/vulkan.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <numeric>
#include <stdexcept>

using namespace mycartgs;
void VkCheck(VkResult r,const char* stage){if(r!=VK_SUCCESS)throw std::runtime_error(std::string(stage)+" error="+std::to_string(r));}
int main(int argc,char** argv) {
    try {
        if(argc<3)throw std::runtime_error("usage: gaussian_render_probe scene.mcgs output.rgba [width height [ratio threads fixedK cameraShift referencePrefix]]");
        SceneDescriptor identity;identity.slug="Replica__office2__rgbd";identity.gaussianCount=43197;identity.anchorCount=16723;
        identity.assetSha256="1d735b0871ff94cc6acbf039ad014e0d1490ca4e209caef92648a0b9e53a6685";
        identity.teacherFingerprint="9e2d413fa7864127465394fa22bb3da4e05d912d4c9e0822a4e928933aa36804";
        identity.studentCheckpointSha256="fd85285c6bd5871f4b41c85690597ccadc965d5a5ec6f474a983e4e09f547a75";
        SceneAsset scene;scene.LoadFile(argv[1],identity);
        FrameCamera camera;camera.width=argc>3?uint32_t(std::stoul(argv[3])):1200;camera.height=argc>4?uint32_t(std::stoul(argv[4])):680;
        camera.fx=600.f*camera.width/1200;camera.fy=600.f*camera.height/680;camera.cx=(camera.width-1)*.5f;camera.cy=(camera.height-1)*.5f;
        camera.center={-.8294860125f,-.8426836729f,1.7652033567f};
        const float r[9]={-.7053378820f,.4040099382f,-.5824726224f,-.1863143742f,.6871432066f,.7022258043f,.6839483976f,.6038295031f,-.4093953669f};
        const float shift=argc>8?std::stof(argv[8]):0.f;for(uint32_t c=0;c<3;++c)camera.center[c]+=shift*.08f*r[c*3];
        for(uint32_t i=0;i<3;++i){for(uint32_t j=0;j<3;++j)camera.view[i*4+j]=r[j*3+i];
            for(uint32_t j=0;j<3;++j)camera.view[i*4+3]-=camera.view[i*4+j]*camera.center[j];}camera.view[15]=1;
        std::array<float,16> projection{};projection[0]=2*camera.fx/camera.width;projection[5]=2*camera.fy/camera.height;
        projection[10]=camera.farPlane/(camera.farPlane-camera.nearPlane);projection[11]=-camera.farPlane*camera.nearPlane/(camera.farPlane-camera.nearPlane);projection[14]=1;
        for(uint32_t i=0;i<4;++i)for(uint32_t j=0;j<4;++j)for(uint32_t v=0;v<4;++v)camera.projection[i*4+j]+=projection[i*4+v]*camera.view[v*4+j];
        const bool full=argc<=5;const float ratio=full?1.f:std::stof(argv[5]);const uint32_t threads=argc>6?uint32_t(std::stoul(argv[6])):2;
        const uint32_t fixed=argc>7?uint32_t(std::stoul(argv[7])):0;const std::string reference=argc>9?argv[9]:"";
        QnnBackend unused;std::unique_ptr<ProbeBackend> network;
        if(!full){network=std::make_unique<ProbeBackend>(argv[2],threads,reference);network->Bind(scene);}
        GaussianFrame frame;frame.Bind(scene);const auto prepareStart=std::chrono::steady_clock::now();
        frame.Prepare(camera,full?static_cast<InferenceBackend&>(unused):static_cast<InferenceBackend&>(*network),ratio,fixed,full);
        const double prepareMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-prepareStart).count();
        ProbeWrite(std::string(argv[2])+"_selected.bin",frame.SelectedRows().data(),frame.SelectedRows().size());
        ProbeWrite(std::string(argv[2])+"_candidates.bin",frame.CandidateRows().data(),frame.CandidateRows().size());
        const auto projectStart=std::chrono::steady_clock::now();GaussianEye eye;frame.Project(camera,eye);
        const double projectMs=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-projectStart).count();
        const auto& stats=frame.Statistics();
        std::ofstream statsOut(std::string(argv[2])+"_stats.json");
        statsOut<<std::setprecision(9)<<"{\"candidates\":"<<stats.candidates<<",\"k\":"<<stats.k<<",\"ratio\":"<<ratio
            <<",\"fixed_k\":"<<fixed<<",\"camera_shift\":"<<shift<<",\"prepare_total_ms\":"<<prepareMs<<",\"project_ms\":"<<projectMs
            <<",\"selector_ms\":"<<stats.selectorMs<<",\"student_ms\":"<<stats.studentMs
            <<",\"max_xyz\":"<<stats.maxXyz<<",\"max_scale_ratio\":"<<stats.maxScaleRatio<<"}";
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};app.pApplicationName="MyCartGS render probe";app.apiVersion=VK_API_VERSION_1_1;
        VkInstanceCreateInfo instanceInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};instanceInfo.pApplicationInfo=&app;VkInstance instance;
        VkCheck(vkCreateInstance(&instanceInfo,nullptr,&instance),"vkCreateInstance");uint32_t devices=0;vkEnumeratePhysicalDevices(instance,&devices,nullptr);
        if(!devices)throw std::runtime_error("No Vulkan device");std::vector<VkPhysicalDevice> physical(devices);vkEnumeratePhysicalDevices(instance,&devices,physical.data());
        VkPhysicalDeviceProperties properties;vkGetPhysicalDeviceProperties(physical[0],&properties);std::cout<<"GPU "<<properties.deviceName<<std::endl;
        uint32_t families=0;vkGetPhysicalDeviceQueueFamilyProperties(physical[0],&families,nullptr);std::vector<VkQueueFamilyProperties> props(families);
        vkGetPhysicalDeviceQueueFamilyProperties(physical[0],&families,props.data());uint32_t family=0;
        while(family<families && !(props[family].queueFlags&VK_QUEUE_COMPUTE_BIT))++family;if(family==families)throw std::runtime_error("No compute queue");
        float priority=1;VkDeviceQueueCreateInfo queueInfo{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};queueInfo.queueFamilyIndex=family;queueInfo.queueCount=1;queueInfo.pQueuePriorities=&priority;
        VkDeviceCreateInfo deviceInfo{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};deviceInfo.queueCreateInfoCount=1;deviceInfo.pQueueCreateInfos=&queueInfo;VkDevice device;
        VkCheck(vkCreateDevice(physical[0],&deviceInfo,nullptr,&device),"vkCreateDevice");VkQueue queue;vkGetDeviceQueue(device,family,0,&queue);
        VkCommandPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};poolInfo.queueFamilyIndex=family;VkCommandPool pool;
        VkCheck(vkCreateCommandPool(device,&poolInfo,nullptr,&pool),"command pool");VkCommandBufferAllocateInfo allocate{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        allocate.commandPool=pool;allocate.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;allocate.commandBufferCount=1;VkCommandBuffer command;VkCheck(vkAllocateCommandBuffers(device,&allocate,&command),"command buffer");
        {
            VulkanGaussians renderer;renderer.Initialize(physical[0],device,nullptr);renderer.Upload(eye,VK_FORMAT_R8G8B8A8_UNORM);
            VkCommandBufferBeginInfo begin{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};VkCheck(vkBeginCommandBuffer(command,&begin),"begin");renderer.Dispatch(command);VkCheck(vkEndCommandBuffer(command),"end");
            VkSubmitInfo submit{VK_STRUCTURE_TYPE_SUBMIT_INFO};submit.commandBufferCount=1;submit.pCommandBuffers=&command;
            const auto start=std::chrono::steady_clock::now();VkCheck(vkQueueSubmit(queue,1,&submit,VK_NULL_HANDLE),"submit");VkCheck(vkQueueWaitIdle(queue),"wait");
            const double millis=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
            const float* pixels=static_cast<const float*>(renderer.Pixels());float maxError=0;
            for(uint32_t i=0;i<64;++i){uint32_t x=(i*7919+173)%camera.width,y=(i*6151+127)%camera.height;
                const uint32_t tile=(y/16)*((camera.width+15)/16)+x/16;const auto range=eye.ranges[tile];float transmission=1,color[3]={};
                for(uint32_t j=range.begin;j<range.end;++j){const auto& s=eye.splats[eye.members[j]];const float dx=s.centerRadius[0]-x,dy=s.centerRadius[1]-y;
                    const float power=-.5f*(s.conicAlpha[0]*dx*dx+s.conicAlpha[2]*dy*dy)-s.conicAlpha[1]*dx*dy;
                    if(power>0)continue;const float alpha=std::min(.99f,s.conicAlpha[3]*std::exp(power));if(alpha<1.f/255)continue;
                    const float next=transmission*(1-alpha);if(next<.0001f)break;
                    for(uint32_t c=0;c<3;++c)color[c]+=s.color[c]*alpha*transmission;transmission=next;}
                for(uint32_t c=0;c<3;++c)maxError=std::max(maxError,std::abs(color[c]-pixels[(size_t(y)*camera.width+x)*4+c]));}
            std::ofstream output(argv[2],std::ios::binary);output.write(reinterpret_cast<const char*>(pixels),size_t(camera.width)*camera.height*16);
            if(!output)throw std::runtime_error("Cannot write render output");
            std::cout<<"RENDER_PASS K="<<frame.Statistics().k<<" projected="<<eye.projected<<" instances="<<eye.members.size()
                     <<" size="<<camera.width<<"x"<<camera.height<<" gpu_submit_wait_ms="<<millis<<" sample_max_abs="<<maxError<<std::endl;
            if(maxError>2e-5f)throw std::runtime_error("Vulkan vs CPU alpha compositing tolerance exceeded");
        }
        vkDestroyCommandPool(device,pool,nullptr);vkDestroyDevice(device,nullptr);vkDestroyInstance(instance,nullptr);
    } catch(const std::exception& error){std::cerr<<"RENDER_FAIL "<<error.what()<<std::endl;return 1;}
}
