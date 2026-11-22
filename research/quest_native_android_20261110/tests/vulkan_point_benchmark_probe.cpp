#include "gaussian_frame.h"
#include "point_benchmark.h"
#include "vulkan_gaussians.h"
#include <vulkan/vulkan.h>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>

using namespace mycartgs;
void Check(bool ok,const char* message){if(!ok)throw std::runtime_error(message);}
void VkCheck(VkResult r,const char* stage){if(r!=VK_SUCCESS)throw std::runtime_error(std::string(stage)+" VkResult="+std::to_string(r));}
struct NoNetwork final : InferenceBackend {
    InferenceStatus state;
    void Bind(const SceneAsset&) override {}
    const InferenceStatus& Status() const override {return state;}
    void RunSelector(uint32_t,const float*,const float*,float*) override {throw std::runtime_error("Unexpected Selector call");}
    void RunStudent(uint32_t,const float*,float*) override {throw std::runtime_error("Unexpected Student call");}
};
using Clock=std::chrono::steady_clock;
double Ms(Clock::time_point begin){return std::chrono::duration<double,std::milli>(Clock::now()-begin).count();}
int main(int argc,char** argv) {
    try {
        Check(argc==3||argc==5,"usage: vulkan_point_benchmark_probe scene.mcgs output.csv [width height]");
        SceneDescriptor id;id.slug="Replica__office2__rgbd";id.gaussianCount=43197;id.anchorCount=16723;
        id.assetSha256="1d735b0871ff94cc6acbf039ad014e0d1490ca4e209caef92648a0b9e53a6685";
        id.teacherFingerprint="9e2d413fa7864127465394fa22bb3da4e05d912d4c9e0822a4e928933aa36804";
        id.studentCheckpointSha256="fd85285c6bd5871f4b41c85690597ccadc965d5a5ec6f474a983e4e09f547a75";
        SceneAsset scene;scene.LoadFile(argv[1],id);NoNetwork network;GaussianFrame frame;frame.Bind(scene);
        FrameCamera camera;camera.width=argc==5?uint32_t(std::stoul(argv[3])):1200;camera.height=argc==5?uint32_t(std::stoul(argv[4])):680;
        camera.fx=600.f*camera.width/1200;camera.fy=600.f*camera.height/680;camera.cx=(camera.width-1)*.5f;camera.cy=(camera.height-1)*.5f;
        camera.center={-.8294860125f,-.8426836729f,1.7652033567f};
        const float r[9]={-.7053378820f,.4040099382f,-.5824726224f,-.1863143742f,.6871432066f,.7022258043f,.6839483976f,.6038295031f,-.4093953669f};
        for(uint32_t i=0;i<3;++i){for(uint32_t j=0;j<3;++j)camera.view[i*4+j]=r[j*3+i];
            for(uint32_t j=0;j<3;++j)camera.view[i*4+3]-=camera.view[i*4+j]*camera.center[j];}camera.view[15]=1;
        frame.Prepare(camera,network,1.f,0,true);GaussianEye original;frame.Project(camera,original);
        frame.PreparePointBenchmark(id.gaussianCount);GaussianEye full;frame.Project(camera,full);
        Check(full.projected==original.projected&&full.members==original.members,"Full point benchmark topology differs");
        Check(full.splats.size()==original.splats.size()&&full.ranges.size()==original.ranges.size(),"Full shapes differ");
        Check(!std::memcmp(full.splats.data(),original.splats.data(),full.splats.size()*sizeof(ProjectedSplat)),"Full projected attributes not bitwise identical");
        Check(!std::memcmp(full.ranges.data(),original.ranges.data(),full.ranges.size()*sizeof(TileRange)),"Full tile ranges differ");
        VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};app.apiVersion=VK_API_VERSION_1_1;
        VkInstanceCreateInfo instanceInfo{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};instanceInfo.pApplicationInfo=&app;VkInstance instance;
        VkCheck(vkCreateInstance(&instanceInfo,nullptr,&instance),"instance");uint32_t count=0;vkEnumeratePhysicalDevices(instance,&count,nullptr);Check(count>0,"No GPU");
        std::vector<VkPhysicalDevice> physical(count);vkEnumeratePhysicalDevices(instance,&count,physical.data());
        VkPhysicalDeviceProperties props;vkGetPhysicalDeviceProperties(physical[0],&props);
        uint32_t families=0;vkGetPhysicalDeviceQueueFamilyProperties(physical[0],&families,nullptr);
        std::vector<VkQueueFamilyProperties> queues(families);vkGetPhysicalDeviceQueueFamilyProperties(physical[0],&families,queues.data());
        uint32_t family=0;while(family<families&&((queues[family].queueFlags&(VK_QUEUE_GRAPHICS_BIT|VK_QUEUE_COMPUTE_BIT))!=(VK_QUEUE_GRAPHICS_BIT|VK_QUEUE_COMPUTE_BIT)))++family;
        Check(family<families,"No graphics/compute queue");
        std::cout<<"GPU="<<props.deviceName<<" timestamp_bits="<<queues[family].timestampValidBits<<" period_ns="<<props.limits.timestampPeriod<<std::endl;
        float priority=1;VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};qi.queueFamilyIndex=family;qi.queueCount=1;qi.pQueuePriorities=&priority;
        VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};di.queueCreateInfoCount=1;di.pQueueCreateInfos=&qi;VkDevice device;VkCheck(vkCreateDevice(physical[0],&di,nullptr,&device),"device");
        VkQueue queue;vkGetDeviceQueue(device,family,0,&queue);
        VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};pi.queueFamilyIndex=family;pi.flags=VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        VkCommandPool pool;VkCheck(vkCreateCommandPool(device,&pi,nullptr,&pool),"pool");
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};ai.commandPool=pool;ai.level=VK_COMMAND_BUFFER_LEVEL_PRIMARY;ai.commandBufferCount=1;
        VkCommandBuffer command;VkCheck(vkAllocateCommandBuffers(device,&ai,&command),"command");
        {
            VulkanGaussians renderer;renderer.Initialize(physical[0],device,nullptr);renderer.EnableTiming(family);
            auto submit=[&](bool timed){
                VkCheck(vkResetCommandBuffer(command,0),"reset");VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
                VkCheck(vkBeginCommandBuffer(command,&bi),"begin");renderer.Dispatch(command,timed);VkCheck(vkEndCommandBuffer(command),"end");
                VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};si.commandBufferCount=1;si.pCommandBuffers=&command;
                const auto begin=Clock::now();VkCheck(vkQueueSubmit(queue,1,&si,VK_NULL_HANDLE),"submit");VkCheck(vkQueueWaitIdle(queue),"wait");return Ms(begin);
            };
            renderer.Upload(original,VK_FORMAT_R8G8B8A8_UNORM);submit(false);
            std::vector<float> reference(size_t(camera.width)*camera.height*4);std::memcpy(reference.data(),renderer.Pixels(),reference.size()*sizeof(float));
            renderer.Upload(full,VK_FORMAT_R8G8B8A8_UNORM);submit(true);
            Check(!std::memcmp(reference.data(),renderer.Pixels(),reference.size()*sizeof(float)),"Timestamp instrumentation changed RGB/alpha bits");
            PointBenchmark benchmark;benchmark.Start(id.gaussianCount);uint32_t prepared=UINT32_MAX;
            PointBenchmarkMetrics metrics;GaussianEye eye;std::map<uint32_t,std::vector<uint32_t>> identities;
            std::vector<uint32_t> previous;
            while(benchmark.Running()) {
                if(prepared!=benchmark.Stage()) {
                    const auto begin=Clock::now();frame.PreparePointBenchmark(benchmark.K());frame.Project(camera,eye);
                    const auto& ids=frame.SelectedRows();Check(ids.size()==benchmark.K(),"Strict benchmark K failed");
                    if(benchmark.Round()==0){Check(std::includes(ids.begin(),ids.end(),previous.begin(),previous.end()),"Subsets are not nested");previous=ids;identities[benchmark.K()]=ids;}
                    else Check(ids==identities.at(benchmark.K()),"Reverse round changed selected IDs");
                    metrics={};metrics.prepareMs=Ms(begin);metrics.projected=eye.projected;metrics.width=camera.width;metrics.height=camera.height;metrics.instances=eye.members.size();
                    for(const auto& range:eye.ranges)metrics.activeTiles+=range.end>range.begin;
                    prepared=benchmark.Stage();
                }
                const auto begin=Clock::now();renderer.Upload(eye,VK_FORMAT_R8G8B8A8_UNORM);metrics.uploadMs=Ms(begin);
                metrics.submitWaitMs=submit(true);metrics.gpuMs=renderer.RasterMilliseconds();benchmark.Record(metrics);
            }
            std::ofstream csv(argv[2]);csv<<benchmark.Csv();csv.close();Check(bool(csv),"CSV write failed");
            for(const auto& s:benchmark.Summaries())std::cout<<"K="<<s.k<<" projected="<<s.projected<<" samples="<<s.samples<<" gpu_median_ms="<<s.medianMs<<" p95_ms="<<s.p95Ms<<std::endl;
            Check(benchmark.Samples().size()==324,"Device benchmark sample count mismatch");
            std::cout<<"POINT_BENCHMARK_PASS Full attributes/tiles/RGBA bitwise, nested K, empty, workspace reuse, 324 timestamp samples\n";
        }
        vkDestroyCommandPool(device,pool,nullptr);vkDestroyDevice(device,nullptr);vkDestroyInstance(instance,nullptr);
    } catch(const std::exception& e){std::cerr<<"POINT_BENCHMARK_FAIL "<<e.what()<<std::endl;return 1;}
}
