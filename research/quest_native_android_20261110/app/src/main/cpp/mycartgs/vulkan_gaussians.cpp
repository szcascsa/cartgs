#include "vulkan_gaussians.h"
#include "point_benchmark.h"
#include <android/asset_manager.h>
#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace mycartgs {
namespace {
void Vk(VkResult value,const char* stage){if(value!=VK_SUCCESS)throw std::runtime_error(std::string(stage)+" VkResult="+std::to_string(value));}
}
VulkanGaussians::~VulkanGaussians(){Close();}
void VulkanGaussians::Close(){
    if(!device_)return;
    if(timestamps_)vkDestroyQueryPool(device_,timestamps_,nullptr);
    timestamps_=VK_NULL_HANDLE;timestampBits_=0;timestampPeriod_=0;
    if(graphics_)vkDestroyPipeline(device_,graphics_,nullptr);if(compute_)vkDestroyPipeline(device_,compute_,nullptr);
    if(layout_)vkDestroyPipelineLayout(device_,layout_,nullptr);if(pool_)vkDestroyDescriptorPool(device_,pool_,nullptr);
    if(setLayout_)vkDestroyDescriptorSetLayout(device_,setLayout_,nullptr);
    for(auto& b:buffers_){if(b.mapped)vkUnmapMemory(device_,b.memory);if(b.buffer)vkDestroyBuffer(device_,b.buffer,nullptr);if(b.memory)vkFreeMemory(device_,b.memory,nullptr);b={};}
    device_=VK_NULL_HANDLE;graphics_=compute_=VK_NULL_HANDLE;layout_=VK_NULL_HANDLE;pool_=VK_NULL_HANDLE;setLayout_=VK_NULL_HANDLE;
}
VkShaderModule VulkanGaussians::Shader(const char* path){
    if(!assets_) {
        std::ifstream file(path,std::ios::binary|std::ios::ate);
        if(!file)throw std::runtime_error(std::string("Probe shader missing: ")+path);
        const size_t bytes=size_t(file.tellg());std::vector<uint32_t> words((bytes+3)/4);
        file.seekg(0);file.read(reinterpret_cast<char*>(words.data()),bytes);
        if(!file||bytes%4)throw std::runtime_error("Invalid probe SPIR-V");
        VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};info.codeSize=bytes;info.pCode=words.data();
        VkShaderModule shader;Vk(vkCreateShaderModule(device_,&info,nullptr,&shader),"probe shader");return shader;
    }
    AAsset* asset=AAssetManager_open(assets_,path,AASSET_MODE_BUFFER);
    if(!asset)throw std::runtime_error(std::string("Missing Vulkan shader ")+path);
    const size_t bytes=size_t(AAsset_getLength64(asset));
    std::vector<uint32_t> words((bytes+3)/4);
    if(bytes%4 || AAsset_read(asset,words.data(),bytes)!=int(bytes)){AAsset_close(asset);throw std::runtime_error("Invalid SPIR-V asset");}
    AAsset_close(asset);VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    info.codeSize=bytes;info.pCode=words.data();VkShaderModule shader;Vk(vkCreateShaderModule(device_,&info,nullptr,&shader),"shader");return shader;
}
void VulkanGaussians::Initialize(VkPhysicalDevice physical,VkDevice device,AAssetManager* assets){
    if(device_)return;device_=device;physical_=physical;assets_=assets;
    std::array<VkDescriptorSetLayoutBinding,4> bindings{};
    for(uint32_t i=0;i<4;++i){bindings[i].binding=i;bindings[i].descriptorCount=1;bindings[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bindings[i].stageFlags=VK_SHADER_STAGE_COMPUTE_BIT|(i==3?VK_SHADER_STAGE_FRAGMENT_BIT:0);}
    VkDescriptorSetLayoutCreateInfo desc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};desc.bindingCount=4;desc.pBindings=bindings.data();
    Vk(vkCreateDescriptorSetLayout(device_,&desc,nullptr,&setLayout_),"descriptor layout");
    VkDescriptorPoolSize size{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,4};
    VkDescriptorPoolCreateInfo pool{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};pool.maxSets=1;pool.poolSizeCount=1;pool.pPoolSizes=&size;
    Vk(vkCreateDescriptorPool(device_,&pool,nullptr,&pool_),"descriptor pool");
    VkDescriptorSetAllocateInfo alloc{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};alloc.descriptorPool=pool_;alloc.descriptorSetCount=1;alloc.pSetLayouts=&setLayout_;
    Vk(vkAllocateDescriptorSets(device_,&alloc,&set_),"descriptor set");
    VkPushConstantRange range{VK_SHADER_STAGE_COMPUTE_BIT|VK_SHADER_STAGE_FRAGMENT_BIT,0,sizeof(Params)};
    VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};layout.setLayoutCount=1;layout.pSetLayouts=&setLayout_;
    layout.pushConstantRangeCount=1;layout.pPushConstantRanges=&range;Vk(vkCreatePipelineLayout(device_,&layout,nullptr,&layout_),"Gaussian layout");
    VkShaderModule shader=Shader("shaders/raster.comp.spv");
    VkComputePipelineCreateInfo pipeline{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};pipeline.layout=layout_;
    pipeline.stage={VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_COMPUTE_BIT,shader,"main",nullptr};
    const VkResult result=vkCreateComputePipelines(device_,VK_NULL_HANDLE,1,&pipeline,nullptr,&compute_);vkDestroyShaderModule(device_,shader,nullptr);Vk(result,"Gaussian compute pipeline");
}
void VulkanGaussians::Ensure(Buffer& b,VkDeviceSize bytes){
    bytes=std::max<VkDeviceSize>(bytes,16);if(b.capacity>=bytes)return;
    VkPhysicalDeviceProperties properties;vkGetPhysicalDeviceProperties(physical_,&properties);
    if(bytes>std::min<VkDeviceSize>(128ull*1024*1024,properties.limits.maxStorageBufferRange))throw std::runtime_error("Gaussian GPU buffer exceeds device/bounded capacity");
    if(b.mapped)vkUnmapMemory(device_,b.memory);if(b.buffer)vkDestroyBuffer(device_,b.buffer,nullptr);if(b.memory)vkFreeMemory(device_,b.memory,nullptr);
    b={};b.capacity=(bytes+4095)&~VkDeviceSize(4095);
    VkBufferCreateInfo info{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};info.size=b.capacity;info.usage=VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;info.sharingMode=VK_SHARING_MODE_EXCLUSIVE;
    Vk(vkCreateBuffer(device_,&info,nullptr,&b.buffer),"Gaussian buffer");
    VkMemoryRequirements requirements;vkGetBufferMemoryRequirements(device_,b.buffer,&requirements);
    VkPhysicalDeviceMemoryProperties memory;vkGetPhysicalDeviceMemoryProperties(physical_,&memory);uint32_t type=UINT32_MAX;
    for(uint32_t i=0;i<memory.memoryTypeCount;++i)if((requirements.memoryTypeBits&(1u<<i)) &&
       (memory.memoryTypes[i].propertyFlags&(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT))==
        (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT|VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)){type=i;break;}
    if(type==UINT32_MAX)throw std::runtime_error("Host coherent Vulkan storage memory unavailable");
    VkMemoryAllocateInfo alloc{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};alloc.allocationSize=requirements.size;alloc.memoryTypeIndex=type;
    Vk(vkAllocateMemory(device_,&alloc,nullptr,&b.memory),"Gaussian buffer allocation");Vk(vkBindBufferMemory(device_,b.buffer,b.memory,0),"Gaussian buffer binding");
    Vk(vkMapMemory(device_,b.memory,0,b.capacity,0,&b.mapped),"Gaussian buffer map");
}
void VulkanGaussians::Upload(const GaussianEye& eye,VkFormat format){
    params_.width=eye.width;params_.height=eye.height;params_.tilesX=(eye.width+15)/16;
    params_.srgb=format==VK_FORMAT_R8G8B8A8_SRGB||format==VK_FORMAT_B8G8R8A8_SRGB;
    const size_t sizes[]={eye.splats.size()*sizeof(ProjectedSplat),eye.ranges.size()*sizeof(TileRange),eye.members.size()*sizeof(uint32_t),size_t(eye.width)*eye.height*16};
    const void* pointers[]={eye.splats.data(),eye.ranges.data(),eye.members.data(),nullptr};
    std::array<VkDescriptorBufferInfo,4> bufferInfo{};std::array<VkWriteDescriptorSet,4> writes{};
    for(uint32_t i=0;i<4;++i){Ensure(buffers_[i],sizes[i]);if(pointers[i]&&sizes[i])std::memcpy(buffers_[i].mapped,pointers[i],sizes[i]);
        bufferInfo[i]={buffers_[i].buffer,0,buffers_[i].capacity};writes[i]={VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        writes[i].dstSet=set_;writes[i].dstBinding=i;writes[i].descriptorCount=1;writes[i].descriptorType=VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;writes[i].pBufferInfo=&bufferInfo[i];}
    vkUpdateDescriptorSets(device_,4,writes.data(),0,nullptr);
}
void VulkanGaussians::EnableTiming(uint32_t queueFamily){
    if(timestamps_)return;
    uint32_t count=0;vkGetPhysicalDeviceQueueFamilyProperties(physical_,&count,nullptr);
    std::vector<VkQueueFamilyProperties> queues(count);vkGetPhysicalDeviceQueueFamilyProperties(physical_,&count,queues.data());
    if(queueFamily>=count || !queues[queueFamily].timestampValidBits)
        throw std::runtime_error("Selected Vulkan queue does not support GPU timestamps");
    VkPhysicalDeviceProperties p;vkGetPhysicalDeviceProperties(physical_,&p);
    timestampBits_=queues[queueFamily].timestampValidBits;timestampPeriod_=p.limits.timestampPeriod;
    VkQueryPoolCreateInfo info{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};info.queryType=VK_QUERY_TYPE_TIMESTAMP;info.queryCount=2;
    Vk(vkCreateQueryPool(device_,&info,nullptr,&timestamps_),"Gaussian timestamp query pool");
}
double VulkanGaussians::RasterMilliseconds() const {
    if(!timestamps_)throw std::runtime_error("GPU timestamps not initialized");
    uint64_t ticks[2]{};
    Vk(vkGetQueryPoolResults(device_,timestamps_,0,2,sizeof(ticks),ticks,sizeof(uint64_t),VK_QUERY_RESULT_64_BIT),"Gaussian timestamp results after fence");
    return double(TimestampTicks(ticks[0],ticks[1],timestampBits_))*timestampPeriod_/1e6;
}
void VulkanGaussians::Dispatch(VkCommandBuffer command,bool timed){
    if(timed){if(!timestamps_)throw std::runtime_error("GPU timing unavailable");
        vkCmdResetQueryPool(command,timestamps_,0,2);vkCmdWriteTimestamp(command,VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,timestamps_,0);}
    VkMemoryBarrier host{VK_STRUCTURE_TYPE_MEMORY_BARRIER};host.srcAccessMask=VK_ACCESS_HOST_WRITE_BIT;host.dstAccessMask=VK_ACCESS_SHADER_READ_BIT;
    vkCmdPipelineBarrier(command,VK_PIPELINE_STAGE_HOST_BIT,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,0,1,&host,0,nullptr,0,nullptr);
    vkCmdBindPipeline(command,VK_PIPELINE_BIND_POINT_COMPUTE,compute_);vkCmdBindDescriptorSets(command,VK_PIPELINE_BIND_POINT_COMPUTE,layout_,0,1,&set_,0,nullptr);
    vkCmdPushConstants(command,layout_,VK_SHADER_STAGE_COMPUTE_BIT|VK_SHADER_STAGE_FRAGMENT_BIT,0,sizeof(Params),&params_);
    vkCmdDispatch(command,params_.tilesX,(params_.height+15)/16,1);
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};barrier.srcAccessMask=VK_ACCESS_SHADER_WRITE_BIT;barrier.dstAccessMask=VK_ACCESS_SHADER_READ_BIT|VK_ACCESS_HOST_READ_BIT;
    vkCmdPipelineBarrier(command,VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT|VK_PIPELINE_STAGE_HOST_BIT,0,1,&barrier,0,nullptr,0,nullptr);
    if(timed)vkCmdWriteTimestamp(command,VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT,timestamps_,1);
}
void VulkanGaussians::Graphics(VkRenderPass pass){
    const VkShaderModule vertex=Shader("shaders/present.vert.spv"),fragment=Shader("shaders/present.frag.spv");
    VkPipelineShaderStageCreateInfo shaders[2]={{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_VERTEX_BIT,vertex,"main",nullptr},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,nullptr,0,VK_SHADER_STAGE_FRAGMENT_BIT,fragment,"main",nullptr}};
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};ia.topology=VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo viewport{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};viewport.viewportCount=viewport.scissorCount=1;
    VkPipelineRasterizationStateCreateInfo raster{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};raster.polygonMode=VK_POLYGON_MODE_FILL;raster.cullMode=VK_CULL_MODE_NONE;raster.lineWidth=1;
    VkPipelineMultisampleStateCreateInfo samples{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};samples.rasterizationSamples=VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo depth{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    VkPipelineColorBlendAttachmentState blend{};blend.colorWriteMask=15;
    VkPipelineColorBlendStateCreateInfo color{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};color.attachmentCount=1;color.pAttachments=&blend;
    VkDynamicState dynamicStates[]={VK_DYNAMIC_STATE_VIEWPORT,VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dynamic{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};dynamic.dynamicStateCount=2;dynamic.pDynamicStates=dynamicStates;
    VkGraphicsPipelineCreateInfo info{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};info.stageCount=2;info.pStages=shaders;
    info.pVertexInputState=&vi;info.pInputAssemblyState=&ia;info.pViewportState=&viewport;info.pRasterizationState=&raster;
    info.pMultisampleState=&samples;info.pDepthStencilState=&depth;info.pColorBlendState=&color;info.pDynamicState=&dynamic;info.layout=layout_;info.renderPass=pass;
    const auto result=vkCreateGraphicsPipelines(device_,VK_NULL_HANDLE,1,&info,nullptr,&graphics_);
    vkDestroyShaderModule(device_,vertex,nullptr);vkDestroyShaderModule(device_,fragment,nullptr);Vk(result,"Gaussian present pipeline");
}
void VulkanGaussians::Draw(VkCommandBuffer command,VkRenderPass pass){
    if(!graphics_)Graphics(pass);
    vkCmdBindPipeline(command,VK_PIPELINE_BIND_POINT_GRAPHICS,graphics_);vkCmdBindDescriptorSets(command,VK_PIPELINE_BIND_POINT_GRAPHICS,layout_,0,1,&set_,0,nullptr);
    vkCmdPushConstants(command,layout_,VK_SHADER_STAGE_COMPUTE_BIT|VK_SHADER_STAGE_FRAGMENT_BIT,0,sizeof(Params),&params_);
    vkCmdDraw(command,3,1,0,0);
}
}  // namespace mycartgs
