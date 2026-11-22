#pragma once
#include "gaussian_frame.h"
#include <vulkan/vulkan.h>
#include <array>
#include <string>
struct AAssetManager;

namespace mycartgs {
class VulkanGaussians {
public:
    ~VulkanGaussians();
    void Initialize(VkPhysicalDevice physical,VkDevice device,AAssetManager* assets);
    void Upload(const GaussianEye& eye,VkFormat targetFormat);
    void Dispatch(VkCommandBuffer command,bool timed=false);
    void EnableTiming(uint32_t queueFamily);
    double RasterMilliseconds() const;
    void Draw(VkCommandBuffer command,VkRenderPass renderPass);
    const void* Pixels() const { return buffers_[3].mapped; }
    void Close();
private:
    struct Buffer { VkBuffer buffer=VK_NULL_HANDLE; VkDeviceMemory memory=VK_NULL_HANDLE; void* mapped=nullptr; VkDeviceSize capacity=0; };
    void Ensure(Buffer& buffer,VkDeviceSize bytes);
    void Graphics(VkRenderPass pass);
    VkShaderModule Shader(const char* asset);
    VkPhysicalDevice physical_=VK_NULL_HANDLE;
    VkDevice device_=VK_NULL_HANDLE;
    AAssetManager* assets_=nullptr;
    VkDescriptorSetLayout setLayout_=VK_NULL_HANDLE;
    VkDescriptorPool pool_=VK_NULL_HANDLE;
    VkDescriptorSet set_=VK_NULL_HANDLE;
    VkPipelineLayout layout_=VK_NULL_HANDLE;
    VkPipeline compute_=VK_NULL_HANDLE,graphics_=VK_NULL_HANDLE;
    VkQueryPool timestamps_=VK_NULL_HANDLE;
    uint32_t timestampBits_=0;
    float timestampPeriod_=0;
    std::array<Buffer,4> buffers_{};
    struct Params { uint32_t width=0,height=0,tilesX=0,srgb=0; float background[4]={0,0,0,1}; } params_;
};
}  // namespace mycartgs
