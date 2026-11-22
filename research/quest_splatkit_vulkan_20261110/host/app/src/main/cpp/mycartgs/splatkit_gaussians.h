#pragma once
#include "gaussian_frame.h"
#include <vulkan/vulkan.h>
#include <memory>

namespace mycartgs {
struct SplatKitMetrics {
    uint32_t input=0,drawn=0;
    uint64_t uploadSerial=0;
    double uploadMs=0,cullMs=0,sortMs=0,rasterMs=0,pipelineMs=0;
};
// The upstream SplatKit cull, stable radix sort and raster stages. This class owns host wiring only.
class SplatKitGaussians {
public:
    SplatKitGaussians();
    ~SplatKitGaussians();
    void Initialize(VkInstance instance,VkPhysicalDevice physical,VkDevice device,VkQueue queue,uint32_t family);
    void Upload(const SceneAsset& scene,const std::vector<uint32_t>& selected,
                const std::vector<DecodedAttribute>& decoded,uint64_t revision);
    void Prepare(const FrameCamera& camera,VkRenderPass pass);
    void Compute(VkCommandBuffer command);
    void Draw(VkCommandBuffer command);
    void End(VkCommandBuffer command);
    void Offscreen(VkCommandBuffer command,const FrameCamera& camera);
    void Collect(); // Caller has waited for this queue submission.
    const SplatKitMetrics& Metrics() const;
    std::array<uint64_t,4> ImageDiagnostics(std::string* rgbaSha256=nullptr,std::vector<uint8_t>* rgba=nullptr) const; // Untimed readback only.
    void Close();
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
