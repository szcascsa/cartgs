set(CMAKE_CURRENT_SOURCE_DIR "${VKROOT}")
include("${VKROOT}/cmake/shaders.cmake")
splatkit_compile_shaders(splatkit_quest_shaders
  rendering/vulkan/raster/splat.vert rendering/vulkan/raster/projected.vert rendering/vulkan/raster/splat.frag
  rendering/vulkan/cull/visibility.comp rendering/vulkan/cull/prepare_indirect.comp
  rendering/vulkan/selection/lod_selection.comp rendering/vulkan/sort/radix_prepare.comp
  rendering/vulkan/sort/radix_histogram.comp rendering/vulkan/sort/radix_scan.comp rendering/vulkan/sort/radix_scatter.comp)
