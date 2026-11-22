cmake_minimum_required(VERSION 3.22)
project(splatkit_quest_baseline LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
get_filename_component(STUDY "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(PACKAGES "${STUDY}/source/splatkit/packages")
set(VKROOT "${PACKAGES}/splatkit-android/src/main/cpp")
set(Vulkan_INCLUDE_DIR "${STUDY}/deps/Vulkan-Headers/include" CACHE PATH "" FORCE)
set(Vulkan_INCLUDE_DIRS "${Vulkan_INCLUDE_DIR}")
find_library(Vulkan_LIBRARY vulkan REQUIRED)
set(Vulkan_LIBRARY "${Vulkan_LIBRARY}" CACHE FILEPATH "" FORCE)
set(VK_BOOTSTRAP_TEST OFF CACHE BOOL "" FORCE)
set(VK_BOOTSTRAP_POSITION_INDEPENDENT_CODE ON CACHE BOOL "" FORCE)
add_subdirectory("${STUDY}/deps/vk-bootstrap" vk-bootstrap EXCLUDE_FROM_ALL)
set(VK_VERSION_HEADERS "${STUDY}/deps/Vulkan-Headers/include")
add_library(splatkit_quest STATIC
  ${VKROOT}/rendering/vulkan/device/VulkanContext.cpp
  ${VKROOT}/rendering/vulkan/device/VulkanBuffer.cpp
  ${VKROOT}/rendering/vulkan/device/VulkanPipelineCache.cpp
  ${VKROOT}/rendering/vulkan/device/Vma.cpp
  ${VKROOT}/rendering/vulkan/world/VulkanWorld.cpp
  ${VKROOT}/rendering/vulkan/cull/VulkanVisibilityPass.cpp
  ${VKROOT}/rendering/vulkan/sort/VulkanRadixSort.cpp
  ${VKROOT}/rendering/vulkan/selection/VulkanLodSelection.cpp
  ${VKROOT}/rendering/vulkan/frame/VulkanFrameCompute.cpp
  ${VKROOT}/rendering/vulkan/frame/VulkanCameraUniforms.cpp
  ${VKROOT}/rendering/vulkan/raster/VulkanSplatPipeline.cpp
  ${VKROOT}/rendering/vulkan/presentation/VulkanRenderTarget.cpp
  ${PACKAGES}/splatkit-engine/src/Log.cpp
  ${PACKAGES}/splatkit-engine/src/rendering/GpuLayout.cpp
  ${PACKAGES}/splatkit-engine/src/rendering/RenderPolicy.cpp
  ${PACKAGES}/splat-core/src/lod/LodSelection.cpp
  ${PACKAGES}/splat-core/src/lod/LodFile.cpp
  ${PACKAGES}/splat-core/src/lod/LodTree.cpp
  ${PACKAGES}/splat-core/src/math/SymmetricEigen.cpp
)
target_include_directories(splatkit_quest PUBLIC ${Vulkan_INCLUDE_DIR} ${VKROOT}
  ${PACKAGES}/splat-core/include ${PACKAGES}/splatkit-engine/include)
target_include_directories(splatkit_quest SYSTEM PUBLIC ${STUDY}/deps/vk-bootstrap/src ${STUDY}/deps/vma/include)
target_compile_definitions(splatkit_quest PUBLIC VK_USE_PLATFORM_ANDROID_KHR SPLATKIT_VULKAN_SH8=0)
target_compile_options(splatkit_quest PRIVATE -O3 -Wno-unused-parameter -Wno-missing-field-initializers)
target_link_libraries(splatkit_quest PUBLIC vk-bootstrap::vk-bootstrap vulkan android log)
set(SPLATKIT_VULKAN_SH8 OFF)
# Preserve the upstream shader compilation flags and GLSL sources.
add_subdirectory(shaders)
target_link_libraries(splatkit_quest PUBLIC splatkit_quest_shaders)
foreach(dependency splatkit_quest vk-bootstrap)
  target_compile_options(${dependency} PRIVATE -Wno-error
    -Wno-error=unused-parameter -Wno-error=unused-argument -Wno-error=missing-prototypes
    -Wno-error=implicit-fallthrough -Wno-error=undef -Wno-error=tautological-type-limit-compare
    -Wno-error=missing-braces -Wno-error=unreachable-code -Wno-error=c++20-extensions)
endforeach()
