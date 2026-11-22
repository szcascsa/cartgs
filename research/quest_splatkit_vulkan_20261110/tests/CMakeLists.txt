cmake_minimum_required(VERSION 3.22)
project(splatkit_quest_probe LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
add_subdirectory(../cmake splatkit)
set(CORE "${CMAKE_CURRENT_SOURCE_DIR}/../host/app/src/main/cpp/mycartgs")
set(JSON "Q:/third_party/OpenXR-SDK-Source/src/external/jsoncpp")
add_library(baseline_adapter STATIC ${CORE}/splatkit_gaussians.cpp ${CORE}/scene_asset.cpp ${CORE}/gaussian_frame.cpp
  ${CORE}/decode.cpp ${CORE}/residual48.cpp ${CORE}/strict_topk.cpp ${CORE}/sha256.cpp
  ${JSON}/src/lib_json/json_reader.cpp ${JSON}/src/lib_json/json_value.cpp ${JSON}/src/lib_json/json_writer.cpp)
target_compile_options(baseline_adapter PRIVATE -O3)
target_include_directories(baseline_adapter PUBLIC ${CORE} ${JSON}/include)
target_link_libraries(baseline_adapter PUBLIC splatkit_quest)
add_executable(splatkit_point_probe splatkit_point_probe.cpp)
target_link_libraries(splatkit_point_probe PRIVATE baseline_adapter)

set(UPSTREAM_TESTS "${CMAKE_CURRENT_SOURCE_DIR}/../source/splatkit/packages/splatkit-android/tests")
foreach(test camera_projection splat_pipeline radix_sort frame_compute)
  if(test STREQUAL "camera_projection")
    set(file frame/VulkanCameraProjectionTest.cpp)
  elseif(test STREQUAL "splat_pipeline")
    set(file raster/VulkanSplatPipelineTest.cpp)
  elseif(test STREQUAL "radix_sort")
    set(file sort/VulkanRadixSortTest.cpp)
  else()
    set(file frame/VulkanFrameComputeTest.cpp)
  endif()
  add_executable(splatkit_${test}_test ${UPSTREAM_TESTS}/rendering/vulkan/${file})
  target_include_directories(splatkit_${test}_test PRIVATE ${UPSTREAM_TESTS})
  target_link_libraries(splatkit_${test}_test PRIVATE splatkit_quest)
endforeach()

add_executable(stereo_runtime_probe stereo_runtime_probe.cpp
  ${CORE}/quest_runtime.cpp ${CORE}/scene_menu.cpp ${CORE}/xnnpack_backend.cpp)
target_include_directories(stereo_runtime_probe PRIVATE Q:/third_party/XNNPACK/include Q:/third_party/pthreadpool/include Q:/third_party/cpuinfo/include)
target_link_libraries(stereo_runtime_probe PRIVATE baseline_adapter -Wl,--start-group
  Q:/build-device/xnnpack/libXNNPACK.a Q:/build-device/xnnpack/libxnnpack-microkernels-prod.a
  Q:/build-device/pthreadpool/libpthreadpool.a Q:/build-device/cpuinfo/libcpuinfo.a -Wl,--end-group)

add_library(cpu_frozen_reference STATIC cpu_reference/gaussian_frame_reference.cpp cpu_reference/xnnpack_backend_reference.cpp)
target_compile_options(cpu_frozen_reference PRIVATE -O3)
target_include_directories(cpu_frozen_reference PUBLIC cpu_reference
  Q:/third_party/XNNPACK/include Q:/third_party/pthreadpool/include Q:/third_party/cpuinfo/include)
target_link_libraries(cpu_frozen_reference PUBLIC baseline_adapter -Wl,--start-group
  Q:/build-device/xnnpack/libXNNPACK.a Q:/build-device/xnnpack/libxnnpack-microkernels-prod.a
  Q:/build-device/pthreadpool/libpthreadpool.a Q:/build-device/cpuinfo/libcpuinfo.a -Wl,--end-group)
target_link_libraries(stereo_runtime_probe PRIVATE cpu_frozen_reference)
add_executable(cpu_optimization_probe cpu_optimization_probe.cpp ${CORE}/xnnpack_backend.cpp)
target_compile_options(cpu_optimization_probe PRIVATE -O3)
target_link_libraries(cpu_optimization_probe PRIVATE cpu_frozen_reference)
option(MYCARTGS_CPU_SANITIZE "Instrument the CPU inference probe and adapter" OFF)
if(MYCARTGS_CPU_SANITIZE)
  foreach(target baseline_adapter cpu_frozen_reference cpu_optimization_probe)
    target_compile_options(${target} PRIVATE -fsanitize=address,undefined -fno-sanitize=vptr -fno-omit-frame-pointer)
  endforeach()
  target_link_options(cpu_optimization_probe PRIVATE -fsanitize=address,undefined -fno-sanitize=vptr)
  add_executable(sanitizer_stdlib_probe sanitizer_stdlib_probe.cpp)
  target_compile_options(sanitizer_stdlib_probe PRIVATE -fsanitize=address,undefined -fno-omit-frame-pointer)
  target_link_options(sanitizer_stdlib_probe PRIVATE -fsanitize=address,undefined)
endif()
