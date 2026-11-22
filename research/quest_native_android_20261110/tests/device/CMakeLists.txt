cmake_minimum_required(VERSION 3.22)
project(mycartgs_qnn_probe CXX)
set(CMAKE_CXX_STANDARD 17)
get_filename_component(STUDY_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
set(CORE "${STUDY_ROOT}/app/src/main/cpp/mycartgs")
set(SDK "${STUDY_ROOT}/tools/qairt-sdk/qairt/2.40.0.251030")
set(JSON "${STUDY_ROOT}/third_party/OpenXR-SDK-Source/src/external/jsoncpp")
add_executable(qnn_device_probe ../qnn_device_probe.cpp ${CORE}/qnn_backend.cpp ${CORE}/scene_asset.cpp
    ${CORE}/sha256.cpp ${JSON}/src/lib_json/json_reader.cpp ${JSON}/src/lib_json/json_value.cpp
    ${JSON}/src/lib_json/json_writer.cpp)
target_include_directories(qnn_device_probe PRIVATE ${CORE} ${SDK}/include/QNN ${JSON}/include)
target_link_libraries(qnn_device_probe PRIVATE android log dl)
add_executable(gaussian_render_probe ../gaussian_render_probe.cpp ${CORE}/gaussian_frame.cpp ${CORE}/vulkan_gaussians.cpp
    ${CORE}/qnn_backend.cpp ${CORE}/scene_asset.cpp ${CORE}/sha256.cpp ${CORE}/decode.cpp ${CORE}/residual48.cpp ${CORE}/strict_topk.cpp
    ${JSON}/src/lib_json/json_reader.cpp ${JSON}/src/lib_json/json_value.cpp ${JSON}/src/lib_json/json_writer.cpp)
target_include_directories(gaussian_render_probe PRIVATE ${CORE} ${SDK}/include/QNN ${JSON}/include)
target_link_libraries(gaussian_render_probe PRIVATE android log dl vulkan)
include("${STUDY_ROOT}/cmake/Xnnpack.cmake")
target_sources(gaussian_render_probe PRIVATE ${CORE}/xnnpack_backend.cpp)
target_link_libraries(gaussian_render_probe PRIVATE XNNPACK pthreadpool cpuinfo)
add_executable(xnnpack_device_probe ../xnnpack_device_probe.cpp ${CORE}/xnnpack_backend.cpp ${CORE}/scene_asset.cpp
    ${CORE}/sha256.cpp ${CORE}/residual48.cpp ${JSON}/src/lib_json/json_reader.cpp ${JSON}/src/lib_json/json_value.cpp
    ${JSON}/src/lib_json/json_writer.cpp)
target_include_directories(xnnpack_device_probe PRIVATE ${CORE} ${JSON}/include)
target_link_libraries(xnnpack_device_probe PRIVATE android log XNNPACK pthreadpool cpuinfo)
add_executable(vulkan_point_benchmark_probe ../vulkan_point_benchmark_probe.cpp ${CORE}/gaussian_frame.cpp
    ${CORE}/vulkan_gaussians.cpp ${CORE}/scene_asset.cpp ${CORE}/sha256.cpp ${CORE}/decode.cpp ${CORE}/residual48.cpp ${CORE}/strict_topk.cpp
    ${JSON}/src/lib_json/json_reader.cpp ${JSON}/src/lib_json/json_value.cpp ${JSON}/src/lib_json/json_writer.cpp)
target_include_directories(vulkan_point_benchmark_probe PRIVATE ${CORE} ${JSON}/include)
target_link_libraries(vulkan_point_benchmark_probe PRIVATE android log vulkan)
