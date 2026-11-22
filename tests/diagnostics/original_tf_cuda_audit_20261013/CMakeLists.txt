cmake_minimum_required(VERSION 3.16)
project(original_tf_audit LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
find_package(Torch REQUIRED)
get_filename_component(REPO "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
add_executable(native_benchmark native_benchmark.cpp
  ${REPO}/src/transform_field.cpp ${REPO}/src/hexplane_field.cpp
  ${REPO}/src/constrained_adam.cpp)
target_include_directories(native_benchmark PRIVATE ${REPO} /usr/local/cuda-11.8/include)
target_link_libraries(native_benchmark ${TORCH_LIBRARIES} /usr/local/cuda-11.8/lib64/libcudart.so /usr/local/cuda-11.8/lib64/libnvToolsExt.so)
target_compile_options(native_benchmark PRIVATE -O3)
