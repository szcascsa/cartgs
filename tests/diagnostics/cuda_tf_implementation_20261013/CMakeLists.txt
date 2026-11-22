cmake_minimum_required(VERSION 3.20)
project(cuda_tf_validation LANGUAGES CXX CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 86)
find_package(Torch REQUIRED)
get_filename_component(REPO "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
add_library(tf_cuda STATIC ${REPO}/src/transform_field_cuda.cpp ${REPO}/src/transform_field_cuda.cu
  ${REPO}/src/transform_field_cuda_neural.cu ${REPO}/src/transform_field_cuda_regularization.cu
  ${REPO}/src/transform_field_cuda_optimizer.cpp ${REPO}/src/transform_field_cuda_optimizer.cu)
target_include_directories(tf_cuda PUBLIC ${REPO} /usr/local/cuda-11.8/include)
target_link_libraries(tf_cuda PUBLIC ${TORCH_LIBRARIES} /usr/local/cuda-11.8/lib64/libcudart.so
  /usr/local/cuda-11.8/lib64/libcublas.so)
target_compile_options(tf_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
add_executable(cuda_transform_field_test ${REPO}/tests/cuda_transform_field_test.cpp
  ${REPO}/src/hexplane_field.cpp ${REPO}/src/transform_field.cpp)
target_link_libraries(cuda_transform_field_test tf_cuda)
add_executable(phase1_benchmark phase1_benchmark.cpp
  ${REPO}/src/hexplane_field.cpp ${REPO}/src/transform_field.cpp)
target_link_libraries(phase1_benchmark tf_cuda)
add_executable(cuda_tf_neural_test ${REPO}/tests/cuda_tf_neural_test.cpp
  ${REPO}/src/hexplane_field.cpp ${REPO}/src/transform_field.cpp)
target_link_libraries(cuda_tf_neural_test tf_cuda)
add_executable(cuda_tf_backward_test ${REPO}/tests/cuda_tf_backward_test.cpp
  ${REPO}/src/hexplane_field.cpp ${REPO}/src/transform_field.cpp)
target_link_libraries(cuda_tf_backward_test tf_cuda)
add_executable(cuda_tf_regularization_test ${REPO}/tests/cuda_tf_regularization_test.cpp
  ${REPO}/src/hexplane_field.cpp ${REPO}/src/transform_field.cpp)
target_link_libraries(cuda_tf_regularization_test tf_cuda)
add_executable(training_benchmark training_benchmark.cpp ${REPO}/src/constrained_adam.cpp
  ${REPO}/src/hexplane_field.cpp ${REPO}/src/transform_field.cpp)
target_link_libraries(training_benchmark tf_cuda /usr/local/cuda-11.8/lib64/libnvToolsExt.so)
add_executable(cuda_tf_adam_test ${REPO}/tests/cuda_tf_adam_test.cpp
  ${REPO}/src/hexplane_field.cpp ${REPO}/src/transform_field.cpp)
target_link_libraries(cuda_tf_adam_test tf_cuda)
add_executable(cuda_tf_preview_test ${REPO}/tests/cuda_tf_preview_test.cpp
  ${REPO}/src/constrained_adam.cpp ${REPO}/src/gradient_constraint.cpp
  ${REPO}/src/hexplane_field.cpp ${REPO}/src/transform_field.cpp)
target_link_libraries(cuda_tf_preview_test tf_cuda)
add_executable(legacy_constrained_adam_test ${REPO}/tests/constrained_adam_test.cpp ${REPO}/src/constrained_adam.cpp)
target_link_libraries(legacy_constrained_adam_test tf_cuda)
add_executable(legacy_gradient_constraint_test ${REPO}/tests/gradient_constraint_test.cpp
  ${REPO}/src/gradient_constraint.cpp ${REPO}/src/constrained_adam.cpp)
target_link_libraries(legacy_gradient_constraint_test tf_cuda)
add_executable(cuda_tf_api_test ${REPO}/tests/cuda_tf_api_test.cpp
  ${REPO}/src/hexplane_field.cpp ${REPO}/src/transform_field.cpp)
target_link_libraries(cuda_tf_api_test tf_cuda)
add_executable(change_distill_cuda_test ${REPO}/tests/change_distill_test.cpp
  ${REPO}/src/change_management.cpp ${REPO}/src/change_distill.cpp
  ${REPO}/src/hexplane_field.cpp ${REPO}/src/transform_field.cpp)
target_compile_definitions(change_distill_cuda_test PRIVATE CARTGS_TEST_CUDA_BACKEND)
target_link_libraries(change_distill_cuda_test tf_cuda)
add_executable(drift_stability_cuda_test ${REPO}/tests/drift_stability_test.cpp
  ${REPO}/src/drift_stability.cpp ${REPO}/src/hexplane_field.cpp ${REPO}/src/transform_field.cpp)
target_compile_definitions(drift_stability_cuda_test PRIVATE CARTGS_TEST_CUDA_BACKEND)
target_link_libraries(drift_stability_cuda_test tf_cuda)
add_executable(mlp_backward_benchmark mlp_backward_benchmark.cpp
  ${REPO}/src/hexplane_field.cpp ${REPO}/src/transform_field.cpp)
target_link_libraries(mlp_backward_benchmark tf_cuda)
add_executable(online_profiler_parity online_profiler_parity.cpp
  ${REPO}/src/hexplane_field.cpp ${REPO}/src/transform_field.cpp)
target_link_libraries(online_profiler_parity tf_cuda)
add_executable(training_chain_parity training_chain_parity.cpp
  ${REPO}/src/hexplane_field.cpp ${REPO}/src/transform_field.cpp)
target_link_libraries(training_chain_parity tf_cuda)
