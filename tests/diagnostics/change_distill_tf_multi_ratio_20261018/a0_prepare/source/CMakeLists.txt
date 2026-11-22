cmake_minimum_required(VERSION 3.20 FATAL_ERROR)
project(photo-slam LANGUAGES CXX CUDA)

if(NOT CMAKE_CUDA_COMPILER_VERSION MATCHES "^11\\.8(\\.|$)")
    message(FATAL_ERROR
        "CaRtGS requires CUDA 11.8; found CUDA compiler ${CMAKE_CUDA_COMPILER_VERSION}")
endif()

if(DEFINED CMAKE_CUDA_ARCHITECTURES
   AND NOT "${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "86")
    message(FATAL_ERROR
        "CaRtGS supports only RTX 3090 (sm_86); found CUDA architectures ${CMAKE_CUDA_ARCHITECTURES}")
endif()

set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING
    "CUDA architecture for RTX 3090" FORCE)

if(NOT CMAKE_BUILD_TYPE)
    # set(CMAKE_BUILD_TYPE Debug)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-declarations")

set(CAFFE2_USE_CUDNN on)

# Uncomment the following line if you want to solve CMake Warning "Failed to compute shorthash for libnvrtc.so"
# set(PYTHON_EXECUTABLE "/usr/bin/python3")

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -pthread")

# set(ENV{CUDA_LAUNCH_BLOCKING} 1)
# set(ENV{TORCH_USE_CUDA_DSA} 1)

find_package(CUDA REQUIRED)
if(NOT CUDA_VERSION_STRING MATCHES "^11\\.8(\\.|$)")
    message(FATAL_ERROR
        "CaRtGS requires the CUDA 11.8 toolkit; found ${CUDA_VERSION_STRING}")
endif()
find_package(Eigen3 REQUIRED)
find_package(OpenCV 4 REQUIRED)
find_package(jsoncpp REQUIRED)

find_package(OpenGL REQUIRED)
find_package(glm REQUIRED)
find_package(glfw3 REQUIRED)

find_package(realsense2)

# Build the default CUDA Selector; runtime config still supports legacy A/B tests.
option(CARTGS_EXPERIMENTAL_SELECTOR_CUDA "Build the CUDA Selector backend" ON)
if(CARTGS_EXPERIMENTAL_SELECTOR_CUDA)
    find_package(CUDAToolkit 11.8 EXACT REQUIRED)
    add_library(selector_cuda STATIC src/selector_cuda.cpp src/selector_cuda.cu)
    set_target_properties(selector_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_include_directories(selector_cuda PUBLIC ${PROJECT_SOURCE_DIR} ${CUDA_INCLUDE_DIRS})
    target_compile_options(selector_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
    target_link_libraries(selector_cuda PUBLIC ${TORCH_LIBRARIES} CUDA::cudart CUDA::cublas CUDA::cublasLt)
    add_compile_definitions(CARTGS_EXPERIMENTAL_SELECTOR_CUDA)
endif()

add_library(transform_field_cuda STATIC
    src/change_distill_transport.cu
    src/transform_field_cuda.cpp src/transform_field_cuda.cu
    src/transform_field_cuda_neural.cu src/transform_field_cuda_regularization.cu
    src/transform_field_cuda_optimizer.cpp src/transform_field_cuda_optimizer.cu
    src/protection_profiler.cpp)
set_target_properties(transform_field_cuda PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(transform_field_cuda PUBLIC ${PROJECT_SOURCE_DIR} ${CUDA_INCLUDE_DIRS})
target_compile_options(transform_field_cuda PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--fmad=false>)
target_link_libraries(transform_field_cuda PUBLIC ${TORCH_LIBRARIES}
    ${CUDA_CUDART_LIBRARY} ${CUDA_CUBLAS_LIBRARIES})
if(CARTGS_EXPERIMENTAL_SELECTOR_CUDA)
    target_link_libraries(transform_field_cuda PUBLIC selector_cuda)
endif()

set(ORB_SLAM3_SOURCE_DIR "${PROJECT_SOURCE_DIR}/third_party/ORB-SLAM3")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

include_directories(
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/third_party
    ${ORB_SLAM3_SOURCE_DIR}
    ${ORB_SLAM3_SOURCE_DIR}/include
    ${ORB_SLAM3_SOURCE_DIR}/include/CameraModels
    ${ORB_SLAM3_SOURCE_DIR}/Thirdparty/Sophus
    ${EIGEN3_INCLUDE_DIR}
    ${OPENGL_INCLUDE_DIR}
)

##################################################################################
##  Build the gaussian splatting component libraries to ${PROJECT_SOURCE_DIR}/lib
##################################################################################

add_library(
    simple_knn
    SHARED
    third_party/simple-knn/simple_knn.cu
    third_party/simple-knn/simple_knn.h
    third_party/simple-knn/spatial.cu
    third_party/simple-knn/spatial.h
)
# target_compile_features(simple_knn PUBLIC cxx_std_17)
target_link_libraries(simple_knn ${TORCH_LIBRARIES})

add_library(
    cuda_rasterizer
    SHARED
    include/operate_points.h
    src/operate_points.cu
    include/stereo_vision.h
    src/stereo_vision.cu
    cuda_rasterizer/auxiliary.h
    cuda_rasterizer/rasterize_points.h
    cuda_rasterizer/rasterize_points.cu
    cuda_rasterizer/backward.cu
    cuda_rasterizer/backward.h
    cuda_rasterizer/config.h
    cuda_rasterizer/forward.cu
    cuda_rasterizer/forward.h
    cuda_rasterizer/operate_points.h
    cuda_rasterizer/rasterizer.h
    cuda_rasterizer/rasterizer_impl.cu
    cuda_rasterizer/rasterizer_impl.h
    cuda_rasterizer/stereo_vision.h
    cuda_rasterizer/adam.cu
    cuda_rasterizer/adam.h
    cuda_rasterizer/conv.cu
)
set_target_properties(cuda_rasterizer PROPERTIES CUDA_ARCHITECTURES "86")
# target_compile_features(cuda_rasterizer PUBLIC cxx_std_17)
target_include_directories(
    cuda_rasterizer
    PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)
target_link_libraries(cuda_rasterizer glm::glm ${TORCH_LIBRARIES} Eigen3::Eigen)

##################################################################################
##  Build the ImGui library to ${PROJECT_SOURCE_DIR}/lib
##################################################################################

add_library(
    imgui
    SHARED
    viewer/imgui/imconfig.h
    viewer/imgui/imgui_demo.cpp
    viewer/imgui/imgui_draw.cpp
    viewer/imgui/imgui_impl_glfw.cpp
    viewer/imgui/imgui_impl_glfw.h
    viewer/imgui/imgui_impl_opengl3_loader.h
    viewer/imgui/imgui_impl_opengl3.cpp
    viewer/imgui/imgui_impl_opengl3.h
    viewer/imgui/imgui_internal.h
    viewer/imgui/imgui_tables.cpp
    viewer/imgui/imgui_widgets.cpp
    viewer/imgui/imgui.cpp
    viewer/imgui/imgui.h
    viewer/imgui/imstb_rectpack.h
    viewer/imgui/imstb_textedit.h
    viewer/imgui/imstb_truetype.h
)
target_link_libraries(imgui glfw OpenGL::GL)

##################################################################################
##  Build the gaussian mapper library to ${PROJECT_SOURCE_DIR}/lib
##################################################################################

add_library(
    gaussian_mapper
    SHARED
    third_party/tinyply/tinyply.h
    third_party/tinyply/tinyply.cpp
    include/gaussian_keyframe.h
    include/gaussian_model.h
    include/gaussian_parameters.h
    include/gaussian_rasterizer.h
    include/gaussian_renderer.h
    include/gaussian_scene.h
    include/gaussian_mapper.h
    include/gi_teacher.h
    include/frozen_teacher_target_cache.h
    include/historical_probe_cache.h
    include/online_gi.h
    include/quaternion_utils.h
    include/hexplane_field.h
    include/transform_field.h
    include/change_management.h
    include/drift_stability.h
    include/recent_window_importance.h
    include/global_gi_bank.h
    include/global_gi.h
    include/historical_outlier_selector.h
    include/constrained_adam.h
    include/gradient_constraint.h
    include/conflict_replay.h
    include/gradient_constraint_solver.h
    include/historical_gradient_render.h
    include/aabb.h
    include/gaussian_transform_ops.h
    include/model_snapshot.h
    include/general_utils.h
    include/graphics_utils.h
    include/loss_utils.h
    include/sh_utils.h
    include/tensor_utils.h
    include/camera.h
    include/point_cloud.h
    include/point2d.h
    include/point3d.h
    include/types.h
    src/gaussian_keyframe.cpp
    src/gaussian_model.cpp
    src/gaussian_parameters.cpp
    src/gaussian_rasterizer.cpp
    src/gaussian_renderer.cpp
    src/gaussian_scene.cpp
    src/gaussian_mapper.cpp
    src/maturity_dynamics.cpp
    src/gi_teacher.cpp
    src/online_gi.cpp
    src/hexplane_field.cpp
    src/transform_field.cpp
    src/change_management.cpp
    src/change_distill.cpp
    src/drift_stability.cpp
    src/recent_window_importance.cpp
    src/global_gi.cpp
    src/historical_outlier_selector.cpp
    src/constrained_adam.cpp
    src/gradient_constraint.cpp
    src/conflict_replay.cpp
    src/historical_gradient_render.cpp
)
target_link_libraries(
    gaussian_mapper
    ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so
    ${OpenCV_LIBRARIES}
    jsoncpp_lib
    ${TORCH_LIBRARIES}
    Eigen3::Eigen
    simple_knn
    cuda_rasterizer
    transform_field_cuda
    ${CUDA_CUDART_LIBRARY}
)
target_include_directories(gaussian_mapper PUBLIC
    ${PROJECT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
)

option(CARTGS_BUILD_TESTS "Build CaRtGS first-party tests" OFF)
if(CARTGS_BUILD_TESTS)
    enable_testing()
    if(CARTGS_EXPERIMENTAL_SELECTOR_CUDA)
        add_executable(change_distill_selector_cuda_test tests/change_distill_test.cpp
            src/change_management.cpp src/change_distill.cpp src/hexplane_field.cpp src/transform_field.cpp)
        target_compile_definitions(change_distill_selector_cuda_test PRIVATE CARTGS_TEST_CUDA_BACKEND CARTGS_TEST_SELECTOR_CUDA)
        target_link_libraries(change_distill_selector_cuda_test transform_field_cuda)
        add_test(NAME change_distill_selector_cuda COMMAND change_distill_selector_cuda_test)
        add_executable(drift_stability_selector_cuda_test tests/drift_stability_test.cpp
            src/drift_stability.cpp src/hexplane_field.cpp src/transform_field.cpp)
        target_compile_definitions(drift_stability_selector_cuda_test PRIVATE CARTGS_TEST_CUDA_BACKEND CARTGS_TEST_SELECTOR_CUDA)
        target_link_libraries(drift_stability_selector_cuda_test transform_field_cuda)
        add_test(NAME drift_stability_selector_cuda COMMAND drift_stability_selector_cuda_test)
        add_executable(joint_replay_selector_cuda_test tests/joint_replay_test.cpp)
        target_compile_definitions(joint_replay_selector_cuda_test PRIVATE CARTGS_TEST_SELECTOR_CUDA CARTGS_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
        target_link_libraries(joint_replay_selector_cuda_test gaussian_mapper)
        add_test(NAME joint_replay_selector_cuda COMMAND joint_replay_selector_cuda_test cuda)
        add_executable(historical_gradient_selector_cuda_test tests/historical_gradient_render_test.cpp)
        target_compile_definitions(historical_gradient_selector_cuda_test PRIVATE CARTGS_TEST_SELECTOR_CUDA)
        target_link_libraries(historical_gradient_selector_cuda_test gaussian_mapper)
        add_test(NAME historical_gradient_selector_cuda COMMAND historical_gradient_selector_cuda_test --cuda)
    endif()
    add_executable(constrained_adam_test tests/constrained_adam_test.cpp src/constrained_adam.cpp)
    target_link_libraries(constrained_adam_test transform_field_cuda)
    add_test(NAME constrained_adam COMMAND constrained_adam_test)
    add_executable(gradient_constraint_solver_test tests/gradient_constraint_solver_test.cpp)
    add_test(NAME gradient_constraint_solver COMMAND gradient_constraint_solver_test)
    add_executable(gradient_constraint_clock_test tests/gradient_constraint_clock_test.cpp)
    add_test(NAME gradient_constraint_clock COMMAND gradient_constraint_clock_test)
    add_executable(conflict_replay_test tests/conflict_replay_test.cpp src/conflict_replay.cpp)
    add_test(NAME conflict_replay COMMAND conflict_replay_test)
    add_executable(joint_replay_test tests/joint_replay_test.cpp)
    target_link_libraries(joint_replay_test gaussian_mapper)
    target_compile_definitions(joint_replay_test PRIVATE CARTGS_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME joint_replay_cuda COMMAND joint_replay_test)
    add_test(NAME joint_replay_tf_cuda COMMAND joint_replay_test cuda)
    set_tests_properties(joint_replay_cuda PROPERTIES SKIP_RETURN_CODE 77)
    add_executable(gradient_constraint_test tests/gradient_constraint_test.cpp
        src/gradient_constraint.cpp src/constrained_adam.cpp)
    target_link_libraries(gradient_constraint_test transform_field_cuda)
    add_test(NAME gradient_constraint COMMAND gradient_constraint_test)
    add_executable(historical_gradient_render_test tests/historical_gradient_render_test.cpp)
    target_link_libraries(historical_gradient_render_test gaussian_mapper)
    add_test(NAME historical_gradient_render COMMAND historical_gradient_render_test)
    add_test(NAME historical_gradient_render_cuda COMMAND historical_gradient_render_test --cuda)
    set_tests_properties(historical_gradient_render_cuda PROPERTIES SKIP_RETURN_CODE 77)
    # Explicit dataset/checkpoint benchmark; never writes the source checkpoint.
    add_executable(gradient_guard_checkpoint_test tests/gradient_guard_checkpoint_test.cpp)
    target_link_libraries(gradient_guard_checkpoint_test gaussian_mapper)
    add_executable(
        online_gi_lifecycle_test
        tests/online_gi_lifecycle_test.cpp
        src/online_gi.cpp
    )
    target_link_libraries(online_gi_lifecycle_test ${TORCH_LIBRARIES})
    add_test(NAME online_gi_lifecycle COMMAND online_gi_lifecycle_test)

    add_executable(
        recent_window_importance_test
        tests/recent_window_importance_test.cpp
    )
    target_link_libraries(recent_window_importance_test gaussian_mapper)
    add_test(NAME recent_window_importance COMMAND recent_window_importance_test)

    add_executable(global_gi_bank_test tests/global_gi_bank_test.cpp)
    target_include_directories(global_gi_bank_test PRIVATE ${PROJECT_SOURCE_DIR})
    add_test(NAME global_gi_bank COMMAND global_gi_bank_test)
    add_executable(global_gi_snapshot_cache_test tests/global_gi_snapshot_cache_test.cpp)
    target_include_directories(global_gi_snapshot_cache_test PRIVATE ${PROJECT_SOURCE_DIR})
    add_test(NAME global_gi_snapshot_cache COMMAND global_gi_snapshot_cache_test)
    add_executable(gi_association_state_test tests/gi_association_state_test.cpp)
    target_link_libraries(gi_association_state_test gaussian_mapper
        ${ORB_SLAM3_SOURCE_DIR}/Thirdparty/DBoW2/lib/libDBoW2.so)
    add_test(NAME gi_association_state COMMAND gi_association_state_test)
    add_executable(global_gi_test tests/global_gi_test.cpp)
    target_link_libraries(global_gi_test gaussian_mapper)
    add_test(NAME global_gi COMMAND global_gi_test)
    add_executable(ultra_light_schedule_test tests/ultra_light_schedule_test.cpp)
    target_link_libraries(ultra_light_schedule_test ${TORCH_LIBRARIES})
    add_test(NAME ultra_light_schedule COMMAND ultra_light_schedule_test)
    add_executable(tail_optimization_schedule_test tests/tail_optimization_schedule_test.cpp)
    target_include_directories(tail_optimization_schedule_test PRIVATE ${PROJECT_SOURCE_DIR})
    add_test(NAME tail_optimization_schedule COMMAND tail_optimization_schedule_test)
    add_executable(ultrafast_rotation_test tests/ultrafast_rotation_test.cpp)
    target_include_directories(ultrafast_rotation_test PRIVATE ${PROJECT_SOURCE_DIR})
    add_test(NAME ultrafast_rotation COMMAND ultrafast_rotation_test)
    add_executable(ultrafast_mapping_test tests/ultrafast_mapping_test.cpp)
    target_link_libraries(ultrafast_mapping_test gaussian_mapper)
    target_compile_definitions(ultrafast_mapping_test PRIVATE CARTGS_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME ultrafast_mapping_cuda COMMAND ultrafast_mapping_test)
    set_tests_properties(ultrafast_mapping_cuda PROPERTIES SKIP_RETURN_CODE 77)
    add_executable(ultralight_mode2_mapping_test tests/ultralight_mode2_mapping_test.cpp)
    target_link_libraries(ultralight_mode2_mapping_test gaussian_mapper)
    target_compile_definitions(ultralight_mode2_mapping_test PRIVATE CARTGS_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME ultralight_mode2_config COMMAND ultralight_mode2_mapping_test config)
    add_test(NAME ultralight_mode2_full_parity COMMAND ultralight_mode2_mapping_test parity)
    add_test(NAME ultralight_mode2_stages COMMAND ultralight_mode2_mapping_test stages)
    add_test(NAME ultralight_mode2_clocks COMMAND ultralight_mode2_mapping_test clocks)
    add_test(NAME ultralight_mode2_special COMMAND ultralight_mode2_mapping_test special)
    set_tests_properties(ultralight_mode2_full_parity ultralight_mode2_stages ultralight_mode2_clocks ultralight_mode2_special PROPERTIES SKIP_RETURN_CODE 77)
    find_package(Python3 COMPONENTS Interpreter QUIET)
    if(Python3_Interpreter_FOUND)
        add_test(NAME ultrafast_eval COMMAND ${Python3_EXECUTABLE}
            ${PROJECT_SOURCE_DIR}/tests/ultrafast_eval_test.py)
    endif()

    add_executable(
        historical_outlier_selector_test
        tests/historical_outlier_selector_test.cpp
    )
    target_link_libraries(historical_outlier_selector_test gaussian_mapper)
    add_test(NAME historical_outlier_selector COMMAND historical_outlier_selector_test)

    add_executable(
        gaussian_model_lifecycle_test
        tests/gaussian_model_lifecycle_test.cpp
    )
    target_link_libraries(gaussian_model_lifecycle_test gaussian_mapper)
    add_test(NAME gaussian_model_lifecycle COMMAND gaussian_model_lifecycle_test)

    add_executable(
        transform_field_test
        tests/transform_field_test.cpp
        src/hexplane_field.cpp
        src/transform_field.cpp
    )
    target_link_libraries(transform_field_test transform_field_cuda)
    add_test(NAME transform_field COMMAND transform_field_test)
    add_executable(continuous_tf_test tests/continuous_tf_test.cpp)
    target_link_libraries(continuous_tf_test gaussian_mapper)
    add_test(NAME continuous_tf COMMAND continuous_tf_test)
    add_test(NAME continuous_tf_cuda COMMAND continuous_tf_test --cuda)
    add_executable(elastic_entry_test tests/elastic_entry_test.cpp)
    target_link_libraries(elastic_entry_test gaussian_mapper)
    target_compile_definitions(elastic_entry_test PRIVATE CARTGS_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
    add_test(NAME elastic_entry COMMAND elastic_entry_test)
    add_test(NAME elastic_entry_cuda COMMAND elastic_entry_test --cuda)

    add_executable(
        hexplane_resample_test
        tests/hexplane_resample_test.cpp
        src/hexplane_field.cpp
        src/transform_field.cpp
    )
    target_link_libraries(hexplane_resample_test transform_field_cuda)
    add_test(NAME hexplane_resample COMMAND hexplane_resample_test)

    add_executable(
        change_distill_test
        tests/change_distill_test.cpp
        src/change_management.cpp
        src/change_distill.cpp
        src/hexplane_field.cpp
        src/transform_field.cpp
    )
    target_link_libraries(change_distill_test transform_field_cuda)
    add_test(NAME change_distill COMMAND change_distill_test)

    add_executable(
        drift_stability_test
        tests/drift_stability_test.cpp
        src/drift_stability.cpp
        src/hexplane_field.cpp
        src/transform_field.cpp
    )
    target_link_libraries(drift_stability_test transform_field_cuda)
    add_test(NAME drift_stability COMMAND drift_stability_test)

    add_executable(change_distill_tf_cuda_test tests/change_distill_test.cpp
        src/change_management.cpp src/change_distill.cpp src/hexplane_field.cpp src/transform_field.cpp)
    target_compile_definitions(change_distill_tf_cuda_test PRIVATE CARTGS_TEST_CUDA_BACKEND)
    target_link_libraries(change_distill_tf_cuda_test transform_field_cuda)
    add_test(NAME change_distill_tf_cuda COMMAND change_distill_tf_cuda_test)
    add_executable(drift_stability_tf_cuda_test tests/drift_stability_test.cpp
        src/drift_stability.cpp src/hexplane_field.cpp src/transform_field.cpp)
    target_compile_definitions(drift_stability_tf_cuda_test PRIVATE CARTGS_TEST_CUDA_BACKEND)
    target_link_libraries(drift_stability_tf_cuda_test transform_field_cuda)
    add_test(NAME drift_stability_tf_cuda COMMAND drift_stability_tf_cuda_test)

    add_executable(
        gi_teacher_test
        tests/gi_teacher_test.cpp
        src/gi_teacher.cpp
    )
    target_link_libraries(gi_teacher_test ${TORCH_LIBRARIES})
    add_test(NAME gi_teacher COMMAND gi_teacher_test)
    add_test(NAME gi_teacher_cuda COMMAND gi_teacher_test --cuda)

    add_executable(performance_primitives_test tests/performance_primitives_test.cpp)
    target_link_libraries(performance_primitives_test ${TORCH_LIBRARIES})
    if(CARTGS_EXPERIMENTAL_SELECTOR_CUDA)
        target_link_libraries(performance_primitives_test selector_cuda)
    endif()
    add_test(NAME performance_primitives COMMAND performance_primitives_test)

    add_executable(mapping_profiler_test tests/mapping_profiler_test.cpp)
    target_link_libraries(mapping_profiler_test ${TORCH_LIBRARIES} ${CUDA_CUDART_LIBRARY})
    target_include_directories(mapping_profiler_test PRIVATE ${CUDA_INCLUDE_DIRS})
    add_test(NAME mapping_profiler COMMAND mapping_profiler_test)

    add_executable(renderer_readonly_test tests/renderer_readonly_test.cpp)
    target_link_libraries(renderer_readonly_test gaussian_mapper)
    add_test(NAME renderer_readonly_cuda COMMAND renderer_readonly_test)
    set_tests_properties(renderer_readonly_cuda PROPERTIES SKIP_RETURN_CODE 77)

    # Dataset-backed integration benchmark; invoked explicitly with a checkpoint.
    add_executable(distill_recovery_test tests/distill_recovery_test.cpp)
    target_link_libraries(distill_recovery_test gaussian_mapper)

    add_executable(
        frozen_teacher_target_cache_test
        tests/frozen_teacher_target_cache_test.cpp
    )
    target_link_libraries(frozen_teacher_target_cache_test ${TORCH_LIBRARIES})
    add_test(NAME frozen_teacher_target_cache COMMAND frozen_teacher_target_cache_test)

    add_executable(historical_probe_cache_test
        tests/historical_probe_cache_test.cpp
        src/hexplane_field.cpp
        src/transform_field.cpp
    )
    target_link_libraries(historical_probe_cache_test transform_field_cuda)
    add_test(NAME historical_probe_cache COMMAND historical_probe_cache_test)

    add_executable(selector_handoff_test tests/selector_handoff_test.cpp)
    add_executable(maturity_ablation_test tests/maturity_ablation_test.cpp)
    add_executable(maturity_module_test tests/maturity_module_test.cpp)
    target_link_libraries(maturity_module_test gaussian_mapper)
    add_test(NAME maturity_module COMMAND maturity_module_test)
    add_test(NAME maturity_module_cuda COMMAND maturity_module_test --cuda)
    target_link_libraries(maturity_ablation_test gaussian_mapper)
    add_test(NAME maturity_ablation COMMAND maturity_ablation_test)
    add_test(NAME maturity_ablation_cuda COMMAND maturity_ablation_test --cuda)
    if(Python3_Interpreter_FOUND)
        add_test(NAME maturity_module_python COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tests/maturity_module_python_test.py)
        add_test(NAME maturity_ablation_python COMMAND ${Python3_EXECUTABLE} ${PROJECT_SOURCE_DIR}/tests/maturity_ablation_python_test.py)
    endif()
    target_link_libraries(selector_handoff_test gaussian_mapper)
    add_test(NAME selector_handoff COMMAND selector_handoff_test)
    add_test(NAME selector_handoff_cuda COMMAND selector_handoff_test --cuda)
    set_tests_properties(selector_handoff_cuda PROPERTIES SKIP_RETURN_CODE 77)
    add_executable(gaussian_protection_test tests/gaussian_protection_test.cpp)
    add_executable(maturity_dynamics_test tests/maturity_dynamics_test.cpp)
    target_link_libraries(maturity_dynamics_test gaussian_mapper)
    add_test(NAME maturity_dynamics COMMAND maturity_dynamics_test)
    add_test(NAME maturity_dynamics_cuda COMMAND maturity_dynamics_test --cuda)
    set_tests_properties(maturity_dynamics_cuda PROPERTIES SKIP_RETURN_CODE 77)
    target_link_libraries(gaussian_protection_test gaussian_mapper)
    add_test(NAME gaussian_protection COMMAND gaussian_protection_test)
    add_test(NAME gaussian_protection_cuda COMMAND gaussian_protection_test --cuda)
    set_tests_properties(gaussian_protection_cuda PROPERTIES SKIP_RETURN_CODE 77)

    add_executable(compact_lod_test tests/compact_lod_test.cpp)
    target_link_libraries(compact_lod_test gaussian_mapper)
    add_test(NAME compact_lod COMMAND compact_lod_test)
    add_test(NAME compact_lod_cuda COMMAND compact_lod_test --cuda)
    set_tests_properties(compact_lod_cuda PROPERTIES SKIP_RETURN_CODE 77)
endif()

##################################################################################
##  Build the viewer library to ${PROJECT_SOURCE_DIR}/lib
##################################################################################

add_library(
    gaussian_viewer
    SHARED
    viewer/drawer_utils.h
    viewer/imgui_viewer.cpp
    viewer/imgui_viewer.h
    viewer/map_drawer.cpp
    viewer/map_drawer.h
)
target_link_libraries(
    gaussian_viewer
    ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so
    gaussian_mapper
    imgui
    ${OpenCV_LIBRARIES}
    jsoncpp_lib
    ${TORCH_LIBRARIES}
    glm::glm
    glfw
    OpenGL::GL
)

##################################################################################
##  Build the test examples to ${PROJECT_SOURCE_DIR}/bin
##################################################################################

# This is a C++ libtorch implementation of gaussian-splatting (https://github.com/graphdeco-inria/gaussian-splatting)
add_executable(train_colmap examples/train_colmap.cpp)
target_link_libraries(train_colmap gaussian_viewer gaussian_mapper)

add_executable(view_result examples/view_result.cpp)
target_link_libraries(view_result gaussian_viewer gaussian_mapper)

##################################################################################
##  Build the mapping examples to ${PROJECT_SOURCE_DIR}/bin
##################################################################################

# Replica Monocular
add_executable(replica_mono examples/replica_mono.cpp)
target_link_libraries(
    replica_mono
    gaussian_viewer
    gaussian_mapper
    ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so
)

# Replica Monocular
add_executable(replica_rgbd examples/replica_rgbd.cpp)
target_link_libraries(
    replica_rgbd
    gaussian_viewer
    gaussian_mapper
    ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so
)

# TUM Monocular
add_executable(tum_mono examples/tum_mono.cpp)
target_link_libraries(
    tum_mono
    gaussian_viewer
    gaussian_mapper
    ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so
)

# TUM RGBD
add_executable(tum_rgbd examples/tum_rgbd.cpp)
if(CARTGS_BUILD_TESTS)
    add_executable(tum_rgbd_maturity_ablation examples/tum_rgbd.cpp)
    target_compile_definitions(tum_rgbd_maturity_ablation PRIVATE CARTGS_MATURITY_ABLATION)
    target_link_libraries(tum_rgbd_maturity_ablation gaussian_viewer gaussian_mapper ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so)
endif()
target_link_libraries(
    tum_rgbd
    gaussian_viewer
    gaussian_mapper
    ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so
)

add_executable(vector_stereo examples/vector_stereo.cpp)
target_link_libraries(
    vector_stereo
    gaussian_viewer
    gaussian_mapper
    ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so
)

add_executable(vector_mono examples/vector_mono.cpp)
target_link_libraries(
    vector_mono
    gaussian_viewer
    gaussian_mapper
    ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so
)

##################################################################################
##  Build the mapping examples to ${PROJECT_SOURCE_DIR}/bin
##################################################################################

# Intel Realsense
if(realsense2_FOUND)
    add_executable(realsense_rgbd examples/realsense_rgbd.cpp)
    target_include_directories(realsense_rgbd PUBLIC ${realsense_INCLUDE_DIR})
    target_link_libraries(
        realsense_rgbd
        gaussian_viewer
        gaussian_mapper
        ${ORB_SLAM3_SOURCE_DIR}/lib/libORB_SLAM3.so
        ${realsense2_LIBRARY}
    )
endif()

if(CARTGS_BUILD_TESTS)
    include(tests/diagnostics/cuda_tf_implementation_20261013/online_driver.cmake)
    if(CARTGS_EXPERIMENTAL_SELECTOR_CUDA)
        include(tests/diagnostics/cuda_selector_revised_20261015/online_driver.cmake)
    endif()
endif()
