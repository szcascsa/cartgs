# Acceptance executable: the original RGB-D input/SLAM/Mapper path, stopping
# at SLAM shutdown instead of entering the repository's post-SLAM tail loop.
# It neither changes online iteration frequency nor any training setting.
file(READ "${PROJECT_SOURCE_DIR}/examples/tum_rgbd.cpp" TF_ONLINE_SOURCE)
set(TF_VALIDATION_CLASS [=[
class CudaTFOnlineMapper : public GaussianMapper {
 public:
  using GaussianMapper::GaussianMapper;
  void reportOnlineValidation() {
    mapping_profiler_.report(mapping_iter_);
    std::cout << "[CUDA TF ONLINE] mapping_cycles=" << mapping_iter_
              << " tf_updates=" << normal_update_clock_.tf_updates
              << " selector_updates=" << selector_step_
              << " gaussian_total=" << gaussians_->xyz_.size(0)
              << " cd_tf_steps=" << cd_tf_steps_
              << " replay_tf_updates=" << replay_tf_updates_
              << " tail_training=OFF" << std::endl;
  }
};
]=])
string(REPLACE "int main(int argc, char **argv) {" "${TF_VALIDATION_CLASS}\nint main(int argc, char **argv) {" TF_ONLINE_SOURCE "${TF_ONLINE_SOURCE}")
string(REPLACE "std::shared_ptr<GaussianMapper> pGausMapper" "std::shared_ptr<CudaTFOnlineMapper> pGausMapper" TF_ONLINE_SOURCE "${TF_ONLINE_SOURCE}")
string(REPLACE "std::make_shared<GaussianMapper>" "std::make_shared<CudaTFOnlineMapper>" TF_ONLINE_SOURCE "${TF_ONLINE_SOURCE}")
set(TF_SHUTDOWN_MONITOR [=[
  std::thread stop_at_shutdown([&] {
    while (!pSLAM->isShutDown())
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    pGausMapper->signalStop();
  });
  pSLAM->Shutdown();
  stop_at_shutdown.join();
]=])
string(REPLACE "  pSLAM->Shutdown();" "${TF_SHUTDOWN_MONITOR}" TF_ONLINE_SOURCE "${TF_ONLINE_SOURCE}")
string(REPLACE "  training_thd.join();" "  training_thd.join();\n  pGausMapper->reportOnlineValidation();" TF_ONLINE_SOURCE "${TF_ONLINE_SOURCE}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/cuda_tf_online_main.cpp" "${TF_ONLINE_SOURCE}")
add_executable(tum_rgbd_cuda_tf_online "${CMAKE_CURRENT_BINARY_DIR}/cuda_tf_online_main.cpp")
target_link_libraries(tum_rgbd_cuda_tf_online ${OpenCV_LIBRARIES} gaussian_mapper gaussian_viewer)

# Separate diagnostic executable. Math is unchanged, but its graph brackets
# and split optimizer traversal must not be used as uninstrumented wall time.
file(READ "${PROJECT_SOURCE_DIR}/src/gaussian_mapper.cpp" TF_PROFILE_MAPPER)
string(REPLACE "transform_field_->applyResidual(" "tf_diagnostic::applyResidual(*transform_field_," TF_PROFILE_MAPPER "${TF_PROFILE_MAPPER}")
string(REPLACE "transform_field_->regularizationLoss()" "tf_diagnostic::regularization(*transform_field_)" TF_PROFILE_MAPPER "${TF_PROFILE_MAPPER}")
set(TF_PROFILE_OPTIMIZER [=[tf_diagnostic::mixedStep(*gaussians_->optimizer_, cuda_tf_groups,
    shared_transform_optimizer_configured_ ? std::vector<std::size_t>{shared_transform_mlp_group_index_, shared_transform_grid_group_index_} : std::vector<std::size_t>{})]=])
string(REPLACE "tf_cuda::tfCudaAdamMixedStep(*gaussians_->optimizer_, cuda_tf_groups, gaussians_->protection_.timing())" "${TF_PROFILE_OPTIMIZER}" TF_PROFILE_MAPPER "${TF_PROFILE_MAPPER}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/cuda_tf_profile_mapper.cpp"
  "#include \"tests/diagnostics/cuda_tf_implementation_20261013/online_tf_profiler.h\"\n${TF_PROFILE_MAPPER}")
string(REPLACE "  pGausMapper->reportOnlineValidation();" "  pGausMapper->reportOnlineValidation();\n  tf_diagnostic::report();" TF_PROFILE_MAIN "${TF_ONLINE_SOURCE}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/cuda_tf_profile_main.cpp"
  "#include \"tests/diagnostics/cuda_tf_implementation_20261013/online_tf_profiler.h\"\n${TF_PROFILE_MAIN}")
add_executable(tum_rgbd_cuda_tf_profile "${CMAKE_CURRENT_BINARY_DIR}/cuda_tf_profile_main.cpp"
  "${CMAKE_CURRENT_BINARY_DIR}/cuda_tf_profile_mapper.cpp")
target_link_libraries(tum_rgbd_cuda_tf_profile ${OpenCV_LIBRARIES} gaussian_mapper gaussian_viewer)
