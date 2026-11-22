# The existing source preserves the real RGB-D/SLAM path and online stop.
set(REVISED_MAIN "${TF_ONLINE_SOURCE}")
string(REPLACE "output_dir, 0," "output_dir, selector_revised::state.seed," REVISED_MAIN "${REVISED_MAIN}")
string(REPLACE "std::thread training_thd(&GaussianMapper::run, pGausMapper.get());"
  "std::thread training_thd([&] { selector_revised::run(*pGausMapper); });" REVISED_MAIN "${REVISED_MAIN}")
string(REPLACE "  pGausMapper->reportOnlineValidation();"
  "  pGausMapper->reportOnlineValidation();\n  selector_revised::report();" REVISED_MAIN "${REVISED_MAIN}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/selector_revised_main.cpp"
  "#include \"tests/diagnostics/cuda_selector_revised_20261015/online_budget.h\"\n${REVISED_MAIN}")
file(READ "${PROJECT_SOURCE_DIR}/src/gaussian_mapper.cpp" REVISED_MAPPER)
string(REPLACE "  mapping_profiler_.end(mapping_iter_);"
  "  mapping_profiler_.end(mapping_iter_);\n  selector_revised::recordStep(*this, mapping_start, mapping_iter_, normal_update_clock_.rejected_updates);" REVISED_MAPPER "${REVISED_MAPPER}")
string(REPLACE "selector_network_->forwardProbability(" "selector_revised::probability(selector_network_," REVISED_MAPPER "${REVISED_MAPPER}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/selector_revised_mapper.cpp"
  "#include \"tests/diagnostics/cuda_selector_revised_20261015/online_budget.h\"\n${REVISED_MAPPER}")
add_executable(tum_rgbd_selector_validation "${CMAKE_CURRENT_BINARY_DIR}/selector_revised_main.cpp"
  "${CMAKE_CURRENT_BINARY_DIR}/selector_revised_mapper.cpp")
target_link_libraries(tum_rgbd_selector_validation ${OpenCV_LIBRARIES} gaussian_mapper gaussian_viewer)
