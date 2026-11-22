#pragma once
#include <torch/torch.h>

// Reuse the existing full lifecycle suites on both devices/backends.
#ifdef CARTGS_TEST_CUDA_BACKEND
inline constexpr auto tf_test_device = torch::kCUDA;
inline constexpr const char* tf_test_backend = "cuda";
#else
inline constexpr auto tf_test_device = torch::kCPU;
inline constexpr const char* tf_test_backend = "legacy";
#endif
