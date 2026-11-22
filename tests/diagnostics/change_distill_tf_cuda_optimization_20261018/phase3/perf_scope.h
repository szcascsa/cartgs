#pragma once
#include <ATen/record_function.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>
#include <nvToolsExt.h>
#include <torch/torch.h>
#include <json/json.h>
#include <chrono>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace cdperf {
using Clock = std::chrono::steady_clock;
inline int mode = 0; // 0: native total; 1: deferred CUDA events; 2: Kineto/NVTX.
inline std::string branch;
struct Entry {
  std::string name;
  int64_t parent = -1;
  double cpu_us = 0;
  cudaEvent_t begin = nullptr, end = nullptr;
};
inline std::vector<Entry> entries;
inline std::vector<int64_t> stack;
inline Json::Value tensor_metadata, optimizer_metadata;
inline void check(cudaError_t status) {
  TORCH_CHECK(status == cudaSuccess, cudaGetErrorString(status));
}
struct Scope {
  int64_t id = -1;
  Clock::time_point start;
  std::unique_ptr<at::RecordFunction> record;
  explicit Scope(const std::string& name) {
    if (!mode) return;
    id = entries.size();
    entries.emplace_back();
    entries[id].name = branch.empty() ? name : branch + "/" + name;
    entries[id].parent = stack.empty() ? -1 : stack.back();
    if (mode == 1) {
      check(cudaEventCreate(&entries[id].begin));
      check(cudaEventCreate(&entries[id].end));
      check(cudaEventRecord(entries[id].begin, c10::cuda::getCurrentCUDAStream()));
    }
    if (mode >= 2) {
      nvtxRangePushA(entries[id].name.c_str());
    }
    if (mode == 2) {
      record = std::make_unique<at::RecordFunction>(at::RecordScope::USER_SCOPE);
      record->before(entries[id].name);
    }
    stack.push_back(id);
    start = Clock::now();
  }
  ~Scope() {
    if (id < 0) return;
    entries[id].cpu_us = std::chrono::duration<double, std::micro>(Clock::now() - start).count();
    if (mode == 1)
      check(cudaEventRecord(entries[id].end, c10::cuda::getCurrentCUDAStream()));
    if (mode == 2) record.reset();
    if (mode >= 2) nvtxRangePop();
    stack.pop_back();
  }
};
template <typename F> decltype(auto) op(const std::string& name, F&& fn) {
  Scope scope(name);
  return fn();
}
inline void tensor(const std::string& name, const torch::Tensor& value) {
  const auto key=name+(torch::GradMode::is_enabled()?"/grad_enabled":"/no_grad");
  if (!mode || tensor_metadata.isMember(key)) return;
  Json::Value j;
  for (auto x : value.sizes()) j["shape"].append(Json::Int64(x));
  for (auto x : value.strides()) j["stride"].append(Json::Int64(x));
  j["contiguous"] = value.is_contiguous();
  j["dtype"] = c10::toString(value.scalar_type());
  j["device"] = value.device().str();
  j["requires_grad"] = value.requires_grad();
  j["grad_fn"] = value.grad_fn() ? value.grad_fn()->name() : "none";
  j["bytes"] = Json::Int64(value.numel() * value.element_size());
  j["alignment_mod_256"] = Json::UInt64(reinterpret_cast<uintptr_t>(value.data_ptr()) % 256);
  tensor_metadata[key] = j;
}
inline void optimizer(torch::optim::Adam& optimizer) {
  if (!mode) return;
  Json::Value j;
  int64_t count = 0, elements = 0;
  for (const auto& group : optimizer.param_groups())
    for (const auto& p : group.params()) { ++count; elements += p.numel(); }
  j["parameter_tensors"] = Json::Int64(count);
  j["parameters"] = Json::Int64(elements);
  j["state_entries"] = Json::UInt64(optimizer.state().size());
  optimizer_metadata[branch] = j;
}
inline void flush(const std::string& path) {
  check(cudaDeviceSynchronize());
  std::ofstream out(path);
  out << "id,parent,component,cpu_us,cuda_span_us\n";
  for (size_t i = 0; i < entries.size(); ++i) {
    float ms = 0;
    if (entries[i].begin) {
      check(cudaEventElapsedTime(&ms, entries[i].begin, entries[i].end));
      cudaEventDestroy(entries[i].begin); cudaEventDestroy(entries[i].end);
    }
    out << i << ',' << entries[i].parent << ',' << entries[i].name << ','
        << entries[i].cpu_us << ',' << (mode == 1 ? ms * 1000.0 : -1.0) << '\n';
  }
  entries.clear();
}
} // namespace cdperf
#define CD_OP(name, ...) cdperf::op(name, [&]() -> decltype(auto) { return (__VA_ARGS__); })
