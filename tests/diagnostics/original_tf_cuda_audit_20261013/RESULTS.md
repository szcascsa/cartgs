## 实测结果与复现材料

16个源码/配置SHA-256全部一致：True。native↔Python镜像的32项forward/input/parameter gradient全部通过，最大绝对差3.41e-13。15组连续训练后的参数、输出、输入/参数梯度、Adam m/v有限值检查全部通过：True（见native_validity.csv）。真实TF共有26个参数tensor、1,186,762个FP32参数。`checkpoint_optimizer.txt`确认该checkpoint shadow optimizer为`groups=2 states=0`。

### Native生产源码CUDA events

以下均为600次连续训练中的后500次，warmup100。每个单元为mean / p50 / p95（ms）。

**core_adam**

| N | forward | loss/regularization | backward | zero_grad | optimizer | total | wall |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 0.915 / 0.910 / 0.950 | 0.062 / 0.061 / 0.062 | 2.851 / 2.869 / 2.933 | 0.021 / 0.022 / 0.022 | 1.034 / 1.029 / 1.070 | 4.883 / 4.894 / 4.971 | 4.894 / 4.905 / 4.985 |
| 5000 | 0.924 / 0.916 / 0.959 | 0.061 / 0.060 / 0.062 | 2.910 / 2.891 / 2.990 | 0.021 / 0.022 / 0.022 | 1.019 / 1.013 / 1.053 | 4.936 / 4.906 / 5.044 | 4.947 / 4.917 / 5.055 |
| 10000 | 0.937 / 0.937 / 0.968 | 0.062 / 0.061 / 0.065 | 2.951 / 2.955 / 3.033 | 0.021 / 0.022 / 0.022 | 1.034 / 1.034 / 1.065 | 5.006 / 5.015 / 5.140 | 5.017 / 5.025 / 5.152 |
| 20000 | 0.968 / 0.976 / 1.007 | 0.065 / 0.066 / 0.068 | 2.928 / 2.913 / 3.026 | 0.021 / 0.022 / 0.022 | 1.058 / 1.066 / 1.095 | 5.040 / 5.053 / 5.168 | 5.051 / 5.065 / 5.179 |
| 50000 | 1.280 / 1.278 / 1.297 | 0.033 / 0.033 / 0.035 | 3.946 / 3.939 / 3.988 | 0.021 / 0.020 / 0.022 | 0.418 / 0.417 / 0.420 | 5.698 / 5.689 / 5.750 | 5.710 / 5.701 / 5.761 |

10k→50k总时间：5.006→5.698 ms，倍率1.138，增加13.8%。

**regularized_adam**

| N | forward | loss/regularization | backward | zero_grad | optimizer | total | wall |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 0.939 / 0.948 / 0.998 | 1.203 / 1.211 / 1.261 | 7.915 / 7.913 / 8.028 | 0.022 / 0.022 / 0.026 | 1.045 / 1.055 / 1.097 | 11.124 / 11.149 / 11.340 | 11.136 / 11.161 / 11.352 |
| 5000 | 0.957 / 0.965 / 1.009 | 1.212 / 1.220 / 1.267 | 8.007 / 7.985 / 8.103 | 0.021 / 0.022 / 0.022 | 1.052 / 1.060 / 1.106 | 11.249 / 11.250 / 11.438 | 11.261 / 11.261 / 11.449 |
| 10000 | 0.950 / 0.964 / 0.986 | 1.203 / 1.215 / 1.240 | 7.981 / 7.976 / 8.093 | 0.021 / 0.022 / 0.023 | 1.046 / 1.057 / 1.102 | 11.202 / 11.166 / 11.398 | 11.214 / 11.177 / 11.409 |
| 20000 | 0.948 / 0.933 / 0.987 | 1.195 / 1.182 / 1.238 | 7.946 / 7.966 / 8.088 | 0.021 / 0.022 / 0.022 | 1.035 / 1.023 / 1.079 | 11.145 / 11.134 / 11.323 | 11.156 / 11.145 / 11.335 |
| 50000 | 1.290 / 1.291 / 1.308 | 0.914 / 0.922 / 0.967 | 9.234 / 9.197 / 9.370 | 0.022 / 0.020 / 0.022 | 0.415 / 0.414 / 0.419 | 11.875 / 11.841 / 12.074 | 11.886 / 11.852 / 12.088 |

10k→50k总时间：11.202→11.875 ms，倍率1.060，增加6.0%。

**regularized_preview**

| N | forward | loss/regularization | backward | zero_grad | optimizer | total | wall |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 0.907 / 0.905 / 0.955 | 1.177 / 1.172 / 1.258 | 8.221 / 8.200 / 8.326 | 0.021 / 0.022 / 0.022 | 2.195 / 2.188 / 2.292 | 12.520 / 12.492 / 12.688 | 12.532 / 12.504 / 12.699 |
| 5000 | 0.944 / 0.931 / 1.012 | 1.187 / 1.176 / 1.240 | 8.041 / 8.002 / 8.276 | 0.023 / 0.022 / 0.023 | 2.218 / 2.197 / 2.321 | 12.412 / 12.357 / 12.660 | 12.423 / 12.369 / 12.671 |
| 10000 | 0.948 / 0.934 / 1.002 | 1.191 / 1.180 / 1.239 | 8.020 / 8.005 / 8.127 | 0.022 / 0.022 / 0.023 | 2.227 / 2.203 / 2.324 | 12.408 / 12.347 / 12.632 | 12.419 / 12.358 / 12.644 |
| 20000 | 0.953 / 0.940 / 0.997 | 1.192 / 1.180 / 1.237 | 8.226 / 8.252 / 8.382 | 0.021 / 0.022 / 0.023 | 2.224 / 2.201 / 2.306 | 12.617 / 12.590 / 12.811 | 12.629 / 12.602 / 12.823 |
| 50000 | 1.287 / 1.280 / 1.321 | 0.875 / 0.856 / 0.935 | 9.300 / 9.295 / 9.512 | 0.021 / 0.022 / 0.023 | 0.935 / 0.913 / 1.203 | 12.418 / 12.359 / 12.652 | 12.429 / 12.370 / 12.662 |

10k→50k总时间：12.408→12.418 ms，倍率1.001，增加0.1%。

### 隔离optimizer / zero_grad

上一段阶段event允许CPU继续enqueue：50k backward更长会隐藏后一阶段CPU launch gap，导致阶段optimizer计时低于1k，不能解释为Adam工作量减少。以下在阶段开始前同步，独立warmup100/measurement500，N=0表示固定TF参数池、没有Gaussian输入。zero_grad保留参考gradbuffer以隔离set_to_none控制成本，内存回收另见native流水阶段。

| 操作 | CUDA mean/p50/p95 ms | wall mean/p50/p95 ms |
|---|---:|---:|
| adam | 0.9653 / 0.9513 / 1.0578 | 0.9726 / 0.9584 / 1.0646 |
| preview | 2.1139 / 2.1053 / 2.2037 | 2.1212 / 2.1126 / 2.2112 |
| zero_grad | 0.0021 / 0.0020 / 0.0020 | 0.0149 / 0.0149 / 0.0154 |

### Nsight Systems：生产C++直接trace，N=8491

Nsight Systems 2022.4.2.1，cuda/nvtx tracing，关闭CPU sampling/context-switch tracing；预热100次后由cudaProfilerStart/Stop采集10步regularized_adam。使用runtime correlation归属NVTX阶段，不依赖Python镜像。GPU active包含该阶段kernel、D2D copy和memset的区间并集。

| 阶段 | CPU NVTX mean ms | kernel数/step | kernel active ms/step | D2D copy数/step | memset数/step | GPU span空隙范围 |
|---|---:|---:|---:|---:|---:|---:|
| zero_grad | 0.029 | 0 | 0.000 | 0 | 0 | 无device工作 |
| forward | 1.470 | 81 | 0.424 | 0 | 0 | 69.3–73.9% |
| loss | 0.100 | 7 | 0.030 | 0 | 0 | 60.7–62.9% |
| regularization | 1.709 | 93 | 0.393 | 0 | 6 | 75.3–77.0% |
| backward | 13.025 | 602 | 2.220 | 156 | 4 | 79.2–80.5% |
| optimizer | 1.883 | 208 | 0.582 | 0 | 0 | 68.3–69.1% |
| synchronize | 0.058 | 0 | 0.000 | 0 | 0 | 无device工作 |

每步991个计算kernel，其中97.0%短于10us。instrumented wall mean=18.279 ms，GPU device work区间并集mean=4.097 ms，平均span空隙77.6%；**这不是无插桩的framework百分比**。Nsight明显放大短kernel的API/dispatch成本，不能拿18.28 ms替代前面的native CUDA event结果，也不能把CPU与GPU时间相加。

10步中cudaLaunchKernel 9,910次、cudaMemcpyAsync 1,560次、cudaMemsetAsync 100次、显式末尾cudaStreamSynchronize 10次。copy均为D2D，计时内没有H2D/D2H或cudaMalloc/cudaFree；allocator的Tensor/缓存复用成本仍可能存在，不能说没有allocation overhead。原始TF内部未新增sync。完整API CPU时间和每步span在native_nsys_summary.json，原始trace为native_nsys.nsys-rep/SQLite。

native GEMM/GEMV占kernel active time的6.8%，支持优先处理正则、plane scatter、内存操作与launch链条，而非只重写MLP。C++与Python/不同LibTorch构建的regularization/backward计数略有差异（native 93/602，镜像87/612）；镜像的32项parity只验证forward及其梯度，不能把镜像trace当作native精确计数。

| 序号 | native kernel（短名，完整名见JSON） | CUDA ms（10步） |
|---:|---|---:|
| 1 | `vectorized_elementwise_kernel<(int)4, at::native::FillFunctor<float>, at::detail::Array<char *, (int)1>>(int, T2, T3)` | 5.872 |
| 2 | `grid_sampler_2d_backward_kernel<float, int>` | 4.851 |
| 3 | `vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctor_add<float>, at::detail::Array<char *, (int)3>>(int, T2, T3)` | 3.915 |
| 4 | `elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda(at::TensorIteratorBase &)::` | 2.589 |
| 5 | `vectorized_elementwise_kernel<(int)4, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, at::de` | 2.053 |
| 6 | `reduce_kernel<(int)512, (int)1, at::native::ReduceOp<float, at::native::MeanOps<float, float, float, float>, unsigned int, float, (int)4>>(T` | 1.957 |
| 7 | `ampere_sgemm_32x128_tn` | 1.515 |
| 8 | `elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::bina` | 1.270 |
| 9 | `vectorized_elementwise_kernel<(int)4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, at::de` | 1.025 |
| 10 | `elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<float>>(at::TensorIteratorBase &, c` | 0.880 |
| 11 | `vectorized_elementwise_kernel<(int)4, at::native::neg_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 2)]::operator ()() const::[` | 0.858 |
| 12 | `vectorized_elementwise_kernel<(int)4, at::native::addcdiv_cuda_kernel(at::TensorIteratorBase &, const c10::Scalar &)::[lambda() (instance 2)` | 0.837 |
| 13 | `vectorized_elementwise_kernel<(int)4, at::native::sqrt_kernel_cuda(at::TensorIteratorBase &)::[lambda() (instance 2)]::operator ()() const::` | 0.770 |
| 14 | `grid_sampler_2d_kernel<float, int>` | 0.734 |
| 15 | `vectorized_elementwise_kernel<(int)4, at::native::addcmul_cuda_kernel(at::TensorIteratorBase &, const c10::Scalar &)::[lambda() (instance 2)` | 0.705 |
| 16 | `vectorized_elementwise_kernel<(int)4, at::native::CUDAFunctorOnSelf_add<float>, at::detail::Array<char *, (int)2>>(int, T2, T3)` | 0.687 |
| 17 | `vectorized_elementwise_kernel<(int)4, at::native::BUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float>>, at::de` | 0.655 |
| 18 | `elementwise_kernel<(int)128, (int)2, void at::native::gpu_kernel_impl_nocast<at::native::BUnaryFunctor<float, float, float, at::native::bina` | 0.646 |
| 19 | `ampere_sgemm_32x32_sliced1x4_nt` | 0.568 |
| 20 | `CatArrayBatchedCopy<at::native::OpaqueType<(unsigned int)4>, unsigned int, (int)2, (int)64, (int)64>(T1 *, at::native::CatArrInputTensorMeta` | 0.482 |

### Python镜像分段归因

全部stage的mean/p50/p95在stage_benchmark.csv。这里列mean ms；每个planequery/product的event分段会放大插桩成本，不能与native total求和。

| N | input prep | 12 plane queries合计 | fusion合计 | MLP | output transform |
|---:|---:|---:|---:|---:|---:|
| 1000 | 0.122 | 1.020 | 0.312 | 0.296 | 0.392 |
| 5000 | 0.124 | 1.025 | 0.310 | 0.315 | 0.397 |
| 10000 | 0.124 | 1.025 | 0.299 | 0.315 | 0.398 |
| 20000 | 0.124 | 1.024 | 0.294 | 0.347 | 0.389 |
| 50000 | 0.123 | 1.022 | 0.292 | 0.721 | 0.079 |

### Python Profiler：真实N=8491，预热100次、3次采集，以下计数均除以3

使用CUDA runtime correlation将kernel归到CPU record_function范围，避免Kineto external-ID歧义。CPU profiler/shape/memory插桩影响绝对gap；launch数和短kernel分布用于结构归因。500次性能测量见native/分段CSV，这3次trace用于补充operator/shape/memory，C++真实launch计数见上面Nsight表。

| 模式 | forward kernels | backward kernels（含probe loss） | optimizer kernels | regularization forward kernels | aten calls | empty allocation calls | <10us kernel占比 |
|---|---:|---:|---:|---:|---:|---:|---:|
| core | 81 | 243 | 208 | 0 | 1445 | 113 | 96.3% |
| regularized | 81 | 612 | 208 | 87 | 3743 | 311 | 97.1% |
| preview | 81 | 612 | 260 | 87 | 4133 | 389 | 97.2% |

计时内没有H2D/D2H；copy为D2D（mean每步core10次、regularized166次、preview270次）。2次cudaDeviceSynchronize来自diagnostic显式末尾同步/profiler收尾，不是TF内部插入的同步。CPU launch gaps明显，详见各stage GPU span与kernel active time；allocator存在大量empty调用，但本次不能量化cudaMalloc占总时间的独立比例。

| 模式 | forward GPU kernel active ms/iter | backward active ms/iter | optimizer active ms/iter | forward span空隙范围 | backward span空隙范围 | resident baseline MiB | peak allocated MiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| core | 0.369 | 1.134 | 0.478 | 90.0–90.8% | 83.3–83.9% | 35.35 | 46.66 |
| regularized | 0.371 | 1.956 | 0.471 | 89.7–90.0% | 89.8–90.2% | 35.35 | 54.18 |
| preview | 0.377 | 1.979 | 0.567 | 89.7–90.0% | 89.5–90.2% | 35.35 | 54.18 |

上述空隙包括launch/CPU执行与copy等待，不能标成纯framework百分比；baseline包含模型参数、Adam、grads、真实input和诊断上下文；peak-resident更适合比较额外autograd工作区，仍不是只算TF常驻池的值。

### 最贵20个operator / kernel

regularized模式。operator按aten名称跨shape汇总self CUDA时间；kernel按实际device duration汇总。两种表不能相加，否则双重计数。完整kernel模板名、shape、次数、CPU/CUDA/memory在profiler_summary.json。

| 序号 | aten operator | self CUDA us（3步） | count（3步） |
|---:|---|---:|---:|
| 1 | `aten::copy_` | 1824.0 | 744 |
| 2 | `aten::fill_` | 1524.0 | 762 |
| 3 | `aten::grid_sampler_2d_backward` | 1406.0 | 36 |
| 4 | `aten::add_` | 905.0 | 396 |
| 5 | `aten::mul` | 710.0 | 282 |
| 6 | `aten::mean` | 608.0 | 63 |
| 7 | `aten::div` | 437.0 | 183 |
| 8 | `aten::addmm` | 417.0 | 21 |
| 9 | `aten::mul_` | 367.0 | 156 |
| 10 | `aten::mm` | 364.0 | 42 |
| 11 | `aten::sub` | 296.0 | 105 |
| 12 | `aten::sum` | 232.0 | 42 |
| 13 | `aten::addcdiv_` | 203.0 | 78 |
| 14 | `aten::neg` | 203.0 | 99 |
| 15 | `aten::grid_sampler_2d` | 200.0 | 36 |
| 16 | `aten::sqrt` | 181.0 | 84 |
| 17 | `aten::addcmul_` | 174.0 | 78 |
| 18 | `aten::add` | 144.0 | 72 |
| 19 | `aten::cat` | 142.0 | 45 |
| 20 | `aten::clamp_min` | 78.0 | 24 |

| 序号 | kernel（短名，完整名见JSON） | CUDA us（3步） |
|---:|---|---:|
| 1 | `vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, at::detail::Array<char*, 1> >` | 1518 |
| 2 | `grid_sampler_2d_backward_kernel<float, int>` | 1406 |
| 3 | `vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, at::detail::Array<char*, 3> >` | 911 |
| 4 | `elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::direct_copy_kernel_cuda` | 664 |
| 5 | `reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::MeanOps<float, float, float, float>, unsigned int, float, 4> >` | 590 |
| 6 | `vectorized_elementwise_kernel<4, at::native::AUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, at::detail` | 499 |
| 7 | `ampere_sgemm_32x128_tn` | 417 |
| 8 | `elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::Mu` | 334 |
| 9 | `vectorized_elementwise_kernel<4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, at::detail` | 244 |
| 10 | `elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::CUDAFunctor_add<float> >` | 236 |
| 11 | `vectorized_elementwise_kernel<4, at::native::addcdiv_cuda_kernel` | 203 |
| 12 | `grid_sampler_2d_kernel<float, int>` | 200 |
| 13 | `vectorized_elementwise_kernel<4, at::native::neg_kernel_cuda` | 194 |
| 14 | `vectorized_elementwise_kernel<4, at::native::sqrt_kernel_cuda` | 181 |
| 15 | `vectorized_elementwise_kernel<4, at::native::addcmul_cuda_kernel` | 174 |
| 16 | `vectorized_elementwise_kernel<4, at::native::CUDAFunctorOnSelf_add<float>, at::detail::Array<char*, 2> >` | 162 |
| 17 | `elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BUnaryFunctor<float, float, float, at::native::binary_internal::Mu` | 162 |
| 18 | `ampere_sgemm_32x32_sliced1x4_nt` | 162 |
| 19 | `vectorized_elementwise_kernel<4, at::native::BUnaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, at::detail` | 158 |
| 20 | `CatArrayBatchedCopy<at::native::OpaqueType<4u>, unsigned int, 2, 64, 64>` | 122 |

GEMM/GEMV kernel合计229.3 us/step，占regularized已观测kernel active time 7.2%。这不支持“MLP GEMM占主导”；内存/填零/copy、plane backward及regularization和大量launch更值得优先处理。未用Nsight Compute测DRAM/L2/occupancy，因此不能给出精确显存带宽占比。

### 真实历史输入N分布

只统计历史原始TF日志中LoD步骤的正tf_N（初始化前0另列）；来自多场景UltraFast轮转与稀疏normal日志，带调度/场景偏差。

| ratio | 正N样本数 | zero N样本数 | min | median | p90 | max |
|---:|---:|---:|---:|---:|---:|---:|
| 0.01 | 4628 | 4434 | 1 | 987.0 | 1918.2 | 6467 |
| 0.05 | 4621 | 4436 | 1 | 1069.0 | 4430.0 | 8253 |
| 0.1 | 4625 | 4430 | 1 | 1662.0 | 10340.0 | 18363 |
| 0.15 | 4621 | 4433 | 1 | 2044.0 | 15922.0 | 27969 |

### 最终性能判断

在该fast模型/realistic重复采样负载下，原始regularization使10k训练步从5.006增加到11.202 ms，额外6.197 ms，占regularized total的55.3%。这是两组诊断的差值归因，含CPU/operator/backward调度差异，不等于正则纯GPU算力成本。regularized和preview对N不敏感，纯core也远小于5倍增长；固定全plane工作和framework/launch/optimizer overhead是主要可优化方向，不能把主要收益只押在MLP FLOPs。

推荐优先顺序：保语义融合正则forward/backward与optimizer事务buffer操作；融合plane query+product/coordinate；再比较GEMM周边融合与fast H32 warp MLP。自定义backward需要plane atomic、MLP dW reduction和normal输入梯度，custom optimizer需要prepare/commit、缺失grad和per-tensorstep。

原始实现确有10k→50k时间接近不变现象，尤其包含正则/preview时；数据已分模式给出。该结果不能推导完整Mapping加速倍数，不能归因于所有GPU配置，也不支持删除rotation/ChangeDistill/regularization。

### 复现命令与交付文件

```bash
cd /home/prmi/zwj/mycartgs/tests/diagnostics/original_tf_cuda_audit_20261013
bash run.sh
bash run_nsight.sh
# 有源码/配置修改时先重新检查inventory.json与local_manifest.json
# 单独重建trace归因：
/home/prmi/anaconda3/envs/zwj/bin/python reparse_profile.py
/home/prmi/anaconda3/envs/zwj/bin/python parse_nsight.py
```

run.sh固定CUDA11.8、Torch_DIR和TORCH_CUDA_ARCH_LIST=8.6；诊断CMake适配服务器3.16.3。checkpoint/input/state/config/sourcehash、native CSV/逐样本CSV/finite检查、分段CSV、profiler trace+summary、native parity archive+JSON、autograd graph、构建log、复现脚本、曲线SVG/PNG均在同目录。没有rendered RGB。原始checkpoint不修改，native使用新模型副本。

曲线点为mean，阴影为p50–p95；完整统计见上表。
![N增长曲线](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/original_tf_cuda_audit_20261013/native_scaling.png)
