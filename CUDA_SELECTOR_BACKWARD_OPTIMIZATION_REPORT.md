# CUDA Selector backward 优化报告

结论：**`CUDA_SELECTOR_CORRECTNESS = PASS`**。建议合入本轮 final 分支梯度路由与 ReLU 融合；`Selector.backend = legacy | cuda` 保留，配置默认仍为 legacy。没有修改 forward 数学或任何 loss/sampling/optimizer/Guard 规则。

本轮 M=219969、100 warmup + 500 measurements 的同步 wall：CUDA isolated backward **2.27463 → 2.03620 ms**（1.117×，降低 10.48%）；同进程交替顺序 paired F+B **4.28667 → 4.03152 ms**（1.063×）；structural caller **5.03120 → 4.80687 ms**（1.047×）。每项before/after使用同binary。完整 Mapping 未测量，不以 caller 的收益代替 Mapping 收益。

## 1. 版本、环境与修改范围

起点 HEAD：`4f056ec0096ba5cc4f6af3659638f3ae78be400e`；分支 `cuda-selector`。基线冻结/profiler 提交 `52fc6b4`；直接路由提交 `434c838`；final+ReLU 提交 `474e8e0`；最终小 batch 修正提交 `33ebe0792e9e4fb06727dc578aaf4034727882ba`。本报告随后单独提交，避免把报告自身 hash 写成循环引用。

RTX3090 / CUDA11.8.89 / GCC9 / Torch2.3.1，FP32、TF32=false、current CUDA stream。Native LibTorch ABI=1，Python Torch ABI=0，分别编译。隔离实验目录 `/home/prmi/zwj/selector-cuda-20261015`；生产 `/home/prmi/zwj/mycartgs` 的本轮训练配置、二进制和代码未覆盖。

产品改动只有 `include/selector_cuda.h`、`src/selector_cuda.cu`。thin autograd 与 forward 源码未改；网络层、参数布局/identity、checkpoint、quaternion/scale activation、Gaussian input gradients、ST、RNG、GI、Protection、Replay/CD/DSD、Adam/Guard 数学保持原样。原有未提交工作保留，未 push。运行验收使用隔离环境中冻结的其他依赖；当前工作区其他未提交修改不属于此报告的验证范围。

所有可复现源、脚本、CSV、源码/binary hash、Nsight trace、候选补丁与原始验收归档位于 `tests/diagnostics/cuda_selector_backward_20261015`（下文简称 D）。最终源 SHA-256：

```json
{
  "src/selector_cuda.cu": "999c67b9a3103ed849b11f13e9168ac095e109917ef71233bd554e1f6efd1bd8",
  "src/selector_cuda.cpp": "1890228a4161c78c5140b46409a4082d2d04828e510f2391d6a12708caee57a0",
  "include/selector_cuda.h": "4c711b9645582a480c0744f39134fe6bd35b9cdfca411c93d023b580fb73e790"
}
```

## 2. 基线重新测量与 profiler 分解

改代码前，冻结 HEAD/status/hash，重新顺序测量 legacy 与 CUDA 八规模。首批 M219969 wall：legacy F/B/F+B=3.96291/2.29648/6.16905 ms，CUDA=2.09733/2.24939/4.27948 ms；不是引用旧报告数字。D/baseline 保存此批数据。最终又在同一候选 binary 内重测 legacy/原 CUDA/final，并额外重复两批；下文最终收益以这组新 A/B 为准。

初始 NVTX+Nsight M219969：43 kernels，active 2.17933 ms，device span 2.23963 ms。主要成本是 bias/ratio reduction 约0.409 ms、split+ReLU2约0.357 ms、ReLU1约0.214 ms、branch2 dX约0.346 ms；final dFeature约0.139 ms，final dW约0.143 ms，branch2/branch1 dW约0.255/0.156 ms。原实现已经有 scalar ratio reduction；没有重复实现两层大 M ratio GEMM。

最终同一 profiler executable、10 个 captured backward 的 before/after：

| backward 阶段 | 原 CUDA active ms | 新 CUDA active ms |
| --- | --- | --- |
| bias/ratio reduction | 0.411018 | 0.412812 |
| final dFeature | 0.139419 | 0.000000 |
| final dW | 0.143604 | 0.138888 |
| split + ReLU2 | 0.357705 | 0.000000 |
| branch2 dX | 0.346460 | 0.345362 |
| branch2 dW | 0.255464 | 0.255529 |
| ReLU1 / scalar ReLU | 0.213960 | 0.213471 |
| XYZ/q/scale dX | 0.156878 | 0.159509 |
| branch1 dW | 0.156335 | 0.154858 |
| ratio small GEMM | 0.005901 | 0.005888 |
| direct final + ReLU2 | 0.000000 | 0.247659 |


表中 cuBLAS split-K reduction 跟随对应 GEMM 归类；bias/ratio reduction 保留 ATen tree。ratio small GEMM 约为微秒级，不是主要收益。input preparation / quaternion normalize / scale exp 在上层图，network profiler 不将其混入 MLP backward；caller 则包含这些直接相连的输入准备与 VJP。

## 3. 候选、验收与取舍

| 候选 | 正确性 | M219969 wall（B / F+B / caller） | 决策 |
|---|---|---|---|
| 原 CUDA strategy1 | frozen oracle | 见最终 A/B | 保留诊断 oracle |
| B：direct final，单独 ReLU | 七快照与7000步 exact | 2.15860 / 4.20197 / 4.93702 ms（探索批） | 已单独提交，随后由 C 改善 |
| C：direct final + ReLU2 | 最终完整 gate exact | 2.03620 / 4.31709 / 4.80687 ms | 采用 |
| warp tree narrow dX | 满足旧 tolerance，但新增梯度 relative L2≈1.17e-7 | 未继续验收 | 拒绝，基线 exact，未放宽标准 |
| ordered-FMA warp narrow dX | 七快照/7000步 exact | 2.54948 / 4.60709 / 5.26858 ms（探索批） | 拒绝，低 lane 利用率使整体回退 |
| shared tile spatial ReLU+dX | 七快照/7000步 exact，小 batch 36个输入梯度 Tensor不 exact | 独占重复 B=2.40994/2.41045 ms，C=1.99468 ms | 拒绝，增加空间未换得速度 |

空间不作为 gate：已按用户补充要求尝试保留三分支 dH 并共享 tile。候选9初始 forward 也异常变慢，存在 CPU 编译并行干扰，初始 `measure_9` 不用于性能结论；`clean9` 顺序重测仍慢，因而拒绝。所有失败候选只保留 patch/证据，不在产品 enum/hot path 中。每个通过单步的候选才进入7000步训练，再进行八规模性能/caller；warp tree 在 exact 检查失败后停止。

## 4. 最终 kernel 结构与小 batch 分派

M>16 时去掉完整 `[M,128] dFeature` 分配/GEMM。每个 branch 用 current-stream CUDA kernel 按原两项乘加顺序生成 `[M,32]` 的第二层 preactivation VJP，同时应用原 ReLU mask；不先写 `[M,128]` 再 split。final dW 继续使用原 packed forward feature 做完整128维 cuBLAS GEMM，forward 累加顺序完全没动。

本轮额外边界扫描发现 cuBLAS 对 M≤16 的 k=2 GEMM 有不同累加顺序，直接公式会新增约1e-8～1e-7差异。因此 M≤16 继续走原 FusedActivation backward；不是用 tolerance 接受差异。修正后与原 CUDA strategy1 的 1408 条 gradient Tensor 比较 exact，覆盖 M=0/1/2/3/7/12～25/31/32/33/63/64/65/127/128/129/257/1000/5000/10000、scalar/row。这项补充测试比较新旧 CUDA，不额外宣称所有小 batch 的 legacy bitwise 结果。

scalar ratio 仍先生成正确逐点 contribution、应用同一个第二层 mask、用原 ATen sum得到 `[1,32]`，只 backward 一次 ratio MLP。row ratio 保持逐行 generic path。没有将 `sum(dZ*W)` 改成 `sum(dZ)*W`，避免 reduction 顺序变化。`[M,32]` ratio contribution 仍存在。

branch2 dX、branch1 XYZ/q/scale dX 和全部 dW 继续 cuBLAS；narrow custom dX 的已测实现没有获得合格整体收益。db 继续 ATen `sum(0)`，没有重新引入此前已拒绝的 fused bias reduction。ReLU 的零点与 NaN 条件沿用原实现的 `h<=0 ? 0 : upstream`，未改为 `>=0` 或新的判断式。

## 5. 七 checkpoint direct parity 与1000-step

复用 `CUDA_SELECTOR_DIRECT_PARITY_REPORT.md` 的同输入 oracle，每个快照只加载一次基础数据，clone 为 legacy A/B/CUDA。四 ratio .01/.05/.10/.15 × scalar/row；全部八层 branch、logits、probability、production elastic probability、cached uniform、六 seed fixed-noise Gumbel。

最终 **318,118 项比较，0 failure，全部 exact**：max abs/mean abs/RMS/p99/relative L2=0，cosine=1；XYZ、activated quaternion/scale、ratio、raw quaternion/log-scale/opacity，以及全部18参数 VJP 均 exact。固定 `grad_logits/grad_probability/grad_ST`，ratio/GI/photo-opacity/combined/Guard uniform/Replay/CD/DSD 的外部 target/upstream 完全共享，没有分别运行 renderer 生成不同 upstream。

cached-uniform 56组、5,151,304个决定，mismatch=0；Gumbel 336组、30,907,824个决定，mismatch=0；compact IDs/count exact，无新增 near/far-boundary disagreement。网络不消费 RNG，保留原 CUDA generator/noise 顺序。all-tie 仅 diagnostic，不作 gate。

| checkpoint | M | steps | parameter / Adam m/v relative L2 | average hard mismatch |
| --- | --- | --- | --- | --- |
| room_current | 102643 | 1000 | 0 / 0 / 0 | 0.0 |
| desk | 52860 | 1000 | 0 / 0 / 0 | 0.0 |
| xyz | 31609 | 1000 | 0 / 0 / 0 | 0.0 |
| long | 129459 | 1000 | 0 / 0 / 0 | 0.0 |
| earlier_5329 | 160289 | 1000 | 0 / 0 / 0 | 0.0 |
| middle_6900 | 109564 | 1000 | 0 / 0 / 0 | 0.0 |
| later_16400 | 57489 | 1000 | 0 / 0 / 0 | 0.0 |


全部7000步的 logits/probability/loss、gradient、参数、Adam m/v/step、selected IDs exact；1/10/100/1000步另有逐 Tensor记录。四组 direct/preview Adam 各1000步同样 exact，覆盖 normal/zero/undefined grad、LR=0、非默认 beta/eps/decay、不同 per-tensor step、commit/discard/shrink/projection。Native Guard 六种 Gaussian/Selector/TF 联合事务 exact；Native CD/DSD/Replay/GradientConstraint 4/4通过。legacy A vs B repeat error=0，CUDA error=0；没有新的 selected-ID、gradient、parameter 或 Adam 漂移。

完整 renderer 质量/SLAM 不作为本轮 gate，也未运行在线统计实验。

## 6. 八规模性能与独立复测

以下均为本次 frozen workload、100 warmup +500 measurements，单位 ms。同步 host wall；CUDA event 的对应 mean/p50/p95保存在CSV/final_summary.json。F+B是fresh graph，B是相同saved graph VJP。没有省略 Gaussian input gradient。

| M | legacy B | 原 CUDA B | 新 CUDA B mean/p50/p95 | CUDA F+B 前→后 | CUDA forward 前→后 |
| --- | --- | --- | --- | --- | --- |
| 1000 | 0.6717 | 0.3421 | 0.3251/0.3233/0.3316 | 0.5130 → 0.5019 | 0.0979 → 0.0973 |
| 5000 | 0.7097 | 0.6361 | 0.6008/0.6009/0.6137 | 0.7466 → 0.7501 | 0.1174 → 0.1180 |
| 10000 | 0.7314 | 0.3714 | 0.3679/0.3667/0.3868 | 0.5609 → 0.5651 | 0.1494 → 0.1508 |
| 20000 | 0.7292 | 0.3650 | 0.3730/0.3720/0.3781 | 0.5698 → 0.5700 | 0.2346 → 0.2332 |
| 50000 | 0.7284 | 0.6805 | 0.6247/0.6164/0.6588 | 1.1491 → 1.0862 | 0.5301 → 0.5289 |
| 100000 | 1.1920 | 1.1776 | 1.0591/1.0585/1.0662 | 2.1065 → 1.9974 | 1.0047 → 1.0104 |
| 200000 | 2.1216 | 2.0746 | 1.8394/1.8385/1.8474 | 3.9169 → 3.6933 | 1.9095 → 1.9222 |
| 219969 | 2.3220 | 2.2746 | 2.0362/2.0359/2.0474 | 4.3319 → 4.3171 | 2.1215 → 2.1216 |

| M | legacy caller | 原 CUDA caller | 新 CUDA caller mean/p50/p95 | 相对原 CUDA | 相对 legacy |
| --- | --- | --- | --- | --- | --- |
| 1000 | 2.6054 | 1.4471 | 1.4312/1.4302/1.4407 | 1.011× | 1.821× |
| 5000 | 2.7027 | 1.4159 | 1.4606/1.4589/1.4682 | 0.969× | 1.850× |
| 10000 | 2.6964 | 1.4469 | 1.4566/1.4507/1.4922 | 0.993× | 1.851× |
| 20000 | 2.7978 | 1.5377 | 1.4972/1.4870/1.5259 | 1.027× | 1.869× |
| 50000 | 3.3634 | 1.7860 | 1.7952/1.7924/1.8116 | 0.995× | 1.874× |
| 100000 | 4.2018 | 2.7750 | 2.6757/2.6723/2.7013 | 1.037× | 1.570× |
| 200000 | 6.6901 | 4.6187 | 4.4050/4.4020/4.4315 | 1.048× | 1.519× |
| 219969 | 7.2510 | 5.0312 | 4.8069/4.7985/4.8385 | 1.047× | 1.508× |


M219969另外两批顺序复测（新→原，和主批原→新交替）：

| 独立批次 | 原 CUDA B | 新 CUDA B | 原 CUDA F+B | 新 CUDA F+B |
| --- | --- | --- | --- | --- |
| 2 | 2.28638 | 2.01869 | 4.93115 | 4.38608 |
| 3 | 2.29959 | 2.01887 | 5.08810 | 4.33633 |


上述顺序批次的M219969主批F+B mean仅改善0.34%，但另两批改善更大，故追加了同一进程、相同参数/输入/allocator环境、每轮交替`1→6 / 6→1`的paired实验。每个strategy仍有100 warmup与500 measurements；没有修改网络或VJP数学。下面是更能隔离host/进程顺序影响的结果：

| M | 原 CUDA paired F+B mean/p50/p95 | 新 CUDA paired F+B mean/p50/p95 | speedup |
| --- | --- | --- | --- |
| 1000 | 0.5346/0.5308/0.5678 | 0.5280/0.5239/0.5590 | 1.013× |
| 5000 | 0.7407/0.7426/0.7511 | 0.7281/0.7301/0.7370 | 1.017× |
| 10000 | 0.5725/0.5643/0.6016 | 0.5660/0.5593/0.5951 | 1.011× |
| 20000 | 0.6659/0.6762/0.6908 | 0.6585/0.6647/0.6866 | 1.011× |
| 50000 | 1.1442/1.1440/1.1533 | 1.0818/1.0813/1.0914 | 1.058× |
| 100000 | 2.0971/2.0968/2.1077 | 1.9786/1.9778/1.9903 | 1.060× |
| 200000 | 3.8861/3.8850/3.9018 | 3.6548/3.6542/3.6694 | 1.063× |
| 219969 | 4.2867/4.2862/4.3063 | 4.0315/4.0306/4.0480 | 1.063× |


paired的M219969 event F+B：4.27630→4.02147 ms；forward part event 2.13083→2.13170 ms，backward part event 2.14547→1.88978 ms。fresh-forward后的cache状态与isolated backward不同，不能将这个backward part数值当成独立Bbenchmark。paired FB trace/解析另存D/paired_fb.*、paired_profile_summary.json。

最终相对 legacy：forward 1.890×、backward 1.140×、F+B 1.449×、structural caller 1.508×。caller包含本 fixture 的概率/ST/Protection/compact/gather、photo-opacity/ratio VJP与Selector update，**不包括 renderer/TF/Gaussian optimizer 或完整 Mapping**。本轮没有 Mapping wall speedup结论。

八规模包含 launch/host 占比较高的小 M，不能把大 M收益外推为所有 M同比例提升。F+B 的host span在重复批次中有可见波动，因此保留每批 mean/p50/p95，不用其中最低值代表最终收益。forward源码未变，其小幅计时差按噪声解释，不能称为新的 forward优化。

## 7. Nsight before/after、资源与内存

| M | 原/新 kernels | 原/新 active ms | 原/新 device span ms |
| --- | --- | --- | --- |
| 10000 | 43 / 42 | 0.2656 / 0.2446 | 0.5824 / 0.5341 |
| 50000 | 43 / 42 | 0.6419 / 0.5764 | 0.7028 / 0.6372 |
| 100000 | 43 / 42 | 1.0902 / 1.0140 | 1.1510 / 1.0768 |
| 219969 | 43 / 42 | 2.1867 / 1.9340 | 2.2472 / 1.9942 |


M219969最终 kernel 列表：

| kernel | launch/iteration | active ms | register/thread | shared bytes/block |
| --- | --- | --- | --- | --- |
| `void at::native::reduce_kernel<(int)128, (int)4, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::...` | 9 | 0.404047 | 48 | 16/2064 |
| `ampere_sgemm_32x32_sliced1x4_nt` | 6 | 0.351724 | 86 | 16384 |
| `ampere_sgemm_64x64_nn` | 3 | 0.342172 | 126 | 8448 |
| `void selector_cuda::<unnamed>::directFinal<(bool)1>(const float *, const float *, float *, const float *, int, int, int)` | 4 | 0.247659 | 16 | 0 |
| `selector_cuda::<unnamed>::reluVjp(float *, const float *, int, int)` | 4 | 0.213471 | 16 | 0 |
| `ampere_sgemm_32x128_nn` | 3 | 0.159509 | 57 | 16384 |
| `void cutlass::Kernel<cutlass_80_simt_sgemm_128x32_8x5_nt_align1>(T1::Params)` | 1 | 0.134913 | 94 | 25600 |
| `void splitKreduce_kernel<(int)32, (int)16, int, float, float, float, float, (bool)1, (bool)0, (bool)0>(cublasSplitKParams<T6>, const T4 *, const T5 *, T5 *, ...` | 7 | 0.059876 | 43 | 0 |
| `void at::native::reduce_kernel<(int)256, (int)2, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::...` | 1 | 0.008765 | 32 | 2064 |
| `std::enable_if<!T7, void>::type internal::gemvx::kernel<int, int, float, float, float, float, (bool)0, (bool)1, (bool)0, (bool)0, (int)7, (bool)0, cublasGemv...` | 1 | 0.003232 | 165 | 544 |
| `void gemvNSP_kernel<float, float, float, float, (int)1, (int)32, (int)4, (int)1024, (bool)0, cublasGemvParams<cublasGemvTensorStridedBatched<const float>, cu...` | 1 | 0.003190 | 64 | 1024 |
| `void gemmk1_kernel<int, float, (int)256, (int)5, (bool)0, (bool)0, (bool)1, (bool)0, cublasGemvTensorStridedBatched<const float>, cublasGemvTensorStridedBatc...` | 1 | 0.002762 | 20 | 1024 |
| `void gemmk1_kernel<int, float, (int)256, (int)5, (bool)0, (bool)0, (bool)0, (bool)0, cublasGemvTensorStridedBatched<const float>, cublasGemvTensorStridedBatc...` | 1 | 0.002656 | 20 | 1024 |


| 版本 | memset/iteration | clear bytes/iteration | memset active ms | captured runtime API/iteration |
| --- | --- | --- | --- | --- |
| 原 CUDA | 8 | 32 | 0.009286 | cudaMemsetAsync_v3020=8, cudaLaunchKernel_v7000=43 |
| 新 CUDA | 8 | 32 | 0.009181 | cudaMemsetAsync_v3020=8, cudaLaunchKernel_v7000=42 |


本次backward capture没有H2D/D2H/D2D copy，也没有cudaMalloc/cudaFree或显式同步API；8次memset仅清零ATen reduction的少量控制/计数空间，共32 bytes，约0.009ms。不能把这部分称为大Tensor清零瓶颈。profile外部的warmup、fixture载入、事件等待和全局同步仍存在，分别用于准备数据和统一计时。

原始trace与解析结果见 D/profile_final_1、D/profile_final_6；另有219969 row-ratio trace。表中长模板名缩写，完整demangled name见原始JSON。register/shared-memory来自 CUPTI trace。Nsight Compute明确返回 `ERR_NVGPUCTRPERM`，因此**DRAM bytes与 achieved occupancy 未实测**，没有用理论值冒充counter。Nsight kernel active/device span与未profile的 wall分开报告。runtime API、copies/memsets也在 profile_summary.json；ATen caching allocator内部CPU开销未逐次归因，不能从GPU空隙倒推出具体Tensor allocation成本。

关键中间量：

| Tensor | 原/新 | 生命周期与逻辑字节（M219969，FP32） |
|---|---|---|
| forward packed `[M,128]` | 都保留 | saved until backward，112,624,128 bytes；确保forward/final dW顺序 |
| final dFeature `[M,128]` | 原保留，新在M>16消除 | 原从final dX到最后branch；112,624,128 bytes |
| branch dPre2 `[M,32]` | 都保留 | 逐branch，28,156,032 bytes；ratio后sum |
| branch dH1 `[M,32]` | 都保留 | 当前branch两层VJP之间，28,156,032 bytes |
| scalar ratio dH `[1,32]` | 都保留 | reduction后仅128 bytes，两层小GEMM |
| input gradients `[M,3/4/3]` | 都返回 | autograd上层继续使用 |

消除dFeature自身一次写回和一次读取，逻辑量为225,248,256 bytes/调用；这仅是该workspace的Tensor流量差，不是整个backward的实测DRAM差，也未扣除分支重复读dZ的缓存行为。

F+B incremental allocated peak：原 CUDA **363.71 MiB**，新 CUDA **270.79 MiB**，减少 92.92 MiB。它是消除冗余往返的副作用；显存不作为本轮目标或gate。未新增全局预分配workspace、异步跨stream事务或CUDA Graph。

## 8. 仍保留的成本与最终判断

主要收益来自取消完整 dFeature GEMM/global roundtrip，并将其按branch生成与ReLU2合并。final/各branch dW的batch reduction、精确兼容的bias/ratio sum、branch2与窄input dX仍是主要剩余成本；它们在当前已验证实现中保留原cuBLAS/ATen顺序。本轮拒绝了更改累加树的方案，也拒绝了correctness通过但完整F+B/caller变慢的方案。

建议合入本轮两个成功的backward优化和小batch回退；启用CUDA backend后自动使用新backward。是否改变产品默认backend、完整Mapping收益和用户最终在线质量测试属于独立决策。本报告不声称已稳定低于2.0ms、达到1.5～1.8ms理想目标，或所有GPU/库版本均bitwise一致。当前证据适用于本轮RTX3090/CUDA11.8/Torch环境及覆盖的输入。

最终回答：isolated backward 2.27463→2.03620ms；paired F+B 4.28667→4.03152ms，caller 5.03120→4.80687ms；correctness仍为 **PASS**，最终被采用实现没有观察到任何新增selected-ID、gradient、parameter或Adam漂移。失败候选已退出生产路径。
