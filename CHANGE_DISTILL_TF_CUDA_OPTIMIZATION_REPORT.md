# ChangeDistill TF Teacher Geometry Transport CUDA 优化报告

日期：2026-10-18。正式项目：`/home/prmi/zwj/mycartgs`。GPU：RTX 3090，CUDA 11.8，LibTorch 2.3.1 ABI1。

## 1. 结论与正式启用状态

保留 event-local batch TransportPlan、compact layout、逐组原 GEMM、融合 gather / translation / scale / quaternion / scatter。CUDA Graph 不进入正式代码。Stage A/B/C 和独立 `selector_epochs` / `tf_epochs` 均保留。

正式共享库编译后，正反顺序各 5 次、每版本每分支合计 **10 个 warm samples**：

| 分支 | 原版 median ms | 新版 median ms | 加速比 | wall 降幅 | 原版 min / max / std | 新版 min / max / std |
|---|---:|---:|---:|---:|---|---|
| tf | 456.201 | 200.012 | 2.281× | 56.16% | 451.923 / 458.973 / 2.118 | 198.559 / 202.001 / 0.938 |
| selector | 189.527 | 188.706 | 1.004× | 0.43% | 187.141 / 190.538 / 1.164 | 187.908 / 189.686 / 0.550 |
| both | 641.264 | 371.056 | 1.728× | 42.14% | 635.129 / 649.240 / 4.879 | 369.941 / 372.465 / 0.729 |

`tf` 是 TF-only finalize 窗口；`both` 是含 Selector、TF、buildEvent、AABB 处理、全部 pre/post/audit evaluation 的 finalize 窗口，沿用原审计 benchmark 定义。snapshot / teacher cloning 在主窗口外，仍正常执行，其时间独立保存在每个 `runtime.json` 的 `event_preparation_ms`；没有通过移动正常执行边界获得提速。

正式构建成功。CTest **61/62**；唯一失败 `ultrafast_mapping_cuda`，消息为 `Mode1 tail advanced clocks without Gaussian/TF optimizer updates`。上一轮已用旧 binary + 旧 library 复现同一失败，证据在 `change_distill_optimization_20261017/deployment_20261018/tail_counterfactual.json`。本轮全部 ChangeDistill、cache、Selector CUDA、TF CUDA 和其他 lifecycle tests 通过；未修改 Mapping clock 来掩盖该失败。

## 2. 冻结基线与改动边界

Git HEAD：`0268edd9780e5dba006efa719f627cc59d0e5a65`，工作树有既存未提交改动。本轮未 reset、未 checkout 整个文件、未 commit。Windows 初始快照与服务器生产源码的归一化检查 `server_source_drift.json=[]`；全部无关 tracked diff 和冻结源文件保持原样，见 `integrity.json`。

| 文件 | 初始 normalized SHA256 | 最终 normalized SHA256 |
|---|---|---|
| src/change_distill.cpp | `31b6a308784ee797b6b57f1a31c8d5820b488f1fdcde137b47c9934ffe7a237a` | `66dbf8135898f1373b10c805d48f7aeb5f9ced34a8b9f4c1ad659f5d4665edb7` |
| CMakeLists.txt | `a0053e032271e54b556b99def9068566de4526c92b22a5df1bcd91d52220c9e2` | `485dcd8a5e21f953443e9585cd448401d8fdb2b051e6fc544072ea0dfa88281c` |
| include/change_distill_transport_cuda.h | 新文件 | `d0fae453196c264630bf3a91800a0766500a3b5d8a0772a2c44ff57edc6badee` |
| src/change_distill_transport.cu | 新文件 | `ab6a7894a4d28a027ad306447be00529ec6c227c34cd789849cfe1581dd9e793` |

完整基线 32 文件清单：`baseline/raw/source_manifest.json`；完整初始 / 最终 Git diff 和独立补丁：`baseline/working_tree.diff`、`final_working_tree.diff`、`production_transport_v2.patch`。

正式 mapper：`674444e1bf191f1afc8d014940f238822c2399dcbf6083a28fca19956493dfeb`。其余产物见 `deployment/binary_manifest.json`。所有编译命令已保存，CUDA 保持 `--fmad=false`，未启用 fast-math / FP16 / BF16 / TF32。诊断 runner 链接冻结的正式 CUDA archives，避免复用旧审计中编译选项不同的 object。

## 3. 固定输入和 benchmark 合约

事件：`results/change_distill_ablation/20261017-2038/events_supplement/local_loop/event_6207_1`，直接复用原 snapshot，没有重跑 SLAM。

| 项目 | 数值 |
|---|---:|
| 全部 Gaussian | 85653 |
| mature / corrected mature / unaffected mature | 44889 / 29687 / 15202 |
| TF in-domain corrected / correction groups | 29683 / 144 |
| batch / ratios（保持顺序） | 8192 / [0.01, 0.05, 0.10, 0.15] |
| Selector / TF epochs | 5 / 5 |
| 每分支实际 optimizer steps | 20 |

每个 run 恢复相同 Teacher / Student 初始参数，构建独立 manager、event-local cache 和新的临时 Adam。normal Gaussian 属性和 normal optimizer 不变均有运行断言。一个现有训练 batch 累加四个 ratio 的 loss 后执行一次 backward 和一次 step，未改变该语义。所有 loss、权重、reduction、anchors、outside_identity、ratio 顺序和评估保持不变。

正式正反对照在全部编译完成后运行，GPU compute-process idle gate 和 0.25 秒进程监测未发现其他实验进程。每次 invocation 含 cold E5、两个恢复状态的 E1 warm-up，然后 E5 warm repetitions；std 为总体标准差。没有在已训练模型上继续测量。

## 4. 原调用路径与已测瓶颈

```text
ChangeCoordinator::finalizeGeometryChange
 -> buildEvent（Gaussian ID -> correction group；重叠/重复约束检查）
 -> ChangeDistillManager::execute
 -> TargetTransport(event)：group_id table D2H，一次
 -> 初始 primary / audit evaluate：batch -> ratio -> errors
 -> FrozenTeacherTargetCache::get，lazy miss
 -> old_state.select(missing) -> frozen Teacher TF::applyLogResidual
 -> transportTarget -> TargetTransport::apply
 -> batch IDs D2H + 隐式同步 -> CPU membership scan -> positions H2D
 -> 每 active group：三个 gather -> applyGaussianTransform
 -> XYZ scale / GEMM / translation；scale add；quaternion multiply / normalize
 -> 三个 out-of-place index_copy（伴随全 batch D2D copy）
 -> cat XYZ / log-scale / quaternion -> event-local target cache
 -> Student TF / loss -> backward -> Adam step
```

Phase 0 真实 trace：20 个 miss，**5 个 unique ordered ID sets**，每个跨四个 ratio 重用。三个 8192 点 primary batch 分别有 36 / 66 / 95 个 active groups，5107 点尾批有 115 个；8192 点 audit batch 为 0。总计 `4*(36+66+95+115)=1248` 个 active-group transform。原始逐 miss ID、hash、group counts 保存在 `baseline/raw/layout_trace.json`。

原 transport 不是在做大型矩阵运算：主要是大量 tiny pointwise/index kernel、重复 dispatcher 调用、CPU 等待、membership 往返和 out-of-place index_copy 的整批复制。

## 5. 新路径与数值约束

```text
event -> TargetTransport：CPU ID table + GPU contiguous [G,23] correction metadata
 -> 每个初始 evaluation population：一次 IDs D2H，CPU stable segmentation
 -> 一次 packed positions / group IDs H2D；各 batch 共享 GPU storage 和 CPU offsets
 -> 四个 ratio：Teacher TF 仍分别计算
 -> cache miss：复用该 batch 的 TransportPlan，无 membership D2H/H2D
 -> 一个 compactXYZKernel：group lookup + gather + 原 FP32 scale multiply
 -> 每 active group：原 Mx3 × 3x3 GEMM，写 compact product buffer
 -> 一个 scatterCompactKernel：translation + scale + quaternion + normalize + scatter
 -> 原 target packing / cache / Student / losses / backward / optimizer
```

`src/change_distill.cpp` 的 `TransportBatchPlan`、`TargetTransport::prepare/apply`、TF `errors/evaluate` 只改 Teacher transport。`src/change_distill_transport.cu` 提供两个 kernel；CMake 将其加入现有 TF CUDA target。

GPU metadata 每组 23 个 FP32：16 个 matrix 元素、4 个 wxyz、scale、log(scale)、flags。两个 int64 packed row 分别存 batch positions 和 group IDs；CPU segment 保存 group 和稳定 offset。存储 O(N+B+G)，没有 [B,G] mask 或逐点扫描所有 groups。

保留 row-vector convention：`(xyz * float_scale) @ matrix[:3,:3] + matrix[3,:3]`。从 `matmul` 的内部 MM 路径改为 `mm_out` 写预分配 compact slice，M、K=3、矩阵 view、组和行顺序保持；XYZ 在真实事件、所有 ratio 和 fixture 上 bitwise 相同。**没有使用逐点 BMM。**

scale 为 `log_scaling + std::log(float_scale)`。rotation 左乘 correction；sum3 使用 `(p0+p2)+p1`，sum4 使用 `(p0+p1)+(p2+p3)`；显式 `__fmaf_rn` 只匹配原 ATen cross 的三个分量。36 组合探针中 mode16 在 100000 个 quaternion 上唯一 bitwise 匹配，见 `quaternion_probe.csv`。保持 `--fmad=false`，没有全局改变浮点融合策略。

未启用的属性和 group=-1 保持原值；只对开启 ROTATION 的 correction 执行原有 normalize。零 selected group 无 kernel。CPU / 非 FP32 继续使用原 group-based ATen 路径。合法的正有限 scale 检查保留。

plans 只用于已知初始 evaluation batch。cache callback 的 missing filter 保持顺序，`missing.numel()==batch.numel()` 证明缺失集合就是完整 planned batch；部分缺失走通用 builder，不假定任何新调用都缓存完毕。Stage B 的 direct cache-hit 路径没有改动。没有跨 optimizer step 缓存 Student features。

| 操作 | 执行位置与分配 | 同步 / 传输 | 频率 |
|---|---|---|---|
| event metadata | CPU ID table；GPU [G,23] FP32，stack/cat 临时 storage | 一次 table D2H；小常量 H2D | EVENT-ONCE |
| population planning | CPU stable vectors；GPU packed [2,P] int64；batch 仅持有 view / offsets | population IDs D2H 阻塞；packed H2D | 每 event 两个 population |
| Teacher gather / forward | GPU 属性 index_select、HexPlane / MLP activations | 保留原 lazy cache-miss pipeline 的同步边界 | 20 misses |
| output base | GPU [B,3]、[B,3]、[B,4] clone | 3 个 D2D copy；无 D2H | 16 corrected misses |
| compact gather | GPU [Bc,3] FP32 新 buffer，一个 kernel | 无主机同步 / copy | 16 corrected misses |
| product workspace / group MM | GPU [Bc,3] 新 buffer；每 group 的 slice 不分配 tensor storage；保留 cuBLAS workspace 策略 | 无主机同步；原 group GEMM | 16 workspace，1248 MM |
| fused postprocess / scatter | 一个 kernel 写 output base；不分配 per-group tensor | 无 D2H/H2D、无额外 synchronize | 16 corrected misses |
| packing / cache | GPU [B,10] cat，原 index_copy_ / cached gather | 保留原 cache 行为 | 20 packs；后续 direct hits |

scale、quaternion 和 normalize 的单独耗时已合并进 fused postprocess，无法再从单个 kernel 内用 CUDA Event 无扰动拆分。

## 6. 分阶段实验与拒绝项

以下为独立诊断 executable 的探索测量，每个通过候选通常 10 warm samples；最终结论使用第 1 节正式库正反对照，不能混加二者。

| 候选 | TF median ms | both median ms | 校验 / 结论 |
|---|---:|---:|---|
| 当前 A+B+C | 461.456 | 635.593 | 本轮 fresh baseline |
| Phase1 population plans | 460.179 | 未单独测 | 20 steps 通过；单独收益接近噪声，不能声称独立稳定提速；作为 compact layout 必需索引计划保留并以最终组合验收 |
| 初次全融合 quaternion | 单次 221.409 | 未测 | **拒绝**：20-step max abs 3.434e-5，逐元素 allclose 失败；forward 很小不足以接受 |
| 保留原 ATen quaternion | 395.986 | 567.365 | 通过，较保守 fallback 实验；被更快且 bitwise target 的最终实现替代 |
| 匹配归约树的逐组融合 | 218.144 | 400.544 | TF-only / both 20-step 通过，target 尚有约 1.19e-7 差异 |
| compact gather / mm_out / scatter | 198.448 | 368.498 | 20-step 通过，减少跨 group launch |
| compact + ATen cross FMA 匹配 | 196.151 | 374.667 | target bitwise；最终保留 |
| 独立 CUDA Graph prototype | 208.936 | 381.375 | 20-step 通过，但比对照 compact 更慢；**拒绝** |

当前 CUDA11.8 cuBLAS 没有 `cublasSgemmGroupedBatched` API；未升级工具链，也未用已拒绝的逐点 BMM 替代。采用共享 compact layout / 跨组 fused postprocess，仍逐组发起原尺寸 GEMM。

Graph prototype 使用 frozen Teacher 输入 persistent buffers、side stream warm-up/capture；三个 full-batch pattern 和尾批各一个 graph，含初次 capture / instantiate 和输入 copy 成本。每个 event 每图仅四个 ratio 使用，训练阶段 cache 已命中，graph 无法从 20 个 optimizer steps 获得额外 transport 重用。实测未摊平 setup 成本。没有捕获 optimizer，也没有把 setup synchronize 放入生产代码。

## 7. 正确性：完整 targets、梯度、20 步与 lifecycle

固定 `atol=2e-5, rtol=2e-4`，从未放宽。真实全 85653 点、全部 144 groups、四个 ratio 的 XYZ / log-scale / quaternion，以及真实 cache miss packed targets：**max abs=0，mean abs=0，bitwise=True**。

TF-only correctness run 首步所有梯度最大差 `6.985e-10`，最大 tensor mean abs `9.542e-14`；首步 Adam 更新 bitwise 相同。完整 event 对应梯度最大差 `5.821e-10`，最大 tensor mean abs `1.053e-13`，首步更新同样 bitwise。40 个记录 loss（cold/warm 各 20 步）最大差分别 `3.056e-10` / `1.164e-10`。原始逐 tensor 记录保存在 `final/raw/parity_baseline_*.json`、`verification.pt`。

独立最终诊断 TF-only 最终参数 max abs `2.31713e-5`、both `9.76026e-6`，均逐元素 allclose。max abs 可以略高于 atol；实际判据包含 rtol，不能仅用全局 max 与 atol 比较。

正式库的正反顺序最终参数，以及同版本重复控制：

| 对照 | max abs | 最大 tensor mean abs | bitwise | allclose |
|---|---:|---:|---|---|
| forward_tf | 1.36345625e-05 | 4.58901582e-07 | False | True |
| reverse_tf | 5.78165054e-06 | 2.18442651e-07 | False | True |
| baseline_production_repeat_tf | 1.50799751e-05 | 5.27844406e-07 | False | True |
| production_repeat_tf | 2.02655792e-06 | 9.92316629e-08 | False | True |
| forward_selector | 0 | 0 | True | True |
| reverse_selector | 0 | 0 | True | True |
| baseline_production_repeat_selector | 0 | 0 | True | True |
| production_repeat_selector | 0 | 0 | True | True |
| forward_both | 1.16676092e-05 | 4.73822581e-07 | False | True |
| reverse_both | 9.11951065e-06 | 2.80115842e-07 | False | True |
| baseline_production_repeat_both | 8.01682472e-06 | 2.79825258e-07 | False | True |
| production_repeat_both | 1.25169754e-05 | 4.98104669e-07 | False | True |

原版本自己重跑也不是 TF 参数 bitwise 一致；TF query backward 原有 `atomicAdd`（`src/transform_field_cuda.cu:116`）提供了数值波动的源码依据。这里不据此放宽 gate：正式全部比较仍通过相同 tolerance；Selector 参数在两种顺序下始终 bitwise 相同。

正式 10 warm pairs 的 full-precision weighted pre/post metrics 也逐一比较通过：TF pre 最大差 0；TF-only post 最大差 `1.922e-10`，both post `3.849e-11`；Selector pre/post 最大差均为 0。untouched audit 三分量在现有四位小数日志精度上全部相同；这里没有把日志精度冒充 full-precision audit aggregate 校验。原始完整 targets / first-step / losses / parameters 另有完整精度验证。见 `production_metrics.json`。

通过的 fixture：local LC6207、真实 global scale、真实 AABB4000 / AABB5000、no groups、all -1 with groups、独立 mixed flags、identity、非单位 correction quaternion、outside domain、empty batch、repeated Gaussian IDs、single/small batch、尾批、零 selected group、one selected point。重叠/重复 correction membership 的既存约束未改。分别见 `final/raw/verify_tf/transport_parity.json`、`lifecycle_summary.json` 和 `deployment/ctest.xml`。

## 8. one-time / per-miss 组件时间

独立 mode1，当前 stream CUDA Events，结束后统一解析；无逐操作同步。下表为 inclusive scope，子项**不能再次加到父项**。

| Component | Frequency | Calls | 原 CPU / span ms | 新 CPU / span ms | 新 CPU / call ms |
|---|---|---:|---:|---:|---:|
| transport_metadata | EVENT-ONCE | 1 | 0.337 / 0.343 | 0.668 / 0.675 | 0.668135 |
| transport_plan_population | EVENT-ONCE / population | 2 | 未单独插桩 | 0.674 / 0.684 | 0.336889 |
| geometry_transport | RATIO-ONCE / cache miss | 20 | 297.135 / 294.396 | 20.169 / 18.029 | 1.008473 |
| transport_compact_gather | RATIO-ONCE / corrected batch | 16 | 未单独插桩 | 0.093 / 0.077 | 0.005841 |
| transport_xyz_gemm | RATIO-ONCE / active group | 1248 | 未单独插桩 | 12.736 / 14.200 | 0.010205 |
| transport_fused_postprocess_scatter | RATIO-ONCE / corrected batch | 16 | 未单独插桩 | 0.065 / 0.094 | 0.004048 |
| target_pack | RATIO-ONCE / cache miss | 20 | 0.223 / 0.262 | 0.233 / 0.274 | 0.011661 |

membership population planning 只发生两次（primary 和 audit），不在 epoch / optimizer loop 重算。teacher targets 仍按 ratio event-local 缓存；E5 的训练和后评估直接 cache gather，未省略评估。

## 9. Nsight kernel / copy / launch 证据

| 指标 | A+B+C baseline | 最终 |
|---|---:|---:|
| TF 总 kernel calls | 49119 | 20452 |
| TF <10us kernel calls | 48414 | 19813 |
| geometry kernel calls | 30024 | 1352 |
| geometry attributed GPU ms | 95.624531 | 4.940699 |
| geometry <10us kernels | 29975 | 1352 |
| geometry D2H calls / bytes | 20 / 1212000 | 0 / 0 |
| geometry H2D calls / bytes | 16 / 949856 | 0 / 0 |
| population-plan D2H / H2D | 无单独阶段 | 2 / 303000 bytes；1 / 474928 bytes |
| geometry D2D calls / bytes | 3744 / 352180640 | 48 / 4749280 |
| geometry runtime API calls / ms | 34604 / 151.896747 | 2164 / 6.097891 |
| geometry inter-kernel gap ms | 385.117817 | 17.445865 |
| geometry between-launch API gap ms | 356.364121 | 17.908335 |

event metadata 还保留一次 group table D2H 和小的常量 H2D，计入 `transport_metadata`，没有隐藏为零 copy。Nsight 原始 `.nsys-rep`、SQLite export、kernel CSV、stats CSV、进程监测全部保存。SQLite 分析只给派生 export 添加索引，原 `.nsys-rep` 不变。

原 geometry gather/index kernels：indexSelect 3744 次、index_copy 3744 次；最终 geometry 内均为 0，由 16 次 compact gather 和 16 次 fused postprocess/scatter 替代。GEMM 1248 次 host call 对应 1320 个 GPU kernels（少量 split-K reduce 等），未删除真实矩阵工作。主要名称仍为 `ampere_sgemm_128x32_nn`、`ampere_sgemm_32x32_sliced1x4_nn`、`gemmSN_NN_kernel`。每种完整 kernel 名、calls、total/average 时间见 `*/raw/nsys_tf/kernels.csv` 和 `nsys_summary.json`。

CUDA Event span 包含 stream 等待和 CPU 发射空隙，不等于纯 kernel time；Nsight GPU time 是相关 kernel duration 求和。Nsight 的 host/API/gap 数字受 CUPTI instrumentation 影响，不可直接加到 native wall。硬件 counters 未获得，SM 利用率、compute-bound / bandwidth-bound 仍 **inconclusive**；可确定的是本路径的 launch、copy 和 orchestration 工作量已大幅下降。

## 10. CPU wall、CUDA span 与内存

| 分支 | 原 CUDA span median ms | 新 CUDA span median ms |
|---|---:|---:|
| TF-only | 456.194 | 200.006 |
| Selector-only | 189.521 | 188.700 |
| both | 641.257 | 371.050 |

native wall 在 end CUDA Event synchronize 后结束，span 与 wall 很接近；这并不代表 GPU 全程忙碌。cold、warm-up、全部 warm samples、preparation 时间保留在 `runs.json` / `runtime.json`。

| 分支 | 原 peak MiB | 新 peak MiB | 新增 peak KiB | 原 / 新 additional MiB |
|---|---:|---:|---:|---:|
| tf | 95.575 | 95.588 | 13.000 | 51.421 / 51.434 |
| selector | 81.145 | 81.145 | 0.000 | 37.499 / 37.499 |
| both | 99.936 | 99.949 | 13.000 | 51.759 / 51.771 |

additional 为事件窗口 peak allocated 减窗口初始 allocated，不是 nvidia-smi 总进程显存。索引 workspace 在初始 evaluation 后释放，训练期峰值主要只增加持续存活的 correction metadata；无 [B,G] 分配。

## 11. 可复现命令与保存位置

服务器目录：`/home/prmi/zwj/mycartgs/results/change_distill_tf_cuda_optimization/20261018`。
本地同名诊断：`tests/diagnostics/change_distill_tf_cuda_optimization_20261018`。

```bash
cd /home/prmi/zwj/mycartgs/results/change_distill_tf_cuda_optimization/20261018
PY=/home/prmi/anaconda3/envs/zwj/bin/python
$PY build.py baseline native
$PY build.py final native
# 新 OUTPUT 名，避免覆盖原始结果；固定 event / E5 / B8192 / 4 ratios。
$PY run.py final:tf:new_tf:native:10 final:selector:new_selector:native:10 final:both:new_both:native:10
# 正反顺序脚本和命令已执行并保存；重跑前应复制到新的优化目录。
$PY paired.py production baseline_production
```

正式 baseline runner 通过 `LD_LIBRARY_PATH=deployment/backup/lib` 加载保存的 A+B+C mapper；优化 runner 加载当前项目 `lib`。两侧 `ldd.txt` 已保存。正式构建和 CTest 命令在 `deployment/commands.json`，CUDA object 以源码+header+编译选项 hash 缓存；升级环境须重编并重验。

`raw_profiling_and_sources.tar.gz` 包含所有新 native/component/profiler CSV/JSON/PT/Nsys、通过和失败 candidate 源码、构建命令和日志；校验值在 `archive_manifest.json`。不包含 core dump、object、binary backup；这些仍在服务器原目录，冻结库和正式二进制 hash 已记录。原审计及上一轮优化输出没有覆盖。

`metrics_supplement.tar.gz` 另存最终报告、追加的正式 metrics 派生校验及脚本；其 manifest 记录主归档 hash，原始 runtime / stdout 均已包含在主归档中。最终权威报告是项目根目录本文件。

## 12. 已知限制与下一步

保留的是 GPU-resident **batch execution plan**，不是全 GPU 动态分组：event-once CPU segmentation 和 host GEMM sizes 仍存在，未声称彻底消灭所有 D2H。未知 partial miss 使用安全通用 builder，可能再次规划；LC6207 的 20 misses 全部重用已知 plans。

quaternion 的 bitwise 匹配依赖当前 LibTorch 2.3.1 / CUDA11.8 的归约树和 cross 实现。换 Torch/CUDA/GPU 后必须重跑 primitive、全 targets、first-step 和 20-step gate，不能只用本次误差结论。

Teacher transport 已从主要瓶颈降为小项。下一优先级是现存 TF Student forward / HexPlane backward / evaluation 与 loss scalar 同步；本轮不改变这些语义。stage B 的 cache coverage 与 stage C 的 Selector shared embedding 没有退化。

**保留：batch plan + compact group layout + 原 group GEMM + strict fused postprocess/scatter。拒绝：未匹配数值的首次 quaternion 融合和无端到端收益的 CUDA Graph。旧逐点 BMM 没有重新引入。**
