# ChangeDistill 运行时优化报告

日期：2026-10-18（延续 2026-10-17 审计与同一优化批次）。

## 结论

保留 Stage A 的保守 group membership 方案、Stage B 的有覆盖证明的 cache-hit 路径、
Stage C 的 batch 内 Selector 属性 embedding 复用。拒绝逐点 BMM transport 实验；
不修改 Stage D 输入梯度路径。没有提交 commit，没有更改正式 YAML，没有重新运行 SLAM。

在同一 LC 6207、batch=8192、四个 ratio、每分支 5 epochs / 20 optimizer steps 下，
完整事件的正反顺序复测中位数从 **759.142 ms** 降至 **619.347 ms**，
提速 **1.226×**，耗时减少 **18.41%**。
Selector 独立分支的改善可重复；Stage C 单独对完整事件的边际收益较小，
不能把所有独立分支的节省直接相加。具体限制与负面结果见下文。

## 1. 输入、源码与测量契约

开始时逐项核对原报告、`raw/source_manifest.json` 和当前生产源码：32 个文件的 SHA-256 全部一致，
`drift=[]`。此基线包含用户已有未提交修改，不等于 Git HEAD。
初始完整 Git diff、staged diff、status、HEAD 保存在 `baseline/`。
独立 `selector_epochs` / `tf_epochs` 及未设置时回退 `epochs` 的行为完整保留。

本任务相对初始工作树只改变 6 个生产文件和 1 个 cache 测试；其他 manifest 文件保持相同哈希。
详见 `integrity.json`、`production_optimization.diff` 和完整 `final_working_tree.diff`。
原审计目录、冻结源码及原始 profiling 文件没有改写。

真实输入：

| 项目 | 保持的值 |
| --- | --- |
| event | Local LC 6207 / event_6207_1 |
| Gaussian 总量 | 85,653 |
| mature / corrected mature / unaffected mature | 44,889 / 29,687 / 15,202 |
| TF in-domain / outside skipped | 29,683 / 4 |
| correction groups | 144 |
| ratios（顺序不变） | 0.01, 0.05, 0.10, 0.15 |
| batch / epochs | 8192 / Selector=5, TF=5 |
| optimizer steps | Selector=20；TF=20；完整事件=40 |

Selector 保持每 epoch 4 个 stratified batches；四个 ratio 的 loss 按原顺序累加，
每 batch 只有一次 backward / Adam step。TF 保持每 epoch 4 batches，最后一批 5107 点。
Selector anchors、TF audit 集合及其随机序列、outside_identity、AABB、loss 权重/reduction 均未改变。
所有 pre/post 和逐 epoch evaluation、untouched audit 都在 native 总时间内。

服务器：`172.19.163.58`，RTX 3090，driver 535.230.02，CUDA 11.8，sm_86。
原生 runner 链接项目 LibTorch 2.3.1 / ABI1；Python torch 2.3.1+cu118 / ABI0 仅做离线分析。
编译保持 `-O3 -DNDEBUG -std=c++17`；FP32、TF32 设置及 Selector CublasStrict / DirectFinalRelu 均不变。
环境与输入哈希见 `environment.json`、`input_sha256.json`、各阶段 `raw/source_manifest.json`。

每次 run 都从完全相同的 teacher 参数恢复 Student，新建 manager、Adam 与 event-local cache。
Gaussian 不变、normal optimizer state 不变和实际 step 数由原有 harness 断言。
先 cold run，再 2 次独立恢复的一 epoch warm-up，再测 warm runs，绝不在已训练模型上接着计时。
初始各阶段 10 次 warm；另做两个正反顺序块、每块 5 次 warm，主表采用这 10 次复测。
`monitor.json` 记录 benchmark PID 和 GPU 进程；交错复测没有其他 GPU 进程污染。
CPU 调度、GPU 时钟和 allocator/plan 状态仍会产生波动，因此同时保留初始结果和两个块的结果。

native mode0 无细粒度 hooks。总 CPU wall 在 final CUDA Event 完成后读取；
CUDA Event 测相同 current stream 的跨度，包含 CPU 发射不足导致的 stream 空闲，**不是纯 kernel time**。
teacher clone/snapshot preparation 单列 `event_preparation_ms`；主表 event wall 对应完整 coordinator finalize。
mode1 组件是独立插桩 executable 的 inclusive CPU/CUDA spans，不能跨嵌套层级相加或直接当作 native wall 分解。
Nsight Systems 的 kernel duration 另列。未改驱动权限、未绕过原审计的 ERR_NVGPUCTRPERM；
本轮没有硬件 counter，因此不宣称整个 CD compute-bound 或 memory-bound。

## 2. 原瓶颈与 Stage A

原 `transportTarget` 每次 miss 对 144 个 group 执行 `ids==g → nonzero → gather → applyGaussianTransform → index_copy`。
20 次 miss 共 2880 次 group 检查；其中 4 次是 untouched audit，实际没有任何应 transport 的点，仍扫描全部 group。
真实 event 的 `correction_group_id[N]` 已由 `buildEvent` 构造并验证无重复/交叠；-1 表示未修正。
本轮不改变 coordinator、group 构造或 Gaussian-ID 对应关系。

从原实现确认的精确语义：

```text
XYZ flag:   matmul(xyz * float_scale, matrix[0:3,0:3]) + matrix[3,0:3]
SCALE flag: log_scaling + std::log(float_scale)
ROT flag:   normalize(multiply(correction_wxyz, target_wxyz))
```

矩阵是转置存储的 row-vector 约定；quaternion correction 左乘；无 flag 的属性保持原值；
-1 点不能被附带 normalize。

### 被拒绝：逐点 BMM

`stage_a/` 将组变换打包一次，按 group ID gather，再按点 BMM；没有 `[B,G,...]` 中间量。
XYZ 最大绝对误差 9.53674e-7，log-scale/quaternion 逐位一致，所有 transported targets 通过前向容差。
native TF 中位数 207.078 ms，但 20 步后参数 p22 最大误差 3.35574e-5，部分元素不满足原
`atol=2e-5, rtol=2e-4`。首步梯度/更新通过仍不足以验收完整训练。
因此撤销该实现，没有放宽容差。BMM 与按组 GEMM 的 FP32 累加路径不同；
L1 在接近 target 的位置对微小数值差异敏感，不能仅凭前向误差小就保留。
完整失败数据在 `stage_a/raw/comparison_baseline_tf.json`。

### 保留：TargetTransport 的 host membership 方案

`src/change_distill.cpp::TargetTransport` 在 TF event 开始只复制一次 group-ID 表到 CPU。
每个 Teacher miss 将当前 batch IDs 复制到 CPU，单次 O(B) 遍历按 group 收集局部位置，
将所有位置合并为一次 H2D，再按原 group 顺序及原组内点顺序调用原 `applyGaussianTransform`。
保留原 GEMM 的形状、原 log/rotation 算式以及三次 out-of-place index_copy。
不在 GPU 上重复动态发现 144 个 group，不会修改 frozen snapshot 的别名。

这是保守替代方案，并未声称实现了完整点级 transport fusion。
它仍有 EVENT-ONCE group ID D2H、每 miss 一次 batch-ID D2H、适用 group 的 gather/transform/index_copy。
audit 全为 -1 时直接返回；空 batch、空 group、无 group、混合 flags 和非单位 identity quaternion 均验证。
group-ID CPU 表额外 685,224 bytes；当前 batch host indices/positions 规模 O(B)，没有 O(BG) tensor。

Stage A 的 TF 正反顺序复测：558.907 → 468.734 ms，**1.192×**。
真实 event 的所有点、四个 ratio 的 Teacher 输出及常规 cache-miss 批次 packed targets：
transport 最大/平均绝对误差均为 0。最终训练误差和真实 fixtures 结果另见正确性表。

## 3. Stage B：有覆盖证明的 cache-hit 路径

`FrozenTeacherTargetCache::getCached` 只执行 NoGrad + cached index_select。
`get(..., fully_cached)` 仅在调用方有证明时分流；未知 coverage 保持原 lazy missing-index 流程。
没有每次 `.all().item()`，没有从这次事件推广“所有 anchors 永远已缓存”的假设。

| 调用集合 | 使用直接 gather 的证明 |
| --- | --- |
| Selector primary 训练/后评估 | baseline evaluate 已遍历整个 primary 集合、每个 ratio |
| Selector 固定 audit anchors 后评估 | baseline anchor_pre 已遍历同一个固定集合 |
| Selector epoch0 训练 anchors | 与 baseline anchor_pre 相同 |
| 后续重采样 anchors | 只有初始 anchors 数量等于全部 unaffected 数量，才证明已覆盖全部；否则继续 lazy |
| TF primary 训练/后评估 | pre evaluation 已填充所有 in-domain primary |
| TF audit_post | audit_pre 已填充同一个固定 audit 集合 |

每个 ratio、每个 event 仍是独立 cache。generic path、partial miss、重复 IDs、重复 hit、
返回值不别名 cache、不同 event 输入和 loss/gradient routing 均有测试。

Selector：231.266 → 196.885 ms，1.175×。
TF 在 Stage A 后：468.734 → 461.060 ms，1.017×，小但两个复测块方向一致。
Selector missing-index 检查 304→24 次，Teacher miss 仍为24；
TF missing-index 检查184→20次，Teacher miss仍为20。target gather 数和实际 Teacher 点集不变。
不把 target_cache 的 inclusive 时间与 cache_missing_indices 再相加。

## 4. Stage C：Selector 跨 ratio 复用

增加 `GumbelNetworkImpl::forwardProbabilities` → `selector_cuda::networkMany` → `spatialForward`。
同一调用中只计算一次 XYZ/rotation/scale 的六个 Linear+ReLU，ratio 两层 embedding 和
最终 packed `[B,128]→[B,2]` CublasStrict GEMM 仍各 ratio 独立执行。
`selectorProbabilities` 同时把 current Gaussian gather 与 activated scale exp 从每 ratio 移到每 batch/stratum。
legacy 路径继续使用原 forwardProbability。

这里只复用 **forward activation**。每个 ratio 的 custom Function 仍显式接入原18个参数、执行原 backward，
梯度通过这些参数边累加；不向已缓存的内部 activation 再回传一份梯度。
原 DirectFinalRelu 的路由、small-N fallback、参数顺序、逐 ratio loss 和一次总 backward/step 均保留。
没有换到 batched-GEMM candidate，没有共享 Student activation 到下一 optimizer step。
这也意味着尚未消除跨 ratio 重复的 embedding backward；本轮选择数值风险较低的 forward 复用。

真实 Selector event 中，Student forwardProbability 对应304个 ratio调用变成76个多-ratio调用，
仍输出304组 probability；六个空间 Linear 的 Student 执行数1824→456，减少1368次。
Teacher 24次 miss 的计算不变；训练 custom network backward 仍160次，optimizer step仍20。

专门测试 B=1、16、17、3800、7430、8192 × ratio_count=1、2、4：
输出、loss、输入梯度、全部参数梯度和单步 CUDA Adam 更新共762项全部逐位一致。
真实 event 重排后存在约1e-9级梯度累加误差，最终参数约1e-8级差异；均通过原容差，详见下表。

Stage C 相对 B 的 Selector：196.885 → 185.844 ms，**1.059×**。
完整事件的边际改善比独立 Selector 小：两个块 B→C 分别约0.16%与2.71%，
不足以宣称稳定的单独完整事件百分比；保留依据是完整 Selector 分支在两个块均改善约5%–6%。
TF 路径未因 Stage C 改变，B/C TF 时间差视为测量波动，不计作 Stage C 收益。

## 5. native 结果与内存

主表为两个正反顺序块合并的10次 warm run。单位 ms；方括号是 min–max，std 是总体标准差。

| 阶段 | 分支 | CPU median | min–max | std | CUDA span median | steps |
| --- | --- | --- | --- | --- | --- | --- |
| 初始 | selector | 231.266 | 228.581–235.525 | 2.625 | 231.259 | 20 |
| 初始 | tf | 558.907 | 544.820–570.096 | 11.669 | 558.898 | 20 |
| 初始 | both | 759.142 | 748.194–769.951 | 7.658 | 759.132 | 40 |
| A | tf | 468.734 | 462.996–473.908 | 4.361 | 468.727 | 20 |
| A | both | 673.509 | 670.488–683.242 | 4.434 | 673.500 | 40 |
| A+B | selector | 196.885 | 195.805–197.943 | 0.615 | 196.878 | 20 |
| A+B | tf | 461.060 | 456.313–473.567 | 5.015 | 461.052 | 20 |
| A+B | both | 625.121 | 622.044–628.159 | 2.215 | 625.112 | 40 |
| A+B+C | selector | 185.844 | 184.811–188.739 | 1.241 | 185.837 | 20 |
| A+B+C | tf | 459.888 | 448.427–463.575 | 4.450 | 459.880 | 20 |
| A+B+C | both | 619.347 | 606.057–624.199 | 6.369 | 619.338 | 40 |

初始单独批次10次 warm 的 native median（完整保留，用于检验批次/时钟差异）：

| 阶段 | Selector | TF | 完整事件 |
| --- | --- | --- | --- |
| 初始 | 228.566 | 545.969 | 757.151 |
| A | 未测/路径未变 | 461.389 | 700.557 |
| A+B | 208.913 | 456.904 | 653.081 |
| A+B+C | 183.695 | 446.763 | 617.664 |

不能把分支独立 benchmark 的时间直接相加作为完整事件时间：独立 TF case 使用原配对审计的固定 TF seed，
完整事件使用生产 RNG 流程；两者各自在对应 before/after 内严格相同。
主结论只比较同一种 case；原报告225.970/556.474 ms只是历史参考。

GPU allocator peak 和相对 event 开始的 additional peak，单位 MiB（不含 CUDA context/driver 的整卡占用）：

| 阶段 | 分支 | peak allocated | additional peak |
| --- | --- | --- | --- |
| 初始 | selector | 96.855 | 53.209 |
| 初始 | tf | 95.575 | 51.421 |
| 初始 | both | 101.613 | 53.436 |
| A | tf | 95.575 | 51.421 |
| A | both | 101.613 | 53.436 |
| A+B | selector | 96.855 | 53.209 |
| A+B | tf | 95.575 | 51.421 |
| A+B | both | 101.613 | 53.436 |
| A+B+C | selector | 81.145 | 37.499 |
| A+B+C | tf | 95.575 | 51.421 |
| A+B+C | both | 99.936 | 51.759 |

A/B没有增加观测到的 GPU peak；A另有约669 KiB CPU group-ID表。
C降低独立 Selector peak约15.71 MiB，完整事件peak约1.68 MiB；不是通过减少样本或loss实现。

## 6. 组件与 kernel 证据

以下为一次代表性 mode1 run 的 inclusive component，总时间不能与子项相加。
EVENT-ONCE metadata 与 per-miss transport 明确分开：

| 阶段 | 组件 | 频率 | calls | CPU ms | CUDA span ms | CPU ms/call |
| --- | --- | --- | --- | --- | --- | --- |
| 初始 | tf/geometry_transport | CACHE-MISS | 20 | 356.498 | 353.706 | 17.8249 |
| A | tf/transport_metadata | EVENT-ONCE | 1 | 0.363 | 0.368 | 0.3629 |
| A | tf/geometry_transport | CACHE-MISS | 20 | 270.128 | 267.350 | 13.5064 |
| 初始 | selector/cache_missing_indices | 每调用 | 304 | 55.025 | 18.066 | 0.1810 |
| B | selector/cache_missing_indices | 每调用 | 24 | 1.374 | 1.581 | 0.0573 |
| 初始 | tf/cache_missing_indices | 每调用 | 184 | 10.242 | 11.185 | 0.0557 |
| B | tf/cache_missing_indices | 每调用 | 20 | 1.249 | 1.409 | 0.0624 |
| B | selector/student_forward | 每调用 | 304 | 48.630 | 41.861 | 0.1600 |
| C | selector/student_forward | 每调用 | 76 | 34.178 | 27.402 | 0.4497 |
| C | selector/backward | 每调用 | 20 | 73.237 | 121.972 | 3.6619 |

Nsight Systems：同一event、E5，各trace一个完整分支，含evaluation。

| 阶段 | 分支 | kernel calls | sum kernel GPU ms | <10us kernels |
| --- | --- | --- | --- | --- |
| baseline | tf | 61479 | 243.579 | 60779 |
| stage_a_grouped | tf | 49939 | 210.999 | 49282 |
| stage_b | tf | 49119 | 200.649 | 48451 |
| baseline | selector | 23073 | 175.727 | 20930 |
| stage_b | selector | 21673 | 172.352 | 19509 |
| stage_c | selector | 18025 | 154.106 | 15874 |

按NVTX的CUDA launch关联得到的代表性归因：

| 阶段 | 组件 | kernel calls | kernel GPU ms |
| --- | --- | --- | --- |
| stage_c | selector/selector_network_backward | 6080 | 104.765 |
| stage_a_grouped | tf/geometry_transport | 30024 | 90.173 |
| stage_b | tf/geometry_transport | 30024 | 86.312 |
| stage_b | tf/hexplane_backward | 1120 | 9.499 |
| stage_b | tf/output_backward | 80 | 0.526 |
| stage_b | tf/tf_mlp_backward | 2560 | 13.849 |
| stage_b | selector/selector_network_backward | 6080 | 105.699 |
| baseline | tf/geometry_transport | 41564 | 118.144 |
| baseline | selector/selector_network_backward | 6080 | 105.332 |

每条 kernel 名称、calls、总/平均时长见各阶段 `raw/nsys_*_kernels.csv`，原始 `.nsys-rep`、SQLite及stats也保留。
归因使用CPU launch correlation，不把GPU执行时间是否落在CPU range里当作归属。
`total.backward()` 会在autograd线程发射kernel，因此其整体范围统计覆盖所有捕获线程；
custom backward内部scope按实际发射线程归因。上述均为单独benchmark进程，不包含其他GPU进程。
组件范围有嵌套，不可叠加成总GPU时间。

kernel 数量的下降和 tiny kernel 比例支持“减少 orchestration/launch/indexing 有效”；
硬件occupancy、SM利用率、带宽瓶颈依然 inconclusive。不能根据 CUDA span 减少就宣称 tensor-core 利用率提高。

## 7. 正确性、失败与验证边界

保持 `atol=2e-5, rtol=2e-4`，没有放宽。下表的“mean最大值”是各 tensor 的 mean absolute error 再取最大，
完整逐 tensor 的 max/mean/shape/bitwise/allclose 数据均在 comparison JSON，不能理解成所有元素的合并mean。

| 比较 | tensor数 | max abs最大值 | mean abs最大值 | 结果 |
| --- | --- | --- | --- | --- |
| A：packed targets | 20 | 0 | 0 | 通过 |
| A：TF首步梯度 | 26 | 5.82077e-10 | 1.02401e-13 | 通过 |
| A：TF首步更新 | 26 | 0 | 0 | 通过 |
| A：TF逐步loss | 20 | 1.01863e-10 | 1.01863e-10 | 通过 |
| A：TF最终参数 | 26 | 9.93162e-06 | 4.38612e-07 | 通过 |
| B：Selector全部校验 | 56 | 0 | 0 | 通过 |
| B：Selector最终参数 | 18 | 0 | 0 | 通过 |
| B：TF最终参数vs A | 26 | 1.16378e-05 | 3.97105e-07 | 通过 |
| C：Selector首步梯度 | 18 | 3.72529e-09 | 3.4197e-10 | 通过 |
| C：Selector首步更新 | 18 | 0 | 0 | 通过 |
| C：Selector逐步loss | 20 | 2.32831e-10 | 2.32831e-10 | 通过 |
| C：Selector最终参数 | 18 | 1.49012e-08 | 1.16415e-10 | 通过 |
| 组合：完整事件校验 | 148 | 3.72529e-09 | 5.23869e-10 | 通过 |
| 组合：完整事件最终TF参数 | 26 | 2.09138e-05 | 6.91412e-07 | 通过 |

`verification.pt` 保存每次TF miss的packed target、每步loss、首步全部参数梯度和首步更新。
native warm_0另存20步后最终参数。完整事件也独立比较；Gaussian、normal Adam和steps断言均通过。
TF grid backward本身含atomicAdd，重复训练参数允许既有容差下的微小非逐位差异；不要求不切实际的全模型逐位确定性。

额外验证：

| 真实fixture | 校验tensor数 | max abs最大值 | mean abs最大值 | 结果 |
| --- | --- | --- | --- | --- |
| global | 106 | 5.82077e-10 | 1.16415e-10 | 全部通过 |
| aabb_4000 | 88 | 1.16415e-10 | 1.16415e-10 | 全部通过 |
| aabb_5000 | 88 | 4.36557e-11 | 4.36557e-11 | 全部通过 |

每个global/AABB fixture还单独执行全部点、全部ratio的直接transport比较；
local全部144组和混合flags/空组/-1/空输入/无group用例均通过。
既有 `change_distill_test.cpp` 在CPU legacy、CUDA Selector+TF两种backend重新编译执行，
涵盖local/global/AABB、outside、anchors、对应关系、split epoch、normal optimizer隔离等；
A、B、C各阶段均通过，cache扩展测试也通过。
本轮没有运行全量SLAM/轨迹质量评估；验收范围是冻结真实event和已有生命周期fixtures，不能推断长期轨迹质量收益。

只接受了有完整分支收益且通过正确性的阶段。失败的向量化源码、初始结果和comparison仍保存在`stage_a/`；
没有保留一段仅因microbenchmark快而未通过最终参数容差的代码。

## 8. Stage D 与下一步

本轮仅检查，不实现输入梯度裁剪：

| 边界 | 真实CD中的无梯度输入 | 仍然必须计算的内容 | 结论 |
| --- | --- | --- | --- |
| Selector NetworkFunction | Gaussian XYZ/rotation/scale和ratio | 两层之间的dh、所有embedding及最终层参数梯度 | 仅最外层输入dx理论可省；当前不改 |
| TF QueryFunction | xyz、ratio、AABB | 每个plane的插值权重、plane参数atomicAdd | queryGradKernel把坐标导数与plane梯度混合；需单独隔离验证 |
| TF OutputFunction | base xyz/log-scale/quaternion | residual/head梯度，含quaternion归一化链 | 可考虑跳过base输出写入，不能切断head梯度 |
| TF NeuralFunction | sampled feature需要梯度 | feature梯度用于训练HexPlane | 不可detach feature |

`ctx->needs_input_grad`或等价mask有潜力，但当前profile不能隔离这些可省子计算的完整event收益。
Selector已有backward kernel数在B/C相同；TF query/output全部kernel时间还包含必需参数梯度。
因此未花费较大工作量重写这些CUDA kernel，也没有将它们的总耗时假定为可节省量。

下一项最高价值工作是继续降低TF transport的逐组算式/整批index_copy成本，
先定位p22容差失败对应的XYZ累加差异，并设计保留原数值语义的融合或grouped transport方案。
保留的A仍有约3万个geometry kernel、显著CPU wall，不能认为这一瓶颈已彻底解决。
第二候选是Selector shared embedding backward，但必须重新验证梯度累加顺序及完整分支收益。

## 9. 文件与复现

本地证据目录：`tests/diagnostics/change_distill_optimization_20261017/`。
服务器镜像：`/home/prmi/zwj/mycartgs/results/change_distill_optimization/20261017`。
第1–9节记录优化验证阶段，当时服务器只在独立目录构建/运行；正式部署结果见第10节。

| 文件/函数 | 改动 |
| --- | --- |
| src/change_distill.cpp::TargetTransport / transportTarget | event-local membership table，按原算式transport |
| include/frozen_teacher_target_cache.h::get / getCached | 已证明coverage的直接cache gather |
| src/change_distill.cpp::execute | 显式传递coverage；同batch跨ratio复用probabilities |
| improvements/selector/gumbel_network.h::forwardProbabilities | CD多ratio入口，legacy回退 |
| include/selector_cuda.h / src/selector_cuda.cu::spatialForward / forward | 复用六个空间embedding activation，保持strict final GEMM |
| src/selector_cuda.cpp::NetworkFunction / networkMany | activation共享，原参数边和逐ratio backward保留 |
| tests/frozen_teacher_target_cache_test.cpp | direct/lazy/partial/repeated-ID覆盖 |

```bash
cd /home/prmi/zwj/mycartgs/results/change_distill_optimization/20261017
PY=/home/prmi/anaconda3/envs/zwj/bin/python
$PY stage_c/build_native.py
$PY build_verify.py stage_c
$PY run_cases.py stage_c:selector:new_selector stage_c:tf:new_tf stage_c:both:new_both
$PY compare.py baseline stage_c selector
$PY build_profile.py stage_c
$PY run_cases.py stage_c:selector:new_profile:1:1
```

原生底层命令：

```text
stage_c/benchmark_native EVENT OUTPUT BRANCH 5 8192 4 10 0 2
```

EVENT固定为原`events_supplement/local_loop/event_6207_1`。
build_commands JSON记录完整编译/链接命令；`cuda_objects.json`验证未变CUDA source的object复用，
改变的Selector CUDA object从源码重新编译，未错误复用baseline object。
Stage C最后仅清理helper未使用的参数签名；初始源码在`stage_c.tar.gz`，最终源码在`stage_c/source`，
主表正反顺序复测使用最终原生构建，最后也重建并执行完整事件correctness runner。

`profiling_outputs.tar.gz`保留全部新raw CSV/JSON、model/target/gradient PT、日志、Nsight原始输出、
各阶段源码、完整diff和构建记录。`artifact_manifest.json`记录归档哈希。
`timings.csv/json`、`paired_summary.json`、`components.csv/json`、`nsys_components.csv/json`可直接继续分析。
生产正常路径未添加逐操作synchronize；profiler hooks只存在隔离副本。

最终建议：保留A保守方案、B和C；逐点BMM已撤销；D不启用。
部署或其他机器比较时按相同输入/完整分支契约重新测量，不沿用本机百分比作为保证。

## 10. 2026-10-18 正式部署

已将验证通过的 A（保守 transport）、B（cache-hit）、C（Selector embedding复用）
更新到 `prmi@172.19.163.58:/home/prmi/zwj/mycartgs`，并完成正式 CMake Release 构建。
本地源码已在上一阶段完成优化；本次同步的是同一份实现。

全量构建最初在历史profiling目标上发现过期的生成文件：旧mapper副本的两参数接口与
当前头文件三参数接口不一致。通过现有CMake配置流程重新生成build目录中的诊断副本后，
增量构建成功。未修改生产接口或CMake源码；初次失败和成功重建分别保留在
`build_initial.log` / `build_initial.exit`与`build_refreshed.log` / `build_refreshed.exit`。

- 只同步6个生产文件和1个cache测试，逐文件哈希/标准化文本核对通过。
- 其他26个审计文件及生产YAML保持原值；独立selector_epochs/tf_epochs配置保留。
- 使用现有 `build-cuda118-sm86-gcc9`，CUDA11.8/sm86、LibTorch2.3.1/ABI1、正式`--fmad=false`编译参数。
- 正式共享库 `lib/libgaussian_mapper.so` 已改变哈希，确认导出新的 `selector_cuda::networkMany`。
- `bin/tum_rgbd`、`tum_mono`、`replica_rgbd`、`replica_mono`、`vector_mono`、`vector_stereo`
  的依赖检查无缺失库；主要入口提供必需的--selector-config后，可正确显示Usage。
- ChangeDistill CPU/CUDA、TF、Selector、cache与相关回归均通过；完整CTest结果见下文。
- 没有提交commit，没有重新运行SLAM，也没有改动其他任务的独立runtime。

完整CTest：**61/62通过**。`ultrafast_mapping_cuda`在已有断言
`Mode1 tail advanced clocks without Gaussian/TF optimizer updates`处失败。
同一新测试程序分别加载备份旧库、新库、再次加载新库，均在同一断言失败；
更新前备份的原测试程序+原共享库也复现相同失败。因此这是更新前已存在的问题，
没有把失败隐藏为通过，也没有为让测试变绿修改Mapping、TF loss或原有测试。
四组原始日志与返回码保留在`tail_counterfactual*.log` / `tail_counterfactual.json`。
本次ChangeDistill优化相关测试和数值验收均通过；上述Mapping tail问题保留为独立已知问题。

正式共享库另通过762项跨ratio输出、loss、输入/参数梯度与单步更新比较；全部逐位一致。

部署期间服务器有独立TUM实验，未将并发GPU负载下的耗时作为性能证据。
更新依据为此前同一LC6207、完整分支/事件的空闲GPU复测：完整事件759.142→619.347ms（1.226×）。
本次正式构建侧验证的是编译、数值正确性、回归与实际动态库加载；未把既有隔离runner的数值冒充为正式库新测值。


更新前源码、bin/lib及关键构建产物已备份：

`/home/prmi/zwj/mycartgs/results/change_distill_optimization/20261017/deployment_20261018/backup`

当前正式mapper库SHA-256：`b98517fb7665c1871cb162c254731f43b26c3c98e2a78e72a1aecd84f2ecce19`。

运行中的 `resultsall/batch_20261018_tum_aligned/runtime` 使用自己的冻结库，保持其原版本。
新启动 `/home/prmi/zwj/mycartgs/bin/*` 会通过正式库使用本次优化。

本目录保留 `source_before.json`、`source_after.json`、`deployment_manifest.json`、
`build.log`、`ctest.log`、退出码、共享库校验和启动日志；后续可按相同快照复测正式库：

```bash
cd /home/prmi/zwj/mycartgs/results/change_distill_optimization/20261017/deployment_20261018
/home/prmi/anaconda3/envs/zwj/bin/python run_benchmark.py baseline
/home/prmi/anaconda3/envs/zwj/bin/python run_benchmark.py optimized
```

baseline明确从`backup/lib`加载旧库，optimized从正式`mycartgs/lib`加载新库；已有输出不会被覆盖。
