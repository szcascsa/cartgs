# ChangeDistill 性能源码审计与 Profiling

审计日期：2026-10-17（Asia/Hong_Kong）。对象：当前本地 `mycartgs` 冻结实现；实测机器：`prmi@172.19.163.58`，RTX 3090。

**本轮没有优化生产代码。** 复用真实 Local LC 6207，分别执行 Selector／TF 的1、2、5、10、20 epochs、5种batch、1/2/4 ratios实验；每个正式性能case包含cold、2次独立warmup、10次从相同输入恢复的重复测量。所有训练loss与正式诊断评估保留。

主要结论：TF 的次线性 epoch scaling 有明确的固定开销来源：**冻结Teacher结果的几何 transport**。当前模型的TF总时间拟合为 `T(E)≈398.82+E×31.37 ms`；Selector为 `T(E)≈28.52+E×39.72 ms`。固定项是回归估计，组件实测和kernel证据见后文。

![Epoch、batch与ratio scaling](tests/diagnostics/change_distill_performance_20261017/plots/scaling_summary.png)

| 主benchmark，B8192/R4/E5 | CPU wall中位数/ms | CUDA span中位数/ms | 实际steps | 10次std/ms |
| --- | --- | --- | --- | --- |
| selector | 225.970 | 225.962 | 20 | 0.360 |
| tf | 556.474 | 556.466 | 20 | 1.535 |

CUDA span 包含stream空闲，不能解读为SM计算时间。绝对时间不同于上一轮报告不代表本轮做了优化；本轮重跑旧runner用于检查测量边界，历史机器状态的剩余差异明确保留。

## 审计边界与版本

本轮遵循 **OBSERVE FIRST / OPTIMIZE LATER**。生产 `src/`、`include/`、配置、loss、精度、训练点集和网络结构未由本轮修改。新增文件全部位于本审计目录，及根目录最终报告。独立 benchmark 的 branch、batch、ratio、epoch 变化只用于指定的 scaling 实验。

主要事件是上一轮保存的真实 `events_supplement/local_loop/event_6207_1`，没有重新运行 SLAM。输入为完整 85,653 个 Gaussian，成熟点 44,889，corrected mature 29,687，未修正成熟点 15,202，TF 域内 29,683、域外跳过 4，correction groups **144**。事件无 AABB 扩张。来源为真实 Loop Closure 的补充捕获运行；其 mapper 阈值为 rotation=0.1、translation=0.01，不把它表述为原宽阈值必然触发的事件。

本地工作树已含用户的独立 epoch 改动。`ChangeDistill.selector_epochs`、`tf_epochs` 未设置时分别回退到 `ChangeDistill.epochs`。本轮两份正式 YAML 都未显式设置这两个覆盖项。`raw/current_configs.json` 保存实际读取值，`raw/source_manifest.json` 保存冻结文件 SHA-256；`raw/working_tree_before.diff` 是任务开始时已有改动。

与上一轮保存的源码按统一换行比较，网络、cache、transform、CUDA 实现均一致；相关差异集中在 CD epoch 接口及 mapper 读写/集成。逐文件结果和 CD diff 见 `raw/previous_source_comparison.json`、`raw/previous_vs_current_cd.diff`。原始字节哈希因 Windows CRLF 与 Linux LF 不同，不能直接用字节差异推断算法发生变化。

| 配置/事件 | Selector backend | TF backend | TF C / levels / hidden | batch | Selector epochs | TF epochs |
|---|---|---|---|---:|---:|---:|
| 本地 fast.yaml | cuda | cuda | 8 / [1,2] / 32 | 8192 | 5 | 5 |
| 本地 selector_config.yaml | cuda | legacy | 16 / [1,2,4] / 64 | 8192 | 1 | 1 |
| 主 benchmark：LC 6207 | cuda | cuda | 8 / [1,2] / 32 | scaling 指定值 | 独立测试 | 独立测试 |

因此，本报告的实测数字不代表默认 YAML 中更大 legacy TF 的性能。两条计算路径均做静态审计，但未将不同网络的时间混合。

## 1. 从事件到 optimizer 的实际调用链

行号对应 `source/` 内冻结的本地文件；报告根目录中的源码引用如发生后续编辑，应以此冻结副本为准。CPU/GPU 表示实际工作归属，所有 LibTorch/CUDA 操作均由 CPU 发起。

### 1.1 Local Loop Closure

```mermaid
flowchart TD
  A[combineMappingOperations: LoopClosingBA] --> B[captureGeometryChange: LOCAL_GEOMETRY]
  B --> C[ChangeCoordinator.captureBeforeGeometryChange]
  C --> D[保存 Gaussian / mature / AABB 快照并 cloneFrozen Teacher]
  D --> E[逐关键帧 scaledTransformVisiblePointsOfKeyframe]
  E --> F[addCorrection: 保存真实修正分组]
  F --> G[finishGeometryChange]
  G --> H[ChangeCoordinator.finalizeGeometryChange / buildEvent]
  H --> I[Selector: primary + unaffected anchors]
  I --> J[TF: corrected mature + outside filter]
  J --> K[返回 report，累计 CD steps]
```

### 1.2 AABB-only

```mermaid
flowchart TD
  A[evaluateDynamicAABB: 周期/强制检查] --> B[makeExpansionPlan]
  B --> C[captureGeometryChange: FIELD_DOMAIN]
  C --> D[旧 Gaussian / mature / AABB + 冻结 TF Teacher]
  D --> E[finishGeometryChange / finalizeGeometryChange]
  E --> F[buildEvent: Selector NONE，TF ALL_SNAPSHOT_MATURE]
  F --> G[resampleDomain + setAABB]
  G --> H[清除受影响 TF 参数的 normal/shared Adam 状态]
  H --> I[仅 TF Distill]
  I --> J[返回 report]
```

### 1.3 Global geometry

```mermaid
flowchart TD
  A[combineMappingOperations: ScaleRefinement] --> B[captureGeometryChange: GLOBAL_GEOMETRY]
  B --> C[快照 + 冻结 Selector/TF Teacher]
  C --> D[applyScaledTransformation + scene transform]
  D --> E[addCorrection: 全局 snapshot 范围]
  E --> F[finishGeometryChange / buildEvent]
  F --> G{AABB 是否扩大}
  G -->|是| H[先重采样空间域并清除相应 Adam 状态]
  G -->|否| I[Selector ALL_SNAPSHOT_MATURE]
  H --> I
  I --> J[TF ALL_SNAPSHOT_MATURE]
  J --> K[返回 report]
```

普通 LocalMappingBA 不是此处的事件入口。仅有 “Loop Closure Detected” 日志不足以触发 CD；需至少一个真实 correction group 或 AABB 变化。允许 append；新增点不属于 `[0,N0)` 快照；prune/reset/reorder 被 correspondence version 断言拒绝。

| 文件 / 函数 / 行号 | 输入 → 输出 | CPU/GPU | 新分配 | 同步风险 | 循环及次数 |
|---|---|---|---|---|---|
| src/gaussian_mapper.cpp / combineMappingOperations / 3345 | MappingOperation、关键帧 pose → Gaussian corrections | CPU pose + GPU 属性变换 | mask、修正 indices、变换结果 | pose/阈值部分 CPU；CUDA→标量见具体实现 | 每次 operation；Local 按关联 KF |
| src/gaussian_mapper.cpp / captureGeometryChange / 2368 | XYZ[N,3]、scale[N,3]、q[N,4]、maturity[N] → ChangeCapture | CPU 编排 + GPU | mature mask、属性 clone、Teacher | mature_mask.any().item；TF clone 的 AABB .cpu | EVENT-ONCE；Teacher 按参数循环 |
| src/change_management.cpp / captureBeforeGeometryChange / 170 | mature mask、GaussianAttrState、AABB、版本 → old_state、mature_indices | GPU | N×10 FP32、N bool、M int64、AABB | nonzero 动态输出；一致性检查部分同步 | EVENT-ONCE |
| improvements/selector/gumbel_network.h / cloneFrozen / 113 | 18 个参数 → 冻结网络 | CPU 初始化 + GPU copy | 完整 Teacher 参数 | CPU→GPU 参数初始化/copy | EVENT-ONCE，18 参数 tensor |
| src/transform_field.cpp / cloneFrozen / 141 | 26/32 个参数及 AABB → 冻结 TF | CPU 初始化 + GPU copy | planes、MLP、buffers | aabbMin/Max.cpu，构造检查 .item | EVENT-ONCE；fast 为 26 参数 tensor |
| src/change_management.cpp / addCorrection / 193 | indices[K]、matrix[4,4]、quaternion[4] → CorrectionGroup | CPU 容器 + GPU | indices/变换 clone | indices>=0 的 .all().item | 每有效修正组；本事件 144 |
| src/gaussian_mapper.cpp / finishGeometryChange / 2413 | capture + current Gaussian → context、report | CPU + GPU | AABB 检查/计划 | render mutex；域检查可能 .item | EVENT-ONCE |
| src/change_management.cpp / buildEvent / 226 | groups + mature + config → correction_group_id[N]、scope | CPU + GPU | full[N] int64、索引/排序临时量 | 每组重叠/重复检查 .item | 144 组；AABB 可附带扩张 |
| src/change_management.cpp / finalizeGeometryChange / 285 | event/capture/context → CD report | CPU | 对象、计时范围 | 无快照/无变化时提前返回 | EVENT-ONCE |
| src/hexplane_field.cpp / resampleDomain / 98 | 新旧 AABB、affected planes → 原 Parameter 原位更新 | GPU | linspace、grid、sample、initialized、where | AABB 检查与 equal；不换 Parameter 身份 | 仅域变化；levels×affected planes |
| src/change_management.cpp / finalizeGeometryChange / 322 | affected parameters → normal/shared Adam state 清除 | CPU 容器 | state 释放 | allocator 行为取决于 stream | 每受影响参数；仅 AABB |
| src/change_distill.cpp / execute / 136 | event/capture/context → 两分支报告 | CPU + GPU | 详见下表 | 多处 .item/nonzero | Selector 全部 epochs 完成后才执行 TF |
| src/gaussian_mapper.cpp / finishGeometryChange / 2462 左右 | report → CD step 累计、change_version | CPU | 少量对象 | DSD 若启用会刷新引用；默认关闭 | EVENT-ONCE |

### 1.4 Selector：训练与评估并非同一 batch 逻辑

文件为 `src/change_distill.cpp`，网络入口为 `improvements/selector/gumbel_network.h:96`。

| 步骤 / 函数 / 行号 | 输入 → 输出 tensor | 设备及分配 | 同步 | 频率与本事件次数 |
|---|---|---|---|---|
| scopeIndices / 31、execute 148 | mature[N] & group_id>=0 → primary[P] | GPU mask + nonzero，新 tensor | nonzero 动态形状同步 | EVENT-ONCE，P=29687 |
| unaffected / 169 左右 | mature & group_id==-1 → U[15202] | GPU mask + indices | nonzero | EVENT-ONCE |
| sampleAnchors / 125 | U、P → anchors[A] | CPU shuffle；GPU permutation + gather | .to(device) 默认阻塞 | 初始一次，之后每 epoch 一次；总 E 次 |
| FrozenTeacherTargetCache 构造 | N → ready[N] bool，各 ratio 一个 | GPU zeros | 通常异步 | EVENT-ONCE ×R |
| evaluate / 176–190 | primary/固定 audit anchors → CPU mean loss | 普通 slice batches；R 循环 | 每 batch×ratio 的 .item<double> | pre 一次，每 epoch 后一次 |
| selectorLoss / 64 | current.select(indices) → XYZ[B,3]、scale[B,3]、q[B,4] | 每调用 3 个 GPU gather；scale.exp 再分配 | gather 通常异步 | 每 batch、每 ratio、primary/anchor 各一次 |
| ratio creation / 69 | CPU float ratio → GPU full[B,1] | 新 FP32 tensor | kernel fill，无显式 H2D tensor copy | 每 selectorLoss 一次 |
| cache.get / header 15 | indices → missing、Teacher probability[B] | ready gather、布尔筛选、target gather | 即使 cache hit 仍 nonzero/动态索引 | 所有训练/评估调用 |
| teacher probability / 73 | old XYZ/q/exp(scale)、ratio → p_teacher[B] | 冻结网络、NoGrad、GPU | 无显式 .item；cache miss 会动态筛选 | 本事件总 24 次网络调用，之后命中 |
| student forwardProbability / 76 | corrected XYZ/q/exp(scale)、ratio → p_student[B] | CUDA network + ATen softmax | 通常异步 | E×32 次训练 + (E+1)×24 次评估 |
| L1 / 78 | p_student[B]−p_teacher[B] → scalar | GPU sub、abs、mean，新 tensor | 无直接同步 | 每 selectorLoss 一次 |
| stratumBatch / 83、222 | shuffled primary/anchors → 分层 views + CPU weight | slice 为 view，不复制数据 | 无 | 每训练 batch 两次 |
| ratio 累积 / 225 | 4 ratio 的主项/anchor loss → total scalar | 多个标量加乘，形成单一 graph | 无 | 每训练 batch；R=4 |
| finite / positive check / 234 | total → CPU bool/double | GPU reduction + D2H scalar | **每 batch 两次显式 .item** | 2×4E |
| backward / 236 | 单一 total → 18 参数 gradients | CUDA custom backward + ATen autograd | host 调度；不会为每 ratio 单独调用 backward | 4E 次，总图含每个 ratio 和 anchors |
| Adam / 238 | 所有参数/grad → 更新参数及 m/v | CUDA backend 调 tfCudaAdamDirect；legacy 调 Adam::step | CPU options/state + GPU kernel | 4E；首步分配 state |
| zero_grad / 221、250 | grad handles → 清理 | LibTorch 默认语义，未改变 | 可能涉及 allocator；不是手写同步 | 4E+1 |

训练 anchors 的数量为 `min(floor(P*ratio), U, max_count)`，本事件 `min(29687,15202,32768)=15202`。因此每次抽样遍历全部未修正成熟点，只改变顺序；固定的第一组 audit anchors 用于 epoch 间可比评估。不能把 anchors 当成单独的 optimizer step。

### 1.5 TF：Teacher、transport、cache、Student

| 步骤 / 文件函数行号 | 输入 → 输出 tensor | 设备及分配 | 同步 | 频率 |
|---|---|---|---|---|
| scopeIndices / change_distill.cpp:149 | corrected mature → ids[29687] | GPU mask/nonzero | 动态输出 | EVENT-ONCE |
| outside_identity / 272 | current XYZ[29687,3] → inside mask → ids[29683] | gather、比较、all、布尔筛选 | nonzero/布尔索引动态形状 | EVENT-ONCE；后续网络内部还会重复检查 |
| old AABB diagonal / 279 | old extent[3] → norm scalar → double | GPU sub/norm | `.item<double>()` D2H | EVENT-ONCE |
| cache 初始化 / 281 | N → R 个 ready[N]，首次 miss 后 values[N,10] | GPU | 通常异步，miss indices 有同步 | EVENT-ONCE ×R |
| teacher applyLogResidual / 287 | old_state.select(missing)、ratio → 旧弹性 XYZ/log-scale/q | GPU insideMask/nonzero + gather + TF forward + output + index_copy | 网络内部 nonzero | cache miss；本事件 20 次 |
| transportTarget / 39 | 旧弹性结果、missing ids、group_id、144 groups → 新坐标系 target | GPU ids gather；每组 nonzero；非空组 transform 与 3 次 index_copy | **每组 nonzero；组内若为空仍需判定形状** | 20 次，累计 20×144=2880 组检查 |
| target pack / 289 | target xyz[B,3]/scale[B,3]/q[B,4] → packed[B,10] | GPU cat；cache index_copy_ 写入 | 通常异步 | 20 次 miss，已 transport 后才缓存 |
| target gather / cache header | ids[B] → packed[B,10] → 3 个 slice view | gather 新 tensor；slices 非 contiguous | missing 筛选仍存在 | 每一次 errors() |
| student applyLogResidual / 293 | current.select(batch)、ratio → student attributes | 每 ratio 重复 gather、insideMask、nonzero、网络、index_copy | nonzero | 训练 16E 次，评估 16(E+1)+8 次 |
| tfErrors / 55 | student、target → 三个 scalar | normalize(qs/qt)、XYZ abs mean/diagonal、scale abs mean、1−abs(dot) mean | 构造不提取 scalar | 每 errors()，训练保留 graph |
| evaluate / 295–309 | ids、R → CPU 3-component means | 每 batch×ratio 构造全部三个 loss | **每次三个 .item<double>()** | primary pre+每 epoch post；untouched pre/post |
| audit / 331 | unaffected mature → inside → shuffle → 最多 B 点 | GPU filter、CPU shuffle、H2D permutation | nonzero/.to | EVENT-ONCE；8192 点，仅诊断不训练 |
| shuffle / 345 | tf_indices → shuffled indices | CPU std::shuffle + H2D + GPU gather | .to 默认阻塞 | EPOCH-ONCE |
| batch / 348 | shuffled indices → slice[B] | view | 无 | ceil(29683/B) / epoch |
| ratio loss 累积 / 350 | R×3 losses → scalar total/R | GPU 标量 graph | 无 | BATCH-ONCE，串行 R forwards |
| finite/positive / 355 | scalar → bool/double | D2H | 每 batch 两次 .item | 2×4E |
| backward / 357 | 一次 total.backward → TF 参数梯度 | OutputFunction、NeuralFunction、QueryFunction、ATen | 详见 operator/kernel 实测 | 4E；不是每 ratio 单独 backward() |
| Adam / 359 | 14 MLP+12 plane 参数/grad → 参数+m/v | CUDA fused Adam；26 参数 tensor | CPU state/options，GPU kernel | 4E；首步建 state |
| zero_grad / 347、369 | 临时 Adam grads → 清理 | LibTorch | 无显式同步 | 4E+1 |

TF 位置误差按旧 AABB 对角线归一化；scale 使用 **log-scale L1**；rotation 为两个归一化 wxyz quaternion 的 `mean(1−abs(dot).clamp(0,1))`。CD 不调用 TF regularizationLoss，不包含渲染 loss、maturity weighting 或额外 TF anchors training。未修正集的 pre/post evaluation 属于正式现有路径，本轮完整保留。

## 2. epochs、batch、ratio 与实际工作量

训练 ratios 为 `[0.01,0.05,0.10,0.15]`；CD 内部 pre/post/audit evaluation 使用完全相同 ratios。上一轮外部质量评估另含 `0.025`，它不属于 CD 训练或本轮计时路径。

设 `P=29687`、`A=15202`、`F=29683`、`B=8192`、`R=4`：

```text
Selector batches/epoch = ceil(max(P,A)/B) = 4
Selector primary batch sizes = [7421,7422,7422,7422]
Selector anchor batch sizes  = [3800,3801,3800,3801]
Selector optimizer steps/epoch = 4（非零 loss 时；runner 对实际计数做断言）

TF batches/epoch = ceil(F/B) = 4
TF batch sizes = [8192,8192,8192,5107]
TF optimizer steps/epoch = 4
```

**一个 optimizer step 包含所有 ratio 的串行 forward/loss 累积，然后一次 total.backward 和一次 Adam。** 每个 ratio 具有独立网络 graph；一次全图 backward 会分别执行这些网络的 backward 节点。把 optimizer step 数乘 R 是错误的。

| 单分支 epochs E | 1 | 2 | 5 | 10 | 20 |
|---|---:|---:|---:|---:|---:|
| Selector optimizer steps | 4 | 8 | 20 | 40 | 80 |
| TF optimizer steps | 4 | 8 | 20 | 40 | 80 |
| Selector 训练 Student forward calls | 32 | 64 | 160 | 320 | 640 |
| Selector 全部 Student forward calls（含正式审计） | 80 | 136 | 304 | 584 | 1144 |
| TF 训练 Student forward calls | 16 | 32 | 80 | 160 | 320 |
| TF 全部 Student forward calls（含正式审计） | 56 | 88 | 184 | 344 | 664 |

Selector 每个 ratio：训练 forward `8E`、训练网络 backward `8E`、评估 forward `6(E+1)`；Teacher 总计 6 次/ratio。TF 每个 ratio：训练 forward/backward 各 `4E`、评估 forward `4(E+1)+2`；Teacher 总计 5 次/ratio。Teacher 的缓存命中不代表 get() 免费，仍有 mask/index/target gather。

不同 microbenchmark batch 的 optimizer steps 会按生产公式变化；这是 batch scaling 指定的实验变量，不宣称跨 batch 具有同一训练轨迹。只有固定 batch/ratio/epochs 的重复运行要求 step 数完全相同。训练样本总暴露为 Selector `(P+A)*E*R`、TF `F*E*R`；表中 samples/s 使用此分子，并注明分母包含正式 pre/post 审计和固定成本。

## 3. 网络结构、shape、FLOPs 与布局

### 3.1 Selector forwardProbability

实际路径为 `forwardProbability → logitsFor → selector_cuda::network(CublasStrict, DirectFinalRelu) → logits/tau → softmax(-1) → select(1,1)`。CD 输入已归一化 quaternion 和 `exp(log_scale)`；本路径没有位置 positional encoding、maturity feature、batch normalization、dropout 或 Gumbel 随机采样。

| 输入/阶段 | shape（典型 B=8192） | 层/activation | FP32 大小/布局 |
|---|---|---|---|
| position | [8192,3] | 原始 world XYZ | contiguous，stride [3,1] |
| rotation | [8192,4] | normalized wxyz | contiguous，stride [4,1] |
| scale | [8192,3] | exp(log_scale) | contiguous，stride [3,1] |
| ratio | [8192,1] | full 的原始 ratio；CD 不是 [1,1] 快捷输入 | contiguous，stride [1,1] |
| 四个独立 embedding 第一层 | 各 [8192,32] | Linear(d,32)+ReLU；d=3/4/3/1 | 每项 1 MiB |
| 四个 embedding 第二层 | 各 [8192,32] | Linear(32,32)+ReLU | 每项 1 MiB |
| packed feature | [8192,128] | packFeatures；legacy 为 cat | 4 MiB，stride [128,1] |
| logits | [8192,2] | Linear(128,2) | 64 KiB，stride [2,1] |
| probability | [8192] | softmax 后 select 第1类 | 通常 stride [2]，非 contiguous |

参数总数 **4,962**，18 个参数 tensor，约 19.38 KiB。单样本 Linear forward FLOPs 约 `2*(11*32+4*32*32+128*2)=9408`，B=8192 时约 **77.07 MFLOPs**，不含 bias、ReLU、softmax 与 tensor 操作。CUDA forward 会分配八个 B×32 activations、B×128 packed、B×2 logits，合计约 **12.06 MiB**；这不是峰值显存，因为 backward 保存、softmax、多个 ratio/strata 的图还会叠加。

当前 C++ custom autograd 保存第一层 activations，以及 packed feature 的第二层 view，避免部分重复保存；第二层 view 为 stride [128,1]。CUDA forward 和 backward 的输入都调用 contiguous()；输入已 contiguous 时不复制，不能只按函数名计算实际 copy 次数。

实测Selector probability为stride [2]、pointer mod256=4；主要gather输入与packed workspace为256-byte对齐。TF target log-scale/rotation是stride [10,1]，pointer mod256分别为12/24。这些非连续view确实存在，不把它们当作dense [B,3]/[B,4]传入后续假想kernel。

网络参数很小，但 B=8192 的 forward 不是零计算。是否由 launch/orchestration 主导，必须结合本轮 kernel busy time、CPU wall、tiny kernel、forward/backward 分解判断，不能只从 4,962 参数得出结论。

### 3.2 TF fast 模型

`applyLogResidual → insideMask/nonzero/gather → forward → tf_cuda::query → tf_cuda::neural → tf_cuda::output(log_scale=true) → 三次 index_copy`。

| 阶段 | shape | dtype/device | 布局/依赖 |
|---|---|---|---|
| Gaussian XYZ/log-scale/q | [B,3]/[B,3]/[B,4] | FP32 CUDA | gathered contiguous |
| normalized XYZ | 逻辑 [B,3] | FP32 CUDA | CUDA query kernel 内计算，不单独分配该 tensor |
| ratio | [B,1] | FP32 CUDA | 每次 forward 新 full |
| L0 spatial xy/xz/yz | 各 [1,8,64,64] | FP32 CUDA parameters | contiguous，stride [32768,4096,64,1] |
| L0 ratio xe/ye/ze | 各 [1,8,150,64] | FP32 CUDA parameters | contiguous，stride [76800,9600,64,1] |
| L1 spatial | 各 [1,8,128,128] | FP32 CUDA | stride [131072,16384,128,1] |
| L1 ratio planes | 各 [1,8,150,128] | FP32 CUDA | stride [153600,19200,128,1] |
| 六平面相乘后的两层 feature concatenate | [B,16] | FP32 CUDA | queryKernel 一次写出 contiguous，stride [16,1] |
| feature_out | [B,32] | FP32 CUDA | Linear(16,32)；三 head 共用该输出的 ReLU |
| 三个 hidden | 各 [B,32] | FP32 CUDA | 各 Linear(32,32)+ReLU |
| XYZ/log-scale/rotation raw head | [B,3]/[B,3]/[B,4] | FP32 CUDA | Linear(32,3/3/4) |
| 输出属性 | [B,3]/[B,3]/[B,4] | FP32 CUDA | XYZ 加残差；log-scale 加残差；quaternion compose+normalize |
| packed Teacher target | [B,10] | FP32 CUDA、NoGrad | contiguous stride [10,1] |
| target 的 xyz/scale/q slice | [B,3]/[B,3]/[B,4] | FP32 CUDA、NoGrad | stride [10,1]，非 contiguous view |

HexPlane 参数 **1,182,720**；MLP **4,042**；总计 **1,186,762**，26 参数 tensor（12 planes+14 MLP），约 **4.53 MiB**。MLP Linear forward 约 `2*(16*32+3*32*32+10*32)=7808 FLOPs/sample`，B=8192 时约 **63.96 MFLOPs**。

单次 query 的逻辑读取约 `B*16*6*4` 个 FP32 plane corners，B=8192 时约 12 MiB；这不是 DRAM 流量实测，因为存在 L1/L2 缓存与复用。queryGradKernel 会重新进行插值并对每个角点 atomicAdd；同时生成不一定被上游需要的 XYZ/ratio gradient。不能据此直接宣布 bandwidth-bound 或 atomic-bound。

当前 forward 对 `F=16,H=32,N<=10000` 使用 **Warp32 MLP**；更大的 batch 使用 **cuBLAS**。所以 16384/32768 batch scaling 跨越现有 dispatch 边界，不能把曲线变化全部解释为 batch 占用率。

legacy TF 使用 `HexPlaneField::forward → 六个 grid_sample/level → 逐平面相乘 → cat → LibTorch Linear`。fast 两层为 12 次 grid_sample/forward；默认三层为18次。samplePlane 输出是 transpose view [B,C]，典型 stride [1,B]；最后 cat 的结果为 [B,F]。默认网络 F=48,H=64、总参数约 **7,370,634**，单样本 MLP forward 约32,000 FLOPs。它未被本事件的 CUDA timing 代表。

### 3.3 CUDA kernel 入口

| 模块 | 文件/位置 | 主要 kernel/底层调用 | 是否属于本事件 |
|---|---|---|---|
| Selector forward | src/selector_cuda.cu:235 | cuBLASLt GEMM、ReLU epilogue、packFeatures；ratio 分支 linear | 是，CublasStrict |
| Selector backward | src/selector_cuda.cu:282 | directFinal<true>、reluVjp、sumBias、cuBLAS GEMM | 是，DirectFinalRelu |
| Selector batched weight 候选 | 同文件 batchedGemm | CPU pointer array→GPU，cublasSgemmBatched | 当前 strategy 不走；不能算活跃 H2D |
| TF query | src/transform_field_cuda.cu:87/97 | queryKernel、queryGradKernel | 是；本事件不调用 aten grid_sampler |
| TF MLP | src/transform_field_cuda_neural.cu:53/190/242 | warpMlpKernel；backward cuBLAS、biasGradKernel、reluGradKernel、sharedGradKernel | 是 |
| TF output | 同文件:95/168 | outputKernel、outputGradKernel | 是 |
| Adam | src/transform_field_cuda_optimizer.cu:16 | adamKernel，最多32参数 tensor一组 | 是；Selector18、TF26，通常各一步一个 kernel |
| TF regularization | src/transform_field_cuda_regularization.cu | regularization kernels | CD 没有调用 |

## 4. 跨 ratio 的数据依赖与重复工作

Selector 三个空间属性 embedding 与 ratio 无关；time embedding 与 ratio 有关，最后拼接后线性输出。当前 CD 给每个 ratio 传入 [B,1] full，导致相同 Gaussian 的 XYZ/q/scale gather、scale.exp、三个 embedding 在四个 ratio 中重复。数学上可将 ratio 维展为 [B,R,D] 或 [B*R,D] 后 batch forward；但简单展平不会自动消除空间 embedding 重算，还会扩大激活图。复用 shared embedding 会改变 autograd 累加次序，strict final GEMM 的 tie/numerical 验收需要保留；本轮不实现。

TF 的 xy/xz/yz 采样与 ratio 无关；xe/ye/ze 采样依赖 ratio，ratio 从 **grid sampling 阶段** 就参与，而非仅在最后 MLP。每层 feature 是六平面逐元素乘积。因此无法保持当前模型而简单替换为 “1×完整 HexPlane sampling + 4×cheap heads”。可调查共享 normalized XYZ、spatial-plane samples/其乘积，再进行四组 ratio-plane sampling 与原 MLP；需正确汇总 spatial plane 的梯度，且全局 geometry/epoch 更新后不能复用过期 features。Student 参数每个 optimizer step 改变，跨 step 缓存已学习 features 不等价。

TF CUDA query 内连 normalized XYZ 也会按输出 feature channel 重算；这属于实际 kernel 数据依赖，不是一个可直接复用的外部 tensor。Teacher 冻结，event 内缓存其最终输出有效；Student features 不可当作 event 常量。

TF 的第四个坐标直接使用原始 ratio（0.01/0.05/0.10/0.15），代码没有再对它做 `2r−1` 映射。空间 XYZ 才按AABB归一化到[-1,1]；不能把两者的encoding混为一谈。

## 5. CPU↔GPU、标量提取与分配审计

完整静态扫描见 `raw/source_sync_allocation_scan.csv`，其中也包含未激活候选；下表仅列活跃 CD 路径。

| 文件/行号 | 操作 | Frequency | Sync/Copy 风险 |
|---|---|---|---|
| change_distill.cpp:116–121 | std::vector + iota + std::shuffle + CPU tensor.to(device) | Selector corrected E次、anchors E次；TF train E次+audit一次 | CPU O(N)；每次 permutation H2D，默认 blocking |
| change_distill.cpp:183 | loss.item<double>() | Selector 24(E+1) 次 | D2H scalar / CUDA同步 |
| change_distill.cpp:234–235 | isfinite(total).item、total.item | Selector 8E次；TF同样8E次 | 每训练 batch 两次同步 |
| change_distill.cpp:279 | old AABB norm.item | EVENT-ONCE | D2H scalar |
| change_distill.cpp:303 | 三项 loss.item | TF 48E+72次 | 频繁同步；评估开销不能当成 backward |
| change_distill.cpp:44 | 每 correction group 的 nonzero | 20×144次 | 动态输出大小需回到 host；cache 固定生成成本 |
| frozen_teacher_target_cache.h:15 | missing=indices.index(~ready.index_select) | 每次 selectorLoss/errors | cache hit 仍动态形状筛选/同步 |
| transform_field.cpp:153 | nonzero(insideMask) | 每次 Teacher/Student TF apply | 本轮外部已过滤，内部仍执行 |
| transform_field.cpp:144 | aabbMin/Max.cpu | EVENT-ONCE / Teacher clone | 两次小 D2H + CPU 构造、H2D 参数初始化 |
| model_snapshot.h:copyFrozenModule | 参数/缓冲 copy_ | EVENT-ONCE，按参数数 | 同设备 D2D；初始化阶段还含 H2D |
| change_management.cpp:241–246 | all().item 的重叠/重复检测 | 每 correction group | EVENT-ONCE；与 execute 内 transport 分开 |
| CUDA .cu 的 data_ptr | 获取 CUDA pointer | kernel/GEMM 调度 | 本身不复制、不提取 tensor 内容到CPU |
| C++ sizes/numel/stride | host 读取 tensor metadata | 所有循环 | 本身通常不发 CUDA 同步；动态结果形成之前可能已同步 |

没有发现 CD 活跃训练 loop 把全部 Gaussian `.cpu()` 后再传回 GPU；实际 H2D 主要是 permutation 与 Teacher 构造，D2H 主要是 scalar 与动态 shape。不得将 `data_ptr` 或 `std::vector<Tensor>` 本身计为设备复制。

| 操作/对象 | shape / 估算量 | 频率 | 固定性与复用信息（仅观察） |
|---|---|---|---|
| GaussianAttrState::select | B×(3+3+4) FP32，即40B bytes | 每 ratio/stratum Student；Teacher miss | batch内跨ratio内容相同，但当前反复 gather |
| scale.exp | [B,3] FP32，12B bytes | Selector 每次 forwardProbability 输入 | Gaussian detached且event内固定；只依赖 row ids |
| ratio full | [B,1] FP32，4B bytes | 每 forward | 每ratio值固定；B可能不同 |
| Selector activations/packed | 8×[B,32]+[B,128]+[B,2] | 每 forward | shape受strata/尾batch影响；训练graph生命周期跨ratio |
| ready maps | 4×[85653] bool =342,612 bytes/branch | EVENT-ONCE | event常量布局；ready内容变化 |
| Selector cache values | 4×[85653] FP32 ≈1.31MiB | 每ratio首次miss分配 | event内只填一次，未用row保持未初始化 |
| TF cache values | 4×[85653,10] FP32 ≈13.07MiB | 每ratio首次miss分配 | transport后的最终target；event内有效 |
| TF packed gather | [B,10] FP32，40B bytes | 每 errors() | cache命中仍复制本batchtarget |
| TF transport | 每非空group的局部gather + transform + 3次 out-of-place index_copy | Teacher miss内144组检查 | 每次 index_copy 拷贝当前整batch目标；未在训练epoch重算 |
| TF query output | [B,16] FP32，64B bytes | 每 TF forward | 当前CUDA直接写contiguous |
| TF query backward plane grads | 共1,182,720 FP32 ≈4.51MiB zeros | 每个 ratio 的 Query backward | 4ratio×4batch=16份/epoch，约72.19MiB清零流量，未计allocator/grad累加 |
| TF query backward dx/dr | [B,3]及[B,1] | 每个 Query backward | custom函数即使Gaussian/ratio不需梯度也计算；详见autograd审计 |
| 独立 Adam m/v | 参数大小×2 | 每事件首个step分配 | 每次事件均全新状态；后续step复用 |

这里估算的是逻辑分配/写入量，不是 DRAM counter，也不是峰值同时驻留。allocator 的 memory trace 与 peak allocated/reserved 另外保留。

代表epoch的Kineto计数：Selector `index_select=618、nonzero=82、cat=0`；TF `index_select=4528、nonzero=3016、cat=1268`。两个分支均没有记录到实际materializing `aten::contiguous`；源码中的contiguous调用在这些输入上通常返回原tensor。各有3次clone来自harness的Gaussian不变性检查，位于计时范围外，但raw Kineto trace仍保留；不能把这3次当成CD循环分配。其D2D copies也不应作为生产数据传输热点。原始计数见 `raw/layout_operator_counts.json`。

## 6. Teacher cache 与 autograd

两个分支都使用 `FrozenTeacherTargetCache`：每 ratio 一个 cache，ready[N] bool，按首次出现的 Gaussian row 惰性生成并写入 GPU values。NoGradGuard 内计算并 detach。Selector values 为 [N] probability；TF values 为 [N,10]，存储 **Teacher TF evaluation + geometry transport 之后**的 xyz/log-scale/q。

LC 6207 的 Selector primary 与 anchors 在第一次 evaluate 中已全部覆盖，因此 24 次非空 miss 发生在 baseline。TF 的 primary baseline 为4 batch×4ratio=16次 miss，untouched audit为1 batch×4ratio=4次 miss，合计20次。后续 epoch 的 cache lookup 不重算 Teacher。固定 audit anchors 与 TF audit 集在本事件保持不变；Selector后续训练 anchors重新抽样，但本事件所有未修正成熟点已全被缓存。

不能把 “cache event-local” 理解为只调用一次 cache.get：Selector E=5 有304次 get，TF有184次；每次仍做 ready gather、missing筛选、target gather。TF transport的144组循环只发生在20次miss内，但即便audit点全部未修正，也仍检查全部144组。

缓存正确避免了Teacher网络/transport逐epoch重算，但仍有固定内容的重复处理：`tfErrors`每次重新normalize已冻结target quaternion；Selector每次重新gather/exp固定Gaussian log-scale；TF每次forward重新检查外部已过滤点的inside mask；target cache hit仍形成missing indices。它们具有event内可复用的数据依赖，但本轮没有改变缓存内容或执行位置。

| 对象 | requires_grad / graph 归属 | 检查结论 |
|---|---|---|
| old/current Gaussian xyz、log-scale、q | false，grad_fn none | capture clone、context.detach；runner断言值未变 |
| indices/masks | false，整数/bool | 不加入可微图 |
| AABB buffers、geometry transforms | false | clone/detach；transport NoGrad |
| Teacher parameters与target | false | cloneFrozen + cache NoGrad；不保留Teacher训练图 |
| Student网络参数 | true | CD只更新这些参数 |
| Student probability/residual/output | 训练时true，pre/post评估时false | metadata按 GradMode 分别保存，避免把第一次NoGrad评估误当训练 |
| CUDA custom forward内部 scratch | 常显示requires_grad=false | custom autograd手工save/backward；不能据此判定训练无梯度 |

还存在“输入不需要梯度，但 custom backward 仍计算”的工作：Selector `selector_cuda::backward` 无条件产生四组输入梯度；TF queryBackward 无条件分配/计算 XYZ 与 ratio 梯度；outputBackward 也分配输入属性梯度。Autograd 最终可丢弃不需要的返回值，但 kernel 内这些计算已发生。此处是 future optimization 信息，未改变返回梯度或数值路径。

## 7. Adam、Graph/Fusion 与精度边界

临时 Adam 仅复制 normal 参数组 options/参数引用，不复制 normal state；CD固定学习率，weight_decay=0。Selector18参数tensor/4,962元素；TF26参数tensor/1,186,762元素。首步为每个有gradient的参数创建 m/v，state entry分别18/26，step保存在CPU state；m/v tensor数分别36/52。零梯度清理与optimizer.step的实测时间分开报告。

| 方向 | Selector | TF |
|---|---|---|
| CUDA Graph compatibility | 当前完整loop不兼容直接capture | 当前完整loop不兼容直接capture |
| 固定参数集 | 是，18 tensors | 是，26 tensors；AABB migration仅在训练前 |
| shape障碍 | stratified batch大小7421/7422、anchor3800/3801；其它batch配置更多形状 | [8192,8192,8192,5107]；inside/nonzero仍动态 |
| 控制流障碍 | .item控制finite/positive；cache miss；CPU shuffle；首步Adamstate | 相同；另外144组transport、inside筛选、Teacher cache |
| ratio loop | 固定R，理论上可展开；当前仍串行 | 固定R，grid sampling已ratio-dependent |
| 随机/地址 | CPU std::mt19937 shuffle、动态gather/allocator | 同左；planes grad每次重新分配 |
| torch.compile | 原生LibTorch C++路径不能直接套Python torch.compile；自定义Function需额外集成 | 同左；动态索引与scalar同步亦会分段 |
| 可观察的fusion候选 | gather/exp、ratio feature复用、loss/标量编排 | cache lookup/transport/索引、loss/quat链、query grad编排 |

固定参数集、固定ratio数不等于整个loop可capture；需要先处理动态结果、host scalar控制、随机排列、尾batch、buffer/state地址及stream生命周期。本轮没有CUDA Graph、compile或fusion实现。

数值 tensor 当前为FP32；indices为int64、mask为bool。Selector/TF dedicated CUDA backend 显式要求CUDA FP32，不能仅打开autocast就变成FP16/BF16。MLP GEMM理论上具备低精度/TF32 Tensor Core适配空间，但当前strict GEMM/pedantic math与Warp32路径不构成Tensor Core利用证明。TF query插值、atomic梯度累加、quaternion normalize/compose、abs(dot)接近1、position按AABB归一化、TF Adam eps=1e-15均需独立数值验证。BF16/FP16/TF32只是可行性调查项；本轮没有开启。

## 8. 计时口径与重复实验约束

生产 `elapsedMs()` 为 `steady_clock::now-start`，分支 timer没有显式CUDA Event或cudaDeviceSynchronize。大量 `.item()` 和动态索引会使它经常间接等待GPU，但它仍是CPU wall，不是单独GPU时间，且不包含外部Teacher clone、capture或协调器前置工作。

本轮 native benchmark 在测量前清空先前CUDA工作，使用同一当前stream的CUDA Events包围 finalize，结束等待end event，同时记录CPU wall。另测snapshot/Teacher preparation。**CUDA Event elapsed是stream上两事件之间的跨度，包含CPU来不及enqueue导致的GPU空闲；它不等于所有kernel运行时间之和。** 真正kernel busy time从Kineto/Nsight trace汇总，不将嵌套scope的inclusive时间重复相加。

细粒度mode1仅记录开始/结束events，在整个run结束后统一读取；没有在每个组件后加同步。mode2用于Kineto，mode3用于Nsight/NVTX，不与Kineto同时占用CUPTI。插桩会增加host/kernel launch开销，因此总性能曲线取mode0，组件表取mode1并报告其相对扰动。

CPU scope遇到同步时，可能等待的是更早已enqueue的GPU工作。例如cache missing筛选的CPU wall会承担前一网络/backward尚未完成的等待；CUDA begin事件却排在那些工作之后。因此“cache CPU时间大、该scope CUDA span小”不能直接证明cache kernel很慢。反向地，backward的CPU enqueue结束时GPU可能仍在执行。kernel归因表用于区分这些排队效应。

每个case先记录cold CD run，然后2次独立1epoch warmup，最后10次测量。每一次都从同一冻结Teacher克隆完整Student并验证初始参数相等，重新创建normal Adam与全新ChangeDistillManager，重建capture和event-local cache；warmup更新的模型不进入后续计时。Gaussian保持全部N，ratio/loss/epochs按该case完整执行，实际step数断言通过才接受。

这里cold指“新进程的首个CD事件”，不包含文件读取、CUDA context创建、网络反序列化。输入I/O不在计时内。渲染专用图像tensor不进入训练常驻集，因此GPU总驻留量不能与上一轮持有整个render archive的peak直接对照。

显存指标来自C10 caching allocator，区分allocated/reserved与CD additional allocated，不等于nvidia-smi设备总占用。peak allocated包含本harness常驻的输入、Teacher、Student与三份Gaussian校验副本；additional allocated从重放前的allocator基线计算。raw runtime逐项保留，不能把harness总peak当成生产地图总显存。

Selector固定sampler seed=0；独立TF固定为上一轮paired runner使用的 `0 XOR 0x54465f5041495245`，隔离不同Selector epochs对TF RNG的影响。此控制位于benchmark输入，不修改生产随机数流程。TF atomic累加可能存在微小非逐bit确定性；参数差异检查与数值容差单独列出。

服务器GPU的其它实验不被暂停或终止。正式case前要求连续三次没有其它compute进程；运行时每250ms保存进程列表，发现其它compute进程即标记污染并拒绝使用。cold/warm run原始值、stdout、CUDA跨度、显存、step、输入/源码哈希全部保留。

## 9. 后续优化设计需要遵守的约束

后续方案应保持全部Gaussian、四个ratio、原始loss/权重、实际step与epoch、outside_identity、transport定义、unaffected anchors、pre/post审计与独立Adam语义。batch scaling只用于识别瓶颈，不能据其速度曲线直接建议更改正式训练batch；batch改变本身会改变优化轨迹。

本轮可支持按“固定成本 / 每epoch成本 / backward / gather / optimizer”提出局部优化假设，并通过原始trace验证。不能仅凭GPU util瞬时值、网络参数量、理论读写量或两点epoch拟合宣布memory-bound/compute-bound；没有硬件counter的结论明确标为 inconclusive。


## 10. Torch／LibTorch／CUDA／GPU 环境

| 项 | 本次记录 |
| --- | --- |
| GPU | NVIDIA GeForce RTX 3090 |
| CUDA runtime / compiler | 11.8 / nvcc 11.8 |
| Python PyTorch | 2.3.1+cu118 |
| 原生benchmark LibTorch | 2.3.1，项目 third_party/libtorch-cu118/libtorch，C++11 ABI=1 |
| Python wheel ABI | ABI=0；未用此wheel链接原生benchmark |
| 编译 | g++，-O3 -DNDEBUG -std=c++17，CUDA compute_86/sm_86 |
| 驱动 / 完整信息 | 535.230.02；raw/environment.json |
| 精度 | FP32；原生run逐次记录allow_tf32_cublas，未更改 |
| Profiler | Nsight Systems 2022.4.2；Nsight Compute 2022.3.0；LibTorch Kineto |

完整compiler、nvcc、nvidia-smi -q、PyTorch build配置、原生编译/链接参数分别保留在 `raw/environment.json`、`raw/build_commands_all.json`，实际动态链接见 `raw/native_ldd.txt`。Python wheel与项目LibTorch版本号相同但ABI不同；最终可执行文件链接项目LibTorch，避免以Python环境信息代替实际native库。

## 11. Cold-start、steady-state 与 epoch scaling

### selector

| epochs | cold/ms | warm mean | median | std | min | max | CUDA median | steps | ms/step含固定项 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 143.986 | 70.365 | 70.275 | 0.451 | 69.624 | 71.046 | 70.268 | 4 | 17.569 |
| 2 | 192.102 | 108.524 | 108.536 | 0.379 | 108.021 | 109.284 | 108.528 | 8 | 13.567 |
| 5 | 309.057 | 226.078 | 225.970 | 0.360 | 225.560 | 226.722 | 225.962 | 20 | 11.299 |
| 10 | 496.035 | 422.995 | 422.571 | 1.942 | 420.740 | 426.256 | 422.562 | 40 | 10.564 |
| 20 | 907.598 | 823.988 | 824.672 | 6.265 | 817.048 | 831.225 | 824.660 | 80 | 10.308 |

最小二乘 `T(E)=fixed+E×per_epoch`：fixed=28.517 ms，per_epoch=39.722 ms，R²=0.999951。所有重复原始值与残差保留在 all_runs.csv、epoch_fit.json；不将两个端点的拟合当作独立因果证据。

### tf

| epochs | cold/ms | warm mean | median | std | min | max | CUDA median | steps | ms/step含固定项 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 531.841 | 438.076 | 438.173 | 0.976 | 436.749 | 439.517 | 438.164 | 4 | 109.543 |
| 2 | 554.594 | 459.720 | 459.688 | 1.402 | 457.919 | 463.046 | 459.680 | 8 | 57.461 |
| 5 | 647.461 | 556.578 | 556.474 | 1.535 | 554.400 | 559.471 | 556.466 | 20 | 27.824 |
| 10 | 795.621 | 699.847 | 699.601 | 0.769 | 698.875 | 700.984 | 699.591 | 40 | 17.490 |
| 20 | 1129.913 | 1032.066 | 1032.343 | 1.553 | 1029.412 | 1034.687 | 1032.332 | 80 | 12.904 |

最小二乘 `T(E)=fixed+E×per_epoch`：fixed=398.824 ms，per_epoch=31.373 ms，R²=0.998854。所有重复原始值与残差保留在 all_runs.csv、epoch_fit.json；不将两个端点的拟合当作独立因果证据。

## 12. Why does TF timing scale sub-linearly with epochs?

上一轮5→20 epochs的TF为611.684→1091.39 ms，两点直线等价于约451.782 ms固定项＋31.980 ms/epoch；Selector为503.643→1914.26 ms，等价于33.437 ms固定项＋94.041 ms/epoch。此计算只展示现象，本轮用独立组件计时和完整epoch曲线验证其来源。

**selector 固定与变化成本（mode1，inclusive，不可直接相加）：**

| 组件 | E1 CPU/ms | E1 calls | E20 CPU/ms | E20 calls |
| --- | --- | --- | --- | --- |
| teacher_forward | 3.390 | 24 | 3.523 | 24 |
| primary_evaluation | 14.902 | 2 | 141.798 | 21 |
| anchor_evaluation | 6.491 | 2 | 53.988 | 21 |
| backward | 15.225 | 4 | 312.757 | 80 |
| adam_step | 0.167 | 4 | 2.961 | 80 |

**tf 固定与变化成本（mode1，inclusive，不可直接相加）：**

| 组件 | E1 CPU/ms | E1 calls | E20 CPU/ms | E20 calls |
| --- | --- | --- | --- | --- |
| geometry_transport | 357.229 | 20 | 366.406 | 20 |
| teacher_forward | 4.803 | 20 | 4.892 | 20 |
| primary_evaluation | 362.833 | 2 | 543.648 | 21 |
| untouched_evaluation | 23.455 | 2 | 23.295 | 2 |
| target_cache | 367.350 | 56 | 422.587 | 664 |
| backward | 13.277 | 4 | 277.045 | 80 |
| adam_step | 0.193 | 4 | 2.777 | 80 |

TF Teacher evaluation与geometry transport的调用数随E保持不变，训练backward/Adam随E增加。144个correction groups ×20个Teacher batch/ratio miss造成大量nonzero、索引和index_copy。后续epoch复用已transport的targets，摊薄了固定项。该现象不需要用“20 epoch每步更少样本”或“跳过loss”解释。

warmup能够移除首次GEMM/allocator等启动影响，但不能消除每事件都必须重新生成的Teacher cache与transport；每轮都恢复模型和新Adam，不能通过复用训练后的状态把固定项人为移出。生产timer不是纯CUDA计时，但本轮同步native total与CUDA spans都保留，固定transport也由kernel/CPU事件证实。

旧runner复跑数据见 `raw/previous_runner_replication/`，历史日志见 `raw/previous_timing_raw.csv`。历史运行的CPU/GPU频率、其它任务、系统调度没有完整同时期counter；若绝对时间不同，不能把剩余差异归因为某一项warmup/allocator/JIT而无证据。本路径没有TorchScript JIT训练编译步骤。

**同机重跑上一轮原始可执行文件（分支内部timer，三次中位数）：**

| epochs | Selector/ms | TF/ms | repeats |
| --- | --- | --- | --- |
| 5 | 208.020 | 535.507 | 3 |
| 20 | 799.179 | 994.760 | 3 |

旧runner现在也显示更低的Selector绝对时间，与本轮独立runner扣除coordinator前置开销后的量级一致。因此不能把历史503/1914 ms到本轮226/825 ms的差值称为代码加速。TF固定transport与约31–32 ms/epoch的规律在两套runner中一致。

## 13. 细粒度组件时间：频率、调用数与扰动

下表为E5/B8192/R4的独立event。CPU和CUDA均为inclusive；网络forward嵌套在evaluation/cache中，不能跨层求和。CUDA Event span可能比CPU enqueue时间长，也可能包含CPU空闲；实际kernel GPU time另见operator表。`raw/component_summary.csv`同时保存exclusive时间与全部scaling case。

**selector CPU wall互斥分解（mode1，含队列等待和计时扰动）：**

| 阶段 | CPU wall/ms | 占比/% |
| --- | --- | --- |
| coordination_batch_logging_and_instrumentation_misc | 30.185 | 12.195 |
| mask_and_filter_preparation | 0.063 | 0.025 |
| anchor_and_order_cpu_preparation | 2.721 | 1.099 |
| permutation_h2d | 1.071 | 0.433 |
| gaussian_or_order_gather | 6.994 | 2.826 |
| evaluation_scalar_and_loop_other | 6.770 | 2.735 |
| target_cache_other | 59.444 | 24.017 |
| teacher_forward | 3.477 | 1.405 |
| student_forward | 49.356 | 19.941 |
| loss_construction | 9.454 | 3.820 |
| zero_grad | 0.287 | 0.116 |
| explicit_scalar_sync | 1.123 | 0.454 |
| backward | 75.867 | 30.652 |
| optimizer_step | 0.702 | 0.284 |

互斥范围合计247.514 ms。该表可求和；后面的inclusive组件表用于看调用粒度，不能重复求和。

**tf CPU wall互斥分解（mode1，含队列等待和计时扰动）：**

| 阶段 | CPU wall/ms | 占比/% |
| --- | --- | --- |
| coordination_batch_logging_and_instrumentation_misc | 27.271 | 4.748 |
| mask_and_filter_preparation | 0.161 | 0.028 |
| explicit_scalar_sync | 1.183 | 0.206 |
| evaluation_scalar_and_loop_other | 8.923 | 1.554 |
| target_cache_other | 15.506 | 2.700 |
| teacher_forward | 4.943 | 0.861 |
| teacher_geometry_transport | 361.580 | 62.956 |
| student_forward | 46.644 | 8.121 |
| loss_construction | 33.810 | 5.887 |
| anchor_and_order_cpu_preparation | 2.029 | 0.353 |
| permutation_h2d | 0.815 | 0.142 |
| gaussian_or_order_gather | 0.082 | 0.014 |
| zero_grad | 0.435 | 0.076 |
| backward | 70.168 | 12.217 |
| optimizer_step | 0.784 | 0.136 |

互斥范围合计574.336 ms。该表可求和；后面的inclusive组件表用于看调用粒度，不能重复求和。

### selector

| Component | Frequency | CPU总/ms | CUDA span总/ms | CPU/Call µs | Calls |
| --- | --- | --- | --- | --- | --- |
| teacher_forward | EVENT-ONCE population，分batch×ratio | 3.477 | 3.854 | 144.870 | 24 |
| target_cache | 每BATCH×RATIO；含命中与miss | 62.921 | 28.108 | 206.977 | 304 |
| cache_missing_indices | 每BATCH×RATIO | 52.185 | 18.651 | 171.661 | 304 |
| cache_miss_evaluate | EVENT-ONCE population | 4.456 | 4.638 | 185.681 | 24 |
| cache_storage_allocate | EVENT-ONCE×R | 0.009 | 0.009 | 2.280 | 4 |
| cache_target_gather | 每BATCH×RATIO | 3.004 | 2.416 | 9.880 | 304 |
| anchor_sampling | EPOCH-ONCE | 1.466 | 1.475 | 293.147 | 5 |
| shuffle_cpu_h2d_gather | EPOCH-ONCE，anchors/primary/audit各有调用 | 3.884 | 3.899 | 388.433 | 10 |
| permutation_h2d | 每shuffle | 1.071 | 1.045 | 107.118 | 10 |
| shuffle_index_gather | 每shuffle | 0.123 | 0.169 | 12.331 | 10 |
| primary_batch_view | BATCH-ONCE | 0.091 | 0.046 | 4.568 | 20 |
| anchor_batch_view | BATCH-ONCE | 0.036 | 0.043 | 1.786 | 20 |
| primary_evaluation | EVENT-ONCE baseline + EPOCH-ONCE | 42.610 | 33.408 | 7101.658 | 6 |
| anchor_evaluation | 同primary evaluation | 16.826 | 16.835 | 2804.263 | 6 |
| student_forward | 每BATCH×RATIO，含正式评估 | 49.356 | 51.393 | 162.355 | 304 |
| gaussian_gather | 每BATCH×RATIO | 6.870 | 7.334 | 22.600 | 304 |
| loss_construction | 每BATCH×RATIO | 9.454 | 8.424 | 31.099 | 304 |
| backward | OPTIMIZER-STEP | 75.867 | 113.744 | 3793.356 | 20 |
| selector_network_backward | RATIO×strata/每step | 39.078 | 102.372 | 244.236 | 160 |
| adam_construction | EVENT-ONCE | 0.003 | 0.002 | 2.862 | 1 |
| adam_step | OPTIMIZER-STEP | 0.702 | 0.105 | 35.104 | 20 |
| zero_grad | BATCH-ONCE+结束一次 | 0.287 | 0.108 | 13.668 | 21 |
| finite_check_d2h | BATCH-ONCE | 0.947 | 0.966 | 47.362 | 20 |
| positive_check_d2h | BATCH-ONCE | 0.175 | 0.204 | 8.773 | 20 |

mode1/mode0 CPU总时间比：1.095。细粒度event创建、记录与metadata读取会扰动host调度；不能把其绝对组件值直接从native总时间中相减。

### tf

| Component | Frequency | CPU总/ms | CUDA span总/ms | CPU/Call µs | Calls |
| --- | --- | --- | --- | --- | --- |
| teacher_forward | EVENT-ONCE population，分batch×ratio | 4.943 | 7.646 | 247.156 | 20 |
| geometry_transport | EVENT-ONCE target生成；内部144组 | 361.580 | 358.821 | 18079.011 | 20 |
| target_cache | 每BATCH×RATIO；含命中与miss | 382.029 | 382.593 | 2076.246 | 184 |
| cache_missing_indices | 每BATCH×RATIO | 10.764 | 12.059 | 58.498 | 184 |
| cache_miss_evaluate | EVENT-ONCE population | 367.050 | 366.951 | 18352.478 | 20 |
| cache_storage_allocate | EVENT-ONCE×R | 0.012 | 0.014 | 2.942 | 4 |
| cache_target_gather | 每BATCH×RATIO | 1.914 | 1.791 | 10.404 | 184 |
| shuffle_cpu_h2d_gather | EPOCH-ONCE，anchors/primary/audit各有调用 | 2.927 | 2.945 | 487.784 | 6 |
| permutation_h2d | 每shuffle | 0.815 | 0.804 | 135.818 | 6 |
| shuffle_index_gather | 每shuffle | 0.082 | 0.113 | 13.734 | 6 |
| primary_evaluation | EVENT-ONCE baseline + EPOCH-ONCE | 404.405 | 404.357 | 67400.902 | 6 |
| untouched_evaluation | EVENT-ONCE pre/post两次 | 23.127 | 23.126 | 11563.590 | 2 |
| outside_identity_filter | EVENT-ONCE外部过滤 | 0.051 | 0.053 | 50.587 | 1 |
| network_inside_filter | 每次TF forward | 12.570 | 12.524 | 61.615 | 204 |
| student_forward | 每BATCH×RATIO，含正式评估 | 46.644 | 72.021 | 253.497 | 184 |
| gaussian_gather | 每BATCH×RATIO | 3.555 | 4.045 | 19.322 | 184 |
| hexplane_query | 每次TF forward | 3.477 | 4.611 | 17.046 | 204 |
| tf_mlp | 每次TF forward | 6.582 | 42.868 | 32.264 | 204 |
| loss_construction | 每BATCH×RATIO | 33.810 | 13.502 | 183.750 | 184 |
| backward | OPTIMIZER-STEP | 70.168 | 70.339 | 3508.419 | 20 |
| tf_mlp_backward | RATIO-ONCE/每step | 11.263 | 14.286 | 140.788 | 80 |
| hexplane_backward | RATIO-ONCE/每step | 6.081 | 12.805 | 76.014 | 80 |
| output_backward | RATIO-ONCE/每step | 1.021 | 1.436 | 12.767 | 80 |
| adam_construction | EVENT-ONCE | 0.006 | 0.003 | 5.787 | 1 |
| adam_step | OPTIMIZER-STEP | 0.784 | 1.680 | 39.188 | 20 |
| zero_grad | BATCH-ONCE+结束一次 | 0.435 | 0.311 | 20.720 | 21 |
| finite_check_d2h | BATCH-ONCE | 0.965 | 0.986 | 48.267 | 20 |
| positive_check_d2h | BATCH-ONCE | 0.170 | 0.197 | 8.498 | 20 |
| aabb_diagonal_d2h | EVENT-ONCE | 0.048 | 0.046 | 48.119 | 1 |

mode1/mode0 CPU总时间比：1.032。细粒度event创建、记录与metadata读取会扰动host调度；不能把其绝对组件值直接从native总时间中相减。

**事件快照与Teacher准备（在finalize总时间之外单列）：**

| branch | Component | Frequency | CPU总/ms | CUDA span总/ms | Calls |
| --- | --- | --- | --- | --- | --- |
| selector | event/snapshot_and_teacher_preparation | EVENT-ONCE | 0.577 | 0.575 | 1 |
| selector | event/clone_selector_teacher | EVENT-ONCE | 0.457 | 0.460 | 1 |
| tf | event/snapshot_and_teacher_preparation | EVENT-ONCE | 5.099 | 5.097 | 1 |
| tf | event/clone_tf_teacher | EVENT-ONCE | 4.978 | 4.981 | 1 |

临时Adam construction属于固定事件成本，Adam state实际分配发生在首个optimizer step。batch构造的slice view、H2D permutation和gather已分别列出；miscellaneous可由raw组件exclusive时间和coordinator总范围核对，不将所有nested inclusive时间求和。

## 14. Batch scaling 与尾 batch

保持所有训练点、四个ratio、相同网络与loss。batch改变实际step数，这是实验变量。完整事件时间包括已有审计；TF audit大小随batch变化，故另给训练batch范围内的samples/s、forward/backward/Adam。训练范围来自mode1，受插桩扰动；native总时间用于稳定吞吐比较。

### selector

| batch | native事件median/ms | steps | ratio样本/s含固定审计 | peak allocated MiB | 训练batch总/ms* | 训练forward/ms* | backward总/ms* | Adam总/ms* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2048 | 641.844 | 75 | 1398750 | 60.638 | 502.158 | 110.509 | 265.383 | 2.856 |
| 4096 | 376.863 | 40 | 2382245 | 72.374 | 288.015 | 55.985 | 141.619 | 1.432 |
| 8192 | 225.970 | 20 | 3973004 | 96.855 | 164.604 | 28.558 | 75.867 | 0.702 |
| 16384 | 122.875 | 10 | 7306454 | 146.668 | 74.627 | 14.524 | 39.522 | 0.434 |
| 32768 | 81.966 | 5 | 10953041 | 241.416 | 38.108 | 7.317 | 20.041 | 0.198 |

### tf

| batch | native事件median/ms | steps | ratio样本/s含固定审计 | peak allocated MiB | 训练batch总/ms* | 训练forward/ms* | backward总/ms* | Adam总/ms* |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2048 | 1386.959 | 75 | 428030 | 80.114 | 447.648 | 78.407 | 250.501 | 2.581 |
| 4096 | 946.893 | 40 | 626955 | 83.349 | 238.617 | 41.557 | 133.752 | 1.405 |
| 8192 | 556.474 | 20 | 1066824 | 95.575 | 123.692 | 21.297 | 70.168 | 0.784 |
| 16384 | 338.250 | 10 | 1755092 | 124.020 | 63.252 | 12.231 | 34.845 | 0.407 |
| 32768 | 221.520 | 5 | 2679938 | 169.030 | 31.299 | 6.108 | 17.244 | 0.230 |

`*`为插桩范围。`raw/batch_training_summary.json`保留训练阶段吞吐和全部Student forward总耗时，后者含正式评估；不要把含评估forward除以训练steps当成每step forward。

Selector B8192并没有一个5100左右的普通tail：生产stratumBatch将两个群体均匀切成4份。primary为7421/7422，anchor为3800/3801；这些小变化不构成极小tail。TF才是3个8192加5107，tail仍为full的62.34%，并非tiny batch。

| branch | 实际shape范围 | calls | CPU µs/call | CUDA span µs/call |
| --- | --- | --- | --- | --- |
| selector | batch_shape_primary_7422_anchor_3801 | 10 | 8604.214 | 8669.798 |
| selector | batch_shape_primary_7422_anchor_3800 | 5 | 8602.128 | 8617.168 |
| selector | batch_shape_primary_7421_anchor_3800 | 5 | 6976.416 | 8731.648 |
| tf | batch_shape_8192 | 15 | 6148.983 | 6160.998 |
| tf | batch_shape_5107 | 5 | 6157.004 | 6169.804 |

tail范围包括该batch的ratio forwards、loss、backward与Adam，不能解释为单一kernel效率；是否出现更高每样本成本应按实际样本数归一化，最终优化仍须保持训练语义。

TF tail/full的CUDA span每样本成本比约1.606。本事件tail耗时接近full，但样本少37.66%，说明该batch范围存在显著非按样本线性缩放的成本；这不等同于硬件counter证明的memory-bound。

## 15. Ratio scaling

### selector

| ratios | 取值 | native median/ms | steps | ratio样本/s |
| --- | --- | --- | --- | --- |
| 1 | [0.01] | 79.744 | 20 | 2814575 |
| 2 | [0.01,0.05] | 128.625 | 20 | 3489905 |
| 4 | [0.01,0.05,0.10,0.15] | 225.970 | 20 | 3973004 |

### tf

| ratios | 取值 | native median/ms | steps | ratio样本/s |
| --- | --- | --- | --- | --- |
| 1 | [0.01] | 163.584 | 20 | 907269 |
| 2 | [0.01,0.05] | 287.931 | 20 | 1030908 |
| 4 | [0.01,0.05,0.10,0.15] | 556.474 | 20 | 1066824 |

每个ratio都会增加forward、loss与custom backward节点，但不增加optimizer.step次数；串行ratio loop的可共享数据依赖已在第4节逐项列出。TF fixed Teacher transport也按ratio重复，所以其ratio scaling同时改变固定开销和训练开销。

## 16. Operator／kernel／indexing 与 backward

以下为单个代表epoch的LibTorch Kineto trace。其CPU/GPU运行时间受profiler影响，适合定位operator和kernel，不替代native median。已按CUDA correlation → launch所在CPU线程scope归因，避免将该build写出的CUDA External id误当CPU operator id；全部kernel均完成归因。

### selector

| 指标 | 结果 |
| --- | --- |
| kernel calls | 6793 |
| total kernel GPU ms | 44.186 |
| backward kernel GPU ms | 21.132 |
| Student forward kernel GPU ms | 9.014 |
| index/gather/nonzero/scatter/where 直接operator GPU ms | 3.383 |
| 上述index占全部kernel时间 | 7.656% |
| duration<10µs kernels | 6174 / 6793 |
| tiny kernel累计GPU ms | 20.456 |

| 互斥阶段 | kernel GPU ms |
| --- | --- |
| index_cache_event_and_misc | 10.463 |
| teacher_forward | 2.844 |
| student_forward | 9.014 |
| loss_construction | 0.721 |
| backward | 21.132 |
| adam | 0.012 |

**backward top operators / custom scopes**

| operator | CUDA/ms | kernel calls | backward GPU % |
| --- | --- | --- | --- |
| selector/selector_network_backward | 17.180 | 928 | 81.299 |
| aten::sum | 3.359 | 288 | 15.895 |
| aten::mul | 0.200 | 100 | 0.946 |
| aten::div | 0.129 | 64 | 0.610 |
| aten::fill_ | 0.072 | 36 | 0.341 |
| aten::sgn | 0.064 | 32 | 0.303 |
| aten::copy_ | 0.064 | 32 | 0.303 |
| aten::_softmax_backward_data | 0.064 | 32 | 0.303 |

**backward top kernels**

| kernel | calls | GPU总/ms | average/µs |
| --- | --- | --- | --- |
| ampere_sgemm_32x32_sliced1x4_nt | 224 | 10.636 | 47.482 |
| ampere_sgemm_64x32_sliced1x4_nt | 32 | 3.520 | 110.000 |
| void at::native::reduce_kernel<128, 4, at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4> >(at::native::ReduceOp<float, at::native::func_wrapper_t<float, at::native::sum_functor<float, float, float>::operator()(at::TensorIterator&)::{lambda(float, float)#1}>, unsigned int, float, 4>) | 256 | 3.191 | 12.465 |
| ampere_sgemm_32x128_nn | 224 | 1.111 | 4.960 |
| void gemvNSP_kernel<float, float, float, float, 1, 32, 4, 1024, false, cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float> >(cublasGemvParams<cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float const>, cublasGemvTensorStridedBatched<float>, float>) | 32 | 0.633 | 19.781 |
| void splitKreduce_kernel<32, 16, int, float, float, float, float, true, false, false>(cublasSplitKParams<float>, float const*, float const*, float*, float const*, float const*, float const*, float const*, float*, void*, long, float*, int*) | 128 | 0.413 | 3.227 |
| void selector_cuda::(anonymous namespace)::directFinal<true>(float const*, float const*, float*, float const*, int, int, int) | 128 | 0.338 | 2.641 |
| selector_cuda::(anonymous namespace)::reluVjp(float*, float const*, int, int) | 128 | 0.320 | 2.500 |

### tf

| 指标 | 结果 |
| --- | --- |
| kernel calls | 47859 |
| total kernel GPU ms | 145.737 |
| backward kernel GPU ms | 6.408 |
| Student forward kernel GPU ms | 15.897 |
| index/gather/nonzero/scatter/where 直接operator GPU ms | 51.820 |
| 上述index占全部kernel时间 | 35.557% |
| duration<10µs kernels | 47503 / 47859 |
| tiny kernel累计GPU ms | 122.645 |

| 互斥阶段 | kernel GPU ms |
| --- | --- |
| index_cache_event_and_misc | 8.781 |
| teacher_forward | 5.546 |
| teacher_geometry_transport | 105.226 |
| student_forward | 15.897 |
| loss_construction | 3.696 |
| backward | 6.408 |
| adam | 0.183 |

**backward top operators / custom scopes**

| operator | CUDA/ms | kernel calls | backward GPU % |
| --- | --- | --- | --- |
| tf/tf_mlp_backward | 2.793 | 512 | 43.586 |
| tf/hexplane_backward | 1.603 | 16 | 25.016 |
| aten::fill_ | 0.488 | 244 | 7.615 |
| aten::mul | 0.400 | 176 | 6.242 |
| aten::div | 0.341 | 132 | 5.321 |
| aten::index_select | 0.149 | 48 | 2.325 |
| tf/output_backward | 0.110 | 16 | 1.717 |
| aten::where | 0.105 | 32 | 1.639 |
| aten::sgn | 0.097 | 48 | 1.514 |
| aten::ge | 0.065 | 32 | 1.014 |
| aten::add_ | 0.065 | 32 | 1.014 |
| aten::neg | 0.064 | 32 | 0.999 |

**backward top kernels**

| kernel | calls | GPU总/ms | average/µs |
| --- | --- | --- | --- |
| tf_cuda::(anonymous namespace)::queryGradKernel(tf_cuda::(anonymous namespace)::Inputs, tf_cuda::(anonymous namespace)::Planes, float const*, float*, float*, int) | 16 | 1.603 | 100.188 |
| ampere_sgemm_32x32_sliced1x4_nt | 112 | 0.862 | 7.696 |
| tf_cuda::(anonymous namespace)::biasGradKernel(float const*, float*, int, int) | 112 | 0.680 | 6.071 |
| ampere_sgemm_32x128_nn | 112 | 0.582 | 5.196 |
| void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, at::detail::Array<char*, 1> >(int, at::native::FillFunctor<float>, at::detail::Array<char*, 1>) | 244 | 0.488 | 2.000 |
| void splitKreduce_kernel<32, 16, int, float, float, float, float, true, false, false>(cublasSplitKParams<float>, float const*, float const*, float*, float const*, float const*, float const*, float const*, float*, void*, long, float*, int*) | 112 | 0.448 | 4.000 |
| void at::native::(anonymous namespace)::indexSelectLargeIndex<float, long, unsigned int, 2, 2, -2, true>(at::cuda::detail::TensorInfo<float, unsigned int>, at::cuda::detail::TensorInfo<float const, unsigned int>, at::cuda::detail::TensorInfo<long const, unsigned int>, int, int, unsigned int, unsigned int, long) | 48 | 0.149 | 3.104 |
| void at::native::elementwise_kernel<128, 2, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}>(int, at::native::gpu_kernel_impl_nocast<at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > >(at::TensorIteratorBase&, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> > const&)::{lambda(int)#1}) | 48 | 0.144 | 3.000 |

专用CUDA路径没有aten grid_sampler_backward；其对应工作为queryGradKernel。MLP GEMM、bias reduction、ReLU、Output quaternion backward和ATen损失链分别保留在原始trace及top表。custom Function下的直接CUDA调用归到相应scope，因此scope与ATen operator均可出现于同一排行榜；百分比分母统一为已归因backward kernel总时间，不使用CPU耗时冒充CUDA。

本模型的grid query并非主要forward热点：代表epoch中全部76次Teacher/Student queryKernel合计1.256 ms，对应76次Warp32 MLP为15.927 ms。backward中MLP占43.59%，HexPlane query backward占25.02%。因此“TF主要慢在grid_sample”不符合该CUDA模型的实测；legacy默认模型没有在此事件中做相同runtime结论。

Selector参数虽只有4962，但backward仍有可测的GEMM与reduction热点；不能把全部成本称为launch overhead。其大量短kernel、cache hit动态索引与串行多ratio编排值得优化调查，须同时保留真正矩阵计算的成本。两分支Adam/zero_grad在E5总事件中的占比很小，未显示为首要瓶颈。

TF完整事件的索引开销很大，且多数来自Teacher transport固定项；仅比较“backward vs event total”会严重低估训练kernel的优化价值。后续应按固定项与steady training分别优化，不把两类瓶颈混为一谈。

## 17. Nsight Systems／Compute 与利用率结论

| branch | Nsight Systems | Nsight Compute | 选中kernel |
| --- | --- | --- | --- |
| selector | completed | permission_denied_ERR_NVGPUCTRPERM | ampere_sgemm_32x32_sliced1x4_nt |
| tf | completed | no_kernel_captured | tf_cuda::(anonymous namespace)::queryGradKernel(tf_cuda::(anonymous namespace)::Inputs, tf_cuda::(anonymous namespace)::Planes, float const*, float*, float*, int) |

| Nsight独立采集 | kernel calls | kernel GPU总/ms | <10µs calls |
| --- | --- | --- | --- |
| selector | 6793 | 46.708 | 6183 |
| tf | 47859 | 171.106 | 47503 |

原始 `.nsys-rep`、SQLite、stats CSV、NCU CSV/stdout全部保留。此版nsys使用gpukernsum/cudaapisum/nvtxsum报告名；不匹配的尝试另保存在profiler_attempt1/2。NCU只有实际捕获硬件counter才算完成；退出码0但“No kernels were profiled”不算counter证据。若驱动权限拒绝则记录原文，不改驱动、不安装工具。

本次NCU实际阻塞为 **ERR_NVGPUCTRPERM**：当前用户无GPU Performance Counter访问权限。已记录原始错误，未修改系统权限。因此SM active、achieved occupancy、DRAM吞吐和SM throughput的硬件counter结论均为 **inconclusive**；Kineto给出的估算occupancy不替代硬件实测。

利用率结论：两个分支均存在大量微小kernel及频繁launch；TF尤其包含与144组transport有关的host/synchronization/索引编排成本。trace中的kernel间空隙和实际kernel时间可支持“launch与编排值得优先调查”，但不能据此声称SM在全部训练中都闲置。完整CD事件的compute-bound／memory-bound分类为 **inconclusive**；单kernel counter即使可用，也不能推广到混合的整个事件。batch8192不是通过本轮证明的最优点；较大batch的吞吐变化还混入step数、Teacher分块、diagnostic audit大小和TF Warp32/cuBLAS dispatch变化。

## 18. 复现、验证与交付文件

| 验证 | 结果 |
| --- | --- |
| 相同event输入 | raw/input_sha256.json；原始event.json保存；snapshot未重建 |
| 重复初始化 | 每run参数逐tensor torch::equal检查；新Manager、新临时Adam、同种子 |
| 实际step数 | 282 条run记录通过公式/实际计数验证 |
| Gaussian / normal Adam | 每run断言未修改 |
| 插桩参数一致性 Selector | True bitwise; allclose=True |
| 插桩参数一致性 TF | False bitwise; allclose=True |
| TF parity容差 | atol=2e-5、rtol=2e-4，逐参数max/mean误差见JSON；atomic非逐bit确定性 |
| 正式配置 | 未修改；全部scaling设置仅存在于独立benchmark |

本地主要证据目录：`tests/diagnostics/change_distill_performance_20261017/`。

此报告中的 `raw/`、`source/`、`instrumented/`、`plots/` 均相对于上述证据目录；根目录报告中的图像链接则使用仓库相对路径。

| 路径 | 内容 |
| --- | --- |
| source/ + raw/source_manifest.json | 当前本地冻结源码与逐文件SHA-256 |
| instrumented/ + instrumentation.diff | 隔离插桩副本及完整差异 |
| raw/cases/*/{cold,warmup_*,warm_*}/runtime.json | 全部逐run CPU/CUDA/显存/step数据 |
| raw/cases/*_m1/warm_0/components.csv | 逐次组件CPU与CUDA Event spans |
| raw/cases/*_m2/profile/trace.json | 原始LibTorch Kineto Chrome trace |
| raw/*_operators.csv / *_kernels.csv / *_transfers.csv | 归因后的operator/kernel/设备传输统计 |
| raw/*_allocation_events.csv | 原始allocation/deallocation事件 |
| raw/nsys_*.nsys-rep / *.sqlite / *_stats.csv | Nsight Systems原始与导出数据 |
| raw/nsys_*_kernels.csv / *_api.csv | 从原始SQLite导出的单表CSV，可直接程序读取 |
| raw/ncu_*.csv / *.log | 有限热点硬件counter或不可用原因 |
| raw/benchmark_summary.csv/json + all_runs.csv | 统计汇总，含mean/median/std/min/max |
| raw/epoch_fit.json + batch_training_summary.json | epoch拟合与训练阶段batch吞吐 |
| raw/previous_runner_replication/ | 本轮同机重跑上一轮runner |
| raw/run_contracts.json + instrumentation_parity.json | step、隔离与插桩一致性验证 |
| raw/*tensor_metadata.json及各case的tensor_metadata.json | shape、stride、contiguous、dtype、device、requires_grad、grad_fn |

服务器完整工作目录：`/home/prmi/zwj/mycartgs/results/change_distill_performance_audit/20261017`。原始完整snapshot继续保存在上一轮目录，SHA-256记录用于复用。原始profiling文件已完整复制本地，并打包为 `raw_profiling_outputs.tar.gz`；`audit_source.tar.gz`为冻结源码、插桩、独立runner和分析脚本。

复现顺序：冻结/核对source → `build_native.py` → `run_matrix.py --environment` → `run_matrix.py` → `run_kernel_profiles.py` → `replicate_previous_runner.py` → `validate_results.py` → `analyze.py` → `make_report.py`。远程脚本包含本次明确主机的绝对路径，不把它们描述为通用安装工具。

**可直接进入下一轮设计的信息：** TF event固定transport是已证实瓶颈；Selector跨ratio重算三个属性embedding、两分支cache-hit动态索引和custom backward多算输入梯度具有明确源码依据；具体fusion、vectorization、Graph或混合精度方案仍需独立数值/性能验收，本轮不实现、不据此删减任何算法工作。
