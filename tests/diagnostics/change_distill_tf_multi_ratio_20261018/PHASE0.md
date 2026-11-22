# TF multi-ratio Phase 0：正式 Transport V2 基线

本轮目录独立于原审计和 transport 优化目录。34 个生产文件已冻结，远端逐文件 normalized SHA-256 比较无漂移；初始用户 Git diff/status 保留在本目录。原 transport、Selector、cache 实现保持不变。

## 真实路径与 ratio 依赖

`ChangeDistillManager::execute` 中 TF `errors` → `TransformFieldImpl::applyLogResidual` → `insideMask/nonzero` → Gaussian attributes gather → `forward` → `tf_cuda::query` → `QueryFunction::forward/queryForward/queryKernel` → `neural/NeuralFunction/mlpForward` → `output/OutputFunction` → 三个 index_copy → `tfErrors`。

训练为 batch → 四个 ratio 原顺序 → 三项加权 loss 原顺序 → 除 ratio 数量 → finite/positive 标量控制检查 → 一次 backward → 一次临时 Adam update。LC6207 每 epoch 4 batches，5 epochs 共 20 steps。Teacher frozen cache miss 独立执行原 geometry transport；已有 coverage direct-hit 保留。

六平面真实顺序为 `XY, XZ, X-ratio, YZ, Y-ratio, Z-ratio`。每 level/channel 按此顺序从 1 开始逐项相乘。ratio 直接进入后三个轴含 ratio 的 bilinear sampling，不存在后置 ratio embedding 或 ratio concat 层。XYZ 经 AABB normalization，rotation/log-scale 只进入 output transform。

因此不能共享完整 `[B,16]` feature，也不能对四个 dFeature 先简单求和再使用一个相同 HexPlane Jacobian。三个纯空间 sample 可共享，但不能提前合并成空间乘积；X-ratio 插在 XZ 和 YZ 之间，改乘法顺序会改 FP32 结果。

网络：2 levels × 8 channels → feature `[B,16]`；共享 Linear16→32/ReLU；XYZ/scale/rotation 三头各 Linear32→32/ReLU→3/3/4。MLP 4042 参数，grid1182720 参数，总计1186762、26 parameter tensors。正式 B≤10000、F16/H32 走 Warp32 fused forward，其余走已有 strict FP32 cuBLAS。query backward 已在一个 kernel 内处理所有 planes，通过 atomicAdd 写 plane gradients。

## 新基线

首轮 restored warm native medians：TF 200.709 ms、Selector195.032 ms、both375.854 ms；仅作为本轮起点，最终以同条件正反顺序配对为准。

| Scope | Calls | CPU wall ms | CUDA span ms |
|---|---:|---:|---:|
| pre evaluation | 1 | 28.019 | 28.020 |
| audit pre | 1 | 3.969 | 3.969 |
| post evaluation | 5 | 40.867 | 40.824 |
| audit post | 1 | 2.126 | 2.124 |
| Teacher forward | 20 | 4.559 | 7.436 |
| geometry transport | 20 | 13.667 | 11.398 |
| Student forward | 184 | 44.867 | 70.835 |
| HexPlane forward（含 Teacher） | 204 | 3.261 | 4.528 |
| MLP forward（含 Teacher） | 204 | 5.724 | 41.841 |
| loss construction | 184 | 32.671 | 14.044 |
| evaluation scalar readback | 312 | 8.138 | 3.247 |
| backward | 20 | 69.156 | 69.483 |
| MLP backward | 80 | 10.812 | 15.432 |
| HexPlane backward | 80 | 6.195 | 14.097 |
| Adam | 20 | 0.750 | 1.544 |

上述 scope 为 inclusive，不能相加；instrumented run 本身受 CUDA Event/CPU dispatch 扰动。CUDA span 包含排队、CPU 供给不足造成的间隙，不能当作 kernel busy time。

Nsight：20452 kernels，19830 个 <10 μs，128.274 ms 累计 kernel GPU duration。MLP forward38.358 ms；query forward3.337 ms；queryGradKernel9.186 ms；MLP backward scopes14.960 ms；Adam0.960 ms。无硬件 counters，不据此宣称 compute-bound 或 memory-bound。

## 同步与重复工作

- 每 ratio 的 insideMask/nonzero 会隐式同步选中点数，并重复三个 attributes gather、三个 output scatter。
- evaluation 的 `.item<double>()` 每 batch×ratio×3 次，共312次，仅进入日志和报告；不控制训练、acceptance、early stop 或 sampling。
- 训练 finite/positive `.item()` 参与控制，保留。
- 原 evaluation 在 CPU double 中按 batch→ratio→component 顺序累加；延迟 readback 候选必须保留相同 FP32 loss 和相同 CPU double fold 顺序，不能换成 GPU FP32 累加。
- 原 autograd feature hooks 在 first step 实测顺序为 `[3,2,1,0]`。
- Adam 已是项目现有 multi-tensor direct update，不是本轮高价值目标。

原始证据位于远端同名目录 `baseline/raw/{native_*,verify_*,profile_tf,nsys_tf}`；最终统一归档到本轮原始数据包。
