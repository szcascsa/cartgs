# Dynamic AABB 与 Change Distill 实现说明

## 执行边界

- LoopClosingBA / ScaleRefinement 在 GaussianMapper 消费 MappingOperation 时串行完成：冻结旧状态 → 原有几何修正 → 即时 AABB 检测 → 必要时重采样 → Selector CD → TF CD。没有新增 SLAM 同步状态机。
- 周期检查使用 mapping_iter，每次正常 KF 优化、densify/prune 和 optimizer step 完成后检查；Replay 与 CD 不推进该时钟。
- GaussianModel 返回真实执行的 indices、变换、scale、rotation 和属性标记。XYZ 执行原始 correction；SCALE / ROTATION 按 1e-5 identity 容差决定是否更新对应属性，并共享同一 transport。未变化属性不改写、不清其普通 optimizer state；不是按 Mapper 的 large-motion 阈值猜测属性。整图结果使用 affects_all + population_count，源头不分配 arange(N)，Coordinator 再裁剪到 snapshot_count。
- 快照保存完整 [0,N0) 的 xyz / log_scaling / normalized wxyz rotation。蒸馏只使用 MatureBefore。允许尾部 append；新增点不进入旧 teacher。Prune/reset 或重复 correction 对应关系立即报错。

## AABB 重采样

当前 TF AABB 是唯一真值。只扩张、不收缩；规划使用配置的成熟点分位数范围，因此极端离群点可能仍留在域外。
下界采用 ceil(q_low*(N-1))，上界采用 floor(q_high*(N-1))，两端向内裁剪；不再因下界向下取整保留唯一负向离群点。如果区间内没有整数 rank（例如默认配置下 N=2），保留夹住该区间的两个相邻样本，避免上下界反转。分位数设为 0/1 时仍使用 min/max。

1. 开启相关 CD 时先冻结旧 TF，再在每个 multires level 重采样受影响的 plane；分辨率和 Parameter identity 不变。关闭 CD 时不克隆 teacher，仍允许独立扩容。
2. new-grid → world → old-grid，统一 bilinear / border / align_corners=true；ratio 轴不变化。
3. 旧域覆盖的格点：双线性采样旧 plane。
4. 新区格点复用 TF 正常初始化：xy/xz/yz 为 Uniform(0.1,0.5)，xe/ye/ze 为 1。不是所有 plane 都填 1，也不使用 zero padding 初始化新区。
5. Coordinator 提交新 AABB，仅清除被重采样 plane 的普通 Adam state；MLP 和未受影响 plane 的 state 不动。
6. 进行 TF CD 修正重采样误差。新区随机 feature 不保证几何残差严格为零；旧域外 teacher 的 identity 监督与后续 photometric training 负责约束。

## 蒸馏集合与目标

| 事件 | Selector | TF |
| --- | --- | --- |
| Loop | corrected mature + unaffected anchors | corrected mature |
| Loop + AABB | corrected mature + unaffected anchors | 全部 snapshot mature |
| ScaleRefinement | 全部 snapshot mature | 全部 snapshot mature |
| AABB-only | 不执行，连已有梯度也不清除 | 全部 snapshot mature |

以上为蒸馏开关全部启用时的行为，各开关仍可分别关闭对应监督。

- Selector 使用确定性 forwardProbability，对比旧输入 / 新输入下的 keep probability；输入 scale 在边界从 log_scaling 转回 exp。
- 每 epoch 全局均匀、无放回采样 unaffected anchors，数量为 min(floor(rho*N_corrected), N_unaffected, anchor_max_count)。Corrected / anchor 分别取 mean 再加权；分层 mini-batch 按各自人口规模加权。
- epochs=1 不额外增加步骤。若只有一个 batch，更新前 anchors loss 为零，不对首次更新产生约束；一个 epoch 有多个 batch 时，后续 batch 的 anchors 可以约束此前更新产生的漂移。
- TF teacher 先计算旧 TF 的最终 Gaussian 属性，再执行对应 geometry transport。旧域外输出 identity 后 transport；新域外的 student 固定 identity 时跳过 TF loss，日志记录 skipped_outside，不排除 Selector。
- TF loss：旧 AABB 对角线归一化的位置 L1、log-scale L1、normalized quaternion 的 1-abs(dot)。没有额外 photometric / regularization / GI loss。每个 batch 覆盖全部 target_ratios，平均后一次 backward。
- pre/post 在同一集合上统计。TF untouched audit 仅作诊断，不加入训练。指标下降是观察项，不强制断言。

## 事件与逐 Epoch 日志

Coordinator 打印可读的事件类型和 scope；实际进入蒸馏后打印 [CD BEGIN]，完成后打印 [CD END]。所有日志使用同一个 event id 串联，BEGIN 同时包含 mapping_iter、epochs、batch_size 和 ratio 数量。

| type | 当前执行链中的含义 |
| --- | --- |
| LOCAL_GEOMETRY | Loop Closure 局部几何事件 |
| GLOBAL_GEOMETRY | ScaleRefinement 全局几何事件 |
| FIELD_DOMAIN | AABB-only 扩域事件 |
| LOCAL_GEOMETRY&#124;FIELD_DOMAIN | Loop Closure 与 AABB 扩域合并事件 |
| GLOBAL_GEOMETRY&#124;FIELD_DOMAIN | ScaleRefinement 与 AABB 扩域合并事件 |

- [SELECTOR CD EPOCH] 和 [TF CD EPOCH] 分别记录两个独立优化阶段；epoch=0/E 是训练前基线，随后每完成一轮打印 epoch=1/E 至 E/E。发生扩域时，该基线位于 resample 之后、CD 更新之前。
- loss 是全体有效 primary、全部 ratio 上的加权评估均值，不是最后一个 batch 的 loss。Selector 为 selector_weight * (primary_loss + anchor_loss_weight * anchor_loss)；TF 为 position/scale/rotation 三项加权和。
- Selector 的评估 anchors 在整个事件内固定为首次采样集合，audit_anchors 显示其数量；训练 anchors 仍按原方案每 epoch 重采样。TF 的有效 primary 集合也保持固定。
- delta_prev = 当前 loss - 上一轮 loss，负数表示下降，正数表示上升；steps 为本轮实际 optimizer step 数。应比较同一个 event、同一个模块的相邻 epoch，不能将不同事件或 Selector/TF 的 loss 混在一起比较。
- Selector 额外打印 primary_loss、anchor_loss；TF 额外打印 position_loss、log_scale_loss、rotation_loss 和 skipped_outside。采用科学计数法保留小数值变化。
- 空集合、零 loss 权重或 TF 全部位于新域外时打印相应 SKIP 原因，不伪造零 loss 的训练 epoch。是否下降仅记录，不强制假设单调下降。

逐 epoch 评估复用 NoGrad 路径，不改变训练集合、梯度、optimizer state、训练时钟或随机数流；每个额外 epoch 增加一次固定集合的前向评估开销。最后一轮评估复用于已有 post 汇总，避免重复计算。

## Optimizer 与随机数

CD 使用事件级独立临时 Adam：引用 live parameters，不复用普通 optimizer state。固定使用专用配置：Selector=1e-4、TF MLP=1.6e-5、TF grid=1.6e-4；不跟随普通训练的学习率调度。beta / epsilon 沿用对应普通 param-group，weight_decay=0。TF 参数组顺序为 MLP、grid，测试与 Mapper 保持一致。每次事件重新创建 Adam，事件结束释放；日志打印实际 CD 学习率。

CD 不更新 Master、GI、Protection、KF scheduler，也不推进 mapping_iter / selector_step / transform_step。cd_selector_steps 与 cd_tf_steps 独立记录，只累计真正的 optimizer step。

anchors / shuffle 使用独立随机引擎。Teacher clone 和 AABB 初始化保存并恢复普通 Torch RNG，避免额外消耗正常训练的随机序列。

## 配置

所有参数保留在 cfg/selector/selector_config.yaml，保存模型时同步写出。新增：

~~~yaml
ChangeDistill.epochs: 1
ChangeDistill.seed: 0
ChangeDistill.enable_unaffected_anchors: 1
ChangeDistill.anchor_ratio_to_corrected: 1.0
ChangeDistill.anchor_max_count: 32768
ChangeDistill.anchor_loss_weight: 1.0
ChangeDistill.selector_learning_rate: 0.0001
ChangeDistill.tf_grid_learning_rate: 0.00016
ChangeDistill.tf_mlp_learning_rate: 0.000016
~~~

保留现有 batch_size=8192，epochs=1 表示 primary population 完整扫一遍，不表示固定一个 optimizer step。这三个 LR 是可调的初始实验设置，不代表收敛保证。post_change_replay 只允许 0；本阶段不加额外 Replay。Selector.enabled 默认值不改，完整链路实验需要显式启用。ChangeDistill.enable=0 时跳过 teacher 克隆和 CD，但周期以及 Geometry Change 后的 AABB 检查仍有效，必要时执行 resample、setAABB 和受影响 plane 的 optimizer state 清理。

## 验收

强制 mapping 目标尚未达到、但当前 iteration 已达到有限 Optimization.iterations 时，Mapper 打印一次 WARNING，包含 iteration、优化上限、mapping_iter 和目标次数。警告明确指出普通 Gaussian/Selector/TF optimizer step 已跳过、Gaussian 梯度可能累积，而 mapping 计数和周期 AABB 检查仍继续。本改动仅告警，不改变停止策略；无限优化（iterations=-1）或未启用强制目标时不触发。

新增 tests/hexplane_resample_test.cpp 与 tests/change_distill_test.cpp，并接入 CARTGS_BUILD_TESTS：

- 多分辨率重采样、ratio 轴不变、新区初始化、未受影响参数不变、禁止收缩；
- 默认分位数下 999/1000/1001/10000 个成熟点的正负离群点对称裁剪，以及 1/2 个样本的有效边界；
- Teacher 深拷贝 / frozen / eval、buffers 不共享、普通 RNG 不前进；
- Sim(3) transport、corrected / anchor / protected / appended 分组、correspondence 拒绝条件；
- AABB-only 不修改 Selector 参数 / 梯度 / optimizer state；
- 旧域外 → 新域内的 identity teacher、新域外排除；
- CD 与普通 Adam state / Master 参数和梯度隔离。
- 关闭 CD / 缺少 distiller 时仍执行 AABB 扩容，仅清受影响 plane state；即时检查排除 protected / appended Gaussian。
- global correction 无显式 indices，自动限制到 snapshot population；空 correction 不产生 group。
- CD 不受普通 LR 改变影响，三个专用 LR 各自控制对应参数组，拒绝零、负值、NaN 和 Inf。
- 日志回归覆盖可读事件/scope、epoch=0 基线与两轮 loss、delta_prev 和实际 step 数、最终 loss 与 report 一致，以及空集合/全部域外时的 SKIP 日志；不改变全局 cout 数字格式。
- 在既有 gaussian_model_lifecycle_test 中补充源头结果回归：平移、缩放、旋转、微小平移的实际属性与 flags 一致；未变化属性及其 optimizer momentum 保持不变（需要 CUDA）。

本次仅执行静态检查，未运行 CMake、编译、LibTorch 单元测试或 CUDA 实验。以上为已编写测试的覆盖范围，不表示运行通过。
