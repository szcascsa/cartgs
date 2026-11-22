# 历史渲染质量约束的联合更新

## 作用与边界

正常 Full/LoD 梯度累积完成后，保留原 densify/prune/resetOpacity 顺序，在共享 Adam 提交点联合检查 Gaussian、Selector、TF。有限的最终质量 CANCEL 可以保留 0.2 原始 Adam 位移，并触发冲突源远端联合 Replay；Change CD、恢复 DSD、GI 的职责不变，恢复 DSD 仍是独立配置项。

这不是 Teacher 输出蒸馏：每步基线是更新前的当前地图/网络，历史 GT 提供渲染质量监督。只保护本次采样历史 KF/ratio、固定随机选点和 post-topology population；不回滚已经完成的拓扑/几何变化，不保证未采样历史视角永不退化。

## 启动与计时

`GradientConstraint.tf_warmup_steps: 1000`：TF 初始化后，正常共享优化器成功提交 TF 更新满 1000 次，下一次正常更新才启用守护。初始化前及预热期间执行原共享 Adam，打印 `WARMUP`，不做历史渲染检查。

- TF 计数要求正常 TF 参数组存在梯度、LR 非零，并成功提交；受保护提交中 TF 参数组零位移不计数。
- CANCEL、NOOP、Gaussian-only、CD、DSD、Replay 不推进 TF 正常更新计数。
- 只有新 TF 初始化重置计数；Loop/AABB Change 不重置，下一步重新读取当前历史基线。
- Gaussian XYZ 学习率调度扣除被拒绝/NOOP 的 mapping 更新；mapping iteration、年龄/拓扑调度和 ratio 轮换继续推进，拒绝不会卡住同一 ratio。
- 保留 `transform_state.json` 的 `transform_step` 字段，含义改为已提交的正常 TF 更新数；另记录 `rejected_mapping_updates`。

fast 配置 `TransformField.start_iter: 3000`，因此守护约在 mapping=4000 启动，具体取决于有效 TF 更新数。TF 关闭或未训练满预热长度时，不启用守护；短序列可能仅在尾优化启用。

## 工作管线

1. 从当前 optimizer 中的 `grad.defined()` 参数构造 Adam 事务预览，保持 LibTorch 2.3.1 的 moments 和参数更新顺序，不写当前参数、正常梯度或 optimizer state。
2. Recent Window 外轮转历史 KF/ratio。在线按 `Selector.lod_render_scale` 缩放，SLAM 结束后原尺寸；只为被检查 ratio 准备 Protected admission。
3. NoGrad/Inference 计算更新前各历史 Full/LoD MSE，再验证原始 Adam 候选。通过则直接提交，不做历史反传或投影。
4. 原始候选失败时才构建逐目标的新 autograd 图，用 `autograd::grad` 计算历史梯度；不污染正常 `.grad`，不把原来 grad=None 的参数加入更新。
5. 仅投影到历史 Full/LoD 损失的一阶不增加半空间。点积、Gram 及修正方向使用 FP64，方向回到参数 dtype 后检查 primal residual。小规模 active-set 使用全主元消元处理秩亏，不新增 QP 依赖、不合并不同质量目标。
6. Dual 最优性未收敛不是独立拒绝条件；实际方向满足一阶可行性后仍可进入真实验证。方向不可行、非有限值才取消。
7. 投影方向依次以 1.0/0.5/0.25 验证；未改变原方向时跳过重复的 1.0。原始候选计入 `max_candidate_checks`，默认最多 3 次候选渲染。
8. 每个候选重新计算 Gaussian 激活、Selector hard selection、TF/insideMask 和 compact indices；固定同一份选择随机数，允许选中集合和数量变化。
9. 所有被检查历史 MSE 分别不增加才提交；只允许数值误差 `1e-7 + 1e-5 * abs(baseline_mse)`。不能用改善视角抵消另一个视角退化。
10. 合格非零候选一次性提交参数和正常梯度产生的 Adam state。NOOP 不提交 moments，不计成功学习。有限候选全部失败，或有限投影不可行时，记录一次 `would_cancel`；开启 fallback 后，额外检查并提交 `before + cancel_fallback_scale * original_adam_delta`，不再要求历史质量门控通过。不使用最后投影方向，不缩放梯度；Adam moments/step 只提交一次。非有限梯度、历史损失、参数或候选不允许 fallback。

启用 fallback 后，这是“冲突更新衰减＋远端补偿”，不再是每步严格保护历史质量。0.2 位移只能减缓风险，不能保证历史损失增加也缩小为 0.2；Replay 的实际补偿能力需用同版消融验证。

## 冲突源与远端联合 Replay

- 一个最终有限 `would_cancel` 只归因给正在训练的当前 KF，按原始次数计数，不按被检查历史视角计数，也不按候选数量重复计数。
- 累积 1000 次后，按次数降序选前两个有效源 KF，平局按 KF ID。每个源独立取空间距离最远的 32 个候选，再从该池用 farthest-point sampling 选 4 个分散视角。两组交错去重后最多 8 个，不因重复补足；排除源 KF、Recent Window、当前 KF、已排队及无效视角。
- 相机距离使用当前 Twc：归一化平移＋归一化旋转角，权重复用 `HistoricalOutlier.translation_weight/rotation_weight`。无有效候选也结束当前计数窗口，不无限堆积。
- FIFO 容量 16，只存 KF ID 和窗口 ID，不缓存 Gaussian 索引、梯度或教师。消费时重新取当前视角/地图；无效、已进入 Recent Window 的任务跳过。地图整体重置时清空队列及窗口；append/prune 不清空。
- 复用 `Selector.replay_enabled/replay_interval/replay_num_frames`：正常 mapping 每 300 次，在当前正常训练前消费最多 2 个有效任务，空队列不做随机 Replay。新触发任务等待后续调度，关闭时不强制排空、不触发不足 1000 的尾窗。
- 每个任务执行原图 Full loss 和正常 LoD loss 两次反传，共享 Gaussian Adam 一次事务提交；GS、Selector、TF 均可更新，Replay 不遵循正常 LoD 的 detach 开关。数量 loss 同正常步骤采用 soft expected count，GI 仅只读使用当前 Recent Window teacher。
- 不执行 densify/prune、opacity reset、SH 调度、KF 使用次数/金字塔消耗/自适应损失统计、GI 更新、seen-count、正常 iteration/ratio/LR 时钟或递归 Guard。独立 Replay ratio cursor 轮换；成功提交后才允许更新时间戳 `last_elastic_supervised_iter_`。
- 每任务梯度在开始和结束清空。共享 Adam 内部 step/moments 正常推进，但正常训练 LR 时钟不推进。Replay 在当前自身视角和最近一次 Guard 检查视角记录固定随机规则下的更新前/提交后质量，只作为诊断，不设置新的质量拒绝条件；数值异常仍禁止提交。日志单独记录诊断渲染次数和 host wall 分阶段耗时；这是主机延迟，不是独立 GPU kernel 时间。

## 数量约束

数量不进入 GradientGuard 的目标、投影或提交门控，不要求所有 ratio 每步的数量偏差都不恶化，不要求前后选中数量相同。

正常 LoD 数量 loss 使用同一次 Selector forward 的 `softmax(logits)` 概率计算 expected mature count，不额外跑 MLP。Gumbel hard selection 的概率律与温度无关，数量监督不依赖随机 hard mask。既有 loss 权重、Mature/Hard/Cold 65/25/10 配额及容量借用不变，不新增 Top-K。

Replay 复用正常 soft-count loss 和共享 Adam，但不参与预热/正常学习率计数。CD/DSD 不改动。预算只作为监控/长期收敛指标，不能保证每步精确数量。

## 配置

```yaml
GradientConstraint.enable: 1
GradientConstraint.tf_warmup_steps: 1000
GradientConstraint.check_interval: 1
GradientConstraint.memory_views_per_check: 1
GradientConstraint.memory_ratios_per_check: 1
GradientConstraint.max_candidate_checks: 3
GradientConstraint.log_interval: 100
GradientConstraint.cancel_fallback_enable: 1
GradientConstraint.cancel_fallback_scale: 0.2
Selector.replay_enabled: 1
Selector.replay_interval: 300
Selector.replay_num_frames: 2
ConflictReplay.trigger_cancel_count: 1000
ConflictReplay.source_view_count: 2
ConflictReplay.views_per_source: 4
ConflictReplay.remote_candidate_count: 32
ConflictReplay.queue_capacity: 16
DriftStability.enable: 0
ChangeDistill.enable: 1
```

删除 `GradientConstraint.budget_relative_tolerance` 的读取与保存，不影响恢复 DSD 的同名后缀设置。fast 开启，其他 Selector 配置仍默认关闭。

检查间隔按正常 optimizer 尝试次数，拒绝也推进；间隔大于 1 的未检查步没有保护。无历史视角打印 `UNPROTECTED` 并正常更新。两个新功能可独立关闭；关闭 fallback 则保留原 CANCEL，关闭 Replay 则不排队。fast 显式开启两项，另外两份配置默认关闭。

## 日志

`[GradientGuard]` 区分 `WARMUP/UNPROTECTED/ACCEPT/SHRINK/FALLBACK/NOOP/CANCEL`。记录正常 TF 更新进度、Gaussian LR step、候选类型、独立 MSE 的 `before/raw/last_checked/committed`、参数组位移范数比、累计状态与 `would_cancel` 及分阶段耗时。所有浮点四位小数，无科学计数法。

CANCEL 没有 `committed` 损失，`last_checked` 不能解释为当前地图；FALLBACK 的 `committed` 来自实际衰减提交。`[ConflictReplay SELECT/UPDATE/SKIP/SUMMARY]` 记录源次数、远端视角、入队/去重/容量、Full/LoD loss、更新前后 MSE、独立计数与未排空队列/尾窗。新增统计另保存在 `transform_state.json`；正常 `transform_step/selector_step` 字段语义不改。

## 验证

启用 `CARTGS_BUILD_TESTS`：

```bash
ctest --test-dir build-cuda118-sm86-gcc9 --output-on-failure \
  -R 'constrained_adam|gradient_constraint|historical_gradient_render|conflict_replay|joint_replay'
```

- Solver：近反向、重复/秩亏、零行、非有限、迭代上限及 200 组随机 PSD Gram。
- Clock：999/1000 启用边界、TF 未初始化、Gaussian-only、拒绝后的 LR 计时。
- Guard：原始候选快速接受、质量改善允许数量偏差增加、投影保留有效学习、缩步/拒绝回滚、零位移不提交、零 LR/None gradient、非有限值。
- Fallback：0.2 原始位移、三次失败后单次提交、moments 不缩放、grad=None 不更新、原始和 fallback 数值失败都不提交。
- Conflict Replay CPU：1000 触发、源次数排序、独立选 4、远端池/FPS、去重、不补足、容量、失效视角及窗口清空。
- Joint Replay CUDA：直接调用 Mapper Replay，确认 LoD 梯度进入 GS、GS/Selector/TF 联合提交一次、结束梯度清空、正常时钟/KF/GI/seen/topology 不改变。
- Historical render：固定随机规则、正常 expected count 复用的梯度与温度无关性、候选输入一致性、scratch AABB、CUDA Training/Inference 一致性。
- Checkpoint benchmark 只断言历史质量，全部 ratio 预算单独记录 `gate=OFF`，不再以数量恶化判失败。

本次本机不做 CUDA/LibTorch 构建。已用 MinGW C++17、`-Wall -Wextra -Werror` 独立编译运行 Conflict Replay、Solver、Clock 三组 CPU 测试，全部通过；三份配置的 YAML 解析、重复键检查、新增键读入/导出闭环、功能默认值与 `git diff --check` 通过。静态检查确认 Replay 单次事务提交、不 detach GS、不调用 mapping 生命周期或 Guard，并清除旧随机 Replay 引用。Fallback/Joint Replay CUDA 测试已加入但未在本机执行，仍需目标环境构建运行；静态检查不等于 CUDA 行为验证。前一版的运行数据见下文，不能作为本次修订的验证。

完整 room 应与同版关闭 Guard 对照，检查非零有效学习、当前视角质量、被检查历史质量、预算收敛、在线吞吐和尾优化占比，而不只看取消率。

## 前一版验证记录（2026-10-09，不代表本次修订的运行结果）

在 mi3090 的 `zwj` 环境、独立目录 `/tmp/cartgs-gradient-guard.buOTzJ` 编译/运行，未覆盖服务器项目或结果：Adam parity、约束/回退 CPU 测试、历史渲染 CPU/CUDA 测试全部退出 0；新 mapper 代码的目标环境编译检查退出 0。新增模块链接现有 renderer 库测试，不等同于整项目完整重新构建。

使用 `results/newestfast/tum_rgbd/fr1_room/16400_shutdown/ply` 的 66884 个 Gaussian（60065 mature）及真实 room RGB，执行四个 ratio 的四步短程对照：

| 模式 | 历史 Full PSNR 初始 | 四步后 | 提交情况 |
|---|---:|---:|---|
| 无约束 | 19.6252 | 19.5539 | 四步正常提交 |
| 受约束 | 19.6252 | 19.6273 | 两步缩步接受（0.25、0.5），两步取消 |

受约束提交后的实际历史 Full/LoD MSE 与所有 ratio 预算通过断言。这里只证明该 checkpoint/视角/随机选择下的有限步预防行为；完整在线 room 吞吐、长期质量与当前视角 plasticity 尚未验证。测试阶段先不据此调整质量容差或缩短历史检查覆盖范围。

Adam 预览运算顺序核验来源：[LibTorch 2.3.1 Adam 源码](https://github.com/pytorch/pytorch/blob/v2.3.1/torch/csrc/api/src/optim/adam.cpp)。
