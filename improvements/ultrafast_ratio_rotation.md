# Ultrafast 在线 LoD Ratio-Rotation 重构报告

> 本文为历史重构记录。当前配置仅使用 `Training.ultra_light_mode`，默认 mode1；旧 enabled/interval 入口已移除。当前 mode1/2 语义见 [训练模式](ultra_light_training.md)。

日期：2026-10-12。分支：`remove-full-render`。

## 结论与验证边界

已实现固定 `0.01 → 0.05 → 0.10 → 0.15 → Full` 的在线正常 Mapping 轮转。LoD 与 Full 正常训练互斥，LoD 保留 Master Gaussian 梯度；KF 质量统计和 Master topology 统计仅由 Full 更新。Ultrafast 跳过所有正常尾优化循环。

按用户确认，ultrafast 就是现有 ultra_light 模式。本次直接以 `Training.ultra_light_enabled: 1` 启用新轮转，删除旧“每 20 步增强”的调度实现，不新增模式开关、独立配置或 legacy 分支。`Training.ultra_light_interval` 仅保留读写兼容性，不再控制训练。现有 `fast.yaml` 和 `selector_config.yaml` 的数值不变，默认仍为关闭；启用时需要 `Selector.enabled: 1`。关闭 ultra_light 时，其他模式的 Full + LoD 和尾优化行为保留。

本机完整构建在 CMake 配置阶段失败：`Failed to find nvcc`。以下结论区分源码检查、独立单元测试与尚未执行的 CUDA 测试；不能据此声称已完成在线 CUDA 性能或模型质量验收。没有启动长训练、提交或推送 Git。

按用户最新要求，服务器负责正式测试；本地只完成编码和简单源码审阅，不再尝试构建或执行测试。表中的 PASS 是该要求之前已经执行的轻量检查，新增 CUDA 集成测试及此后计时同步修改尚未运行。

## A. 旧调用链

以下来自修改前的真实 `GaussianMapper` 源码，而非按任务描述推断：

```text
run()
  → trainForOneIteration()
      → increaseIteration(1)
      → useOneRandomSlidingWindowKeyframe()             每次独立选 KF
      → processConflictReplay()                        仅满足原触发条件时
      → trainMappingIteration()
          → updateGlobalGIBank()                       增强步
          → protection_.advance(mapping_iter_ + 1)
          → GaussianRenderer::render(Master Gaussian)   Full
          → refreshRecentWindowContribution()          原采集条件
          → protection_.beginFullMapping(visibility)
          → Full L1/SSIM + opacity regularization
          → full_loss.backward()
          → accumulateLodGradients()                   增强步
              → Protection / ElasticBudgetManager
              → initializeTransformFieldIfNeeded()
              → ensureSharedMappingOptimizer()
              → selector_network_->forwardHard()
              → transform_field_->applyResidual()      满足原 readiness/mask
              → GaussianRenderer::render(compact LoD)
              → 现有 LoD photo/ratio/GI/TF regularization loss
              → selector_loss.backward()
          → kfs_loss_ 与 ema_loss_for_log_ 写入 Full loss
          → max_radii2D_ / addDensificationStats()       Full 的 screen-space 梯度
          → densifyAndPrune() / resetOpacity()           原 iteration 条件
          → stepWithHistoricalGuard()
              → Gaussian 共享 Adam optimizer
          → optimizer_->zero_grad(true)
          → protection_.finishFullMapping()
      → ++mapping_iter_
      → GI refresh / AABB / DriftStability              原 mapping 时钟
```

重要顺序：旧实现先 densify/prune，再提交共享 optimizer。Topology 会重建 Gaussian 参数而不保留当步梯度，maintenance iteration 因此可能不实际更新 Gaussian。Ultrafast 改为先提交 optimizer，再按 Full 条件处理 topology，保证 Full 校准梯度不会因此丢失；其他模式保留原顺序。Selector/TF 已接入 Gaussian optimizer 的额外 param groups，正常训练没有单独调用它们各自 optimizer 的 `step()`。`gaussians_->optimizer_` 的接口类型是 `torch::optim::Adam`，本路径调用其普通 `step()`，没有把非 Gaussian 参数交给 `SparseGaussianAdam::step(visibility, N)`。

旧 `ultra_light` 的普通步只有 Full，增强步为 Full + LoD，默认每 20 次 Mapping 一次增强；不开 ultra_light 时每次都走增强路径。在线结束后旧 ultra_light 恢复每步 Full + LoD，forced-tail、max-mapping 补齐及 legacy densification/keep-training 尾循环原本均存在。本次替换 ultra_light 本身，不保留这条旧调度。

## B. 新 Scheduler 与文件

| 文件 / 函数 | 实际职责 |
|---|---|
| `include/ultrafast_rotation.h` / `UltrafastRotation` | 固定 ratio、phase、独立正常训练计数、在线结束边界、阶段频率和时间汇总 |
| `src/gaussian_mapper.cpp` / `trainForOneIteration()` | 保留每次 KF 选择，拒绝在线结束后的新训练，保留辅助模块 mapping 时钟 |
| `trainMappingIteration()` | 单一正常 Render 分支、共享 optimizer、Full 专属 KF/topology 更新、完成一个 slot |
| `accumulateLodGradients()` | 复用原 LoD loss、Protection、Selector、TF；本模式强制不 detach Gaussian |
| `run()` / `normalMappingLimit()` | 在线结束即停止；max_mapping_iterations 仅为安全上限；本模式不进入尾循环 |
| `reportUltrafastRotation()` / `savePly()` | 日志和 `ultrafast_rotation_state.json`，并检查频率平衡、render 数及零 tail 不变量 |
| `include/ultrafast_training_timer.h` | 可复用 CUDA Events、同步后 elapsed time 和训练 wall time |
| `readSelectorConfig()` / `saveSelectorConfig()` | 复用 `Training.ultra_light_enabled` 模式入口；兼容导出旧 interval，不使用旧 cadence |
| `python/eval_lod.py` / `python/run.py` | ultrafast 的四个 LoD ratio 与同一 shutdown checkpoint 的 Full 评估 |

只有正常 Mapping 的 optimizer 决策完成后调用 `UltrafastRotation::complete()`。Guard 的取消仍是一个完成的正常训练决策，消费本 slot，但日志记录实际未提交的 optimizer；CD、Replay、GI、viewer、evaluation 和保存均不调用 `complete()`。

## C. 五种 Phase 行为

| Phase | Gaussian | Selector | TF | kfs_loss / Full EMA | Densify / Prune / Statistics |
|---|---|---|---|---|---|
| 0.01 | 是 | 是，存在有效梯度时 | 是，原启动条件满足时 | 不更新 | 不执行、不累计 |
| 0.05 | 是 | 是，存在有效梯度时 | 是，原启动条件满足时 | 不更新 | 不执行、不累计 |
| 0.10 | 是 | 是，存在有效梯度时 | 是，原启动条件满足时 | 不更新 | 不执行、不累计 |
| 0.15 | 是 | 是，存在有效梯度时 | 是，原启动条件满足时 | 不更新 | 不执行、不累计 |
| Full | 是 | 不 step | 不进行正常 TF step | 更新一次 | 按原条件执行、正常累计 |

共享 Adam 的 Full phase 使用 `zero_grad(true)` 清除辅助梯度，确保未定义的梯度使辅助参数和 Adam moments 均跳过。LoD 保留 Gaussian 梯度，即使旧 `Selector.detach_gaussian_parameters` 被设为 1，本模式仍不 detach。

TF readiness、`start_iter`、mature-only、outside-identity、Protection admission 及现有 loss 均未重新设计。无 mature Gaussian、无被选中的有效梯度、零学习率或 Guard 取消时，不能保证该模块参数发生变化；逐步日志记录真实提交情况，不能将允许更新等同于每步必然改变参数。

本模式不缓存整个 cycle 的 KF。adaptive sampling 继续读取 `kfs_loss_`；仅在尚未得到任何 Full loss 时跳过原代码会越界的空质量 ranking，基本 KF pool/shuffle/使用次数逻辑保持原样。

densify 与 opacity-reset 的旧 `iteration_` 截止点若落在 LoD phase，会延至下一 Full phase执行；不改成 cycle 时钟。在 Full 之前不会累计 `xyz_gradient_accum_`、`denom_` 或 `max_radii2D_`。超出原 densification 窗口或在线结束时不会补执行。

## D. 在线结束与 Tail

```text
输入示例调用 System::Shutdown()
  → ORB-SLAM3 的 mbShutDown 置 true
  → ultrafast 正常训练入口拒绝新 Mapping
  → 冻结 online_end_mapping_iteration / online_end_rotation_step
  → run() 跳过 forced-tail、max_mapping 补齐和 legacy tail 三条路径
  → 保存最终状态
```

停止检查位于正常训练入口、选 KF 后、Replay 后及在线 Mapping 外循环。已经在在线状态下开始的一个正常 iteration 可以完成，随后记录结束边界；不会再发起下一个正常 iteration。`tail_training_steps` 是结束边界后的 rotation-step 增量，并要求为 0。

如果安全上限先于输入结束触发，则保存提前停止的状态，`online_ended=0`；此时未声称已观测到真正的 SLAM 结束边界。正常输入结束且未提前触发上限时，才输出冻结后的 online-end 两个计数。

独立 scheduler 模拟测试得到 `tail_training_steps=0`；真实 CUDA Mapper 的停止测试已经加入，但本机尚未执行。不能把模拟结果称为真实 SLAM 序列的零 tail 运行证据。

## E. Tests 与命令

以下命令在仓库根目录执行，路径按 Windows PowerShell 表示。

| 测试 / 检查 | 结果 | 说明 |
|---|---|---|
| 固定 ratio sequence，rotation_step 0..11 | PASS | 0,1,2,3,Full,0,1,2,3,Full,0,1 |
| 阶段计数平衡、render 总数、P95 | PASS | 12 个 slot 的正常 Full/LoD 数为 2/10，阶段次数最大差为 1 |
| 在线结束、安全 cap、零 tail | PASS | 独立 scheduler 模拟，不是实际 SLAM 运行 |
| LoD 截止点延至 Full 的 topology 调度 | PASS | 使用真实 `DeferredMappingTask` |
| ultrafast 的 4 LoD + 匹配 checkpoint Full 评估命令 | PASS | 2 个 Python 测试之一；mock 指标仅验证调度，不是质量实测 |
| 其他模式仍仅评估 4 LoD | PASS | 第二个 Python 测试 |
| 旧 forced-tail schedule 回归 | PASS | 原独立测试编译并执行，退出码 0 |
| Python AST / git diff --check | PASS | 三个 Python 文件语法及 diff 空白检查 |
| 现有两份配置数值不变 | PASS | 仅更新注释；无新模式键或独立配置 |
| 完整 CMake 配置 / 构建 | FAIL，环境阻塞 | `Failed to find nvcc`，未进入项目 C++/CUDA 编译 |
| 真实 KF 独立性、kfs_loss、topology、optimizer、TF readiness、零 tail | 未执行 | 已加入 `ultrafast_mapping_cuda`，需要目标 CUDA 11.8 / sm_86 环境 |
| 更新后的 ultra_light 共享 Adam 状态测试 | 未执行 | `ultra_light_schedule` 改为轮转/Full 冻结断言，需要项目 LibTorch 构建 |

```powershell
g++ -std=c++17 -Wall -Wextra -pedantic -I . tests/ultrafast_rotation_test.cpp -o C:/Users/17739/AppData/Local/Temp/cartgs_ultrafast_rotation_test.exe
& C:/Users/17739/AppData/Local/Temp/cartgs_ultrafast_rotation_test.exe
python tests/ultrafast_eval_test.py
g++ -std=c++17 -Wall -Wextra -pedantic -I . tests/tail_optimization_schedule_test.cpp -o C:/Users/17739/AppData/Local/Temp/cartgs_tail_regression_test.exe
& C:/Users/17739/AppData/Local/Temp/cartgs_tail_regression_test.exe
git diff --check
cmake -S . -B C:/Users/17739/AppData/Local/Temp/cartgs-ultrafast-configure -G "MinGW Makefiles" -DCARTGS_BUILD_TESTS=ON
```

CUDA 测试不是玩具 optimizer：它调用生产 `useOneRandomSlidingWindowKeyframe()` 与 `trainMappingIteration()`，检查 KF1/KF8/KF3/KF11/KF2、Master/Selector/TF 参数变化、Full 辅助参数及 Adam moments 不变、空 LoD 无 Full fallback、Full topology deadline，以及生产入口拒绝模拟在线结束后的训练。该测试不提供完整输入序列的 tracking FPS 或质量结论。

在已有正确配置的目标构建目录中，应执行：

```text
cmake --build <build-dir> --target ultrafast_rotation_test ultrafast_mapping_test ultra_light_schedule_test tail_optimization_schedule_test
ctest --test-dir <build-dir> -R 'ultrafast_rotation|ultrafast_mapping_cuda|ultrafast_eval|ultra_light_schedule|tail_optimization_schedule' --output-on-failure
```

Python 语法检查命令：

```powershell
python -c "import ast,pathlib; files=['python/eval_lod.py','python/run.py','tests/ultrafast_eval_test.py']; [ast.parse(pathlib.Path(f).read_text(encoding='utf-8'),filename=f) for f in files]; print('PASS: Python AST, 3 files')"
```

配置比对使用 `git show HEAD:<path>` 与 `Get-Content`，分别提取非注释 YAML 行并比较，确认两份现有配置只改注释、不改数值。这不是 OpenCV FileStorage 运行验证。

## F. Performance 与质量

没有运行新旧短序列对比，本次无法报告下列实测数值。

| 指标 | 本次结果 | 已实现输出 |
|---|---|---|
| 0.01 / 0.05 / 0.10 / 0.15 / Full step ms | 未测量 | `lod_001_mean_ms` 等与 `full_mean_ms` |
| 各 phase GPU Render ms | 未测量 | 对应 `*_render_mean_ms` |
| mean / P95 mapping training ms | 未测量 | `mean_mapping_training_ms` / `p95_mapping_training_ms` |
| 实际在线 Full / LoD render count | 未运行 | `full_render_count` / `lod_render_count` |
| 实际 online_end mapping / rotation | 未运行 | `online_end_mapping_iteration` / `online_end_rotation_step` |
| 实际在线 Selector / TF 更新数 | 未运行 | `selector_online_lod_update_count` / `tf_online_lod_update_count` |
| 序列总耗时、tracking FPS、Gaussian 数量变化 | 未运行 | 现有 example FPS 和逐步 `post_topology_gaussians` 日志 |
| 四种 ratio 与 Full 的 PSNR / SSIM / LPIPS | 未运行 | ultrafast `eval_lod.py` 汇总至 `summary.json` / `summary.csv` |

CUDA Render 时间通过 current stream 的 Event start/end 统计，读取 elapsed time 前同步完成事件。原 rasterizer 部分 kernel 使用默认 stream，因此训练和 Render 边界还使用 device synchronize，确保两种 stream 的工作都完成。阶段训练 ms 是隔离此前 pending GPU work 后、直到本阶段 GPU 完成的 host wall time，含本函数中的 host、GI bank、Guard 和 topology 工作；外围 Replay、mapping operations、后续 GI refresh / AABB / DSD 和 checkpoint evaluation 不属于这个阶段训练计时。Event 时间包含 Render 调用的 host launch gaps 和同步开销，不能称为纯 kernel 时间。事件复用，但同步计时本身会限制流水重叠，性能对比需采用一致的计时方式。

每 5 个正常 Mapping 在结构上只有 4 LoD + 1 Full。计数仅统计正常 Mapping Render；现有 GI refresh、historical Guard、Replay、viewer 或 evaluation 的额外渲染不计入其中，不能把正常 render 计数直接宣称为整个程序的总 Render 数。

日志的 `densify_triggered/prune_triggered` 表示调用了现有 `densifyAndPrune()` 例程；不保证当次实际新增或删除 Gaussian。`post_topology_gaussians` 提供当次处理后的数量。

评估入口保留现有参数：`python python/eval_lod.py <result-path> <gt-path> --selector <明确的在线结束 checkpoint/selector.pt>`。ultrafast 自动追加 Full，使用同一 shutdown model；不会因另一个较新模型目录而评估错 Master Map。Full 不加载 Selector mask 或 TF transform，输出 active/total/actual-target 统计。

## G. Known Issues 与剩余验收

- Full Master Map 是否退化尚无实测证据；必须在目标机器完成同一数据、同一配置的短序列对比和质量评估。不能用 tail optimization 弥补下降。
- Protection 的 `optimized_count` 和 unique-KF evidence 原本由 `beginFullMapping/finishFullMapping` 累计，现仅在 Full phase 累计。未引入 LoD 证据替代算法，成熟速度可能变慢，并进一步推迟 TF 初始化或实际应用。已有首次初始化、首次应用和实际更新数日志用于定位。
- Full 可比较的 KF loss 变得稀疏，部分 KF 可能长期没有 Full quality observation。未重新设计 adaptive sampling；解决了第一轮没有任何 Full loss 时的空 ranking 越界。
- Full densification 样本数降低到约 1/5。densify/prune 的旧周期仍按原 iteration 计算，若截止点与 Full 相位对齐，实际决策次数未必降低 5 倍；Gaussian 最终数量影响需要实测。
- Ultrafast 改变了 optimizer 与 topology 的提交顺序，属于此次 Full 校准语义所需的调整。screen-space 统计仍来自该 Full Render；Protection 使用已有 prune 后可见索引重映射。新增 CUDA 测试检查发生 Full topology 时 Gaussian color 仍更新，但尚无目标机运行证据。
- Recent GI 从正常 Full Render 取得的 contribution 观测也变得稀疏。Streaming GI 的 incremental bank、refresh interval 和 topology synchronization 算法与 mapping 时钟保持原样，没有额外 Full render 补观测。
- KF 输入整合、点云追加和 loop-closure/BA 几何传输仍属于原 Mapping operation，未改成由 phase 控制；Full-only 限制的是正常训练里的 densify/clone/split/prune 决策，不是禁止在线输入新增地图点。
- Guard 的原取消/投影机制仍可能拒绝或缩小正常更新；允许 G/S/TF 更新不代表每个 iteration 都成功提交。Full 没有 S/TF 当前梯度，现有 Adam preview 只包含定义了梯度的参数。
- 本机未完成项目 C++/CUDA 编译，新增 CUDA 集成测试也未编译运行。最终验收仍缺真实 KF/optimizer/topology/零-tail 执行证据、在线更新次数、GPU 性能和 Full/LoD 质量数据。
