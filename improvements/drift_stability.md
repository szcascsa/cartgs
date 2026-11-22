# Historical / Drift Stability Distill

历史退化后的恢复 DSD 保留为独立可选功能；Recent Window GI 只监督近期有观测证据的选择质量。
Replay 保持关闭，Change CD 的快照、transport、loss、optimizer 不变。

## 独立的恢复开关

复用已有配置 `DriftStability.enable`，不新增同义开关。代码默认值及
`selector_config.yaml`、`slower.yaml`、`fast.yaml` 均默认设为 `0`；设为 `1`
可恢复原有历史检测与恢复流程，下文描述的是启用后的行为。

关闭时不创建恢复 Manager，不维护其 Teacher/EMA/Horizon，不执行历史 Probe、
QUALITY 渲染或临时恢复 Adam；启动日志明确打印恢复开关。
正常 Gaussian/Selector/TF 训练、Recent Window GI、Dynamic AABB 和 Change CD 不受影响。
几何及 AABB 变化的 `change_version` 仍独立推进，不依赖恢复功能是否启用。

后续在线更新预防机制使用独立配置与 optimizer 接入点，不复用此开关。
当前尚未实现 Gradient Constraint，因此关闭恢复不代表已经获得在线预防能力。

## GI 监督边界

RecentWindow 为每个关键帧保存独立的 observed mask，以有效 contribution_count > 0
定义已观测；窗口内使用 observation_count 合并。不通过浮点 GI 是否大于零推断
覆盖范围。换入换出、重渲染、append、prune、reset 和几何失效均同步该状态。

GI loss 和分位数阈值只使用 mature ∩ observed；窗口未观测的历史点是 unknown，
不作为负标签。全局 ratio loss、mature/hard/cold 配额和容量借用不变。
无有效贡献的关键帧不提供监督；EMA decay=0 时只使用最新关键帧的观测。

## 触发与 QUALITY

1. 在完整 mapping iteration 和 AABB 处理后，到达 probe interval 才检查。
2. 冻结当前 Gaussian 输入，在历史 outlier 的 visible_mature_indices 上计算
   Selector probability L1 和 TF XYZ/log-scale/rotation drift。
3. 连续超阈值的模块才请求触发 QUALITY。Selector 对照固定当前 TF，只替换
   Selector；TF 对照固定当前 Selector，只替换 TF。两个模块分别建立坏视角集合，
   不使用旧 Selector + 旧 TF 组合的优势给两个模块同时归因。
4. Teacher/Student 使用全部相同当前 Gaussian 输入、相同 Hard/Cold 准入。
   Mature 使用同一份逐行 uniform 的 Bernoulli hard selection（u < p），
   不使用 Top-K，也不再使用 soft opacity。不同 ratio、模块替换、训练前后及
   EMA 候选复用这份 uniform；它随 probe 重建，不推进正常训练 RNG。
   采样概率对应 Gumbel hard argmax 的分布，与 relaxation temperature 无关。
   Function drift 和蒸馏 target 仍使用配置温度下的确定性概率。
5. Selector 预算资格检查当前全体 mature 的期望参与数：
   abs(sum(p) - B_mature) / max(B_mature, 1)。
   budget_relative_tolerance 默认 0.25；缺失或非有限预算证据直接拒绝。
   B_mature 来自当前人口的预算分配器，不使用历史人口或采样子集估算。
   TF 对照使用同一个当前 Selector，因此不会受超预算旧 Selector 影响。
6. 每个模块独立要求合格预算、至少 min_bad_views 个坏视角、
   加权正向 PSNR regression 超过 psnr_regression_threshold。
   若只有旧组合好、单模块替换均无益，第一版不会强行分别触发两个模块。

## 训练和提交

只对该模块坏视角 visible mature 的 union 蒸馏；每个 ratio 最多 max_gaussians。
当前 Gaussian 全部 detach，任何 master 参数、正常 optimizer 和训练时钟都不更新。

Selector 优化 soft-target BCEWithLogits，使用二分类 keep/drop logit 差；
probability L1 继续作为 drift 指标。BCE 包含 Teacher entropy，不能用原 L1 阈值
判断 BCE 是否收敛到零。TF 保留原输出 loss，并打印三个分量。

每个模块、每个事件只创建一个临时 Adam。每 epoch 交错各 ratio 的 batch，
不再依次训完某个 ratio 的全部 epoch。学习率、epoch 和人口上限保持原配置。
Teacher targets 复用当前 probe 的 frozen cache；Student 反传永远重新 forward。

更新前冻结全部历史视角、全部 ratio 的实际 Student QUALITY 基线，并追加一个
仍存在于 scene 的最新近期关键帧作为 guard。近期 guard 只用于验收，绝不进入
function drift、触发计数或 distill population。

Selector 更新后若提交，TF 必须在新的当前 Selector 下重新检查自己的质量优势
和坏视角集合。两模块都检查相对各自更新前及整场事件开始前的质量，避免每个
模块的小损伤叠加超过总容差。

每个模块独立分类：

- IMPROVED：目标视角加权 PSNR 增益 > 0.0001 dB，function drift 下降，安全检查通过，
  但尚未达到绝对恢复阈值。保留更新，不立即替换未完成的 Teacher。
- RECOVERED：满足 IMPROVED，且触发人口 drift 回到阈值内、原坏视角距离该模块
  Teacher 的 PSNR 差距回到容差内。可以申请 EMA 候选。
- NO_PROGRESS：没有可测质量改善或 drift 没下降。恢复更新前参数。
- REJECTED：非有限输出、视角损伤或预算失控。恢复更新前参数。

安全检查覆盖未触发 ratio、正常历史视角和近期 guard。更新不得突破预算容差；
某个未触发 ratio 在更新前就不合格时，至少不能进一步恶化其相对预算误差。
回滚和 optimizer 更新都会失效相应 Student 输出、prepared inputs 和 PSNR 缓存。
正常 optimizer 的状态从未被临时 Adam 修改；回滚仅恢复本模块参数。

## Reference / EMA / Horizon

- min_age：开始常规质量确认的最小年龄，之前不额外渲染来累积刷新资格。
- horizon：软等待窗口，不跳过 probe，不清空连续 drift 计数。
- non_regression_refresh_count：连续预算合格、质量不退化且加权质量不下降的次数。
  不再要求 function drift 小；有益的函数变化可以取得更新资格。
- 部分改善或拒绝的 DSD 不以“执行过训练”为理由更新 Teacher。
- EMA candidate = decay * old + (1-decay) * current；验证实际候选的预算、
  当前质量和合格旧参考质量之后才发布。所有模块比较固定另一模块的当前网络。
- 如果旧 Selector 超预算，不把它的高 PSNR 当作否决条件，也不混入它的权重。
  当前网络通过连续预算/质量检查后直接 clone，日志 REBASE_PENDING。
- max_age：完整重建时钟，不随 EMA 延长。到期立即退休旧目标、clone 当前模块，
  日志 expired_pending_qualification；不再对已过期 Teacher 做最后一次蒸馏。
  新快照仍需预算、单模块质量优势及连续 drift 证据才能教学，重建不表示恢复成功。
- 没有历史视角时不能发布常规 EMA；硬年龄仍生效。
- Geometry CD 后刷新两个模块，AABB-only 后仅刷新 TF；同一个结构变化 cycle 跳过
  DSD，后续 probe 正常继续。EMA 不跨结构/域版本。

## 性能与日志

保留单 probe 的输出/Teacher target/PSNR/prepared input 缓存，不恢复跨 probe 的
历史 visibility 缓存。历史可见性仍每次按当前几何重新查询。坏视角 union 继续
使用已有稠密 bool mask / 稀疏 sort 去重策略。

SLAM 在线 QUALITY 宽高各减半，结束后全分辨率；单 probe 内保持相同分辨率。
QUALITY 使用 inference-only rasterizer，仍接收同一完整 Gaussian population，
以 hard opacity gate 表达选择；本修改不额外引入 QUALITY compact renderer。

日志均固定四位小数，不使用科学计数法：
- HistoricalProbe QUALITY：独立模块 PSNR、预算资格、mature 期望/目标/实际数量。
- HistoricalProbe：drift、连续计数、各模块坏视角数和触发理由。
- HistoricalDistill SELECTOR/TF：epoch=0..N、loss、delta、steps；Selector 附 L1，
  TF 附 XYZ/scale/rotation。BCE 不与 L1 混称。
- HistoricalDistill RESULT：IMPROVED/RECOVERED/NO_PROGRESS/REJECTED、质量增益、
  剩余差距、drift 和 rollback。
- KEEP/EMA_UPDATE/REBASE_PENDING/HARD_RESET：参考更新或保留原因。
- COST：实际渲染次数、输出复用次数、处理行数和分辨率。

## 验证

新增/更新的 C++ 测试覆盖：GI unknown 零梯度、观测窗口及 topology 生命周期、
预算拒绝与缺失证据、单模块归因、Selector BCE 饱和梯度、ratio 交错训练、
部分/完整恢复、无进展/损伤/预算失控回滚、未触发 ratio 和近期 guard、
缓存回滚失效、质量驱动 EMA、候选拒绝、预算重建、硬年龄及结构变化范围。

目标 CUDA/LibTorch 环境可运行：
```text
ctest -R "gi_teacher|recent_window_importance|drift_stability|historical_probe_cache" --output-on-failure
```

本机无 CUDA，不进行工程构建。静态检查不能替代上述测试及真实序列验证。
实际效果应在相同在线计算预算下比较 DSD 关闭/修复前/修复后，并检查后续正常训练
是否保留恢复收益；不能只用 recovery 日志或刚结束时的 loss 判断防遗忘有效性。
