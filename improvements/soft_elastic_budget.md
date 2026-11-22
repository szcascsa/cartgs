# Evidence-Aware Gaussian Protection

正式算法只保留一套 Protection。状态由正常 Mapping 年龄、有效优化次数和独立 KF support 推导，GI validity 不参与成熟判定。源码入口为 `include/gaussian_protection.h`，预算入口为 `include/elastic_budget.h`。

## 证据与状态

`M=min(clamp(age/A0),clamp(opt/U0),clamp(KF/K0))`；三项同时达到阈值才是 MATURE。年龄不足为 WARMUP；年龄足够但证据不足为 UNDER_OBSERVED；后者达到 stale_age 为 STALE。STALE 仅用于诊断，继续参与 Full Mapping 和同一 Protection admission，没有任何清理、损失或训练触发行为。

优化证据仅由正常 `trainMappingIteration` 在 Full visibility 有效、正常 Gaussian optimizer 确认 commit 后更新。Replay、ChangeDistill、历史 Distill、GI、Selector-only、TF-only 和 Viewer 没有更新入口。超轻量普通步也属于正常 Full Mapping。

KF IDs 使用 Gaussian device 上的 int64 `[N,K0]` tensor，只记录前 K0 个不同 KF，空位为 -1；重复 KF 只增加优化次数。CUDA kernel 将 KF 扫描、支持写入、优化计数及相关状态转换合并执行。CPU 单元测试使用同义 Tensor 实现。

## 动态预算

`B=eN`，`P=1-M`，`D=sum(P)`，`Bp=min(D,rho*B)`。同一 Gaussian 的 cached uniform 用于 `u<q`，固定证据下不同 ratio 的 Protected admission 嵌套。

由于 Bp 不超过 D，`lambda=Bp/D<=1`，且 P 不超过 1，`q=lambda*P` 即为精确解，不需要二分。D=0 时概率为 0。

采样后使用实际 Protected 数 `Np`，Mature target 为 `clamp(B-Np,0,Nm)`。Selector/TF conditioning 保持用户 global ratio；预算损失为 `abs(sum(mature_keep_probability)-Bm)/max(B,1)`。GI supervision 仅覆盖 Mature 且 GI valid 的节点。

预算是期望数量，仍存在 Mature Gumbel 随机波动，不保证严格上限。Mature 容量不足时预算会未用满；全未成熟时 Protection 最多消耗总预算的 rho，即使 ratio=1 也不承诺全量选择。原生 Full Render 不受这一限制。

## 拓扑与 checkpoint

新 KF 点 birth 使用当前正常 Mapping clock，优化次数和 KF support 为零。Clone/Split child birth 重新开始、优化次数为零，继承 parent KF IDs 和 unique count。CaRtGS 的 `exist_since_iter_` 保持原语义。

所有 metadata、uniform 及本轮待 commit 的 Full visibility 使用 GaussianModel 同一 survivor 映射。Append 的 child 不继承 parent 本轮 visibility；Split 删除 parent 时，旧 visible indices 重映射到 survivor 索引，child 不获得父节点的渲染证据。

`selector_metadata.ply` 保存 birth、optimized_count、unique_keyframe_count、KF IDs、uniform、Mapping clock 和转换计数。PLY 不支持原生 int64，birth/KF IDs 使用两个 uint32 word 无损保存，保留 -1 sentinel。Python 与 C++ 读取同一协议；旧 checkpoint 缺少证据字段时明确报错，不推造证据。

## 配置与 TF

```yaml
Selector.protection_enabled: 1
Protection.age_ready: 100
Protection.optimization_ready: 80
Protection.unique_keyframe_ready: 4
Protection.stale_age: 1000
Protection.max_budget_fraction: 0.35
Protection.weighted_bernoulli: 1
Protection.clone_split_inherit_keyframe_support: 1
Protection.performance_logging: 1
```

Weighted Bernoulli 与 child KF inheritance 是本方案的固定规则，两个字段必须为 1，设为 0 会报配置错误。关闭 protection_enabled 的实验将全部节点交给 Mature Selector，仍使用 global ratio 预算。

TransformField 继续由 `TransformField.mature_only` 决定：为 1 时只作用于选中的 Mature；为 0 时可作用于所有准入节点。普通、Guard、历史 QUALITY 与 Python 评估采用相同规则。

## 诊断与性能

状态/预算日志保持低频，checkpoint 输出 `protection_state.json`。日志包含四类状态、M/A/U/K/P 均值、优化/KF 均值、五类状态转换、STALE 证据均值及 target/actual 数量，不输出旧分类字段。

性能记录各阶段 host mean/p95/max，CUDA 每 20 次正常 Mapping 采样，通过事件 query 延迟收集，不主动 synchronize。CUDA event 耗时包含该 stream 内的 host launch gaps。GPU KF 更新与优化计数融合，unique_kf_support_update_ms_cuda 是融合 kernel 区间；protection_total_ms 是一次正常 Mapping 中各个不重叠 Protection 区间之和，排除 Full Render/optimizer 的耗时。状态日志自身的统计/输出不用于算法额外开销比较。

默认 K0=4，长期 metadata 为每 Gaussian 52 bytes（birth 8、opt 4、unique 4、IDs 32、uniform 4），另有 5 个 int64 转换计数。相对旧 birth/seen/uniform 共 12 bytes，增加 40 bytes/Gaussian。分类/采样的临时 tensor 与 allocator reserved memory 另行记录，不能与长期 metadata 混为一谈。

## 边界

KF support 达标后不记录更多 ID；不同 KF 不等同于不同视角方向。Child 继承局部区域 support 是近似，仍必须积累自身 age 与 optimization。STALE 没有清理行为，没有 view-direction diversity、stability、Fisher/Hessian 或 Mature 回退。正常优化计数使用 int32，并在最大值饱和，避免极长运行溢出。
