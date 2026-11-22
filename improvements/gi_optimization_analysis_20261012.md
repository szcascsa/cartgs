# Streaming GI 效率优化分析

日期：2026-10-12。范围：当前 Streaming Global GI 的数据获取、选帧、贡献缓存、刷新、Gaussian 拓扑同步和 Selector teacher。

本文基于当前源码及上一轮服务器 room 实测，仅进行分析，没有修改训练代码、配置或启动新的训练。性能基线见 [服务器诊断报告](<H:/AIProjects/cartgs优化更细致版/mycartgs/improvements/ultrafast_server_performance_20261012.md>)。以下收益推算不是优化后的实测结果。

## 1. 结论与优先级

**优先消除每个 Mapping iteration 对全部 KF/MapPoint 的重复扫描，再让 view bank 接收发生变化的 KF。** 当前 GI 内部已经实现贡献缓存和增量 coverage；主要重复工作发生在这些机制之前。

| 工作 | native 累计 | 占完整 Mapping | 优先级 |
|---|---:|---:|---|
| KF/MapPoint snapshot，含锁等待 | 29.7549 s | 51.97% | 最高 |
| Bank 输入排序、比较、覆盖维护及选帧 | 3.3958 s | 5.93% | 高，配合 snapshot 缓存 |
| 新 Bank 成员 render + aggregation | 0.9260 s | 1.62% | 低 |
| Gaussian topology 同步 | 0.1998 s | 0.35% | 低 |
| 两次 periodic full refresh | 0.0734 s | 0.13% | 低 |
| 上述 GI 管理工作合计 | 34.3498 s | 60.00% | |
| GI teacher 构造及 loss | 未独立计时 | 不包含在上面的 GI 合计中 | 应补计时，可做小范围优化 |

Snapshot + bank 合计 33.15 s，占已计时 GI 工作的 **96.51%**。平均每个 Mapping 为 snapshot 11.14 ms、bank 1.27 ms。Bank 的 3.40 s 不能全部归因于 greedy 选帧，其中也包含输入标准化、比较和容器维护。

当前已经具备的优化包括：增量 coverage 计数、unique gain/owners XOR、持久 Bottom-K 优先级、新加入视角才渲染、已有视角缓存、Inference renderer、teacher 使用 kthvalue。实施时不应重复建设这些功能。

## 2. 真实数据依赖

```text
每个正常 Mapping
  → updateGlobalGIBank
    → 获取 ORB map update lock
    → 遍历当前 Map 中全部 KF，过滤 bad/尚未进入 Mapper 的 KF
    → 每个 KF 枚举 MapPoint、检查 bad、复制 ID
    → 释放 ORB map lock
    → bank.update：排序去重、比较历史关联、维护 coverage/成员
    → 新成员渲染，旧成员移除，更新 sum/importance/valid
  → LoD phase：构造 GI teacher，计算 Selector GI loss
  → Full phase 等既有路径发生 Gaussian topology 变化时立即同步 GI
  → 完成 Mapping，按既有 mapping_iter_ deadline 检查 periodic refresh
```

Bank 策略取决于有效 KF 集合及其有效 MapPoint ID 集合，不依赖 MapPoint 坐标或 KF pose 数值。单纯 BA pose 更新不要求重做覆盖选帧；contribution 的几何和可见性更新继续由现有刷新策略负责。两种失效原因应分别处理。

源码：[Mapper snapshot](<H:/AIProjects/cartgs优化更细致版/mycartgs/src/gaussian_mapper.cpp:2475>)、[Bank](<H:/AIProjects/cartgs优化更细致版/mycartgs/include/global_gi_bank.h:47>)、[GI manager](<H:/AIProjects/cartgs优化更细致版/mycartgs/src/global_gi.cpp:138>)。

## 3. 第一阶段：缩短现有 snapshot

### 3.1 使用 vector snapshot，删除重复工作

当前 `KeyFrame::GetMapPoints()` 在 feature lock 下遍历关联，对每个 MapPoint 调用 `isBad()`，构造临时 `std::set<MapPoint*>`；Mapper 再次调用 `isBad()`，最终 bank 还会按 ID 排序去重。

推荐只替换 GI 这一处调用：

```text
GetMapPointMatches() 返回 vector 副本
  → GI 外层 null/bad 过滤
  → 复制 MapPoint ID
  → 既有 bank 排序去重
```

这样取消临时 pointer tree 的插入/分配和一层重复 bad 检查。Feature lock 只用于复制 vector，不再覆盖逐点 bad 查询和 set 插入。预留输出容量可以同时减少 vector 扩容。

约束：保留现有 ORB map 锁边界；不要为其他模块修改 `GetMapPoints()` 的公共行为。重复关联最终仍须去重。对冻结的 ORB 状态，两条路径应输出完全相同的 KF → sorted unique MP IDs；并发 SLAM 下采样时刻可能不同，不能承诺两次在线运行逐帧一致。

这是改动最小的起点，但仍然是每步全量 O(E) 扫描，也仍会申请 map lock，不能据此承诺消除大部分 snapshot 时间。E 表示全部有效 KF 的关联条目数。

源码：[GetMapPoints / GetMapPointMatches](<H:/AIProjects/cartgs优化更细致版/mycartgs/third_party/ORB-SLAM3/src/KeyFrame.cc:405>)、[MapPoint::isBad 的双锁](<H:/AIProjects/cartgs优化更细致版/mycartgs/third_party/ORB-SLAM3/src/MapPoint.cc:373>)。

### 3.2 先区分等待时间与持锁工作

上一轮 OSRT 记录 GI snapshot 路径 mutex wait 7.975 s，Tracking::Track mutex wait 8.705 s，来自带 profiler 的运行。它们证明存在竞争，不能当成 native 的精确等待时间，也不能相加作为可节省 wall time。

建议在原计时范围内加入：map lock wait、map lock hold、KF 枚举、关联复制/过滤、扫描 KF/关联数量。仅 CPU 时间戳即可覆盖这些 CPU 区间，无需插入 CUDA synchronize。

## 4. 第二阶段：按真实关联变化更新，是主要收益来源

### 4.1 推荐架构

维护独立的 GI 关联变化版本及 dirty KF 集合，CPU 缓存每个有效 KF 的排序去重 MP IDs：

```text
正常 Mapping
  → 检查 Map generation、关联变化版本、Mapper KF 可用性变化
  → 全部未变且 GI contribution cache 完整：复用 bank
  → 有变化：仅重建 dirty/new KF，处理 removed KF
  → 批量更新 bank 一次
  → 按原规则处理成员增删
```

无变化时避免 map lock 和全量关联扫描；有变化时扫描成本从全部 E 降为 dirty KF 的关联数 E_dirty。初版也可以用每 KF version 的 O(KF) 检查，只要安全维护 KF 列表及对象生命周期，仍远小于逐条 MapPoint 加锁。

推荐分两步实施：先建立可靠的失效通知与缓存、在无变化时跳过；再给 bank 增加批量 delta 输入。即使变化时暂时保留原 `bank.update(snapshot)`，也能先验证无变化快速路径的收益和正确性。

### 4.2 必须覆盖的失效入口

| 变化 | 需要的处理 |
|---|---|
| `KeyFrame::AddMapPoint` | 标记该 KF 的关联变化 |
| 两个 `EraseMapPointMatch` 重载 | 标记该 KF，包含 BA 删除 outlier 路径 |
| `ReplaceMapPointMatch` | 标记该 KF，包含 fusion/loop closure |
| `MapPoint::SetBadFlag / Replace` | 使受影响的观察 KF 失效；覆盖 bad 标志与关联改写的全过程 |
| KF 插入、删除、`SetBadFlag` | 更新有效候选池及容量 |
| Map merge、切换、clear、PostLoad | 更新 Map generation，重建对应 CPU snapshot 状态 |
| KF 首次进入 `historical_keyframes_`、满足相机可用条件 | 即使 ORB 关联版本未变，也纳入新候选 |
| Gaussian reset 清空 contribution caches | 即使 CPU bank 没变，也执行现有 bootstrap 路径 |

不能仅监听“新 KF”，旧 KF 的关联也会变化；不能仅使用现有 `MapChangeIndex`。源码中的 Add/Erase/Replace 关联接口没有统一递增它，LocalMapping 的新增点与 culling 也不统一受 map update lock 保护。

只记录 MappingOperation 的 BA/loop closure 事件同样不完整，因为没有涵盖所有底层关联写入。

### 4.3 并发与提交要求

这是实现中风险最高的部分，应先定义完整协议，再编写缓存：

1. dirty 通知必须与关联/bad 状态的修改正确排序。简单在操作开始时 `dirty=true`，读端读完再设回 false，会吞掉读取期间发生的新修改。
2. 使用单调 version 或交换 dirty 批次；读端只确认自己消费的版本，新版本保留到下一次处理。读取期间版本变化需重读/重新排队；严格同步方案应等待完整提交，而不是发布半个 Replace/SetBad 操作。
3. version/atomic 只能用于失效通知，不能替代对 `mvpMapPoints` 的同步读取。当前 `ReplaceMapPointMatch` 和 pointer 版本 `EraseMapPointMatch` 自身没有获取 feature lock，不能假设所有写端已经遵守同一把锁。
4. 修订锁协议时审查实际调用者与锁顺序；不能在已有外层持锁路径中重复获取非递归 mutex。GI 通知不能反向调用 Gaussian Mapper 或在 ORB 锁内执行 GPU 工作。
5. CPU cache 长期只保存稳定 ID 与版本，不保留跨生命周期使用的裸 ORB 指针。Map generation 区分 reset/切换；移除事件与新事件都不能遗失。
6. 在验证模式中定期构造完整 reference snapshot，检查增量 snapshot 完全相同。验证通过前不要移除对照机制；性能跑和完整对照跑分开。

纯 pose 变化不标 dirty；真正的 observation 替换必须标 dirty。这是避免缓存仍被每次 BA 全量打脏的关键。

源码：[KeyFrame 关联写入](<H:/AIProjects/cartgs优化更细致版/mycartgs/third_party/ORB-SLAM3/src/KeyFrame.cc:383>)、[MapPoint 失效和替换](<H:/AIProjects/cartgs优化更细致版/mycartgs/third_party/ORB-SLAM3/src/MapPoint.cc:292>)、[Mapper KF 注册](<H:/AIProjects/cartgs优化更细致版/mycartgs/src/gaussian_mapper.cpp:2439>)、[GI reset](<H:/AIProjects/cartgs优化更细致版/mycartgs/src/global_gi.cpp:54>)。

## 5. Bank 本身如何优化

当前 `GlobalGIViewBank::update` 每次构造完整 `next` map，排序每个 KF 的 MP IDs，再比较完整集合；无变化返回发生在这些工作之后。

Delta 输入应只包含新增/改变的规范化关联集合及 removed IDs，旧的 `pool_` 原地复用。规范化只做一次；重复 dirty 但实际集合未变时，应仍识别为 unchanged，避免无效选帧。

必须保持原策略：相同 KF ID 顺序、相同 strict-gain replacement、相同 tie 行为、相同 quota 与 Bottom-K。每个 Mapping 边界将这一批变化一次性应用，再做一次选择，不能每收到一个 KF 事件就立即改 bank；后者会改变 greedy 的中间状态和最终成员。

当 Anchor 关联发生变化，当前策略会重新评估历史候选。不能为追求速度只检查 dirty candidate；这会改变覆盖最优候选的搜索范围。第一版保留此逻辑，后续仅在新 profile 显示它成为热点时，考虑反向覆盖索引等更复杂结构。

已有 [global_gi_bank_test](<H:/AIProjects/cartgs优化更细致版/mycartgs/tests/global_gi_bank_test.cpp>) 对 reference 策略比较 members、anchors、representatives、coverage；扩展它对照 full snapshot 与 delta 批次是合适的验收方法。

## 6. GI teacher：减少全量临时 Tensor

当前 builder 每个 LoD 构造 `[N]` supervised mask、`nonzero` 和 `[N]` label，之后 loss 又通过布尔 mask 取回有效行。GI manager 的 60% 计时没有涵盖这一段。

推荐顺序：

1. `mature_count == 0` 或 `selector_gi_loss_weight_ == 0` 时，不构造仅供 GI loss 使用的 teacher。保留 Selector/photo/ratio loss 和 GI manager 本身的运行逻辑。
2. 复用本次 LoD 已有的 `mature_indices`，在该子集上读取 valid、构造 eligible indices，返回 compact labels；GI loss 直接 gather 对应 Selector 输出，避免先 scatter 成 `[N]` 后再 gather。
3. 保持 eligible 集合、loss 归一化、effective mature ratio、strict `score > kthvalue` 和梯度完全一致。相同分数不能改为强行凑齐 Top-K；未知 GI 行不能变成负样本。

首版不建议缓存四个 nominal ratio 的完整 teacher。实际使用的是 `budget.mature_ratio`，它随 Protection 实际 admission 改变；Mature 集合也会随 age/evidence 改变，且可能在 Gaussian 数量和 GI version 都不变时发生。仅用 `(ratio, gi_version, topology_version)` 作为 key 会得到过期监督。

如果后续引入缓存，必须包含真实 eligible 集合的变化版本、effective ratio 与 GI/索引版本；生成这些版本的代价也要测量。当前 room 最终 mature 只有 51，排序本身不是已证实的主热点，减少全量 mask/label 更值得先验证。

源码：[teacher build](<H:/AIProjects/cartgs优化更细致版/mycartgs/src/gi_teacher.cpp:6>)、[LoD 调用](<H:/AIProjects/cartgs优化更细致版/mycartgs/src/gaussian_mapper.cpp:2699>)、[实际 mature ratio](<H:/AIProjects/cartgs优化更细致版/mycartgs/include/elastic_budget.h:74>)。

## 7. Gaussian topology 与 contribution cache

当前每个视角缓存长度 N；append 对每个 view 做 cat，prune 对每个 view 做 bool indexing，然后再遍历所有 view 聚合。当前只占 0.35%，放在主要 CPU 优化之后。

较小的优化是 prune 时只构造一次 survivor indices，所有 view 使用相同 index_select，避免重复将同一 bool mask 转换为索引。保留 survivor 顺序、即时对齐、新生行零占位且 invalid、split 先 append 再 prune 的现有语义。

数学上 append/prune 是共同线性行变换，因此可以对 `sum` 和 `importance` 使用同一个变换，避免每次重新聚合所有 views。但现有 `aggregate()` 也会重新求和、消除增量加减积累的舍入误差；直接变换旧 sum 不保证 bitwise 一致。该优化需误差与 teacher 标签对照，不能未经验证称为完全等价。

不建议首轮改成大块 `[K,N]` cache、稀疏 cache 或 FP16。当前最终 view contributions 约为 `72 * 248281 * 4 = 68.19 MiB`；长序列的 O(KN) 扩展需要关注，但布局更改会扩大 append/prune、容量管理及错误恢复的复杂度。FP16、稀疏截断还可能影响 GI 排序。

源码：[append / prune / aggregate](<H:/AIProjects/cartgs优化更细致版/mycartgs/src/global_gi.cpp:79>)。

## 8. GI Render、增量聚合和 full refresh

这些路径已使用 NoGrad + Inference，不分配 Training backward 的 bucket/sample 状态；不要把已有功能重复列为新收益。

可保留到后续的等价方向：

- 同一批 GI views 共享一次 Gaussian activation/attribute 准备，限于持有相同模型快照的这一批，不能跨 optimizer iteration 复用过期属性。
- 专用 contribution-only renderer 跳过 SH/RGB 累加与图像写出；GI 保持现有 `con_o.w * T`、alpha 阈值、深度顺序和 transmittance 提前终止。Streaming consumer 不使用 contribution_count，可研究其单独关闭，但 Recent/其他调用仍需要它。
- 减少 incremental aggregation 的临时复制或复用工作区，保留统一发布与 finite/valid 语义。

它们在 room 的实测收益上限很小：incremental + periodic 全部消失也仅对应约 1.75% 的完整 Mapping 时间，而且实际不可能全部消失。专用 CUDA kernel 的实现和验证成本目前不划算。

full refresh 必须继续按真实 `mapping_iter_` 执行，并在同一个模型状态上计算整批 bank 后发布。跨多个训练 step 摊开 refresh 会产生混合模型时刻的估计；后台 GPU refresh 还需稳定的参数快照与 topology 对齐，属于更大的方法/架构选择。

源码：[renderGlobalGIContribution](<H:/AIProjects/cartgs优化更细致版/mycartgs/src/gaussian_mapper.cpp:2463>)、[refreshIfDue](<H:/AIProjects/cartgs优化更细致版/mycartgs/src/global_gi.cpp:205>)、[现有 raw contribution](<H:/AIProjects/cartgs优化更细致版/mycartgs/cuda_rasterizer/forward.cu:419>)。

## 9. 计时与检查自身的开销

- `assertAligned` 每次遍历全部 view Tensor 元数据，但不读取 GPU 内容。可在 GI mutation 后完整检查，普通读取保留 O(1) aggregate/shape 检查；先测量，勿将它称为同步热点。
- `recordGuidance` 已在报告节奏调用，不是每步三个 `.item()`。无需解决一个当前不存在的逐步统计问题。
- GI timer 在真实增量/拓扑/refresh 时创建 CUDA events 并等待。事件复用和延迟 query 可降低仪器开销，但不会消除 CPU snapshot；更改后必须分别报告 CPU wall 与 GPU elapsed，不能用未同步的 CPU submission time 代替 GPU 时间。
- 现有每 20 步 sampled profiler 与 5 phase 轮转对齐，总采 Full；新增 GI teacher 计时应覆盖全部 phase 或采用不混叠采样。

## 10. 方案比较与不推荐的捷径

| 方案 | 优点 | 代价/风险 | 建议 |
|---|---|---|---|
| Vector snapshot + 单次过滤 | 小改动，可独立验证 | 仍每步 O(E)，仍有全局锁竞争 | 第一阶段 |
| 关联版本 + dirty KF cache + batch delta | 消除无变化扫描，长期扩展更好 | 必须完整覆盖并发 mutation 和生命周期 | 主要优化目标 |
| 每 N 步才检查 bank | 实现简单 | 有效 KF/关联更新延迟，会改变 bank 演化与监督时机 | 不作为等价优化 |
| 只在 Full phase 维护 bank | 少检查 | 错误绑定 ratio phase，四个 LoD 使用延迟 bank | 不推荐 |
| 增大 refresh_interval | 简单 | 当前仅 0.13%，还延迟 newborn GI 有效化 | 不优先 |
| 降低 view_ratio / max_views | 减少 render/cache | 全量 snapshot 仍在，改变覆盖与估计 | 作为独立质量消融 |
| 删除锁或后台读取裸指针 | 表面减少等待 | 生命周期、数据竞争、半提交状态风险 | 不采用 |
| 使用 LoD render 替代 Full GI | 复用渲染 | mask 改变遮挡与 T，GI 不再是现有 Master contribution | 不等价 |

## 11. 可期待的收益范围

在固定 2671 个 Mapping、相同 KF/Gaussian 工作量、其他时间不变的假设下，原完整 Mapping 均值为 21.43 ms，snapshot + bank 为 12.41 ms。

| 假设减少的 snapshot + bank 耗时 | 推算 Mapping mean | 推算吞吐提升 |
|---:|---:|---:|
| 50% | 15.23 ms | 1.41x |
| 75% | 12.13 ms | 1.77x |
| 90% | 10.26 ms | 2.09x |

这些是预算分析，不是收益承诺。当前没有记录 dirty KF 比例、无变化步比例和 native 的精确锁等待，因此无法预测落在哪一行。在线训练变快后可能完成更多 Mapping，改变 Gaussian 数、maturity、TF 启动时间及 refresh 次数；输入本身还按时间戳限速，所以不能将上述提升直接换算成端到端 FPS。

## 12. 实施与验收顺序

1. **小范围 baseline 修正**：分离 snapshot wait/hold；实施 vector snapshot；给 teacher 加范围并跳过无监督空路径。冻结 snapshot/teacher 输入，验证输出及梯度一致。
2. **建立可靠增量输入**：覆盖关联写入、bad/replace、候选可用性、map generation 和 cache bootstrap；先验证完整 snapshot 与缓存逐批一致，再启用无变化快速路径。
3. **Bank batch delta**：沿用原选择策略。用相同批次回放对比 members、anchors、representatives、coverage、membership-changed 标志和 render 次数。
4. **重新 profile**：确认主热点是否已转移，再决定 compact teacher、topology gather 或 GPU 数据准备优化。

已有测试可作为后续实现的回归入口，以下命令本轮未执行：

```bash
ctest --test-dir <existing-build-dir> -R '^(global_gi_bank|global_gi|gi_teacher|renderer_readonly_cuda)$' --output-on-failure
<existing-build-dir>/global_gi_bank_test --benchmark
```

新增验收重点：关联保持不变却 pose 改变；旧 KF 被增删关联；MapPoint SetBad/Replace；同批多次修改与删除；读取期间再次修改；Map merge/reset/PostLoad；ORB KF 已存在但 Mapper 延迟注册；同 bank IDs 下 Gaussian reset；append/clone/split/prune 索引一致；重复分数和无有效 GI 的 teacher。

性能对照分两类：固定 CPU snapshot/事件序列测单次算法成本；相同 room 命令做原生在线运行并记录实际 Mapping 次数、KF/Gaussian 数、Tracking mean/P95、GI 分项和全流程 wall time。完整 reference 对照、Nsight profiling 与最终速度测量分别运行，避免把验证开销混入速度结论。

质量继续评估 Full 及 1%/5%/10%/15% 的 PSNR、SSIM、LPIPS 和实际 Gaussian 数；保持在线结束即停、tail=0。算法等价不意味着异步在线两次运行得到完全相同的最终模型。

推荐落地组合：**vector snapshot 起步，完整关联失效通知 + dirty KF cache 作为核心，batch delta bank 跟进；teacher/topology/GPU 优化按第二轮 profile 决定。**
