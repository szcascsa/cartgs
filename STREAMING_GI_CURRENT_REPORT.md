# Streaming Global Importance（SGI）当前实现详细报告

- 报告日期：2026-10-19
- 代码基线：`f7e2ce5`（工作区 `include/`、`src/`、`third_party/ORB-SLAM3/` 无未提交改动）
- 适用范围：ORB-SLAM3 在线 Mapping（RGB-D / 单目 / 双目）；COLMAP 离线训练不支持 `streaming`
- 阅读对象：需要维护、调参、排查或评估 SGI 的开发者

> 本报告以**当前工作区源码**为准。`improvements/streaming_global_importance.md`（2026-10-10）和
> `CONTINUOUS_MATURITY_PIPELINE_AUDIT.md`（2026-10-14）里的部分 GI teacher 语义、`recent` 模式状态和
> 性能数字已被后续提交改变，差异集中列在第 11 节。

---

## 0. 结论摘要

| 维度 | 当前状态 |
|---|---|
| 默认来源 | `GlobalGI.mode: "streaming"`（`cfg/selector/selector_config.yaml`、`cfg/selector/fast.yaml`） |
| 本质 | 用一组**全局选出的历史关键帧**渲染每个 Gaussian 的可见性贡献，取等权平均作为全局 importance |
| 用途 | 只作为 Selector 的 GI teacher；**不参与**候选过滤、LoD 预算或渲染 mask 计算 |
| Bank 构成 | `30%` 有效历史 KF，其中 `80%` coverage anchor、`20%` 确定性 Bottom-K representative |
| 更新节奏 | 每个正常 Mapping boundary 增量更新；每 `1000` 次 Mapping 全量重渲染 |
| 主要成本（2026-10-12 实测） | GI 维护 `1.0891 ms/正常 step`（Mapping wall 的 `8.35%`）；含 teacher + GI loss 为 `1.2706 ms`（`9.74%`） |
| 主要剩余瓶颈 | ORB 关联快照 `0.4296 ms/step`，仍高于选帧 `0.3409 ms/step` 和增量渲染 `0.2259 ms/step` |
| 最关键限制 | 新生 Gaussian 在下次全量刷新前 `valid=false`，最多约 `1000` 次 Mapping 得不到 GI 监督 |
| 显存量级 | 约 `(4K + 9)·N` 字节（`K` 为 Bank 视角数，`N` 为 Gaussian 数）；全量刷新时新旧缓存短暂共存 |
| 当前验证 | 本机 CPU 独立测试（Bank + snapshot cache）通过；服务器历史记录含 3015/3015 boundary 权威对照 |

一句话：**SGI 已经是一个稳定、可观测、成本已降到次要位置的全局重要性教师信号；它的主要问题不再是速度，而是新生点的监督延迟、Bank 对 Gaussian 可见性的近似，以及随地图规模线性增长的缓存。**

---

## 1. 术语与定位

### 1.1 三个容易混淆的概念

| 概念 | 含义 | 当前代码 |
|---|---|---|
| **GI importance** | 每个 Gaussian 在一组视角下的**原始可见性贡献均值** | `GlobalGIManager::importance_`，`[N] float32` |
| **GI teacher** | 由 importance 经 percentile 离散化得到的二值标签、mask 和权重 | `GITeacher` / `GITeacherBuilder` |
| **Selector probability** | Selector 网络输出的保留概率，用于 LoD 采样 | `GumbelNetwork::forwardProbability` |

importance 只是标签来源，不是训练目标本身；Selector 学的是“在给定 ratio 下哪些点应被保留”。

### 1.2 `streaming` 的含义

`streaming` 指 Bank 随 ORB-SLAM3 的**关键帧流**持续增删，覆盖当前 Atlas Map 的全部有效历史 KF：

- 不是时间滑窗；
- 不是最近 N 帧；
- 不因时间流逝自动淘汰旧 KF；
- KF 失效、被 cull、Map 重置时才移除。

### 1.3 三种 GI 模式

| 模式 | 行为 | 当前状态 |
|---|---|---|
| `disabled` | 不创建 `GlobalGIManager`，不采集 GI，不计算 GI loss；`RecentWindow.collect_importance=false` | 支持；COLMAP 训练应使用 |
| `recent` | 旧版 RecentWindow 路径：逐视角归一化、EMA 时间衰减 | 代码保留，但已于 2026-10-15 归档到 `cfg/selector/archive/retired_features.yaml`，日常配置不再启用 |
| `streaming` | 本报告主题：全局 Bank、原始贡献等权平均 | 默认 |

`online_gi.h/.cpp` 是更早的独立模块，当前 Mapper 没有实例化它，仅保留编译与生命周期测试；不要把它和 SGI 混淆。

---

## 2. 端到端调用链

```text
ORB-SLAM3 KeyFrame / MapPoint 关联
   │  GIAssociationState 读写屏障、KF revision、bad revision、Map generation
   ▼
GaussianMapper::updateGlobalGIBank()                 src/gaussian_mapper.cpp:2799-2865
   │  GlobalGISnapshotCache::update()  ->  GlobalGIBatch{upserts, removals}
   ▼
GlobalGIManager::updateBankBatch()                   src/global_gi.cpp:149-214
   │  GlobalGIViewBank::updateBatch()  ->  members
   │  只渲染新成员 -> contribution [N]
   │  sum / mean / valid 增量更新
   ▼
importance() / validMask() / observedMask()          include/global_gi.h:49-52
   │
   ▼
GaussianMapper::accumulateLodGradients()             src/gaussian_mapper.cpp:2992-3242
   │  GITeacherBuilder::build()                       src/gi_teacher.cpp:5-33
   ▼
GITeacher::loss(selector_mask)                       include/gi_teacher.h:11-17
   │
   ▼
selector_loss += gi_loss_weight · ratio_weight · gi_loss
                                                     src/gaussian_mapper.cpp:3204-3210
```

### 2.1 一个正常 Mapping iteration 内的顺序

1. `trainForOneIteration()` 选择 KF，进入 `trainMappingIteration()`。
2. 获取 `mutex_render_`（整个渲染/更新期间持有）。
3. `updateGlobalGIBank()`：
   - 在 GI 读事务内读当前 Map、KF 元数据和关联 ID；
   - 只重建 revision 变化的行；
   - Bank 选帧；
   - 只对新 Bank 成员渲染 contribution。
4. 按当前 phase 执行 Full render 或 LoD：
   - Full 分支本身：普通渲染 + backward，不构建 GI teacher；
   - LoD 分支：构建 teacher、计算 GI loss、与 render/ratio loss 合并后 backward。
5. optimizer 提交，释放 `mutex_render_`。
6. `++mapping_iter_`，`gi_refresh_task_.observe(...)`；若到达刷新周期，重新获取锁并全量刷新。

GI loss 只可能在 `accumulateLodGradients()` 内构建，当前有四个调用点：

| 调用点 | 场景 |
|---|---|
| `src/gaussian_mapper.cpp:1505` | mode1 正常 rotation 的 1%/5%/10%/15% LoD step |
| `src/gaussian_mapper.cpp:1710` | mode1 尾部：每轮 Full 之后的一个 native LoD |
| `src/gaussian_mapper.cpp:1776` | mode2 的 detached LoD |
| `src/gaussian_mapper.cpp:2912` | Conflict Replay 的 joint LoD（默认 Replay 关闭） |

因此：**Full-only 路径不计算 GI loss，但仍会执行 Bank 增量维护**；如果配置既没有 rotation、也没有 mode2/tail/replay 的 LoD，SGI 会持续付出维护成本却不产生监督。

---

## 3. 贡献度（importance）的数学定义

### 3.1 单视角贡献

CUDA 前向在每个采样像素上累加：

```text
frame_importance[i] = Σ_p  α(i,p) · T(i,p)
α(i,p)              = o_i · exp(-0.5 · d(i,p))
d(i,p)              = (p - μ_i)ᵀ Σ_i⁻¹ (p - μ_i)
```

其中：

- `o_i`：Gaussian 的 opacity activation；
- `d(i,p)`：像素中心到该 Gaussian 投影中心的马氏距离平方；
- `T(i,p)`：该 Gaussian 之前的累计透射率；
- kernel 中的 `con_o.w` 就是 `α(i,p)`；
- 累加发生在 `cuda_rasterizer/forward.cu:421`，同一处还累加 `contribution_count`。

它**不是**：

- 像素覆盖数（那是 `contribution_count`）；
- 颜色加权贡献；
- 归一化后的可见概率；
- 可微张量（`mark_non_differentiable`，`src/gaussian_rasterizer.cpp:65`）。

直观理解：一个 Gaussian 覆盖的像素越多、越靠近可见表面（透射率高）、opacity 越大，贡献越大。

### 3.2 多视角聚合

```text
sum(i)        = Σ_{v ∈ Bank} frame_importance_v[i]
importance(i) = sum(i) / |Bank|              // 等权平均
```

代码：`src/global_gi.cpp:79-88` 与 `:199`。

关键设计：

- **不做逐视角归一化**。旧 `recent` 模式会先按视角内总贡献归一化，再做 EMA；streaming 故意保留原始量纲，避免把不同视角的尺度差异二次放大。
- **等权平均**。每个 Bank 视角权重相同，不按分辨率、重叠面积或 map point 数加权。
- **sum 与 mean 同时维护**。增量添加/移除视角时直接对 `sum` 加减，再除以当前视角数；这与“对当前缓存中的视角重新求和”一致，但不会重新渲染旧视角。

### 3.3 数值边界

`checkedContribution()`（`src/global_gi.cpp:69-77`）要求：

- 维度、dtype（float32）、device 与 `sum_` 完全一致；
- 非有限值或负值 → 该行 `valid=false`；
- 非有限值在聚合时替换为 `0`，不会污染均值。

因此“非有限”不会变成“被监督的 0 分”，只会变成“未知”。

### 3.4 重要性质与偏差来源

| 性质 | 后果 |
|---|---|
| 分辨率相关 | 不同分辨率的相机贡献量级不同；Bank 默认假设各视角分辨率接近 |
| 足迹相关 | 大 Gaussian 天然贡献大，importance 不是“单位面积重要性” |
| 透射率相关 | 被遮挡的 Gaussian 贡献趋近 0，即使它存在 |
| 等权平均 | Bank 成员变化会改变分母，importance 时间序列在成员变更处可能跳变 |
| 非归一化 | 不同场景/不同 ratio 的 importance 绝对值不可直接横向比较，只有排序有意义 |

GI teacher 只用排序（percentile），所以上述尺度差异不会直接破坏标签，但会影响跨视角混合后的排序质量。

---

## 4. Global View Bank：如何选视角

### 4.1 配额

`include/global_gi_bank.h:110-114`：

```text
M   = 当前有效历史 KF 数
K0  = min( M, max(1, ceil(view_ratio · M)) )
K   = max_views > 0 ? min(K0, max_views) : K0
K_A = ceil(coverage_anchor_fraction · K)
K_R = K - K_A
```

默认 `view_ratio=0.30`、`coverage_anchor_fraction=0.80`，即 30% 的 KF，其中 80% 为 coverage anchor，20% 为随机代表。

### 4.2 候选资格

只有同时满足以下条件的 KF 进入候选池：

1. 属于当前 Atlas Map；
2. `KeyFrame::GIBad()` 为 false；
3. 已注册到 Mapper 的 `historical_keyframes_`；
4. 相机宽高有效、`full_proj_transform_` 已定义。

代码：`src/gaussian_mapper.cpp:2816-2825`。

这保证了 Bank 只包含“能真正渲染”的 KF，但也意味着新 KF 在 Mapper 完成注册前不会进入 Bank。

### 4.3 Anchor：贪心最大覆盖

- 覆盖对象是 **ORB MapPoint ID**，不是 Gaussian；
- 每个 KF 的点集先排序去重；
- 初始填充：每次选择“能覆盖最多未覆盖 MapPoint”的候选；
- 满额后：只接受**严格增加覆盖**的替换；
- 相同收益按稳定 ID 顺序决定，保证可复现。

代码：`include/global_gi_bank.h:121-131`（填充）、`:134-156`（替换）。

替换收益的精确形式（`sgi_cpu_optimization.md`）：

```text
Δ(candidate, anchor) = uncovered(candidate)
                     + retained_unique(candidate, anchor)
                     - unique(anchor)
```

- `uncovered`：candidate 能新覆盖的点；
- `retained_unique`：当前只被该 anchor 覆盖、但 candidate 也能覆盖的点（替换后不会丢）；
- `unique(anchor)`：该 anchor 当前独占覆盖的点。

只有 `Δ > 0` 才替换，取最大者。

### 4.4 Representative：确定性 Bottom-K

- 对非 Anchor 的 KF，用 `SplitMix64(KF_ID XOR sampling_seed)` 计算持久优先级；
- 取优先级最小的 `K_R` 个；
- 与 Anchor 不重复；
- 同一 ID、同一 seed 在任何运行中顺序一致。

代码：`include/global_gi_bank.h:157-161`、`:181-186`。

### 4.5 覆盖率

```text
coverage_ratio = |被 Anchor 覆盖的 MapPoint| / |所有候选 KF 的 MapPoint 并集|
```

代码：`include/global_gi_bank.h:167`。它衡量 Bank 对 ORB 地图点的覆盖，不是对 Gaussian 的覆盖。

### 4.6 生命周期与复杂度

- 旧 KF 不因时间淘汰；
- KF 失效/移除时从池中删除，并回收其覆盖计数；
- 关联变化时增量更新覆盖计数（`Coverage{count, owners_xor}`），避免每次重建完整并集；
- 每次 boundary 的选帧复杂度约为 `O(候选 × 点 + 候选 × Anchor)`，而不是原来的 `O(候选 × Anchor × 点)`。

2026-10-13 合成基准（243 KF、每帧 1000 关联、30 次采样）：Anchor 关联变化场景平均 `306.34 ms → 24.16 ms`（`12.68×`），候选变化场景 `1.17×`，无变化场景基本持平。该数字**不含** ORB 快照读取。

---

## 5. 缓存、增量更新、拓扑与刷新

### 5.1 每个 boundary 的增量流程

```text
snapshot:
  begin GIAssociationState::Read
  map = Atlas.GetCurrentMapForGI()
  if map == null: 结束事务, yield, retry
  for kf in map.GetAllKeyFrames():
      if kf.GIBad() or 未注册 or 相机无效: skip
      metadata += (kf.id, kf.GIAssociationRevision())
  batch = snapshot_cache.update(map.GIGeneration(),
                                GIAssociationState.badRevision(),
                                metadata, kf.GIPointIds)
  end Read

update:
  bank.updateBatch(batch) -> members
  for id in members - views:
      contribution[id] = render(id)          // 只渲染新增
  for id in views - members:
      sum -= contribution[id]; erase
  sum += new contributions
  mean = sum / |members|
  valid = (既有 valid) & finite(new views)
```

代码：`src/global_gi.cpp:149-214`。

关键点：

- **保留成员不重渲染**；普通参数更新、重复使用同一 KF 都不刷新旧缓存；
- 新增成员只渲染一次；
- 移除成员只做减法，不触发渲染；
- 成员未变化、已初始化且缓存完整时直接返回（`src/global_gi.cpp:161-166`）；
- 实际更新时 `++version_`，供诊断使用。

### 5.2 `valid` 与 `observed` 的区别

| 状态 | 定义 | 含义 |
|---|---|---|
| `valid` | 全量刷新后 = 当前 Bank 所有视角贡献均有限且非负；两次刷新之间 = 历史增量中所有**加入过**的视角的有限性累计 AND | “这个估计没有被 NaN 污染” |
| `observed` | `importance_ > 0` | “至少在一个 Bank 视角里真的产生了贡献” |

注意 `valid` 的保守性：视角被移除时不会把 `valid` 恢复为 true，因此两次刷新之间可能出现“旧视角留下的 false”。下一次全量刷新会按当前 Bank 重新计算。

二者独立：

| valid | observed | 是否进入 GI teacher |
|---|---|---|
| true | true | 是 |
| true | false | 否（importance 恰为 0，没有证据） |
| false | true | 否（估计不可信，例如某个视角 NaN） |
| false | false | 否 |

`observedMask()` 在 `include/global_gi.h:52`，实现为 `importance_ > 0`。

**这一点非常关键**：`valid=true` 只代表数值有限，不代表这个 Gaussian 被看见。真正决定是否监督的是 `observed`。

### 5.3 拓扑同步

`GaussianModel` 的 append/clone/split/prune/reset 回调在 `src/gaussian_mapper.cpp:137-162` 注册。

| 操作 | 模型行为 | SGI 行为 | 渲染 |
|---|---|---|---|
| Append | 旧行 + 新行 cat | 每个视角补 `count` 个 0；新行 `valid=false` | 不渲染 |
| Clone | 追加子行 | 子行 0、不继承 GI；父行保留 | 不渲染 |
| Split | 先追加子行，再按真实 mask 删除父行 | 两次回调分别同步 | 不渲染 |
| Prune | 对属性/元数据用 `~mask` | 对每个视角用同一个 survivor mask 做 `index` | 不渲染 |
| Reset | 重建模型 | 清空 `views_`、sum/mean/valid 归零、版本递增 | 不渲染 |

关键保证：

- 使用**模型真实的 survivor mask**，不根据长度猜测；
- 全过程持有 `mutex_render_`，Selector 看不到中间状态；
- 新生行在下次全量刷新前保持 `valid=false`；
- 旧的 `valid=true` 行不会因为普通拓扑变化整体失效。

拓扑同步的成本是 `O(K·N)` 的 aggregate，实测约 `0.046 ms/step`，远低于快照成本。

### 5.4 周期全量刷新

`refreshIfDue()`（`src/global_gi.cpp:216-258`）：

- 默认每 `refresh_interval=1000` 次 Mapping 一次；
- 只接受 `iteration % interval == 0`，且同一 iteration 不重复；
- `force=true` 允许非整周期刷新（`DeferredMappingTask` 使用）；
- 重新渲染**全部 Bank 成员**，在临时状态中构造 `next_sum/next_valid/mean`，完成后一次性 `swap` 发布；
- 刷新后所有有限行重新 `valid=true`，新生点也在此刻获得有效估计；
- 刷新期间持有 `mutex_render_`，Gaussian 索引不会中途变化。

调度代码：`src/gaussian_mapper.cpp:1327-1336`。

```text
gi_refresh_task_.observe(mapping_iter_, refresh_interval);
if (gi_refresh_task_.take(enhanced)) {
    lock_render;
    refreshIfDue(mapping_iter_, render, force=true);
}
```

`DeferredMappingTask` 只负责“到期后等待允许执行”，真正的时间判断仍由 `refreshIfDue` 完成（`include/ultra_light_schedule.h:6-17`）。

### 5.5 空 Bank、Map 重置与 map switch

- **空 Bank**：`views_` 为空 → `sum/importance/valid` 全 0 → GI teacher `supervised_k=0` → 跳过 GI loss；
- **Gaussian reset**：清空所有视角缓存，下一次 `updateBankBatch` 检测到缓存缺失，强制重新引导（`src/global_gi.cpp:171` 的 `!initialized_estimate_` 分支）；
- **Map reset / generation 变化**：`GlobalGISnapshotCache` 清空旧行并重建（`include/global_gi_snapshot_cache.h:14-18`）；
- **Map switch**：`Atlas::GetCurrentMapForGI()` 只在读事务内返回当前有效 Map，不等待、不持 `mMutexMapUpdate`（`third_party/ORB-SLAM3/src/Atlas.cc:238-242`）。

### 5.6 新生点的监督延迟（设计上的重要副作用）

1. Append 后新行 `valid=false`；
2. 正常 boundary 加入新视角时，`next_valid` 从旧 `valid` 克隆，新行仍为 false；
3. **在 Bank 已初始化且非空的常态下**，只有下一次全量刷新会把新行重新评估为 `valid=true`；
4. 刷新后还要 `observed=true`（在某个 Bank 视角里有正贡献）才会真正进入 teacher。

例外：Bank 尚为空（`initialized_estimate_=false`）或 Gaussian reset 后的首次 bootstrap，`next_valid` 从全 true 开始，此时新行可能立即变为 `valid=true`。这是一种“重新建立 Bank”的路径，不是常规增量路径。

因此：

- 最坏延迟约 `refresh_interval=1000` 次 Mapping；
- 若新生点始终不在任何 Bank 视角中产生正贡献，即使刷新也不会被监督；
- 这属于设计取舍：避免用“零贡献”去监督未知点，但牺牲了新生点的响应速度。

如果训练对新生点响应敏感，应降低 `refresh_interval`（见第 13 节），而不是关闭 `observed` 检查。

---

## 6. 并发与一致性

### 6.1 两把锁的职责

| 锁 | 保护对象 | SGI 使用方式 |
|---|---|---|
| `GaussianMapper::mutex_render_` | Gaussian 属性、Selector、渲染与所有 GI 缓存 | 所有 GI 写/读都在此锁内 |
| `GIAssociationState` | ORB KF/MapPoint 关联 | GI 快照持**独占读事务**；ORB 关联写者持共享写事务 |

SGI 不再获取 `Map::mMutexMapUpdate`，这是 2026-10-12 优化的核心之一。

### 6.2 ORB 读事务

`third_party/ORB-SLAM3/include/GIAssociationState.h`：

- `Write`：共享锁（`lock_shared`），可嵌套，只有最外层加锁/解锁；
- `Read`：独占锁（`lock`），可嵌套；
- 通过 thread_local 深度计数，保证嵌套 bad/replace 事务不会被半读；
- 记录 writer/tracking 等待时间，供日志使用。

`third_party/ORB-SLAM3/src/GIAssociationState.cc:13-33`。

### 6.3 失效协议

| 变化 | 机制 | 结果 |
|---|---|---|
| KF 关联插入/删除/替换 | `KeyFrame::GIAssociationRevision` 递增 | 只重建该行 |
| MapPoint bad / replace | 全局 `badRevision` 递增 | 全部行重新读取，只有点集变化才 upsert |
| KF cull / removal | 写事务 + 元数据缺失 | 下次快照删除该行 |
| Map clear / PostLoad / NewMap | `Map::GIGeneration` 变化 | 旧行清空、全部重建 |
| 仅 pose/scale 变化 | revision 不变 | 不重建 |

测试 `tests/gi_association_state_test.cpp` 覆盖了插入、替换、擦除、bad、pose-only、KF culling、Map clear、Atlas map switch 和嵌套写事务互斥。

### 6.4 一致性边界

- 快照中只复制**稳定 ID**，不保存 ORB 裸指针；
- 离开读事务后才渲染 contribution；
- 渲染使用当前 Gaussian 状态，因此“关联快照”和“Gaussian 快照”不是同一时刻，但都在 `mutex_render_` 下，Gaussian 索引不会变；
- 没有后台线程、没有跨 boundary 的部分发布。

### 6.5 已知并发风险

- 任何新增的 ORB 关联修改路径都必须使用 `GIAssociationState::Write`，否则快照可能读到半事务；
- 未运行 ThreadSanitizer，测试通过不等于所有未来路径无竞态；
- `shared_timed_mutex` 的公平性依赖标准库实现；
- 所有 GI 工作同步执行在 Mapping 线程，没有多 stream 或异步发布。

---

## 7. Teacher 与 GI loss

### 7.1 构造（`src/gi_teacher.cpp:5-33`）

输入：

- `importance`：`[N] float32`；
- `valid_mask`：`[N] bool`；
- `observed_mask`：`[N] bool`；
- `maturity`：`[N] float32`；
- `ratio`：当前 LoD ratio；
- `entry`：`Elastic.entry_maturity`，默认 0。

步骤：

```text
P = { i | valid(i) ∧ observed(i) ∧ (entry = 0 ∨ maturity(i) ≥ entry) }
K = |P|
target_k = floor(ratio · K)

q = floor((1 - ratio) · (K - 1))
threshold = P 中 importance 第 (q+1) 小的值
label(i) = 1[ importance(i) > threshold ]   (i ∈ P)
label(i) = 0                                  (i ∉ P)

w(i) = clamp((maturity(i) - entry) / (1 - entry), 0, 1) · 1[i ∈ P]
       （entry = 0 时直接取 maturity(i)）
```

代码：`include/elastic_entry.h:4-9`、`src/gi_teacher.cpp:18-32`。

### 7.2 关键语义

1. **监督 population 只看 valid ∧ observed**（`entry=0` 时），不看 maturity 高低；
2. maturity 只作为**连续权重**，不把低成熟度行从 percentile 排序中剔除；
3. `maturity=0` 的行仍参与排序、仍占用 `K`，但梯度权重为 0；
4. `entry>0` 时才会把 `maturity<entry` 的行从 mask 中剔除，并重新缩放权重；
5. `target_k` 只用于诊断/统计，实际标签数由严格 `>` 的 percentile 决定，遇到并列可能少于 `target_k`。

第 3 点容易被误读：**“低成熟度点不参与 GI 监督”是错的**，准确说法是“低成熟度点参与排序但不产生梯度”。

### 7.3 Loss

```text
per_row = smooth_l1(prediction, label) 或 |prediction - label|
loss    = Σ w(i) · per_row(i) / (Σ w(i) + 1e-8)
```

代码：`include/gi_teacher.h:11-17`。

`prediction` 是 `selector_mask`：

- `sampleSelectorST(probability, cached_uniform)` 生成 0/1 硬采样；
- 前向是硬 mask，反向通过 `(probability - probability.detach())` 把梯度回传给概率；
- 因此 GI loss 实际训练的是“采样后的保留决策”，与 LoD 渲染使用的 mask 一致。

代码：`include/selector_handoff.h:14-21`、`src/gaussian_mapper.cpp:3067`。

### 7.4 与其它 loss 的组合

```text
selector_loss = render_weight · photo_loss
              + ratio_weight · ratio_loss
              + gi_loss_weight · importance_ratio_weight[ratio_index] · gi_loss
              + TF regularization
```

代码：`src/gaussian_mapper.cpp:3199-3215`。

默认：

- `Selector.gi_loss_weight = 1.0`；
- `Importance.ratio_aware_weight = 0` → `ratio_weight = 1.0`；
- 若开启 ratio-aware，则按 `Importance.ratio_weights` 对四个 ratio 分别加权。

### 7.5 数值例子

`N=100`，`K=100`，`ratio=0.05`：

```text
q = floor(0.95 · 99) = 94
threshold = 第 95 小的 importance
label = importance > threshold  → 约 5 个正样本
target_k = floor(0.05 · 100) = 5
```

若 maturity 为 `[0, 0.5, 1, ...]`，则对应权重为 `[0, 0.5, 1, ...]`；排序仍按全部 100 行，梯度只来自权重大于 0 的行。

### 7.6 版本字段

`GITeacher` 保存 `gi_version` 和 `topology_version`（`src/gi_teacher.cpp:25-26`），当前只作为诊断元数据，不参与 loss 计算。

---

## 8. 配置

### 8.1 参数表

| 参数 | 默认 | 约束 | 作用 |
|---|---:|---|---|
| `GlobalGI.mode` | `"disabled"`（缺省时）；当前配置为 `"streaming"` | `disabled` / `recent` / `streaming` | GI 来源 |
| `GlobalGI.view_ratio` | `0.30` | `(0, 1]` | Bank 占有效 KF 的比例 |
| `GlobalGI.coverage_anchor_fraction` | `0.80` | `[0, 1]` | Anchor 占 Bank 的比例 |
| `GlobalGI.sampling_seed` | `0` | 非负整数 | Representative 的 SplitMix64 种子 |
| `GlobalGI.refresh_interval` | `1000` | `> 0` | 全量刷新周期（Mapping 次数） |
| `GlobalGI.max_views` | `0` | 非负整数 | Bank 硬上限；0 表示不限制 |
| `GlobalGI.performance_logging` | `1` | 0/1 | 是否输出 `[GlobalGI]` 性能日志 |
| `GlobalGI.verbose_logging` | `0` | 0/1 | 是否输出每次 Bank/拓扑详情 |
| `RecentWindow.enable` | `0` | 0/1 | 仅 `recent` 模式需要 |
| `Selector.gi_loss_weight` | `1.0` | 浮点 | GI loss 总权重 |
| `Importance.ratio_aware_weight` | `0` | 0/1 | 是否按 ratio 加权 GI loss |
| `Importance.ratio_weights` | `[1,1,1,1]` | 长度匹配 ratio 数 | 各 ratio 的权重 |
| `Elastic.entry_maturity` | `0` | `[0,1)` | teacher 的 maturity 硬门槛与权重重缩放 |

补充：`cfg/selector/selector_config.yaml` 中 `GlobalGI.performance_logging: 1`，`cfg/selector/fast.yaml` 中为 `0`；两者都使用 `streaming`、`view_ratio=0.30`、`refresh_interval=1000`。生产运行若不需要 GI 分项日志，可使用 `fast.yaml` 的设置以减少同步读回。

### 8.2 配置校验

`include/global_gi_bank.h:25-33` 与 `src/gaussian_mapper.cpp:879-894`：

- mode 非法 → 抛异常；
- `view_ratio` 非有限或越界 → 抛异常；
- `coverage_anchor_fraction` 越界 → 抛异常；
- `refresh_interval=0` → 抛异常；
- `recent` 但 `RecentWindow.enable=0` → 抛异常；
- `seed/max_views` 为负 → 抛异常。

### 8.3 保存与加载

- 有效配置随 `saveSelectorConfig()` 写入 PLY 旁的 YAML（`src/gaussian_mapper.cpp:4426-4433`）；
- 加载时若文件缺少 `GlobalGI.*`，按 `disabled` 处理；
- 只读 viewer 仍可加载带 `streaming` 配置的 checkpoint，但不会重建 Bank。

### 8.4 COLMAP

`trainColmap()` 在 `streaming` 下直接抛异常（`src/gaussian_mapper.cpp:1169-1170`），因为离线 COLMAP 没有 ORB MapPoint 覆盖。COLMAP 训练必须显式选择 `recent` 或 `disabled`。

---

## 9. 日志、指标与报告工具

### 9.1 日志事件

前缀 `[GlobalGI]`，事件类型：

| event | 频率 | 内容 |
|---|---|---|
| `bank` | `verbose_logging=1` 时每次增量 | 增量渲染/聚合耗时、CUDA time |
| `bank_state` | `verbose_logging=1` 时每次增量 | 完整状态快照 |
| `topology` | `verbose_logging=1` 时每次拓扑 | remap/aggregate/total、点数、有效/无效点数 |
| `full_refresh` | 每次全量刷新 | 视角数、wall time、CUDA event time |
| `refresh_state` | `verbose_logging=1` 时每次刷新 | 完整状态快照 |
| `mapping` | 第 1 次及每 1000 次 | 周期汇总、P95、吞吐、GI 占比 |
| `final` | 保存/退出 | 最终汇总 |

### 9.2 关键字段

| 组 | 字段 | 说明 |
|---|---|---|
| Bank | `global_gi_total_kfs`、`global_gi_target_views`、`global_gi_anchor_views`、`global_gi_representative_views`、`global_gi_coverage_ratio` | 选帧结果 |
| Bank 成本 | `global_gi_bank_update_ms`、`global_gi_bank_snapshot_ms`、`global_gi_bank_selection_ms`、对应 `*_total_ms` | 快照 vs 选帧拆分 |
| 增量 | `global_gi_incremental_count`、`global_gi_incremental_total_ms`、`global_gi_incremental_render_total_ms`、`global_gi_incremental_aggregate_total_ms` | 新成员渲染与聚合 |
| 拓扑 | `global_gi_topology_update_count`、`global_gi_append/clone/split/prune_count`、`global_gi_topology_cumulative_ms`、`global_gi_topology_total_ms_mean/max/p95` | 拓扑同步 |
| 刷新 | `global_gi_full_refresh_count`、`global_gi_full_refresh_total_ms`、`global_gi_full_refresh_ms_mean/max` | 全量刷新 |
| 缓存 | `global_gi_cache_memory_mb`、`global_gi_cache_peak_memory_mb`、`global_gi_view_cache_estimate_mb` | 逻辑峰值，不是 CUDA allocator reserved 峰值 |
| 有效性 | `global_gi_valid_gaussians`、`global_gi_invalid_gaussians`、`global_gi_valid_observed_gaussians`、`global_gi_observed_gaussians`、`global_gi_valid_observed_fraction`、`global_gi_guidance_enabled` | 监督覆盖诊断 |
| 吞吐 | `mapping_iteration_ms_mean`、`mapping_iterations_per_second` | Mapping 吞吐，不是 Tracking FPS |
| 开销 | `global_gi_overhead_fraction` | 已计时 GI 成本 / Mapping wall |
| 锁 | `global_gi_map_wait_total_ms`、`global_gi_map_hold_total_ms`、`global_gi_association_wait_total_ms`、`global_gi_association_hold_total_ms`、`global_gi_association_writer_wait_total_ms`、`tracking_gi_association_wait_total_ms` | ORB 竞争 |

### 9.3 指标口径（容易误读）

1. `global_gi_overhead_fraction` 只包含 **Bank + 增量 + 拓扑 + 全量刷新**，**不包含** GI teacher 和 GI loss；
2. 它不是“相对 disabled 的速度降幅”，只是已计时 GI 操作占 Mapping wall 的比例；
3. `mapping_iterations_per_second` 是 Mapping 周期吞吐，不是 Tracking FPS；
4. `cache_peak_memory_mb` 不是进程显存峰值，也不包含 renderer scratch buffer；
5. `valid_gaussians` 是“估计有限”的行数，不是“被监督”的行数；被监督看 `valid_observed_*`。

### 9.4 报告工具

```bash
# 三模式日志 -> 中文 Markdown + CSV 时间序列
python scripts/report_global_gi.py \
  --disabled-log disabled.log \
  --recent-log recent.log \
  --streaming-log streaming.log \
  --output sgi-performance.md

# 在线 A/B 基准（服务器）
python scripts/benchmark_streaming_gi.py --root results/gi_runtime_20261012 \
  --label NEW_LABEL --head DEPLOYED_COMMIT --runs 3
python scripts/report_streaming_gi_benchmark.py results/gi_runtime_20261012
```

`tests/global_gi_report_test.py` 与 `tests/streaming_gi_benchmark_test.py` 只验证解析/生成逻辑，合成数据不是 GPU 性能结果。

---

## 10. 实测性能与验证现状

### 10.1 历史测量（不同配置，不可直接相加或混比）

| 日期 | 场景 | 关键结果 |
|---|---|---|
| 2026-10-10 | Replica room0，前 400 帧，2100 Mapping，三模式短测 | 26 KF，Bank=8（7 Anchor + 1 Rep），覆盖率 0.8687；拓扑 30 次（18/3/3/6），平均 0.3757 ms、P95 0.6104 ms、最大 1.9599 ms；两次刷新 25.4762/22.5004 ms；缓存 0.7001 MiB、逻辑峰值 1.4002 MiB；最终 17905 点 invalid=0、有效 Mature 比例=1、guidance=1 |
| 2026-10-10 | 用户 TUM 部分日志（Mapping 537–646） | 54 KF、17 视角（14+3）、覆盖率 0.8352；valid 1991/54338（3.664%）；Bank CPU 累计 4723.0 ms，增量 233.6 ms，拓扑 62.9 ms；`overhead_fraction=0.4645`；瓶颈是每次 Mapping 的 ORB 快照/锁 |
| 2026-10-12 | TUM fr1_room，1352 帧，基线与优化各 3 次 | GI 维护 `12.0825 → 1.0891 ms/正常 step`（`-90.99%`，`11.09×`），占 Mapping wall `58.17% → 8.35%`；含 teacher/loss `1.2706 ms`（`9.74%`）；完整 Mapping `20.7720 → 13.0466 ms`（`1.592×`） |
| 2026-10-12 | 同上，分项 | 快照 `10.5005 → 0.4296`；Bank 选帧 `1.1391 → 0.3409`；增量渲染 `0.3462 → 0.2259`；拓扑 `0.0687 → 0.0464`；周期刷新 `0.0280 → 0.0463`（ms/正常 step） |
| 2026-10-12 | 同上，锁竞争 | GI 对 `mMutexMapUpdate` 的 wait/hold 从 `6.4717 s / 21.5570 s` 降到 `0 / 0` |
| 2026-10-13 | 243 KF 合成选帧基准 | Anchor 关联变化平均 `306.34 → 24.16 ms`（`12.68×`）；候选变化 `1.17×`；无变化持平 |
| 2026-10-13 | Replica room0 短测，2100 Mapping | 33 KF、10 视角（8+2）、25711 Gaussian；刷新 31.0761/31.3755 ms；快照累计 9429.98 ms、选帧 842.14 ms（快照占 91.8%）；最终 invalid=0、guidance=1 |

**口径提醒**：

- 2026-10-12 的 `1.592×` 是完整 Mapping 吞吐，不是纯 GI 微优化收益；优化后同一在线时段完成了更多训练步，Gaussian/成熟度状态也不同；
- 2026-10-12 使用归档的 `server_fast.yaml`，当前 `fast.yaml` 已改动，不能直接复现；
- 2026-10-12 测量时 Selector 仍以 legacy 后端为主；2026-10-15 起 `Selector.backend` 默认改为 `cuda`。GI 路径未变，但 teacher/loss/优化器耗时不可直接与旧数字逐项对比；
- 2026-10-10 的 3.664% valid 是新生点尚未等到第 1000 次刷新的中间状态，不是最终状态；
- 2026-10-13 的选帧加速不含 ORB 快照读取，不能等同于整体 Mapping 加速。

### 10.2 质量对照（历史，单 checkpoint）

2026-10-12 同一 TUM 序列、各版本 1 个 checkpoint：

| 渲染 | Before PSNR / SSIM / LPIPS | After PSNR / SSIM / LPIPS |
|---|---|---|
| LoD 1% | 8.435 / 0.3394 / 0.8385 | 8.893 / 0.3377 / 0.8096 |
| LoD 5% | 8.704 / 0.3412 / 0.7992 | 9.837 / 0.4039 / 0.7692 |
| LoD 10% | 9.176 / 0.3619 / 0.7787 | 10.029 / 0.4087 / 0.7551 |
| LoD 15% | 9.429 / 0.3651 / 0.7744 | 10.196 / 0.4116 / 0.7509 |
| Full | 12.727 / 0.5103 / 0.7132 | 13.308 / 0.5289 / 0.6869 |

这是**单 checkpoint、单序列**对照，优化后训练推进更多、Gaussian 数更少（244781 → 187183），不能作为当前版本的普遍质量结论；1% SSIM 还略有下降。

### 10.3 自动化测试

当前 `CMakeLists.txt` 注册 56 个 CTest，其中 GI 相关 7 个：

| 测试 | 覆盖 |
|---|---|
| `global_gi_bank` | 配额、贪心覆盖、严格增益替换、Bottom-K、历史保留、覆盖率；对冻结的旧算法做 2000 次变化 + 2000 次无变化差分 |
| `global_gi_snapshot_cache` | 600 个 boundary，含 generation/reset、bad revision、删除/替换、批量 delta；与权威全量快照逐步比对 |
| `global_gi` | 增量不重渲染、成员替换、valid/observed、拓扑序列、NaN/负值、空 Bank、延迟刷新 |
| `gi_teacher` / `gi_teacher_cuda` | mask 组合、percentile 标签、连续 maturity 权重、零权重 loss、梯度隔离 |
| `gi_association_state` | 插入/替换/擦除/bad/pose-only/culling/Map reset/map switch、嵌套事务互斥 |
| `recent_window_importance` | legacy recent 的归一化、EMA、拓扑同步 |

历史服务器验证：

- 2026-10-10：28/28 CTest、`compute-sanitizer memcheck` 0 error；
- 2026-10-12：GI 回归 9/9 PASS，完整在线权威快照 3015/3015 boundary 一致；
- 2026-10-13：28/28 CTest（12.72 s）。

### 10.4 本次本地复核

2026-10-19，在本工作区用 MSYS2 GCC 15.2.0、`-std=c++17 -O2 -Wall -Wextra -pedantic` 编译并运行：

```bash
g++ -std=c++17 -O2 -Wall -Wextra -pedantic -I. tests/global_gi_bank_test.cpp -o bank_test
g++ -std=c++17 -O2 -Wall -Wextra -pedantic -I. tests/global_gi_snapshot_cache_test.cpp -o snapshot_test
./bank_test
./snapshot_test
```

```text
global_gi_bank_test            PASS（差分 2000+2000）
global_gi_snapshot_cache_test  PASS（600 boundary）
```

未在本机重新编译 CUDA/Torch 目标，也未重跑 GPU 在线测试；上述 GPU 数字来自历史服务器记录。

---

## 11. 与旧文档的差异（以当前代码为准）

| 项目 | 2026-10-10 / 10-14 文档 | 当前代码 |
|---|---|---|
| GI teacher 输入 | `mature_mask` 硬过滤（“Mature ∩ GIValid”） | `valid ∧ observed`，maturity 作为**连续权重**；`entry>0` 才硬过滤 |
| teacher 签名 | 含 mature/protected 参数 | `build(importance, valid, observed, maturity, ratio, gi_version, topology_version, entry)` |
| `observed` 来源 | 旧文档描述不统一 | 明确定义为 `importance > 0` |
| `recent` 模式 | 默认对比模式之一 | 2026-10-15 归档，日常配置只保留 `streaming` / `disabled` |
| ORB 快照 | 持 `mMutexMapUpdate`，逐 MP 锁 | 专用 `GIAssociationState` 事务 + revision/generation 缓存 |
| Bank 更新 | 每 boundary 全量 rows + 选帧 | 只重建变化行，boundary batch |
| 刷新调度 | 直接在 Mapping 后调用 | `DeferredMappingTask` + `force=true` |
| 日志 | 无快照/选帧拆分 | 新增 snapshot/selection、association lock、writer/tracking wait |
| Selector 默认 | 旧文档称 `Selector.enabled=0` | 当前 `selector_config.yaml` 为 1，`fast.yaml` 也为 1 |
| 测试数量 | 28 | 56（GI 相关 7） |

**结论**：阅读旧报告时，凡是涉及 teacher mask、`recent` 默认状态、ORB 锁和性能数字的部分，都应以本报告和当前代码为准。

---

## 12. 已知限制与风险

| 级别 | 问题 | 影响 | 当前缓解/建议 |
|---|---|---|---|
| 高 | 新生点最多 ~1000 次 Mapping 无 GI 监督 | 新区域/新物体学习慢；训练早期监督覆盖低 | 降低 `refresh_interval`；监控 `invalid_gaussians` 与刷新周期 |
| 高 | 缓存显存 `(4K+9)N`，K 随 KF 数线性增长 | 长序列/大地图可能 OOM；全量刷新瞬时约 2× 视角缓存 | 设置 `max_views`；监控 `cache_peak_memory_mb`；必要时降低 `view_ratio` |
| 中高 | Bank 用 ORB MapPoint 覆盖选视角，不是 Gaussian 可见性 | 视角可能覆盖地图点但看不到当前 Gaussian 分布；重要性有偏 | 提高 `view_ratio`；结合 `valid_observed_fraction` 判断 |
| 中 | 贡献缓存直到下次刷新才更新 | 几何/opacity 变化后 importance 变旧；时间序列混合不同 vintage | 缩短刷新周期；不要在两次刷新之间做精细质量归因 |
| 中 | 每个 boundary 都做 ORB 快照 | 地图越大、关联变化越多，CPU 成本越高（当前 0.43 ms/step） | 已做 revision 缓存；继续优化需减少全量 bad 重建 |
| 中 | 所有 GI 同步在 Mapping 线程 | 刷新造成周期性延迟尖峰；无后台/多 stream | 保持 `refresh_interval` 与训练预算匹配；避免在刷新边界做实时性要求高的操作 |
| 中 | 并发协议依赖所有 ORB 写路径使用 `GIAssociationState::Write` | 漏加会导致快照读到半事务 | 新增 ORB 关联修改必须审计；未来引入 TSan |
| 中 | importance 只用于排序，绝对值无跨场景可比性 | 不能直接拿阈值做质量判断 | 只用 percentile 标签；跨场景比较看监督覆盖率与渲染质量 |
| 低中 | 缓存不随 checkpoint 保存/恢复 | 重启后需重新建 Bank；短时间无 GI 监督 | 训练重启后关注前 1000 步的 guidance 状态 |
| 低 | COLMAP 不支持 streaming | 离线训练无法用 SGI | 显式使用 `recent`/`disabled` |
| 低 | `valid=true` 可能只是“有限”，不是“正确” | 误读日志会高估监督覆盖 | 同时看 `observed` 相关字段 |
| 低 | 并列 importance 使正标签数少于 `target_k` | 实际正样本略少 | 属于 percentile 语义，通常影响很小 |
| 低 | 只覆盖当前 Atlas Map，不合并不同 Map 的坐标系 | 多 Map 场景的全局重要性只对当前 Map 有效 | 单 Map 在线训练；跨 Map 需额外对齐设计 |

---

## 13. 调参与运维建议

### 13.1 何时用哪种模式

- **在线 ORB-SLAM3（RGB-D/单目/双目）**：`streaming`，默认参数即可；
- **COLMAP 离线**：`disabled`；
- **只想验证 Selector 而不想付 GI 成本**：`disabled`；
- **需要复现旧实验**：`recent` + `RecentWindow.enable=1`，但注意该配置已归档。

### 13.2 参数调节

| 目标 | 调整 | 代价 |
|---|---|---|
| 降低 GI 成本/显存 | `view_ratio` 0.30 → 0.20/0.15；或 `max_views` 16–24 | 覆盖率下降，importance 噪声增大 |
| 加快新生点响应 | `refresh_interval` 1000 → 500 | 全量刷新更频繁，周期性尖峰更多 |
| 提高视角多样性 | 降低 `coverage_anchor_fraction`（如 0.6–0.7） | 覆盖收益下降，代表性更随机 |
| 稳定复现实验 | 固定 `sampling_seed` | 改变 seed 会换 representative 集合 |
| 减少日志开销 | `performance_logging: 0` / `verbose_logging: 0` | 失去诊断字段 |
| 调整各 ratio 的 GI 权重 | `Importance.ratio_aware_weight: 1` + `ratio_weights` | 需要重新调参验证 |

### 13.3 上线监控清单

1. `global_gi_valid_observed_fraction`：应随训练上升；长期偏低说明 Bank 看不到 Gaussian；
2. `global_gi_guidance_enabled`：应为 1；为 0 检查 Selector、`gi_loss_weight`、优化预算；
3. `global_gi_invalid_gaussians`：刷新后应接近 0；持续偏高说明拓扑/NaN 问题；
4. `global_gi_bank_snapshot_ms` vs `selection_ms`：判断剩余 CPU 瓶颈；
5. `global_gi_full_refresh_ms`：观察是否随地图增长恶化；
6. `global_gi_cache_peak_memory_mb`：与进程显存一起看，不能只看这一项；
7. `global_gi_overhead_fraction`：目标 < 10%；注意它不含 teacher/loss。

### 13.4 排查路径

| 现象 | 优先检查 |
|---|---|
| GI loss 一直为 0 / guidance=0 | `Selector.enabled`、`gi_loss_weight`、`valid_observed_fraction`、Bank 是否为空 |
| 新生区域学习慢 | `invalid_gaussians` 是否在刷新前很高；`refresh_interval` 是否过大 |
| 显存持续上涨 | `cache_peak_memory_mb`、`K`、`N`；是否设置了 `max_views` |
| Mapping 周期性卡顿 | `full_refresh_ms`、`refresh_interval`、刷新时是否与 Full render 重叠 |
| 日志中 coverage_ratio 骤降 | KF 失效/Map 切换/关联 revision 大量变化 |
| `bank_snapshot_ms` 高 | ORB 关联写竞争、bad 重建、地图规模 |

---

## 14. 关键文件索引

| 文件 | 作用 |
|---|---|
| `include/global_gi.h` | `GlobalGIManager` 接口、`GIViewCache`、`GITopologyKind` |
| `src/global_gi.cpp` | 增量聚合、拓扑 remap、全量刷新、日志与计时 |
| `include/global_gi_bank.h` | `GlobalGIConfig`、`GlobalGIViewBank`（配额/覆盖/Bottom-K） |
| `include/global_gi_snapshot_cache.h` | 稳定 ID + revision/generation 的 boundary 缓存 |
| `src/gi_teacher.cpp`、`include/gi_teacher.h` | teacher mask/label/weight 与 loss |
| `include/elastic_entry.h` | 连续 maturity 权重与 entry 缩放 |
| `include/recent_window_importance.h`、`src/recent_window_importance.cpp` | legacy `recent` 模式 |
| `src/gaussian_mapper.cpp` | 调用链、配置读取、ORB 快照、teacher/loss 组合、保存 |
| `third_party/ORB-SLAM3/include/GIAssociationState.h` | GI/ORB 关联读写屏障 |
| `third_party/ORB-SLAM3/src/GIAssociationState.cc` | 嵌套事务与等待计时 |
| `third_party/ORB-SLAM3/src/KeyFrame.cc`、`Map.cc`、`Atlas.cc` | revision/generation、GI 读接口 |
| `cuda_rasterizer/forward.cu` | `frame_importance` / `contribution_count` 累加 |
| `cuda_rasterizer/rasterize_points.cu` | 统计张量分配与 Visibility 限制 |
| `src/gaussian_rasterizer.cpp`、`src/gaussian_renderer.cpp` | Inference 采集 GI、SH/covariance 适配 |
| `include/mapping_profiler.h` | GI/GITeacher/GILoss 等阶段计时 |
| `include/ultra_light_schedule.h` | `DeferredMappingTask` |
| `tests/global_gi_bank_test.cpp`、`tests/global_gi_test.cpp`、`tests/global_gi_snapshot_cache_test.cpp` | 算法与生命周期回归 |
| `tests/gi_teacher_test.cpp`、`tests/gi_association_state_test.cpp` | teacher 语义与并发协议 |
| `scripts/report_global_gi.py`、`scripts/benchmark_streaming_gi.py`、`scripts/report_streaming_gi_benchmark.py` | 报告与基准工具 |
| `include/online_gi.h`、`src/online_gi.cpp` | 遗留模块，当前 Mapper 未使用 |

---

## 附录 A：一个 boundary 的伪代码

```text
def on_mapping_boundary():
    with GIAssociationState.Read():                 # 独占读，排除关联写者
        map = Atlas.GetCurrentMapForGI()            # 非阻塞；null 则释放后重试
        metadata = []
        for kf in map.GetAllKeyFrames():
            if kf.GIBad(): continue
            if not registered_in_mapper(kf): continue
            if not camera_valid(kf): continue
            metadata.append((kf.id, kf.GIAssociationRevision()))

        batch = snapshot_cache.update(
            map.GIGeneration(),
            GIAssociationState.badRevision(),
            metadata,
            read_points = kf.GIPointIds)

    members = view_bank.updateBatch(batch)
    for id in members - cached_views:
        cached_views[id] = render_contribution(id)   # 只渲染新增
    for id in cached_views - members:
        sum -= cached_views[id]
        erase
    sum += new_contributions
    mean = sum / len(members)
    valid = valid & finite(new_contributions)   # 新行在 append 时已为 false

def maybe_refresh():
    if mapping_iter % refresh_interval != 0: return
    next = {}
    next_sum = zeros(N)
    next_valid = ones(N)
    for id in bank.members():
        c = render_contribution(id)
        next[id] = c
        next_sum += c
        next_valid &= finite(c)
    mean = next_sum / K
    publish(next, next_sum, mean, next_valid & finite(mean))
```

## 附录 B：快速判断当前配置是否在跑 SGI

启动日志中查找：

```text
[GlobalGI CONFIG] mode=streaming view_ratio=0.3 anchor_fraction=0.8 refresh_interval=1000 verbose_logging=0
```

若 `mode=disabled`，则不会创建 `GlobalGIManager`，也不会出现后续 `[GlobalGI]` 事件。
