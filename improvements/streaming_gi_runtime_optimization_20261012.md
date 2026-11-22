# Streaming GI 在线运行时优化报告

## 结论与验收

在服务器 RTX 3090 上，完整 TUM fr1_room、相同配置、相同在线 LoD 轮转，基线与优化版各运行三次。逐指标中位数显示：

- complete Mapping mean：20.7720 -> 13.0466 ms，吞吐提升 **1.592139x**。严格的 >=1.60x 目标未达到；相对于本次实际基线，该目标要求 <=12.9825 ms，不能用早先 21.4343 ms 的参考数替代实际基线。
- GI maintenance：12.0825 -> 1.0891 ms/正常 Mapping step，降低 **90.9863%**，加速 **11.0942x**；占 Mapping wall 的 58.17% -> **8.35%**。
- 纳入单独测量的 teacher 和 GI loss 后，优化版 GI 为 1.2706 ms/step，占 **9.74%**。Mature preparation 另列，不混入 teacher。
- GI 已成为次要开销。LoD backward 的 host wall 为 5.8443 ms/正常 step；LoD 整段、Full render/backward 和 optimizer 合计约占 Mapping wall 的 80%。这构成本次停止优化 GI 的实测依据，不代表 1.60x 在物理上不可达到。
- 满足任务的强成功条件：GI 降低 >=70%、占比 <=20%、训练成为主要开销，GI 引起的 Tracking 锁竞争显著降低。正常训练 tail steps 在全部完成运行中均为 **0**。
- 服务器回归 **9/9 PASS**；完整在线权威快照对照 **3015/3015 Mapping boundaries 一致**。

## 基线、配置与可比性

基线 HEAD 为 `6294748c3ca292d73084665b7e7f74aa3438f117`，本地分支为 `remove-full-render`。服务器项目 `/home/prmi/zwj/mycartgs` 不包含自身 Git 元数据，使用本地 commit、source SHA256 和 binary SHA256 关联部署版本。

环境：服务器 `172.19.163.58`、conda `zwj`、RTX 3090、CUDA 11.8、GCC 9、LibTorch 2.3.1。数据集为 `/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_room`，association 输入 **1352 帧**，全部完成运行的 Tracking 帧数也为 1352。保持数据回放、正常 Mapping clock、1%/5%/10%/15%/Full 轮转和无尾训练。

使用服务器原有 `cfg/selector/fast.yaml`，没有调整 GI 或训练配置。关键值：Streaming GI、view_ratio=0.30、anchor_fraction=0.80、refresh_interval=1000 Mapping iterations、max_views=0、sampling_seed=0、在线 LoD scale=0.5、TF start_iter=800、Protection age/optimization/KF=800/300/5。

**本地默认 fast.yaml 与服务器配置不同**：本地 ultra_light=0、TF start_iter=3000、Protection=100/80/4、Replay=1；服务器 ultra_light=1、TF start_iter=800、Protection=800/300/5、Replay=0。复现此报告必须使用归档的服务器配置，不能直接把本地默认配置当成 benchmark 配置。完整配置保存在 `evidence/gi_runtime_20261012/server_fast.yaml`，其他两份配置也已归档。

| 配置 | 基线及优化版相同的 SHA256 |
|---|---|
| Selector fast.yaml | `1f66ab767f64106427f88f5453793b9808789f9503da6ec54906919c0e72ac71` |
| Gaussian Mapper TUM RGB-D | `4b4a812edcf5e1ffc4357625800e447971d3b954b3172705ce901b1eacebbfeb` |
| ORB-SLAM3 TUM fr1_room | `9002890cd1537cbc7e4a5a4117e22fd3a351aca7b7204fc42101acaca72f4f17` |

`baseline_manifest.json`、`instrumented_manifest.json`、`optimized_manifest.json` 保存 source/config/binary hashes。优化测量时的 manifest 标签为 `8afccf4+gi-snapshot-batch`；该候选核心 source hashes 已逐文件核对，与提交 `baa916a5aefd1a7f2209fc3aef2f5c3e1276396d` 一致。保留原始标签，不改写测量历史。

在线系统异步运行，优化后在同一输入时长内完成更多正常训练，Gaussian/TF/Mature 状态随之变化。因此这是相同在线输入与方法设置的吞吐比较，不是固定 Gaussian 状态或固定训练步数的微基准。没有通过限步、减少输入、冻结 Bank 或降低 GI 频率制造加速。下表完整列出实际工作量；后续所有 GI 数据均逐轮按实际正常训练步数归一化，再取中位数。

| Native run | 正常 steps | Mapping mean ms | KF | Gaussian | GI Bank | Selector / TF 在线更新 | tail |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline 1 | 2796 | 20.7720 | 234 | 244781 | 71 | 937 / 937 | 0 |
| baseline 2 | 2953 | 19.2793 | 221 | 236853 | 67 | 971 / 971 | 0 |
| baseline 3 | 2664 | 21.4655 | 240 | 246201 | 72 | 872 / 872 | 0 |
| optimized 1 | 4294 | 13.1514 | 239 | 172837 | 72 | 2236 / 2236 | 0 |
| optimized 2 | 4327 | 13.0466 | 232 | 187183 | 70 | 2198 / 2198 | 0 |
| optimized 3 | 4362 | 12.9475 | 227 | 173535 | 69 | 2290 / 2290 | 0 |

## 原瓶颈与实际调用链

当前 clean HEAD 已包含 Ratio Rotation；本任务没有恢复旧 Full+LoD 或增加尾训练。GI 原调用链是：

```text
trainForOneIteration：原 KF scheduler 选择 KF
  -> trainMappingIteration
  -> updateGlobalGIBank
       -> 获取 Atlas 当前 Map
       -> 等待并持有 Map::mMutexMapUpdate
       -> GetAllKeyFrames
       -> 对所有 eligible KF 调用 GetMapPoints
       -> 每个 MP 做 bad 检查、锁访问、set 构建
       -> 为全部 KF 构造并排序/去重完整 association snapshot
       -> GlobalGIManager::updateBank：处理全量 rows、Bank 选择
       -> 对新 Bank member 渲染 contribution、增量 aggregate
  -> 本轮 LoD 或 Full 分支
  -> LoD 中构造 Mature/Protection/teacher，backward 与 optimizer
  -> Full 中 kfs_loss、densification stats 与 topology
  -> 按原 mapping iteration clock 执行 GI periodic refresh
```

复现结果中，snapshot 达 10.5005 ms/step，Bank processing 为 1.1391 ms/step，而增量 GI render/aggregation 仅 0.3462 ms/step。主要瓶颈是 CPU association 重建和共用 map mutex，非 GI rasterizer。instrumented baseline 记录 GI map-lock wait=6.472 s、hold=21.557 s，Tracking map-lock wait=5.534 s。

## 最终架构与语义

### 一致 association snapshot

新增 ORB 专用 `GIAssociationState` barrier：association writers 获取 shared lock，GI reader 获取 exclusive lock。原 ORB 对象锁继续负责 writer 之间的同步；barrier 用于排除 snapshot 与正在进行的 association mutation。通过 thread_local 嵌套深度，bad/replace 等嵌套事务只在最外层锁定/释放，避免半事务读取。

GI 不再申请 `mMutexMapUpdate`。同一个 GI read transaction 内捕获当前 Map、eligible KF metadata 和变化行；`GIPointIds()` 直接读取受 barrier 保护的数据，取消逐 MP 通用锁与临时 `std::set`。离开事务后只传递稳定 IDs，持久缓存不保存 ORB 裸指针。

每 KF 使用 association revision；MP bad/replace 使用全局 bad revision 保守失效所有行，即使反向 observations 不完整也不会漏变更。Map 使用独立 generation，clear/PostLoad 重新生成，避免 map switch/reset 和 ID 复用污染缓存。

| 状态变化 | 更新/失效机制 |
|---|---|
| 新 KF、Mapper 延迟注册或相机数据变为可用 | 每 boundary 重新检查 eligible KF metadata；新行重建 |
| association 插入、删除、替换 | KF revision 增加；仅变化行重建 |
| MapPoint::SetBadFlag / Replace | 嵌套写事务 + 全局 bad revision；全部行保守重建 |
| KF culling/removal | 写事务；下次 metadata 中移除该行 |
| Map switch、clear、PostLoad | 原子事务边界 + 唯一 generation；旧行移除，新行重建 |
| 仅 pose/scale 变化 | 不改 association revision；原 Mapping camera 更新及 refresh 机制保持 |
| Gaussian reset | 原 GI reset 清 contribution；empty batch 也检查并重建缺失 contribution |
| append/clone/split/prune | 原 topology 同步和索引语义保持，由现有测试验证 |

### Boundary batch 与 Bank

`GlobalGISnapshotCache::update()` 只重建 revision 变化的行，生成完整 boundary 的 upserts/removals。`GlobalGIViewBank::updateBatch()` 先应用整个 batch，再运行一次原全局 coverage/replacement/representative selection，保留 deterministic ordering 和 Bottom-K sampling，不逐事件贪心发布结果。

没有后台 GI、bounded staleness 或跨 boundary 的部分发布。GPU contribution/aggregate、teacher、topology、refresh 的数学定义保持原逻辑；view_ratio、anchor 和 representative 行为没有改变。每个正常 boundary 仍处理 Bank，无变化时只跳过等价的重复 association/container 工作。

`Atlas::GetCurrentMapForGI()` 在读事务内非阻塞获取有效 Map。通用 `GetCurrentMap()` 的等待改为释放 Atlas mutex 后等待，避免等待 ChangeMap 时持锁；GI retry 也先释放读事务，不阻塞 map switch/reset 的 writers。

## 尝试、取舍与修改文件

实测保留一套完整的 revision cache + 事务 barrier + batch Bank 实现，没有维护 legacy runtime 分支，也没有声称做过不存在的失败 A/B 实验。

| 方案 | 结果与依据 |
|---|---|
| 增加轻量 host 分项与抽样 CUDA events | 保留；定位真实 CPU/锁瓶颈并单列 teacher |
| association cache + 专用 ORB 读取 + boundary batch | 保留；native 和权威对照均通过，GI 归一化降低 90.99% |
| 修复 empty LoD SH backward shape | 保留；原 `[0,15,3]` 被误返回 `[0,0,3]`，仅修复 empty 情况；非空行为不变 |
| 仅把 set 换 vector、仍每步全量重建 | 审阅后未实施；无法消除重复遍历与主 map-lock 等待 |
| 后台异步 GI / 延迟发布 | 未实施；当前 GI 已次要，引入生命周期和一致性复杂度没有实测必要 |
| compact teacher / 更多 Tensor 缓存 | 未实施；teacher 约 0.11 ms/正常 step，收益上限远小于 LoD backward；保留原 teacher 语义 |

主要 API 和文件：

- `src/gaussian_mapper.cpp::updateGlobalGIBank`：事务读取、cache batch、权威对照、锁/teacher/训练分项统计。
- `include/global_gi_snapshot_cache.h::GlobalGISnapshotCache`：稳定 ID、revision/generation 行缓存。
- `include/global_gi_bank.h::GlobalGIViewBank::updateBatch`、`src/global_gi.cpp::GlobalGIManager::updateBankBatch`：完整 batch 处理、无变化路径与 reset bootstrap。
- `third_party/ORB-SLAM3/include/GIAssociationState.h`、`src/GIAssociationState.cc`：barrier、嵌套事务、revision/generation 和等待计数。
- ORB `KeyFrame`、`MapPoint`、`Map`、`Atlas` 对应 `.h/.cc`：GI 读取接口及完整失效钩子；`Tracking.cc`：锁等待记录。
- `include/mapping_profiler.h`：抽样 CUDA 区间、host mean/P95、Mature/teacher/loss/render/backward 分项。
- `cuda_rasterizer/rasterize_points.cu`：empty SH backward shape 修复。
- `CMakeLists.txt`、ORB `CMakeLists.txt` 及相关 GI/rotation tests：构建和回归覆盖。
- `scripts/benchmark_streaming_gi.py`、`report_streaming_gi_benchmark.py`、`eval_headless.py`：版本记录、逐轮归一化中位数、原评价流程的无界面包装。

## 性能结果与测量口径

下面除特别标注外均为各组三次成功 native 运行的逐指标中位数。不同指标可能来自不同轮，表内中位数不能当成同一轮做精确求和。complete Mapping 包含边界处理；原 `mean_mapping_training_ms` timer 包含 `trainMappingIteration` 内的 GI，不能再把 GI 加到该数值上。

| 指标 | Before | After |
|---|---:|---:|
| complete Mapping mean ms | 20.7720 | 13.0466 |
| complete Mapping P95 ms | 50.2608（instrumented baseline 单轮） | 30.9733 |
| normal training mean ms | 18.9071 | 11.7934 |
| Mapping throughput steps/s | 48.1417 | 76.6483 |
| GI maintenance total wall s | 33.7828 | 4.7125 |
| GI ms / normal step | 12.0825 | 1.0891 |
| GI share of Mapping wall | 58.1674% | 8.3477% |
| process wall s | 65.1575 | 64.2811 |
| GPU utilization，1 s 采样均值 | 17.7424% | 31.6462% |
| Tracking mean ms | 23.7050 | 21.3045 |
| Tracking P95 ms | 40.2000 | 30.5000 |
| GI view-cache estimate MB | 66.2973 | 47.4711 |
| GI peak cache memory MB | 69.5655 | 72.3448 |

GI 绝对 wall 降低约 86.05%，归一化降低 90.99%；区别来自优化版完成更多 Mapping steps。process wall 仅小幅降低，因为输入回放占用固定时间，Mapping 吞吐改善主要表现为同一在线时段完成更多有效更新。

| GI maintenance 分项，ms/正常 step | Before | After |
|---|---:|---:|
| KF/MapPoint snapshot / state maintenance | 10.5005 | 0.4296 |
| Bank selection / processing | 1.1391 | 0.3409 |
| Incremental GI render + aggregation | 0.3462 | 0.2259 |
| Topology synchronization | 0.0687 | 0.0464 |
| Periodic full refresh | 0.0280 | 0.0463 |

refresh 次数中位数由 2 -> 4，按原 1000 Mapping iteration clock 触发；绝对 refresh wall 增加是训练推进更快的结果，没有拉长 interval。

teacher 新分项的 Before 仅来自 **1 次 instrumented baseline**，After 为 3 次中位数；该 baseline complete mean=20.5112 ms，与原 native 基线接近，但不是原三次样本。

| Teacher/训练分项，host wall ms/正常 step | Before（单轮） | After |
|---|---:|---:|
| Mature preparation | 0.1498 | 0.1101 |
| GI teacher build：supervised indices、ranking、labels/masks | 0.0958 | 0.1115 |
| GI loss preparation / indexing | 0.0434 | 0.0717 |
| GI maintenance + teacher + GI loss | 11.7209 | 1.2706 |
| 上述合计占 Mapping wall | 57.1437% | 9.7388% |
| LoD 整段（内含 teacher/loss/render/backward） | 5.5172 | 8.8681 |
| 其中 LoD backward | 3.2448 | 5.8443 |
| Gaussian / Selector / TF optimizer | 0.7878 | 1.2496 |
| Full render | 0.1515 | 0.1350 |
| Full backward | 0.1561 | 0.1488 |
| Mapping operations | 1.8082 | 1.0669 |

teacher 保留现有 full-size label/mask，没有用不完整 cache key 缓存结果。本次 build/loss scope 足以确认它不是主瓶颈，未把约 0.1 ms 工作再拆成噪声较大的逐 kernel 计时。

CUDA 每 19 个 boundary 抽样，避开五 phase 混叠；异步 event 收集，没有为本任务新增每 Mapping step 的 `cudaDeviceSynchronize`。保留 clean HEAD 原 `UltrafastTrainingTimer` 的同步行为，两组一致。优化版 sampled LoD backward CUDA elapsed=7.3480 ms/抽样 LoD step，optimizer=1.2756 ms/抽样 step；这是 event 区间，包含 host launch gaps，不代表独占 kernel 时间，也不能与按全部正常 steps 归一化的 host wall 直接相加。

| 原 phase training timer，ms/对应 phase | Before | After |
|---|---:|---:|
| LoD 1% | 19.6223 | 13.2243 |
| LoD 5% | 19.1452 | 13.6486 |
| LoD 10% | 19.3090 | 13.7604 |
| LoD 15% | 20.0416 | 13.7113 |
| Full | 16.4158 | 4.5408 |
| 正常 Full render count | 559 | 865 |
| 正常 LoD render count | 2237 | 3462 |

这里的 render counts 是正常训练分支，GI refresh/incremental 和评价的独立渲染另行记录。没有在 LoD 前后追加 Full 正常训练。

## Tracking 竞争

| 锁计数，累计时间 | Before（instrumented 单轮） | After（3 次中位数） |
|---|---:|---:|
| GI 对 mMutexMapUpdate 的 wait | 6.4717 s | 0 |
| GI 对 mMutexMapUpdate 的 hold | 21.5570 s | 0 |
| Tracking 对 mMutexMapUpdate 的 wait | 5.5344 s | 1.9428 s |
| GI association reader wait | 未存在 | 2.7817 ms |
| GI association reader hold | 未存在 | 1.8456 s |
| 全部 association writers wait | 未存在 | 265.9024 ms |
| Tracking association writer wait | 未存在 | 3.9304 ms |

Tracking 的主 map-lock 等待降低约 64.9%，同时新增 barrier 的 Tracking 等待仅累计约 3.93 ms。不是把原大额等待转移到新锁；其他 ORB/Mapping 操作仍可能使用 map-lock，因此 Tracking 总等待不会归零。

## 正确性与执行命令

服务器在已有构建目录完成 ORB 和 CaRtGS native build，最终测试日志见 `evidence/gi_runtime_20261012/candidate_regression.log`。

```bash
conda activate zwj
cd /home/prmi/zwj/mycartgs
cmake --build third_party/ORB-SLAM3/build-cuda118-sm86-gcc9
cmake --build build-cuda118-sm86-gcc9
cd build-cuda118-sm86-gcc9
ctest -R '^(global_gi_bank|global_gi_snapshot_cache|gi_association_state|global_gi|gi_teacher|mapping_profiler|renderer_readonly_cuda|ultrafast_mapping_cuda|ultrafast_rotation)$' --output-on-failure
```

| 验证 | 结果 |
|---|---|
| global_gi_bank | PASS：2000 changed/empty + 2000 unchanged snapshots，成员/anchor/representative/coverage 与独立原策略 reference 一致 |
| global_gi_snapshot_cache | PASS：600 个混合多事件 boundary；new/changed/bad/replaced/removed/reset/no-change，完整 batch 对照 |
| gi_association_state | PASS：实际 ORB 插入/删除/replace/bad/culling/reset/switch；反向 observations 缺失；嵌套事务排他 |
| global_gi | PASS：aggregate、valid Gaussian indices、append/clone/split/prune、Gaussian reset/bootstrap、periodic refresh |
| gi_teacher | PASS：现有 teacher 行为及有效性回归 |
| mapping_profiler / renderer_readonly_cuda | PASS：计时与只读渲染回归 |
| ultrafast_mapping_cuda / ultrafast_rotation | PASS：真实参数更新、Full 辅助 Adam freeze、KF 独立、kfs_loss、topology、empty LoD、no-tail |
| 完整在线权威 snapshot | PASS：1352 帧、3015/3015 boundaries 一致、exit=0、tail=0 |
| 本地 benchmark parser/report tests | PASS：`python -B tests/streaming_gi_benchmark_test.py`，teacher 单位、逐轮归一化、失败运行排除 |

权威 snapshot 使用 `CARTGS_GI_VALIDATE_SNAPSHOT=1`，在同一事务中对照原 `GetMapPoints()` 全量重建，验证运行不参与性能统计；三次优化 native 均关闭该开关。

Native benchmark 工具保留用户原命令，只把输出目录隔离到每轮；每轮 `command.json` 记录实际命令。性能汇总调用：

```bash
cd /home/prmi/zwj/mycartgs
python scripts/benchmark_streaming_gi.py --root results/gi_runtime_20261012 --label NEW_UNIQUE_LABEL --head DEPLOYED_COMMIT --runs 3
python scripts/report_streaming_gi_benchmark.py results/gi_runtime_20261012
```

`NEW_UNIQUE_LABEL` 为复现时的新目录名，不复用已有证据目录；baseline/optimized 切换须部署对应 commit 后重新构建，并使用同一归档服务器配置。

## 渲染质量

使用已有 `python/eval_lod.py` 与 `python/run.py` 评价完整 1352 帧。`scripts/eval_headless.py` 只在进程内将 evo backend 设为 Agg，不修改 `~/.evo/settings.json`。质量评价与 native benchmark 错开运行。

按 Mapping mean 的中位样本选择 baseline run 1 和 optimized run 2，分别为 244781 与 187183 Gaussians；这是每版本 **1 个 checkpoint** 的相对对照，不是多次质量均值。

| 渲染 | Before PSNR / SSIM / LPIPS | After PSNR / SSIM / LPIPS | active Before -> After | actual/target Before -> After |
|---|---|---|---|---|
| LoD 1% | 8.43545 / 0.339423 / 0.838468 | 8.89254 / 0.337683 / 0.809608 | 1021 -> 2231 | 0.417108 -> 1.191882 |
| LoD 5% | 8.70353 / 0.341229 / 0.799200 | 9.83710 / 0.403947 / 0.769222 | 4491 -> 6294 | 0.366940 -> 0.672497 |
| LoD 10% | 9.17631 / 0.361916 / 0.778679 | 10.02873 / 0.408703 / 0.755071 | 8748 -> 10037 | 0.357381 -> 0.536213 |
| LoD 15% | 9.42859 / 0.365066 / 0.774438 | 10.19580 / 0.411557 / 0.750914 | 12858 -> 13287 | 0.350191 -> 0.473227 |
| Full | 12.72733 / 0.510306 / 0.713191 | 13.30838 / 0.528938 / 0.686899 | 244781 -> 187183 | 1 -> 1 |

Full 和各 LoD 的 PSNR/LPIPS 均改善，1% SSIM 下降约 0.00174。未观察到本组 checkpoint 的系统性质量退化，但绝对质量仍偏低，不据此声称方法质量优秀。active/target 偏差来自保留的 soft Selector/Protection 行为，不在本任务重新设计 budget。

baseline 评价 Mature=150，TF transform_applied=false；optimized Mature=4608，四种 LoD TF=true、Full=false。优化后更多正常训练推进原 readiness clock，Selector/TF 实际在线更新明显增多，LoD backward 与 optimizer 单步开销也增加。Gaussian 减少伴随 prune 次数中位数 8 -> 14，未调整拓扑规则、densify/prune 频率配置或 GI workload；不能把整个变化仅归因于 GI 计算微优化。

复现评价命令示例（两个模型各执行一次，输出不同目录）：

```bash
python python/eval_lod.py results/gi_runtime_20261012/baseline_retry/run_1/model /home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_room --run-script scripts/eval_headless.py --output results/gi_runtime_20261012/quality_baseline --ratios 0.01 0.05 0.10 0.15
python python/eval_lod.py results/gi_runtime_20261012/optimized/run_2/model /home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_room --run-script scripts/eval_headless.py --output results/gi_runtime_20261012/quality_optimized --ratios 0.01 0.05 0.10 0.15
```

使用 checkpoint 随附 Selector 配置；ultrafast 评价流程自动追加 Full。最终 checkpoint 来自 online-end 状态，没有通过 post-online optimization 修复质量。

## Git、证据与剩余风险

逻辑分离提交：

1. `6e8e465`：native GI stage、teacher 和 map contention instrumentation。
2. `8afccf4`：empty LoD SH 梯度修复、隔离 rotation fixture 与服务器配置差异。
3. `baa916a`：一致 association cache、ORB 事务接口与 Bank batch 优化。
4. 本报告所在的交付提交：benchmark/report/headless 工具、测试及证据归档；确切 hash 见当前 `git log` 和最终交付消息。

未 push。未跟踪的 `build-cuda118-sm86.zip` 保留。

本地证据目录：`improvements/evidence/gi_runtime_20261012/`。`comparison.json` 保存全部成功运行、逐轮归一化、逐指标中位数、验证运行和两份完整质量 summary；三份 manifest 保存源代码与配置 hashes。完整 run logs、模型、GPU 采样、`time.txt` 和逐轮 `command.json` 保留在服务器 `/home/prmi/zwj/mycartgs/results/gi_runtime_20261012/`。

风险与限制：

- 严格 1.60x 理想目标未达，实测为 1.592139x；当前主要剩余瓶颈是 LoD backward/训练，继续增加 GI 架构复杂度收益有限。
- 本次性能依据为一个完整 TUM 序列；质量依据为每版本一个 checkpoint。其他场景、长序列和 KF 更多的地图需要额外验证。
- 原始基线最早一次出现 Sophus `SO3::exp` NaN、exit=134，之后同参数三次完整运行成功；失败日志保存在服务器 `baseline/run_1`，未纳入成功样本中位数。没有定位上游原因，不能归因于 GI。
- MP bad/replace 采用全局 conservative invalidation：变化频繁时仍会重建较多 rows，但正确性优先；当前 snapshot 已低于 0.5 ms/step。
- 新增 ORB association 修改入口必须加入事务/revision 协议；现有路径经过静态审计、实际 mutation 测试和完整在线权威对照。未运行 ThreadSanitizer，不能把测试通过等同于所有未来路径无竞态。
- `shared_timed_mutex` 公平性依赖实现；本次 reader/writer/Tracking wait 已实测很小，仍应在更高写入压力下观察。
- CUDA event 分项不是独占 kernel profile；未用 Nsight 每轮测量，也未通过逐步新增全局同步干扰 native 性能。
- 无新增 GI 频率限制、Bank 冻结、Full-only GI 或 teacher 禁用。所有正常 rotation counts 均衡，tail=0。
