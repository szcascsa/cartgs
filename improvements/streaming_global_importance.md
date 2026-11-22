# Streaming Global Importance 实现与验证报告

日期：2026-10-10。工作分支：`large-scale-adjustment`。检查基线：`9c7a73f`。

## 结论与验收状态

设计没有阻止实施的结构冲突，已完成 SGI 代码接入、配置、测试用例和性能报告工具。服务器完整编译、28 项 CTest、性能日志解析测试和 CUDA renderer 内存检查均通过。

目标机 `172.19.163.58:/home/prmi/zwj/mycartgs/` 已完成 CUDA 11.8 / sm_86 / LibTorch 2.3.1 构建和真实数据短测。三个 GI 模式各完成 2100 次 Mapping，Streaming 第 1000、2000 次完整刷新后 invalid 都为零。真实成本已记录；三模式运行期间出现用户 TUM 实验并发，速度数据不能用于公平的消融结论。详见 `sgi_server_performance.md`。

实现保留 Selector 结构、全局候选集合、Gumbel/Bernoulli admission、TF、Replay、GradientGuard、Change Distill 及正常 Mapping 的优化顺序。现有未跟踪文件 `build-cuda118-sm86.zip` 保留。

## 源码审查与适配点

原调用链为：

```text
GaussianModel 的 reset / append / prune 回调
    → RecentWindowImportanceManager
正常 Full Render 的 frame_importance / contribution_count
    → RecentWindow 逐视角归一化 + EMA
accumulateLodGradients
    → GITeacherBuilder（Mature、Protection、observed mask）
    → GI loss + 原有 selected-render / ratio loss
```

需要处理的适配点：

1. RecentWindow 还提供 Replay、Guard、Historical Probe 使用的近期 KF 列表。SGI 模式保留这份时间窗口，只关闭其 Gaussian contribution 缓存，避免改变这些模块的近期/历史划分。
2. RecentWindow 对每个视角归一化且使用 EMA，SGI 必须直接保存当前 kernel 的原始 contribution，并等权平均。两者不混用。
3. 原 Inference renderer 禁止输出 GI。现仅允许 Inference 的 contribution 输出，仍不分配 backward bucket/sample 状态；Visibility 模式继续禁止收集 GI。
4. 配置采用一个 `GlobalGI.mode` 选择 GI 来源，代表样本比例由 Anchor 比例推导。eager 同步、Bottom-K、ORB MapPoint、uniform coverage、新点 invalid 是固定行为，没有增加互相覆盖的开关。
5. 服务器测试发现最下层 `FORWARD::render` 的 Inference dispatcher 仍将 GI 收集参数硬编码为 false。已将三个参数传给既有 kernel，未修改 contribution 公式；Training/Inference 的原始 contribution 和计数一致性测试通过。
6. 既有 combined SH/预计算 covariance 测试暴露了输入适配问题。`GaussianRasterizer::forward` 将合并 SH 拆为 DC 和高阶系数、将 `[N,3,3]` covariance 打包为 kernel/backward 所需的 `[N,6]`，并提供已定义的空 DC Tensor。修复后原 compact LOD CUDA 梯度测试通过，compute-sanitizer 检查 renderer 为零错误。

### ORB 接口与有效历史 KF

`updateGlobalGIBank()` 从当前 ORB Map 的 `GetAllKeyFrames()` 读取候选，通过 `KeyFrame::isBad()` 排除失效 KF，并通过 `KeyFrame::GetMapPoints()` 与 `MapPoint::isBad()` 获取有效 MapPoint ID。

本实现的有效 KF 是：当前 ORB Map 中未失效、已进入 Gaussian Mapper 的历史表、已具备尺寸与投影 Tensor 的 KF。它们不受正常 Mapping 的训练滑动窗口限制。

读取关联时持有 `Map::mMutexMapUpdate`，只复制稳定 ID，释放 ORB 锁后才计算 GI。该 ORB 分支的 `MapChangeIndex` 不覆盖全部关联和 culling 路径，因此每次正常 Mapping 检查关联快照；没有变化时跳过 Bank 选帧。该 CPU 扫描、锁等待和选帧时间全部计入 Bank 耗时。

### Gaussian 拓扑的实际索引处理

| 操作 | 当前源码行为 | GI 处理 |
|---|---|---|
| Append | `densificationPostfix()` 对六个属性执行 `cat(old, new)` | 所有缓存追加零列，valid 追加 false |
| Clone | 取父行生成新 Tensor，再调用 postfix | 原父行保留；子行追加零列、不继承 GI |
| Split | 先追加子行，再用真实 prune mask 删除父行 | 两次回调立即同步；其余旧行保持对应关系 |
| Prune | `prunePoints()` 对属性和元数据使用 `~mask` | 对所有 GI 视角执行同一个 survivor mask |

没有依据长度猜测 survivor，也没有建立额外的 Gaussian 身份系统。现有回调新增可选的 `AppendKind` 事件参数，仅用于区分 append/clone/split 的性能统计；原三参数源码调用保持兼容，项目需要重新构建。

## 实际实现

### Global View Bank

`GlobalGIViewBank` 是可独立编译的 CPU 策略：

- 默认 `K=ceil(0.30*N_KF)`、`K_A=ceil(0.80*K)`、`K_R=K-K_A`，没有有效 KF 时为空。
- Coverage 使用 MapPoint ID 覆盖计数；新增、删除、关联变化时增量更新计数，不重复构造所有完整集合并集。
- 初始化和新增名额从全部候选中按最大覆盖增益填充；满额时仅接受严格增加覆盖的替换。
- 删除 Anchor 或其关联变化后重新评估历史候选。相同覆盖增益使用稳定 ID 决定顺序。
- Representative 使用 `SplitMix64(KF_ID XOR seed)` 的持久优先级，取非 Anchor 中优先级最小的 Bottom-K。Anchor 与 Representative 无重复。
- 按当前历史池维护容量，旧 KF 不因时间流逝被自动丢弃。ORB 已失效或移除的 KF 会被排除。

### 缓存、增量更新与拓扑同步

`GlobalGIManager` 保存每个 Bank 成员的 `[N] float32 contribution`、`sum`、平均 `importance` 和 `[N] bool valid`。

Bank 加入新成员时只渲染新成员；退出成员扣除旧缓存。普通参数更新、普通重复使用 KF 不刷新旧缓存。首次建立 Bank 时全部成员都有当前贡献，建立初始完整有效性。

每次真实 append/prune 回调对全部视角立即同步，并从这些缓存重新求和，更新平均 GI。Split 的追加和父点删除分别同步；全过程沿用 `mutex_render_`，Selector 不能读取中间状态。拓扑回调不调用 renderer。

新生点零值是占位值，valid 为 false。新增视角只提供部分估计，不会将已有 invalid 点改回 valid。原有有效行不会因为普通拓扑变化整体失效。

非有限或负 contribution 的对应行也保持 invalid。完整 Bank 渲染中数值有限且贡献为零的行是真实零贡献，可以 valid。

### 周期刷新与 Selector 接入

每次正常 `trainMappingIteration()` 完成并增加 `mapping_iter_` 后，检查刷新条件。默认在第 1000、2000…次真实渲染全部当前 Bank 成员；Replay、CD、DSD 的内部优化步不增加该刷新计数。

刷新在模型锁内同步执行，Gaussian 索引和 Bank 成员不会在渲染中改变。全部新缓存、sum、mean、valid 在临时状态中构造，完成后统一发布。普通拓扑变化只对旧缓存重排，不额外触发全量渲染。

GI Teacher 仅使用 `Mature ∩ GIValid`，复用当前 Builder 的 mask 接口及其严格 percentile 标签规则。Selector 的候选集合和预算计算仍使用原人口；valid 不参与 admission 或 render mask 的候选过滤。没有有效 Mature 行时只跳过 GI loss。

当前 kernel 在 `cuda_rasterizer/forward.cu` 中累积 `con_o.w * T`，并记录 contribution count。本次保持该定义，不加入体积归一化或新 kernel。

## 配置与消融

`cfg/selector/selector_config.yaml` 默认 GI 来源已切换为：

```yaml
GlobalGI.mode: "streaming"
GlobalGI.view_ratio: 0.30
GlobalGI.coverage_anchor_fraction: 0.80
GlobalGI.sampling_seed: 0
GlobalGI.refresh_interval: 1000
GlobalGI.max_views: 0
GlobalGI.performance_logging: 1
```

消融只修改 `GlobalGI.mode`，分别为 `disabled`、`recent`、`streaming`。其他配置应保持一致。`recent` 需要 `RecentWindow.enable: 1`，其 size、EMA 和归一化逻辑保留。缺少 GlobalGI 配置的旧文件仍按原 RecentWindow 设置运行。

有效配置会随 `saveSelectorConfig()` 写入 PLY 旁的实验 YAML。已有 Python LOD 评估不使用 GI 训练缓存，原 Selector/TF checkpoint 解析没有新增要求。

默认文件的 `Selector.enabled` 保持原值 0；切换 GI 来源本身不会启用 Selector。若需要 GI guidance，使用实验原有的 Selector 启用设置。

## 性能记录与报告工具

日志前缀为 `[GlobalGI]`，分别记录 `bank`、`topology`、`full_refresh`、`mapping`、`final`：

默认 `GlobalGI.verbose_logging: 0`，只保留第 1 次及每 1000 次 Mapping 的性能摘要、完整刷新和最终汇总；高频 `bank`、`bank_state`、`topology` 及重复的 `refresh_state` 详情仅在设为 1 时输出。`GlobalGI.performance_logging: 0` 可进一步关闭 GI 性能输出。关闭详情后，汇总计数继续累计，但不再为逐次打印同步读回有效点数量。

默认日志级别已在服务器验证：相关目标编译和 GI 测试通过；2100 次 Mapping 仅输出 6 条 GI 日志（3 次周期摘要、2 次完整刷新、1 次最终汇总）和 2 条 Mapping 性能摘要，逐次 Bank/拓扑详情为零；最终 invalid=0、有效 Mature 比例=1、guidance=1。

- Bank 的有效 KF、目标数量、Anchor/Representative 数量、覆盖率、更新与累计 CPU 时间。
- 新成员 contribution 渲染、聚合、CUDA event 时间、累计视角数和累计耗时。
- 每次拓扑操作的来源、remap/aggregate/total 时间、CUDA event 时间、点数和有效/无效点数。
- 拓扑次数、累计耗时、平均值、P95、最大值；P95 在周期汇报和最终汇报计算。
- 每次完整刷新的全部视角数、wall time、CUDA event 时间、累计/平均/最大耗时。
- 实际缓存 Tensor 的元素字节数、缓存与已跟踪临时 Tensor 的逻辑峰值、`K*N*4` 视角缓存估算。
- GI valid Mature 的数量/比例、guidance 状态、Mapping 平均周期/iterations/s、已计时 GI 操作占 Mapping wall time 的比例。

CUDA 时间通过当前 stream 的 Events 取得，GI 操作的 wall time 等待对应 event 完成后记录。普通不涉及 GI 更新的 Mapping 不因精确计时执行额外 synchronize。有效 Mature 统计只在汇报周期和保存时读回。

峰值字段不是 CUDA allocator reserved 峰值，也不包含 renderer scratch buffer；不能将它当作整个进程的显存峰值。全量刷新原子发布时新旧缓存会同时存在，目标机仍需记录进程实际显存。

`scripts/report_global_gi.py` 读取三个模式的现有日志，生成中文 Markdown 报告和 CSV 数量/耗时/显存时间序列，不启动训练：

```text
python scripts/report_global_gi.py --disabled-log disabled.log --recent-log recent.log --streaming-log streaming.log --output sgi-performance.md
```

三组速度使用相同的 `[Mapping PERF]` 周期数据比较。日志缺少数据时标为“未记录”，不填假值。Tracking 是否掉帧仍需结合 Tracking 日志，不能单凭 SGI 成本推断。

## 修改文件与关键接口

| 文件 | 作用 |
|---|---|
| `include/global_gi_bank.h` | 配置、CPU Coverage/Bottom-K Bank |
| `include/global_gi.h`、`src/global_gi.cpp` | `updateBank`、`appendGaussians`、`pruneGaussians`、`refreshIfDue`、性能统计 |
| `include/gaussian_mapper.h`、`src/gaussian_mapper.cpp` | ORB 快照、只读 contribution 渲染、周期刷新、GI Teacher、配置保存 |
| `include/gaussian_model.h`、`src/gaussian_model.cpp` | 可选 `AppendKind` 回调及 Clone/Split 来源 |
| `include/recent_window_importance.h`、`src/recent_window_importance.cpp` | 保留近期列表，SGI/Disabled 不分配 Recent GI Tensor |
| `src/gaussian_rasterizer.cpp`、三处 CUDA rasterizer 入口 | Inference 收集既有 GI kernel 输出；SH/covariance 输入适配 |
| `CMakeLists.txt`、`cfg/selector/selector_config.yaml` | 编译/测试注册、SGI 默认配置 |
| `tests/global_gi_bank_test.cpp` | CPU Bank 独立测试 |
| `tests/global_gi_test.cpp` | contribution、拓扑、有效性、1000/2000 刷新测试 |
| 三个既有 C++ 测试文件 | Recent temporal-only、GaussianModel 真正回调、Inference/Training GI 一致性 |
| `scripts/report_global_gi.py`、`tests/global_gi_report_test.py` | 实测日志报告与解析测试 |

## 已执行验证与未完成验证

| 检查 | 结果 |
|---|---|
| `git diff --check` | 通过 |
| CPU Bank：g++ C++17，`-O2 -Wall -Wextra -pedantic` | 编译和执行通过 |
| Bank：250 次增长、配额、Bottom-K、输入顺序复现、历史保留 | 通过 |
| Bank：Greedy、严格增益替换、失效删除、关联更新、400 次随机覆盖计数与独立集合并集对照 | 通过 |
| 性能报告 Python AST 与合成日志解析/报告生成测试 | 通过；合成数据不是 GPU 性能结果 |
| 服务器 CMake、`CARTGS_BUILD_TESTS=ON`、Release、CUDA 11.8 / sm_86 | 完整编译通过 |
| Test B–F 的 LibTorch C++ 缓存/监督测试 | `global_gi` CTest 通过 |
| GaussianModel 真正 Clone/Split 回调测试 | `gaussian_model_lifecycle` CTest 通过，真实 CUDA |
| Training/Inference 原始 GI contribution 一致性 | `renderer_readonly_cuda` 通过，覆盖空场景、多种 SH/covariance 模式 |
| 原 compact LOD CUDA 结果与梯度回归 | `compact_lod_cuda` 通过 |
| 全部 CTest | 28/28 通过，12.36 秒，未跳过 CUDA 项 |
| CUDA renderer 内存检查 | `compute-sanitizer --tool memcheck --error-exitcode 99`，0 errors |
| Test G：实际数据集、真实 CUDA 拓扑 P95/刷新/显存/速度 | 已采集 400 帧、2100 Mapping 的三模式日志；并发使速度比较不成立 |

本机最初的 `No CUDA toolset found` 诊断保留在 `build-cuda118-sm86-sgi-validation/CMakeFiles/CMakeConfigureLog.yaml`；后续按用户要求只在服务器编译。服务器测试原始日志回传到同一忽略目录的 `server-logs/`。

## 服务器验证记录

- 环境：Ubuntu 20.04，RTX 3090，驱动 535.230.02，CUDA 11.8.89，GCC 9.4.0，CMake 3.27.9，LibTorch 2.3.1+cu118。
- 构建目录：`/home/prmi/zwj/mycartgs/build-cuda118-sm86-gcc9`，复用已安装的 Eigen/OpenCV/JSONCPP/ORB 依赖。
- 源码首次同步 543 个文件并逐个 SHA256 校验；15 个被覆盖文件保存在 `.codex-sgi-backup-20261010-160211/overwritten-source.tar.gz`。没有同步本地构建包或删除远端依赖/数据。
- 运行日志与独立配置：`results/sgi-validation-20261010-160420/`。Replica room0 前 400 帧使用符号链接，原始数据未修改。
- 三模式都启用 Selector，固定 `Selector.max_mapping_iterations=2100`、`GlobalGI.refresh_interval=1000`，其他参数相同。TF.start_iter=5000，本次未进入 TF 训练；没有启动大规模训练。
- Streaming：26 个有效 KF，Bank=8，其中 Anchor=7、Representative=1，MapPoint 覆盖率 0.8687。Append/Clone/Split/Prune 分别为 18/3/3/6。
- 拓扑：平均 0.3757 ms、P95 0.6104 ms、最大 1.9599 ms、累计 11.2707 ms。两次完整刷新分别为 25.4762/22.5004 ms。
- 缓存：最终 0.7001 MiB，逻辑峰值 1.4002 MiB。每 2 秒外部采样的进程显存峰值为 1022 MiB，该采样可能遗漏瞬时峰值。
- 第 1000、2000 次 Mapping 的 invalid=0、有效 Mature 比例=1；最终 17905 个 Gaussian 中 valid=17905、invalid=0，GI guidance 开启。
- 用户后续正在运行 TUM 实验并修改了远端 Selector 实验配置。保留该配置，不以本地默认 YAML 覆盖；本次三模式独立测试配置与其分离。

```text
cmake -S . -B build-cuda118-sm86-gcc9 -DCARTGS_BUILD_TESTS=ON
cmake --build build-cuda118-sm86-gcc9 -j 4
OMP_NUM_THREADS=1 ctest --test-dir build-cuda118-sm86-gcc9 --output-on-failure --timeout 120 -j 1
python3 -B tests/global_gi_report_test.py
OMP_NUM_THREADS=1 /usr/local/cuda-11.8/bin/compute-sanitizer --tool memcheck --error-exitcode 99 bin/renderer_readonly_test
```

## 用户 TUM 部分日志分析

日志范围为 Mapping 537–646，未覆盖第 1000 次刷新。第 646 次状态：54 个 KF、17 个 Bank 视角、14 个 Anchor、3 个 Representative，配额正确，覆盖率为 0.8352。

54338 个点中 valid=1991（3.664%）、invalid=52347（96.336%）。该阶段所有 topology 都是 append，新点在下一次完整刷新前继续 invalid，这是本次设计的监督延迟。日志没有显示刷新错误，也不能仅凭该片段证明后续第 1000 次刷新通过。

Bank CPU 路径累计 4723.0193 ms，增量 contribution/aggregate 累计 233.5597 ms，拓扑累计 62.9449 ms，完整刷新累计为零。Bank 占这三类已计时 GI 成本约 94.09%；瓶颈集中在每次 Mapping 的 ORB 关联快照、锁等待和 CPU 选帧路径，现有日志尚不能拆分这几部分。拓扑平均 1.2846 ms、最大 3.9433 ms，GI 缓存 3.9902 MiB。

`global_gi_overhead_fraction=0.4645` 是已计时 GI 成本 / Mapping wall time，不是相对 Disabled 的速度降幅。`mapping_iterations_per_second=59.6835` 是 Mapping 周期吞吐，不是 Tracking FPS。片段中的 94 次增量视角渲染包含 Bank 增长和替换，历史视角总数不需要等于最终 Bank 的 17 个成员。

回传的远端实验配置中 `Selector.enabled=0`、`Selector.max_mapping_iterations=16400`、`GlobalGI.mode=streaming`。SGI manager 不以 Selector 开关为条件，因此配置为 0 时仍维护缓存，但不以 GI loss 训练 Selector。该日志片段缺少 guidance 字段，不能证明已启用监督；若目标是测试 SGI guidance，需要核对启动时的配置和命令行 override。本次独立短测配置明确启用了 Selector，日志中的 guidance=1。

## 仍需验证的边界

1. 独占服务器条件下的三模式速度消融尚未完成；不以本次并发数据给出速度降幅结论。完整长序列质量、轨迹与大地图显存扩展仍需专门实验。
2. 本次真实 CUDA 短测验证了两次完整刷新和真实 append/clone/split/prune；TF、Guard、DSD 的完整数据集联合长期运行未由本次 2100 次短测覆盖。
3. 已完成 CPU 覆盖收益优化和快照/选帧计时拆分，原算法逐步差分测试通过。243 KF 的 Anchor 关联变化合成基准平均加速 12.68 倍；真实短测剩余 Bank 成本主要来自 ORB 快照读取，关联增量缓存仍需单独实现。验证与性能边界见 `improvements/sgi_cpu_optimization.md`。
4. 30% Bank 无默认固定上限，持久缓存为 `4*K*N + 9*N` 字节，完整刷新还会暂时持有旧缓存。大地图显存风险须实测。
5. 当前实现覆盖当前 ORB Map 的全部有效历史 KF，不处理不同 Atlas Map 之间未对齐的坐标系。没有为离线 COLMAP 合成 ORB coverage；COLMAP 训练须显式选择 recent/disabled。只读 viewer 仍可加载带 streaming 配置的 checkpoint。
6. 沿用 Mapper 当前执行线程与默认 CUDA stream。多 stream 并发更新、可恢复 OOM、训练 checkpoint 恢复全部 SGI 缓存不在本次实现内。
7. 普通参数更新及几何变化使数值可能变旧，按设计由下次完整刷新纠正；它们不会让已有 Gaussian 索引缓存因拓扑变化整体失效。
