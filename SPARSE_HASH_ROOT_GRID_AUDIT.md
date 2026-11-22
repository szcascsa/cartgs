# Target-Occupied Sparse Hash Root Grid 结构审计

结论：**SPARSE ROOT STRUCTURAL GO**。在同一 fr1_room 冻结输入上，occupied L1 roots 从36变为2048，最大 root occupancy 从39334降至1188；L3 unresolved Raw-SI 占比从97.70%降至16.05%，全局 Raw-SI 加权 leaf Q 从11.84%升至88.38%。Representative 数学、mass conservation、determinism 和关闭路径 parity 均通过。

本结论仅针对 root partition 的空间尺度和结构正确性。单次同步构建时间从4166.64 ms增至8920.76 ms，当前实现没有获得性能加速。本轮没有训练、GI重渲染、质量评价或在线SLAM运行。

## 1. 基线、完成标准与实验边界

- HEAD：`40751d07e194aaaa997d06bdea14d90f813b1222`；分支：`redundancy-aware-si`。
- 开始时工作树已经包含上一轮未提交的 Hierarchical SI 实现；本轮保留这些改动及原报告，未回滚其他工作。开始状态与原始文件SHA见 `audit_before.json`。
- 本轮只替换 Level-1 root partition，增加结构诊断和测试。固定参数仍为：target occupied roots=2048、每leaf最多3代表、Q<0.80 split、最大L3、effective maturity=1。
- 完成标准：occupied sparse grouping、确定性h搜索、synthetic A–G、原数学/关闭路径回归、同fixture旧后端exact复现、新结构及mass独立复核。
- 验证环境：RTX3090、CUDA11.8、PyTorch/LibTorch2.3.1+cu118、g++9、Linux；独立NumPy复核另在Windows完成。
- 服务器验证仅使用 `/home/prmi/zwj/sparse-hash-root-grid-20261019/source` 与独立build，没有覆盖 `/home/prmi/zwj/mycartgs` 的production源码或二进制。未commit、未stage、未push。

本报告数据及脚本统一保存在：

[tests/diagnostics/sparse_hash_root_grid_20261019](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/sparse_hash_root_grid_20261019)

## 2. 旧AABB-2048失败的原因

旧实现从所有有限Master xyz计算全局AABB，再选16×8×16=2048个理论bins。远处Gaussian拉大AABB；主要场景只占据36个root，其中一个root包含78.15%的candidate和82.94%的Raw SI。

旧Level-1 physical size为 `[8.8391022876, 17.6458335311, 9.7073910431]` 米。仅两次octree split无法消除这种粗尺度，主要SI留在L3低覆盖leaf里。原实现已经只为occupied bins创建node；本轮修正的是决定物理尺度的规则，不将旧实现误述为已经分配了整个dense 3D tensor。

新实现直接用candidate产生的occupied key数量选择统一cubic h，不用AABB跨度定义competition domain，也不删除或clip outlier。

## 3. 本轮源码修改位置

| 文件 | 本轮改动 |
|---|---|
| `include/sparse_hash_root_grid.h` | 新增int64 triplet、固定target、搜索trace及SparseRootGrid接口 |
| `src/sparse_hash_root_grid.cpp` | candidate-only h搜索、floor key、sort/unique计数、ordered sparse grouping |
| `include/hierarchical_representative_si.h` | backend enum、root key、搜索与阶段计时字段；默认实验backend改为SparseHash |
| `src/hierarchical_representative_si.cpp` | 调用新root partition；保留旧backend；root cube bounds、child partition assert、diagnostic扩展 |
| `tests/sparse_hash_root_grid_test.cpp` | synthetic A–G、mask隔离、Raw-SI不影响h、empty、真实fixture对照与重复确定性 |
| `tests/hierarchical_representative_si_test.cpp` | 原固定cell synthetic/旧smoke显式选Legacy，避免改变已有测试输入的空间预设 |
| `CMakeLists.txt` | 注册新源文件和测试target |
| `cfg/selector/fast.yaml` | 仅更新实验说明注释；开关数值不变 |

本轮没有修改 `src/gaussian_mapper.cpp`、`src/global_gi.cpp`、`src/gi_teacher.cpp`、`include/global_gi_bank.h`、renderer contribution kernel、Selector或handoff实现；它们与本轮开始的SHA一致。当前Git中这些文件的既有dirty改动属于上一轮，不能归因于本轮。详细证明见 `local_unchanged_modules.json` 和 `mathematical_parity.json`。

`build(..., RootBackend::LegacyAabb2048)` 保留 `legacy_aabb_2048` reference；默认参数为 `SparseHashTarget2048`，对应 `sparse_hash_target_2048`。旧backend只供显式测试/诊断，不增加第二套production配置。

功能总开关和诊断开关在两个现有配置中仍为0，并经OpenCV FileStorage实际解析：

```yaml
GlobalGI.HierarchicalRepresentative.enable: 0
GlobalGI.HierarchicalRepresentative.diagnostics: 0
```

## 4. Sparse Hash定义与存储

固定world origin为 `(0,0,0)`。每个candidate的root key为：

```text
g_i = floor(x_i / h_root)                    # 三轴分别floor，int64 triplet
root_min = g_i * h_root
root_max = root_min + (h_root,h_root,h_root)
```

`std::map<array<int64_t,3>, vector<row>>` 仅保存occupied roots，按 `(gx,gy,gz)` 字典序遍历，cell内按当前source row升序。搜索计数使用长度N的triplet vector，sort/unique后计数。内存规模随candidate数与occupied roots变化，不根据min/max key span分配任何dense tensor、array或linearized 3D table。

负坐标使用floor，例如x=-0.1、h=1得到gx=-1；没有integer truncation。int64转换溢出、非有限候选或无法在FP64分辨的root bounds均显式报错。source row只是当前snapshot的tie-break，不是跨topology稳定身份。

新backend保留candidate AABB作为diagnostic摘要；它不参与正常h选择、origin设置或root bounds生成。Sparse的`dimensions=[0,0,0]`表示没有dense grid shape。

## 5. Deterministic h搜索

搜索接口只接收candidate xyz、effective radius和source rows，不接收Raw SI、Q、split结果、Selector、图像、GT或质量指标。

固定搜索规则在构建hierarchy之前执行：

1. activated scale计算 `r_i=exp(mean(log(s_i)))=cbrt(sx*sy*sz)`；`d50=2*median(r_i)`，偶数population取中间两半径的平均；seed为 `h0=8*d50`。
2. 计算 `N_occ(h)=|unique floor(x/h)|`。高于2048时逐次×2，低于2048时逐次÷2，最多256次；达到crossing或安全数值边界停止。
3. 在所得bracket中固定生成128个log-space候选，包含端点。
4. 以 `abs(N_occ-2048)` 最小者为winner；误差相同时取较大的h。
5. 只做一次固定128点log-space refinement，区间为winner两侧已测试邻点。最终在全部已测h中按同一目标与tie-break选择。

seed恰好命中target时检查其一半至两倍范围；若target超过distinct xyz数，搜索在可达occupied上限停止dyadic缩小，仍按固定候选规则选择，不虚构2048个root。数值安全边界只用于保护FP64/int64，不以米单位手调h。

floor/hash count对任意连续h不保证全局单调。本实现保证**固定已测候选中的最优值**，不声称穷尽全部实数h。此fixture已命中误差0，因此计数目标无需进一步优化。没有根据Q、SI、unresolved或质量结果更改搜索规则。

| 搜索量 | 实测 |
|---|---:|
| target occupied roots | 2048 |
| achieved occupied roots | 2048 |
| absolute / relative count error | 0 / 0 |
| d50 | 0.0134610466172 m |
| seed=8*d50 | 0.107688372938 m |
| bracket | [0.215376745875, 0.430753491751] m |
| 去除重复h后的计数evaluation | 255 |
| h_root | 0.299867557464 m |
| h_root/d50 | 22.2766896209 |
| L1 physical size | [0.299867557464, 0.299867557464, 0.299867557464] m |
| L2 / L3边长 | 0.149933778732 / 0.0749668893659 m |
| unique key min | [-224, -66, -184] |
| unique key max | [92, 204, 192] |

独立NumPy重新计算全部255个h的occupied count，验证winner目标和较大h的tie-break，全部一致。完整trace保存在 `fixture/sparse_hash_target_2048_h_search.csv`。

## 6. Candidate、数学和octree语义

参与hierarchy及h搜索的集合严格为 `valid AND observed`。当前总N=52635，candidate N=50330，excluded=2305；effective maturity仍为1。invalid、unknown、unobserved不进入h搜索或tree，不因Grid被升级为valid；其score=0、leaf assignment=-1，Teacher继续使用原mask。

本轮逐字保持以下原实现块，SHA与变更前一致：geometry、完整 `selectCell` greedy、final-leaf redistribution。继续使用：

```text
sigma_ij = (r_i+r_j)/2
kappa_ij = exp(-||x_i-x_j||² / (2*sigma_ij²+1e-12)); kappa_ii=1
cover_j(S) = max_s kappa_sj
Delta_i(S) = sum_j I_j * max(kappa_ij-cover_j,0)
Q_C = sum_j I_j*cover_j / (sum_j I_j+1e-12)
```

最多选择3个代表，gain相同按当前source row ID；Q<0.80且level<3才split。child继续使用 `point >= midpoint` 的三轴bit，child0..7遍历，每个Gaussian恰好进入一个child；新增occupancy求和assert，不重新全局floor分配child。L3禁止继续split。

Split parent的provisional代表与Delta仍全部丢弃；每个final leaf独立写入代表真实Delta，以及非代表 `I_j*(1-cover_j)`。没有mass pre-dedup、density correction、renormalization或跨root竞争。

## 7. Synthetic与回归结果

| 验证 | 结果与证据 |
|---|---|
| A 空空间 | origin附近与1e6远处的4点，最多4个occupied roots；没有跨度dense allocation |
| B 单个远outlier | 固定6000点主体加入1e6 outlier；h从约0.162125变为0.162153 m，比例1.00017，约+0.017%，无数量级膨胀 |
| C Negative key | (-0.1,0,-1)/h1得到(-1,0,-1) |
| D Determinism | 同输入h、keys、membership、traversal、完整search trace一致；真实fixture的assignment、代表、Delta、Q、score重复exact一致 |
| E Huge key range | 1e6/h1e-6产生约1e12 key，超过int32；容器仅存occupied entries；int64不可表示输入显式报错 |
| F Octree partition | 4096点synthetic确实触发split；每个parent occupancy=直接children总和；final leaves完整覆盖candidate，无L4 |
| G 同leaf数学parity | 两backend进入相同5点duplicate leaf，代表IDs、Delta、Q、最终SI exact一致 |
| mask隔离 | 修改invalid/unobserved行至极远坐标及大scale，不改变h、assignment或score |
| Raw与h隔离 | 同xyz/scale下将Raw SI统一×7，h和全部search trace exact一致 |
| empty | 0个candidate生成空tree，无dense dimensions |
| 原数学测试 | 原duplicate/independent/cluster、threshold、depth、parent discard、守恒、dense-reference、topology/geometry/scale测试，CPU和GPU均通过 |
| enable=false | Raw handle、valid/observed、Teacher labels/weights/loss梯度、Selector forward、ST及ratio loss exact parity，CPU/GPU均通过 |
| 原GlobalGI regression | 原Bank维护、topology callbacks和Teacher回归测试通过 |

同leaf测试由shared `selectCell` 执行，三个核心数学块另有源码字节一致证明；原257点blocked-vs-dense测试继续验证非平凡greedy公式。真实fixture的全部记录Delta/Q、tree membership、split判定、parent discard和输出由独立NumPy重建；未把“核对记录代表的gain”误称为全scene全部候选的独立argmax穷举。

## 8. 同一fr1_room冻结输入

复用上一轮final Master、activated scales、valid/observed与69-view Raw SI，未重算contribution或刷新Bank。输入二进制SHA256为：

```text
e54e055fd9be3a211ba87cf6997fd088960659ff3f415d2affeebc7636d09a55
```

Master来源：

```text
/home/prmi/zwj/mycartgs/resultsall/batch_20261018_tum_aligned/jobs/tum/rgbd/fr1_room/full/attempt_1/model/16900_shutdown/ply/point_cloud/iteration_16900/point_cloud.ply
```

输入/原始数据清单SHA重新核验。清单含历史图像/GT文件的完整性校验，但h搜索和hierarchy从未读取其内容或质量信息。旧AABB reference先执行，新Sparse后执行；新后端再重复一次做exact确定性检查。

旧reference与上一轮输出的score二进制、leaf assignment和原tree全部字段逐值一致；新增key/计时字段不改变旧结果。原报告约4158 ms的结构与本轮约4167 ms重现一致，时间本身不要求bitwise相同。

## 9. AABB与Sparse直接对照

| 指标 | AABB-2048 | SparseHash-2048 |
|---|---:|---:|
| occupied L1 roots | 36 | 2048 |
| root occupancy max | 39334 | 1188 |
| root occupancy p99 | 27961.10 | 307.78 |
| largest root N fraction | 78.1522% | 2.3604% |
| 最大occupancy root的SI fraction | 82.9367% | 0.6539% |
| 所有root中最大的SI fraction | 82.9367% | 1.5397% |
| L1 split count | 5 | 469 |
| L2 split count | 13 | 549 |
| L3 leaves | 48 | 2355 |
| L3 unresolved | 28/48，58.3333% | 481/2355，20.4246% |
| unresolved L3 Raw SI | 2838548.668903 | 466175.284613 |
| unresolved SI/global Raw SI | 97.7048% | 16.0461% |
| 全局Raw-SI加权leaf Q | 11.8437% | 88.3762% |
| final leaves | 92 | 5601 |
| final representatives | 202 | 13715 |
| total hierarchy build | 4166.64 ms | 8920.76 ms |

最大occupancy root减少约33.11倍。unresolved绝对数量增加源于tree覆盖的空间区域数增多；占L3比例及承载的全局SI比例均明显下降。结构判断依据是这些归一化指标，不用绝对unresolved数量判断是否改善。

## 10. Root occupancy与Raw-SI mass分布

统计单位是occupied L1 root；quantile使用线性插值，因此整数occupancy的p99可以不是整数。

| 统计量 | AABB occupancy | Sparse occupancy | AABB Raw-SI mass | Sparse Raw-SI mass |
|---|---:|---:|---:|---:|
| min | 1 | 1 | 0.00057938 | 0.00015561 |
| mean | 1398.0556 | 24.5752 | 80700.842008 | 1418.569488 |
| median | 1 | 5 | 66.963111 | 236.845283 |
| p90 | 860 | 57 | 33138.994534 | 3389.795843 |
| p95 | 3491.25 | 105 | 172109.474423 | 6697.465877 |
| p99 | 27961.10 | 307.78 | 1667602.305111 | 20359.968666 |
| max | 39334 | 1188 | 2409501.544800 | 44730.656368 |

Sparse最大occupancy root key=(2,0,3)：1188 Gaussian，占2.3604%；Raw SI=18997.229211，占0.6539%。

Sparse最大SI root key=(-2,0,3)：467 Gaussian；Raw SI=44730.656368，占全局1.5397%。这两个root不是同一个，报告分别列出，避免混淆“最大N”和“最大SI”。完整逐root表见两个 `*_root_statistics.csv`。

## 11. L1/L2/L3详细统计

node统计包括该level随后被split的parent；reps仅统计该level最终leaf中的有效代表。每level weighted Q以该level已评估node的Raw mass作分母，不与全局final-leaf weighted Q混用。

| backend | level | evaluated nodes | splits | final leaves | final reps | occupancy mean / median / p90 / max |
|---|---|---:|---:|---:|---:|---|
| AABB | L1 | 36 | 5 | 31 | 47 | 1398.056 / 1 / 860 / 39334 |
| AABB | L2 | 26 | 13 | 13 | 25 | 1933.077 / 17.5 / 4838.5 / 24083 |
| AABB | L3 | 48 | 0 | 48 | 130 | 1046.146 / 23 / 2922.4 / 16613 |
| Sparse | L1 | 2048 | 469 | 1579 | 3390 | 24.575 / 5 / 57 / 1188 |
| Sparse | L2 | 2216 | 549 | 1667 | 4255 | 18.223 / 8 / 42 / 390 |
| Sparse | L3 | 2355 | 0 | 2355 | 6070 | 11.087 / 5 / 28 / 161 |

| backend | level | Q mean | Q median | Q p10 | Q p90 | Raw-SI weighted Q |
|---|---|---:|---:|---:|---:|---:|
| AABB | L1 | 0.887523 | 1.000000 | 0.468318 | 1.000000 | 0.037014 |
| AABB | L2 | 0.618492 | 0.764468 | 0.081206 | 1.000000 | 0.060269 |
| AABB | L3 | 0.611491 | 0.670326 | 0.100972 | 1.000000 | 0.114310 |
| Sparse | L1 | 0.884240 | 0.997759 | 0.588568 | 1.000000 | 0.649656 |
| Sparse | L2 | 0.875684 | 0.956432 | 0.620505 | 1.000000 | 0.726228 |
| Sparse | L3 | 0.900995 | 0.984923 | 0.681582 | 1.000000 | 0.809205 |

Sparse全体final-leaf Q：mean=0.938136，median=0.997605，p10=0.811677，p90≈1。Q表中的1.000000为四舍五入，不表示忽略原epsilon。

## 12. Mass conservation与独立复核

原证明仍成立：代表的Delta之和是最终covered mass F；代表因self-kappa=1而没有自身residual；非代表residual之和为M-F，所以每leaf及全局总mass守恒，无renormalization。

| 量 | AABB | Sparse |
|---|---:|---:|
| Raw SI sum | 2905230.3122825678 | 2905230.3122825678 |
| hierarchical SI sum，LibTorch reduction | 2905230.3122825678 | 2905230.3122825674 |
| absolute error | 0 | 4.6566128731e-10 |
| relative error | 0 | 1.6028377693e-16 |
| 独立逐leaf最大absolute error | 1.1641532183e-10 | 5.4569682106e-12 |
| 独立重算记录Delta最大差 | 7.2759576142e-12 | 7.2759576142e-12 |
| 独立重算Q最大差 | 3.3306690739e-16 | 5.5511151231e-16 |
| 独立重建最终score最大逐row差 | 6.8212102633e-13 | 4.5474735089e-13 |

误差远小于1e-6相对容差。NumPy的全局求和可恰为零误差，与LibTorch约4.66e-10的末位差来自reduction顺序，不进行归一化“修正”。两个backend均无NaN/Inf、负score、丢失/重复leaf assignment或unknown行升级。Windows独立复核得到相同结构结论，保留各平台原始末位数值。

## 13. Runtime breakdown与成本解释

以下为最终源码在同一进程、同一GPU上的一次同步correctness构建。build包含h搜索、CPU grouping、tensor计算与同步；不含输入文件读取、diagnostic写盘、独立NumPy分析。不是多次隔离benchmark，没有给出在线amortized开销。

| 阶段 | AABB ms | Sparse ms |
|---|---:|---:|
| h search | 0 | 1201.579 |
| root key construction/grouping | 1.250 | 4.525 |
| geometry similarity | 3310.578 | 4154.458 |
| representative greedy/gain/coverage | 732.096 | 2266.500 |
| recursive child membership partition | 1.020 | 3.736 |
| 其余AABB摘要、tensor copy/index/output及host bookkeeping | 121.690 | 1289.967 |
| total hierarchy build | 4166.635 | 8920.765 |

recursive split计时只计child分类自身，不重复计入子树similarity和greedy。总耗时约2.141倍，增加114.10%；本轮没有获得性能提升。

按现有精确执行流程计数，所有被评估node的kappa计算元素数为 `sum_node min(3,n)*(n*n+n)`，从8,179,319,966降至47,034,222，约减少173.90倍。这是公式计算量计数，不是实测吞吐量或缓存pair数。与此同时，evaluated nodes从110增至6619，现有shared greedy对每cell多次启动小tensor操作并显式同步；阶段计时包含这些launch、同步和host成本，不能当成纯GPU kernel时间。该实现特征与计时结果支持“小cell执行成本主导”的解释，但本轮没有做kernel timeline或独立消融来量化每种开销占比。

未引入batching、similarity cache、近似邻域或custom CUDA。没有实测peak RAM/VRAM，不能把稀疏容器的空间复杂度保证表述为实测峰值内存收益。

## 14. 编译、关闭路径与复现证据

隔离CMake编译并链接新core及测试，CPU/GPU执行通过；使用原production编译参数，对完整mapper translation unit作 `-fsyntax-only` 检查通过。没有重新链接或运行完整SLAM二进制。

证据目录主要文件：

| 文件 | 用途 |
|---|---|
| `sparse_root_results.json` | 两backend完整结构统计与结构判定 |
| `fixture/*_tree.csv` | 每node/root key/path/AABB/occupancy/M/F/Q/代表/Delta/残余/守恒 |
| `fixture/*_h_search.csv` | 固定h搜索完整trace |
| `*_root_statistics.csv` | 逐root occupancy及Raw-SI mass |
| `fr1_room_input.bin`、`fr1_room_inputs.npz` | 相同冻结输入，含activated scale、Raw及mask |
| `old_reference/*` | 上轮冻结输出，用于exact复现检查 |
| `analyze_structure.py`、`independent_h_search.json` | 独立membership、公式、search winner及输出复核 |
| `final_sparse_synthetic.log`、`final_hierarchy_cpu.log`、`final_hierarchy_cuda.log` | 最终synthetic与关闭路径parity |
| `global_gi_regression.log`、`status.json`、`final_status.json` | 原回归和分阶段成功状态 |
| `mapper_compile_check.json` | 实际mapper TU检查命令及边界 |
| `final_verification.json`、`source_expected.json` | 源码SHA、环境与配置默认关闭验证 |
| `mathematical_parity.json`、`local_unchanged_modules.json` | 三个核心块未改、Raw/Teacher/Selector文件未改 |
| `sparse_root_partition.diff` | 相对于上一轮未提交core的本轮partition差异 |
| `artifact_manifest.json`、`artifact_download_verified.json` | 72个证据文件SHA及下载完整性验证 |
| `local_independent_results.json`、`local_independent_verified.json` | Windows独立NumPy复核 |

独立验证命令：`python analyze_structure.py`，工作目录为证据目录，需要NumPy。C++复现入口及独立CMake见 `run_validation.py`、`run_finalize.py`、`CMakeLists.txt`；其中服务器依赖路径记录在脚本中，迁移机器需替换环境路径，不需要renderer或GT。

分析脚本首次保存JSON时修正了NumPy bool序列化问题，失败日志留在 `analysis_before_json_scalar_fix.log`；修正不涉及h搜索、tree或SI。最终所有gate和独立复核均通过。

## 15. 已知限制及判定依据

- 不处理cross-root/leaf boundary redundancy；没有neighbor、halo、ghost或重叠cell。
- 2048仍是occupied-count目标，任意输入不保证exact达到；本fixture恰好达到2048。
- h规则只保证固定候选集合内最优；各平台libm/硬件的bitwise跨平台搜索一致性未作要求。同一输入、同一环境重复exact通过。
- 保留上一轮stateless每次score调用重建，不复用旧geometry/topology缓存；SGI refresh与Teacher接口不改。本轮没有定义增量grid或在线刷新优化。
- final representatives从202增加到13715，说明更多局部区域获得代表容量；这不等价于低budget排序或渲染质量收益，未运行质量实验。
- 仍有481个L3 unresolved leaf，承载16.05%的Raw SI；未为改善此数字改threshold、depth、target或h规则。
- 单次构建约8.92秒，当前执行方式存在明显成本。本轮按任务只作结构判定，未扩展性能优化或在线验证。

结构依据逐项成立：occupied roots达到target；最大root的N和SI占比不再吞掉主体；unresolved L3的Raw-SI占比显著下降；Raw-SI加权leaf Q显著上升；representative数学、mass conservation、determinism及关闭路径parity继续通过。

**SPARSE ROOT STRUCTURAL GO**
