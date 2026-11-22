# Hierarchical 3-Representative SI 实现与 correctness 报告

结论：**IMPLEMENTATION PASS**。最终固定参数为 **初始总数2048 cells / 3 representatives / Q<0.80 split / 最大L3 / effective maturity=1**。CPU、GPU synthetic tests、legacy关闭路径 parity、原GlobalGI regression、mapper translation-unit编译以及真实fr1_room一次构建和独立公式复核均通过。

该PASS只表示数学与工程正确性，不代表质量GO、online GO或production GO。没有训练、没有修改Gaussian/TF、没有测渲染质量，也没有自动接入在线实验。

## 1. 基线与范围

- HEAD：`40751d07e194aaaa997d06bdea14d90f813b1222`；分支：`redundancy-aware-si`。
- 开始时tracked clean；既有 `STREAMING_GI_CURRENT_REPORT.md` 和 pycache 未跟踪，原样保留，详情见 `audit_before.json`。
- 用户后续明确：maturity统一为1；初始grid总数由256改为2048，不是最短边划分2048格。最终参数见 `protocol_2048.json`；256版本只保留在 `before_grid2048/` 作为变更前工程记录，不作为参数搜索或质量选择。
- 本地改动未commit、未push。服务器只建立隔离source/build进行测试，未覆盖 `/home/prmi/zwj/mycartgs` 的production源码或二进制。

## 2. 当前真实SGI调用链

`GaussianMapper` Mapping boundary → `updateGlobalGIBank` → `GlobalGIManager::updateBankBatch` → 原Bank selection/GIViewCache增删 → `renderGlobalGIContribution` → `GaussianRenderer::render` → `cuda_rasterizer/forward.cu` 返回 `frame_importance` → `sum_ / Bank view count` → `importance_`。

当前kernel真实统计为：通过production alpha/T筛选后，`atomicAdd(frame_importance + row, con_o.w * T)`。本轮没有把它改写成文档中的alpha*T，也没有改变任何per-view normalization或kernel。

`src/global_gi.cpp::aggregate/refreshIfDue` 保留 `sum_`、`importance_` 为[N] FP32，`valid_` 为[N] bool；`GIViewCache` 继续保留每个Bank view的原始[N] contribution。`observedMask()` 仍为原 `importance_>0`。已有full refresh cadence不改。

Teacher入口在 `GaussianMapper::accumulateLodGradients`：新开关打开时，以当前xyz、activated scale、Raw SI、valid和observed重建hierarchy，把输出score交给原 `GITeacherBuilder::build`。其strict-percentile label、requested ratio、监督population、GI loss、ratio loss和ST sampling均保持原实现。

权威geometry来自 `GaussianModel::getXYZ()` 与 `getScalingActivation()`；后者执行exp(log-scale)。append/clone按old+new拼接；split先append children再prune parent；prune使用精确survivor mask；reset替换map。source row仅作为当前snapshot确定性tie-break，不是跨topology稳定ID。

## 3. 修改文件

| 文件 | 改动 |
|---|---|
| `include/hierarchical_representative_si.h` | 固定常量、cell/tree/result接口、stateless score入口 |
| `src/hierarchical_representative_si.cpp` | grid、cell级tensor相似度、3-rep greedy、octree、SI重分配、守恒检查和诊断 |
| `tests/hierarchical_representative_si_test.cpp` | CPU/GPU synthetic、dense-reference、legacy及topology测试；冻结输入smoke入口 |
| `include/global_gi_bank.h` | 两个默认关闭bool及模式互斥校验 |
| `include/gaussian_mapper.h` | 新模块include |
| `src/gaussian_mapper.cpp` | 配置读写、effective maturity=1、score替换、条件诊断/周期摘要 |
| `cfg/selector/fast.yaml`、`selector_config.yaml` | 明确默认关闭的配置 |
| `CMakeLists.txt` | 注册新源文件与CTest target |

原 `global_gi.cpp`、`gi_teacher.cpp`、renderer/kernel、Selector网络及TF实现未修改；原fixed-h8 marginal模式保留，但新开关不调用它。

## 4. 配置及关闭语义

```yaml
GlobalGI.mode: "streaming"
GlobalGI.HierarchicalRepresentative.enable: 0
GlobalGI.HierarchicalRepresentative.diagnostics: 0
```

设置enable=1才启用。diagnostics=1才导出tree与telemetry，不默认大量写磁盘。2048、3、0.80、L3均为代码常量，无sweep配置。开启要求Raw `streaming`，与旧 `streaming_marginal_geo`/marginal shadow互斥。

关闭时直接使用原score handle，既不读取geometry也不建hierarchy；原Raw/marginal两种既有teacher选择路径、原maturity、entry、Selector行为保持不变。测试覆盖返回handle相同、importance/mask/labels/weight/loss梯度、Selector forward、cached-u ST以及ratio loss exact parity；legacy fixture明确包含正负两类label。两个默认配置均经OpenCV FileStorage实际读取，enable/diagnostics均为0。

用户追加的maturity=1仅作用于新实验路径：LoD中有效maturity与selection maturity为全1、entry=0；Teacher不再受旧maturity/protection资格或权重影响。未删除旧状态模块；Full流程仍可维护它以保持关闭路径兼容，新hierarchy本身完全不读它。

## 5. 初始2048-cell grid

从所有当前有限xyz计算scene AABB，每轴padding为 `max(1e-9, max_extent*1e-9, max_abs_coordinate*32*FP64_epsilon)`。枚举 `nx=2^ex, ny=2^ey, nz=2^ez`，其中ex+ey+ez=11；在78个组合中最小化三轴physical cell size的log variance，目标数值相同则取字典序最小dims。

总格数严格2048，空格不分配pairwise matrix；只保存occupied roots。point用floor分配，极小padding保护max-boundary，浮点边界再clamp到有效索引。当前scene：

- AABB min：`[-67.14266983305264, -79.83248916899014, -97.58124557768154]`
- AABB max：`[74.28296676908779, 61.3341790798788, 57.7370111111288]`
- dims：`[16, 8, 16]`，乘积2048。
- Level-1 physical size：`[8.839102287633777, 17.645833531108618, 9.707391043050647]`。

## 6. 几何与representative数学

输入mass始终为冻结Raw SI `I_i=mean_v(c_iv)`。没有importance-mass预去重、density normalization或重复乘cell importance。

物理尺度半径 `r_i=cbrt(sx*sy*sz)`，实现采用等价的 `exp(mean(log(activated_scale)))` 避免中间乘积overflow/underflow。`sigma_ij=(r_i+r_j)/2`；`kappa_ij=exp(-||xi-xj||²/(2*sigma_ij²+1e-12))`，clamp到[0,1]，显式diag=1。无view cosine、orientation、covariance、opacity或appearance。

Cell中先令cover=0。每步对未选择candidate计算 `Delta_i=sum_j I_j*max(kappa_ij-cover_j,0)`，取最大Delta，完全相同gain时按当前source row ID升序。记录真实选中时Delta，再更新 `cover=max(cover,kappa_rep)`。连续选min(3,occupancy)个；第一个代表可能是Raw SI较低、但能代表更多mass的点，已有专门synthetic检查。

每cell使用LibTorch FP64 tensor运算，candidate按128-row block计算对cell内全部Gaussian的relation。没有Python双重Gaussian循环的production路径，没有全场景N×N matrix，也没有custom CUDA或近似邻域。CPU/GPU跟随输入device；block大小只限制执行内存，不改变公式。

## 7. Coverage、split和parent discard

`M_C=sum I_j`；`F_C=sum I_j*cover_j`；`Q_C=F_C/(M_C+1e-12)`。全零mass cell同样按原式得到Q=0，deterministic选代表后仍按threshold/depth规则处理，不另设早停特例。生产observed=Raw>0，因此实际候选cell不会出现全零mass；synthetic显式覆盖该边界。

若Q>=0.80，则成为leaf；若Q<0.80且level<3，按该cell三轴中点分为8个child，只递归非空child。每child重新从原Raw I开始选择最多3代表。L3无论Q多低均停止。没有occupancy触发、最少child数条件、旧residual(2)规则或跨boundary竞争。

split parent的provisional representatives只供split判定，完全不写最终score；所有输出仅由final leaves产生。node traversal固定root升序、child 0..7。synthetic与真实输出独立重建验证均排除了parent score泄漏。

## 8. SI redistribution与mass证明

对于final leaf，代表的输出为选中当时的真实Delta；非代表为 `I'_j=I_j*(1-cover_j)`。代表因为self-kappa=1，其最终cover=1，故自身未覆盖残余为0。

每步Delta是coverage目标的真实增量，因此 `sum_rep Delta=F_C`。非代表残余之和等于 `sum_all I_j*(1-cover_j)=M_C-F_C`。二者相加严格为M_C。不同leaf互不重叠，所以全局同样守恒。代码对每leaf和全局均显式检查，**没有renormalization补偿**。

真实测得：

- Raw总mass：`2905230.3122825678`。
- Hierarchical总mass：`2905230.3122825678`。
- 全局absolute/relative error：`0` / `0`。
- 独立NumPy逐leaf最大absolute error：`1.1641532182693481e-10`。
- 独立重建全部I'最大逐row差：`6.8212102632969618e-13`。
- 独立重算代表Delta最大差：`7.2759576141834259e-12`；Q最大差 `3.3306690738754696e-16`。

Smoke保留原冻结Raw FP64数组，未改其值；production接受原FP32 Raw并在FP64中计算、最后转换回原dtype。synthetic另验证FP32输出守恒容差<=1e-6。

## 9. Candidate与unknown语义

参与hierarchy的集合严格为原valid&observed，不使用maturity过滤。当前数据valid为全图有效、observed为原Bank正贡献行：总N=52635，candidate N=50330。excluded行不进入tree、leaf_assignment=-1、输出score=0，但其valid/observed仍完全由原SGI决定；模块不把它们变成known/valid，Teacher继续用原mask。

candidate的xyz/activated scale/Raw必须有限、scale正、Raw非负，否则显式报错。全零、空map、低occupancy和完全相同gain均有确定性行为。

## 10. Topology与geometry correctness

第一版无跨调用hierarchy缓存，开启时每次在原render lock保护的teacher读取处重建。输入N、append/clone/split/prune、正确同步后的row reorder、xyz原位变化、scale变化、map reset均不会复用旧membership、radius或representative score。synthetic覆盖append未知row、prune+reorder、geometry和scale编辑，并与全新输入clone的结果exact比较。

原Raw GI仍必须遵守既有row-alignment callback契约。绕过所有同步、只重排Master而不重排Raw GI本身属于上游数据破坏，本模块不声称能凭无稳定ID的tensor自动修复它。

## 11. Synthetic / regression验证

CPU与RTX3090 GPU均通过下列同一组assert；不以CPU/GPU彼此bitwise相等为要求，要求各设备重复确定且符合数学语义。

| 测试 | 结果 |
|---|---|
| 3完全重复，I=[10,9,8] | Delta=[27,0,0]，Q≈1，不split |
| 3完全独立 | 三者Delta=各自Raw，Q≈1 |
| 4完全独立等权 | Q≈0.75，L1/L2必须split |
| 100 duplicates | 1个主要representative覆盖mass，occupancy不触发split |
| 2独立duplicate clusters | 前2代表覆盖mass，不需split |
| 4独立duplicate clusters | parent Q≈0.75，split到child |
| 无split/一次/两次split | leaf与全局守恒 |
| 同输入重复 | dims、assignment、完整tree、IDs、Delta、Q、I' exact |
| L3仍Q<0.80 | 记录unresolved，绝无L4 |
| Parent score discard | 仅final-leaf Delta/残余可重建输出 |
| Empty/zero mass、小leaf、非顺序source ID tie | 无NaN/Inf，确定性 |
| 257点chunk边界 | blocked greedy与独立dense公式的order/gains一致 |
| 首代表非Raw最大 | 取weighted coverage更高的中心代表 |
| FP32 output | mass容差通过 |
| append/prune/reorder/xyz/scale变化 | 无旧row/cache复用 |
| enable=false | 原Raw handle、valid/observed、teacher labels/weights/loss gradients、Selector forward/ST/ratio loss exact |

另执行未改动的原 `tests/global_gi_test.cpp`，覆盖Bank增量/全刷、invalid/newborn、append/clone/split/prune与teacher监督语义，通过。原Raw/renderer/Teacher/Selector实现SHA保持原样。边界复核去除了最初零mass早停特例，最终严格按Q=0和L3规则处理；这项修正未改变正mass fr1_room fixture的tree或输出，保留 `before_zero_mass_fix.json` 和输出SHA核对。

## 12. 编译验证边界

隔离CMake实际编译并链接新core、当前GlobalGI/Teacher及测试，CPU/GPU执行通过。使用现有production的编译参数，对修改后的完整 `gaussian_mapper.cpp` translation unit执行编译检查，通过。记录于 `mapper_compile_check.json`。

未重新链接/运行整个SLAM二进制，未替换服务器production build。本轮不把synthetic/单TU编译称为完整online系统验证。

## 13. fr1_room固定map smoke

Master：`/home/prmi/zwj/mycartgs/resultsall/batch_20261018_tum_aligned/jobs/tum/rgbd/fr1_room/full/attempt_1/model/16900_shutdown/ply/point_cloud/iteration_16900/point_cloud.ply`。复用原69个Bank views的冻结contribution/Raw数组；总N=52635，candidate N=50330。旧Master、TF、camera、Bank与GT相关数据SHA保持一致，Raw/contribution NPZ与旧manifest SHA exact匹配；没有重新渲染contribution、训练Selector/TF或优化Gaussian。

历史production source/.so在上轮源码同步与构建后发生已知变化，初始准备阶段对这些旧软件路径的校验因此失败。它们本轮不作为smoke输入加载；已记录在 `input_manifest.json::historical_unused_software_drift`。本轮实际使用的数据均重新核验，未跳过数据SHA gate。没有使用GT/quality选择grid或split。

| level | occupied/evaluated cells | split | final leaves | final reps | occupancy mean | median | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| L1 | 36 | 5 | 31 | 47 | 1398.056 | 1.000 | 39334 |
| L2 | 26 | 13 | 13 | 25 | 1933.077 | 17.500 | 24083 |
| L3 | 48 | 0 | 48 | 130 | 1046.146 | 23.000 | 16613 |

总final leaves=92；final representatives=202。这里occupancy按该level被评估的全部node统计，包含随后split的parent，不与final-leaf统计混为一谈。

Final-leaf Q：mean=0.79353572，median=0.99999505，p10=0.14861178，p90=1.00000000。

**L3 unresolved：28/48=58.33%**；这些未充分覆盖leaf携带 **97.70%** 的全局Raw SI。Raw-mass加权leaf Q仅 **11.84%**。不能被unweighted median≈1掩盖：大量小leaf已覆盖，但主要mass仍在复杂dense leaf里以residual保留。

初始最大root occupancy=39334；L3最大仍=16613。当前较大的全map AABB与有限depth造成很粗的competition domain。该限制已保留，没有追加shortest-edge grid、KD partition、neighbor correction或更深level来改善数字。

## 14. Runtime breakdown

| 阶段 | 单次构建ms |
|---|---:|
| 总build | 4158.040 |
| geometry similarity | 3300.286 |
| representative gain/reduction/coverage | 729.634 |
| 其余AABB/membership/copy/host bookkeeping | 128.121 |

计时以同步后的host wall计量GPU tensor调用，包含本次build内部同步；输入读取和diagnostic文件写入不在build时间内。这是一次correctness smoke成本，不是正式benchmark或性能验收。当前每次teacher读都重建，显然不能凭此称online-ready，本轮不做缓存/加速设计。

## 15. Tree debug及独立复核

diagnostics输出 `*_tree.csv` 与 `*_telemetry.json`。tree每node含leaf标记、level、root、octree path、AABB、occupancy、M/F/Q、rep IDs/raw/Delta、残余mass与守恒误差；非leaf记录仅作调试，不参与SI。

独立NumPy脚本从实际最终输出及tree重新计算每个node的coverage/记录Delta、重建全部final-leaf输出，并检查split条件、父子occupancy、row partition、unknown mask及mass；另对固定随机抽样的2605个candidate gain验证winner上界。完整greedy argmax parity由257点dense synthetic reference验证，不把抽样称为全scene exhaustive oracle。

原始结果在 `/home/prmi/zwj/hierarchical-representative-si-20261019`，本地同名目录。包括 `smoke/fr1_room_tree.csv`、`smoke/fr1_room_telemetry.json`、`hierarchical_fr1_room_output.npz`、`independent_smoke_verification.json`、输入/源码SHA和所有编译测试日志。

## 16. 已知限制与停止

1. Boundary不处理，跨leaf Gaussian不竞争。
2. 当前AABB很大，2048 roots仍有明显occupancy集中；L3大部分Raw mass未达到0.80 coverage。Implementation PASS不意味着representative capacity已足够。
3. Stateless rebuild优先保证geometry/topology correctness，成本高；没有宣称online/performance通过。
4. Raw Bank本身仍沿用production的缓存/refresh freshness语义，新层没有重新定义或修复Raw GI。
5. 原Teacher strict-percentile存在score tie时不保证精确positive K，本轮没有修改这项既有逻辑。
6. 没有进行可选render ranking sanity，没有PSNR/SSIM/LPIPS收益结论，也没有用GT调任何参数。
7. 新enable的LoD路径effective maturity=1；关闭路径保留旧模块以满足legacy regression，未全项目删除maturity状态。

**最终：IMPLEMENTATION PASS。** 本轮只完成规定结构的实现与correctness validation；没有质量GO、online GO或production GO。报告完成后停止，未commit、未push。
