# Sparse Learned Feature Grid — Current-Code Audit

审计对象：`H:\AIProjects\cartgs优化更细致版\mycartgs`
审计日期：2026-10-18
审计范围：只读源码、git 状态、配置、现有测试与调用链；**未实现 Grid、未修改算法、未修改源码、未 commit、未 push**。

> 重要说明：本审计开始时工作树仍包含 2026-10-18 的 GI freshness / Oldest / maturity-ablation 研究改动；审计过程中回退被实际应用（核心文件 mtime 为 18:31）。下面的所有行号和结论都以**回退后的最终工作树**为准，并已重新核对。当前工作树不是 HEAD 的干净状态，必须按“工作树源码 SHA”而不是按 commit 识别基线。

---

## 0. 结论先行

1. **当前源码没有 Selector 侧的 Sparse Learned Feature Grid，也没有任何 `position -> learnable latent` 机制。** 当前 Selector 是一个逐 Gaussian 的 pointwise MLP：输入只有 raw world `xyz`、归一化 rotation、activated scale、标量 ratio；没有 opacity、SH、age、seen count、visibility、camera/frame、邻域、global feature、positional encoding、AABB 归一化或外部 latent。`[VERIFIED]`
2. **当前源码仍然有 Global GI / GI Teacher / GI loss，也有 continuous maturity 和 ChangeDistill。** 回退删掉的是 10-18 新增的 GI selective refresh / Oldest / freshness、maturity ablation mode、utility probe 等研究路径，不是更早的 streaming GI Teacher 本体。不要按“完全 pre-GI”理解当前代码。`[VERIFIED]`
3. **当前 Selector 的 photometric gradient 确实能到达 Selector 参数，也能到达任何输入侧可微 latent**，但路径是 `p -> ST mask -> selected opacity -> render -> photo`，而且被 maturity 乘权：`p=(1-m)r+m s`，所以 `dp/dθ=m ds/dθ`。m=0 的新点没有 photometric/ratio/GI 的 Selector 梯度。`[VERIFIED]`
4. **加 Grid 不需要改 renderer，不需要改 Gaussian storage，不需要按 Gaussian ID 存。** 最自然的是 world-space 空间查询，在 `accumulateLodGradients` 构造 Selector 输入的位置接入。`[VERIFIED/INFERRED]`
5. **当前 CUDA Selector 的 kernel 固定为 4 输入 / 18 参数 / final K=128。** 直接 concat Grid 特征会破坏 `selector_cuda` 的接口与参数顺序。最小路线要么两条 arm 都用 legacy 后端做忠实 concat，要么保留 CUDA base network、在 logits 上加一个 zero-init 的 Grid 校正；前者科学上更干净，后者工程上更小。`[VERIFIED]`
6. **当前没有可用于 online 训练的 replay/fixed-KF-trace。** `std::random_device` 决定 KF shuffle，topology 会改变 row 顺序和 `cached_uniform` 流。要做干净 A/B，应优先做 fixed-map / fixed-camera / fixed-topology 的 Selector capacity harness，而不是直接在线跑 Baseline vs Grid。`[VERIFIED]`
7. **没有 stable Gaussian ID。** prune 用 boolean survivor mask 重排所有 row；clone/split 在末尾 append，parent indices 只在操作内短暂存在。因此 Grid 应按 world `xyz` 查询，而不是按 Gaussian ID。`[VERIFIED]`
8. **当前 TF 是一个独立的、比 Selector 大得多的 HexPlane 空间场，且在同一个 LoD loss 中继续学习。** Grid 的梯度不会经过 TF 参数，但 TF 会吸收/混淆 selection 的收益；第一轮 capacity audit 应冻结 TF（或至少固定其 LR/参数）并禁用 ChangeDistill/GI。`[VERIFIED/INFERRED]`

---

## 1. Git / baseline state

### 1.1 Git

| 项 | 值 |
|---|---|
| branch | `main` |
| HEAD | `0268edd9780e5dba006efa719f627cc59d0e5a65` |
| HEAD subject | `fix(build): correct mode2 test tensor and enforce LF scripts` |
| HEAD date | 2026-10-16 15:42:32 +0800 |
| staged | 0 |
| stash | 空 |
| modified tracked | 38 |
| untracked | 23（不含本报告；本报告本身为第 24 个 `??`） |
| `git diff --stat` | 38 files, +1116 / -220 |

`git status --porcelain=v1` 的代表性条目：

```text
 M CMakeLists.txt
 M cfg/selector/fast.yaml
 M cfg/selector/selector_config.yaml
 M examples/tum_rgbd.cpp
 M improvements/selector/gumbel_network.h
 M include/change_management.h
 M include/gaussian_mapper.h
 M include/selector_cuda.h
 M include/ultrafast_rotation.h
 M python/gaussian_model.py
 M python/run.py
 M src/change_distill.cpp
 M src/gaussian_mapper.cpp
 M src/selector_cuda.cpp
 M src/selector_cuda.cu
 M tests/ultrafast_mapping_test.cpp
 ...
?? SPARSE_LEARNED_GRID_CODE_AUDIT.md   (本报告)
?? tests/diagnostics/...
```

**这不是一个可直接复现的 commit 基线。** 回退是“工作树覆盖”，git pointer 仍是 `0268edd`；`git diff` 里还保留着 ChangeDistill、mode1 tail、CUDA Selector 等更早的未提交工作。任何后续实验必须固定工作树文件 SHA（或先做一次 baseline commit），不能只写 `HEAD=0268edd`。`[VERIFIED]`

### 1.2 回退事实

回退 staging 目录：`H:\AIProjects\cartgs优化更细致版\research-retirement-20261018\`

- `rollback_plan.json`：`base_head=0268edd...`，`archive=6c6ada8...`；22 个核心文件恢复到“2026-10-17 研究前快照”或 `HEAD` 原 maturity 机制。
- `apply_local.py`：按 `before_raw_sha256` / `after_sha256` 覆盖 22 个文件，并断言 `HEAD` 不变。
- `rollback_plan.json` 的 `remove` 列表包括：`include/maturity_ablation.h`、`include/maturity_gi_audit.h`、`include/online_mapping_budget.h`、`tests/maturity_ablation_test.cpp`、`tests/gi_staleness/`、`tests/oldest_online/`、`tests/teacher_upperbound/`、`tests/dynamic_teacher_upperbound/`。这些路径当前均不存在。`[VERIFIED]`
- 当前 `git diff` 已不再包含 `include/global_gi.h`、`src/global_gi.cpp`、`include/gaussian_protection.h`、`include/maturity_dynamics.h`、`src/maturity_dynamics.cpp` 等文件；这些文件的内容与 HEAD 一致。`[VERIFIED]`
- 由于 CRLF 归一化，部分文件的 raw SHA-256 与 rollback plan 的 `after_sha256` 不完全相同；但 `git diff` 内容层面已经是回退版本。不要用 raw hash 误判“回退没生效”。`[VERIFIED]`

### 1.3 关键源码 SHA-256（当前工作树）

```text
b338eee724b5ebd656a1ae69a5cc13c90d84c8a81c30b83ac94b1e8603186daa  python/selector.py
666f20bba9d4e563668249ea7ccca5ddbb5bb266b193071e6c24bfbacffda3e6  improvements/selector/gumbel_network.h
0249481d2f70fa31063ffc109498dee52b7cd01a1ef7f1a835151f8981a6711f  include/selector_cuda.h
f0e72dcdb300265f5d44da40c7ea27a4b37d81194bcb8a7cbe566c96b3836a1c  src/selector_cuda.cpp
fc638b3ddb4a2fb412c7b5a667f0edcb1ea709a1a5deb1db20dea8276912beab  src/selector_cuda.cu
26ce0703bbca5391f1b1d01a2db28e974bd66abcb3078c800bc6d30d8946bc0e  src/gaussian_mapper.cpp
810d9c45584ed3bab8d1a5084ddd462e3197f67c8ce3a912bf666171630c27b0  include/gaussian_mapper.h
3fa96ccfadd9b1b32aed0b18803f57e0f268c758683eb4d163cd475acdfdfc52  include/gaussian_model.h
b56ade7eb1de526652311c6fee4732640897bc92021362d5ae9271f88199acef  src/gaussian_model.cpp
c35d8da0d753f71ce33679476f31bbd50eb93fa6b84a869e8396d4ec881118a7  python/gaussian_model.py
89445d875b355bf21387a7d0419d7673402ddd6cab50f94c3faf9570646b3128  python/transform_field.py
7c8e58501299ebda4027699b2eec362c30b114ea944b109d3caf39bf17508f47  src/transform_field.cpp
0f35390f1c13315e3faec1f5e6d6a023ca2d73b2e98a0ea991c6dccb720fd1a8  include/transform_field.h
28ac7922cd36d25bbfb64748cff8613f5ac1d65ac0880821e249b8ca4e795534  src/hexplane_field.cpp
a283c6c124373b94f6f60485db1ec9f9191602193bcd36b6d67f7dfbb1c37797  cfg/selector/fast.yaml
fa83886300808022d88f8305c42cd623555a881b6617b8b16286514cfafa3dc8  cfg/selector/selector_config.yaml
de0b0acc9a9af8da72e5d17dde72b9842cc1bb65fa0e8cef825c6c30f1e0be5e  include/change_management.h
7f7b6348a0752c1bcc6e60996ce474c2b94c2ccdce3aa6b752b54673f6868b46  src/change_management.cpp
1b07fdbad5ede15864218c77044211cb8d40c45afea67d02b196bbd1a6421768  src/change_distill.cpp
a9fd5c75a04a8bacc1177ddae01174fb904c2a8ed50a7d9651c7fc4c9aadce9d  include/global_gi.h
93a22b400142381d3390d820aa989b3cbe9013898218f0c5a4e6a27ee6a5bd8d  src/global_gi.cpp
7207642039af90bba1f9260eb19e93bc68881f4449b9898d48b6445da4284768  include/gaussian_protection.h
0d9db2e66cb4f1805f9a40d87cfb26be905c33965cb2d7fd94b7fc6ce0cce7cb  include/protection_config.h
ad43e00e3cef4d29fbec83cf9ca3a6c64a358750e2c02fd316d36c6df0cd2102  include/ultrafast_rotation.h
```

### 1.4 Feature presence: A（生产路径使用）/ B（残留但不调用）/ C（不存在）

| 机制 | 结论 | 证据 |
|---|---|---|
| Global GI (`GlobalGIManager`) | **A** | `src/gaussian_mapper.cpp:132-133` 在 `streaming` 且 SLAM 存在时构造；`trainForOneIteration` 调 `refreshIfDue`（1327-1335）；`trainMappingIteration` 调 `updateGlobalGIBank`（1421-1427）。 |
| GI Teacher (`GITeacherBuilder`) | **A** | `src/gaussian_mapper.cpp:3070-3096` 在每次 LoD 构造 teacher 并加 `gi_loss`；`src/gi_teacher.cpp:5-34`。 |
| GI loss | **A** | `Selector.gi_loss_weight: 1.0`（`cfg/selector/*.yaml:34-35`）；`selector_loss += selector_gi_loss_weight_ * ratio_weight * gi_loss`（`gaussian_mapper.cpp:3204-3210`）。 |
| Oldest refresh / GI freshness / selective refresh | **C** | 在 `include/`, `src/`, `improvements/` 中精确搜索 `oldest`, `Oldest`, `staleness`, `freshness`, `selective_refresh` 均为 0；`GlobalGIConfig` 只有 streaming/recent/disabled、view_ratio、refresh_interval、max_views（`include/global_gi_bank.h:13-34`）；`refreshIfDue` 只做 full refresh（`src/global_gi.cpp:216-258`）。 |
| utility proxy | **C** | 核心源码只有 `#include <utility>`；没有 utility 类型/字段/损失。 |
| Selector 侧 grid / embedding table / spatial memory | **C** | 核心源码没有 `SparseGrid` / `FeatureGrid` / `cell_id` / `latent` / embedding table；Selector 只有四个 per-branch embedding（`gumbel_network.h:31-37`）。 |
| maturity（continuous cycle-geometric） | **A** | `include/gaussian_protection.h:15-19, 39-57, 110-153`；`trainMappingIteration` 在 Full commit 后 `observeFull`（`gaussian_mapper.cpp:1565-1569`）。 |
| maturity ablation mode / `effectiveMaturity` / gate06 | **C** | 搜索 `maturity_ablation_mode`, `effectiveMaturity`, `MaturityConsumer`, `gate06` 均为 0；`ProtectionConfig` 只有 drift/beta/rise/fall/diagnostics/compatibility（`include/protection_config.h:5-24`）。 |
| ChangeDistill | **A** | `ChangeDistill.enable: 1`（`cfg/selector/*.yaml:105`）；`ChangeDistillManager::execute` 在 `finishGeometryChange` 中调用（`src/change_management.cpp:334-337`）。 |
| newborn protection / forced keep / seen count | **C** | 搜索 `newborn`, `forced_keep`, `age_threshold`, `seen_count` 均为 0；`mixSelectorProbability` 在 m=0 时给出 p=r，不是强制 keep（`include/selector_handoff.h:6-13`）。 |
| TF HexPlane grid | **A（但属于 TF）** | `include/hexplane_field.h` / `src/hexplane_field.cpp`；不是 Selector 输入侧 grid。 |

### 1.5 分支/其它 checkout 里的“稀疏网格”

用户在任务中提到“稀疏自细化网格可能在别的分支”。我检查了所有本地 branch / worktree：

| 分支/位置 | 内容 | 与 Selector Grid 的关系 |
|---|---|---|
| `spatial-selector` (`db97461`) | 只有 `ADAPTIVE_SPATIAL_BUDGET_STUDY.md`、`SPATIAL_BUDGET_SELECTOR_AUDIT.md` 和 `tests/diagnostics/spatial_budget_20261016/`、`same_k_swap_20261016/`；`git diff main..spatial-selector` 为 48 个文件、+26880 行，全部是文档/诊断，没有生产 Selector 源码改动。 | 研究的是 fixed spatial budget / cell coverage / same-K，不是 learned feature grid。文档明确写：`当前证据不支持正式加入 Sparse Adaptive Grid；停止本轮方向，不接在线训练`（`ADAPTIVE_SPATIAL_BUDGET_STUDY.md:234`），总决策 No-Go（`:3`）。 |
| `lightweight-tf` (`00a855c`) | `include/sparse_fine_lattice.h`，`SPARSE_FINE_IMPLEMENTATION.md`：Hierarchical Gradient-Grown Sparse Compensation Lattice；`FineKey=(level,ix,iy,iz)`、stable vertex ID、page 1024、CUDA hash query、per-row Adam、checkpoint。 | 这是 **TF compensation lattice**，不是 Selector feature grid；但它是当前代码库中“sparse learned spatial latent”的最佳工程先例。 |
| `tf-post-optimization` (`4d3c033`) | `include/residual_field.h`、`local_feature_adapter.*`、`ROR_CAPACITY_AND_ADAPTER_STUDY.md`；sparse residual field / cell cache / feature adapter。 | 也是 **TF residual**，不是 Selector；文档结论是 Feature Adapter Gate B 失败、停止 automatic growth（`ROR_CAPACITY_AND_ADAPTER_STUDY.md:5`）。 |
| `archive/gi-utility-maturity-20261018` (`6c6ada8`) | 10-18 研究改动归档（GI utility/staleness/Oldest、maturity ablation、teacher upperbound 等）。 | 是**被回退掉的研究快照**，不是当前生产路径；其中也没有 Selector learned grid。 |
| `tf-optimization`, `cuda-tf`, `protection-v2`, `feature/gi` 等 | 各自是 TF / GI / protection 历史分支。 | 没有 Selector 输入侧 learned spatial latent。 |

**结论：整个本地仓库没有 Selector 侧 Sparse Learned Feature Grid 的可复用实现。** 若后续要复用工程结构，`lightweight-tf` 的 `SparseFineLattice` / `tf-post-optimization` 的 `ResidualField` 是最接近的 sparse spatial latent 代码，但它们绑定 TF residual 语义，不能直接搬到 Selector。`[VERIFIED]`

### 1.6 两个 config 的关键差异（必须先 pin）

| 项 | `cfg/selector/selector_config.yaml` | `cfg/selector/fast.yaml` |
|---|---|---|
| `Selector.backend` | `cuda`（`:19`） | `cuda`（`:17`） |
| `TransformField.backend` | `legacy`（`:73`） | `cuda`（`:73`） |
| `TransformField.start_iter` | `5000`（`:74`） | `3000`（`:74`） |
| `Selector.lod_render_scale` | `1.0`（`:38`） | `0.5`（`:37`） |
| `Protection.performance_logging` | `1`（`:31`） | `0`（`:30`） |
| `Mapping.iteration_logging` | 未设置（默认 0） | `0`（`:8`） |
| `ChangeDistill.epochs` | `1`（`:117`） | `5`（`:117`） |
| `GlobalGI.mode` | `streaming`（`:58`） | `streaming`（`:58`） |
| `GlobalGI.performance_logging` | `1`（`:64`） | `0`（`:64`） |

`scripts/*.sh` 默认使用 `selector_config.yaml`（例如 `scripts/tum_rgbd.sh:3`），因此“默认生产脚本”是 **CUDA Selector + legacy TF + start 5000 + LOD scale 1.0 + CD epochs 1**，不是 `fast.yaml`。`[VERIFIED]`

---

## 2. Runtime Mapping call graph

### 2.1 真实训练入口

`examples/tum_rgbd.cpp:117-121`：

```cpp
std::shared_ptr<GaussianMapper> pGausMapper =
    std::make_shared<GaussianMapper>(pSLAM, gaussian_cfg_path, output_dir, 0,
                                     device_type, selector_enabled_override,
                                     *selector_config_path);
std::thread training_thd(&GaussianMapper::run, pGausMapper.get());
```

`main` 负责 SLAM 图像循环；Mapping 在独立线程中运行。`seed` 硬编码为 `0`（`tum_rgbd.cpp:118`），entry 不解析 CLI seed。

### 2.2 顶层调用链（文件 / 函数 / 行）

| 层 | 位置 | 说明 |
|---|---|---|
| entry | `examples/tum_rgbd.cpp:117-121` | 构造 mapper，启动 `run()` 线程 |
| `GaussianMapper::run` | `src/gaussian_mapper.cpp:932-1164` | 初始 mapping、增量 mapping、tail |
| 初始步骤 | `src/gaussian_mapper.cpp:1048-1049` | `trainForOneIteration()` |
| 增量循环 | `src/gaussian_mapper.cpp:1062-1097` | `combineMappingOperations()` + `trainForOneIteration()` |
| tail | `src/gaussian_mapper.cpp:1108-1143` | forced tail / fixed budget / 原 densify tail |
| `trainForOneIteration` | `src/gaussian_mapper.cpp:1253-1361` | 选 KF、调 `trainMappingIteration`、GI refresh、AABB、DSD |
| `trainMappingIteration` | `src/gaussian_mapper.cpp:1363-1800` | Full / LoD / commit / topology / tail / mode2 |
| `accumulateLodGradients` | `src/gaussian_mapper.cpp:2992-3242` | Selector forward、ST、GI、TF、LoD render、loss、backward |
| `stepWithHistoricalGuard` | `src/gaussian_mapper.cpp:2118-2140` | 无 Guard 时直接 mixed Adam step；默认 Guard=false |
| `ensureSharedMappingOptimizer` | `src/gaussian_mapper.cpp:2086-2116` | 把 Selector / TF 参数挂到 Gaussian shared Adam |
| `renderAndRecordAllKeyframes` / `savePly` | `src/gaussian_mapper.cpp:4014-4191` | 保存 |

### 2.3 一次 Mapping iteration 到底几次 render？

| 模式/阶段 | render 次数 | 证据 |
|---|---:|---|
| mode1 online，Full slot | **1**（Full render） | `full_step` 分支 `GaussianRenderer::render`（`gaussian_mapper.cpp:1459-1469`），LoD 分支条件 `(!rotation || !full_step)` 为 false（1496）。 |
| mode1 online，LoD slot | **1**（selected LoD render） | `accumulateLodGradients` 内 `selected_render_pkg = GaussianRenderer::render`（3174）。 |
| mode1 tail | **2**（Full + joint LoD） | Full 在 1459，tail LoD 在 1701-1712。 |
| mode2 | **2**（Full + detached LoD） | Full 在 1459，mode2 LoD 在 1766-1778。 |
| mode0 | **1**（Full only） | `rotation=false`，`full_step=true`，LoD 条件不成立。 |
| ConflictReplay | 额外 Full + LoD + quality renders | `trainJointReplay` 2903-2913；默认 `Selector.replay_enabled:0`（`fast.yaml:42`, `selector_config.yaml:43`）。 |
| DriftStability | 额外 historical renders | `evaluateDriftStability` 2611+；默认 `DriftStabilityConfig::enabled=false`（`include/drift_stability.h:19`），配置未开启。 |
| GlobalGI full refresh | 额外 GI view renders | `trainForOneIteration` 1327-1335 在 `refreshIfDue` 触发时对 bank members 调 `renderGlobalGIContribution`；`refresh_interval=1000`（`fast.yaml:62`, `selector_config.yaml:62`），不是每个 mapping。 |
| ChangeDistill | 不额外 render | CD Selector loss 只用 `forwardProbability`；TF loss 只用 `applyLogResidual` errors（`src/change_distill.cpp:158-177, 397-495`）。 |

`UltrafastRotation` 的五 slot 是 `1% -> 5% -> 10% -> 15% -> Full`，`phase=step%5`（`include/ultrafast_rotation.h:12-31`）。测试 `tests/ultrafast_rotation_test.cpp:10-24` 验证了 12 步中 2 次 Full / 10 次 LoD。`[VERIFIED]`

### 2.4 Full 与 LoD 是否分开？

**是，代码是两个独立分支。** `trainMappingIteration`：

```cpp
const bool full_step = mode1_tail || !rotation || ultrafast_rotation_.full();   // 1375
...
if (full_step) { ... Full render ... full_loss.backward(); ... }              // 1459-1494
if (!mode2 && enhanced && (!rotation || !full_step)) { ... LoD ... }          // 1496-1511
```

- Full：所有 Gaussian，`GaussianRenderer::render`，photometric loss，`full_loss.backward()`，然后 Gaussian optimizer step；**不调用 Selector / TF**。
- LoD：`accumulateLodGradients` → Selector → ST → selected compact → TF → selected render → `selector_loss.backward()`；Selector/Gaussian/TF 一起 step（mode1），或分段 step（mode2）。
- Full slot 的测试断言明确：`Full updated Selector/TF` 必须为 false（`tests/ultrafast_mapping_test.cpp:172-174`）。

### 2.5 ratio 在哪里生成？schedule 是什么？

- 配置解析：`readSelectorConfigFromFile` 读 `Selector.target_ratios`（`gaussian_mapper.cpp:537-555`）；mode1 下强制覆盖为 `UltrafastRotation::ratios = {0.01,0.05,0.10,0.15}`（`551-554`）。
- 每步选择：`phase = mode1_tail ? tailRatioIndex() : ultrafast_rotation_.phase()`（`gaussian_mapper.cpp:1377`）；LoD 用 `ratio_index=phase`（1497），`target_ratio=selector_target_ratios_[ratio_index]`（3024-3025）。
- mode2：`ratio_index = normal_lod_branch_count_ % selector_target_ratios_.size()`（1771）。
- schedule：固定 1/5/10/15/Full 循环，按**normal mapping iteration**推进；`UltrafastRotation::complete` 在每次正常 Full/LoD 分支结束时调用（`gaussian_mapper.cpp:1690-1692`），`step++`。它不是按成功 optimizer update 推进；Guard 拒绝时 schedule 仍前进。`[VERIFIED]`
- tail：`tailRatioIndex()` 从 online 结束后的下一个 LoD ratio 开始，按 tail LoD attempt 继续（`ultrafast_rotation.h:43-46`）；测试 `ultrafast_rotation_test.cpp:34-65`。

### 2.6 Selector 在哪些 branch 被调用？

只有 **LoD 分支**：

- mode1 online 的非 Full slot：`trainMappingIteration` 1496-1511 → `accumulateLodGradients`；
- mode1 tail：1701-1712 → `accumulateLodGradients(..., native_lod=true)`；
- mode2：1766-1778 → `accumulateLodGradients(..., normal_mode2=true)`；
- ConflictReplay：2912-2913（默认关闭）；
- ChangeDistill：`selectorProbabilities` 直接调 `forwardProbabilities`（`src/change_distill.cpp:171-177`），不是 mapping branch；
- DriftStability：`forwardKeepLogit`（`src/drift_stability.cpp:177`），默认关闭。

**Full 分支不调用 Selector。** `[VERIFIED]`

### 2.7 TF 在哪些 branch 被调用？

只在 `accumulateLodGradients` 内，且只在 selected compact 行上：

```cpp
if (transform_field_ && compact_count > 0) {                       // 3117
  inside = transform_field_->insideMask(selected_input.xyz);       // 3118
  const auto transform_mask = transform_field_->candidateMask(...); // 3121
  auto transformed = transform_field_->applyResidual(...);          // 3135-3139
  selected_input.xyz/scaling/rotation = index_copy(...);            // 3140-3145
}
```

即 mode1 LoD / mode1 tail LoD / mode2 LoD / Replay LoD；Full 不调用。TF 也可以被 ChangeDistill 的 `applyLogResidual` 调用（`src/change_distill.cpp:397-398`）。`[VERIFIED]`

### 2.8 Gaussian optimizer 在哪些 branch step？

- **mode1**：每个 normal mapping iteration 都 step；Full 和 LoD 都走 `commit_optimizer` → `stepWithHistoricalGuard`（`gaussian_mapper.cpp:1556`）。LoD 是 joint：Gaussian + Selector + TF 一起更新。
- **mode2**：Full 阶段 `commitMode2Stage(true)` step Gaussian（`1521`）；LoD 阶段 `commitMode2Stage(false)` 只 step 非 Gaussian groups（`1782`）。
- **mode0**：Full 分支末尾 `commit_optimizer()`（`1665`）step 所有 group；没有 LoD。
- **tail**：mode1 tail 先 Full step（1563/1728），再 LoD step；mode2 tail 同上。
- `SparseGaussianAdam` 6 个 group：xyz / f_dc / f_rest / opacity / scaling / rotation（`include/gaussian_model.h:203-207`, `src/gaussian_model.cpp:513-531`）。

### 2.9 Selector optimizer 在哪些 branch step？

**正常 mapping 不使用 `selector_optimizer_` 直接 step。** `ensureSharedMappingOptimizer` 把 `selector_network_->parameters()` 追加到 `gaussians_->optimizer_`（`gaussian_mapper.cpp:2090-2099`）；normal step 走 shared optimizer。`selector_optimizer_` 只在构造时创建（`167-169`），并作为 ChangeDistill 的 `normal_selector_optimizer` 模板（`finishGeometryChange` 2440；`ChangeDistillManager::execute` 313-315）。ChangeDistill 用 `independentAdam` 复制参数组、重置 m/v、按事件 LR 训练（`src/change_distill.cpp:16-29, 313-349`）。`[VERIFIED]`

### 2.10 TF optimizer 在哪些 branch step？

- 正常 mapping：shared optimizer 的 `shared_transform_mlp_group_index_` / `shared_transform_grid_group_index_` 组（`gaussian_mapper.cpp:2101-2115`），在 LoD 分支 step。
- `transform_optimizer_`（构造于 `2057-2062`）保存 MLP+grid 参数组，正常 mapping 不直接 step；ChangeDistill 用 `context.normal_tf_optimizer` 复制独立 Adam（`src/change_distill.cpp:457-478`）。
- `transform_optimizer.pt` 会保存（`gaussian_mapper.cpp:4188-4190`），但当前没有 C++ load 它；Python `run.py` 也不 load。

### 2.11 densify / prune / topology update 在哪里？

只在 **Full step 提交后**、`topology_window` 内：

```cpp
const bool topology_window = getIteration() < opt_params_.densify_until_iter_ ||
    opt_params_.densify_until_iter_ == -1;                                  // 1597-1598
if (full_step && topology_window) { ...                                   // 1606
  gaussians_->addDensificationStats(...);                                 // 1614
  if (rotation ? ultrafast_densify_task_.take(true) : ...)
    gaussians_->densifyAndPrune(...);                                     // 1626
  ...
  gaussians_->resetOpacity();                                             // 1640
}
```

`densifyAndPrune` 内部顺序：`densifyAndClone` → `densifyAndSplit` → `prunePoints`（`src/gaussian_model.cpp:872-893`）。LoD 分支不更新 densification stats，也不 prune。`[VERIFIED]`

---

## 3. Selector architecture

### 3.1 类与文件

| 实现 | 位置 | 说明 |
|---|---|---|
| C++ 训练/推理 | `improvements/selector/gumbel_network.h:14-194` | `GumbelNetworkImpl`；`TORCH_MODULE(GumbelNetwork)` |
| C++ CUDA 后端 | `include/selector_cuda.h`, `src/selector_cuda.cpp`, `src/selector_cuda.cu` | `selector_cuda::network` 自定义 autograd Function |
| Python eval | `python/selector.py:13-57` | `SelectorNetwork`，与 C++ legacy 对应 |
| Python CUDA eval | `python/selector_cuda_backend.py`, `python/selector_cuda_bindings.cpp` | 编译同一份 `selector_cuda` |

### 3.2 网络结构（legacy 与 CUDA 数学等价）

```text
pos_emd:      3 -> 32 (Linear+ReLU) -> 32 (Linear+ReLU)
rotation_emd: 4 -> 32 (Linear+ReLU) -> 32 (Linear+ReLU)
scale_emd:    3 -> 32 (Linear+ReLU) -> 32 (Linear+ReLU)
time_emd:     1 -> 32 (Linear+ReLU) -> 32 (Linear+ReLU)
concat 128 -> soft_net: Linear(128, 2)
output: logits [N,2]
probability: softmax(logits / tau)[:, 1]
```

C++ 注册顺序：`pos_emd`, `rotation_emd`, `scale_emd`, `time_emd`, `soft_net`（`gumbel_network.h:31-37`）。参数总数 **4,962**：

```text
pos:      3*32+32 + 32*32+32 = 1184
rotation: 4*32+32 + 32*32+32 = 1216
scale:    3*32+32 + 32*32+32 = 1184
time:     1*32+32 + 32*32+32 = 1120
soft_net: 128*2+2            =  258
total                        = 4962
```

CUDA `validate` 固定 4 输入 / 18 参数 / final `[2,128]`（`src/selector_cuda.cu:77-94`）；`packFeatures` 把四个 32-d 分支拼成 128（`123-127`）；`gemmBiasRelu(packed, p[16], p[17], h[8], false)`（`259-268`）。`[VERIFIED]`

### 3.3 输入构造（实际 tensor shape）

调用点：`src/gaussian_mapper.cpp:3055-3065`

```cpp
const auto mixed_probability = elasticSelectorProbability(
    maturity,
    gaussians_->protection_.selectionMaturity(selector_protection_enabled_),
    entry, target_ratio, entry_indices,
    [&](const torch::Tensor& indices) {
      const auto positions = indices.defined() ? xyz.index_select(0, indices) : xyz;
      const auto rotations = indices.defined() ? rotation.index_select(0, indices) : rotation;
      const auto scales = indices.defined() ? scaling.index_select(0, indices) : scaling;
      return selector_network_->forwardProbability(
          positions, rotations, scales, target_ratio, selector_temperature_);
    });
```

其中 `xyz / rotation / scaling` 来自 `gaussians_->getXYZ()` / `getRotationActivation()` / `getScalingActivation()`（`3036-3043`）：

| 输入 | shape | dtype | 来源 | 说明 |
|---|---|---|---|---|
| `position` | `[M,3]` | FP32 | `GaussianModel::xyz_`（raw world） | 未归一化、无 PE、TF 前、loop closure 后的当前 xyz |
| `rotation` | `[M,4]` | FP32 | `normalize(rotation_)` | raw quaternion 归一化；不是 world rotation matrix |
| `scale` | `[M,3]` | FP32 | `exp(scaling_)` | activated scale（世界尺度） |
| `target_ratio` | `[1,1]`（C++ 正常路径）或 `[M,1]`（Python eval） | FP32 | `selector_target_ratios_[ratio_index]` | 标量 ratio；广播到 M 行 |
| `maturity` | `[N]` | FP32 | `protection_.maturity.detach()` | **不作为网络输入**；只用于 `mixSelectorProbability` |
| `cached_uniform` | `[N]` | FP32 | `protection_.cached_uniform` | **不作为网络输入**；只用于 ST/Bernoulli |
| `entry_indices` | `[E]` int64 | — | `indicesAbove(entry)` | 仅 entry>0 时使用；当前 entry=0 |

M = N（entry=0）或 eligible 行数（entry>0）。`mixed_probability` 返回 `[N]`（ineligible 行被填成 ratio）。`[VERIFIED]`

**没有**：opacity、SH、`exist_since_iter_`、`last_elastic_supervised_iter_`、`touched_in_cycle`、GI importance/valid/observed、camera/frame id、visibility、frustum、global code、邻域、voxel/hash/embedding、normalized AABB 坐标。`[VERIFIED]`

### 3.4 坐标到底是什么？

- **world coordinate**：`gaussians_->getXYZ()` 直接来自 `xyz_`（`src/gaussian_model.cpp:70`）；没有 AABB 归一化、没有相机坐标、没有 positional encoding。
- **TF 前**：Selector 在 `accumulateLodGradients` 的 TF 之前计算 p（`3055-3065` vs `3117-3145`）；TF 只作用于已选中的 compact 行，不会反过来改变 p。
- **Loop Closure 后**：`applyScaledTransformation` / `scaledTransformVisiblePointsOfKeyframe` 直接原地修改 `xyz_`（`src/gaussian_model.cpp:399-497`）；之后的 Selector 看到的是修正后的 world xyz。
- **没有 normalized coordinates**：`python/selector.py:124-129` 同样直接用 `gaussians.get_xyz`；`python/transform_field.py:73-77` 的 `2*(xyz-aabb_min)/(aabb_max-aabb_min)-1` 只属于 TF，不属于 Selector。

### 3.5 ratio conditioning

- 不是 concat 原始 scalar；先经过 `time_emd: 1->32->32`（`gumbel_network.h:153-158`）。
- 与三个空间 branch 的 32-d embedding 在通道维 concat 成 128，再进 `soft_net`。
- 同一个 batch 内 ratio 是同一个标量（[1,1] 广播）；没有 per-row ratio，也没有 Fourier feature / condition layer / separate head。
- CUDA 后端同样把 ratio 走 `linear(x[3], p[12], p[13])` + `linear(h[6], p[14], p[15])`（`selector_cuda.cu:249`），然后 `packFeatures` 把 ratio 分支作为第 4 个 32-d 块（`263-268`）。
- `forwardProbabilities` 可以一次算多个 ratio；CUDA 路径 `networkMany` 复用空间 branch 的 activation，但每个 ratio 仍然单独 final GEMM（`gumbel_network.h:122-141`, `src/selector_cuda.cpp:53-68`）。

### 3.6 选择机制：训练 vs 评估

**训练（normal mapping）**

```cpp
const auto selector_mask = sampleSelectorST(mixed_probability,
                                            gaussians_->protection_.cached_uniform); // 3067
```

`sampleSelectorST`（`include/selector_handoff.h:14-21`）：

```cpp
hard = (uniform < probability)
mask = hard + (probability - probability.detach())
```

- 前向是 **hard Bernoulli**，不是 threshold、不是 Top-K、不是 Gumbel。
- 反向是 **straight-through**：`d mask / d p = 1`。
- `cached_uniform` 是 checkpoint 中保存的固定 per-Gaussian uniform（`gaussian_protection.h:17,54,89`；`selector_metadata.ply` 保存/加载：`gaussian_model.cpp:1078-1134`）。
- 因此给定 checkpoint、ratio、p，mask 是确定的；normal training 不再每步抽随机数。`[VERIFIED]`

**评估（Python `build_selector_mask`）**

```python
mask = metadata["cached_uniform"] < probability
```

`python/selector.py:112-144`；无 ST、无 Gumbel、无梯度。`run.py` 用这个 mask 选行、应用 TF、导出 PLY，然后渲染导出的 PLY（`python/run.py:292-343`）。`[VERIFIED]`

**Gumbel / 其它接口**

`GumbelNetwork::forward` 使用 `gumbelSoftmax`（`gumbel_network.h:39-51,170-182`），`forwardHard` 也用 Gumbel noise（`54-65`），`forwardHardWithUniform` 是历史 Guard 的固定 uniform ST（`81-94`）。这些**不是 normal training 路径**；normal 只用 `forwardProbability` + `sampleSelectorST`。不要因为类名叫 GumbelNetwork 就假设训练用 Gumbel。`[VERIFIED]`

**实际 K**

`K = sum_i [cached_uniform_i < p_i]`，不是严格的 `round(rN)`；`ratio_loss` 只惩罚 `sum(p)` 与 `rN` 的偏差（`selector_handoff.h:22-25`）。评估脚本只报告 `actual_active` / `actual_vs_target`，不做 Top-K（`python/selector.py:130-137`）。`[VERIFIED]`

---

## 4. Selector 的梯度到底来自哪里

### 4.1 总 loss

`accumulateLodGradients` 的 `selector_loss`（`src/gaussian_mapper.cpp:3193-3215`）：

```text
L_selector =
    w_render * [ (1-λ_dssim) * L1(selected_render, gt)
                 + λ_dssim * (1 - SSIM(selected_render, gt)) ]
  + w_ratio  * |sum(p_mix) - r*N| / max(1, r*N)
  + w_gi     * ratio_weight * L_gi
  + [if transform_count > 0] L_tf_reg
```

其中：

- `w_render = Selector.render_loss_weight = 1.0`
- `w_ratio = Selector.ratio_loss_weight = 0.01`
- `w_gi = Selector.gi_loss_weight = 1.0`
- `L_gi`：`GITeacher::loss(selector_mask, smooth_l1)`（`src/gi_teacher.cpp` / `include/gi_teacher.h:11-17`）
- `L_tf_reg = transform_field_->regularizationLoss()`（`src/transform_field.cpp:132-139`）

Full 分支的 `full_loss`（`gaussian_mapper.cpp:1481-1487`）是另一个 loss：

```text
L_full = (1-λ_dssim) * L1(full_render, gt) + λ_dssim*(1-SSIM) + opacity_reg
```

Full 分支没有 Selector mask，也没有 `ratio/GI` loss。`[VERIFIED]`

### 4.2 autograd 关系（实际图）

```text
Selector θ ─────────────────────────────┐
xyz, rotation, scale (mode1 不 detach) ──┤
ratio (constant) ────────────────────────┴─> p [N]
                                                │
maturity m (detach) ──> p_mix = (1-m)*r + m*p  │
                                                │
cached_uniform u ──> hard = [u < p_mix]        │
                     mask = hard + (p_mix - p_mix.detach())
                                                │
              ┌─────────────────────────────────┼─────────────────────────────┐
              │                                 │                             │
              ▼                                 ▼                             ▼
   compact_to_global = nonzero(mask.detach())   ratio_loss(p_mix)      GI loss(mask, label)
              │                                 │                             │
              ▼                                 │                             │
 selected_input = gather(full_input)            │                             │
 selected_input.opacity *= mask                 │                             │
              │                                 │                             │
              ▼                                 │                             │
    TF (optional) transforms selected xyz/scale/rot
              │                                 │
              ▼                                 │
        selected render ──> photo loss ─────────┴──────────────> L_selector
```

关键点：

- `compact_to_global` 用 `selector_mask.detach() > 0`（`3100-3101`），所以 **indices 不携带梯度**；梯度只通过 `selected_input.opacity = compact_mask * opacity`（`3108-3111`）回到 `p`。
- `mask` 前向 0/1，反向 `d mask/d p = 1`；所以 photo loss 对 `p` 的梯度 = `dL/dopacity_selected * opacity_selected`（在 selected 行），未选行没有 photo 梯度。
- `p_mix = (1-m)*r + m*p`，`m.detach()`；因此 **所有通过 `p_mix` 的梯度都被乘以 m**：photo、ratio、GI 都受影响。测试 `tests/selector_handoff_test.cpp:11-18` 明确验证 `p.grad == m`、`m.grad` 不存在。
- `mode1` 不 detach Gaussian 参数（`3036-3052`，`selector_detach_gaussian_parameters_` 默认 0），所以 `p` 对 `xyz/rotation/scale` 的梯度也会回到 Master；`mode2` detach（`3003`），只训练 Selector/TF。
- `full_loss` 不经过 Selector；Full 分支不会给 Selector 梯度。

### 4.3 回答重点问题

1. **photometric loss 是否真的回传到 Selector？**
   **是**，但只在 LoD 分支，且通过 ST mask 的 opacity 乘权；梯度被 maturity 乘权。Full 分支不回传。`[VERIFIED]`
2. **Gaussian 参数是否 detach？**
   mode1 不 detach（joint）；mode2 detach；`Selector.detach_gaussian_parameters` 只在 mode0 生效（`gaussian_mapper.cpp:3003` 对 mode2 强制 true）。`[VERIFIED]`
3. **TF 是否 detach？**
   TF 参数不 detach；selected TF residual 参与 photo loss，TF 参数从 `L_selector` 收到梯度。Selector 和 TF 是 loss 的两条并行参数路径；Selector 的 p 只决定哪些行被 TF 处理，TF 的输出不会反传给 p。`[VERIFIED]`
4. **mask 是 soft / hard / ST 哪一种？**
   前向 hard Bernoulli；反向 ST（`hard + p - p.detach()`）。不是 soft mask，不是 Gumbel，不是 Top-K。`[VERIFIED]`
5. **哪些 loss 只更新 Selector？**
   没有“只更新 Selector”的 normal loss：photo/ratio/GI 在 mode1 同时更新 Gaussian（通过 p 的输入梯度）和 TF（通过 selected residual）；mode2 中 photo/ratio/GI 更新 Selector+TF，不更新 Gaussian。ChangeDistill 的 Selector CD loss 是 Selector-only（独立 Adam，不经过 renderer/Gaussian）。`[VERIFIED]`
6. **哪些 loss 同时更新 Gaussian？**
   Full photometric（Gaussian only；Selector/TF 无梯度）；LoD photo/ratio/GI 在 mode1 同时更新 Gaussian/Selector/TF；TF regularization 只更新 TF。`[VERIFIED]`

### 4.4 branch-specific backward 列表

| backward | 位置 | 更新对象 |
|---|---|---|
| `full_loss.backward()` | `gaussian_mapper.cpp:1489-1490` | Gaussian（Full） |
| `selector_loss.backward()` | `gaussian_mapper.cpp:3225-3228` | mode1：Gaussian+Selector+TF；mode2：Selector+TF |
| Replay `full_loss.backward()` + `accumulateLodGradients` | `gaussian_mapper.cpp:2903-2913` | 受 Guard preview/commit 控制；默认关闭 |
| CD `total.backward()` | `src/change_distill.cpp:343-349` | Selector only（独立 Adam）；TF CD 在 469-478 |
| DriftStability | `src/drift_stability.cpp` | 默认关闭；不是 normal mapping |

---

## 5. 当前 Selector 的 scene-specific 信息有多少

### 5.1 分类

| 类别 | 当前有什么 | 是否进入 Selector |
|---|---|---|
| A. Gaussian 自带信息 | raw `xyz`、raw rotation（归一化后输入）、raw scaling（exp 后输入）；另有 opacity、SH、`exist_since_iter_`、`last_elastic_supervised_iter_` 等 | 只有 xyz/rotation/scale；opacity/SH/age/supervision 都不进入 |
| B. 网络全局参数 | Selector θ（4,962 参数） | 是（共享） |
| C. 当前 ratio | `target_ratio` 标量 | 是（time_emd） |
| D. 邻域信息 | 无 KNN / graph / conv / voxel / cell statistics | **否** |
| E. scene-specific persistent external state | `maturity`、`cached_uniform`、GI importance/valid/observed、`last_elastic_supervised_iter_` | 都不作为网络输入；maturity 只乘在输出混合，uniform 只用于采样，GI 只作为 teacher label |

### 5.2 是否存在 `position -> learnable latent`？

**没有。** 我检查了以下机制，当前 core source 均不存在：

- voxel feature
- hash grid / `cell_id` / embedding table
- spatial memory
- feature plane（TF 的 HexPlane 是独立 TF 模块，不是 Selector 输入）
- local MLP / octree feature
- per-cell statistics
- per-Gaussian latent / stable ID

**准确表述：**

> 当前 Selector 是 per-Gaussian pointwise MLP；位置只以 raw world `xyz` 的形式进入共享 MLP，没有显式的、persistent 的 scene-specific spatial latent，也没有邻域/全局上下文。它可以在共享权重里学到一个粗糙的全局空间先验，但不能在不改变 Gaussian 参数的情况下独立更新某个 cell/region 的选择知识。

这也是为什么 raw xyz 虽然存在，仍然不能等同于 Grid：Grid 提供的是 **空间共享的额外可学习参数**，而当前网络只能通过共享 θ 和非线性 MLP 间接表达位置。`[VERIFIED/INFERRED]`

### 5.3 关键含义

- 加 Grid 不会增加新的观测信息；它增加的是 **空间参数化 / 空间共享先验 / 额外容量**。因此“Grid 有用”必须与“更多参数”和“raw xyz 已经能表达位置”这两个替代解释区分开。
- 当前 `pos_emd` 直接吃 raw xyz（米制尺度），没有归一化/PE；不同 scene 的坐标尺度会改变 MLP 的有效频率。Grid 若用固定 cell size，也会受 scene 尺度影响。`[VERIFIED]`

---

## 6. Gaussian 数据结构与 Grid 查询点

### 6.1 核心 storage

`GaussianModel`（`include/gaussian_model.h:74-212`）全部是 LibTorch `torch::Tensor`，不是 `std::vector`，不是 CPU 结构体数组。训练时 `device_type_ = CUDA`；CPU fallback 时在 CPU。

| 字段 | shape | dtype | 含义/备注 |
|---|---|---|---|
| `xyz_` | `[N,3]` | FP32 | raw world xyz；`requires_grad` |
| `features_dc_` | `[N,1,3]` | FP32 | SH DC |
| `features_rest_` | `[N,15,3]` | FP32 | SH rest（degree 3） |
| `opacity_` | `[N,1]` | FP32 | raw logit；`sigmoid` 后才是 opacity |
| `scaling_` | `[N,3]` | FP32 | log scale；`exp` 后才是 scale |
| `rotation_` | `[N,4]` | FP32 | raw quaternion；`normalize` 后才是 rotation |
| `max_radii2D_` | `[N]` | FP32 | densify/prune 统计 |
| `xyz_gradient_accum_` | `[N,1]` | FP32 | densify 统计 |
| `denom_` | `[N,1]` | FP32 | densify 统计 |
| `exist_since_iter_` | `[N]` | Int32 | 创建 iteration；**不进入 Selector，也不保存到 PLY** |
| `last_elastic_supervised_iter_` | `[N]` | Int64 | 最近 LoD supervision iteration；不进入 Selector |
| `protection_.maturity` | `[N]` | FP32 | continuous maturity，`[0,1]` |
| `protection_.drift_ema` | `[N]` | FP32 | drift EMA |
| `protection_.cached_uniform` | `[N]` | FP32 | 固定 Bernoulli uniform |
| `protection_.prev_full_xyz` | `[N,3]` | FP32 | Full 快照 |
| `protection_.prev_full_log_scale` | `[N,3]` | FP32 | Full 快照 |
| `protection_.prev_full_rotation` | `[N,4]` | FP32 | Full 快照 |
| `protection_.snapshot_valid` | `[N]` | Bool | 快照有效 |
| `protection_.touched_in_cycle` | `[N]` | Bool | 本 cycle 是否被 touched |
| `protection_.drift_initialized` | `[N]` | Bool | drift 是否初始化 |
| optimizer state | 6 组 | FP32 | `exp_avg`, `exp_avg_sq`, step；见 `SparseGaussianAdam` |

`exist_since_iter_` 在 `createFromPcd` 置零（`src/gaussian_model.cpp:167-169`）；`increasePcd` 设为当前 iteration（`256-258`）；clone/split 继承 parent（`832, 864-865`）。`savePly` 只写 xyz/normals/f_dc/f_rest/opacity/scale/rot（`src/gaussian_model.cpp:1136-1230`），**不写 `exist_since_iter_`、不写 stable ID、不写 optimizer**。`[VERIFIED]`

### 6.2 有没有 stable ID？

**没有。** 没有 `id_` / `gaussian_id` / `stable_id` 字段。row index 是唯一身份；prune 用 boolean mask 重排（`prunePoints` 611-679），append 在末尾（`densificationPostfix` 697-785）。clone/split 的 `parent_indices` 只在 `densificationPostfix` 调用内传给 `protection_.append`（768），之后不保留 child→parent 关系。`[VERIFIED]`

### 6.3 Grid 要为每个 Gaussian query `F(x_i)`，最自然在哪里？

**在 Selector 输入构造点**，不是在 Gaussian storage：

- `src/gaussian_mapper.cpp:3058-3064`：`positions/rotations/scales` 已经在 `[M,*]` 形状；在这里调用 `F(positions)`，拼到 Selector 输入即可。
- 不需要给 `GaussianModel` 加字段；不需要把 `F` 存回 `xyz_` 或 optimizer。
- 如果未来要缓存 per-Gaussian F，才需要改 storage；v0 不建议（会引入 topology 对齐、checkpoint 和 optimizer 的额外复杂度）。

---

## 7. topology: densify / clone / split / prune

### 7.1 append/new Gaussian

- **新 map point**：`combineMappingOperations` 的 `LocalMappingBA` / `LoopClosingBA` 在 `initial_mapped_ && points.size() >= 30` 时调用 `gaussians_->increasePcd(points, colors, getIteration())`（`gaussian_mapper.cpp:3312-3315, 3411-3414`）。
- `increasePcd` 设 `new_exist_since_iter = iteration`（`gaussian_model.cpp:256-258`），然后 `densificationPostfix`：
  - optimizer 6 组都 `cat` 新行，Adam state 为新行补零（`710-747`）；
  - `exist_since_iter_` cat（766-767）；
  - `protection_.append(count, parents={})`：maturity/drift/valid 等补零，`cached_uniform` 新抽 `torch::rand`（75-91）；
  - `last_elastic_supervised_iter_` 新行设为 `mapping_iteration`（769-771）；
  - `online_gi_append_callback_` / `gi_append_event_callback_` 通知 GI（781-784）。
- **没有 stable ID**；新行就是末尾 row。

### 7.2 clone/split

`densifyAndClone`（`src/gaussian_model.cpp:846-870`）：

- 选出 `selected_pts_mask`；`new_xyz/features/opacity/scaling/rotation` 直接 `index_select` 复制；
- `new_exist_since_iter` 也复制 parent 的 `exist_since_iter_`；
- `parent_indices = nonzero(selected_pts_mask)`；
- `densificationPostfix(..., AppendKind::Clone, parent_indices)`；
- `protection_.append` 用 parents 继承 maturity/drift/drift_initialized（`gaussian_protection.h:75-91`）；`prev_full_*` 补零、`snapshot_valid/touched` 补零、`cached_uniform` 新抽。

`densifyAndSplit`（797-844）：

- `new_xyz = parent.xyz + cov * N(0,1)`（815-820）；scale 除以 `0.8*N`（821-823）；
- `new_rotation/features/opacity` 复制 parent；
- `parent_indices = nonzero(selected_pts_mask).repeat(N)`；
- `AppendKind::Split`；
- 然后 `prunePoints(prune_filter)` 删掉被 split 的 parent rows（838-843）。

**parent/child 没有持久 ID relationship**；只在 append 时把 `parent_indices` 传给 maturity 继承。`[VERIFIED]`

### 7.3 prune

`prunePoints`（611-679）：

- `valid_points_mask = ~mask`；
- 6 个 optimizer group 的 param 和 Adam state 都 `index({valid_points_mask})`；
- `exist_since_iter_`、`protection_` 全部 `index`；
- `last_elastic_supervised_iter_`、densify stats 同样 `index`；
- 行顺序按 survivors 重新压缩，**没有 stable ID 保持**；
- `online_gi_prune_callback_(valid_points_mask)` 通知 GI。

`densifyAndPrune` 的顺序是 clone → split → prune（872-893），所以同一轮里 append 的新行会插在末尾，然后 prune 重排。`[VERIFIED]`

### 7.4 对 Grid 的结论

- **按 Gaussian ID 存 feature 不可行**（没有 ID）。
- **按 world-space coordinate 存 feature 可行**：topology 改变 row 顺序不影响 `F(x)`；新点 query 到 cell 就得到对应 latent；prune 不需要删除 cell（可能留下 dead cell）。
- clone/split 的新点与 parent 位置接近，会自然 query 到相同/相邻 cell；不需要显式 feature 继承。若希望 split 后 child 继承 parent 的 cell feature，world-space 查询天然近似做到。
- 如果一定要 per-Gaussian feature，需要在 `densificationPostfix` / `prunePoints` 增加 append/prune 回调（类似 GI 的 callback），并维护 child→parent 继承；这是额外工程量，v0 不建议。

---

## 8. Loop Closure / Scale Refinement / geometry change

### 8.1 会改变 Gaussian XYZ 的地方

| 操作 | 代码 | 影响范围 | 修改 |
|---|---|---|---|
| LoopClosingBA，大旋转/大平移 | `gaussian_mapper.cpp:3319-3421`；`GaussianModel::scaledTransformVisiblePointsOfKeyframe`（`gaussian_model.cpp:427-497`） | 该 KF 可见且未 transform 的点；`point_unstable_flags` 基于 `exist_since_iter_`（444-447） | `xyz` 应用 `diff_pose` + `loop_kf_scale`；`scaling` 乘 scale；`rotation` 乘 rotation；`replaceTensorToOptimizer` 更新 param；`protection_.transportSnapshot` 更新 prev_full 快照 |
| ScaleRefinement | `gaussian_mapper.cpp:3423-3467`；`GaussianModel::applyScaledTransformation`（399-425） | **全部 Gaussian**（`affects_all=true`） | `xyz = s*R*xyz + t`；`scaling *= s`；`rotation *= R`；`replaceTensorToOptimizer`；`scene_->applyScaledTransformation` |
| Local BA 只改 pose | `gaussian_mapper.cpp:3289-3303` | KF pose，不改 Gaussian xyz | 后续 LoD/Full 渲染使用新 pose |
| TF 域扩展 | `evaluateDynamicAABB`（2470-2531）/ `finishGeometryChange`（2406-2468） | 只改 TF AABB / 平面，不改 Master xyz | `resampleDomain` + `setAABB`；`transform_optimizer_` / shared optimizer 的受影响参数 state erase（`change_management.cpp:318-333`） |
| TF 本身对 selected 的 residual | `accumulateLodGradients:3135-3145` | 只改 selected_input 的副本，不回写 Master | `selected_input.xyz/scaling/rotation` |
| ChangeDistill teacher transport | `src/change_distill.cpp:47-146` | 只改 teacher 的 target，不改 live Master | `TargetTransport::apply` |

### 8.2 发生在哪个线程/阶段？

- 全部在 Mapping 线程；geometry mutation 在 `mutex_render_` 下（`gaussian_mapper.cpp:3314, 3374-3388, 3435-3442`）。
- Loop close / scale refinement 来自 `combineMappingOperations`，在 `trainForOneIteration` 之前的增量 mapping 条件块中执行（`run` 1077-1080）。
- CD / TF domain resample 在 `finishGeometryChange` 内，仍在 `mutex_render_` 下（`2409`, `2415`）；CD 会继续占用 render lock 直到 selector/TF distillation 完成。
- `GaussianProtection::transportSnapshot` 只更新 `prev_full_*`，不更新 `maturity` 本身；maturity 由 Full 后的 `observeFull` 更新。`[VERIFIED]`

### 8.3 如果 Grid key 是 world xyz，geometry correction 后会不会跨 cell？

**会。** `applyScaledTransformation` 对全部点做 Sim(3)，`scaledTransformVisiblePointsOfKeyframe` 对可见子集做局部变换；点会移动到新的 world cell。

- 如果 Grid 查询的是**当前** `xyz`（推荐 v0），那么修正后自然 query 新 cell；旧 cell 的 latent 不会跟随点。语义是“这个空间位置的 selection 知识”，不是“这个点的历史”。
- 如果希望 latent 跟随点，需要 transport：`ResidualField` 的 `transportGlobal` / `transportSupport` 是现成工程先例（`tf-post-optimization:include/residual_field.h`）；当前 Selector 没有这种机制。
- 如果 Grid 是 dense fixed AABB，跨 cell 本身没问题；问题是旧 cell 的统计/学习信号不会迁移。若 loop closure 后场景整体平移/缩放，fixed AABB 也可能把点推到外面。

### 8.4 当前有没有 `transformed_indices`？

- `GaussianTransformResult` 有 `indices`、`affects_all`、`population_count`（`include/gaussian_transform_ops.h`）；`applyScaledTransformation` 设 `affects_all=true`；`scaledTransformVisiblePointsOfKeyframe` 设置局部 `indices`。
- `ChangeCapture` / `ChangeEvent` 有 `correction_groups`，每组 `CorrectionGroup` 继承 `GaussianTransformResult`，带 `indices` 和 `keyframe_id`；`ChangeEvent.correction_group_id` 是 `[N0]`，-1 表示 unaffected，>=0 是组 id（`include/change_management.h:69-95`；`src/change_management.cpp:193-248`）。
- **但没有持久化的全局 `transformed_indices` 文件/张量**；事件结束后 `ChangeEvent` 不保存。若 Grid 需要 event-time transport，需要在 `finishGeometryChange` 里加 hook 或导出事件。

**因此：当前回退版本不是“没有 ChangeEvent”**，ChangeEvent/CorrectionGroup 是存在的，且可用于知道哪些点被修正；只是没有 Grid 所需的持久 event trace。`[VERIFIED]`

---

## 9. 当前 AABB / spatial normalization

### 9.1 哪里创建 scene AABB？

- TF 初始化：`GaussianMapper::initializeTransformFieldIfNeeded`（`gaussian_mapper.cpp:2017-2084`）。
  - `xyz = getXYZ().detach()`；
  - `mature_xyz = xyz[elasticDomainMask(xyz, maturity, domain_threshold)]`；
  - `aabb_min = min(mature_xyz)`，`aabb_max = max(mature_xyz)`；
  - `center=(min+max)/2`，`half_extent=(max-min)*0.5*max(aabb_scale,1)`，`clamp_min(1e-3)`；
  - `TransformField(aabb_min, aabb_max, config)`。
- 这不是 Selector 的 AABB；Selector 完全不使用 AABB。`[VERIFIED]`

### 9.2 会不会动态扩大？

**会。** `dynamic_aabb_config_.enabled` 默认 true（`include/change_management.h:32-38`），`TransformField.dynamic_aabb:1`（`cfg/selector/*.yaml`）；`evaluateDynamicAABB` 每 `aabb_check_interval`（默认 1000）检查一次（`gaussian_mapper.cpp:1337-1342, 2470-2531`）：

- `AABBPressure` 统计 outside 数量和方向；
- `makeExpansionPlan` 用低/高 quantile + `expansion_margin`（`change_management.cpp:120-164`）；
- 扩张是**单调的**（新 min <= 旧 min，新 max >= 旧 max）；
- `HexPlaneField::resampleDomain` 保持平面分辨率不变，把旧域重采样到新域（`src/hexplane_field.cpp:80-131`）；
- `setAABB` 更新 buffer；受影响参数的 optimizer state 被 erase（`change_management.cpp:318-333`）。

### 9.3 Selector 是否使用 AABB？

**不。** `python/selector.py` 与 `src/gaussian_mapper.cpp:3058-3064` 都直接传 raw `xyz`；没有 `aabb_min/max`。`python/transform_field.py:73-77` 的归一化只属于 TF。`[VERIFIED]`

### 9.4 TF 是否使用 AABB？

**是。** `HexPlaneField::forward` 计算 `normalized_xyz = AABB3D{aabb_min_,aabb_max_}.normalize(xyz)`（`src/hexplane_field.cpp:170-171`），再与 ratio 拼成 4D 坐标采样 6 平面（153-186）。`insideMask` / `candidateMask` 也用 AABB（`49-54`, `include/transform_field.h:80-87`）。`[VERIFIED]`

### 9.5 长序列越界怎么处理？

- TF：`outside_identity:1`；`applyLogResidual` 只对 `insideMask` 的点做 residual（`src/transform_field.cpp:150-173`）；`applyResidual` 用 `candidateMask` 排除 outside（`89-130`）。所以越界点保持 identity，不会报错。
- Dynamic AABB 会在 outside 超过阈值时单调扩张；扩张后 `resampleDomain` 把旧 feature 映射到新域，分辨率不变（`fixed_resolution:1`），因此**有效空间分辨率会随 AABB 变大而变粗**。
- Selector：没有 AABB，越界不是概念；Grid 若用 fixed AABB，需要自己定义 out-of-bounds（zero feature / clamp / skip）。`[VERIFIED/INFERRED]`

### 9.6 对 Grid v0 的含义

- 如果 Grid 复用 TF AABB，会把 Grid 和 TF 域绑定：TF 扩张 → Grid 也要 resample；TF 的 fixed resolution 会限制 Grid 有效分辨率。
- 如果 Grid 用独立 frozen AABB（从 checkpoint xyz 计算一次），则不受 TF 动态扩张影响，但长序列/新区域会 out-of-bounds。
- 如果 Grid 用 world-space hash + 固定 cell size，则不需要 AABB；这是 long-sequence 更自然的选择。`[INFERRED]`

---

## 10. TF 当前真实实现

### 10.1 输入/输出

| 项 | 值 |
|---|---|
| 输入 | `xyz [K,3]`（selected compact 行，TF 前）、`ratio` 标量、`maturity`（仅用于 residual 权重） |
| 输出 | `delta_xyz [K,3]`、`delta_log_scale [K,3]`、`delta_rotation_raw [K,4]` |
| 应用 | `xyz' = xyz + w*delta_xyz`；`scale' = scale * exp(w*delta_log_scale)`；`rotation' = normalize(normalize(identity+delta_rot) * rotation)`，w=maturity |
| 位置 | `src/transform_field.cpp:64-139`；`include/transform_field.h:56-112` |

### 10.2 结构

- `HexPlaneField`：6 个平面 `xy, xz, xe, yz, ye, ze`，每层共享 `feature_dim`，多分辨率 `multires`；坐标是 `[normalized_x, normalized_y, normalized_z, ratio]`；每个 level 的 6 个平面采样结果逐元素相乘，再 concat 各 level（`src/hexplane_field.cpp:153-186`）。
- `feature_out: (feature_dim * len(multires)) -> hidden_dim`
- 3 个 head：`position_hidden/out`、`scale_hidden/out`、`rotation_hidden/out`，每个都是 `Linear+ReLU`（`src/transform_field.cpp:64-87`）。
- `ratio` 不是额外 MLP 输入，而是作为第 4 个坐标进入 `xe/ye/ze` 平面（`hexplane_field.cpp:161-171`）。
- `regularizationLoss`：spatial smoothness + ratio smoothness + ratio L1（`transform_field.cpp:132-139`）。

### 10.3 启动条件

- `TransformField.enable:1`（两个 config）。
- `start_mode: fixed_iteration`；`start_iter=5000`（`selector_config.yaml:74`）或 `3000`（`fast.yaml:74`；mode2 强制 3000，`gaussian_mapper.cpp:513-516`）。
- `initializeTransformFieldIfNeeded` 在每次 LoD 开始时调用（`accumulateLodGradients:3027`），满足 `ready()` 才创建；`ready()` 在 `include/transform_field.h:22-28`。
- 默认 `Elastic.entry_maturity:0`、`TransformField.maturity_compute_epsilon:0`；因此所有 finite Gaussian 都可能是 TF candidate。

### 10.4 哪些 Gaussian 使用 TF？

只有 selected compact 行中：

```cpp
insideMask(selected_input.xyz) &&
maturity >= maturity_compute_epsilon &&
(entry == 0 || maturity >= entry)
```

（`gaussian_mapper.cpp:3117-3123`；`TransformField::candidateMask` `include/transform_field.h:83-87`）。不是所有 Gaussian，也不是 Full 的所有行。`[VERIFIED]`

### 10.5 optimizer

- 初始化时 `transform_optimizer_ = Adam(mlpParameters(), mlp_lr_init)`，再 `add_param_group(gridParameters())`，grid LR（`gaussian_mapper.cpp:2054-2062`）。
- 正常 mapping 用 shared optimizer 的两个 group（`2101-2115`）；`updateTransformLearningRate` 同时更新 `transform_optimizer_` 和 shared groups（`2329-2359`）。
- CUDA backend 时 `tfCudaAdamMixedStep` 对 TF groups 用自定义 CUDA Adam（`src/transform_field_cuda_optimizer.cpp:99-118`）。
- `transform_optimizer.pt` 保存（`4188-4190`），但没有 load 入口。`[VERIFIED]`

### 10.6 Selector 与 TF 的梯度关系

- **前向数据流**：Selector → mask → selected rows → TF → render → photo。
- **反向参数梯度**：photo 对 TF 参数有梯度（通过 selected residual）；photo 对 Selector 参数也有梯度（通过 mask→opacity）；两条路径在 loss 处汇合，但 **Selector 的 p 不进入 TF 的输入**，TF 的输出也不进入 p。因此不存在 `TF -> p -> Selector` 的参数梯度路径。
- 如果 Grid 只拼到 Selector 输入，Grid 的梯度不会经过 TF 参数；它经过 `p -> mask -> opacity -> render -> photo`。`[VERIFIED]`

### 10.7 如果 Selector 加 learnable Grid，Grid 梯度会不会通过 `TF -> render -> photo` 回传？

**不会通过 TF 参数回传。** Grid 的梯度路径是 Selector 的 p → mask → selected opacity → render → photo；TF 只是被选择后的几何变换，是 photo 的另一条参数分支。除非你把 Grid 特征也喂给 TF，否则 Grid 与 TF 权重之间没有 autograd 边。`[VERIFIED]`

### 10.8 TF 自己继续学习，会不会让 Grid capacity test 难以归因？

**会，而且这是主要 confound 之一。**

- TF 和 Grid 在同一个 `L_selector` 中联合训练；TF 可以吸收/补偿 selection 造成的几何误差。
- TF 的 HexPlane 本身就是一个大的 scene-specific spatial field（fast 配置约 657k 参数；selector_config 配置约 1.97M 参数；见下方估算），比 Selector 的 4,962 参数大两个数量级。Grid 的额外空间容量可能被 TF 的持续学习掩盖。
- TF 在 `start_iter` 之后才存在；不同 checkpoint 的 TF 初始化/训练量不同，会改变 LoD 质量。
- 第一轮 capacity audit 应冻结 TF（至少 lr=0 / 不 step / 不更新），并固定 checkpoint 的 TF 参数；否则无法把 same-K 质量差异归因给 Selector Grid。`[VERIFIED/INFERRED]`

TF 参数量估算：

```text
fast: feature_dim=8, multires=[1,2], spatial=64, ratio=150
  per level: 3*(8*64*64) + 3*(8*150*64) = 98,304 + 230,400 = 328,704
  2 levels: 657,408
  MLP: feature_out 16->32 (544) + 3*(32*32+32) (3,168) + 3/3/4 heads (330) ≈ 4,042
  total ≈ 661k

selector_config: feature_dim=16, multires=[1,2,4], hidden=64
  per level: 3*(16*64*64) + 3*(16*150*64) = 196,608 + 460,800 = 657,408
  3 levels: 1,972,224
  MLP ≈ 16k
  total ≈ 1.99M
```

（这些是参数元素数，不是字节；FP32 约 2.6 MB / 8 MB。）`[INFERRED from config + shapes]`

---

## 11. protection / newborn / maturity 类机制

### 11.1 当前真实存在什么？

| 机制 | 是否存在 | 说明 |
|---|---|---|
| continuous maturity | **是** | `GaussianProtection::maturity [N]`，`reset` 为 0，`append` 继承 parent，`prune` 重排；Full commit 后 `observeFull` 更新。 |
| drift EMA / geometric stability | **是** | `drift_ema`，`observeFull` 计算 xyz/scale/rotation drift，`stability=exp(-drift/T)`。 |
| age threshold / newborn protection | **否** | 没有 `newborn` / `age_threshold`；`exist_since_iter_` 只用于 loop closure 的 unstable mask（`gaussian_model.cpp:444-447`）和外部统计。 |
| seen_count threshold | **否** | 没有 seen count；`touched_in_cycle` 是 bool，`last_elastic_supervised_iter_` 是最近一次 LoD supervision。 |
| mature Gaussian / forced keep | **否（作为 admission 规则）** | `compatibility_mature_threshold` 只用于 `getSelectorMatureMask()` / CD / DSD 统计；正常 Selector candidate 不受它限制（`protection_config.h:12` 注释；`gaussian_protection.h:105`）。 |
| budget exclusion | **否** | 没有 Hard/Cold quota；`p=(1-m)r+m s`，m=0 时 p=r，m=1 时 p=s。 |
| protection 强制 keep | **否** | `mixSelectorProbability` 在 m=0 时给 r，不是 1；低 maturity 点是“随机以 ratio 概率进入”，不是 forced keep。 |
| maturity 作为 Selector 输入 | **否** | 只在输出混合中使用，且 `.detach()`。 |

### 11.2 公式

```text
m_i in [0,1]
p_i(r) = (1 - m_i) * r + m_i * s_i(r)
```

（`include/selector_handoff.h:6-13`；`python/elastic_budget.py:23-29`）

- m=0：p=r；Selector 网络输出被完全忽略。
- m=1：p=s；纯 Selector。
- entry>0 时，`elasticSelectorProbability` 只对 eligible 行 forward，ineligible 行保持 p=r（`include/elastic_entry.h:20-30`）；当前 `Elastic.entry_maturity:0`，全部行 eligible。

### 11.3 对 Grid 的影响（重要）

因为 `dp/dθ = m * ds/dθ`：

- **m=0 的 Gaussian 对 Selector/Grid 没有 photometric、ratio、GI 梯度**。新 append 的点 maturity=0，要经过多个 Full cycle 才上升；这些点所在的 Grid cell 可能长期没有学习信号。
- 只有 m>0 的行贡献 Selector 梯度；m 越接近 1，信号越强。
- 如果 Grid cell 里只有低 maturity 点，Grid 基本学不到东西；如果 cell 里混有 mature 点，才能学到。
- ChangeDistill selector loss 不走 `p_mix`，直接对 `forwardProbabilities` 求 L1（`src/change_distill.cpp:158-177`），因此不受 maturity 乘权；但它的 scope 只包含 `mature_mask_before`（`compatibility_mature_threshold=0.8`）的 corrected/all-snapshot mature 行（`src/change_distill.cpp:32-37, 246-248`），**低 maturity 行不进入 CD**。
- TF 的 residual 也在 `applyResidual` 里乘 `maturity`（`src/transform_field.cpp:95-107`）；m=0 的点即使被选为 TF candidate，residual 也是 0。
- 第一轮 capacity audit 若要研究“空间 latent 的表达能力”，应明确选择：
  - 保持 `protection_enabled:1`（真实语义，但梯度被 m 缩放、低 m 点学得慢）；
  - 或 `protection_enabled:0`（`selectionMaturity(false)=1`，p=s，Selector/Grid 获得完整梯度；但改变了 baseline 语义）。
  两者不能混着比较。`[VERIFIED/INFERRED]`

---

## 12. optimizer / checkpoint / serialization

### 12.1 optimizer 构造

| optimizer | 构造 | 正常 mapping 是否 step | 用途 |
|---|---|---|---|
| Gaussian + shared | `GaussianModel::trainingSetup`（`gaussian_model.cpp:499-538`）：`SparseGaussianAdam` 6 组；`ensureSharedMappingOptimizer` 追加 Selector 组、TF MLP 组、TF grid 组（`gaussian_mapper.cpp:2086-2116`） | 是 | Gaussian + Selector + TF 联合更新 |
| Selector 独立 | `GaussianMapper` 构造：`Adam(selector_network_->parameters(), lr=0.001)`（`gaussian_mapper.cpp:167-169`） | **否** | CD 的 `normal_selector_optimizer` 模板；不保存 state |
| TF 独立 | `initializeTransformFieldIfNeeded`：`Adam(mlpParameters, mlp_lr)` + `add_param_group(gridParameters)`（`2054-2062`） | **否** | CD 的 `normal_tf_optimizer` 模板；`transform_optimizer.pt` 保存 |
| CD 事件内 | `independentAdam`（`src/change_distill.cpp:16-29`） | CD 内部 step | 复制参数组，fresh m/v，weight_decay=0 |

注意：`independentAdam` 复制的是 **参数张量本身**，不是 source optimizer 的 state；它 new 一个 `torch::optim::Adam`，所以 CD 每个事件从 fresh moments 开始。`[VERIFIED]`

### 12.2 checkpoint save

`GaussianMapper::savePly`（`src/gaussian_mapper.cpp:4073-4191`）在 `result_dir / "<iter>_shutdown" / "ply"` 下写：

| 文件 | 内容 | 位置 |
|---|---|---|
| `point_cloud/iteration_<iter>/point_cloud.ply` | xyz, normals, f_dc, f_rest, opacity, scale, rot | `gaussian_model.cpp:1136+` |
| `selector_metadata.ply` | maturity, drift_ema, cached_uniform + comments `protection_schema`, `protection_mapping_iteration` | `gaussian_model.cpp:1078-1134` |
| `selector.pt` | `selector_network_->save(OutputArchive)`，包含所有注册 parameters/buffers | `gaussian_mapper.cpp:4177-4183` |
| `transform_field.pt` | TF module save | `4184-4187` |
| `transform_optimizer.pt` | TF Adam state | `4188-4190` |
| `transform_state.json` | TF/AABB/启动/计数等 | `4192+` |
| `protection_state.json` | maturity 统计 | `4164-4175` |
| `mapping_stage_state.json` / `ultrafast_rotation_state.json` | 调度与计数 | `4090-4154` |
| `selector_config.yaml` | 实际配置 | `4361+` |

**没有保存**：Gaussian optimizer state、shared Adam state、Selector optimizer state、`exist_since_iter_`、`last_elastic_supervised_iter_`、Grid（如果未来独立模块未注册）。`[VERIFIED]`

### 12.3 checkpoint load

- C++ `GaussianMapper::loadPly`（`4845-4925`）只 load `point_cloud.ply` + `selector_metadata.ply` + 可选 camera；**不 load `selector.pt` / `transform_field.pt` / optimizer**。它也没有“从 checkpoint 继续 online training”的入口。
- Python `load_selector`（`python/selector.py:60-88`）：`torch.jit.load` 尝试，失败再 `torch.load`；`state_dict`；`load_state_dict(strict=False)`，但随后如果 `missing or unexpected` 就抛错。
- Python `load_transform_field`（`python/transform_field.py:125-151`）：`load_state_dict(strict=True)`。
- Python `run.py` 在 eval 时 load selector/TF/metadata；`gaussians.load_ply` 不 load `exist_since_iter_`（它也没有被保存）。`[VERIFIED]`

### 12.4 如果未来增加 learnable Grid tensor / embedding

| 问题 | 回答 |
|---|---|
| 加入哪个 optimizer？ | 如果 Grid 参数注册在 `GumbelNetworkImpl` 内，`selector_network_->parameters()` 会自动包含；`ensureSharedMappingOptimizer` 会把它加入 shared Gaussian optimizer 的 selector group；`selector_optimizer_` 构造时也会包含。正常 step 走 shared optimizer。若 Grid 是独立 module，需要单独 `add_param_group`，并决定 LR/weight decay。 |
| 独立 optimizer 更自然还是 Selector optimizer？ | 对 v0，**放在 Selector group 最自然**：同一个 LR、同一个 checkpoint、同一个 CD teacher clone。独立 optimizer 会引入第二个 step 语义和额外 state，不利于 A/B。若未来 sparse grid 需要 per-row Adam，则应独立（参考 `SparseFineLattice` / `ResidualField` 的 per-row Adam）。 |
| optimizer state 如何 checkpoint？ | 当前 `selector.pt` **不包含 optimizer state**；shared Adam 也不保存。如果只做 fixed-map eval/短训练，fresh Adam + 固定 checkpoint 可以接受；如果要精确 resume，需要新增 `selector_optimizer.pt` / `shared_optimizer.pt` 并实现 load。 |
| 当前 checkpoint 格式支持新增 module parameter 吗？ | `Module::save` 会序列化所有注册 parameters/buffers，所以**格式层面支持**；但 Python `load_selector` 的 `strict=False + missing/unexpected 抛错` 会拒绝旧 checkpoint（缺 Grid key）。需要版本化或允许 missing Grid 并初始化为零。`selector_config.yaml` 已随 checkpoint 保存，可用来恢复 Grid 配置。 |
| clone checkpoint 做 shared-start A/B 是否容易？ | **eval / fixed-map training 容易**：同一 PLY + metadata + selector.pt + TF。**online training resume 不容易**：没有 Gaussian/shared/selector optimizer state，也没有 KF/topology trace。要做 shared-start online A/B，需要新增 resume + trace。 |

### 12.5 对 Grid v0 的 checkpoint 设计建议

- 把 Grid 注册在 `GumbelNetworkImpl` 内（`register_module` / `register_parameter` / `register_buffer`），这样 `selector.pt` 自动包含。
- 把 Grid 的 `origin`、`cell_size`、`resolution`、`dim` 作为 buffers 或 `selector_config.yaml` 字段保存；不要依赖运行时重新计算（scene AABB 可能在 checkpoint 后变化）。
- 若要保持旧 checkpoint 可用：`load_selector` 允许 missing grid keys，初始化为零；旧 selector 的 `soft_net` 权重保持不变。
- 如果 Grid 用独立 module，必须新增 `grid.pt` 和 save/load 路径，否则 eval 会静默丢失 Grid。

---

## 13. 当前 stochastic state / reproducibility

### 13.1 RNG 来源

| 来源 | 位置 | 是否可控 |
|---|---|---|
| `std::srand(seed)` | `gaussian_mapper.cpp:74` | seed 固定 0（entry 传 0）；只影响 `std::rand()` fallback |
| `torch::manual_seed(seed)` | `gaussian_mapper.cpp:75` | 固定 0；影响 CPU / 默认 generator（CUDA 默认 generator 是否全部 seed 需按 LibTorch 版本确认） |
| `std::random_device rd_` | `include/gaussian_mapper.h:502`；`generateKfidRandomShuffle` `gaussian_mapper.cpp:3559` | **不可控**；KF shuffle 每次进程不同 |
| `std::rand()` | `gaussian_mapper.cpp:3635`（无 shuffle 时的 fallback） | `srand(0)` 可控，但正常路径用 shuffle |
| `cached_uniform = torch::rand(...)` | `gaussian_protection.h:54`（reset）、`:89`（append） | 由 torch RNG 决定；checkpoint 保存后固定 |
| Gumbel `exponential_()` | `gumbel_network.h:173` | 只在 `forward` / `forwardHard` 使用；normal 不用 |
| `ChangeDistillManager::rng_` | `src/change_distill.cpp:196`，`std::mt19937_64 rng_(seed)` | seed=0 可控；用于 shuffle/anchors/nextSeed |
| `ScopedTorchRNG` | `include/model_snapshot.h:10-28` | 保存/恢复 generator state；可临时 set seed |
| AABB resample | `src/change_management.cpp:318-322` | 用 `distill_manager_->nextSeed()` 作为临时 seed；事件序列相同才可复现 |
| `ORB-SLAM3` 内部 RNG/线程 | third_party | 不受 GaussianMapper seed 完全控制；online 轨迹/关键帧可能变化 |

### 13.2 能否做到 Baseline vs Grid 的 same KF / same ratio / same uniform / same camera / same topology？

| 条件 | 能否 | 说明 |
|---|---|---|
| same ratio | **能** | 固定 `UltrafastRotation` 序列；配置固定。 |
| same uniform | **能（固定 checkpoint）** | `cached_uniform` 在 `selector_metadata.ply`；只要不 append/prune，mask 输入相同。但 append 会 `torch::rand` 新行，topology 不同则 uniform 流不同。 |
| same camera | **不能（当前 online）** | `std::random_device rd_` 决定 `kfid_shuffle_`；没有 CLI seed/trace。`tests/ultrafast_mapping_test.cpp:116-117` 只能通过测试子类手动注入 `kfid_shuffle_`。 |
| same KF trace | **不能（当前 online）** | 同上；SLAM 关键帧/回环事件也受多线程影响。 |
| same topology | **不能（当前 online）** | densify/prune 条件依赖 gradient/visibility；Grid 改变 selection 会改变 densify stats，从而改变 topology。没有 topology replay。 |
| same RNG | **部分** | torch seed 固定；但 camera shuffle 用 random_device；topology append 会消耗 torch RNG；AABB resample/CD 用独立 rng。 |

### 13.3 结论

- **fixed-map / fixed-camera / fixed-topology 的离线 Selector capacity 实验可以做到严格 same checkpoint / same uniform / same frames / same ratio**；这是第一轮最干净的路径。
- **online Baseline vs Grid 目前不能保证 same camera/topology**；必须先补 KF trace replay 或禁用 online 变动（fixed map、no densify、no CD/GI），否则差异会混入 data order/topology。
- `torch::manual_seed` 是否能完全控制 CUDA generator 取决于 LibTorch 版本；但 checkpoint 保存的 `cached_uniform` 让 eval 不受影响。`[VERIFIED/INFERRED]`

---

## 14. 现有 evaluation / replay 基础设施

### 14.1 已有

| 工具 | 位置 | 能力 |
|---|---|---|
| `python/eval_lod.py` | `python/eval_lod.py:121-197` | 对 4 个 ratio + Full 批量调用 `run.py`；每个 ratio 独立 output；汇总 `summary.json` / `summary.csv`。 |
| `python/run.py` | `python/run.py:292-343, 472-482, 489-583` | 加载 selector/TF/metadata；`build_selector_mask`；应用 TF；导出 selected PLY + `ply_manifest.json`；按 ratio 渲染；`--evaluation-frame-set` 固定 eval 帧；`--no-save-images`。 |
| `--ply` | `python/run.py:163-164` | 直接渲染 standalone PLY，不经 Selector/TF。 |
| `tests/ultrafast_eval_test.py` | 全文件 | 验证 eval command routing / 4 ratio + Full。 |
| `tests/ultrafast_mapping_test.cpp` | `tests/ultrafast_mapping_test.cpp:24-180` | 用合成 3 Gaussian/5 camera 驱动**生产** `trainMappingIteration`；断言 ratio rotation、Full 不更新 Selector/TF、LoD 更新 Selector/TF、densify stats 不污染。它通过手动设置 `kfid_shuffle_` 绕过 random_device。 |
| `tests/selector_handoff_test.cpp` | 全文件 | 验证 `mixSelectorProbability`、ST hard forward、`p.grad=m`、ratio loss、maturity config。 |
| `tests/ultrafast_rotation_test.cpp` | 全文件 | 验证 1/5/10/15/Full、tail continuation、P95、topology deadline。 |
| `tests/diagnostics/cuda_selector_*` | 当前工作树未跟踪的诊断目录 | 固定 checkpoint / 固定输入 / 固定 uniform 的 legacy-vs-CUDA parity、梯度/Adam replay；不是通用 production replay。 |
| `tests/diagnostics/same_k_swap_20261016` | **只在 `spatial-selector` 分支**，当前工作树没有 | same-K swap 协议/分析；可作为评估设计参考，不能直接跑当前分支。 |
| `tests/diagnostics/spatial_budget_20261016` | 当前工作树存在 | fixed spatial budget / cell coverage / starvation 诊断；不修改生产路径。 |

### 14.2 没有

- 没有 production 的 “load checkpoint → continue online mapping” 入口。
- 没有 KF trace / camera order replay hook。
- 没有 topology event replay。
- 没有 per-row probability / mask / cell-id dump（`selector_stats.json` 只有 aggregate counts：`python/run.py:579-583`；`build_selector_mask` 的 stats 在 `python/selector.py:130-143`）。
- 没有 same-K 强制（actual K 由 cached uniform 决定）。
- 没有 Grid feature / gradient norm / cell occupancy 诊断。

### 14.3 为 clean Grid validation 最少需要补什么 instrumentation

1. **fixed-state Selector train/eval harness**：加载同一 PLY + `selector_metadata.ply` + `selector.pt` + `transform_field.pt`，固定训练/评估相机列表，固定 ratios，固定 `cached_uniform`；只训练 Selector（可冻结 Master/TF），不 append/prune。
2. **per-row dump**：`iteration, ratio, index, xyz, p, mask, cell_id, grid_feature_norm, grid_grad_norm`；至少 CSV/`.npy`。
3. **same-K evaluation**：训练/评估时记录 baseline actual K；Grid 评估用同一 K（例如按 p 排序取 top-K，或 offline subsample），并明确这是 evaluation-only 决策规则。
4. **camera trace**：若必须 online，记录 `kfid_shuffle_` / 实际 `fid_` 序列，并提供 replay hook；或提供 `--kf-order` 文件。
5. **topology trace**：记录 append/prune 的 row 数、survivor mask hash、event iteration；若要 replay，需要完整 survivor mask/append 内容。
6. **Grid diagnostics**：zero/shuffle/per-cell variance/gradient norm（见 §22）。

---

## 15. 性能路径 / timing

### 15.1 已有 instrumentation

| 工具 | 位置 | 覆盖 | 默认 |
|---|---|---|---|
| `MappingProfiler` | `include/mapping_profiler.h` | `Operations`, `FullRender`, `GI`, `FullBackward`, `LOD`, `Densify`, `Optimizer`, `AABB`, `Historical`, `MaturePreparation`, `GITeacher`, `GILoss`, `LodRender`, `LodBackward`, `GaussianOptimizer`, `LodOptimizer`, `FullLoss`, `LodLoss`, `TransformField`, `Total`；host wall + 每 19 cycle 采样 CUDA event | `Mapping.iteration_logging`（`fast.yaml:8` = 0；`selector_config.yaml` 未设置，默认 0）→ `mapping_profiler_.enabled=false`（`gaussian_mapper.cpp:496-501`） |
| `UltrafastTrainingTimer` | `include/ultrafast_training_timer.h` | `wall_ms`, `render_ms`, `cuda_elapsed_ms`；per-iteration log | 仅 mode1 + `mapping_iteration_logging_`；默认关闭 |
| `ProtectionProfiler` | `include/protection_profiler.h` | `SelectorAllRows`, `TFForward`, `TFRegularization`, `TFBackward`, `TFOptimizer`, `MaturityUpdate`, `MaturitySnapshot` | `Protection.performance_logging`（`fast.yaml:30` = 0；`selector_config.yaml:31` = 1） |
| `ContinuousTFStatistics` | `include/continuous_tf_statistics.h` | per-ratio selected/inside/candidate/GI/maturity 统计；`report()` 在 save 时打印 | 由 `performance_logging` 控制 |
| `run.py` render timing | `python/run.py:499, 549, 558-563` | 每帧 `rendering_ms`；`render_time.txt` | Python eval always |
| `UltrafastRotation` counters | `include/ultrafast_rotation.h:17-27` | full/lod render count、selector/tf updates、elapsed mean/P95 | `mapping_iteration_logging_` 控制 elapsed；counters 总是有 |

### 15.2 本机是否能跑 smoke test？

**不能。** 本审计环境：
- `build-cuda118-sm86/` 只有 `CMakeCache.txt` / `CMakeFiles`，没有可执行文件；
- `nvidia-smi` 不存在；
- `python -c "import torch"` 报 `ModuleNotFoundError: No module named 'torch'`。

因此无法报告真实的 `Selector forward ms` / `total LoD step ms` / `N Gaussian`。这是 **UNKNOWN**，不是 0。`[VERIFIED for environment]`

### 15.3 现有计时的 caveat

- `MappingProfiler` 只在 `Mapping.iteration_logging=1` 时启用；`UltrafastTrainingTimer` 会 `cudaDeviceSynchronize()`，显著改变时序；它适合相对比较，不适合“无 overhead” baseline。
- `Protection.performance_logging=1` 会给 Selector all-row / TF 各阶段加 CUDA event；`fast.yaml` 关闭它。
- `UltrafastRotation` 的 `elapsed` 只在 `record_timing` 时保存；`mapping_iteration_logging_` 关闭时 `elapsed` 为空，`mean/p95` 为 null（`gaussian_mapper.cpp:4131-4132`）。
- `python/run.py` 的 render time 是 inference rasterizer，不是 training forward/backward；不能代表 LoD training step。
- 若未来要判断 `[occupied_cells, 8]` sparse latent query 是否值得 CUDA，应先测：
  - `N`（Gaussian 数）、`K`（selected）、cell occupancy；
  - Python/C++ `F(x)` 的 gather/index 时间；
  - selector forward 时间（legacy vs CUDA）；
  - LoD render/backward 时间；
  - 在 `mapping_iteration_logging=1` 下的完整 step 时间。
  当前没有这些数据。

---

## 16. 当前代码是否适合最小 Grid v0？

目标概念（用户描述）：

```text
cell_id_i = spatial_hash(x_i)
F_i = embedding[cell_id_i]
relative_xyz_i = xyz_i - cell_center_i
selector_input_i = concat(current_selector_features_i, F_i, relative_xyz_i)
p_i = Selector(selector_input_i, ratio)
```

Grid 是 learnable、scene-specific、persistent；只接受原有 Selector / LoD loss；不做 maturity/refinement/new loss。

### 16.1 最小需要改哪些文件？

**忠实 concat 版本（legacy 后端，推荐第一轮）**

| 文件 | 改动 | 必要性 |
|---|---|---|
| `improvements/selector/gumbel_network.h` | 新增 Grid module / buffers；`logitsFor` 中 `F=grid(position)`、`rel=position-cell_center`；`soft_net` 输入从 128 改为 `128+D+3`；legacy 分支 concat。 | 必需 |
| 新 `include/selector_spatial_grid.h` + `src/selector_spatial_grid.cpp`（或 header-only） | dense/sparse grid lookup、cell center、参数注册、checkpoint save/load。 | 必需 |
| `python/selector.py` | 镜像 Grid + `soft_net` shape；`load_selector` 构造/加载；保持 C++/Python 权重 key 一致。 | 必需（eval/checkpoint） |
| `src/gaussian_mapper.cpp` | 读 Grid 配置（resolution/dim/origin/cell size）；构造 Selector 时传入；如果 Grid 需要初始 AABB，从 initial xyz 计算。 | 可能需要；若 Grid 默认参数硬编码则不需要 |
| `cfg/selector/fast.yaml` / `selector_config.yaml` | 新增 `Selector.grid_*` 配置；第一轮建议显式 pin。 | 建议 |
| `CMakeLists.txt` | 新源文件 + 测试 target。 | 必需（若新 .cpp） |
| `python/selector_cuda_backend.py` / `src/selector_cuda.*` | 仅当坚持 CUDA backend 时；忠实 concat 需要改 kernel final K=128+D+3、参数顺序/validate。 | 非 v0 必需，若用 legacy 可跳过 |
| `python/run.py` | 若 selector model 构造需要 Grid 配置，load config 传入；否则不需要。 | 可能需要 |
| renderer / Gaussian storage | **不需要**。 | — |

**CUDA 兼容的 zero-init additive 版本（工程最小）**

| 文件 | 改动 |
|---|---|
| `improvements/selector/gumbel_network.h` | 保留 `selector_cuda::network` base 18 参数；新增 `grid_proj: Linear(D+3, 2, bias=False)` 零初始化；`logits = base_logits + grid_proj([F, rel])`。 |
| 新 Grid module | 同上；只被 `grid_proj` 使用。 |
| `python/selector.py` | 镜像 `grid_proj` + Grid；CUDA Python 调用仍只传 base 18 参数。 |
| CUDA kernel | **不需要改**；Grid 参数由 shared optimizer 的 CUDA Adam 直接 step（dense 参数）。 |
| renderer / storage | 不需要。 |

但这个版本只给 logits 加一个**可加的 per-cell bias**，不提供 Grid 特征与 base feature 的非线性交互；它回答的是“空间先验是否有用”，不是“Grid 作为 concat feature 是否有用”。科学上不如忠实 concat。`[INFERRED]`

### 16.2 哪些 tensor shape 会变？

- `positions [M,3]` 不变。
- `cell = floor((positions - origin)/cell_size)`：`[M,3]` int64。
- `F = grid[cell]`：`[M,D]`（dense）或 `[M,D]`（sparse gather）。
- `relative = positions - cell_center`：`[M,3]`。
- `selector_input`：`[M,128+D+3]`（concat 在 128-d embedding 后）。
- `soft_net.weight`：`[2,128+D+3]`；`soft_net.bias`：`[2]`。
- 如果 additive：`grid_proj.weight`：`[2,D+3]`；base `soft_net` 仍是 `[2,128]`。
- Grid table：dense `[Rx,Ry,Rz,D]`；sparse `[K,D]` + keys。
- Checkpoint keys：`grid.*` 或 `grid_proj.*`；`selector.pt` 大小增加。

### 16.3 哪些 CUDA/C++ 接口会受影响？

- `selector_cuda::network`：当前 `validate` 强制 4 输入 / 18 参数 / final `[2,128]`（`src/selector_cuda.cu:77-94`）。忠实 concat 必须改这个接口，或走 legacy。
- `selector_cuda::forward` 的 `spatial` 参数：当前是 **6 个预计算 branch activation**（`src/selector_cuda.cu:237-247`），不是 raw grid feature；不要误把它当 Grid 接口。
- `GumbelNetwork::forwardProbabilities` 的 CUDA `networkMany`：共享 spatial activations；加 Grid 后要么每个 ratio 重算 grid logits，要么扩展 `networkMany`。
- `tfCudaAdamMixedStep` / `tfCudaAdamDirect`：只要 Grid 参数是 contiguous FP32 CUDA tensor，就能 step；不需要改 optimizer kernel（dense）。如果 sparse per-row Adam，需要独立实现。

### 16.4 是否需要改 renderer？

**不需要。** Renderer 只接收 `GaussianRenderInput`（xyz/opacity/scaling/rotation/features）和 camera；Grid 只影响 Selector 的 p/mask，不改变 renderer 输入 schema。`selected_input` 的 TF 变换和 opacity 乘权都在 `accumulateLodGradients` 内完成（`3104-3147`），renderer 的 `render` 调用签名不变（`3174-3176`）。`[VERIFIED]`

### 16.5 是否需要改 Gaussian storage？

**不需要（v0）。** 如果 Grid 按 `xyz` 查询，不需要把 per-Gaussian feature 存进 `GaussianModel`；也不需要 stable ID。若要做 per-Gaussian latent 或 cache F，则要改 storage + topology + checkpoint；不建议 v0。

### 16.6 是否需要改 checkpoint？

**需要（对 Grid 参数）**，但不需要改 Gaussian PLY 格式：

- 把 Grid 注册在 Selector module 内 → `selector.pt` 自动包含；Python 需要同一 module 结构。
- 把 Grid 的 `origin/cell_size/resolution/dim` 存为 buffer 或 config；否则 eval 重建时可能不一致。
- 旧 checkpoint 没有 Grid key：需要 `load_selector` 允许 missing 并零初始化，或做 checkpoint 版本转换。
- 若 Grid 独立 module → 新增 `grid.pt`/save/load；否则 eval 会丢。

### 16.7 是否需要改 optimizer？

- 若 Grid 在 Selector module 内：**不需要新增 optimizer 代码**；`selector_network_->parameters()` 自动进入 shared optimizer 的 selector group 和 `selector_optimizer_`。
- 但 `tfCudaAdamDirect` 会对 selector group 的所有参数 step；dense Grid 参数是 contiguous FP32，可行。若 sparse grid 需要只 step touched rows，则要独立 sparse Adam（参考 `SparseFineLattice`/`ResidualField`）。
- 如果 Grid 用独立 optimizer，需要新增 param group、LR、checkpoint、CD clone 逻辑；不建议 v0。

### 16.8 最小侵入实现路线（建议）

1. **先在 legacy 后端做忠实 concat**：两条 arm 都用 legacy（config 复制并设 `Selector.backend:"legacy"`；不要用 CUDA baseline vs legacy grid）。这是一个 backend-controlled capacity 实验，结论只针对 legacy 数学；不要把它当作 production CUDA baseline。若必须与 production CUDA 对齐，则改用 §16.1 的 CUDA additive，或扩展 CUDA kernel 做忠实 concat。
2. Grid 用 **dense bounded table**（见 §17），固定 AABB/分辨率/D；`F` 用 integer floor + `index`；relative xyz 用 cell center。
3. 把 Grid 注册在 `GumbelNetworkImpl` 内；`soft_net` 输入 `128+D+3`；旧 `soft_net` 权重复制到前 128 列，新增列零初始化；这样 step 0 的 p 与 baseline 完全一致。
4. 同步 `python/selector.py` 结构和 checkpoint key。
5. 配置 pin；在 fixed-map/fixed-camera harness 中训练，冻结 Master/TF/topology，禁用 CD/GI/Guard/Replay。
6. 先做 zero/shuffle/parameter-matched 诊断（§21/§22），再考虑 CUDA kernel 移植。

---

## 17. 三种可能的 Grid 数据结构

| 维度 | A. Dense bounded embedding `[Rx,Ry,Rz,D]` | B. Sparse occupied-cell hash `cell_key -> D` | C. Multi-resolution hash grid |
|---|---|---|---|
| 动态场景增长 | 固定 box；新区域需扩 box/重采样；长序列不友好 | 天然 append 新 cell；适合增长 | 天然 append；多级增长 |
| CUDA query | 整数 floor + 索引；易实现；可用 LibTorch advanced indexing | 需要 hash/table/CUDA kernel；完整 key 比较；可参考 `SparseFineLattice` | 多级 hash + 插值；kernel 复杂 |
| optimizer | dense 参数，标准 Adam；空 cell 无梯度 | per-row sparse Adam 或 dense table + mask；需要 stable row ID | 多级 per-row Adam；状态多 |
| topology | 与 row 顺序无关 | 与 row 顺序无关 | 与 row 顺序无关 |
| loop closure | query 新 xyz；旧 cell latent 不迁移；可接受/可 transport | 同 A；transport 成本更高 | 同 A；多级 transport 更复杂 |
| checkpoint | 一个 tensor；最小 | keys + values + m/v/step；需要版本化 | keys/levels/values/m/v/step；最大 |
| 长序列内存 | `R^3 D` 固定；分辨率随 box 变粗 | `O(occupied cells)`；可控 | 多级 `O(occupied)`；更大 |
| implementation complexity | **最低** | 中 | 高 |
| 当前代码先例 | TF HexPlane 是 dense plane，但不是 per-cell latent | `SparseFineLattice` / `ResidualField`（TF） | 无 |

### 17.1 对“第一轮 capacity audit”的建议

**A. Dense bounded embedding** 最适合当前代码的第一轮 capacity audit，理由：

1. 当前实验可在**单个 room/desk checkpoint、固定 AABB**下进行；dense `16^3` 或 `32^3` 的参数量/显存可控（`16^3*4=16,384`；`32^3*8=262,144`）。
2. 不需要 custom CUDA hash / sparse optimizer / stable ID；一个 `torch::Tensor` + `index` 就能 query，legacy/CUDA 都能用。
3. 与 `spatial-selector` 分支已有的 fixed 32^3 spatial partition（`spacing=max(extent,0.32m)/32`, floor indexing）语义一致；诊断/对照容易复用。
4. 先回答“spatial latent 是否有 signal”；如果 no-go，不需要投入 sparse hash 工程。
5. 如果 go，再把 A 的接口替换为 B；A 的 cell key/AABB 语义可以保留。

**B. Sparse occupied-cell hash** 适合第二轮/生产：

- 长序列、动态增长、无固定 AABB、内存 O(occupied)；
- 当前 `SparseFineLattice`/`ResidualField` 已证明 page/stable row ID/CUDA hash/per-row Adam/checkpoint 的工程可行性；
- 但实现量明显大于 A，且需要 topology/geometry transport 的设计。

**C. Multi-resolution hash** 不建议第一轮：

- TF 已经是一个多分辨率 HexPlane；再加 multi-res grid 会把“Selector 空间容量”和“多尺度表示”混在一起；
- hash collision、level 选择、optimizer/checkpoint 复杂度高；
- 没有当前代码先例可直接复用。

---

## 18. 特别风险：Grid 会不会只记位置？

### 18.1 风险机制

如果 Grid 是 per-cell latent，Selector 又已经吃 raw xyz，网络可能学成：

```text
cell_id -> F -> p
```

即“位置 lookup -> keep probability”，忽略 Gaussian 的 scale/rotation/opacity/SH 和 ratio。特别是：

- 当前 p 是 pointwise MLP；Grid 特征如果维度小、cell 粗，可能变成 cell-level bias；
- `maturity` 混合和 `cached_uniform` 会让 per-cell bias 在训练中产生稳定的 selected set，网络可能利用它记忆训练帧的 coverage，而不是学习可泛化的 selection；
- TF 的 residual 又会补偿几何，进一步让“位置”成为最省力的解释变量。

### 18.2 与当前实现兼容的诊断（不实施）

| 诊断 | 做法 | 期望/判据 |
|---|---|---|
| zero grid | `F=0`（或 `grid_proj=0`） | 性能与 baseline 相同 → Grid 未被使用/无 signal；若下降 → Grid 有贡献 |
| shuffle grid across cells | 固定 permutation π，query `F(π(cell))` | 若性能不掉 → 不是空间结构，只是参数/正则；若掉 → 空间对应重要 |
| shuffle grid across rows | 对每个 Gaussian 随机打乱 F 行 | 若性能不掉 → 不是 per-cell 信息；若掉 → per-row 空间信息重要 |
| freeze selector | 只训练 Grid，Selector MLP 冻结 | 若 Grid 单独能把 same-K 拉起来 → Grid 有独立 capacity；若不能 → 需要与 Selector 共同学习 |
| freeze Grid | 只训练 Selector（Grid 随机/零） | 对照：Grid 的增益是否来自联合适应 |
| coordinate perturbation within cell | 在 cell 内随机移动 xyz，保持 cell id | 若 p 不变 → 只依赖 cell id；若 p 变 → 仍使用 fine xyz/attributes |
| ratio sensitivity | 同一 cell/同一行，ratio 0.01 vs 0.15 | 若 p 的 ratio 斜率消失 → ratio 被 Grid 淹没 |
| cell-to-cell variance | 计算 p 的 between-cell vs within-cell variance | within-cell 方差接近 0 → collapse 到 cell lookup |
| probe cell id from F | 线性/MLP probe `F -> cell_id` | 高准确率说明 F 编码位置；但不等于 collapse，需要结合 shuffle |
| gradient norm ratio | `||∂L/∂grid||` vs `||∂L/∂selector||`，以及 grid grad 的 cell 分布 | grid 梯度集中在少数 cell → 记忆；均匀 → 更泛化 |
| feature norm | 每 cell `||F||`、cell occupancy、选中率 | 空 cell/低 occupancy cell 的 F 可能死掉 |
| same-K swap | 在相同 K 下交换 baseline/grid 的 selected set，比较 render error | 区分“选得更多”与“选得更准” |

---

## 19. 特别风险：Grid 是否只是增加参数量？

### 19.1 混杂

- Grid 参数从 `16k` 到 `262k`，而 Selector 只有 `4,962`。任何质量提升都可能只是容量。
- TF 本身已经是一个大空间场（~0.66M / ~1.99M 参数），Grid 可能只是重复 TF 的空间容量。
- Grid 的 cell 共享会改变有效正则（同 cell 的点共享 latent），这本身可能有益，也可能只是平滑先验。

### 19.2 可用的 control

| control | 参数是否匹配 | 是否保留空间结构 | 评价 |
|---|---|---|---|
| A. 增宽 Selector MLP | 可精确匹配（调 embedding width / hidden） | 否 | **最公平的纯参数 control**；如果 Grid 打不过它，说明增益只是参数 |
| B. global learnable embedding | 只匹配一个向量（通常远小于 Grid） | 否（所有行共享） | 太弱；只能检验 global bias，不适合作为主 control |
| C. non-spatial latent table + shuffled assignment | 可匹配（同样的 table 大小） | 否（随机/打乱 cell 对应） | 保留 per-row/per-cell 容量但破坏空间语义；适合 mechanism control，不适合 pure parameter control |
| D. shuffled-cell grid | 完全匹配（同一 tensor） | 否（空间对应被打乱） | **最公平的空间机制 control**；与 Grid 共享参数、optimizer、初始化 |
| E. zero grid | 完全匹配 | — | 检验“Grid 是否被使用”的下界 |

### 19.3 建议

- **主参数 control：A（增宽 Selector MLP）**。让总参数与 Grid 相同；初始化时保持原 soft_net 行为（新增宽度零初始化/等价）。
- **主空间 control：D（shuffled-cell grid）**。同一 Grid tensor，cell→latent 用固定随机 permutation；参数、optimizer、初始化、训练步数完全相同，只破坏空间对应。
- **可选：C（non-spatial latent table）**。如果担心“per-Gaussian identity”本身就能带来收益，这是重要的补充；但需要 fixed map/row order，topology 变动会破坏。
- **不要只用 B（global embedding）** 作为唯一 control；它太弱，容易把“参数增加”误判为“空间有用”。

### 19.4 参数匹配的做法（分析，不实现）

- 记录 baseline Selector 参数 `P0=4,962`。
- Grid 参数 `Pg = R^3 * D`（dense）或 `K * D`（sparse）。
- 增宽 MLP：把 `kEmbeddingWidth` 从 32 增到 W，或给 `soft_net` 加 hidden；计算新增参数直到 `P0 + Δ ≈ P0 + Pg`。需要保持旧权重可加载（新增列/行零初始化）。
- 若无法精确匹配，至少报告 `ΔP/P0`，并用 D 控制做机制对照；不要把不同参数量的结果直接称为“Grid 更好”。

---

## 20. 后续 clean Grid Capacity Audit 需要哪些控制

### 20.1 能冻结 / 不能冻结

| 项 | 能否冻结 | 当前代码怎么做 | 风险 |
|---|---|---|---|
| Master Gaussian | **能**（fixed-map） | 所有 Gaussian LR=0；不 densify；不 opacity reset；不 Full? | 若仍跑 Full，`observeFull` 会更新 maturity；建议 LoD-only 或冻结 maturity |
| TF | **能** | `TransformField.grid_lr_init/final=0`、`mlp_lr_* = 0`；或 `TransformField.enable:0`（但 baseline 就无 TF） | `updateTransformLearningRate` 在 init/final <=0 时返回 0；`tfCudaAdamDirect` 仍可能 step m/v，但 lr=0 不改参数 |
| topology | **能** | `Optimization.densify_until_iter:0` 或 `densify_from_iter` 大于训练步；`opacity_reset_interval:0` | 需要确认 `topology_window` 条件；`getIteration() < 0` 为 false |
| cameras | **能（离线）** | fixed-map harness 固定 camera list；online 需 KF trace | 当前 online 无 trace |
| KF trace | **不能（当前 online）** | 需新增 replay/`kfid_shuffle_` 注入 | random_device |
| ratio | **能** | 固定 `UltrafastRotation` 或 config ratios | 无 |
| RNG | **部分** | torch seed + fixed uniform；camera 用 random_device | 见 §13 |
| K | **能（eval）** | same-K evaluation / top-K offline；训练仍用 Bernoulli | 不能改训练选择 |
| eval frames | **能** | `--evaluation-frame-set`（`run.py:472-482`） | 必须两 arm 同一文件 |
| GI | **能** | `GlobalGI.mode:"disabled"` + `Selector.gi_loss_weight:0` | 关闭后 baseline 不再是原 GI 语义 |
| ChangeDistill | **能** | `ChangeDistill.enable:0` | 关闭后不再有 CD selector 梯度 |
| Guard/Replay/DSD | **能** | 默认已关；显式 0 | 无 |

### 20.2 冻结后会不会破坏正常 Selector photometric learning？

- **不会破坏“Selector 在固定地图上学习选择”**：固定 Master/TF/topology 后，LoD render 仍然对 selected subset 产生 photometric loss，Selector/Grid 仍能通过 ST mask 获得梯度。这是一个合法的 fixed-state capacity test。
- **但它不再是完整的 online CaRtGS 动力学**：没有新 KF、没有 densify、没有 TF 继续学习、没有 maturity 更新（若 LoD-only）。结论只能回答“在固定场景/固定 TF 下，spatial latent 是否能提升 same-K selection”，不能直接外推到 online。
- **maturity 是最大细节**：如果保持 `protection_enabled:1` 且 LoD-only，maturity 保持 checkpoint 值；低 m 点的梯度被缩小。若设 `protection_enabled:0`，p=s，Selector 获得完整梯度，但 baseline 语义改变。两种都要在报告中明确。
- **TF 冻结**：如果 TF 已初始化，设 lr=0 仍会应用 residual；这保持了几何但不让 TF 吸收 Grid 的误差。若 `TransformField.enable:0`，baseline 就没有 TF，和原 online 不同；不要混用。

### 20.3 推荐的第一轮控制组合

```text
同一 checkpoint:
  PLY + selector_metadata.ply + selector.pt + transform_field.pt
固定:
  camera list (train/heldout), ratio sequence, cached_uniform,
  Master xyz/scale/rot/opacity/SH (no optimizer), topology (no densify),
  TF params (lr=0), maturity (no Full), eval frame set, actual K
关闭:
  GlobalGI (mode disabled, gi_loss_weight=0)
  ChangeDistill (enable=0)
  Guard/Replay/DSD (0)
训练:
  only Selector (+ Grid) on fixed LoD renders
对照:
  Baseline selector / Grid / shuffled-grid / widened-MLP / zero-grid
```

这不是最终实验设计，只是根据源码说明哪些东西可冻结。`[INFERRED]`

---

## 21. 最终必须明确回答的问题

### A. 当前 Selector 是否真的缺乏 persistent scene-specific spatial context？

**是。** 它没有 position→latent、cell feature、embedding table、邻域或外部 scene state；位置只以 raw world xyz 进入共享 pointwise MLP。它可以在共享 θ 中编码粗糙空间先验，但没有可独立更新的 spatial memory。`[VERIFIED]`

### B. 当前 photometric loss 是否能直接训练一个 Selector 输入侧的 learnable spatial latent？

**能。** 只要 `F(x)` 进入 p，且 p 通过 ST mask 乘到 selected opacity，photo loss 就会给 F 梯度；ratio/GI 也会。但梯度被 maturity 乘权，且只在 selected rows 上有 photo 项；m=0 的点没有梯度。`[VERIFIED]`

### C. 加 Grid feature 后，gradient 最终会走到哪里？

- Grid 参数：`photo -> selected opacity -> mask -> p -> F`；`ratio_loss -> p -> F`；`GI loss -> mask -> p -> F`；CD（若启用）-> `p -> F`。
- Master Gaussian：mode1 下还会通过 p 对 xyz/rot/scale 的输入梯度更新；mode2 detach。
- TF：photo 对 TF 参数有梯度，但 Grid 不经过 TF 参数；TF 也不回传 p。
- 不会走到 renderer 代码（无参数）或 Gaussian ID。`[VERIFIED]`

### D. Grid 是否必须修改 renderer？如果不需要，明确说明。

**不需要。** Renderer 的输入 schema 不变；Grid 只改 Selector 的 p/mask。`[VERIFIED]`

### E. Grid 是否必须绑定 Gaussian ID？还是 world-space query 更自然？

**world-space query 更自然。** 当前没有 stable ID，prune 会重排 row，clone/split 只在末尾 append；per-ID Grid 需要额外的 append/prune/clone/split remap 和 checkpoint，且 loop closure 后 ID 与位置脱节。world-space `F(x)` 对 topology 免疫，代价是 geometry correction 后 latent 不随点迁移。`[VERIFIED]`

### F. topology change 对 Grid v0 有什么影响？

- world-space Grid：append/prune 不影响查询；新点自然落到 cell；旧 cell 可保留 dead latent。rank/row 顺序无关。
- per-ID Grid：必须在 `densificationPostfix` / `prunePoints` 增加 hook；clone/split 继承 parent feature；否则 row 重排会错位。
- densify 还会改变 `cached_uniform`（新行 `torch::rand`），从而改变 mask；如果 Grid 改变 selection，会反过来改变 densify stats/topology，形成反馈。第一轮应禁用 topology。`[VERIFIED]`

### G. Loop Closure / Scale Refinement 对 spatial key 有什么影响？

- 两者都修改 world xyz；scale refinement 还改 scale/rotation。点会跨 cell。
- world-space Grid 查询当前 xyz → 使用新 cell；旧 cell 的 latent 不迁移。
- 当前有 `GaussianTransformResult.indices` / `ChangeEvent.correction_groups` 可知道哪些点变了；但没有持久 `transformed_indices`。若要 transport，需要在 `finishGeometryChange` 加 hook，或复用 `ResidualField` 的 transport 设计。`[VERIFIED]`

### H. 当前 AABB 是否适合 fixed grid？

**不适合直接复用 TF AABB。** TF AABB 是动态、单调扩张、fixed-resolution resample 的，并且依赖 maturity/entry；Selector 没有 AABB。第一轮 fixed grid 应用**独立 frozen AABB**（从 checkpoint xyz 计算一次），或直接 world-space hash + 固定 cell size；不要把 Grid 绑到 TF 域。`[VERIFIED/INFERRED]`

### I. 最小 Grid v0 应该用 dense / sparse / multi-res 哪一种？

**dense bounded embedding**（单分辨率、固定 AABB、`16^3` 或 `32^3`、D=4/8）最适合第一轮 capacity audit；**sparse occupied-cell hash** 适合第二轮/生产；**multi-res hash** 暂不建议。理由见 §17。`[INFERRED]`

### J. 如何做 parameter-matched non-spatial control？

- 主 control：**增宽 Selector MLP**，总参数匹配 Grid；新增权重零初始化以保持 step 0 行为。
- 空间机制 control：**shuffled-cell grid**（同一 tensor，固定 permutation）；参数完全匹配。
- 可选：**non-spatial latent table**（per-row/per-code latent，随机/打乱 assignment）；需要 fixed row order。
- 不建议只用 global embedding；它太弱。`[INFERRED]`

### K. 如何检测 Grid 只是死记位置？

见 §18：zero grid、shuffle cells/rows、freeze selector/grid、coordinate perturbation within cell、ratio sensitivity、between/within-cell variance、cell-id probe、gradient norm/cell occupancy、same-K swap。`[INFERRED]`

### L. 如何做 shared-checkpoint Baseline vs Grid，而不让 Master / TF / topology 差异污染结果？

- 同一 PLY + `selector_metadata.ply` + `selector.pt` + `transform_field.pt`。
- 固定 camera list、ratio 顺序、`cached_uniform`、eval frames。
- 冻结 Master/TF/topology；禁用 GI/CD/Guard/Replay；LoD-only（避免 Full 更新 maturity）。
- Grid 与 baseline 的 `soft_net` 共享旧权重；Grid 新路径零初始化，step 0 的 p 完全一致。
- 用 same actual K 评估；记录 actual K。
- 如果必须 online：先补 KF trace + topology trace；否则不要把差异归因给 Grid。`[INFERRED]`

### M. 当前代码里需要改动的最小文件集合是什么？

**忠实 concat + legacy：**
`improvements/selector/gumbel_network.h`、新 `include/selector_spatial_grid.h`（+ `.cpp`）、`python/selector.py`、`cfg/selector/*.yaml`、`CMakeLists.txt`；`src/gaussian_mapper.cpp` 仅在需要传 Grid 配置/AABB 时改。`python/run.py` 仅在需要传配置时改。

**CUDA additive 最小：**
`gumbel_network.h`、新 Grid module、`python/selector.py`、config/CMake；CUDA kernel 不改。

不需要：renderer、Gaussian storage、`GaussianModel`、TF、ChangeDistill、GI。`[INFERRED]`

### N. 如果只为了回答“scene-specific learned spatial feature 能不能提高 same-K Selector quality”，最小实现应该有多复杂？

**一个 fixed-map、fixed-camera、fixed-TF 的 Selector-only 训练/评估 harness + 一个 dense grid + 一个 zero-init concat 分支**即可；不需要 online SLAM、不需要 topology change、不需要 CUDA kernel 改造、不需要新 loss。复杂度主要是：

- 数据/相机/ratio/uniform 固定；
- Grid 查询与 checkpoint/optimizer 接入；
- same-K 评估与 zero/shuffle/parameter-matched controls；
- 训练循环（可以复用 `ultrafast_mapping_test` 的 production `trainMappingIteration` 或写一个最小 fixed-state harness）。

如果直接上 online + CUDA + sparse hash + topology transport，复杂度会高一个数量级，且混杂因素太多，不适合第一轮。`[INFERRED]`

---

## 22. Risks / unknowns

| 风险 | 级别 | 说明 / 缓解 |
|---|---|---|
| 工作树不是 commit | 高 | 先固定工作树 SHA 或 commit baseline；否则后续 `git checkout` 会丢失回退状态。 |
| 误以为“pre-GI” | 高 | 当前仍有 GlobalGI streaming Teacher / GI loss / maturity / CD；不要把它们从 baseline 中误删。 |
| CUDA Selector 固定 18 参数 / K=128 | 高 | 忠实 concat 必须改 kernel；第一轮用 legacy 或 additive。 |
| maturity 梯度缩放 | 高 | m=0 点无 Selector/Grid 梯度；决定 `protection_enabled` 并固定。 |
| TF 继续学习 | 高 | 冻结 TF；否则 capacity 归因不清。 |
| CD 事件 | 中高 | `ChangeDistill.enable:1` 且 epochs 5（fast）；会直接训练 Selector。第一轮关闭。 |
| GI teacher | 中高 | `gi_loss_weight:1.0`；GI label 依赖 valid/observed/maturity。第一轮关闭或固定。 |
| random_device KF | 高 | online A/B 不能 same camera/topology；用 fixed-map 或加 trace。 |
| no stable ID | 中 | world-space Grid；不要 per-ID。 |
| topology / densify | 中高 | 会改变 row/uniform/topology；第一轮禁用。 |
| loop closure / scale refinement | 中 | 点跨 cell；v0 固定 checkpoint/不 online；或接受 query 新 cell。 |
| AABB dynamic | 中 | 不要复用 TF AABB；独立 frozen AABB 或 world hash。 |
| parameter count confound | 高 | 必须做 widened-MLP / shuffled-cell control。 |
| position memorization | 中高 | zero/shuffle/probe/within-cell variance。 |
| checkpoint incompatibility | 中 | 旧 `selector.pt` 缺 Grid key；Python loader 会抛错；做版本化/零初始化。 |
| 没有 GPU/torch/build | — | 本环境无法跑 smoke/timing；所有性能数字 UNKNOWN。 |
| `exist_since_iter_` 不保存 | 中 | 如果未来 Grid/Selector 需要 age，checkpoint resume 会丢；当前 Selector 不用。 |
| `cached_uniform` 新行随机 | 中 | append 改变 mask；fixed map 避免。 |
| `std::rand` fallback / ORB-SLAM RNG | 中 | 多线程/第三方 RNG 不可完全复现。 |

### Unknowns

- 真实 GPU 上的 Selector forward / LoD step ms（本机无环境）。
- `torch::manual_seed` 是否覆盖所有 CUDA generator（取决于 LibTorch 版本；checkpoint uniform 可绕过）。
- 不同 TUM/Replica/VECtor checkpoint 的 maturity 分布/occupancy；没有在当前环境跑。
- `spatial-selector` 分支的 same-K 诊断是否能直接迁移到当前回退源码（它基于 `0268edd` 附近的源码；当前回退后的 `run.py`/mapper 有 ChangeDistill/mode1 tail 差异，需逐文件核对）。
- `lightweight-tf` / `tf-post-optimization` 的 sparse latent 工程能否无冲突地抽到 Selector（它们绑定 TF residual、Guard、CD transport；不能直接复制）。

---

## 23. Recommended minimal implementation surface

### 23.1 推荐 v0（科学优先）

```text
backend: legacy for both arms (do not mix CUDA baseline vs legacy grid)
grid: dense bounded [R,R,R,D], frozen AABB from checkpoint xyz, floor indexing
input: [pos_emd(3->32), rot_emd(4->32), scale_emd(3->32), ratio_emd(1->32)] = 128
       concat [F (D), relative_xyz (3)] -> soft_net(128+D+3 -> 2)
init: copy old soft_net weights into first 128 columns; new columns zero
optimizer: same shared selector group (no new optimizer)
checkpoint: selector.pt includes grid; Python selector mirrors; grid origin/size buffers
```

### 23.2 推荐 v0（工程最小、可保留 CUDA）

```text
backend: CUDA base network unchanged (18 params, K=128)
grid: dense [R,R,R,D] + zero-init grid_proj: Linear(D+3, 2, bias=False)
logits = selector_cuda::network(base) + grid_proj([F, rel])
optimizer: grid_proj params in selector group; CUDA Adam steps them
```

代价：只提供 additive per-cell bias，不是 concat feature interaction。

### 23.3 文件/接口清单

| 文件 | v0 改动 |
|---|---|
| `improvements/selector/gumbel_network.h` | Grid module / zero-init projection；`logitsFor` 查询；`forwardProbabilities` 同步 |
| `include/selector_spatial_grid.h` / `src/selector_spatial_grid.cpp` | dense grid lookup、cell center、参数/buffer、save/load |
| `python/selector.py` | 镜像结构；`load_selector` 兼容旧 checkpoint |
| `cfg/selector/fast.yaml` / `selector_config.yaml` | Grid 配置（resolution/dim/origin/cell size/init） |
| `CMakeLists.txt` | 新源文件 / test |
| `src/gaussian_mapper.cpp` | 只在需要从 initial xyz 计算 frozen AABB/传配置时改；输入构造点不变 |
| `src/selector_cuda.*` | 仅忠实 concat 时需要；additive 不需要 |
| renderer / GaussianModel / TF / CD / GI | **不改** |

### 23.4 测试清单（未来实现时）

1. shape/参数计数：`soft_net` 输入 `128+D+3`；Grid 参数数量；checkpoint round-trip。
2. step 0 parity：baseline vs Grid 的 p/mask 完全相同（zero-init）。
3. gradient：`p.sum().backward()` 时 Grid grad 非零；m=0 行为符合预期；zero grid / zero proj 时 grad 为 0。
4. CUDA/legacy parity（若用 CUDA）：同一输入 p 在容差内一致。
5. same-K：baseline actual K 与 Grid actual K 记录；evaluation top-K 一致。
6. zero/shuffle/parameter-matched controls。
7. checkpoint load 旧 selector.pt（无 Grid）不崩，Grid 零初始化。

---

## 24. 最终回答摘要（给后续设计的一句话版本）

- **当前代码的 Selector 是 per-Gaussian pointwise MLP，没有 persistent spatial latent；它并不缺“位置输入”，缺的是可独立更新、空间共享的 scene-specific 参数。**
- **photometric loss 可以训练输入侧 Grid，但路径是 ST mask→opacity→render，受 maturity 乘权；m=0 点没有梯度。**
- **Grid 不需要改 renderer / Gaussian storage；world-space query 比 Gaussian ID 更自然。**
- **第一轮最小、干净的做法是：fixed checkpoint + fixed cameras + fixed TF/topology + 一致 backend + dense bounded grid + zero-init concat + same-K eval + zero/shuffle/widened-MLP controls。**
- **当前代码里没有 Sparse Learned Feature Grid 的可复用实现；`spatial-selector` 只有 No-Go 文档，`lightweight-tf`/`tf-post-optimization` 的 sparse lattice 属于 TF。**
- **当前工作树不是干净 commit；先固定源码 SHA/commit，再开始 Grid v0。**

---

## Appendix A. 关键源码索引

| 主题 | 文件:行 |
|---|---|
| 训练入口 | `examples/tum_rgbd.cpp:117-121` |
| Mapping 主循环 | `src/gaussian_mapper.cpp:932-1164` |
| 单步训练 | `src/gaussian_mapper.cpp:1253-1361` |
| Full/LoD 分支 | `src/gaussian_mapper.cpp:1363-1800` |
| Selector 输入/ST/GI/TF/loss | `src/gaussian_mapper.cpp:2992-3242` |
| shared optimizer 挂载 | `src/gaussian_mapper.cpp:2086-2116` |
| optimizer step | `src/gaussian_mapper.cpp:2118-2140` |
| TF 初始化 | `src/gaussian_mapper.cpp:2017-2084` |
| 几何变化 | `src/gaussian_mapper.cpp:2361-2468, 3319-3467` |
| checkpoint save | `src/gaussian_mapper.cpp:4073-4191` |
| checkpoint load | `src/gaussian_mapper.cpp:4845-4925` |
| Selector 网络 | `improvements/selector/gumbel_network.h:14-194` |
| Selector CUDA | `src/selector_cuda.cu:77-382` |
| ST / ratio loss | `include/selector_handoff.h:6-25` |
| elastic entry | `include/elastic_entry.h:4-30` |
| maturity | `include/gaussian_protection.h:39-153` |
| Gaussian storage | `include/gaussian_model.h:74-212` |
| prune/densify | `src/gaussian_model.cpp:611-893` |
| geometry correction | `src/gaussian_model.cpp:399-497` |
| metadata save/load | `src/gaussian_model.cpp:1078-1134` |
| TF HexPlane | `src/hexplane_field.cpp:9-236` |
| TF residual | `src/transform_field.cpp:64-185` |
| ChangeDistill selector | `src/change_distill.cpp:158-177, 258-370` |
| GI refresh | `src/global_gi.cpp:216-258` |
| GI teacher | `src/gi_teacher.cpp:5-34` |
| ratio schedule | `include/ultrafast_rotation.h:12-88` |
| Python Selector | `python/selector.py:13-144` |
| Python eval | `python/run.py:292-343, 472-482, 489-583` |
| Python LOD eval | `python/eval_lod.py:121-197` |
| mapping harness | `tests/ultrafast_mapping_test.cpp:24-180` |
| selector handoff test | `tests/selector_handoff_test.cpp:11-32` |
| rotation test | `tests/ultrafast_rotation_test.cpp:10-74` |

## Appendix B. 本次审计没有做的事

- 没有修改 Selector / TF / renderer / Gaussian storage / CUDA kernel / optimizer / config / CMake。
- 没有实现 Grid、maturity、self-refinement、新 loss。
- 没有 commit / push。
- 没有运行正式 TUM / Replica / VECtor 实验。
- 没有跑 smoke test（本机无 torch/CUDA/build）。
- 没有把历史分支的 GI/Oldest/utility 设计映射到当前代码；所有“存在/不存在”的判断均来自当前工作树源码。
