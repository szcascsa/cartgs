# Scaffold-inspired Anchor-Conditioned Selector 重构报告

日期：2026-10-20。状态：**架构接入完成，功能与梯度验证通过；重建质量尚未验收。**

本次完整 `fr1_xyz` 接入测试正常结束。固定 8 个共同视图上，新系统 Full PSNR 为 **20.1653 dB**，原生产 legacy 系统为 **20.5732 dB**，差值 **−0.4079 dB**；SSIM 和 LPIPS 同样较弱。因此不能把功能通过表述为质量改进或 production-ready。

未 stage、commit、push。起始及交付 HEAD 均为 `e5d1d57bb1c9e152b24f8deb2faee92d087d57f3`。远程测试使用隔离源码、构建和输出目录，没有替换已有运行目录。

## 1. 已采用的用户后续决定

最初源码审查发现 mode1 实际为 joint LoD，且 Change Distill 已含 Selector teacher-student。用户随后明确决定：

1. 保留 joint LoD：同时更新 Master Gaussian、Selector、Anchor Features 和已就绪 TF。
2. Full Render 只优化 Master Gaussian。
3. mode2 退出可选运行模式；配置值 2 直接拒绝。历史辅助实现仍留存，已移除其测试目标注册。新 Anchor Selector 仅允许 mode1；原 Full-only/legacy 用途的 mode0 保留。
4. 几何事件保留 Selector Change Distill，迁移 Anchor 及成员 Gaussian 的选择关系。
5. 新路径固定 `maturity=1`，关闭动态成熟度/drift 更新。

这些决定替代了最初任务中的 LoD Gaussian detach、禁止既有 Selector distillation 和保留动态成熟度要求。不存在尚待确认的核心接入冲突。

## 2. 本地实现审查与架构对照

审查覆盖 `gaussian_mapper`、`gaussian_model`、Selector、maturity/elastic entry、共享 Adam、TF、Change Distill、几何事件、PLY/Selector 保存及 Python 导出。

原生产网络位于 `improvements/selector/gumbel_network.h`，对应本地 FlexGS 官方参考 `reference/FlexGS-official-20260915/scene/deformation.py:237`：XYZ、rotation、scale、ratio 四路 embedding，再接线性二类头。Raw SI 不直接进入该网络；可选 Streaming GI 通过辅助 loss 指导训练。此前研究目录中的 PG-ST/tiny XYZ MLP 是独立实验，不是这条生产主线。

本地未找到 Scaffold-GS 实现，补充审查了[官方模型源码](https://github.com/city-super/Scaffold-GS/blob/main/scene/gaussian_model.py)。本次借鉴稀疏 Anchor 和共享 feature；显式 Gaussian 与现有 clone/split/prune 继续保留，不采用固定 offset slots、Gaussian 属性生成网络或视角输入。

| 项目 | 原生产路径 | 新 Anchor 路径 |
|---|---|---|
| Selector 输入 | XYZ、rotation、scale、ratio | Anchor feature、local offset、log ratio |
| 重要性指导 | 可选 GI teacher loss | 无 SI/GI 输入、base 或监督 |
| 区域状态 | 无 Anchor latent | 每个 occupied voxel 一个 16 维 feature |
| Gaussian | 显式属性及原 optimizer | 保留相同属性及优化语义 |
| 正常训练/导出选择 | cached Bernoulli | 保留 cached Bernoulli |
| 成熟度 | 动态 Full-to-Full 几何成熟度 | 固定 1 |
| 几何事件 | 原 Selector/TF CD | Anchor 关联迁移及 Selector/TF CD |

## 3. 数据结构和数学定义

新增 `improvements/selector/anchor_bank.h`。默认固定 voxel 边长 `h=0.1`，单位为地图世界坐标单位；原点为零。没有用 GT、SI、C、PG coefficient 或渲染质量确定网格，也没有做 h 搜索。

```text
key_i = floor(XYZ_i / h)
anchor_center_j = (key_j + 0.5) * h
anchor_feature: [M,16]，可学习 Parameter
anchor_centers: [M,3]，buffer
anchor_keys:    [M,3]，int64 buffer
gaussian_rows: [N]，Gaussian -> Anchor 的 int64 buffer
```

Anchor position 和尺度是空间参照，不是第二套可学习几何。Gaussian XYZ 继续由 Master 保存。每次 forward 从当前 XYZ 计算：

```text
offset_i = (XYZ_i - anchor_center[a(i)]) / h
input_i  = concat(feature[a(i)], offset_i, log(r))   # 16+3+1 = 20
```

最终网络：

```text
Linear(20,32) -> ReLU -> Linear(32,32) -> ReLU -> Linear(32,2)
q_i(r) = logit_keep - logit_drop
p_i(r) = sigmoid(q_i(r) / temperature)
```

静态网络共 **1,794 个参数**，另有 `M×16` 个 Anchor feature 参数。最终只产生一个标量选择分数；两个内部 logits 复用现有 softmax/ST 接口。没有 camera、view、distance、SH、opacity、scale、rotation 或 SI 输入。同 Anchor 的 Gaussian 共享 feature，通过不同 local offset 得到不同分数。

生产入口为 `GumbelNetworkImpl::forwardAnchorProbability`，`src/gaussian_mapper.cpp:3133` 调用。新架构使用 LibTorch GPU；旧专用 CUDA kernel 的输入和权重布局不同，因此禁止将 Anchor 网络交给该 kernel。配置中 `architecture` 决定网络结构，`backend=legacy` 是现有 LibTorch 后端枚举。

### 已修复的 ratio 结构问题

最初复用 FlexGS 的 late-fusion 线性头时，数学形式为 `q_i(r)=A(local_i)+B(r)`，Gaussian 相对排序与 ratio 无关。第一次真实快照中，21,522 个 Gaussian 在四个 ratio 下均选中 626 个。

因此改为上述联合输入小 MLP，并使用 `log(r)` 表示预算尺度。依据是结构可分离性与实际选择数量退化，决定发生在读取评估质量之前；没有用 PSNR 选架构、h、lr 或 checkpoint，没有做网络容量搜索。损失、采样器、Adam 和调度保持不变。失败快照及诊断保存在远程 `online-fr1_xyz-v1-additive/`、`validation-v1-additive/`；Python 仍能读取该 19 维输入版本。正式实现为 `anchor_layout_version=2`。

## 4. Ratio 调度、选择和梯度分流

`src/gaussian_mapper.cpp:1423` 保留 mode1：

```text
在线：1% LoD -> 5% LoD -> 10% LoD -> 15% LoD -> Full
尾部：每次 Full 提交后，执行一个轮转 ratio 的 native LoD
```

没有补额外 Full，也没有改变 iteration/LR 时钟、正常步数规则或生产尾优化默认配置。实际数据集测试专用配置将强制尾优化设为 20 步，两条系统对照相同。

LoD 保留原 reconstruction photo、ratio budget loss 和 TF regularizer；没有新增 loss。图像 loss 仍由原 `lambda_dssim`、L1/smooth-L1 和 fast-SSIM 实现确定。新路径 GI loss 为零。几何事件之外不引入 teacher。

- **LoD**：Gaussian 渲染参数保持 joint 梯度；Selector MLP、Anchor feature 及 TF 接收原 opacity-ST/rendering 梯度。local offset 保留到显式 XYZ 的梯度，符合用户要求的 joint 路径。
- **Full**：保留原 Master reconstruction/已有 opacity regularizer；不 forward Selector/TF。以 undefined gradient 清除辅助组梯度，避免 Adam momentum 在 Full 中继续更新辅助参数。
- **拓扑维护**：Full 的 clone/split/prune 可以增删 Anchor 表行，这是组织维护，不是用 Full loss 训练 Anchor feature。

`ensureSharedMappingOptimizer`（`gaussian_mapper.cpp:2146`）继续使用共享 normal Adam。Gaussian 仍是前六组；Selector 的独立组包含 MLP 和 Anchor feature，TF 组继续独立。Gaussian 原有 cat/prune 只处理前六组，Anchor 表通过自己的迁移函数管理。

## 5. SI/GI 与成熟度处理

`Selector.architecture=anchor` 时，配置解析强制关闭 GlobalGI、geometry marginal/hierarchy/shadow/diagnostic 及 GI teacher 权重；不创建 Streaming GI manager，不进行其 Bank refresh 或 contribution render。旧 GI callbacks 仍保留 topology/correspondence version 职责，并接入 Anchor 同步；回调名称不表示新网络依赖 SI。

`ProtectionConfig.fixed_maturity_one` 使初始化、append、metadata load 后 maturity 均为 1；关闭 drift dynamics、Full 成熟度观察及 snapshot transport。Selector、TF 和 CD 不再受动态成熟度门槛影响。cached uniform 和旧 checkpoint 字段保留，便于原选择机制及文件格式复用。真实快照验证 maturity 全 1、drift 全 0。

显式 `legacy` 架构保留历史实现用于 baseline。复现原系统需使用原 SI 配置；仅切换 architecture 不会自动恢复原 GI 参数。本次 control 配置见 `tests/anchor_selector/validation/legacy_selector_config.yaml`，来自起始 HEAD 的 `fast.yaml`，仅覆盖相同的 20 步测试尾预算。

## 6. 在线生命周期与优化器状态

接入点为 `gaussian_mapper.cpp:142` 的现有模型生命周期 callbacks。

| 事件 | 处理 |
|---|---|
| 初始 PCD / reset / PLY load | 根据显式 XYZ 初始化 occupied voxels、零 feature 和逐行关联 |
| append / clone / split | 为新 Gaussian 分配 voxel；已有 voxel 共享 feature，新 voxel 零初始化；不限制 slot 数 |
| prune | 使用同一 survivor 行集合裁剪关联；删除空 Anchor，重映射 ID，并裁剪 feature Adam moments |
| 普通 Gaussian 优化 | 保持稳定关联；每次从最新 XYZ 计算 offset，不缓存独立 offset 几何 |
| loop closure / scale refinement | 在事件结束时重建校正后 voxel 关联，迁移 feature/moments，再运行 CD |
| 显式行置换 | `AnchorBank::selectRows` 支持同一 permutation；已做组件测试 |

不是每个 Mapping step 重建 Scaffold。CPU voxel hash 只在初始化、新增和明确几何事件等维护阶段使用；正常 feature lookup/offset/MLP 在 GPU 上运行。

feature 表改变形状时，不能只修改 `.data`：测试确实捕获过旧 AccumulateGrad 形状残留。最终实现替换 leaf Parameter，同时更新 Module 注册、normal/shared optimizer 引用及 Adam state key；保留既有行 moments，新行 moments 为零。几何重关联按新 cell 内原成员均值迁移 feature 与 moments，沿用原 Adam step。这是明确定义的迁移策略，不声称合并时优化轨迹无损。

本地没有独立的 Master 任意 reorder 入口；当前 append/prune 路径已接入。未来若新增同数量的任意重排，仍须同步调用 Anchor 与原 metadata 的行置换，不能只改 XYZ。

## 7. TF 与 Change Distill

`captureGeometryChange`（`:2421`）冻结完整 Selector，包括 feature、centers、keys、Gaussian 行关联及网络。teacher 使用旧 XYZ/旧关联，student 使用校正后的显式 XYZ/新关联。compact teacher/student 调用显式传递 source-row 索引。

`finishGeometryChange`（`:2466`）在蒸馏前完成 feature/关联/Adam 迁移。`src/change_distill.cpp:249` 对新 Anchor 路径扩大 Selector CD 到：受校正 Gaussian 所在的**旧 Anchor 与新 Anchor 的所有旧成员**。这样一个 Anchor 只有部分 Gaussian 被 loop correction 移动时，未移动的同伴也被约束。新追加、没有 teacher 的 Gaussian 不伪造 target。

沿用现有 probability distillation loss、ratio 列表、event-local Adam、epochs 和固定学习率；不是增加 anchor-feature regression loss。旧 CD 中的“unaffected anchors”是 Gaussian 样本，已排除与扩大后的 primary population 重复的行。

TF 的属性 residual、AABB、regularizer 和几何/CD transport 保留。Selector 不预测 Gaussian 属性。实际 Mapper 测试通过 global scale 和 local correction 下 Selector+TF CD；专门的 partial-anchor 测试确认只移动一个 Gaussian 时，两个旧同 Anchor 成员都进入 Selector CD，teacher 和显式几何保持正确。

## 8. 保存、推理与实际预算

`selector.pt` 保存网络、feature 表及 Anchor buffers；现有 `selector_metadata.ply` 保存逐 Gaussian cached uniform 和固定 maturity。`anchor_selector_state.json` 记录 v2 schema、h、M/N、feature width、ratio encoding 和选择策略。

Python `selector.py` 自动识别新 payload，复用现有 Gaussian PLY、TF、renderer 和导出路径。默认配置已切换为 Anchor。现有 C++ viewer 仍显示 Full；没有额外增加交互 LoD viewer。

训练 forward 与正式导出均为：

```text
hard_i = cached_uniform_i < p_i(r)
训练 backward：hard + (p - p.detach())
```

**没有改为 exact-K，也没有增加校准、refill 或 truncate。** 原 ratio loss 只约束期望数量，不保证 `floor(r*N)`。周期日志可检查实际/期望数量以及与 probability Top-K 的集合重合；Top-K 仅作诊断。

最终真实快照：N=24,561，M=1,393，h=0.1。occupancy min/median/max 为 1/3/532，全部 Anchor 有成员且 feature 非零。

| ratio | 名义 floor(rN) | 期望数量 | 实际数量 | 实际保留比例 |
|---|---:|---:|---:|---:|
| 1% | 245 | 282.7415 | 284 | 1.1563% |
| 5% | 1,228 | 1,237.1663 | 1,217 | 4.9549% |
| 10% | 2,456 | 2,494.2090 | 2,493 | 10.1502% |
| 15% | 3,684 | 3,742.7141 | 3,728 | 15.1785% |

四 ratio native/Python 集合 exact 一致。跨 LibTorch/PyTorch 版本的概率最大误差为 5.3644e-7，使用明确的 1e-6 FP32 概率容差；集合一致性没有放宽。1% 档仍有明显预算偏差，应保留实际点数，不能当 exact same-K。

保存继续是 snapshot/export 机制，不是完整 training resume；原系统没有完整保存/恢复 normal Gaussian+Selector Adam、所有时钟和运行状态，本轮没有声称补齐。

## 9. Correctness Tests

最终编译通过 production `tum_rgbd`、Gaussian/renderer 依赖及指定测试目标。8 组测试全部通过：

| 测试 | 验证 |
|---|---|
| anchor_selector CPU / CUDA | voxel/共享 feature、不同 offset 分数、rotation/scale 输入无影响、ratio 输入、compact 行身份、append/prune/reorder、Adam 迁移、冻结快照、RNG 隔离、archive roundtrip |
| Python parity CPU / CUDA | 四 ratio native/Python 概率与 cached mask；小 fixture 概率最大误差 0 |
| anchor_mapping --parity | 实际 compact renderer、joint Gaussian/Selector/feature/TF 更新、Full 辅助 Adam 冻结、ratio/KF 调度、尾部 pyramid Full→native LoD、拓扑维护、非 finite 分支、global/local CD |
| legacy_mapping --parity | 原生产路径调度与优化器回归 |
| anchor_change_distill | partial-anchor 全成员 CD、frozen teacher 与显式几何不变 |
| legacy_change_distill | 原 CD 行为回归 |

Anchor Mapping fixture 特意将 ratio loss 权重设为零，以隔离真实 photo 经 opacity-ST 到 MLP/feature 的梯度。这只是测试 fixture，不改变正式训练 loss。它证明移除 GI 后 photo 梯度链仍成立，不证明 learned ranking 质量足够。

额外完成真实 24,561 行 checkpoint 的 native/Python 四 ratio 校验。30 个本地/远程源码文件 SHA exact 匹配，记录于 `source_manifest.json` / `source_verification.json`。Renderer 源码和 CUDA rasterizer 算法未修改。

## 10. 完整序列与基本质量回归

在独立目录完成 `fr1_xyz` 全部 792 个 RGB-D association，两条系统均正常退出，尾部测试预算均为 20。旧系统配置来自起始 HEAD。

| 项目 | 新 Anchor | Legacy-SI |
|---|---:|---:|
| 最终 Master Gaussian | 24,561 | 27,206 |
| 在线 normal Mapping | 6,808 | 5,854 |
| 尾部 Mapping | 20 | 20 |
| Full renders（含尾部） | 1,381 | 1,190 |
| LoD renders（含尾部） | 5,467 | 4,704 |
| 程序 wall time | 39.907 s | 40.199 s |
| PyTorch allocator peak allocated | 552.250 MiB | 548.778 MiB |

两条路径均保持 balanced rotation。新路径 TF 按原配置在 iteration 3000 初始化，3001 首次应用。

质量使用现有 evaluator 和同一组按时间均匀选取的 8 个视图，无质量筛选、无 checkpoint 选择。仅在测试进程中将 evo 的 GUI backend 改为 Agg，未修改全局配置或 metric 实现。

| 系统/导出 | Gaussian 数 | PSNR ↑ | SSIM ↑ | LPIPS ↓ |
|---|---:|---:|---:|---:|
| Legacy-SI Full | 27,206 | 20.5732 | 0.71485 | 0.33116 |
| Anchor Full | 24,561 | 20.1653 | 0.70352 | 0.36369 |
| Anchor 1% | 284 | 14.4222 | 0.53581 | 0.71148 |
| Anchor 5% | 1,217 | 17.2258 | 0.60602 | 0.57749 |
| Anchor 10% | 2,493 | 18.1468 | 0.63538 | 0.52128 |
| Anchor 15% | 3,728 | 18.3557 | 0.64324 | 0.49660 |

Full paired PSNR：mean −0.4079 dB、median −0.4296、p10 −0.8132、p90 +0.0251，改善视图 1/8；LPIPS 8/8 更差。这是本次整体替换的负向回归证据。

这不是冻结地图 same-K 对照：两条路径的在线 Mapping 数、最终地图规模不同，且新系统按用户要求同时关闭了成熟度并移除了 GI。不能把差异归因于单个 MLP 设计，也不能将 wall time 接近解释为生产速度提升。

## 11. 性能初测、限制与状态

RTX 3090、FP32、N=20,000 / M=7,307，5 次 warmup 后 20 次平均：Anchor forward+backward **0.6897 ms**；旧 LibTorch 网络 **1.2558 ms**；Anchor 初始化 **5.5603 ms**；feature tensor 467,648 bytes。比较对象是旧 LibTorch 实现，没有据此声称优于旧专用 CUDA 或 SLAM 端到端加速。完整程序 allocator peak 不是全部 GPU 显存占用。

| 范围 | 状态 |
|---|---|
| Anchor、联合 ratio 网络、无 SI/GI、maturity=1、joint mode1、保存/导出 | IMPLEMENTED / TESTED |
| 当前 append/prune、事件迁移、共享 optimizer 状态、Selector/TF CD | IMPLEMENTED / TESTED |
| 一条完整真实序列及四 ratio 推理/渲染 | TESTED |
| 跨 scene/seed 质量与严格同训练预算比较 | PARTIALLY VERIFIED；仅一个 scene/seed、8 个质量视图 |
| 质量替代原系统 | 尚未建立；本次 Full 回归负向 |
| 新 Anchor 专用 CUDA、完整 resume、交互 LoD viewer | NOT IMPLEMENTED |
| 退役 DriftStability EMA 恢复功能与新 Anchor 快照组合 | BLOCKED：新路径显式拒绝该 opt-in 配置，未静默复用不兼容的可变表 EMA |
| adaptive grid/split、Gaussian 生成 MLP、新 loss/regularizer | NOT IMPLEMENTED，按任务边界未加入 |

普通 Full 更新期间关联保持稳定，offset 总是反映最新 XYZ；不保证 Gaussian 永远位于最初 voxel 范围内。大幅几何事件的 score 保真依赖迁移与 CD，不声称合并 Anchor 时精确无损。大场景维护延迟、复杂 loop closure 后质量、更多数据集均需后续验证。

## 12. 交付与后续建议

主要源码：

- `improvements/selector/anchor_bank.h`
- `improvements/selector/gumbel_network.h`
- `src/gaussian_mapper.cpp`、`src/change_distill.cpp`、`src/historical_gradient_render.cpp`
- `include/gaussian_protection.h`、`include/protection_config.h`、`src/gaussian_model.cpp`
- `python/selector.py`、`cfg/selector/fast.yaml`、`cfg/selector/selector_config.yaml`

测试入口：`tests/anchor_selector/`；证据总入口：

- [audit_summary.json](tests/anchor_selector/validation/evidence/validation/audit_summary.json)
- [correctness.json](tests/anchor_selector/validation/evidence/validation/correctness.json)
- [quality_summary.csv](tests/anchor_selector/validation/evidence/validation/quality_summary.csv)
- [paired_full_statistics.json](tests/anchor_selector/validation/evidence/validation/paired_full_statistics.json)
- [真实 checkpoint 校验](tests/anchor_selector/validation/evidence/validation/online_checkpoint.json)
- [源码一致性](tests/anchor_selector/validation/evidence/validation/source_verification.json)

远程完整模型和隔离构建位于 `/home/prmi/zwj/scaffold-selector-refactor-20261020/`。本地证据包不包含大型点云或模型。

下一步建议先用固定 normal Mapping 步数、相同尾预算、多 scene/seed 和完整共同视图，检查 Full 负向回归及低 budget 偏差，再讨论发布或优化。应分别分析 joint Gaussian 更新和新 Selector 的作用；本轮不自动更改 loss、采样器、h、网络容量或训练步数，不继续实现其他算法。
