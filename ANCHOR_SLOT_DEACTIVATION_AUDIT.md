# Anchor slot 可逆停用：验证、实现与一次实际训练

**结论：本次重新训练后，5%／10%／15% LoD 改善，1% 与 Full 退化；不能判定全面有效。** 功能已实现并通过正确性测试，默认关闭。没有 stage、commit 或 push。

## 1. 本次实际做了什么

先在旧冻结模型验证，再按用户追加授权，实现停用并进行一次 fr1_xyz、seed2026、10,000 Mapping 训练。基线为已有的 Octree-GS 原版 Anchor opacity pruning 训练；新训练从头运行，输入配置只追加 `Selector.slot_deactivation: 1`。没有从评价质量选择 LR、阈值、seed 或 checkpoint。

保留 27 维 Selector、十点 Anchor、原 Cached Bernoulli、原 ratio 轮转、joint／Full、TF、Change Distill 和已有损失。未增加补点算法、蒸馏或 loss。

冻结阶段：仅移除符合资格的点，在四 ratio × 64 视图上，最大渲染图像差 **0**。但用保存的 Cached Uniform、统一 logit 平移得到同 K 时，10%／15% PSNR 分别下降 0.299／2.032 dB，原完整 gate 失败。随后用户明确要求实际训练，因此没有把冻结 gate 改成“通过”，也没有据此调参。

## 2. 停用规则和边界

- Anchor 删除公式不变：只有 `T > 0.8 × check_interval` 且 `O < 0.005 × T`，才整组删除。
- slot 在充分闭合窗口的每次 Anchor 可见访问都低于 `1/255`，且当前仍低于截断，才停用；观察不足、未观察和短暂低 opacity 不直接停用。
- 停用只使 LoD 有效选择概率为零。十个显式 Gaussian、UID、Anchor Feature、Gaussian 和 Adam 状态都保留；Full 仍使用完整地图。
- 当前 opacity 恢复到 `≥1/255` 时解除停用，并标记窗口已恢复；旧窗口证据不能立即再次停用，须经过新的完整资格窗口。恢复是代码能力，不表示低 opacity 点必然能在本次训练中自行恢复。
- `p_eff = eligibility × p`，仍用原 `cached_uniform < p_eff`。原预算损失读取 `p_eff`，目标仍为 `r × N_total`，不改成剩余候选数，也没有强制 exact-K。
- 状态是两个 `[Anchor,10]` bool buffer，随创建补零、整 Anchor pruning 压缩、UID 几何迁移、teacher snapshot 和 checkpoint 保存。原始 MLP 概率与 Change Distill 目标保持原定义。

代码开关：

```yaml
Selector.slot_deactivation: 1
```

只允许显式十点 Anchor 路径使用；缺省为 0。新 checkpoint 自带停用状态，Python 推理／导出按保存的资格执行。旧 checkpoint 没有这些 buffer 时保持旧行为。

## 3. 正式 Cached Bernoulli 质量

64 个预先锁定的共同时间均匀视图。Δ = 停用训练 − 基线；PSNR／SSIM 越高越好，LPIPS／photo 越低越好。photo 使用原生 `0.8 L1 + 0.2 (1-fast_ssim)`，不是用评价 SSIM 替代。

| 档位 | 基线点数 | 停用点数 | 基线 PSNR | 停用 PSNR | ΔPSNR | ΔSSIM | ΔLPIPS | Δphoto |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1% | 438 | 447 | 19.1333 | 18.8471 | -0.2862 | -0.00528 | -0.00159 | +0.002676 |
| 5% | 2219 | 2135 | 20.4513 | 20.6425 | +0.1912 | +0.00683 | -0.01858 | -0.003015 |
| 10% | 4257 | 4216 | 20.5622 | 21.0699 | +0.5077 | +0.01379 | -0.02755 | -0.005817 |
| 15% | 6138 | 6157 | 20.5995 | 21.1361 | +0.5367 | +0.01475 | -0.02822 | -0.006161 |
| Full | 41730 | 42640 | 22.1548 | 22.0451 | -0.1097 | -0.00253 | +0.00227 | +0.000583 |

## 4. 相同绝对 K 质量

共同 K 在正式渲染前锁定为 `floor(r × min(N_baseline,N_enabled))`。使用保存 Uniform 的 `logit(p_eff)-logit(u)` 顺序取共同 K，按 source row 打破 tie。这只是固定随机数下的同预算诊断，未用于训练或生产导出。

| 档位 | 双方相同 K | 基线 PSNR | 停用 PSNR | ΔPSNR | ΔSSIM | ΔLPIPS | Δphoto | PSNR 改善视图 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 1% | 417 | 18.9932 | 18.6653 | -0.3278 | -0.00806 | +0.00794 | +0.003553 | 17/64 |
| 5% | 2086 | 20.4192 | 20.6355 | +0.2163 | +0.00805 | -0.01987 | -0.003577 | 37/64 |
| 10% | 4173 | 20.5580 | 21.0699 | +0.5119 | +0.01407 | -0.02809 | -0.005902 | 46/64 |
| 15% | 6259 | 20.5995 | 21.1290 | +0.5295 | +0.01485 | -0.02860 | -0.006148 | 48/64 |

5%／10%／15% 在同 K 下四指标一起改善，说明本次中高比例收益不依赖多选点。1% 同 K 四指标均退化，Full 四指标也略退化，不能只报告有利档位。

## 5. 预算利用与地图规模

| ratio | 基线已选截断以下点 | 占预算 | 停用训练已选截断以下点 | 占预算 |
|---|---:|---:|---:|---:|
| 1% | 2 | 0.46% | 0 | 0.00% |
| 5% | 727 | 32.76% | 241 | 11.29% |
| 10% | 2021 | 47.47% | 873 | 20.71% |
| 15% | 3515 | 57.27% | 1752 | 28.46% |

新模型最终 4264 Anchor / 42640 Gaussian，层级 L0／L1／L2 为 1106／2642／516；基线是 4173 Anchor / 41730 Gaussian，新模型存储点数多 910（2.18%）。创建 7203 Anchor，原版公式整组删除 2939。

最终停用 20739 个 slot（48.64% population）。当前低于截断共 29272 个；其中 8533 个没有被长期窗口规则停用，仍可成为候选。因此预算浪费明显减少，但没有消除。首次非零停用出现在 Mapping 5000，后期形成主要作用。

新增常驻状态 85280 字节（约 83.3 KiB），不进入 Adam。单次训练 wall time：基线 73.25 s，停用 73.63 s；异步轨迹和运行时机不同，不能将此差值认定为可信速度开销。未完成独立性能 benchmark。

## 6. 验证、结论与适用范围

**TESTED**：15 项 CPU／CUDA、窗口边界、暂时低 opacity、恢复后新证据、预算分母、被屏蔽梯度、十点增删、几何迁移、teacher／checkpoint、旧模型兼容、production Mapping joint／Full、Change Distill 回归均通过。最终真实 checkpoint 四 ratio 的 native mask、Python mask 和导出 selected rows exact 相同，停用点被选数为 0。

测试中曾因夹具在拓扑修改后未按生产边界关闭 autograd、且缺少真实 Adam 状态而发生段错误。已仅修复夹具，使其在 NoGrad 边界执行拓扑修改并建立 Adam 状态；未据此改变生产几何算法。

**PARTIALLY VERIFIED**：一次真实训练支持“停用后联合训练能适应中高比例预算重分配”，不支持“所有 ratio 或 Full 都改善”。1% 原来仅两个已选点低于截断，可释放预算本就很少；这与收益主要出现在较高 ratio 的现象一致，但不足以单独解释其退化原因。

**未证明**：多场景／多 seed 稳定收益、严格 held-out 泛化、同一冻结地图上的训练因果效应、无干扰性能和峰值显存对照。两次在线 SLAM 虽同 seed／配置／10,000 步，但关键帧、估计外参、地图增长／删除与训练采样可能不同；本次 64 视图未严格拆分 Bank／non-Bank。same-K 控制实际 LoD 点数，不消除这些系统差异。

建议保留默认关闭的实验开关。本轮没有继续调窗口、改阈值或增加新的候选规则；1% 和 Full 的退化应先作为反证保留。

## 7. 证据与复现位置

服务器实验目录：`/home/prmi/zwj/anchor-slot-deactivation-20261021`；基线、原源码和 server production 的 SHA 均核对未变。训练与评估权重分离，评价期间模型 SHA 未变。

本地证据：`tests/anchor_selector/slot_deactivation_validation/`。

- `evidence/frozen/`：原冻结 gate、逐视图、移除前后图像误差和相同 K 结果。
- `online_evidence/validation/quality_summary.csv`、`quality_per_view.csv`、`paired_statistics.csv`：四指标及 paired mean／median／p10／p90／改善比例。
- `online_evidence/validation/correctness.json`、`online_diagnostics.json`、`activity_windows.csv`：功能门控、最终状态与停用动态。
- `online_evidence/validation/training_protocol.json`、`evaluation_plan.json`：锁定配置、共同视图、共同 K 和模型 SHA。
- `online_evidence/enabled_seed2026/10000_shutdown/ply/`：最终 Selector／TF、Master PLY、metadata、停用及窗口状态。
- `source_manifest.json`、`local_delivery_checks.json`、`online_evidence/online_delivery_verification.json`：16 个本地／远程源码路径与 296 个原始交付文件 SHA；1152 行质量指标独立复算通过。

主要实现：`improvements/selector/anchor_bank.h`、`explicit_anchor_blocks.h`、`src/gaussian_model.cpp`、`src/gaussian_mapper.cpp`、`src/historical_gradient_render.cpp`、`python/selector.py`。原网络结构与原重建损失不变。

最终分类：**中高比例 PROMISING；总体 MIXED，默认不开启。**
