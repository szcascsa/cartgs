# 21D Anchor 分布统计 Selector

日期：2026-11-02。

## 启用与范围

独立配置：`cfg/selector/anchor_stats21_residual48.yaml`。

```yaml
Selector.anchor_set_encoder: 0
Selector.anchor_set_statistics: 1
CUDA.fused_anchor_input_pack: 1
CUDA.fused_flexgs_selector: 1
```

两个集合开关互斥。统计方案复用集合模式入口，以`selector_set_encoder_schema=3`标识；没有可学习成员Encoder。原11D模型、schema 1和schema 2继续按各自原始定义加载，新架构需要重新训练Selector。

Compensator继续使用Residual48；Maturity Loss、1000 Full预热、ratio、停止策略、tail Full→LoD、Anchor名义格子视锥候选及全图N预算沿用所复制的集合配置。本轮没有修改生产默认配置或恢复暂停队列。

## 输入与数学定义

对每个通过Anchor视锥筛选的成员，读取Master上的局部位置：

\[
q_i=(x_i-c_{a(i)})/h_{a(i)}.
\]

`c`和`h`分别为现有Anchor名义格子中心和offsetScale，不使用Gaussian动态包围盒。

成员资格严格沿用现有slot规则：

\[
v_i=\neg(\operatorname{slot\_inactive}_i\land\alpha_i<1/255),
\quad E_a=\{i:a(i)=a,v_i=1\},\quad m_a=|E_a|.
\]

没有slot deactivation时全员eligible。该资格查询detach，不修改slot停用或恢复状态。

\[
\mu_a=\frac1{m_a}\sum_{i\in E_a}q_i,
\qquad
\Sigma_a=\frac1{m_a}\sum_{i\in E_a}(q_i-\mu_a)(q_i-\mu_a)^T.
\]

使用总体协方差，分母为`m_a`，不是`m_a−1`。中心化后求二阶矩，避免`E[qqᵀ]−μμᵀ`的相减抵消。

| 输入位置 | 内容 | 维度 |
|---|---|---:|
| 0–2 | Gaussian名义格子相对位置q | 3 |
| 3–5 | activated Scale | 3 |
| 6–8 | SH DC | 3 |
| 9 | activated Opacity | 1 |
| 10 | 原log(ratio) | 1 |
| 11–13 | eligible成员位置均值μ | 3 |
| 14–19 | 协方差xx、xy、xz、yy、yz、zz | 6 |
| 20 | eligible成员数m/10 | 1 |
| 总计 | 每点固定输入 | **21** |

不可选成员的前10D物理输入置零，不参与统计；均值/协方差/Count均使用同一eligible集合。全无效Anchor的10D上下文为零。最终选点仍经过原slot mask，不会因为统计归零而重新允许这些成员入选。单成员协方差为零，但其q和μ仍保留位置。

## 网络与梯度

\[
z_i=[q_i,s_i,DC_i,\alpha_i,\log r,\mu_a,\operatorname{SymCov}(\Sigma_a),m_a/10],
\qquad \ell_i=MLP_\theta(z_i),\quad21\to32\to32\to2.
\]

中间层均为ReLU；沿用原零初始化输出Head。统计分支没有参数，没有Anchor/Gaussian私有Selector特征。

| 参数 | 数量 |
|---|---:|
| `local_emd.0.weight/bias`：21→32 | 704 |
| `local_emd.2.weight/bias`：32→32 | 1,056 |
| `soft_net.weight/bias`：32→2 | 66 |
| 总计 | **1,826** |

有效成员的物理输入保持Joint梯度。均值和协方差可将同Anchor多个评分的梯度传回有效成员XYZ；没有为提速而detach统计。无效成员的统计输入梯度为零。Maturity继续只对Selector私有参数求VJP，统计分支没有额外可训练参数，不会因加入上下文而将Maturity梯度传到Master或Compensator。

## CUDA实现与生命周期

`src/anchor_set_statistics_cuda.cpp/.cu`提供自定义Autograd：

1. 在当前CUDA Stream上初始化分组计数，用设备端索引将当前候选行分到Anchor；不读取Host计数。
2. 每Anchor一个Warp，最多10个成员。先排序成员行，固定归约顺序；计算eligible数量、均值、中心协方差，直接写最终`N_candidates×21`输入。
3. Backward每Anchor一个Warp，汇聚上下文VJP并写回XYZ、Scale、DC、Opacity梯度；不对Master使用跨Anchor原子累加。

当前每次forward重新建立有界的`A×10`成员表，避免依赖尚未建立的拓扑缓存失效协议。Count、成员表、均值由该Autograd图私有保存，多次forward或两个VJP之间不会覆盖。额外分组与均值保存约96 bytes/Anchor，加上21D输出；不是无界缓存。

Clone/Split/Prune/重排/Anchor迁移后使用当前owners重新分组，slot恢复后使用当前Opacity与slot状态重新判断。Master属性、名义中心或offsetScale更新后下次forward立即读取新值，不复用过期统计。

`CUDA.fused_anchor_input_pack=0`可回退LibTorch统计实现。原FlexGS评分CUDA kernel扩展为11D/21D模板，21D仍保留独立权重；采样、随机数生成与Mask规则不变。`CUDA.fused_flexgs_selector=0`回退原评分MLP。原11D实例保留原运算结构。当前设备/人口Dispatcher不支持时仍走LibTorch路径。

Python使用相同数学定义的PyTorch参考实现，支持保存加载、原eval入口及逐视图Anchor候选。Python尚未绑定新统计CUDA kernel，不把C++加速直接视为Python eval加速。

## 验证与性能

独立验证目录：`research/anchor_statistics21_20261102/`；服务器为`/home/prmi/zwj/experiments/anchor_statistics21_20261102/`。

首批Python检查已通过：手算协方差、FP64 gradcheck、零Encoder参数、1826网络参数、schema 0/1/2/3加载、compact行、拓扑与slot恢复。

## 已完成验证

独立服务器验证已完成，证据位于`research/anchor_statistics21_20261102/`：

| 检查 | 结果 |
|---|---|
| Python数学与生命周期 | 46项对照通过；最大绝对差`1.91×10^-6` |
| 原生CPU/CUDA统计 | 均值、总体协方差、Count、单成员、混合occupancy、全inactive、空输入、VJP和schema/frozen通过 |
| CUDA统计与参考实现 | 前向最大差`7.45×10^-9`，输入VJP最大差`9.54×10^-7` |
| 21D CUDA MLP | 1、1,031、20,000行 forward/VJP 对照通过 |
| 原11D回归 | 30项 forward/共享输出/梯度逐位一致 |
| 实际LoD Mapping | 全图预算、Anchor视锥候选、Maturity私有VJP、Joint Adam、空候选和inactive成员通过 |
| 旧路径回归 | 原集合86项对照、原11D采样/Residual48回归通过 |
| 生产保护 | 334个受保护文件哈希一致；两批队列保持暂停 |

### 合成CUDA微测

RTX3090、FP32、occupancy=4，单位为单次 forward 或 forward+backward 毫秒。该表是 Selector 子路径微测，不是完整Mapping墙钟时间。

| N | 原11D融合 F/B | 当前50D集合 F/B | 21D ATen F/B | 21D融合 F/B |
|---:|---:|---:|---:|---:|
| 10,000 | 0.029 / 0.299 | 0.411 / 1.498 | 0.389 / 1.423 | **0.037 / 0.324** |
| 50,000 | 0.110 / 0.335 | 0.756 / 1.496 | 0.462 / 1.441 | **0.140 / 0.386** |
| 100,000 | 0.221 / 0.578 | 1.440 / 2.288 | 0.845 / 1.477 | **0.268 / 0.683** |

在合成子路径中，21D融合方案相对当前50D集合方案减少约 **3.4–11.1倍 forward**、**3.5–4.7倍 forward+backward**；相对原11D融合路径仍约 **1.2倍** 的开销。21D统计若关闭输入融合，性能改善明显减弱，因此生产启用时应同时保留两个融合开关。

上述结果不代表完整SLAM提速，也没有证明渲染质量提升。尚未进行完整场景训练、PSNR/SSIM/LPIPS或多Seed质量比较；新方案应先以独立配置完成与当前Selector的匹配在线实验。

证据包：`VALIDATION_EVIDENCE.zip`，SHA256为`a9bae92b7ee3329392ba64ff630d69bc05c17c03a62dd51bf4c34396c4dd2f4f`。本轮没有修改服务器主项目、生产默认或暂停队列，没有commit/push。
