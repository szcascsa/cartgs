# 21D Anchor 分布统计 Selector 实现报告

## 结论

已完成21D Anchor分布统计方案的代码实现、CUDA融合、自动微分和隔离验证。它用固定统计量替代当前50D集合方案中的`10→16→16`成员Encoder及Mean/Max learned pooling，同时保留Anchor内成员之间的分布信息。

当前可通过以下独立配置启用：

```yaml
Selector.anchor_set_encoder: 0
Selector.anchor_set_statistics: 1
CUDA.fused_anchor_input_pack: 1
CUDA.fused_flexgs_selector: 1
```

没有切换生产默认，没有启动完整场景训练。

## 输入与网络

对每个Gaussian构造21D：

`q_xyz(3) + activated_scale(3) + SH_DC(3) + activated_opacity(1) + log_ratio(1) + anchor_mean(3) + symmetric_covariance(6) + eligible_count/10(1)`。

其中`q_xyz=(xyz-anchor_center)/offsetScale`。均值、协方差和Count只使用当前slot-eligible成员；无效成员不参与统计，单成员仍保留自身位置。

评分网络为`21→32→ReLU→32→ReLU→2`，可训练参数为1,826个，全部为场景共享参数；统计分支没有Anchor或Gaussian私有参数。

## 代码改动

- `improvements/selector/anchor_set_encoder.h`、`python/anchor_set_encoder.py`：schema 3数学参考与接口。
- `src/anchor_set_statistics_cuda.cpp/.cu`、`include/anchor_set_statistics_cuda.h`：Forward/Backward统计融合。
- `src/flexgs_selector_cuda.cu`、`include/flexgs_selector_cuda.h`：Selector MLP支持11D和21D模板。
- `improvements/selector/gumbel_network.h`、`python/selector.py`：schema加载、Frozen clone、compact owners和CUDA路径。
- `src/gaussian_mapper.cpp`、`include/gaussian_mapper.h`：配置解析、状态导出和统计开关。
- `cfg/selector/anchor_stats21_residual48.yaml`：独立实验配置。

## 性能结果

RTX3090合成子路径测量如下，F为forward，B为forward+backward：

| N | 原11D融合 | 当前50D集合 | 21D ATen | 21D融合 |
|---:|---:|---:|---:|---:|
| 10,000 | 0.029 / 0.299 ms | 0.411 / 1.498 ms | 0.389 / 1.423 ms | **0.037 / 0.324 ms** |
| 50,000 | 0.110 / 0.335 ms | 0.756 / 1.496 ms | 0.462 / 1.441 ms | **0.140 / 0.386 ms** |
| 100,000 | 0.221 / 0.578 ms | 1.440 / 2.288 ms | 0.845 / 1.477 ms | **0.268 / 0.683 ms** |

21D融合相对50D集合方案减少约3.4–11.1倍forward耗时、3.5–4.7倍forward+backward耗时；相对当前11D融合路径仍增加约20%左右。关闭输入融合后，统计路径会回退到多个ATen小算子，收益明显下降。

## 正确性证据

- 46项Python统计对照通过，最大绝对差`1.91×10^-6`。
- CUDA统计相对参考实现前向最大差`7.45×10^-9`，输入VJP最大差`9.54×10^-7`。
- 21D MLP不同人口的Forward/VJP通过。
- 原11D路径30项逐位回归通过。
- 实际LoD Mapping、全图N预算、Anchor视锥候选、inactive隔离、Maturity私有VJP、Joint Adam和checkpoint/frozen通过。
- 服务器主项目334个受保护文件SHA256未变化，队列仍暂停。

完整证据目录为`research/anchor_statistics21_20261102/`，证据包SHA256：
`a9bae92b7ee3329392ba64ff630d69bc05c17c03a62dd51bf4c34396c4dd2f4f`。

## 当前限制

统计方案使用固定二阶位置统计，表达能力低于learned Mean/Max；协方差只保留6个对称分量，不编码外观分布。当前只完成合成性能和功能验证，尚未证明在线PSNR、选择集合质量、低ratio收益或完整SLAM墙钟时间改善。推荐下一步在同一Replica office2协议下，用当前50D集合方案和21D融合方案各跑一个seed，比较1%、5%、10%、40%质量、实际K、Selector梯度和完整Mapping耗时。
