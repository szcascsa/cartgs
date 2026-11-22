# Anchor × Ratio 14D 潜在特征补偿实验

结论：**本次未超过当前R3**。non-Bank same-K 平均差为 **-0.3858 dB**；训练档 **-0.5272 dB**，插值档 **-0.1972 dB**。14个 seed×ratio 对照中 2 个 PSNR 上升。

## 实现

- 原 Anchor16、27D Selector 输入、32D 共享编码及选择头保持原结构；额外为每个 Anchor 的1/5/10/15%分别保存一张14D潜在特征表。
- 先按 Cached Bernoulli 选点，再为选中K行查询所属 Anchor 的潜在特征；补偿头 `(shared32 + latent14) -> 32 ReLU -> 14`。14维输出仍是 XYZ3、log-scale3、quaternion4、opacity-logit1、SH DC3。
- 未见ratio按原始ratio线性插值潜在特征，再解码；范围外用端点特征。原共享编码中仍保留 log(ratio)。这不是R3直接插值输出系数。
- 新表零初始化；头原有32列、bias和最后层复制，新增输入列保留随机非零初始化。初始零特征不改变原输出；从零补偿头训练时，特征梯度需要末层开始更新后才能传入。
- 四张独立Parameter，未使用ratio表梯度为None。特征及Adam随Anchor增长、prune、重排和闭环成员加权迁移；同层增长沿用逐通道max继承、细层零初始化。
- 配置开关：`Selector.compensation_method: "anchor_ratio_latent14"`。原全局默认R3未改；本轮未stage、commit、push。

## 真实在线训练与评价

fr1_xyz，从头SLAM，两个seed2026/2027。两臂使用同一个源码及二进制，配置仅补偿方法不同。保留当前joint、Full-only Master、原loss、ratio轮转、TF/SI/GI关闭和自然停止；不强制10000步。

当前配置为Variable Anchor1..10，`anchor_only_pruning=0`，Gaussian/Anchor交替拓扑；这与部分旧报告的Anchor-only配置不同。本轮只比较新跑的匹配R3，不拿旧报告质量直接相减。原16D Feature保留其正常joint训练，并非冻结其权重。

服务器直接更新 `/home/prmi/zwj/mycartgs`，复用 `build-performance-20261023-121515` 构建tum_rgbd；未进行隔离构建或运行测试套件。更新前差异源码和主程序已备份至实验目录backup。

| 运行 | 最终Mapping iteration | Gaussian | Anchor | 墙钟秒 |
|---|---:|---:|---:|---:|
| r3_independent14_seed2026 | 6900 | 7469 | 2338 | 44.40 |
| anchor_ratio_latent14_seed2026 | 6400 | 7055 | 2001 | 42.96 |
| r3_independent14_seed2027 | 6425 | 7090 | 2103 | 42.35 |
| anchor_ratio_latent14_seed2027 | 6400 | 7579 | 2316 | 43.39 |

相同64个RGB评价帧；剔除四次运行Bank并集后共 **49** 个non-Bank。PSNR/SSIM/LPIPS/photo均保存逐视图；same-K按`logit(p)-logit(cached_uniform)`排序，K为共同最小population乘ratio并受共同eligible上限约束。Full另评无补偿Master。

## Same-K non-Bank PSNR

| Ratio | K | R3 | Anchor-ratio14 | 平均差dB | seed2026 / 2027差dB |
|---|---:|---:|---:|---:|---:|
| 1% | 70 | 15.9674 | 15.3386 | -0.6289 | -0.7318 / -0.5260 |
| 2.5% | 176 | 16.0452 | 15.9822 | -0.0630 | +0.2392 / -0.3653 |
| 5% | 352 | 18.3713 | 18.0154 | -0.3558 | -0.1676 / -0.5440 |
| 7.5% | 529 | 18.4606 | 18.4546 | -0.0060 | +0.3261 / -0.3381 |
| 10% | 705 | 19.3265 | 18.5031 | -0.8234 | -0.8349 / -0.8119 |
| 12.5% | 881 | 19.1510 | 18.6284 | -0.5226 | -0.3225 / -0.7226 |
| 15% | 1058 | 19.1166 | 18.8159 | -0.3007 | -0.2085 / -0.3928 |

| 范围 | Cached差PSNR | same-K差PSNR | same-K差SSIM | same-K差LPIPS | same-K差photo |
|---|---:|---:|---:|---:|---:|
| train | -0.5219 | -0.5272 | -0.01882 | +0.01252 | +0.00984 |
| unseen | -0.1391 | -0.1972 | -0.01988 | +0.00250 | +0.00769 |
| all | -0.3578 | -0.3858 | -0.01927 | +0.00823 | +0.00892 |

Full non-Bank PSNR差：+0.1854 dB。SSIM/PSNR越高越好，LPIPS/photo越低越好。

## 参数与局限

每Anchor额外56个FP32参数；头增加448个权重。私有特征参数量为56M，对比R3为56N；因多个Gaussian共享同Anchor，新方案通常更省特征参数，但解码头可达的个体补偿仍受共享与非线性限制。表的参数+梯度+Adam一二阶状态估算为896M bytes，不等同于峰值显存。

本次两seed的R3表参数分别418,264 / 397,040，新方案112,056 / 129,696，减少约73.2% / 67.3%（不含头）。新增特征表均finite且RMS非零；每张表Adam实际步数为：R3的780 / 685，新方案680 / 680。自然停止及在线拓扑差异没有通过伪造optimizer步数补齐；这些优化曝光差异必须保留在解释中。

在线地图、位姿、实际更新步数可能分叉；same-K控制点数，不能消除地图差异，也不能独立证明潜在特征参数化的因果收益。两个seed来自同一场景，并非跨场景验证。服务器有其他GPU作业，本轮墙钟不能作为独占性能benchmark。

生命周期和原生checkpoint加载代码已接入，但按用户要求未跑对应单元/回归测试。实际完成在线增长、优化、保存和Python载入渲染；不能据此宣称所有闭环边界都已覆盖。三个插值ratio不构成对所有连续ratio的质量保证。

## 证据

- `tests/diagnostics/anchor_ratio_latent14_20261026/aggregate_comparison.csv`
- `evidence/validation/quality_summary.csv`、`quality_per_view.csv`、`paired_statistics.csv`、`selection_statistics.csv`
- `evidence/validation/training_runtime.json`、`evaluation_plan.json`、`preregistration.json`
- `source_manifest.json`、`evidence/sync_receipt.json`、`evidence/validation/build.log`
- 服务器完整模型：`/home/prmi/zwj/anchor-ratio-latent14-20261026/`各方法_seed目录。
