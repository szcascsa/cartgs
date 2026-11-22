# Anchor Selector 新增 Gaussian 属性输入：单次无 SI 训练

## 结果概览

cached_bernoulli：相对原20维无SI，四档LoD的PSNR 4/4 改善（平均+0.6941 dB），SSIM 1/4，LPIPS 4/4。
same_k：相对原20维无SI，四档LoD的PSNR 4/4 改善（平均+0.6347 dB），SSIM 0/4，LPIPS 4/4。

fr1_xyz / seed2026 / 10,000 次 Mapping / 792 个共同图像帧。仅新训练一次；对照为此前相同 seed/预算的 Anchor-20 无 SI，另列 Anchor-20 + SI 作为参考。

输入按顺序为 Anchor feature16、local offset3、activated scale3、SH DC RGB3、activated opacity1、log ratio1，共27维。SH DC 直接取第0阶三通道系数，不转成 RGB；scale/opacity 为 TF 前显式 Gaussian 的 exp/sigmoid 激活。网络27→32→32→2，ReLU，共2018个共享参数（原1794），另有每Anchor16维特征。

保留 joint LoD：新增属性输入不额外detach，Gaussian、Selector/Anchor、TF 都可接收 LoD 梯度；Full 仍只更新 Master Gaussian。Change Distill 使用旧/新属性的 detached 快照，不修改 Gaussian。

SI/GI mode=disabled、SI loss权重0；原photo/ratio/TF loss、LR、h=0.1、Cached Bernoulli、ratio轮转和maturity=1保持。默认新增属性开关为0，实验单独开启1；旧19/20维和新27维Python checkpoint显式识别。

## 生产 Cached Bernoulli 质量

| 档位 | 原20维无SI 点数 / PSNR | 新27维无SI 点数 / PSNR | ΔPSNR | ΔSSIM | ΔLPIPS | 原20维+SI PSNR |
|---|---:|---:|---:|---:|---:|---:|
| Full | 24,043 / 21.9407 | 24,388 / 21.9514 | +0.0108 | -0.00125 | +0.00485 | 21.8718 |
| LOD_1 | 235 / 15.1111 | 245 / 16.4924 | +1.3813 | +0.00793 | -0.02661 | 15.8584 |
| LOD_5 | 1,213 / 17.7323 | 1,256 / 18.6460 | +0.9138 | -0.00372 | -0.03105 | 18.0874 |
| LOD_10 | 2,458 / 19.2102 | 2,527 / 19.5268 | +0.3166 | -0.01475 | -0.03071 | 19.4574 |
| LOD_15 | 3,656 / 19.5061 | 3,683 / 19.6710 | +0.1649 | -0.02248 | -0.02280 | 19.8934 |

Δ=新27维无SI−原20维无SI；PSNR/SSIM正数较好，LPIPS负数较好。

## 相同 K 概率排序诊断

沿用此前固定K：202 / 1010 / 2020 / 3031。该诊断不改变生产 Cached Bernoulli 选择。

| 档位 | 原20维无SI 点数 / PSNR | 新27维无SI 点数 / PSNR | ΔPSNR | ΔSSIM | ΔLPIPS | 原20维+SI PSNR |
|---|---:|---:|---:|---:|---:|---:|
| LOD_1 | 202 / 14.5863 | 202 / 15.2601 | +0.6738 | -0.00051 | -0.02166 | 15.1363 |
| LOD_5 | 1,010 / 17.0377 | 1,010 / 17.9567 | +0.9190 | -0.01551 | -0.03006 | 17.9188 |
| LOD_10 | 2,020 / 18.6581 | 2,020 / 19.2830 | +0.6249 | -0.01155 | -0.03564 | 19.3732 |
| LOD_15 | 3,031 / 19.3187 | 3,031 / 19.6399 | +0.3212 | -0.01831 | -0.02983 | 19.8508 |

## 训练与适用范围

新训练 wall=67.55s；Full renders=4658，LoD renders=8665，online Mapping=6677，tail Mapping=3323。
同一总Mapping预算不意味着相同Full/LoD次数；异步SLAM在线结束时刻、地图规模、Gaussian初始化RNG消耗、相机估计和densification轨迹都可能不同。不是冻结地图因果消融；一次训练不能证明跨seed/scene稳定收益。

native/Python概率及cached mask、输入依赖与梯度、compact source-row、Full/LoD梯度分流、全局/局部几何变化蒸馏、冻结teacher和旧模型加载均通过专项检查。逐视图差异保存mean/median/p10/p90/improved fraction。评价前后checkpoint SHA不变。

本轮没有调LR、扩大网络、加入SI或其他loss，没有stage、commit、push。
