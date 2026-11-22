# Anchor LoD 与 Legacy-SI 对照

在当前 fr1_xyz 单次运行、8 个共同评价帧上，Anchor 四档 LoD 均弱于 Legacy-SI：PSNR 低 1.80–2.05 dB，SSIM 更低、LPIPS 更高。四档共 32 个视图对中，三项质量指标均由 Legacy-SI 胜出。Full PSNR 也低 0.4079 dB。

本次补齐旧生产 Legacy-SI 的四档 LoD；没有重新训练，也没有修改 Selector、预算或损失。
Anchor 使用最终 joint ratio-conditioning v2；Legacy-SI 指原生产 SI-guided Selector，不是 Raw SI ranking，也不是 PG-ST。

## 实测质量

PSNR、SSIM 越高越好，LPIPS 越低越好；ΔPSNR = Anchor − Legacy-SI。

| 档位 | Anchor 点数 | Legacy-SI 点数 | Anchor PSNR | Legacy PSNR | ΔPSNR | Anchor SSIM | Legacy SSIM | Anchor LPIPS | Legacy LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1% | 284 | 266 | 14.4222 | 16.4752 | -2.0529 | 0.53581 | 0.59649 | 0.71148 | 0.64573 |
| 5% | 1,217 | 1,403 | 17.2258 | 19.0291 | -1.8034 | 0.60602 | 0.65419 | 0.57749 | 0.50546 |
| 10% | 2,493 | 3,067 | 18.1468 | 20.0709 | -1.9241 | 0.63538 | 0.68507 | 0.52128 | 0.43206 |
| 15% | 3,728 | 4,311 | 18.3557 | 20.2679 | -1.9122 | 0.64324 | 0.69426 | 0.49660 | 0.40068 |
| Full | 24,561 | 27,206 | 20.1653 | 20.5732 | -0.4079 | 0.70352 | 0.71485 | 0.36369 | 0.33116 |

## 协议与适用范围

- 场景为 TUM fr1_xyz，两次完整在线运行均使用 792 条 RGB-D association、seed 0、尾优化 20 步。
- 两者使用同一组 8 个时间均匀评价视图：84、175、265、355、446、536、626、717；没有按质量挑视图。
- 共同的是 GT 图像索引；渲染姿态沿用各自运行的估计轨迹，不是同一组冻结相机外参。
- 使用各自冻结 checkpoint、各自 TF 和原有 Cached Bernoulli 导出；实际点数不保证等于 nominal ratio × Master 点数。
- Anchor Master 为 24,561 点、iteration 6828；Legacy-SI Master 为 27,206 点、iteration 5874。在线 Mapping 次数不同，地图不是同一个冻结状态。
- Anchor 成熟度固定为 1、关闭 SI；Legacy 保留原成熟度与 SI 训练。该结果包含整个在线系统差异，不能单独归因于 Anchor MLP。
- 本表是相同 nominal ratio、相同视图对照，不是 exact same-K 或固定地图消融；只覆盖一个场景、一个 seed、8 个视图。
- 1% 档 Anchor 选了更多点（284 对 266），质量仍较低；5%/10%/15% 档 Anchor 点数更少，因此不能把全部差距归因于 Selector 的选点能力。
- 使用既有 evaluator 的 PSNR/SSIM/LPIPS；本次未新增 photo 指标，也未进行可信的独立速度 benchmark。

## 逐视图证据

| 档位 | PSNR 改善视图 | SSIM 改善视图 | LPIPS 改善视图 |
|---|---:|---:|---:|
| 1% | 0/8 | 0/8 | 0/8 |
| 5% | 0/8 | 0/8 | 0/8 |
| 10% | 0/8 | 0/8 | 0/8 |
| 15% | 0/8 | 0/8 | 0/8 |
| Full | 1/8 | 1/8 | 0/8 |


原始逐视图指标、实际点数、paired mean/median/p10/p90 均保存在 `tests/anchor_selector/validation/lod_comparison/`。
Legacy 四档原始 evaluator 输出保存在 `tests/anchor_selector/validation/evidence/quality_legacy/`。
没有 stage、commit 或 push。
