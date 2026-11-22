# Anchor 排序分档 Viewer

日期：2026-11-04。新增独立 `Ranked Anchor LoD` 模式，原 Budget、Global ratio、Full 模式保留。训练网络、配置、checkpoint 均不改变。

## 使用

在原生 Viewer 左侧选择 `Ranked Anchor LoD`，再切换：

- `Projected size`：名义格子投影尺寸从大到小。
- `Camera depth`：Anchor 中心在相机坐标系中的 Z，从近到远；不是到相机的欧氏距离。

默认三档 ratio 为 **0.5／0.3／0.1**，高／中／低组的 Gaussian 数量占比为 **20%／30%／50%**。两个参数各自可调，分组占比决定哪些 Anchor 属于哪档，ratio 决定该档保留多少成员。

服务器正常启动方式：

```bash
cd /home/prmi/zwj/realLOD
/home/prmi/anaconda3/envs/zwj/bin/python scripts/view_budget.py \
  results/fr1_xyz_frozen_master_20261103_01/r20_40_60_80/training/13052_shutdown \
  --dataset /home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz
```

沿用当前终端已有的 X11 DISPLAY。原生 Viewer 仍为首选；浏览器备用加 `--viewer browser`，同样支持新模式。可选推理后端仍为 `--inference-backend pytorch|cuda|cuda_graph`，默认 `pytorch`。

## 分组和预算

只考虑当前视锥内的非空 Anchor。名义格子由 `center ± offsetScale/2` 构造，投影8个角点得到屏幕包围矩形：

\[
s_a=\frac{\max(w_a^{pixel},h_a^{pixel})}{H}.
\]

穿过近裁剪面的格子以无限投影尺寸保守排前；屏幕边缘不先裁剪矩形，以免进出画面时尺寸突然缩小。它不代表成员实际 Gaussian footprint，也不扩张名义可见包围体。

排序同分时使用稳定 Anchor UID。按每个 Anchor 成员数累计，用其累计区间中点判定高／中／低档，整个 Anchor 不拆分。因此实际分组比例可能有一个 Anchor 粒度的偏差，并不强制各组都有成员。

同档成员合并为一个候选池，Selector 和 Compensator 使用同一档 ratio：

\[
K_l=\min(\lfloor r_lN_l\rfloor,M_l).
\]

`N_l` 是该组全部成员数，`M_l` 是Slot允许选取的成员数。每档严格K，不额外保点，不跨档借预算。三个结果恢复全局 Gaussian 原行序，合并后只做一次正常渲染，保持跨档深度排序和Alpha合成。

## 对比控件

- A/B/C/D：学习选择或随机选择，分别搭配启用或关闭补偿；同一排序和参数下，四组使用相同分档预算，A/B与C/D分别配对同一组选点。
- `Rank hysteresis`：累计Gaussian分位边界±2个百分点滞回，默认关闭。开启后实际分组比例可能偏离目标更多；参数或排序变化自动重置历史。
- `Reset LoD history`：手动重置档位历史。
- `Color levels`：高档红、中档绿、低档蓝。**PSNR始终来自正常外观图**；着色额外执行一次渲染，耗时单独显示。
- 保留正常RGB PSNR、Full参考、自由相机、数据集相机、实际K和每档Anchor/N/K统计。

当前三档模式不是固定总点数预算；视角变化会改变总K。两种排序因Anchor不拆分，实际K可能相差少量点，不能自动称作严格same-K质量消融。

## 实现位置

- `python/anchor_rank_lod.py`：投影/深度、加权排序分组、三档网络调用与合并。
- `python/budget_viewer.py`：服务接口和正常外观PSNR/着色显示分离。
- `viewer/budget_viewer.cpp`：原生窗口控件。
- `python/budget_viewer.html`：浏览器备用控件。

推理基于冻结Master与Feature；状态发生更新或拓扑改变时沿用预算Renderer的版本检查与rebuild机制。分档历史不会写回模型，不新增训练参数。

独立构建与备份目录：`/home/prmi/zwj/experiments/ranked_anchor_viewer_20261104/`。只更新 `realLOD` 的可选推理/Viewer文件与独立Viewer动态库、二进制；服务器 `/home/prmi/zwj/mycartgs` 不修改。旧文件保存在该目录的 `backup/`。

## 已有验证与限制

在用户要求停止追加测试前，已完成原生Viewer编译、13项新分档单测、真实checkpoint上1,281项等价检查、原15项预算及3项协议回归，并执行浏览器脚本语法/控件请求检查。此后停止追加测试，不进行新的场景训练或质量排名。

两种排序均已实际渲染；未将少量视角观测解释为一种排序全面优于另一种。历史checkpoint训练ratio为0.2／0.4／0.6／0.8时，0.1在训练范围外，界面保留范围提示。用户可以自行比较实际效果。
