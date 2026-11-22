# 明亮视角替换版

Replica RGB-D改为office4第825帧，Replica Mono改为office4第774帧；TUM两组保持原视角。索引从0计数。这是按明亮画面重新选择的展示案例，不再称四类绝对PSNR最优。图片直接复用或补渲染冻结模型，未调亮或改变像素。

| 场景／模式 | RGB帧 | 1% PSNR | 5% PSNR | 10% PSNR | 30% PSNR |
|---|---:|---:|---:|---:|---:|
| Replica/office4/rgbd | 825 | 26.84 | 34.46 | 37.11 | 39.21 |
| Replica/office4/mono | 774 | 26.09 | 29.79 | 32.19 | 37.22 |
| TUM_RGBD/freiburg2_xyz/rgbd | 551 | 20.42 | 23.52 | 24.73 | 26.85 |
| TUM_RGBD/freiburg2_xyz/mono | 1622 | 21.33 | 23.79 | 25.05 | 26.98 |

每组selected/目录包含GT.png及ratio_01／05／10／30.png原尺寸图片；PSNR来自保存PNG前的FP32渲染。指标、实际K和选点hash与原评价完全一致，详见BRIGHT_SELECTION.json和SELECTED_VIEW_METRICS.csv。

Replica两组GT的平均显示亮度分别为0.497、0.528（0—1，BT.709加权RGB）；这是原图统计，不是调色。
