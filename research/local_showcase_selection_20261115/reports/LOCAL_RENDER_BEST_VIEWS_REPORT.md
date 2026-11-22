# Replica／TUM局部渲染最佳视角筛选

覆盖Replica 8个场景×Mono／RGB-D，以及TUM 5个场景×Mono／RGB-D，共26组、49,740个评价视角。每个视角检查1%、5%、10%、30%四档局部渲染，共198,960条历史指标。

## 固定视角筛选规则

每组先按同一视角四档PSNR的算术平均降序排名，LPIPS为并列时的次级排序。随后在Replica RGB-D、Replica Mono、TUM RGB-D、TUM Mono四类中各取最高者。这代表四档整体最优共用视角，不代表该帧在每一个Ratio分别达到全轨迹最大值。

选图使用完整图像；没有裁剪、改亮、锐化或图像修复。每个Ratio复用相同相机、GT关联与冻结模型。图上的PSNR来自保存PNG前的FP32渲染，与原评价相同；PNG为8位展示图。

## 四组数值最优

| 数据集／场景／模式 | RGB索引（0起） | 轨迹索引（0起） | 1% PSNR／K | 5% PSNR／K | 10% PSNR／K | 30% PSNR／K |
|---|---:|---:|---:|---:|---:|---:|
| Replica/office1/mono | 889 | 883 | 29.73／40 | 35.21／200 | 36.68／401 | 40.28／1204 |
| Replica/office1/rgbd | 1805 | 1805 | 29.79／107 | 36.79／537 | 38.92／1074 | 40.89／3223 |
| TUM_RGBD/freiburg2_xyz/mono | 1622 | 1609 | 21.33／148 | 23.79／740 | 25.05／1480 | 26.98／4441 |
| TUM_RGBD/freiburg2_xyz/rgbd | 551 | 551 | 20.42／139 | 23.52／698 | 24.73／1396 | 26.85／4189 |

## 更明亮的Replica备选

Replica最高PSNR视角集中在较暗、低纹理区域，不宜用高PSNR直接推断细节恢复更强。额外保留office4 RGB-D、office0 Mono各自的四档整体最优帧，便于展示选择；它们不是绝对PSNR冠军。

| 数据集／场景／模式 | RGB索引（0起） | 轨迹索引（0起） | 1% PSNR／K | 5% PSNR／K | 10% PSNR／K | 30% PSNR／K |
|---|---:|---:|---:|---:|---:|---:|
| Replica/office0/mono | 547 | 543 | 28.63／30 | 31.54／150 | 33.55／300 | 36.89／900 |
| Replica/office4/rgbd | 825 | 825 | 26.84／48 | 34.46／241 | 37.11／482 | 39.21／1446 |

## 全部26组的最佳共用视角

| 场景／模式 | RGB索引 | 1% PSNR | 5% PSNR | 10% PSNR | 30% PSNR | 平均PSNR |
|---|---:|---:|---:|---:|---:|---:|
| Replica/office1/mono | 889 | 29.73 | 35.21 | 36.68 | 40.28 | 35.48 |
| Replica/office0/mono | 547 | 28.63 | 31.54 | 33.55 | 36.89 | 32.65 |
| Replica/office4/mono | 774 | 26.09 | 29.79 | 32.19 | 37.22 | 31.32 |
| Replica/room2/mono | 1979 | 23.40 | 26.98 | 31.44 | 32.06 | 28.47 |
| Replica/room1/mono | 495 | 22.15 | 27.04 | 28.38 | 32.16 | 27.43 |
| Replica/room0/mono | 92 | 20.51 | 25.71 | 27.43 | 31.99 | 26.41 |
| Replica/office2/mono | 573 | 21.27 | 24.41 | 26.65 | 31.20 | 25.88 |
| Replica/office3/mono | 554 | 20.03 | 23.95 | 25.68 | 30.82 | 25.12 |
| Replica/office1/rgbd | 1805 | 29.79 | 36.79 | 38.92 | 40.89 | 36.60 |
| Replica/office4/rgbd | 825 | 26.84 | 34.46 | 37.11 | 39.21 | 34.40 |
| Replica/office0/rgbd | 287 | 27.48 | 33.26 | 35.66 | 38.37 | 33.69 |
| Replica/room2/rgbd | 1190 | 24.32 | 29.61 | 31.97 | 36.19 | 30.52 |
| Replica/room1/rgbd | 494 | 24.73 | 29.29 | 31.69 | 35.59 | 30.32 |
| Replica/room0/rgbd | 163 | 22.85 | 26.68 | 28.97 | 33.84 | 28.09 |
| Replica/office2/rgbd | 1964 | 19.73 | 27.86 | 31.14 | 32.38 | 27.78 |
| Replica/office3/rgbd | 1720 | 19.96 | 26.58 | 29.04 | 32.41 | 27.00 |
| TUM_RGBD/freiburg2_xyz/mono | 1622 | 21.33 | 23.79 | 25.05 | 26.98 | 24.29 |
| TUM_RGBD/freiburg1_xyz/mono | 727 | 17.69 | 21.81 | 22.92 | 25.24 | 21.91 |
| TUM_RGBD/freiburg3_long_office_household/mono | 116 | 17.54 | 21.11 | 22.09 | 24.45 | 21.30 |
| TUM_RGBD/freiburg1_desk/mono | 310 | 18.43 | 20.60 | 21.92 | 23.30 | 21.06 |
| TUM_RGBD/freiburg1_room/mono | 827 | 10.75 | 16.69 | 19.14 | 23.75 | 17.58 |
| TUM_RGBD/freiburg2_xyz/rgbd | 551 | 20.42 | 23.52 | 24.73 | 26.85 | 23.88 |
| TUM_RGBD/freiburg3_long_office_household/rgbd | 1918 | 19.99 | 23.39 | 24.59 | 26.37 | 23.58 |
| TUM_RGBD/freiburg1_desk/rgbd | 291 | 17.57 | 21.71 | 23.35 | 25.59 | 22.05 |
| TUM_RGBD/freiburg1_xyz/rgbd | 724 | 18.41 | 21.88 | 22.77 | 24.43 | 21.87 |
| TUM_RGBD/freiburg1_room/rgbd | 196 | 17.44 | 19.42 | 21.86 | 25.58 | 21.08 |

## 语义与正确性

- 局部Ratio按当前名义Anchor视锥候选池计算预算，K是实际送入渲染的点数；1%不代表全图Master数量N的1%。
- 双网络真实视角＋LI=0.1，沿用已完成正式批次的checkpoint；没有重训或改变选点／补偿。
- Mono沿用原评价尺度处理，RGB-D沿用原评价处理；每行四档共用同一估计相机视角。
- 共补渲染6组×3个候选视角×4档=72张；选点hash、K以及PSNR／SSIM／LPIPS与历史记录全部相同，最大指标差值为0。
- 所有候选模型与原指标输入的SHA256保持不变；原评价目录和生产配置未修改。
- 此处是明确挑选的最佳展示案例，不是数据集平均表现。Replica两种模式最佳恰好都是office1，TUM两种模式最佳恰好都是freiburg2_xyz，因此四组不等于四个不同场景名称。
- 全部帧排名见ALL_FRAME_RANKING.csv；每组每Ratio各自最优见PER_RATIO_BEST_VIEW.csv；导出指标和实际K见SELECTED_VIEW_METRICS.csv。

## 图片位置

selected/best/<数据集>/<场景>/<模式>/：四组正式最佳视角，各有GT.png及ratio_01／05／10／30.png。
selected/alternatives/：两组Replica明亮备选；images/：每组前三个高质量候选的原尺寸PNG。
