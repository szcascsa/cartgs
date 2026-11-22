# 各场景结果图

每个数据集、场景、相机模式单独一张图，共32张，包含CaRtGS Vanilla Full及当前双网络真实视角＋LI方案的Full、全局LoD、局部视角LoD。

统一采用CaRtGS未校正原图评价口径。VECtor三个双目场景的69条结果重新渲染评价；其余667条复用已完成的同口径评价。没有重新训练或保存渲染图片。

训练Ratio：0.01、0.05、0.1、0.3、0.5、0.7；未见Ratio：0.03、0.08、0.2、0.4、0.6。图中空心点和表中*表示未见Ratio。

横轴按实际Ratio数值使用线性坐标：0.01至0.1的距离为0.1至0.2距离的90%。低Ratio刻度倾斜以避免重叠，不对横轴做非线性拉伸，所有指标保持原值。

Full点数为地图总数N，全局LoD为实际选中K，局部LoD为跨视角平均K；CSV另有最小／最大K。这些数值不是Alpha合成可见点数，不能当作统一same-K比较。

CaRtGS按每个场景及相机模式Full全轨迹PSNR最高的一次重复取值，SSIM／LPIPS／N绑定同次。corridors-dolly双目原版只有两次完成，其余组为三次。

图中的CaRtGS原版来自vanilla_all_datasets_20261031_2011归档实验，指当前项目关闭扩展模块后的Vanilla算法路径；来源checkpoint及评价目录逐行记录在CSV，不将其标为未修改上游仓库重新构建的结果。

评价使用各自完整估计轨迹，未对共同RGB帧重聚合。图中明确展示两方评价帧数；相同帧数也不证明轨迹及RGB关联相同。只作各自全轨迹成绩展示，不作为共同帧配对显著性检验。

当前训练模型的Full起点与选模记录保持不变。双目原图评价会包含训练校正图像与未校正评价图像之间的域差异，不将其解释为训练退化。

| 数据集／场景／模式 | 结果图 | 矢量图 | 精确数值 |
|---|---|---|---|
| Replica/office0/mono | [PNG](figures\Replica\Replica__office0__mono.png) | [SVG](figures\Replica\Replica__office0__mono.svg) | [CSV](figures\Replica\Replica__office0__mono.csv) |
| Replica/office0/rgbd | [PNG](figures\Replica\Replica__office0__rgbd.png) | [SVG](figures\Replica\Replica__office0__rgbd.svg) | [CSV](figures\Replica\Replica__office0__rgbd.csv) |
| Replica/office1/mono | [PNG](figures\Replica\Replica__office1__mono.png) | [SVG](figures\Replica\Replica__office1__mono.svg) | [CSV](figures\Replica\Replica__office1__mono.csv) |
| Replica/office1/rgbd | [PNG](figures\Replica\Replica__office1__rgbd.png) | [SVG](figures\Replica\Replica__office1__rgbd.svg) | [CSV](figures\Replica\Replica__office1__rgbd.csv) |
| Replica/office2/mono | [PNG](figures\Replica\Replica__office2__mono.png) | [SVG](figures\Replica\Replica__office2__mono.svg) | [CSV](figures\Replica\Replica__office2__mono.csv) |
| Replica/office2/rgbd | [PNG](figures\Replica\Replica__office2__rgbd.png) | [SVG](figures\Replica\Replica__office2__rgbd.svg) | [CSV](figures\Replica\Replica__office2__rgbd.csv) |
| Replica/office3/mono | [PNG](figures\Replica\Replica__office3__mono.png) | [SVG](figures\Replica\Replica__office3__mono.svg) | [CSV](figures\Replica\Replica__office3__mono.csv) |
| Replica/office3/rgbd | [PNG](figures\Replica\Replica__office3__rgbd.png) | [SVG](figures\Replica\Replica__office3__rgbd.svg) | [CSV](figures\Replica\Replica__office3__rgbd.csv) |
| Replica/office4/mono | [PNG](figures\Replica\Replica__office4__mono.png) | [SVG](figures\Replica\Replica__office4__mono.svg) | [CSV](figures\Replica\Replica__office4__mono.csv) |
| Replica/office4/rgbd | [PNG](figures\Replica\Replica__office4__rgbd.png) | [SVG](figures\Replica\Replica__office4__rgbd.svg) | [CSV](figures\Replica\Replica__office4__rgbd.csv) |
| Replica/room0/mono | [PNG](figures\Replica\Replica__room0__mono.png) | [SVG](figures\Replica\Replica__room0__mono.svg) | [CSV](figures\Replica\Replica__room0__mono.csv) |
| Replica/room0/rgbd | [PNG](figures\Replica\Replica__room0__rgbd.png) | [SVG](figures\Replica\Replica__room0__rgbd.svg) | [CSV](figures\Replica\Replica__room0__rgbd.csv) |
| Replica/room1/mono | [PNG](figures\Replica\Replica__room1__mono.png) | [SVG](figures\Replica\Replica__room1__mono.svg) | [CSV](figures\Replica\Replica__room1__mono.csv) |
| Replica/room1/rgbd | [PNG](figures\Replica\Replica__room1__rgbd.png) | [SVG](figures\Replica\Replica__room1__rgbd.svg) | [CSV](figures\Replica\Replica__room1__rgbd.csv) |
| Replica/room2/mono | [PNG](figures\Replica\Replica__room2__mono.png) | [SVG](figures\Replica\Replica__room2__mono.svg) | [CSV](figures\Replica\Replica__room2__mono.csv) |
| Replica/room2/rgbd | [PNG](figures\Replica\Replica__room2__rgbd.png) | [SVG](figures\Replica\Replica__room2__rgbd.svg) | [CSV](figures\Replica\Replica__room2__rgbd.csv) |
| TUM_RGBD/freiburg1_desk/mono | [PNG](figures\TUM_RGBD\TUM_RGBD__freiburg1_desk__mono.png) | [SVG](figures\TUM_RGBD\TUM_RGBD__freiburg1_desk__mono.svg) | [CSV](figures\TUM_RGBD\TUM_RGBD__freiburg1_desk__mono.csv) |
| TUM_RGBD/freiburg1_desk/rgbd | [PNG](figures\TUM_RGBD\TUM_RGBD__freiburg1_desk__rgbd.png) | [SVG](figures\TUM_RGBD\TUM_RGBD__freiburg1_desk__rgbd.svg) | [CSV](figures\TUM_RGBD\TUM_RGBD__freiburg1_desk__rgbd.csv) |
| TUM_RGBD/freiburg1_room/mono | [PNG](figures\TUM_RGBD\TUM_RGBD__freiburg1_room__mono.png) | [SVG](figures\TUM_RGBD\TUM_RGBD__freiburg1_room__mono.svg) | [CSV](figures\TUM_RGBD\TUM_RGBD__freiburg1_room__mono.csv) |
| TUM_RGBD/freiburg1_room/rgbd | [PNG](figures\TUM_RGBD\TUM_RGBD__freiburg1_room__rgbd.png) | [SVG](figures\TUM_RGBD\TUM_RGBD__freiburg1_room__rgbd.svg) | [CSV](figures\TUM_RGBD\TUM_RGBD__freiburg1_room__rgbd.csv) |
| TUM_RGBD/freiburg1_xyz/mono | [PNG](figures\TUM_RGBD\TUM_RGBD__freiburg1_xyz__mono.png) | [SVG](figures\TUM_RGBD\TUM_RGBD__freiburg1_xyz__mono.svg) | [CSV](figures\TUM_RGBD\TUM_RGBD__freiburg1_xyz__mono.csv) |
| TUM_RGBD/freiburg1_xyz/rgbd | [PNG](figures\TUM_RGBD\TUM_RGBD__freiburg1_xyz__rgbd.png) | [SVG](figures\TUM_RGBD\TUM_RGBD__freiburg1_xyz__rgbd.svg) | [CSV](figures\TUM_RGBD\TUM_RGBD__freiburg1_xyz__rgbd.csv) |
| TUM_RGBD/freiburg2_xyz/mono | [PNG](figures\TUM_RGBD\TUM_RGBD__freiburg2_xyz__mono.png) | [SVG](figures\TUM_RGBD\TUM_RGBD__freiburg2_xyz__mono.svg) | [CSV](figures\TUM_RGBD\TUM_RGBD__freiburg2_xyz__mono.csv) |
| TUM_RGBD/freiburg2_xyz/rgbd | [PNG](figures\TUM_RGBD\TUM_RGBD__freiburg2_xyz__rgbd.png) | [SVG](figures\TUM_RGBD\TUM_RGBD__freiburg2_xyz__rgbd.svg) | [CSV](figures\TUM_RGBD\TUM_RGBD__freiburg2_xyz__rgbd.csv) |
| TUM_RGBD/freiburg3_long_office_household/mono | [PNG](figures\TUM_RGBD\TUM_RGBD__freiburg3_long_office_household__mono.png) | [SVG](figures\TUM_RGBD\TUM_RGBD__freiburg3_long_office_household__mono.svg) | [CSV](figures\TUM_RGBD\TUM_RGBD__freiburg3_long_office_household__mono.csv) |
| TUM_RGBD/freiburg3_long_office_household/rgbd | [PNG](figures\TUM_RGBD\TUM_RGBD__freiburg3_long_office_household__rgbd.png) | [SVG](figures\TUM_RGBD\TUM_RGBD__freiburg3_long_office_household__rgbd.svg) | [CSV](figures\TUM_RGBD\TUM_RGBD__freiburg3_long_office_household__rgbd.csv) |
| VECtor/corner-slow/mono | [PNG](figures\VECtor\VECtor__corner-slow__mono.png) | [SVG](figures\VECtor\VECtor__corner-slow__mono.svg) | [CSV](figures\VECtor\VECtor__corner-slow__mono.csv) |
| VECtor/corner-slow/stereo | [PNG](figures\VECtor\VECtor__corner-slow__stereo.png) | [SVG](figures\VECtor\VECtor__corner-slow__stereo.svg) | [CSV](figures\VECtor\VECtor__corner-slow__stereo.csv) |
| VECtor/corridors-dolly/mono | [PNG](figures\VECtor\VECtor__corridors-dolly__mono.png) | [SVG](figures\VECtor\VECtor__corridors-dolly__mono.svg) | [CSV](figures\VECtor\VECtor__corridors-dolly__mono.csv) |
| VECtor/corridors-dolly/stereo | [PNG](figures\VECtor\VECtor__corridors-dolly__stereo.png) | [SVG](figures\VECtor\VECtor__corridors-dolly__stereo.svg) | [CSV](figures\VECtor\VECtor__corridors-dolly__stereo.csv) |
| VECtor/robot-normal/mono | [PNG](figures\VECtor\VECtor__robot-normal__mono.png) | [SVG](figures\VECtor\VECtor__robot-normal__mono.svg) | [CSV](figures\VECtor\VECtor__robot-normal__mono.csv) |
| VECtor/robot-normal/stereo | [PNG](figures\VECtor\VECtor__robot-normal__stereo.png) | [SVG](figures\VECtor\VECtor__robot-normal__stereo.svg) | [CSV](figures\VECtor\VECtor__robot-normal__stereo.csv) |
