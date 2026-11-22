# 渲染质量、平均实际渲染点数与完整地图规模

已完成32组逐场景图和一张总览，覆盖Replica 16组、TUM 10组、VECtor 6组。

## 点数口径

- 完整源地图N：读取checkpoint的完整Gaussian数；原版和我们可能不同。
- 全局导出K：全图选择并补偿后固定导出包的完整点数；同一包在所有视角保持不变。
- 局部提交K：当前名义Anchor视锥内按ratio预算选择并补偿后，实际送入Renderer的点数；随视角变化，表中给出平均K，完整源地图N仍需保留。
- CSV中source_map_N表示完整源地图，render_model_N在全局模式为固定导出包K、在局部模式为需要保留的完整地图N；exported_gaussians仅对固定包填写。
- 平均实际渲染点数：每帧至少对一个像素产生有效Alpha合成贡献的独立Gaussian数量，再按该次评价全部帧求平均。复用生产Rasterizer已有GI整数贡献计数，不用Opacity阈值近似，不等同于radii>0或K。
- 全局／局部LoD均在选点、补偿和原有属性变换之后的最终渲染输入上计数；Full使用未补偿Master，未用名义Anchor可见性代替真实像素贡献。
- 另记录radii>0的投影有效点数，所有帧满足贡献数≤投影有效数≤提交K≤源地图N。

## 质量及验证

PSNR／SSIM／LPIPS直接复用此前CaRtGS未校正原图口径的完整逐帧评价，768条汇总质量数值不变。仅为缺失的实际点数补做冻结前向计数，不重训、不保存渲染图片。

全部局部选点身份哈希与原评价逐帧一致，全局导出身份一致；每个工作项在首／中／末帧验证计数版与普通Rasterizer图像和radii逐bit一致，并用原图复核PSNR。模型与历史评价文件哈希前后保持一致。

统计器先使用原有GI整数贡献次数，后续使用独立编译的单调原子0/1标记，只记录是否贡献，免去不需要的浮点GI累加。三个数据集的72组验收输入逐Gaussian贡献集合、图像和radii完全一致；后续每个计数项仍在首／中／末帧与原接口对照。独立统计器不替代生产GI或训练模块。

原版继续按每场景、每相机模式Full轨迹PSNR最高的一次重复取值，SSIM、LPIPS、N和实际贡献点数绑定同一次checkpoint。未重新挑选最优模型。双方使用各自完整轨迹，不作为共同帧的同点数配对实验。

所有ratio横轴使用实际线性距离，质量不做平滑或插值；空心点及*为未见ratio。计数纵轴及质量—点数图横轴使用对数，已明确标注。

总览：[PNG](figures/ALL_SCENES_OVERVIEW.png) / [SVG](figures/ALL_SCENES_OVERVIEW.svg)。总表：[QUALITY_RENDERED_COUNTS.csv](QUALITY_RENDERED_COUNTS.csv)。

| 场景／模式 | 图 | 矢量图 | 数据 |
|---|---|---|---|
| Replica/office0/mono | [PNG](figures/Replica/Replica__office0__mono.png) | [SVG](figures/Replica/Replica__office0__mono.svg) | [CSV](figures/Replica/Replica__office0__mono.csv) |
| Replica/office0/rgbd | [PNG](figures/Replica/Replica__office0__rgbd.png) | [SVG](figures/Replica/Replica__office0__rgbd.svg) | [CSV](figures/Replica/Replica__office0__rgbd.csv) |
| Replica/office1/mono | [PNG](figures/Replica/Replica__office1__mono.png) | [SVG](figures/Replica/Replica__office1__mono.svg) | [CSV](figures/Replica/Replica__office1__mono.csv) |
| Replica/office1/rgbd | [PNG](figures/Replica/Replica__office1__rgbd.png) | [SVG](figures/Replica/Replica__office1__rgbd.svg) | [CSV](figures/Replica/Replica__office1__rgbd.csv) |
| Replica/office2/mono | [PNG](figures/Replica/Replica__office2__mono.png) | [SVG](figures/Replica/Replica__office2__mono.svg) | [CSV](figures/Replica/Replica__office2__mono.csv) |
| Replica/office2/rgbd | [PNG](figures/Replica/Replica__office2__rgbd.png) | [SVG](figures/Replica/Replica__office2__rgbd.svg) | [CSV](figures/Replica/Replica__office2__rgbd.csv) |
| Replica/office3/mono | [PNG](figures/Replica/Replica__office3__mono.png) | [SVG](figures/Replica/Replica__office3__mono.svg) | [CSV](figures/Replica/Replica__office3__mono.csv) |
| Replica/office3/rgbd | [PNG](figures/Replica/Replica__office3__rgbd.png) | [SVG](figures/Replica/Replica__office3__rgbd.svg) | [CSV](figures/Replica/Replica__office3__rgbd.csv) |
| Replica/office4/mono | [PNG](figures/Replica/Replica__office4__mono.png) | [SVG](figures/Replica/Replica__office4__mono.svg) | [CSV](figures/Replica/Replica__office4__mono.csv) |
| Replica/office4/rgbd | [PNG](figures/Replica/Replica__office4__rgbd.png) | [SVG](figures/Replica/Replica__office4__rgbd.svg) | [CSV](figures/Replica/Replica__office4__rgbd.csv) |
| Replica/room0/mono | [PNG](figures/Replica/Replica__room0__mono.png) | [SVG](figures/Replica/Replica__room0__mono.svg) | [CSV](figures/Replica/Replica__room0__mono.csv) |
| Replica/room0/rgbd | [PNG](figures/Replica/Replica__room0__rgbd.png) | [SVG](figures/Replica/Replica__room0__rgbd.svg) | [CSV](figures/Replica/Replica__room0__rgbd.csv) |
| Replica/room1/mono | [PNG](figures/Replica/Replica__room1__mono.png) | [SVG](figures/Replica/Replica__room1__mono.svg) | [CSV](figures/Replica/Replica__room1__mono.csv) |
| Replica/room1/rgbd | [PNG](figures/Replica/Replica__room1__rgbd.png) | [SVG](figures/Replica/Replica__room1__rgbd.svg) | [CSV](figures/Replica/Replica__room1__rgbd.csv) |
| Replica/room2/mono | [PNG](figures/Replica/Replica__room2__mono.png) | [SVG](figures/Replica/Replica__room2__mono.svg) | [CSV](figures/Replica/Replica__room2__mono.csv) |
| Replica/room2/rgbd | [PNG](figures/Replica/Replica__room2__rgbd.png) | [SVG](figures/Replica/Replica__room2__rgbd.svg) | [CSV](figures/Replica/Replica__room2__rgbd.csv) |
| TUM_RGBD/freiburg1_desk/mono | [PNG](figures/TUM_RGBD/TUM_RGBD__freiburg1_desk__mono.png) | [SVG](figures/TUM_RGBD/TUM_RGBD__freiburg1_desk__mono.svg) | [CSV](figures/TUM_RGBD/TUM_RGBD__freiburg1_desk__mono.csv) |
| TUM_RGBD/freiburg1_desk/rgbd | [PNG](figures/TUM_RGBD/TUM_RGBD__freiburg1_desk__rgbd.png) | [SVG](figures/TUM_RGBD/TUM_RGBD__freiburg1_desk__rgbd.svg) | [CSV](figures/TUM_RGBD/TUM_RGBD__freiburg1_desk__rgbd.csv) |
| TUM_RGBD/freiburg1_room/mono | [PNG](figures/TUM_RGBD/TUM_RGBD__freiburg1_room__mono.png) | [SVG](figures/TUM_RGBD/TUM_RGBD__freiburg1_room__mono.svg) | [CSV](figures/TUM_RGBD/TUM_RGBD__freiburg1_room__mono.csv) |
| TUM_RGBD/freiburg1_room/rgbd | [PNG](figures/TUM_RGBD/TUM_RGBD__freiburg1_room__rgbd.png) | [SVG](figures/TUM_RGBD/TUM_RGBD__freiburg1_room__rgbd.svg) | [CSV](figures/TUM_RGBD/TUM_RGBD__freiburg1_room__rgbd.csv) |
| TUM_RGBD/freiburg1_xyz/mono | [PNG](figures/TUM_RGBD/TUM_RGBD__freiburg1_xyz__mono.png) | [SVG](figures/TUM_RGBD/TUM_RGBD__freiburg1_xyz__mono.svg) | [CSV](figures/TUM_RGBD/TUM_RGBD__freiburg1_xyz__mono.csv) |
| TUM_RGBD/freiburg1_xyz/rgbd | [PNG](figures/TUM_RGBD/TUM_RGBD__freiburg1_xyz__rgbd.png) | [SVG](figures/TUM_RGBD/TUM_RGBD__freiburg1_xyz__rgbd.svg) | [CSV](figures/TUM_RGBD/TUM_RGBD__freiburg1_xyz__rgbd.csv) |
| TUM_RGBD/freiburg2_xyz/mono | [PNG](figures/TUM_RGBD/TUM_RGBD__freiburg2_xyz__mono.png) | [SVG](figures/TUM_RGBD/TUM_RGBD__freiburg2_xyz__mono.svg) | [CSV](figures/TUM_RGBD/TUM_RGBD__freiburg2_xyz__mono.csv) |
| TUM_RGBD/freiburg2_xyz/rgbd | [PNG](figures/TUM_RGBD/TUM_RGBD__freiburg2_xyz__rgbd.png) | [SVG](figures/TUM_RGBD/TUM_RGBD__freiburg2_xyz__rgbd.svg) | [CSV](figures/TUM_RGBD/TUM_RGBD__freiburg2_xyz__rgbd.csv) |
| TUM_RGBD/freiburg3_long_office_household/mono | [PNG](figures/TUM_RGBD/TUM_RGBD__freiburg3_long_office_household__mono.png) | [SVG](figures/TUM_RGBD/TUM_RGBD__freiburg3_long_office_household__mono.svg) | [CSV](figures/TUM_RGBD/TUM_RGBD__freiburg3_long_office_household__mono.csv) |
| TUM_RGBD/freiburg3_long_office_household/rgbd | [PNG](figures/TUM_RGBD/TUM_RGBD__freiburg3_long_office_household__rgbd.png) | [SVG](figures/TUM_RGBD/TUM_RGBD__freiburg3_long_office_household__rgbd.svg) | [CSV](figures/TUM_RGBD/TUM_RGBD__freiburg3_long_office_household__rgbd.csv) |
| VECtor/corner-slow/mono | [PNG](figures/VECtor/VECtor__corner-slow__mono.png) | [SVG](figures/VECtor/VECtor__corner-slow__mono.svg) | [CSV](figures/VECtor/VECtor__corner-slow__mono.csv) |
| VECtor/corner-slow/stereo | [PNG](figures/VECtor/VECtor__corner-slow__stereo.png) | [SVG](figures/VECtor/VECtor__corner-slow__stereo.svg) | [CSV](figures/VECtor/VECtor__corner-slow__stereo.csv) |
| VECtor/corridors-dolly/mono | [PNG](figures/VECtor/VECtor__corridors-dolly__mono.png) | [SVG](figures/VECtor/VECtor__corridors-dolly__mono.svg) | [CSV](figures/VECtor/VECtor__corridors-dolly__mono.csv) |
| VECtor/corridors-dolly/stereo | [PNG](figures/VECtor/VECtor__corridors-dolly__stereo.png) | [SVG](figures/VECtor/VECtor__corridors-dolly__stereo.svg) | [CSV](figures/VECtor/VECtor__corridors-dolly__stereo.csv) |
| VECtor/robot-normal/mono | [PNG](figures/VECtor/VECtor__robot-normal__mono.png) | [SVG](figures/VECtor/VECtor__robot-normal__mono.svg) | [CSV](figures/VECtor/VECtor__robot-normal__mono.csv) |
| VECtor/robot-normal/stereo | [PNG](figures/VECtor/VECtor__robot-normal__stereo.png) | [SVG](figures/VECtor/VECtor__robot-normal__stereo.svg) | [CSV](figures/VECtor/VECtor__robot-normal__stereo.csv) |
