# 六条数据集/模式加速比曲线

|Figure|Data file|Real/mock|Source|Script|Outputs|
|---|---|---|---|---|---|
|纯渲染/完整调用|SIX_CURVES_DATA.csv|真实测量|reports/SUMMARY.csv|plot_dataset_modes.py|600dpi PNG、PDF、SVG|

本版每视角1轮。Replica每模式8个场景、TUM每模式5个场景、VECtor每模式3个场景。每个场景先以自己的Full为基准得到全轨迹平均逐视角加速比，再在数据集/模式内对场景等权平均。不按轨迹长度加权，不合并不同场景的耗时分母。六条线共享一个坐标系，真实ratio线性横轴，未平滑。纯渲染不含准备或网络；完整调用单独出图。VECtor Stereo为双目训练模型的单张左图推理。
