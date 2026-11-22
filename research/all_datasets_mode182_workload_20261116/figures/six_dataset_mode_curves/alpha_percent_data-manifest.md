# Alpha合成量与ratio六条曲线（百分比）

|图|数据|类型|来源|脚本|输出|
|---|---|---|---|---|---|
|Alpha Compositing Workload|ALPHA_COMPOSITING_PERCENT_DATA.csv|真实计数|reports/SIX_CURVE_DATA.csv|plot_alpha_compositing.py|600dpi PNG、PDF、SVG|

32组、59,507个有效全轨迹视角；Replica每模式8场景、TUM每模式5场景、VECtor每模式3场景。每帧有效Alpha合成次数除以同视角未选择、未补偿完整地图次数，再场景内平均，最后对场景等权平均。纵轴50%表示合成次数为完整地图的一半，不表示加速2倍。CSV同时保留原始相对值和百分比列。六条线不做平滑；虚线圆点为Mono，实线方点为RGB-D/Stereo。100%参考线名称为Our Complete Map (100%)。沿用mode182原始分辨率与名义Anchor局部预算；VECtor Stereo只计单张左图。
