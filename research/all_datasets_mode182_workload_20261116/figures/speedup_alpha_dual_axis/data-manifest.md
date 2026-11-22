# 渲染加速比与Alpha合成负载双轴图

|图|数据|性质|来源|脚本|输出|
|---|---|---|---|---|---|
|六组曲线、左右纵轴|SPEEDUP_ALPHA_DATA.csv|真实测量|既有加速比SIX_CURVES_DATA.csv及Alpha百分比ALPHA_COMPOSITING_PERCENT_DATA.csv|plot_speedup_alpha.py|600dpi PNG、PDF、SVG|

左轴保留纯渲染加速比，右轴为有效Alpha合成次数占完整地图的百分比。六组各有两条曲线：空心标记为左轴，填充标记与点线为右轴。三数据集颜色沿用原图；圆点为单目，方点为RGB-D/双目。32场景模式、59,507个视角，原有11档ratio，不平滑、不重测。全部66个数据对的场景数、相机模式、ratio及视角数匹配，输入输出哈希见FIGURE_VALIDATION.json。两纵轴尺度独立，曲线交点不表示两个物理量相等。
