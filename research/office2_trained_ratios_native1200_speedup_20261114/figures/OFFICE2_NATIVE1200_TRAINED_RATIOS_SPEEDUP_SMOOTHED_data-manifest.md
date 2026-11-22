# 原始分辨率加速比平滑图数据清单

|Figure|Data file|Real/mock|Source|Script|Outputs|
|---|---|---|---|---|---|
|OFFICE2_NATIVE1200_TRAINED_RATIOS_SPEEDUP_SMOOTHED|data/OFFICE2_NATIVE1200_TRAINED_RATIOS_SPEEDUP_SMOOTHED.csv|真实测量|reports/PER_VIEW_SPEEDUP.csv|plot_smoothed_speedup.py|PNG、SVG、PDF|

显示值采用31帧居中中值滤波，再进行σ=6帧的Gaussian平滑，反射处理边界；不降采样。纵轴沿用0–5.5×。原始计时、统计结果及未平滑图不变。
