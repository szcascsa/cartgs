# 论文ratio–加速比图数据清单

|Figure|Data file|Real/mock|Source|Script|Outputs|
|---|---|---|---|---|---|
|OFFICE2_MODE182_RATIO_VS_RENDERING_SPEEDUP|OFFICE2_MODE182_RATIO_VS_RENDERING_SPEEDUP.csv|真实测量|reports/PER_VIEW_SPEEDUP.csv、SUMMARY.csv|plot_ratio_speedup.py|PDF、SVG、600dpi PNG|

横轴是真实ratio，采用线性比例；每个标记对应已测ratio，直线连接测量点，不补造ratio样本。纵轴为2000个视角的Full/ratio纯渲染加速比算术平均；每视角先分别对三次计时取中位数。浅色区域是5%–95%视角分位区间，展示不同视角的变化，不能称为均值置信区间。未使用此前时间轴平滑数据，原始计时和统计结果保持不变。
