# mode182图表数据清单

|Figure|Data file|Real/mock|Source|Script|Outputs|
|---|---|---|---|---|---|
|纯渲染/完整调用加速比，原始/平滑|reports/PER_VIEW_SPEEDUP.csv|真实测量|formal/TIMINGS.csv|analyze.py|450dpi PNG、SVG、PDF|

加速比逐视角使用同轮Full耗时中位数除以ratio耗时中位数。平滑只作用于显示加速比，原始数据和摘要不变。
