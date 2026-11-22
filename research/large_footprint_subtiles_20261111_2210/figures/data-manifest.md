# 图表数据清单

全部性能数据来自本轮实测。合成压力图使用人工构造输入的真实计时，未使用规划占位数值。
PNG为450 DPI，同时保存SVG；中文字体为系统微软雅黑或Noto CJK。

|图|数据文件（reports/AGGREGATE下）|性质|生成脚本|
|---|---|---|---|
|RESULT_OVERVIEW|CONDITIONS.csv, COUNT_SCALING_SUMMARY.csv, SUBTILE_REJECTION.csv, ATTRIBUTION_SUMMARY.csv|真实实验|plot_results.py|
|SCENE_LATENCY_01|PER_SCENE_RATIO_ROUND.csv|真实实验|plot_results.py|
|SCENE_COUNT_LATENCY_01|PER_SCENE_RATIO_ROUND.csv|真实实验|plot_results.py|
|SCENE_LATENCY_02|PER_SCENE_RATIO_ROUND.csv|真实实验|plot_results.py|
|SCENE_COUNT_LATENCY_02|PER_SCENE_RATIO_ROUND.csv|真实实验|plot_results.py|
|KERNEL_BREAKDOWN|../PROFILE/MODULES.csv|真实Profile|plot_results.py|
|SYNTHETIC_FOOTPRINT_STRESS|SYNTHETIC_STRESS.csv|合成输入的真实计时，非规划数据|plot_results.py|
