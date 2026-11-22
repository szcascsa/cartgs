# 三方法渲染耗时数据清单

数据：data/OFFICE2_MODE40_RENDERING_MS_THREE_METHODS_UNSMOOTHED.csv；源文件：reports/PER_FRAME.csv、formal/TIMINGS.csv；脚本：plot_rendering_ms_three_methods.py。

三方法均为Mode40，完整2000帧，同轮三次计时取每帧中位数，直接展示毫秒。没有时间轴平滑。纯渲染时间排除准备、网络和显示；动态预算沿用离线冻结K。导出450 DPI PNG和SVG、PDF。
