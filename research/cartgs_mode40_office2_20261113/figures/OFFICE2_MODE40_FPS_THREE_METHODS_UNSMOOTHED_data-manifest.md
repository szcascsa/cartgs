# 三方法等效渲染FPS数据清单

数据：data/OFFICE2_MODE40_FPS_THREE_METHODS_UNSMOOTHED.csv；源文件：reports/PER_FRAME.csv、formal/TIMINGS.csv；脚本：plot_fps_three_methods.py。

三方法均为Mode40，完整2000帧，同轮三次计时取每帧中位数后转换FPS。没有时间轴平滑。纯渲染时间排除准备、网络和显示；动态预算沿用离线冻结K。导出450 DPI PNG和SVG、PDF。
