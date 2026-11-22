# 等效渲染FPS图数据清单

| 数据 | 性质 | 来源 | 脚本 | 输出 |
|---|---|---|---|---|
| data/OFFICE2_EQUIVALENT_RENDERING_FPS.csv | 真实测量转换 | reports/PLOT_DATA.csv及formal/TIMINGS.csv | plot_equivalent_fps.py | PNG/SVG/PDF |

逐帧由三次计时中位数转换FPS，再做21帧居中移动平均；点数和选择预算没有改变。展示的是纯渲染的等效FPS，不代表含网络、准备和显示的完整运行帧率。
