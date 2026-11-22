# 三方法图数据清单

| 图表 | 数据 | 性质 | 来源 | 脚本 | 输出 |
|---|---|---|---|---|---|
| OFFICE2_DYNAMIC_RATIO_THREE_METHODS | data/office2_dynamic_ratio_three_methods.csv | 真实测量 | reports/PLOT_DATA.csv | plot_three_methods.py | PNG/SVG/PDF |

仅保留CaRtGS、Ours Full和动态ratio。数据覆盖完整2000帧；沿用已验收的同轮纯渲染计时。PSNR曲线沿用σ=35帧平滑；点数、耗时不平滑；虚线为逐帧三次中位数的均值±1个总体标准差。动态预算来源仍为离线逐帧搜索，不因图例改名而成为实时控制结果。
