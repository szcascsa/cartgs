# 纯渲染曲线数据清单

| Figure | Data file | Real/mock | Source | Script | Outputs |
|---|---|---|---|---|---|
| 原图加入方形纯渲染 | reports/SQUARE2000_RESULTS.csv | Real | samples.csv中2000×2000的raster_event，72条/条件中位数 | plot_with_render_only.py | GPU_STUDENT_SQUARE2000_WITH_RENDER_ONLY.png / .svg |

保留原四条曲线，增加绿色LoD纯渲染与Full纯渲染两条曲线。24条绘制曲线逐项核对数据；原CSV哈希未改变。未进行训练、渲染或平滑。纯渲染不含推理/上传/CPU调度；完整流程独立计时。详细SHA及绘制数值见同名JSON。
