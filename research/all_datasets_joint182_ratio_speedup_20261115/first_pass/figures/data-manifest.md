# 三数据集mode182加速比曲线数据清单

|Figure|Data file|Real/mock|Source|Script|Outputs|
|---|---|---|---|---|---|
|各数据集及合图，纯渲染/完整调用|reports/SUMMARY.csv|真实测量|runs/*/formal/TIMINGS.csv|analyze.py|600dpi PNG、PDF、SVG|

每个场景每种相机模式一条曲线；同场景颜色相同，Mono虚线圆点，RGB-D/Stereo实线方点。每个ratio点先计算每视角本版各轮耗时中位数的Full/ratio，再对有效全轨迹等权取均值。不平滑、不插值；纯渲染和完整调用分开。N表示对应模型的Master总数，实际K见CSV。
