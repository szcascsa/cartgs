# 数据清单

|图|数据|真实／模拟|统计|输出|
|---|---|---|---|---|
|PSNR、SSIM、LPIPS、Gaussian总数各一张及总览|../data/COMBINED_MEANS.csv|真实|32个场景／模式等权平均|450 DPI PNG、SVG、PDF|

源数据：Replica 16组、TUM 10组、VECtor 6组。仅保留ours_view_ratio和cartgs_opacity。质量不重新渲染；局部U为每个场景／模式／Ratio全部评价视角选中母Gaussian ID并集大小；Top-K取固定保留集大小。
