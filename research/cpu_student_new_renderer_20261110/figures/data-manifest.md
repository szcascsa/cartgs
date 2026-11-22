# 可视化数据清单

所有图使用真实实验数据，未平滑、未重采样或重新训练。Ratio为线性数值轴。

| Figure | Data file | Real/mock | Source | Script | Outputs |
|---|---|---|---|---|---|
| CPU/GPU分离耗时 | reports/CPU_GPU_SEPARATED_RESULTS.csv | Real | 本轮双轮正式测速 | plot_results.py | CPU_GPU_SEPARATED_LATENCY.png / .svg |
| GPU渲染收益 | reports/CPU_GPU_SEPARATED_RESULTS.csv | Real | 同输入Full/LoD新G Renderer | plot_results.py | GPU_RENDERING_BENEFIT.png / .svg |

总地图N来自reports/TIMING_SUMMARY.csv；实际K为六视角中位数，可为半整数。原始样本位于reports/samples.csv。各分项独立计时，不是加和分解；完整流程使用独立墙钟样本。Full属性常驻GPU，CPU-LoD每帧上传选中属性。输入、脚本和输出SHA256见DATA_MANIFEST.json。
