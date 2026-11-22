# 数据清单

| Figure | Data file | Real/mock | Source | Script | Outputs |
|---|---|---|---|---|---|
| 网络/渲染分离 | reports/samples.csv | Real | 本轮A/C/D方形18视角双轮 | analyze.py | TENSOR_CORE_RENDERING_SEPARATED.png / .svg |
| 网络组件 | reports/components/samples.csv | Real | 固定身份Graph Event | analyze.py | TENSOR_NETWORKS_ONLY.png / .svg |

真实数据，未平滑；纯渲染不含两网络。完整流程/组件独立计时，不机械相加。原数据及脚本SHA见DATA_MANIFEST.json。
