# Full不退化的低Ratio渲染研究版本

推荐独立推理入口为 `SparseRenderer(4)`，未安装到生产项目或改动训练。

服务器目录：`/home/prmi/zwj/experiments/sparse_rasterizer_balanced_20261110`。

- `FULL_SAFE_LOW_RATIO_CUDA_OPTIMIZATION_REPORT.md`：中文结论、机制、验证和全部结果。
- `RASTERIZER_RESULTS.csv`：16场景纯渲染，包括实际K、中位数、p90和每轮对比。
- `PIPELINE_RESULTS.csv`：4场景每次重新计算视角候选、Selector/Compensator输出后的准备＋渲染。
- `GPU_PROFILE.csv`：独立CUDA Event与Kineto Kernel统计。
- `TILE_WORKLOADS.csv`：实际Tile成员数量、空位表比例与短列表覆盖。
- `FINAL_VERIFICATION.json`：预声明门限和完整性验收。
- `reports/formal` 与 `reports/quiet`：服务器原始样本、进程身份、监控和日志；本地在 `results/reports` 中。

## 接口

```python
import torch
from runtime import SparseRenderer

renderer = SparseRenderer(4)  # 每device/Stream一个实例
with torch.no_grad():
    rgb, radii = renderer.render(camera, packet, background)
```

`packet`沿用现有 `render_prepared` 输入格式，XYZ、Scale、Rotation、Opacity和SH均为未改变的FP32属性。每次调用重新构建几何和Tile结构，支持切换视角；不复用上一帧Mask或网络特征。渲染后仍须沿用原产品流程完成展示，不包含Viewer或传输加速。

模式0/1/2保留参考/旧Warp/旧Bitset控制，模式3为目录消融，模式4为本轮推荐。没有按地图K或Ratio切换回旧算法。128项只是每Tile短列表容量，超出会完整枚举目录。

空集合沿用原CUDA primitive的零输出；现有产品外层的背景图特殊处理需要继续保留。不能跨device或Stream共享同一Workspace；返回RGB/radii不与后续帧共享输出存储。

## 重复验证

服务器使用现有 `/home/prmi/anaconda3/envs/zwj/bin/python`。`runtime.py`在本研究目录的build下编译，使用CUDA11.8、sm86，不修改conda。实际加载路径和版本见 `reports/RUNTIME_IDENTITY.json`。归档.so仅对应本次服务器环境，其他设备需重新编译。

运行 `validate.py`、`test_lifecycle.py`、`test_boundaries.py`、`validate_render_inputs.py`。正式测速分别运行 `benchmark_balanced.py --all-scenes --mode 4 --output <新目录>` 和 `benchmark_pipeline.py --output <新目录>`；输出目录必须不存在。低频监控版本入口 `run_quiet_confirmation.py` 是本次记录流程，不能重复覆盖已完成目录。

本地先把原始 `reports` 解压至 `results/reports`，再运行 `analyze_results.py` 重建中文报告、CSV和性能图。所有阶段保留原始Baseline与独立开关，未commit/push。
