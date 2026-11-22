# 低点数 CUDA 渲染研究入口

服务器独立目录：`/home/prmi/zwj/experiments/sparse_rasterizer_sixhour_20261110_0053`。本地同名 `research` 子目录保留源码、报告和数值证据。原生产和训练不导入本目录。

最终数值基准为实际原入口 A；上一版 B 只读保留，用作历史速度对照，其数值缺陷单列。不能将 B 当成逐位正确性判据。实际采用的选项由 `reports/FROZEN_CANDIDATE.json` 和最终验收报告决定。

## 冻结渲染器

```python
import torch
from frozen_renderer import create_renderer

# 在与后续render相同的device/Stream上，模型加载时初始化。
with torch.no_grad():
    renderer = create_renderer(camera, background, maximum_points=map_count)
    # 每次调用都使用当前视角和当前选择/补偿结果；没有同帧内容缓存。
    image, radii = renderer.render(current_camera, current_packet, background)
```

`current_packet` 沿用既有输入：`sh_degree` 和 `inputs`，其中包含 `means3D`、`opacities`、`scales`、`rotations`、`dc`、`shs`，或原有 `colors_precomp` / `cov3D_precomp`。严格 K 与 Gaussian 身份由原准备流程决定。

`use_graph=False` 使用已验收普通 CUDA 路径 C，作为完整可回退版本；仍不触发 JIT。Full 与低 Ratio 使用同一算法，不按 K/Ratio 绕开。不同 Stream 应分别创建实例，错误跨 Stream 复用明确拒绝。地图上限或分辨率变化必须重新准备容量计划；每帧几何/SH/补偿属性始终重读。输出 Tensor 每帧独立持有。

Q 的相机准备使用 `camera_graph_preparation.GraphCameraPreparer(MiniCam)`，然后逐帧调用 `create(width, height, FoVx, FoVy, world_view_transform)`。传入当前世界到相机矩阵，保留原 CPU 位姿求逆；不要改传上帧矩阵。该组件只预计算固定内参、捕获原 BMM/inverse 执行，仍逐帧检查 inverse info 并复制独立输出。F 回退为 `camera_preparation.CameraPreparer`，原相机回退为原 `MiniCam`。是否通过最终验收以 `CAMERA_GRAPH_FINAL_VERIFICATION.json` 为准。

若 `CAMERA_ASYNC_FINAL_VERIFICATION.json` 也通过，最终使用独立 R：`camera_async_preparation.AsyncGraphCameraPreparer(MiniCam)`，接口相同，仅将上传改为同 Stream 的非阻塞操作；原逆矩阵检查仍每帧立即执行。Q/F/G/C 均作为已验收回退保留。R 的实际配对收益见 `FINAL_ASYNC_PIPELINE_RESULTS.csv`，不把 Q 的收益重复归入 R。

正式调用可使用 `frozen_renderer.create_camera_preparer(MiniCam, variant='R')`；该入口在模型加载时检查正式验收文件与相机源码SHA，未通过或源码变化则拒绝加载。`variant='Q'`、`'F'` 显式选择已有回退。它只作用于研究推理入口，未修改生产默认。

## 构建与复验

仅使用既有 `zwj` 环境，CUDA 11.8，RTX 3090 / sm86。未安装或更改共享环境。其他 GPU 需重新编译与验收，不能直接引用本服务器的延迟。

```bash
cd /home/prmi/zwj/experiments/sparse_rasterizer_sixhour_20261110_0053
export CUDA_HOME=/usr/local/cuda-11.8
export TORCH_CUDA_ARCH_LIST=8.6
export MAX_JOBS=2
/home/prmi/anaconda3/envs/zwj/bin/python runtime.py
/home/prmi/anaconda3/envs/zwj/bin/python graph_runtime.py
```

源码变更或重新编译后必须重新验证哈希与数值，不允许改写现有冻结清单来掩盖变更。已有结果在输入和代码未变化时无需重复运行。`launch.py` 根据本目录任务 PID/start_ticks 防止重叠；新任务名和输出目录必须唯一。原二进制、场景、checkpoint 和分组在 `reports/PROTOCOL.json`、`SCENES.json`。

## 结果索引

- `LOW_POINT_RENDERING_SIX_HOUR_REPORT.md`：最终中文报告及全部边界。
- `RASTERIZER_RESULTS.csv`：A/B/C/G 的纯渲染两轮统计，含实际 K、p95、重复波动。
- `ROUND_RESULTS.csv`、`FULL_GATES.csv`：逐场景逐轮 Full 验收。
- `FINAL_PIPELINE_RESULTS.csv`：固定内参 F 的独立配对结果。
- `FINAL_CAMERA_GRAPH_PIPELINE_RESULTS.csv`：若 Q 最终通过，记录相机执行图的独立配对结果。
- `FINAL_ASYNC_PIPELINE_RESULTS.csv`：若 R 最终通过，记录最终异步相机上传组合的独立配对结果。
- `TILE_WORKLOADS.csv`：计时外 K、radii 可见、真实 Alpha 参与、Tile/成员分布诊断。
- `COUNT_VS_SPEED_PER_VIEW.csv`、`SAME_ID_COMPENSATION_CONTROL.csv`：点数与 footprint 归因，不更改最终补偿算法。
- `BASELINE_B_NUMERICAL_DIFFERENCES.csv`：旧 B 的非零误差，不能隐藏。
- `FAILURE_CANDIDATES.md`：失败与拒绝候选，含受干扰轮次。
- `figures/FINAL_SCENE_INDEX.md`：全部场景曲线，线性 Ratio 坐标、实际 K；无渲染图片。
- `reports/INTEGRITY_FINAL.json`、`CLEANUP_MANIFEST.json`：保护文件完整性与仅本轮可重建产物清理记录。

没有自动 commit、push、部署或切换生产默认。相机优化作为独立组件，只有通过其最终验收后才使用，不能仅引用调优集收益。
