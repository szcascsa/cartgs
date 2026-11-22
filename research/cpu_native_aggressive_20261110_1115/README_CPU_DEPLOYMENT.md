# 激进 CPU 冻结推理入口

本目录提供独立研究后端，不接管生产默认或训练。Selector、Compensator、候选筛选和属性准备全部在 CPU；渲染仍由原 CUDA Rasterizer 执行。使用原 Teacher 权重，没有重新训练或量化。

## 启动条件

- 实测环境：Linux x86-64、Intel i5-9600K、AVX2/FMA、PyTorch 2.3.1；不支持没有 AVX2/FMA 的处理器。
- 正式扩展位于 `build_v4/mycartgs_cpu_aggressive_v4.so`；源码和二进制身份见 `reports/FROZEN_CANDIDATE.json`。
- 需要现有冻结 `python` 模块提供 checkpoint 加载和网络定义。测试使用 `/home/prmi/zwj/experiments/compensator_distillation_replica_all_20261108/source/python`。
- 相同 seed 在 CPU/CUDA 上不保证相同随机键。复用原 GPU 选点身份时，必须使用已导出的 `noise_snapshots/<scene>_seed0.pt`，不要在 CPU 上重新采样替代。
- 单个实例固定一份模型和地图，加载时复制状态；模型更新、Clone/Split/Prune 或 Anchor 迁移后重新创建实例。不能将它接入训练循环。

## 线程策略

在**新进程启动前**设置：

```bash
env OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0 GOMP_SPINCOUNT=0 \
    /home/prmi/anaconda3/envs/zwj/bin/python your_application.py
```

这些设置仅影响该进程。原生部署进程可设置 `torch.set_num_threads(1)`、`torch.set_num_interop_threads(1)`，使 Torch 辅助操作不另起一套多线程竞争；真正的并行由 C++ 的 6 线程池执行。不要同时保留多个各占 6 线程的实例来衡量单帧速度。跨场景正式对照双方都使用相同休眠策略和 6 线程，原始默认策略的反例已保留。

## 最小使用方式

```python
import sys
sys.path.insert(0, "/home/prmi/zwj/experiments/compensator_distillation_replica_all_20261108/source/python")
import torch
from cpu_runtime import CpuRenderer

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
renderer = CpuRenderer.from_checkpoint(checkpoint, original_noise_path, threads=6)

# camera 的矩阵与中心必须为 CPU FP32；矩阵约定沿用现有 Viewer。
packet, ids, stats = renderer.prepare(camera, ratio_budget=0.1)
# 固定点数：renderer.prepare(camera, budget=1000)
# 全局导出：renderer.prepare(camera, global_ratio=0.1, view_conditioned=False)
```

局部预算沿用名义 Anchor 候选池，`K=floor(ratio×M)`，资格不足时截断。全局保持原 FlexGS 分位数阈值和同分语义。返回的普通输出拥有独立内存，后续请求不覆盖；传入调用方 Buffer 时由调用方管理生命周期。

## CPU→GPU 一次上传

```python
from cpu_gpu_camera_handoff import CpuGpuCameraHandoff
from renderer import render_prepared

handoff = CpuGpuCameraHandoff(renderer, slots=3)
ticket = handoff.prepare(camera, ratio_budget=0.1)
packet = ticket.wait()
image = render_prepared(ticket.camera, packet, background, return_radii=True)
ticket.finish()
```

属性及本帧 51 个相机 FP32 连续上传。三个槽位有界；事件保证上传和消费次序。`finish()` 后槽位内容可被后续帧覆盖，不能继续持有 ticket 内的相机或属性视图使用。需要长期保留的 IDs/属性由调用方显式复制。**原 Rasterizer 含默认 Stream 的 CUDA 核，正式路径必须在默认 Stream 渲染**；不能简单地把上面代码放到另一个 Stream 上下文。通用数据消费者可在自定义 Stream wait、使用并 finish，已单独验证属性及相机数据传递，详见 `handoff_boundaries.py`。这个对象本身按单一提交线程使用。

Full 原入口本来就没有 Selector/Compensator 工作。本轮的 `kind=3` 是整图重打包的诊断接口，会比直接返回 Master 视图慢，不应将它当作 Full 渲染提速入口。

## 验证与复现

完整结果见 `CPU_AGGRESSIVE_INFERENCE_OPTIMIZATION_REPORT.md`。`reports/CPU_PAIRED_RESULTS.csv` 保留逐场景、逐条件、两轮中位和 p95；`reports/CPU_QUALITY_ALL_FRAMES.csv` 保留全部图像指标与最差项。测试不保存渲染图片。

独立 CPU 恢复测试为 `standalone_check.py`；预算、Buffer 和错误恢复为 `boundaries.py`；跨 Stream 交接为 `handoff_boundaries.py`。复跑请使用新的 `--output` 名称，并通过 `launch.py` 的进程身份检查避免重复任务。当前结果只证明这台 CPU 及已测 checkpoint 的表现，不能直接当成 ARM/移动端实测。
