# CPU 深度推理候选

本目录是上一版 `cpu_native_aggressive_20261110_1115` 的独立后续研究。生产默认、训练和原checkpoint均未修改。

保留4点×16输出AVX2微核和C++准备/交接调度。8点候选未获得稳定收益，源码与数据仅作失败证据保留。数值及速度结论以 `CPU_DEEP_INFERENCE_OPTIMIZATION_REPORT.md` 为准。

## CPU独立启动

```bash
env OMP_WAIT_POLICY=PASSIVE KMP_BLOCKTIME=0 GOMP_SPINCOUNT=0 \
    /home/prmi/anaconda3/envs/zwj/bin/python application.py
```

```python
import sys
sys.path.insert(0, "/home/prmi/zwj/experiments/compensator_distillation_replica_all_20261108/source/python")
import torch
from deep_runtime import CpuRenderer, extension

torch.set_num_threads(1)
torch.set_num_interop_threads(1)
engine = CpuRenderer.from_checkpoint(checkpoint, original_gpu_noise_path, threads=6)
packet, ids, stats = engine.prepare(camera, ratio_budget=0.1)
```

`camera` 的矩阵、中心为CPU FP32，坐标约定沿用现有Viewer。固定预算用 `budget=1000`；全局导出用 `global_ratio=0.1, view_conditioned=False`，保持原分位数/同分语义。换场景或更改冻结地图/模型时重新建实例；不能在原实例内继续训练。

## 原生相机/属性上传调度

```python
from renderer import render_prepared

handoff = extension().NativeHandoff(engine.native, engine.sh_degree, slots=3, device=0)
ticket = handoff.prepare(camera, ratio_budget=0.1)
packet = ticket.wait()
result = render_prepared(ticket.camera, packet, background, return_radii=True)
ticket.finish()
```

相机与属性一次H2D。3槽有界，`ticket`持有槽位和owner；未finish的槽拒绝覆盖。finish后不要继续使用ticket里的属性、相机或IDs，后续帧可以复用；需要长期保留时显式clone。这个对象按一个提交线程使用。

CUDA数据消费者可在自定义Stream wait/使用/finish，方法取当前Stream。**原Rasterizer仍需在默认Stream渲染**，本轮未改动其内部默认Stream核。不要将任意Stream的数据交接能力误解为原Rasterizer的任意Stream支持。

## 开关与基线

```python
# 仅调度：
engine = CpuRenderer(model, pc, noise, threads=6, kernel_mode=0)
# 仅矩阵微核，保留旧包装及调度：
engine = CpuRenderer(model, pc, noise, threads=6,
                     fused_schedule=False, native_wrapper=False, kernel_mode=1)
# 同扩展控制路径：
engine = CpuRenderer(model, pc, noise, threads=6,
                     fused_schedule=False, native_wrapper=False, kernel_mode=0)
```

真正上一版V4保留在原目录，通过其 `cpu_runtime.CpuRenderer` 调用。最终扩展身份见 `reports/FROZEN_CANDIDATE.json`。首次源码编译不计入稳态推理时间；当前共享库依赖安装的Torch/CUDA运行库，即使CPU prepare未初始化CUDA，也不能直接复制到没有对应运行库的ARM设备。

CPU计时外构造相机Tensor，但每个请求都重算候选、视角、网络、选点和属性。不使用旧帧结果缓存。正式CPU测量只驻留一个原生线程池；先前多池短块测试的争抢反例保留。切勿在仅6核机器上同时常驻多套各6线程的活跃实例，然后把争抢当作单实例部署延迟。

完整CSV、失败项、质量统计、性能曲线和Sanitizer边界见报告；不保存渲染图片，不自动部署。
