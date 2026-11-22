# Student32专用CPU推理

此研究入口接续已绑定的蒸馏学生，只优化训练完成后的推理。旧Teacher/学生入口、训练、原权重和生产默认不变。

```python
from student_fast_runtime import CpuRenderer, extension

engine = CpuRenderer.from_checkpoint(
    teacher_checkpoint, original_gpu_noise_snapshot,
    threads=6, student_checkpoint=student_checkpoint,
    student_kernel=2, student_decode=1, student_context=1,
)
packet, ids, stats = engine.prepare(cpu_camera, ratio_budget=0.3)
handoff = extension().NativeHandoff(engine.native, engine.sh_degree)
ticket = handoff.prepare(cpu_camera, ratio_budget=0.3)
image = render_prepared(ticket.camera, ticket.wait(), background)
ticket.finish()
handoff.synchronize()
```

相机矩阵和中心是CPU FP32，GPU只使用原Rasterizer。wait、consume、finish须使用同一Stream；本轮渲染仍在默认Stream。学生必须匹配Teacher/地图/Feature/行序指纹；仍使用原GPU固定噪声快照，不能在CPU重新采样替代。

默认研究设置是2/1/1：四Head派生权重合并调度、32D学生专用循环、AVX2批量解码、6 Anchor批量View投影。`student_kernel=0,student_decode=0,student_context=0`关闭本轮新增路径；原后端可直接从上一轮`student_runtime`加载。

原始学生权重及3534个参数不变。额外Head Buffer只有约2KiB，没有跨帧结果缓存。更新学生、Master、Anchor Feature或拓扑后必须重建快照。使用本机glibc libmvec的AVX2 expf/logf，FP32解码存在经过实际渲染检查的微差；不声称所有属性逐位相同。

该优化不能消除学生相对Teacher既有的蒸馏损失。新增质量差异与上一轮学生分别评价，详见本轮报告。当前验证限定x86 AVX2/FMA硬件和已有Replica16组权重，不是ARM测速结果。未部署生产。

复现最终构建直接运行`kernel_runtime.py`；`evolve.py/add_*.py/finalize_source.py`是开发过程脚本，请勿在冻结目录再次运行覆盖最终源码。旧候选源与二进制在development_v1/v2/v3仅供复核。
