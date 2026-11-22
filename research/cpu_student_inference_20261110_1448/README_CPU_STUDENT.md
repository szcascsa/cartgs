# 蒸馏Student Compensator的CPU推理入口

此目录为可选冻结推理后端，默认Teacher。需要显式提供对应场景的学生checkpoint才替换Compensator；Selector、Master、Anchor Feature、严格K、训练和生产默认不变。

```python
from student_runtime import CpuRenderer, extension

engine = CpuRenderer.from_checkpoint(
    checkpoint_path,       # 冻结Teacher的ply目录
    original_noise_path,   # 原GPU固定随机键的CPU快照，禁止在CPU重新采样替代
    threads=6,
    student_checkpoint=student_path,
)
packet, ids, stats = engine.prepare(cpu_camera, ratio_budget=0.3)
print(engine.compensator_info)

# 实际GPU只消费CPU准备的属性并使用原Rasterizer渲染。
handoff = extension().NativeHandoff(engine.native, engine.sh_degree)
ticket = handoff.prepare(cpu_camera, ratio_budget=0.3)
image = render_prepared(ticket.camera, ticket.wait(), background)
ticket.finish()
handoff.synchronize()
```

`cpu_camera`与原入口相同，4×4相机矩阵和中心均为CPU FP32。消费、wait和finish保持相同Stream；本轮原Rasterizer在默认Stream渲染。

已有CPU对象可直接传`CpuRenderer(model, pc, noise, 6, student_checkpoint=path)`，或传绑定好的`student=student`。两者互斥。加载会验证学生的Teacher/地图/Feature/UID行序指纹；误配直接报错，禁止静默换Teacher。学生输入对象会复制，后续修改不会改变冻结快照。

取消`student_checkpoint`或`student`即使用Teacher。已有`selection_mode`、`comp_batch`等优化开关继续有效。权重、地图或拓扑变化后必须重建快照，不能边训练边更新同一实例。

Replica学生路径模式：

`/home/prmi/zwj/experiments/compensator_distillation_replica_all_20261108/models/Replica__<scene>__<mono或rgbd>/continuous_calibration/best.pt`

各学生只适用于自己的绑定场景，不是跨场景通用权重。需要原项目Python依赖可导入，研究环境使用冻结`source/python`。CPU原生后端直接复用上一轮已编译二进制，不修改训练或重新编译CUDA。

学生存在相对Teacher的蒸馏质量代价，详见本轮报告及逐场景逐Ratio CSV；不能将部署正确性通过当作蒸馏质量等价。此功能未部署生产。
