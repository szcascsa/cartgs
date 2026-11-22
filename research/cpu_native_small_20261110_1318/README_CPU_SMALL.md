# CPU小输出与选点顺序压缩后端

本目录是训练结束后的冻结推理研究后端，不修改生产默认或训练。上一版`cpu_native_deep_20261110_1222`作为配对基线；本轮只保留小输出精确微核和等价顺序压缩。

## 使用

在项目原Python推理依赖可导入的环境中：

```python
from small_runtime import CpuRenderer, extension

# noise_path必须是离线保存的原GPU固定随机键，不能在CPU重新采样代替。
engine = CpuRenderer.from_checkpoint(checkpoint_path, noise_path, threads=6)
packet, ids, stats = engine.prepare(cpu_camera, ratio_budget=0.1)

# 可选：原生有界三槽位交接。本帧51个相机FP32与属性同包上传。
handoff = extension().NativeHandoff(engine.native, engine.sh_degree)
ticket = handoff.prepare(cpu_camera, ratio_budget=0.1)
gpu_packet = ticket.wait()
image = render_prepared(ticket.camera, gpu_packet, background)
ticket.finish()  # 与wait/消费使用同一CUDA Stream。
handoff.synchronize()
```

`cpu_camera`需提供原入口的`world_view_transform`、`projection_matrix`、`full_proj_transform`、`camera_center`及图像尺寸/FoV/近平面等元数据；矩阵和中心均CPU FP32。RGB光栅化仍使用项目原GPU实现。准备耗时不含磁盘加载、相机矩阵构造、显示。

## 独立开关与回退

构造函数的`small_kernel=False, ordered_mode=0`关闭本轮两项优化；其余配置继承上一轮。`ordered_mode=1/2/3`分别为串行压缩、并行压缩和按工作量选择。正式候选默认`small_kernel=True, ordered_mode=3`。

相同实例的请求由C++互斥串行化。输出默认独立；可传有界`output/id_output`。三槽位交接禁止覆盖未完成的ticket；修改网络、Master、Anchor或Feature后必须销毁并重建冻结快照。

只验证了当前Teacher架构和x86 AVX2/FMA硬件。Student包装接口没有在本轮独立验收，ARM设备需另做实现与测量。本轮未修改Rasterizer的Stream假设，正式渲染在默认Stream进行。

`reports/FROZEN_CANDIDATE.json`记录源码/二进制SHA；完整结果、失败候选和实际适用范围见中文技术报告。所有低Ratio和Full慢例均保留，不能把高K收益泛化为所有预算。
