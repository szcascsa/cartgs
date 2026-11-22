# CPU中Ratio直方图选点与6点补偿微核

本目录是训练结束后的冻结推理研究后端，不修改生产默认或训练。上一版`cpu_native_small_20261110_1318`作为配对基线；本轮研究主区间Ratio 0.1–0.5，保留精确直方图选点和6点Compensator微核。

## 使用

在项目原Python推理依赖可导入的环境中：

```python
from mid_runtime import CpuRenderer, extension

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

构造函数`selection_mode=0, comp_batch=4`关闭本轮两项优化；`selection_mode=1`开启256桶精确边界分区，`comp_batch=6`开启6点微核。研究后端默认1/6。两者独立可回退，上一轮小输出/顺序压缩配置保留。

Score、UID及所有非Rotation属性要求逐位一致；CPU编译布局导致最终旋转解码出现微差，验收最大绝对差≤1e−6，并单独验证真实渲染质量。该后端不能称为全属性bitwise一致。

相同实例的请求由C++互斥串行化。输出默认独立；可传有界`output/id_output`。三槽位交接禁止覆盖未完成的ticket；修改网络、Master、Anchor或Feature后必须销毁并重建冻结快照。

只验证了当前Teacher架构和x86 AVX2/FMA硬件。Student包装接口没有在本轮独立验收，ARM设备需另做实现与测量。本轮未修改Rasterizer的Stream假设，正式渲染在默认Stream进行。

`reports/FROZEN_CANDIDATE.json`记录源码/二进制SHA；完整结果、失败候选和实际适用范围见中文技术报告。所有低Ratio和Full慢例均保留，不能把高K收益泛化为所有预算。
