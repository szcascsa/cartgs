# 低点数渲染研究入口

服务器目录：`/home/prmi/zwj/experiments/sparse_rasterizer_20261109`。

`runtime.py`提供独立、冻结推理用的`SparseRenderer`。不修改正常训练入口、原渲染器或生产默认。传入当前`DeploymentRenderer.prepare`或`prepare_render_data`生成的同一包数据即可；本轮没有蒸馏、修改或重新训练模型。

```python
import sys
sys.path.insert(0, '/home/prmi/zwj/experiments/sparse_rasterizer_20261109')
from runtime import SparseRenderer

# 0：逐字参考展开控制；1：稳定Warp展开；2：共享深度排序＋Tile Bitset。
renderer = SparseRenderer(mode=2)
with torch.no_grad():
    image, radii = renderer.render(camera, packet, background)
```

短Tile列表候选可通过`SparseRenderer(mode=1, short_lists=True)`显式测试。当前该项未显示稳定独立收益，默认关闭；Bitset是低点数下值得保留复测的候选，高密度Full可能变慢。没有按K设置自动分支门槛。

实例的Scratch只复用容量，投影、深度、颜色和Tile成员逐次重算。每个实例仅供同一设备、同一CUDA Stream顺序调用，其他Stream应创建独立实例；返回图像与radii使用独立存储。输入支持原SH或预计算RGB、Scale/Rotation或预计算协方差。`means2D`在冻结推理中不参与梯度。

先执行正确性和内存检查，再执行性能测试：

```bash
cd /home/prmi/zwj/experiments/sparse_rasterizer_20261109
/home/prmi/anaconda3/envs/zwj/bin/python validate.py
/home/prmi/anaconda3/envs/zwj/bin/python test_lifecycle.py
/usr/local/cuda-11.8/bin/compute-sanitizer --tool memcheck --error-exitcode 99 /home/prmi/anaconda3/envs/zwj/bin/python sanitizer_smoke.py
/usr/local/cuda-11.8/bin/compute-sanitizer --tool initcheck --error-exitcode 99 /home/prmi/anaconda3/envs/zwj/bin/python sanitizer_smoke.py
```

空闲GPU正式测速使用新输出目录，不带`--allow-concurrent`。并发诊断已有结果，不需要重复训练：

```bash
/home/prmi/anaconda3/envs/zwj/bin/python benchmark.py --output reports/idle_benchmark_01 --profile
```

当前质量门限为FP32 RGB、radii逐位一致及Tile成员/顺序一致。所有速度均需区分并发诊断和正式空闲GPU数据，也要区分同研究入口控制组和原部署入口。

2026-11-10空闲GPU复测结果单独保存在`reports/idle_retest_20261110/`，技术报告为`SPARSE_RASTERIZER_IDLE_BENCHMARK_REPORT.md`。接受与因外部图形活动剔除的轮次记录在该目录的`COMPLETE.json`及监控文件中。准备＋渲染的完整推理测试另存`pipeline/`，不把纯渲染加速直接当作总推理加速。
