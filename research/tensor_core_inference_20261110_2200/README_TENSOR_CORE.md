# Selector与学生Compensator的Tensor Core研究入口

只用于训练完成后的GPU推理。原生产、训练、模型及原v31后端不变。源码在`source/`，构建缓存只在本目录`build/`。需要CUDA11.8、PyTorch2.3.1及Ampere或更新GPU，实测硬件RTX3090。

```python
from source.deployment_runtime import DeploymentRenderer

engine = DeploymentRenderer.from_checkpoint(
    teacher_checkpoint,
    student_checkpoint=student_checkpoint,
    tensor_selector=2,
    tensor_compensator=2,
    tensor_warps=4,
    tensor_compensator_warps=2,
)
with torch.no_grad():
    packet, ids, stats = engine.prepare(camera, ratio_budget=0.3)
```

`0`表示原FP32 CUDA，`1`表示TF32 WMMA候选，`2`表示FP16乘法/FP32累加。Selector的模式2使用寄存器MMA，学生补偿模式2使用共享内存WMMA；物理解码和概率仍FP32。可单独关闭两网络Tensor路径；两者0回退原融合实现。它是允许少量数值误差的版本，选点集合可能微变，严格K与资格/行序不变。

静态Selector物理前缀保持开启。加载时保存私有冻结Master、Feature、权重与派生布局；权重、拓扑或地图变化时必须新建实例。只缓存有限派生权重，不缓存旧视角/旧Ratio的预测结果。沿用原Stream与Graph调用约束，同实例按原API串行调用。

首轮大地图继续原host计数流程，网络同样走Tensor实现。用户已允许后续调整调度，当前是否更改由大点数实测结果决定。加载学生必须匹配原Teacher、Master、Feature和行序指纹，不能混用别的地图。

复现构建与验收使用`study.py`、`smoke.py`、`components.py`、`quality.py`、`benchmark.py`、`boundaries.py`。`integrate.py`和`host_wrappers.txt`是早期开发生成器，仅作历史记录，不要重新运行覆盖最终源码。最终源码和补丁、冻结哈希、原始CSV、安全日志及报告为复现依据。

运行需复用已有只读研究依赖和数据路径，见`study.py`；不复制大模型/数据集。所有正式结果及适用边界见中文报告，组件固定输入Graph计时与每帧完整准备分开报告。
