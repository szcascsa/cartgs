# Tensor Core与准备/调度联合研究入口

仅用于训练完成后的冻结GPU推理。生产默认、训练、原checkpoint和已验收网络C未修改。最终证据以`reports/FROZEN_JOINT_CANDIDATE.json`、中文报告与逐样本CSV为准。

正式混合方案的可复用API在`source/`，研究驱动在根目录；旧开发候选在`development_p*`。不要重新运行旧生成器或把失败候选覆盖到最终源码。

```python
import sys
import torch
from pathlib import Path

study = Path('/home/prmi/zwj/experiments/tensor_core_inference_20261110_2200/preparation_phase')
sys.path.insert(0, str(study / 'source'))
from deployment_runtime import DeploymentRenderer
from preparation_camera import NativeCameraPreparer

engine = DeploymentRenderer.from_checkpoint(
    teacher_checkpoint,
    student_checkpoint=student_checkpoint,
    tensor_selector=2,
    tensor_compensator=2,
    tensor_warps=4,
    tensor_compensator_warps=2,
    preparation_frustum=1,
    preparation_projection=1,
    device_schedule=1,
)
camera_preparer = NativeCameraPreparer(MiniCam, engine._ext)
with torch.no_grad():
    camera = camera_preparer.create_from_pose(width, height, fov_x, fov_y, current_camera_to_world)
    packet, ids, stats = engine.prepare(camera, ratio_budget=0.3)
```

依赖项目原`python/`和蒸馏框架的`compensation_student.py`，必须与训练出的同场景学生匹配；研究脚本`prep_study.py`已指定既有只读路径。二进制为CUDA11.8、PyTorch2.3.1、Ampere RTX3090实测；其他设备需重建与重新测速。

`preparation_frustum=1`为普通CUDA融合点积、半径、可见性和计数；`0`为原路径。TF32视锥因改变候选Mask已排除，正式选项不接受`2`。

`preparation_projection=1`预计算第一层的冻结Offset/Base/Residual线性投影，当前Ratio/View组合每帧重新计算；`0`关闭。未跨越ReLU。FP16隐藏层和FP32重排存在小误差，按图像质量门限验收，不声称逐位渲染等价。

`device_schedule=1`把大地图候选N、预算K和控制留在GPU，返回前保留最终finite检查；`0`恢复既有地图规模调度。原Graph容量与高Ratio普通发射仍保留，按通用容量组织。

三个新选项均为独立显式开关；本研究模块默认新开关为0，以便关闭时对照。两网络Tensor模式仍可各自设0恢复原FP32融合计算。只有准备/调度开启、网络为FP32的组合也已单独测量。

新相机是一般4×4逆矩阵，不假设严格正交；`current_camera_to_world`必须是当前帧CPU 4×4位姿。保留奇异和非有限值检查，一次Pinned 140字节上传由ATen记录Stream事件。固定内参可缓存，不能缓存旧位姿/旧视锥/旧预测。默认非并发调用同一个实例，跨Stream调用须按原API串行建立依赖。

Cache容量为`N×32+(L−1)×Anchor×32`个FP32静态数和`Anchor×32`当前工作区；严格有限。切换准备选项会清除Graph并重建投影。训练、权重更新、地图/Feature更新、Clone/Split/Prune/重排或Anchor迁移后必须重建整个冻结实例，不能继续使用旧Cache。

复现顺序：确认保护哈希与无重复任务 → `prep_study.py`编译 → `boundaries_preparation.py` → `quality_preparation.py` → 冻结参数 → `formal_controller.py`双轮配对 → 组件、安全、迁移及Profile → `analyze_preparation.py` → 图表/完整性/报告。

`inline_projection.py`、`projected_register_bench.py`是失败候选对应的历史驱动，只能针对各自留档源码/二进制运行。正式版本已撤回这些无稳定完整收益的实现。
