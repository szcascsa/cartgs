# 训练完成后的默认优化推理入口

默认实现为 `JointReleaseRenderer(..., mode=182)`，命令行为 `python/deployment_infer.py`。它使用 checkpoint 中的 Selector、与冻结地图指纹匹配的 Student Compensator，以及保守版 Mode182 Rasterizer。公开入口只接受 `mode=182`，没有启用 Alpha3。

本次切换范围是训练完成后的独立优化推理入口。训练、常规评价和普通 Viewer 的入口保持现状；它们不会自动改用要求 Student 的新接口。

## 使用

从项目根目录运行：

```bash
python python/deployment_infer.py /path/to/finished/ply \
  --student /path/to/matching/student_best.pt \
  --dataset /path/to/dataset --ratio 0.1 --output /path/to/new_metrics.csv
```

`--ratio`、`--budget`、`--full` 三选一。`--frames 0` 使用加载到的全部轨迹帧。CSV 记录逐帧候选 N、实际 K、完整准备＋渲染同步墙钟时间；不保存渲染图片，不覆盖已有 CSV。模块加载、静态预投影和首帧 Graph 建立在预热阶段完成，不混入稳态逐帧时间。

Python 应用将项目的 `python` 目录加入模块搜索路径后使用：

```python
import torch
from joint_release_renderer import JointReleaseRenderer

torch.backends.cuda.matmul.allow_tf32 = False
runtime = JointReleaseRenderer.from_checkpoint(
    '/path/to/finished/ply',
    student_checkpoint='/path/to/matching/student_best.pt',
    mode=182,
)

# pose：当前相机到世界的 CPU 4×4 矩阵。
# calibration：width、height、fx、fy，与保存 checkpoint 的相机口径一致。
with torch.no_grad():
    result = runtime.render_pose(pose, calibration, ratio_budget=0.1)
    # 也支持 budget=2000 或 full=True。
    rgb, ids, radii, stats = (result[k] for k in ('render', 'ids', 'radii', 'stats'))
```

已有 CUDA 相机可调用 `runtime.render(camera, ratio_budget=...)`。低提交开销路径是 `render_pose`：CPU 位姿直接进入 C++，本帧相机与请求统一提交。`full=True` 使用原生 Full 位姿入口及只读冻结属性，不执行 Selector 或 Student。

局部 N 为当前名义可见 Anchor 的 Gaussian 成员数，比例预算为 `floor(ratio*N)`，实际 K 受合资格成员数量限制；固定预算采用相同候选域。缺口记录为 `budget_shortfall`。输入 K、`radii>0` 和真正参与 Alpha 合成的点数含义不同。

## 模型与执行身份

支持已有 joint schema 1–4 的 Residual48 视角条件 checkpoint，包含 `selector.pt`、`selector_config.yaml` 和唯一的 `point_cloud/iteration_*/point_cloud.ply`。Student 为冻结的 32/32 共享主干、四个独立线性 Head；带 View 投影时共 3534 个参数。加载时检查 Student 来源指纹，覆盖模型状态、Feature、Master、Gaussian 行序与相关 schema；缺少或不匹配时明确报错，不替换为 Teacher。

默认网络与调度参数来自已验收的 `new_prepare_182`：

| 部分 | 执行配置 |
| --- | --- |
| Selector | `tensor_selector=2`，FP16 PTX MMA、FP32 累加 |
| Student | `tensor_compensator=11`，FP16 WMMA、FP32 累加，2 warp，16 lane/Anchor View 投影 |
| 准备 | 冻结输入预投影，当前 Ratio/View 每帧重算，动态 N/K 留在 GPU，合并相机和请求提交 |
| 排序 | 严格 K，开启已验收的 small-radix finish |
| Rasterizer | build mode 15、pixel mode 200，对应 Mode182；64 线程预处理 |

Mode182 保留原有 bit-exp 近似和 `T<0.01` 早停，Alpha 下限仍为 `1/255`。它相对旧 Mode182 的新准备调度通过了 ID/RGB/radii 逐位一致检查，**不等于相对原 CaRtGS/F2 无损**。历史同 Student 对照的 18 组验收中，相对原 FP32 Student＋F2 的等权平均 PSNR 损失约 0.00849 dB，最差条件均值约 0.20278 dB，最差帧约 0.30253 dB。这些是已有研究证据，本次代码集成没有重新运行服务器 GPU 验收。

## 生命周期、容量与构建

模型加载时建立私有冻结快照和静态投影；缓存只依赖地图、权重、Feature、拓扑与编码规则。更换其中任一项需重建实例。当前位姿、Ratio、候选、选点及补偿每帧重算，不缓存上帧预测或图像。

同一实例仅在一个 CUDA device/Stream 上串行使用；多请求由调用者串行化，跨 Stream 使用独立实例。返回的 RGB、radii 和 ID 有独立存储，下一次调用不覆盖旧输出。保留 finite 检查、严格预算和 Stream 依赖。

Rasterizer 的共享工作区上限为 1 GiB，其中位表上限为 256 MiB；Graph 计划最多 512 个，容量或分辨率变化时按 native 规则失效重建。`precompile(width, height)` 可提前建立有界计划，超出容量明确报错。它的启动时间与稳态单帧延迟应分别报告。

原生实现随项目保存于 `python/joint_release_native`，67 份 CUDA/C++ 文件与已验收快照逐文件 SHA256 一致，来源记录见 `SOURCE_PROVENANCE.json`。加载器使用这些源码完整构建，不导入研究目录、不链接历史对象文件、不复制研究二进制。首次在目标设备使用时，需要已有 PyTorch CUDA、CUDA Toolkit/NVCC、C++ 编译器、Ninja、GLM 及项目模型读取依赖；构建目录可通过 `TORCH_EXTENSIONS_DIR` 指定。

历史 GPU 证据来自 RTX 3090／CUDA 11.8。新打包入口在本次仅完成 CPU 契约测试和源码身份核验；按用户要求没有进行服务器编译、部署或新二进制测速。
