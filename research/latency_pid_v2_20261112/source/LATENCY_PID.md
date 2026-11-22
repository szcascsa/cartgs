# 推理延迟PI/PID控制

该模块仅用于训练完成后的冻结推理。它改变每帧请求预算B，调用原有GPU逻辑计算候选上限和最终K；不修改Selector、Compensator、训练或Rasterizer。既有固定点数及ratio入口保持原样。

候选充足时K=B，网络条件为ratio=K/M。候选不足时，当前后端沿用K=min(B,有效候选数)、ratio=min(1,B/M)，并报告budget_shortfall；控制器保留这个原始语义，不会擅自改用裁剪后的K重新计算网络条件。计时反馈绑定的是实际K及有效候选上限。

## 使用

```python
import torch
from latency_pid import LatencyPIDRenderer

# 在推理模式外按原流程加载冻结engine、rasterizer、相机和background。
# rasterizer.render(camera, packet, background)须返回(image, radii)。
controlled = LatencyPIDRenderer(
    engine, rasterizer, background,
    config=dict(target_ms=0.4, initial_budget=3000,
                min_budget=256, max_budget=8192,
                kp=3.0, ki=12.0, kd=0.0),
    build_directory='/path/to/isolated/build',
)

with torch.inference_mode():
    for camera, host_camera_to_world in views:
        result = controlled.render(camera, camera_to_world=host_camera_to_world)
        image = result['render']
        actual_K = len(result['ids'])
        # result['stats']仍是原推理后端的真实候选数量、ratio和预算状态。

    # 仅在应用需要时读取已完成样本；此函数不会等待GPU。
    telemetry = controlled.poll()

controlled.close()
```

可调用`controlled.set_target(new_ms)`切换目标；待完成的旧目标样本将失效。`render(..., reset=True)`支持应用显式标记相机切换。自动切换检测使用调用方已有的CPU camera-to-world矩阵，不能为了控制器而从GPU下载矩阵。不提供pose时仍可运行，但不会自动检测视角突变。

## 默认参数

| 参数 | 默认 | 含义 |
|---|---:|---|
| target_ms | 0.4 | 所选计时边界下的单图目标；原生Hook与外层API边界见下文 |
| initial_budget | 3000 | 初始请求点数 |
| min_budget / max_budget | 256 / 8192 | 请求预算范围；候选不足时GPU可裁剪至更低甚至零点 |
| kp / ki / kd | 3 / 12 / 0 | 默认增量PI；ki单位为每秒，kd单位为秒 |
| ema | 0.25 | 耗时EMA新样本权重 |
| derivative_ema | 0.2 | 归一化耗时变化率的EMA权重 |
| deadband | 0.05 | 相对目标的允许误差；默认±5% |
| max_increase / max_decrease | 0.08 / 0.25 | 单次有效反馈最多增加8%或减少25%请求点数 |
| max_dt | 0.1 | 积分/微分采用的最大有效采样间隔，避免暂停后出现大跳变 |
| cut_translation | 0.5 | 相邻主机位姿的平移切换阈值，单位与地图相同 |
| cut_rotation_degrees | 30 | 相邻位姿的旋转切换阈值 |
| event_slots | 8 | CUDA Event环容量，范围1–64；每槽两个Event |

参数通过`config`传入，未知键或非法范围立即报错。`timestamp_s`可传应用的严格递增秒时间戳；默认使用C++ steady_clock。传入时间戳须代表实际提交节奏，不能将视频帧编号直接当秒。

## 控制与生命周期

- V2在log(K)上使用增量PI及测量值微分。连续死区后的误差为sign(e)max(0,abs(e)−deadband)；默认关闭D。
- 控制状态直接保存经上下限和每步限速裁剪后的命令，消除固定初始K偏置。死区内保持当前命令；禁止仍超时却因P回弹而加点，或仍低于目标却反向减点。
- integral在V2仅作裁剪后累计状态的诊断，不能与V1的时间误差积分直接数值比较。
- 每个Event样本绑定其frame、epoch、实际K、有效候选上限及提交时间。晚到的时间只反馈其原始样本，不与新帧K错配。
- CUDA Event在构造时分配。完成状态使用cudaEventQuery；未完成就继续使用原控制状态，环满则丢弃该次计时，不等待、不覆盖在途Event。多条已完成反馈只使用最新一条决策；feedback_samples统计收据，superseded_samples记录被更新样本取代的反馈，updates统计控制更新次数。
- 视角切换或目标切换会重置滤波和微分、清空积分，并把当前有效预算作为新基准，减小切换跳动；旧epoch样本仅回收，不参与控制。
- 每实例须串行、单CUDA Stream使用；跨Stream请求会报错。模型、地图、GPU设备、Rasterizer或分辨率改变时新建实例。模块拒绝在相同实例上切换分辨率。
- 异常渲染的计时标记为无效，不会污染反馈。close只销毁Event，不主动等待；应用仍需遵守原Rasterizer输出和GPU内存生命周期。

## 计时和边界

默认通用Renderer使用`start/finish`外层计时，可能包含CPU发射间隙。声明`native_latency_timing=True`的Renderer使用`start_native()`提供的一对Event，在原生图提交处记录begin/end；此时目标只覆盖输入更新及渲染CUDA Graph，不包含Python、输出分配及Graph冷构建。两种计时不能混为同一结果。

V2计时Hook的Mode40独立实现位于`research/latency_pid_v2_20261112/timer_runtime.py`。它只添加Event记录，不改变任何像素运算；Wrapper通过`render(..., timer=(begin_handle,end_handle))`传递句柄。必须在相同device/Stream按顺序记录两端，成功返回前记录end；异常通过finish(valid=False)取消反馈。不要对不支持此协议的Renderer设置该属性。

准备、网络、PID求解及文件输出不计入目标。控制器中没有cudaStreamSynchronize、cudaEventSynchronize或torch.cuda.synchronize；原推理后端已有的finite检查及发布同步完整保留。

既有engine.prepare(camera, budget=...)内部已经在GPU裁剪候选预算，因此控制器不需要先prepare一次获取M再重复prepare，也无需新增GPU控制Kernel。新增工作是C++标量计算、Host API调用及Event记录，仍需在目标设备上单独测量，不能称零开销。

纯PID依赖历史帧。相机突变第一帧、外部GPU竞争或最低预算仍超时，都无法获得硬时延保证。当前默认增益是可用初值，不是对任意设备和场景已验证的最佳值。应根据目标设备和帧率标定，分别报告控制开销、单次调用尾延迟及质量变化。

V1快照位于`research/latency_pid_inference_20261112/`，V2诊断及配对实验位于`research/latency_pid_v2_20261112/`。模块未启用生产默认，也不自动修改训练或Viewer。
