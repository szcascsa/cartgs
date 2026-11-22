# 训练后推理PID预算控制实现与验证

## 结果

已实现独立、可选择启用的PI/PID预算控制模块，并在服务器RTX3090、PyTorch 2.3.1+cu118上编译验证。默认目标是单图纯渲染0.4 ms，默认D=0，即PI；PID增益、预算范围、误差死区、EMA、变化限速和视角切换阈值均可配置。

完成14项C++核心场景、7项GPU测试、24组真实模型等K逐位对照以及office2 RGB-D完整2000帧因果单次渲染回归。选点、补偿网络和Rasterizer未修改，3612个受保护文件SHA不变。未训练、未修改服务器生产目录、未切换默认、未commit/push，未保存渲染图片。

共享GPU上观察到其他活跃计算任务，因而本次2000帧结果仅用于功能回归，不能作为独占条件下的延迟命中率、质量收益或控制器开销结论。目标0.4 ms不代表硬上限；本次也没有把既有离线预算表当成实时控制结果。

## 实际改动

本地仓库增加以下文件，既有代码文件保持原样：

| 文件 | 作用 |
|---|---|
| python/latency_pid_core.h | 不依赖Torch的C++ PID核心，控制log预算，按真实提交时间间隔更新 |
| python/latency_pid_bindings.cpp | 原生CUDA Event计时环、异步Query、帧/预算/epoch绑定、视角切换和Python接口 |
| python/latency_pid.py | 可组合的冻结推理适配器，每帧仅一次prepare、一次render |
| python/LATENCY_PID.md | 中文使用说明、默认参数、生命周期与计时边界 |
| tests/anchor_selector/test_latency_pid_core.cpp | PID核心行为与闭环收敛测试 |

独立服务器研究目录：`/home/prmi/zwj/experiments/latency_pid_inference_20261112/`。源码在source，构建在build，测试和报告分别在tests、logs、reports。服务器mycartgs、realLOD及共享conda未改动。

## 算法与数据流

1. C++控制器异步查询之前已完成的CUDA Event，使用与该Event绑定的实际K、有效候选上限和提交时间更新PID。
2. PID输出本帧请求预算B，现有engine.prepare(camera,budget=B)执行一次。原GPU候选控制决定最终K，保留原来的严格K、索引顺序、选点与补偿语义。
3. 从准备返回的真实K及候选上限建立新计时样本，仅包围Rasterizer一次调用。
4. 返回当前渲染结果；未完成的Event留到后续帧。环满时跳过该帧计时，不等待或覆盖在途样本。

采样误差：`e=(target-filtered_time)/target`。控制量为`log(B)`，包含比例、时间积分和滤波后的测量值微分。误差进入默认±5%死区时保持实际预算；超时最多每次反馈减少25%，富余最多增加8%。积分采用条件抗饱和，不在预算/候选边界继续积累越界方向的误差。

保留当前后端的原始ratio语义：候选充足时K=B、ratio=K/M；候选不足时K=min(B,eligible)，网络ratio仍为min(1,B/M)，而不是擅自改为裁剪后的K/M。计时样本始终绑定实际K。

视角切换由调用方已有CPU camera-to-world位姿判断；不下载GPU位姿，不缓存/复用旧视角Mask。切换目标或检测到相机突变时递增epoch，重置滤波、积分和微分，以当前预算重设基准；旧epoch的在途样本仅回收，不更新控制器。

## 测试

| 验证 | 结果 | 证据 |
|---|---|---|
| C++核心14场景 | 通过 | logs/test_1.log |
| 原生GPU测试7项 | 通过 | logs/test_2.log、reports/TEST_RESULTS.json |
| 真实模型等K对照 | 24/24组IDs、RGB、radii逐位一致 | reports/PAIRED_BITWISE.csv |
| office2完整轨迹 | 2000次请求，2000次控制更新，0次采样环丢弃，0条最终待完成样本 | reports/SCENE_VALIDATION.json、CONTROLLER_FINAL.json |
| 训练隔离 | 启用梯度时拒绝执行 | GPU适配器测试 |
| 源码与模型保护 | 3612个SHA不变 | reports/PROTECTED_BEFORE.json、PROTECTED_AFTER.json |

GPU测试覆盖非法参数、传入配置不被修改、未完成Event导致环满时不等待、切换目标/视角后的旧反馈丢弃、单Stream与时间顺序约束、空候选、异常渲染样本失效，以及一次prepare/一次render。

真实模型使用当前冻结的蒸馏Compensator＋Tensor Core D＋新准备＋Mode40，分辨率2064×2208。逐位对照覆盖请求预算256/2000/4000/8192与0/500/722/1000/1500/1999六个视角。对照固定请求预算以验证相同输入时的输出；正常动态控制会改变预算，本来就不要求与某个固定K图像逐位一致。

2000帧记录中保存1999个逐帧计时快照，控制器内部实际消费了2000个样本；原因是日志读取的是“最新已消费反馈”，一次Query消费多个已完成样本时不会把每个内部样本都复制到Python。本轮没有把该CSV当成完整单次尾延迟统计。

实现初次真实场景验证在加载阶段触发现有模型对Inference Tensor版本计数的限制，测试入口改为no_grad后通过；没有修改模型或缓存检查。失败现场保留在logs/scene_validation.log。

最终编译动态库SHA256：`5fd42102c95d1b3017ae7ac1be2be62b8d91dbc171312777da5714a58e110507`。逐文件实现指纹见reports/IMPLEMENTATION_SHA256.json。

## 开销与适用限制

- PID标量计算在C++，无额外预测网络或GPU计算Kernel；GPU最终预算裁剪复用原路径。
- 计时环默认8槽，共16个预创建Event，显式元数据规模有界；未建立历史视角或网络输出缓存。
- 新模块不调用cudaStreamSynchronize、cudaEventSynchronize或torch.cuda.synchronize；既有准备后端的finite检查和必要发布同步完整保留。测试入口末尾的同步只用于验收。
- CUDA Event记录、Query、C++接口及Python包装仍有开销，没有声称零成本或已达到10微秒目标。独占GPU下的完整额外开销尚未验收。
- PI依赖过去已完成帧，不能提前知道快速转向后的第一帧工作量。外部竞争或最低预算仍超时，不能通过继续积分解决。
- 默认增益是可用初值，不是跨设备、场景、帧率的最优配置。D默认关闭以减少噪声放大。
- 每实例限串行、单Stream、固定模型/设备/渲染器/分辨率；变更这些依赖时重新创建实例。

## 使用与下一项验收

使用`from latency_pid import LatencyPIDRenderer`包装现有冻结engine和Rasterizer，详见python/LATENCY_PID.md。无需替换固定预算入口；构造时可指定独立build_directory加载该模块。

下一项最有价值的验证是在独占GPU、真实显示帧率下对比固定K与PI，记录完整单次请求序列、控制器额外Host/GPU开销、目标命中率、p95/p99、转向后恢复帧数及PSNR。不使用离线最优K或三次中位数替代实时输入；通过后再判断是否需要D或当前视角预测。
