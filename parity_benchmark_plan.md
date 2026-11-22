# Anchor / Selector / R3 CUDA Parity与性能验收基线

版本：`4269d7e3bad6cf5af7e1df1763634b61e98e0b51`，2026-10-22。本轮未实现CUDA候选；本文件定义下一轮门控。

## 1. 已保存的真实fixture

目录：`tests/diagnostics/anchor_selector_r3_performance_20261022/evidence/`。

| 文件 | 内容 |
| --- | --- |
| `native1/Full_parity.pt` | 六组Master、相机/GT、Full图像/loss/grad、Adam前后状态 |
| `native1/lod_0.010000_parity.pt` | 1% Cached，N15891，实际K156 |
| `native1/lod_0.050000_parity.pt` | 5% Cached，实际K799 |
| `native1/lod_0.100000_parity.pt` | 10% Cached，实际K1582 |
| `native1/lod_0.150000_parity.pt` | 15% Cached，实际K2368 |
| `anchor_input_buffers.pt` | 原Selector所有buffer，含owner、center、offset_scale、levels、keys、UID、成员数、slot状态等 |
| `native1/native_metadata.json` | 命名参数映射、shape/stride/device/dtype、真实相机、重放协议 |
| `fixture_verification.json` | 789项重复比较、62个固定状态Adam检查；保留超容差项 |
| `fixture_details.json` | 层级/成员分布、C末层精确0、Adam放大误差位置与数值 |
| `native_profile.sqlite/.nsys-rep` | 独立Profiler原始证据，非正式计时样本 |
| `native1/native_stages.csv`、`native2/native_stages.csv` | 两次正式计时，单case60次有效计时、10次warmup |
| `trace1/trace2/mapping_trace.csv` | 真实在线每Mapping host scope；含拓扑事件 |
| `lifecycle_private.log`、`lifecycle_mapping.log` | 当前原生生命周期回归通过 |

原始真实checkpoint本地已存在于上一轮`tests/anchor_selector/private_compensation_online_20261022/evidence/models/r3_independent14_seed2026/`，远程为`/home/prmi/zwj/private-compensation-online-20261022/r3_independent14_seed2026/7465_shutdown/ply/`。不将其与旧13,919/5,177地图混用。

fixture的参数编号：0 XYZ、1 SH DC、2 SH rest、3 opacity logit、4 log_scale、5 raw quaternion；其后按`native_metadata.json.parameters`排列Selector/Anchor/R3参数。四张R3表按各自编号保存；没有grad的参数不写该gradient字段，不得解释为一张全零梯度。

每组还保存：Gaussian UID、owner、cached uniform、ratio、Input27、h32、概率、STmask、selected ids、14D残差、变换后属性、view/projection、radii、render、loss、所有定义的grad、before/after m/v/step/params。Full没有Input27或LoD残差是正常现象。

C++ `OutputArchive`文件可通过Python `torch.jit.load(path,map_location='cpu').state_dict()`读取。新增buffer补充包为TorchScript buffer archive。fixture CPU拷贝不属于正式计时；Nsight排除warmup及fixture那一轮。

## 2. 必须分别验收的四层

### 层A：纯函数Forward

- Anchor输入pack逐列、MLP h/logits/p、ST正向、14D decode、变换后所有Gaussian属性与原函数比较。
- 精确：owner、UID、level/key、cached uniform、mask、selected ids、K、slot eligibility和成员数。
- FP32网络起始参考：`atol=1e-7, rtol=1e-6`；属性通用初始门槛：`atol=2e-5, rtol=2e-4`，同时报告max/RMS/p95误差。
- p在uniform附近须单列统计distance-to-threshold。即使p满足allclose，集合变了也不能宣称等价；需消除误差来源或将该候选判为不通过。
- Quaternion同时比较原raw输出及物理旋转。等价符号不能掩盖后续不同raw残差/optimizer行为。

### 层B：固定上游梯度VJP

- 使用相同的保存d-output，不让Renderer非确定性混入MLP/属性VJP测试。
- 比较六组Master、Anchor Feature、Encoder、Selector、当前R3表梯度；C冻结且R3照片不经C回Encoder。
- 非当前ratio表grad必须None；Full下Selector/Anchor/R3的grad必须None。不能用零梯度伪造skip。
- 重复owner Feature梯度需求和，ids散射回原Gaussian行；默认FP32`atol2e-5/rtol2e-4`，报告接近零梯度的绝对误差。
- 新增fixture：重复owner、非连续stride、极大Offset、K0/K1/全选、q范数clamp、exp/sigmoid饱和、零/非零C。

### 层C：固定gradient与Adam状态

- 使用保存的参数/grad/m/v/step，逐叶LR/epsilon/betas与base Adam完全一致；检查parameter、m/v、step、version和注册alias。
- dense语义：本轮未选中但曾有moment的行仍可能更新；非当前ratio表gradNone完全skip。两个情况都要独立fixture。
- 比较至少连续20次真实step，不调用伪造零grad step来凑clock。
- append/clone/split新行value继承或零初始化、UID新建、moments零；prune/reorder保持旧行moments/step；几何transport只清受影响行。
- 当前62组固定状态公式检查通过；未来候选必须在相同输入下通过，不能用端到端原子误差作为Adam纯函数放宽理由。

### 层D：真实Renderer / 短程Mapping

- 固定同相机/GT、相同ids与K，分别比较Full和四LoD的render/photo/PSNR/SSIM，必要时LPIPS；输出逐视图误差。
- 重复baseline→baseline、candidate→candidate、baseline→candidate，区分Renderer atomic的误差包络。
- 当前native-repeat一个Scale更新分量误差`7.71e-4`超过通用容差，保留`end_to_end_allclose=false`。其极小梯度/累积moments受eps1e-15放大，不代表允许更宽泛的候选误差。
- 如端到端误差超过baseline-repeat包络，暂停候选；不能改Loss、epsilon或选集来通过。需要精确隔离时，回到层B/C的固定梯度。
- 最后执行相同训练预算/ratio顺序/seed/Full step的短程原生产训练；比较Full和same-K LoD，记录实际Topology/Adam clocks。若地图轨迹无法严格一致，明确不做纯性能因果结论。

## 3. 生命周期矩阵

| 操作 | 必验 |
| --- | --- |
| 新观测append | 同层occupied跳过重复创建；新点零残差；Feature/owner/UID/统计对齐 |
| clone | parent残差value继承，新UID，新增m/v为0，Anchor容量不超10 |
| split | `R diag(scale)`扰动；两子点替代父，净增1；残差/identity/optimizer同步 |
| prune/reorder | Master六组、所有R3表、m/v、owner、metadata使用同keep/order |
| Anchor growing/prune | 层级与固定原点，Feature max/zero规则；原opacity/访问窗口严格不等式 |
| local correction | 同层接收容量、满10丢弃、Feature成员数加权，Gaussian UID跟点走 |
| Sim3 / loop | XYZ/Quaternion残差transport、受影响m/v重置；完整事务后发布snapshot |
| save/load | 四张表/UID/Anchor/slot/schema和原ratio插值可恢复；不宣称现有导出含完整Adam |
| 非默认stream | Tensor准备→所有算子→backward→step建立event依赖；两stream不互读未完成状态 |

旧的private compensation/variable mapping测试本轮已重跑；未来新kernel不仅要通过小fixture，还要通过真实地图归属和ID检查。

## 4. 正式性能验收

1. 原实现与候选使用同GPU、同软件/编译flags、同输入fixture、相同N/K、同Loss/grad/clock；代码及fixture均有SHA。
2. 必须先记录/确认GPU无其他计算进程，固定clock/power条件或完整记录。当前审计有外部持续负载，因此尚无独占速度验收结果。
3. Profiler与计时分开：10～50 warmup，至少3轮交错原/候选，每轮>=100样本，报告mean/median/p95/p99和重复波动；CUDA Events与同步wall同时记录。
4. 单独记录H2D/D2H、CPU API wait、kernel纯执行、GPU queue、allocator请求/真实cudaMalloc、Forward/Backward/Adam；不拿异步CPU调用时长当kernel耗时。
5. 覆盖1/5/10/15%、Full、在线五槽循环、Tail Full+LoD、Topology事件；Topology子段单独计时，避免均值掩盖卡顿。
6. 真实15,891点优先；10K/50K/100K合成压力只用于规模趋势，不能冒充真实地图质量/在线占比。
7. allocator峰值使用对应runtime的实际计数器；nvidia-smi采样仅是下界，不能称精确peak。额外记录参数/m/v/grad和workspace字节。
8. 只有完整Mapping与独立模块都稳定改善，且所有parity门控通过，才建议替换生产；小MLP微基准更快不足以验收。

## 5. 复现与可移植性

隔离工具目录包含`build_native.py`、`run_native.py`、`run_online.py`、`stress_and_verify.py`、`fixture_details.py`、`analyze.py`、`make_outputs.py`。

远程脚本固定当前已验证source/build路径，复用编译flags/link；迁移环境时显式改路径并重新SHA核对。LibTorch ABI1与Python Torch ABI0使用不同进程，禁止通过LD_LIBRARY_PATH把两者混载。

`run_online.py`拒绝覆盖已有run输出；复跑用新的隔离ROOT。基准过程只恢复隔离重放中的参数值，真实moment步骤仍执行，未改生产默认配置。

原始证据manifest验证由`analyze.py`执行，附加buffer与生命周期日志在交付验证中另外计算SHA；生产源码最终tracked diff必须仍为空。
