# 原始 Transform Field CUDA backend 实现与验收报告

状态：CUDA实现及分阶段验证已完成；完整在线质量验收尚未通过，默认 backend 仍为 legacy。
正文最终在线、benchmark与Nsight数据对应修正版`462d53f`；`839a1e1`旧数据保留在`before_reciprocal_fix/`及不含`reciprocal_fix`前缀的证据文件，不能与最终表混用。
2026-10-14用户明确停用GradientConstraint；fast配置及最终在线A/B均关闭该模块。
此前Guard开启的在线结果保留为历史诊断，不用于最终生产性能与质量结论。
所有数据来自目标服务器 RTX 3090 / CUDA 11.8 / LibTorch 2.3.1+cu118；未用静态检查替代 CUDA 实测。

## 1. Git 与实现范围

分支 `cuda-tf`，原始基线 `99a9bb1f05e458617f50fabb30a910f14aaf7c9c`。
已完成提交：

- `6a2c587`：fused HexPlane query 与 parity。
- `fbc29a9`：cuBLAS / warp32 两种 forward 候选。
- `51516c9`：完整 neural / output backward 与 autograd bridge。
- `2516890`：CUDA regularization。
- `9c52383`：完整训练口径 benchmark / Tensor materialization / Nsight。
- `a39bdae`：多 tensor CUDA Adam direct 与连续多步 parity。
- `876153d`：CUDA Adam prepare/commit/discard 与混合 Guard adapter。
- `77b3d7c`：完整成本复测、最终 Nsight 与生产 MLP 策略选择。
- `598a3c5e1a65bc81a8d5836a01c480c0bbe2da3b`：Mapping / CD / DSD / Replay / checkpoint API 接入，主工程测试。
- `62ab7f9ade068e361628db1bd758dde007978eb3`：无尾训在线验收 executable、完整图像评估与细分 MLP backward benchmark。
- `839a1e1`：停用GradientConstraint的最终协议、批量逐帧指标验证、独立在线TF计时、固定输入训练链与legacy复跑工具。
- `462d53f`：mean backward对齐ATen的FP32 reciprocal运算顺序，增加near-zero Adam regression与修正后完整复验。

主要文件：`include/transform_field_cuda.h`、`src/transform_field_cuda.cpp`、
`src/transform_field_cuda.cu`、`src/transform_field_cuda_neural.cu`、
`src/transform_field_cuda_regularization.cu`、`include/transform_field_cuda_optimizer.h`、
`src/transform_field_cuda_optimizer.cpp`、`src/transform_field_cuda_optimizer.cu`。
阶段测试与原始结果：`tests/diagnostics/cuda_tf_implementation_20261013/`。
原审计材料保留；未 push；远程没有有效仓库 HEAD，不能把本地 commit 当远程 Git 状态。

生产与高层接入修改文件：

| 范围 | 文件 |
|---|---|
| CUDA计算 | `include/transform_field_cuda.h`、`src/transform_field_cuda.cpp`、`src/transform_field_cuda.cu`、`src/transform_field_cuda_neural.cu`、`src/transform_field_cuda_regularization.cu` |
| CUDA optimizer | `include/transform_field_cuda_optimizer.h`、`src/transform_field_cuda_optimizer.cpp`、`src/transform_field_cuda_optimizer.cu` |
| 原TF API/config | `include/transform_field.h`、`src/transform_field.cpp` |
| mixed optimizer/事务 | `include/constrained_adam.h`、`src/constrained_adam.cpp`、`include/gradient_constraint.h`、`src/gradient_constraint.cpp` |
| 生产入口 | `src/gaussian_mapper.cpp`、`src/change_distill.cpp`、`src/drift_stability.cpp` |
| 构建/用户停用配置 | `CMakeLists.txt`、`cfg/selector/fast.yaml` |
| CUDA回归测试 | `tests/cuda_transform_field_test.cpp`、`tests/cuda_tf_neural_test.cpp`、`tests/cuda_tf_backward_test.cpp`、`tests/cuda_tf_regularization_test.cpp`、`tests/cuda_tf_adam_test.cpp`、`tests/cuda_tf_preview_test.cpp`、`tests/cuda_tf_api_test.cpp` |
| 既有流程测试接入 | `tests/change_distill_test.cpp`、`tests/drift_stability_test.cpp`、`tests/joint_replay_test.cpp`、`tests/tf_backend_test_options.h` |

验收driver、profiler、所有CSV/JSON及报告表在诊断目录内，未改原`examples/tum_rgbd.cpp`、Renderer/Loss/Selector/Protection算法。
文件级SHA256核对记录将随最终验收保存；服务器源码通过SHA与本地对应，不依赖其无效Git HEAD。

## 2. 算法与存储

保留原始 fast：L=2、每级6 planes（XY/XZ/XE/YZ/YE/ZE）、C=8、F=16、H=32、7 Linear、dx/ds/dq=3/3/4，总参数1,186,762。
所有参数与 AABB 继续是原注册 Tensor；未改变名称、shape、NCHW / Linear row-major layout。
未加入 tanh、bound、ratio encoding 或混合精度。生产调用与 checkpoint 流程已通过目标机器测试；完整在线质量验收见第9节。

## 3. Forward / backward 数据流

Query forward 在一个 kernel 中完成 AABB normalize、12 planes border bilinear sampling、逐 channel 六项 product 和 level concat。
Query backward 重算采样，使用 prefix product 与反向乘积链；plane gradient 使用 atomicAdd；XYZ / Tensor ratio 梯度累加全部贡献。
保留 exact boundary 的 grid_sample derivative，未使用 product/sample。

MLP forward 同时实现7 SGEMM + fused bias/ReLU，以及 H32 的一 warp/Gaussian 候选。
MLP backward 使用 cuBLAS dX/dW、CUDA db reduction、三分支 shared gradient 累加。
Output CUDA 保留 activated scale / log-scale 两入口、wxyz 左乘与两次 clamp-normalize，并返回全部输入梯度。
cuBLAS 使用当前 handle / stream 与 PEDANTIC_MATH，调用后恢复 handle 设置。
CUDA 编译禁用隐式 FMA，必要处显式 fmaf 对齐原 ATen 运算顺序。
生产CUDA wrapper检查CUDA/FP32/device/shape/contiguity，使用CUDAGuard、current CUDA stream与launch check。
生产代码不执行全device synchronize；显式同步只出现在诊断benchmark中。

薄 autograd bridge 分为 query、neural、output、regularization；SavedVariable 检查参数、AABB、输入版本。
未使用的 head 保留 undefined gradient，避免错误推进 optimizer state；Gaussian identity 梯度通过 autograd 累加。
暂不支持 higher-order derivative，此项目验收使用一阶 VJP。
低层CUDA不强制mature_only/outside_identity；applyResidual/applyLogResidual和上层compact/scatter mask保持原有分工。
默认detach_gaussian_parameters=0时，XYZ、activated scale、base quaternion梯度完整返回；Full已存在的Gaussian梯度正常累加。

## 4. Regularization 与保存策略

Regularization forward 为 partial reduction + scalar reduction 两个 kernels，backward 为一个 stencil kernel。
保持沿 dim2 / height 的先 d1 再 d1 差分、逐 plane mean 与原 weights。
N=0 query 不产生 plane edge，但独立 regularization 仍训练 planes。

保存 feature、共享 ReLU 与三个 head ReLU；不保存 plane grid/sample/product 链。
Backward 重算 normalized coordinates、corner indices/weights 与 plane samples。

## 5. Parity

- Phase A：feature / XYZ / ratio / 每 plane 梯度通过；随机、checkpoint、边界、zero sample、非默认 stream。
- Phase B/C：381 条完整 backward 检查通过，并在 C 数值修正后重跑 B forward。
- Quaternion near-eps 的归约与 q*q 两条梯度 edge 按原 FP32 链实现；exact zero 保留原 NaN gradient，未新增保护。
- Phase D：362 条检查通过；正则 gradient 使用更严格 atol=2e-10、rtol=2e-4。
- Phase F1：311 条 p/m/v 检查最大绝对误差均为0；1/10/37/100步的 per-tensor step 一致。

F1 覆盖 grad=None、全零 gradient、LR0、weight decay、plane state reset、eps1e-15/1e-8、独立 normal/CD/DSD optimizer、非默认 stream。
最新 F1 接入后复测共373条检查，p/m/v最大绝对误差仍全部为0；初版311条结果单独保留。
完整 CD/DSD 生命周期与实际 Mapper Replay 入口测试也已通过，见第9节。

| Parity 证据 | 检查数 | 最大绝对误差 | 结果 |
|---|---:|---:|---|
| Query A | 493 | 8.34465e-7 | 全部通过 |
| Forward B，C修正后重跑 | 207 | 1.90735e-6 | 全部通过 |
| Backward C | 381 | 7.62939e-5 | 全部通过；最大值为随机MLP权重梯度，满足rtol=2e-4 |
| Regularization D，修正后 | 374 | 1.49012e-8 | 全部通过；包含near-zero Adam回归 |
| Adam F1，生产接入后 | 373 | 0 | 全部通过，逐tensor step一致 |
| Guard Preview F2 | 136 | 0 | 全部通过 |
| 公开TF API | 1810 | 9.53674e-7 | 全部通过 |

先测旧实现自身重复梯度：A最大4.47035e-7、C最大2.21189e-9、D为0。
各case的atol/rtol与实际误差保存于对应parity CSV；未通过放宽起始容差掩盖错误。

## 6. CUDA event microbenchmark

均为100次预热、500次测量；完整 p50/p95 见各阶段 CSV。真实输入 N=8491。
此表是独立 TF probe，使用原审计 loss / LR / Adam 协议，不是在线 Mapping wall time。
修正前的 benchmark 保留在 `before_reciprocal_fix/`，以下只列 `462d53f` 修正版数据。

独立阶段 N=8491 的修正版 mean：query forward 0.361638→0.035994 ms，query backward 1.44491→0.197406 ms；
total backward 2.81709→0.492716/0.488135 ms；regularization forward+backward 6.61752→0.107540 ms；
Adam direct 0.995019→0.065272 ms。独立阶段不可直接相加替代 total 测量。
最终修正版 Custom optimizer 后54组合全部finite；N=8491同机重测：

| 测量项，mean ms | legacy | cuBLAS CUDA | warp32 CUDA |
|---|---:|---:|---:|
| core，CUDA Adam | 5.06206 | 0.825000 | 0.771828 |
| regularized，CUDA Adam | 11.3292 | 0.939813 | 0.892774 |
| regularized CUDA preview | 12.5122 | 1.15662 | 1.13007 |

生产策略：H32/F16且N<=10000选择Warp32，其他选择cuBLAS。
理由：Warp32在1k/5k/8491/10k完整成本较低，cuBLAS在20k/50k更快；H64继续cuBLAS。

最终regularized训练总成本，单元格为mean / p50 / p95 ms：

| N | legacy | CUDA cuBLAS | CUDA warp32 |
|---:|---:|---:|---:|
| 1000 | 11.2323 / 11.2364 / 11.4330 | 0.925311 / 0.920576 / 0.957440 | 0.869345 / 0.869376 / 0.888832 |
| 5000 | 11.3083 / 11.2957 / 11.5364 | 0.948359 / 0.943104 / 0.991232 | 0.890671 / 0.891904 / 0.933888 |
| 8491 | 11.3292 / 11.3408 / 11.5313 | 0.939813 / 0.935936 / 0.980992 | 0.892774 / 0.892928 / 0.930816 |
| 10000 | 11.3216 / 11.3234 / 11.5999 | 0.970451 / 0.967680 / 1.00352 | 0.923187 / 0.919552 / 0.959488 |
| 20000 | 11.3523 / 11.3234 / 11.5200 | 1.09082 / 1.08851 / 1.11206 | 1.32393 / 1.32301 / 1.33530 |
| 50000 | 12.0009 / 11.9880 / 12.1610 | 1.72312 / 1.72134 / 1.75104 | 2.70363 / 2.70234 / 2.72691 |

全部测量项的mean/p50/p95原始CSV按阶段保留：A的query forward/backward、
B/C的MLP/output/total forward与total backward、D的regularization forward/backward、
F1/F2的Adam direct/preview、Phase3/5的core/regularized/preview总成本。
独立阶段测量与完整训练中的阶段时间受launch gap、缓存与调度影响，不能将前者逐项相加作为后者。
Warp32的独立forward慢于cuBLAS，生产选择依据完整训练总成本，未只挑forward结果。

全部876行mean/p50/p95数据集中列于[benchmark完整表](tests/diagnostics/cuda_tf_implementation_20261013/BENCHMARK_TABLES.md)。
该表覆盖query forward/backward、MLP forward/backward、output、total forward/backward、
正则forward/backward、Adam direct/preview，以及全部N的core/regularized/preview总成本。
补测的N8491 MLP reverse-only：legacy=0.575428/0.567296/0.635904ms，CUDA=0.167856/0.165888/0.178176ms（mean/p50/p95）。
MLP backward两种CUDA forward候选共用cuBLAS反向；补测内含gradient parity，exit=0。

## 7. Nsight 与 GPU memory

最终修正版同机采集100预热后10个iteration，CUDA/NVTX、无CPU sampling。修正前的初步 trace 保留在 `before_reciprocal_fix/`，不与以下数据混用。

grid_sampler=0，CatArrayBatchedCopy=0；D2D copy 156→0/step。
大量旧Adam短kernel仍存在，不能仅以 `<10us` 比例判断收益。
Nsight active/span/gap 与 memset 原始统计保存在 summary；插桩 wall time 不作为无插桩性能。
Query backward183 registers，warp32 MLP48，regularization backward34；无local/stack spill。
N=8491 regularized peak allocated：57,263,104→48,643,584 bytes。
reserved 包含 allocator 先前缓存，不能作为独占显存。
最终CUDA preview peak allocated为61,564,928 bytes，高于同口径legacy的57,263,104 bytes，原因是独立prepared参数/moments缓冲；未隐藏这一内存代价。
用户已停用GradientConstraint，因此最终生产对照使用direct路径；preview仍完成兼容测试。
F后最终Nsight：regularized Adam total kernels legacy/cuBLAS/warp32=991/94/81；preview=1043/96/83。
CUDA direct optimizer=1kernel，prepare/candidate/commit=3kernels；两候选grid_sampler/CatArrayBatchedCopy/D2D copy均为0。
Warp32 direct的<10us比例92.47%，legacy96.92%；raw trace保存在服务器，summary下载到诊断目录。

最终direct trace，按NVTX范围归属（每step）：

| 项目 | legacy | CUDA cuBLAS | CUDA warp32 |
|---|---:|---:|---:|
| forward kernels | 81 | 17 | 4 |
| backward kernels | 602 | 66 | 66 |
| regularization forward NVTX范围kernels | 93 | 3 | 3 |
| optimizer kernels | 208 | 1 | 1 |
| loss kernels | 7 | 7 | 7 |
| total kernels | 991 | 94 | 81 |
| <10us比例 | 96.9223% | 94.3617% | 92.4691% |
| D2D copies | 156 | 0 | 0 |
| memsets | 10 | 0 | 0 |
| GPU device work active，ms | 4.14026 | 0.694064 | 0.832479 |
| 插桩iteration span，ms | 18.7416 | 1.49246 | 1.40333 |
| span gap比例 | 77.9022% | 53.3820% | 40.3791% |

NVTX regularization范围包含loss接入的add；其反向kernel计入backward行。
CUDA正则自身是两个forward与一个backward kernel，不能将范围统计误当独立完整正则cost。
上述active/span/gap均为Nsight插桩统计；不作为CUDA event microbenchmark或无插桩在线wall time。
Preview对应total kernels为1043/96/83；全部六份summary保留每阶段active、span、copy、memset、runtime API与top kernels。

## 8. Optimizer / Guard 事务

CUDA direct 使用原 optimizer.state() 的 AdamParamState 与原注册参数，逐 tensor step；无gradient tensor完全不动。
Moment 运算与 addcdiv FP32 顺序已对齐 LibTorch2.3.1；fast26参数一个 metadata kernel。
CUDA prepare/candidate/commit/discard 与混合group adapter已通过136条bitwise检查；实际Guard accept/shrink/project/fallback/reject决策与旧版一致。
非法/过期candidate在任何live写入前拒绝，F1与原CPU Guard测试重跑通过。Mapping调用已接线，主工程rebuild及41/41 CTest通过。
独立preview/candidate/commit：legacy=2.18247ms，CUDA=0.239213ms。
prepare读取live p/grad/m/v/per-tensor step，生成独立proposed/delta/m/v/step；live在质量决策前不变。
delta使用FP32实际proposed-p，Guard只修改candidate参数方向与幅度；prepared moments始终来自原gradient，不能乘alpha或重算。
non-TF groups继续旧preview，TF groups使用CUDA preview；所有groups共用一次质量决策，统一commit/discard。
direct一个批量kernel，preview的prepare/candidate/commit三个kernels；no-grad不推进，zero-grad与LR0仍推进state。
注册参数identity保持；原Torch state map支持copy/snapshot/affected-plane reset；raw CUDA参数写入后递增version。
未启用AMSGrad；该扩展选项被显式拒绝，不静默更改算法。
原 shared normal Adam checkpoint resume 缺口保持原状，未在本轮修复。

## 9. 特殊流程与在线验收

公开TF API 1810条检查通过，max_abs=9.53674e-7；包括activated/log-scale、outside、重复索引、teacher、旧checkpoint加载与新save/legacy-load、非默认stream。
既有完整ChangeDistill与DriftStability生命周期套件已在CUDA backend运行通过；不将微测试的独立state当完整流程替代。
主工程完整rebuild成功，41/41 CTest通过，包括实际Mapper Replay、Protection、Guard、CD/DSD与旧CPU套件。
LOCAL / GLOBAL / FIELD_DOMAIN继续使用同一TF API；teacher冻结拷贝、student CUDA backward、fresh CD optimizer与normal state隔离。
FIELD_DOMAIN resample仍保留原LibTorch算法，affected plane state按原Torch state map擦除；MLP与unaffected plane state保留。
Replay保持normal参数/state/LR与原clock；DSD保持临时eps1e-8、无进展回滚；normal/CD维持eps1e-15。

最终生产对照为GradientConstraint关闭的四场景8次Mapping，以及用户授权的desk共同start=1000两次Mapping。
全部10次退出码0；Full与LoD 1%/5%/10%/15%共50个正式评估全部退出码0、指标finite。
long/room/desk/xyz分别完整使用2515/1352/592/3397个association帧，所有对应评估使用相同数量的帧。
旧Guard开启批次保存在`results/cuda_tf_online_20261013`及`results/cuda_tf_online_desk_start1000_20261013`，不用于以下结论。

两边使用相同fast architecture、Selector/Protection/CD/Replay/DSD、loss、ratio、数据与SLAM配置。
room使用有效的dataset内association与fr1相机标定；两边相同。原先空association导致的失败日志保留。
单独验收main在SLAM shutdown时发出stop，完成在途iteration后保存，未更改iteration频率、预算或原生产example。
全部`transform_state.json`确认`post_slam_mappings=0`、`tail_completed_steps=0`。
默认detach_gaussian_parameters=0、Training.ultra_light_enabled=0，未关闭任何其他训练模块。

### 9.1 原生产Mapping与Full质量

| 场景与TF start | Mapping mean ms，legacy→CUDA | p95 ms | cycles | committed TF updates | Full PSNR dB | SSIM | LPIPS |
|---|---:|---:|---:|---:|---:|---:|---:|
| long，3000 | 21.1370→15.7077 | 43.2063→29.7352 | 5490→7382 | 2490→4382 | 20.7735→18.1037 | 0.7337→0.6527 | 0.2759→0.3875 |
| room，3000 | 15.2978→13.1164 | 35.1376→28.2088 | 3697→4293 | 697→1293 | 16.1631→16.7399 | 0.5981→0.6122 | 0.5707→0.5440 |
| desk，3000 | 11.2682→11.2429 | 18.5124→20.2651 | 2175→2172 | 0→0 | 18.1454→18.1066 | 0.6618→0.6617 | 0.4501→0.4506 |
| xyz，3000 | 17.6246→10.4966 | 25.7340→12.9060 | 8113→13747 | 5113→10747 | 21.3604→21.9259 | 0.7180→0.7285 | 0.2855→0.2630 |
| desk补充，1000 | 16.1096→12.5594 | 33.4987→23.4619 | 1503→1938 | 503→938 | 16.6386→18.1352 | 0.6139→0.6592 | 0.4895→0.4544 |

上述mean/p95来自原验收executable的完整host Mapping cycle，保留既有每19轮CUDA event采样，未增加同步。
场景整体进程wall主要受原timestamp播放间隔约束：long128.191→128.205s、room64.403→64.255s、
desk31.931→31.787s、xyz156.896→157.297s；desk补充见完整在线表。
因此进程总时长接近不表示Mapping没有加速；实际cycle mean、cycles、TF updates明确变化。
原desk两边TF未激活，该组不能验证CUDA TF训练速度；补充组单独保留，不替代原组。

[完整在线结果表](tests/diagnostics/cuda_tf_implementation_20261013/ONLINE_RESULTS.md)列出全部50个PSNR/SSIM/LPIPS、
各ratio active count、Gaussian total、trajectory、CD步骤、显存及生产Mapping分阶段采样。
最终模型与原日志保存在服务器`results/cuda_tf_online_guard_off_reciprocal_fix_20261014`及`results/cuda_tf_online_guard_off_reciprocal_fix_desk_start1000_20261014`。
前一版本Guard关闭批次完整保留于不含`reciprocal_fix`的同名目录，单独报告表保存在`before_reciprocal_fix/`。

### 9.2 显存、trajectory与TF population

| 场景与TF start | Gaussian total，legacy→CUDA | nvidia-smi进程峰MiB | 平移APE RMSE m | 旋转APE RMSE，unit-less |
|---|---:|---:|---:|---:|
| long，3000 | 190448→141644 | 2782→2650 | 0.009831→0.028578 | 0.014375→0.026305 |
| room，3000 | 152762→136691 | 2972→2532 | 0.053593→0.044329 | 0.061347→0.066376 |
| desk，3000 | 66127→63254 | 1384→1406 | 0.015501→0.014617 | 0.038829→0.038094 |
| xyz，3000 | 29499→29558 | 1014→1198 | 0.003152→0.003217 | 0.034466→0.034388 |
| desk补充，1000 | 64842→80320 | 1404→1330 | 0.015753→0.015024 | 0.038879→0.040537 |

在线显存为1s采样的整个进程观察峰，包含SLAM，不是TF独占或Torch allocator peak；峰值之间也受Gaussian数量影响。
trajectory使用SE(3) Umeyama alignment，没有scale alignment；逐场景原始metrics_traj.txt保留。
原生产日志的transformed N仅逐1000轮打印，完整记录列于在线表，不能冒充逐轮mean N。
独立诊断运行另记录每次TF forward的N与总和，见后续分阶段计时；不混入上述原生产run。

### 9.3 评估口径

Full与LoD均使用原Python renderer/TF/Selector；新CUDA checkpoint已通过Python `load_state_dict(strict=True)`并完成全部LoD推理。
评估没有抽帧、训练或保存render图片；模型、逐帧指标、trajectory、selector_stats与日志保留。
PSNR/SSIM/LPIPS仅按8帧批量计算，reduction=none，每帧分别输出；无任何训练算子改动。
64 Full+64 LoD1%帧与原scalar指标对照：PSNR max_abs=1.90735e-6，SSIM=1.19209e-7，LPIPS=1.26362e-5。
首次指标batch验证atol=3e-6在LPIPS失败，随后明确采用指标专用atol=2e-5、rtol=1e-6通过；
原失败与最终结果均保存于`batch_metric_parity.json`。TF parity容差没有因此改变。
所有正式日志`parity_frame_limit=0`。evo仅在本评估进程设置Agg，不修改用户settings.json。

### 9.4 在线质量差异的排查

最终修正版long Full PSNR下降2.66981dB且所有LoD同时退化，按验收标准视为待排查问题，不能解释为新方法收益。
最终两次long运行在TF尚未启用的第1000轮已经出现active count=574/564、visible=536/449、transformed N均为0；
第2000轮active count=1242/1266。即输入拓扑与在线选择在CUDA TF首次执行前已不同。
最终SLAM平移APE也由0.009831变为0.028578m，Gaussian总数与Mapping cycles明显不同。
这些证据说明独立在线run存在调度/SLAM差异，不能单凭最终PSNR判断是TF数学错误；也不能据此排除实现问题。
追加固定真实输入、相同参数/state的训练链，以及相同legacy配置的独立复跑，结果在后续补充。
修正前同一legacy binary/config/dataset复跑：5456 cycles、2456 TF updates、169698 Gaussians，
Mapping mean/p95=20.9635/43.2435ms；Full PSNR=18.4036、SSIM=0.6745、LPIPS=0.3456，
平移/旋转APE RMSE=0.015734m/0.020140。对比首次legacy PSNR20.3187，旧系统自身已有1.9151dB在线波动。
该单次复跑不足以建立质量置信区间，也不能将CUDA long持续退化判为已解释或通过。

### 9.5 独立在线TF阶段诊断

构建`tum_rgbd_cuda_tf_profile`，串行运行long legacy/CUDA，配置及无尾训协议保持相同。
仅在诊断executable中给原TF图加autograd边界，使用同一backend的原forward/backward，TF与非TF optimizer分区执行。
622条offline包装对照通过，max_abs=1.62981e-9；正则梯度与10步split optimizer p/m/v/step保持bitwise。
在线两边五个阶段calls均大于0，pending_samples=0，证明实际Mapper入口被包装；没有只用offline测试代替。
每19次调用记录当前stream event，harvest仅cudaEventQuery，没有显式synchronize。

| TF阶段 | legacy event mean ms | CUDA event mean ms | legacy host mean ms | CUDA host mean ms |
|---|---:|---:|---:|---:|
| forward | 1.352416 | 0.598394 | 1.288631 | 0.548834 |
| backward | 4.795849 | 1.532737 | 4.933392 | 1.634652 |
| regularization forward | 1.305177 | 0.068488 | 1.335242 | 0.033319 |
| regularization backward | 6.237793 | 0.164577 | 6.190510 | 0.164998 |
| TF optimizer direct | 0.987957 | 0.091103 | 1.151049 | 0.116409 |

legacy共2307次TF forward，N总和22,344,924，各阶段121个event样本；
CUDA共4084次，N总和37,156,409，各阶段214个样本。
这些是独立诊断run的完整normal Mapper TF调用统计；不包含CD内部全部调用，不替代原生产run的精确N统计。
包装、nested autograd与optimizer分区有额外host开销，event包含launch gap和在线GPU竞争。
不能将该表加总当成原生产TF总时间，或与第9.1节的wall拼接计算精确TF占比。
详细command、calls与原日志保存在`online_tf_profile_reciprocal_fix_runs.json`及服务器`results/cuda_tf_stage_profile_reciprocal_fix_20261014`。

### 9.6 完整Mapping剩余成本

最终原生产xyz的LoD CUDA范围由12.5726降至6.0303ms，但LoD backward范围仍为3.4503ms，
另有Full render/backward约0.8204/1.7253ms、全optimizer约1.0930ms；这些嵌套范围不可相加。
long的LoD范围仍为8.9309ms、LoD backward范围4.4560ms；TF core已变快，Selector/Renderer/Loss、
compact gather/scatter、Gaussian梯度链与CPU提交成本仍保留。完整Mapping收益小于独立TF probe，符合现有范围测量。
没有为追求系统加速继续修改这些算法或Mapping频率。原desk没有TF训练，整体均值不改善符合其实际执行路径。

### 9.7 固定输入训练链与near-zero Adam修正

使用原审计真实N=8491输入、同一旧checkpoint、相同ratio轮换与LR/eps，连续100步执行TF+正则+Adam，
并先反传独立Full输入loss，再反传LoD，验证输入梯度累加。
teacher-forced每步复制legacy参数与optimizer state后比较，隔离单次计算；free-rollout两边独立更新100步，观察累计差异。
另外执行legacy-vs-legacy重复及cuBLAS候选，保留原失败记录，不用forward/backward的小误差直接推断optimizer一致。

修正前`grid.L1_xy`在第一步index=10118处，梯度为legacy1.11022e-16、CUDA2.22045e-16；
normal Adam eps=1e-15将这一差异放大为p差1.30832e-4，尽管逐gradient起始atol检查通过。
根因为原ATen mean backward使用CPU scalar产生的FP32 reciprocal乘法，而CUDA直接浮点除法造成near-zero cancellation不同。
修正为host计算FP32 reciprocal后CUDA乘法，保持原loss、权重、eps及其他算法不变。
near-zero Adam regression新增12个参数比较，最终D共374条全部通过，max_abs=1.49012e-8；主工程重建41/41通过。

| 修正后100步训练链 | 比较条数 | 超出起始容差 | 最大绝对差异 |
|---|---:|---:|---:|
| legacy-vs-CUDA warp32，teacher-forced | 11100 | 0 | 1.66893e-6 |
| legacy-vs-cuBLAS，teacher-forced | 11100 | 0 | 1.78814e-7 |
| legacy-vs-legacy，teacher-forced | 11100 | 0 | 1.19209e-7 |
| legacy-vs-CUDA warp32，free-rollout抽查1/10/100步 | 333 | 12 | 1.39669e-3 |
| legacy-vs-cuBLAS，free-rollout抽查1/10/100步 | 333 | 5 | 5.32687e-4 |
| legacy-vs-legacy，free-rollout抽查1/10/100步 | 333 | 13 | 1.39663e-3 |

修正后第一步及10步比较全部通过；free-rollout失败均在100步，warp32与native重复累计差异同量级。
这证明旧实现自身存在会被小eps放大的训练累计数值floor，不能把全部free-rollout差异归因于custom Adam。
也不将这些失败改写成通过：`training_chain_control.exit=1`与逐条CSV保留，未放宽TF起始容差。
optimizer独立同gradient的p/m/v/step仍bitwise；端到端训练gradient不同导致的累计结果不承诺bitwise。
修正后的online、event benchmark与Nsight重新运行，旧批次全部保留并明确revision。

## 10. 当前结论

不改optimizer的regularized独立probe（旧revision）已由11.2229降至1.82194ms；最终修正版包含CUDA Adam后由11.3292降至0.892774ms。
kernel由991降至81，完整Mapping的long/xyz/room及desk补充均实际加速，cycles与TF updates增加。
在线质量没有四场景一致通过：long明显退化；即使单算子parity与多步Adam通过，也不能宣称完整验收成功。

尚未通过项：

- long 场景 Full PSNR/SSIM/LPIPS 与 trajectory 均明显劣于 legacy，根因尚未被固定输入训练链完全解释。
- 100步 free-rollout 训练链仍有累计差异；legacy-vs-legacy 也出现同量级差异，因此不能宣称端到端 bitwise parity。
- CUDA preview 峰值显存高于 legacy，需在后续工作中单独评估 prepared state 缓冲的代价。

结论：当前不建议替换 legacy 默认 backend。最终修正版的完整表格、在线结果和证据索引见
`tests/diagnostics/cuda_tf_implementation_20261013/BENCHMARK_TABLES.md`、
`tests/diagnostics/cuda_tf_implementation_20261013/ONLINE_RESULTS.md`、
`tests/diagnostics/cuda_tf_implementation_20261013/guard_off_reciprocal_fix_acceptance_verification.json`；
证据压缩包为同目录的`guard_off_reciprocal_fix_acceptance_evidence.zip`。
