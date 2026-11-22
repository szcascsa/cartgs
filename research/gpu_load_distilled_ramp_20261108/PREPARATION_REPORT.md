# Replica蒸馏Compensator连续升载：测速准备报告

## 当前状态

CPU准备已完成，新的GPU验收和有效正式测速尚未开始。用户已明确要求等待其通知GPU空闲；无自动启动、空闲轮询启动或后台测速任务。最后一个本轮GPU测量进程2589744已退出。

以下“通过”仅指编译和CPU验证。最终负载控制器能否在全部方法中实现目标实际占空比，必须等新的GPU验收，不能提前认定。

## 已锁定的模型与实验

| 项目 | 配置 |
|---|---|
| 场景 | Replica office3 RGB-D，完整2000帧 |
| 学生 | 连续Ratio校准后的32/32 Compensator，四个直接线性Head，3534参数 |
| 部署 | 已支持该学生的v31，FP32，TF32关闭；保持原Selector和Anchor Feature |
| 目标负载 | 第i帧从0开始，u=0.2+0.6i/1999，即20%线性升到80% |
| Adaptive预算 | B=floor(4000−3000i/1999+0.5)，4000线性降到1000 |
| 实际K | 名义Anchor可见候选池中，K=min(B,合格候选数) |
| 对照 | CaRtGS Full、我们Full、学生固定4000、学生固定1000、学生Adaptive |
| 重复 | 三轮轮换方法顺序，每方法每轮2000帧，共30000请求 |
| 准备隔离 | 准备Stream最高优先级−5；人工负载−4；渲染0。准备阶段人工负载暂停并排空 |
| 输出 | 原始请求、实际负载、K、PSNR、准备/渲染/总耗时、mean/p95/p99/std/CV；不保存渲染图像 |
| 绘图 | 1×3，英文图例；耗时51帧、PSNR81帧居中均值；同时保留未平滑耗时与原始统计 |

学生checkpoint：

`/home/prmi/zwj/experiments/compensator_distillation_replica_all_20261108/models/Replica__office3__rgbd/continuous_calibration/best.pt`

学生SHA256：`20bcf17294a4eeec7d493246742882d04767a74e7ddfd3dff50fd4349fb6ac90`。

v31二进制SHA256：`fa7b855685ecd4f75ce7118ee07fadd27640af08099a6775ffee5fe87fa7d601`。

## 负载计量修正

旧发生器把CUDA Event或整包驻留包络作为ON时间。Renderer已有Block不能被Stream优先级抢占，短负载包因此可能分批进入SM；包络中包含部分SM空闲或调度等待，不能代表实际人工负载。已有尝试中，我们Full的某负载区间比目标低12.85个百分点，故该批正式尝试作废，不能拿来比较速度。

最终准备版本直接按**归一化SM驻留时间面积**计费：同一SM内先对人工负载Block区间求并集，再跨SM求和并除以SM数量，得到等效整卡ON时间Q。每包计费ceil(Q)，各渲染窗口累计许可u×active，条件尾部结清OFF，约束真实SM时间占空比不超过80%。

这不是整卡排他时间，不能声称每个ON包都抢占全部GPU。整包驻留包络和CUDA Event仍记录，且包络占空比可能高于SM面积占空比，两个口径分别导出。所有方法都使用同一口径、同一帧序的目标负载。

保留无跨SM栅栏的普通短包、2ms包时长上限和准备阶段零重叠。此前Cooperative Launch、有限启动握手及旧计费测试全部保留为失败证据，均不并入最终结果。具体诊断见PROTOCOL.md。

## 编译与CPU验证

- CUDA库已用CUDA11.8、sm_86重新编译成功，7个C ABI入口符号齐全。
- 当前负载库SHA256：`5f26a08344c1139d25c0c29508ce90575fd6ba57b5e93b9b9160ddcb03b1eab0`。
- C++测试通过：区间重叠/嵌套/接触/空集合、多SM归一化、部分覆盖、长时间戳精度、向上计费、非法占空比拒绝。
- Python 5项测试通过：实际2000帧日程、真实审计函数的SM面积计量与账本一致性、准备/渲染阶段隔离、缺失/陈旧/不完整GPU验收拒绝、运行源码及二进制变化后的哈希失效。
- 后处理脚本通过语法和本地依赖导入检查；未用模拟数据生成实验结果图。
- 上述验证只在CPU执行；NVCC编译没有运行CUDA Kernel。

## 正式测速前的强制门槛

正式入口要求`GPU_ACCEPTANCE.json`通过且绑定当前源码、部署库、负载库和保护清单共59个文件的哈希；源码或二进制变化后必须重新GPU验收。旧smoke01–06不能作为此版本的验收证明。

当前正式入口的CPU核查结果为：**“当前实现尚未通过GPU验收，禁止正式测速”**，符合预期，未绕过此限制。

用户通知空闲后，按以下顺序执行。两条命令是操作清单，未启动，不能在其他实验占用GPU时执行。

```bash
cd /home/prmi/zwj/experiments/gpu_load_distilled_ramp_20261108/source/research/gpu_load_distilled_ramp_20261108
/home/prmi/anaconda3/envs/zwj/bin/python -B -u experiment.py --label smoke07 --smoke
```

第一阶段包含40组Graph/Native输入和选择ID逐位检查，以及CaRtGS Full、我们Full、学生固定1k各2000帧验收。每250帧实际SM占空比相对时间加权目标偏差≤5个百分点、包驻留≤2ms、准备重叠为0、累计SM占空比≤80%；全部通过后才产生证明。

```bash
/home/prmi/anaconda3/envs/zwj/bin/python -B -u experiment.py --label formal03
```

正式阶段15条件共30000次请求，同帧首尾图像与无负载参考逐位复核；出现新计算进程、既有Viewer活动、占空比不匹配或数值异常即停止并保留现场。Adaptive是预设负载到预算的映射，不是自动反馈控制器。

## 保护与文件

服务器独立目录：`/home/prmi/zwj/experiments/gpu_load_distilled_ramp_20261108`。

保护清单847个生产/模型/环境文件SHA256均一致；冻结推理源码与部署库无非预期变化。准备核查时剩余17.96GiB，正式开始前仍需重新核查≥5GiB。未改生产默认、共享conda、模型权重，未commit/push。

`reports/PREPARED_MANIFEST.json`保存待用户通知状态、59文件身份、模型和库SHA、CPU验证及下一运行标签；`reports/PLANNED_MATRIX.csv`保存15个计划条件，属于计划，不是测量结果。原先`reports/STATUS.json`保留smoke06失败现场，不能将其当作当前CPU准备状态。

准备证据已保存到本地`results/reports`及服务器独立目录。完整技术测速报告和图表必须等待实际GPU测试完成后生成。
