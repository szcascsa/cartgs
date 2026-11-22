# CUDA TF 实现与验收记录

基线分支：cuda-tf；HEAD：99a9bb1f05e458617f50fabb30a910f14aaf7c9c。
开始时已有未跟踪：ORIGINAL_TF_CUDA_AUDIT.md、build-cuda118-sm86.zip、tests/diagnostics/。
旧审计材料保留，不覆盖原有测量；本目录保存各阶段独立产物。

完成门槛：A query parity/benchmark → B/C MLP与输出完整梯度及双策略benchmark → D正则 → E profiler → F Adam事务 → 特殊流程 → 四场景在线A/B。
默认legacy，未通过前置门槛不进入下一阶段；不将静态检查写成CUDA运行通过。

Phase A：目标机器编译与所有feature/XYZ/ratio/plane梯度parity通过。
真实checkpoint使用8491个随机domain内点；补测脚本另使用原审计真实input.bin。
每个N/阶段预热100次、测量500次。N=8491时query forward 0.361638→0.035994 ms，backward 1.44491→0.197406 ms。
首版曾因重复重算采样使50k backward慢于legacy；已保留FP32 corner/channel累加顺序，消除重复采样，复测全部通过。
边界测试、零sample、scalar/per-row ratio、非默认stream已通过。详见phase_a_parity.csv与phase_a_benchmark.csv。
Phase 1 全训练口径补测已通过（phase_1.exit=0）：保持旧MLP/output/regularization/Adam。
真实input.bin循环至各N，100预热+500测量；N=8491 total_regularized 11.3323→9.50063 ms。
这是query阶段收益，不作为最终完整CUDA backend性能结论。
Phase B：两种MLP forward与输出变换全部parity通过（phase_b.exit=0）。
随机/零feature、真实checkpoint、N=0、默认H64、非默认stream，以及quaternion零/近eps/符号翻转均覆盖。
N=8491 CUDA event均值：MLP legacy/cuBLAS/warp32=0.151107/0.068659/0.227565 ms；
total forward=0.733995/0.083589/0.240613 ms。各独立阶段不可直接相加替代total测量。
两种候选保存相同activation；暂以cuBLAS优先，最终选择仍需完整forward/backward/reg/optimizer测量。
Phase C：381条逐Tensor/边界检查全部通过，phase_c.exit=0。
CUDA query、MLP、output通过三个薄autograd节点连接；参数/输入SavedVariable检查revision。
保留undefined head梯度与N=0 query无plane边；验证Full+LoD累加及Guard型autograd::grad。
Quaternion near-eps曾发现FP32归约与q*q两条backward edge累加顺序问题，现与旧chain对齐；
精确零范数保留旧NaN梯度行为，未添加保护或放宽容差。诊断oracle和失败日志保留。
N=8491 total backward legacy/cuBLAS/warp32=2.81709/0.492716/0.488135 ms。
Phase C数值修正后重跑完整Phase B forward parity与benchmark，通过（phase_b_after_c/）。
Phase D：362条正则loss/gradient/边界检查通过；正则梯度使用更严格atol=2e-10、rtol=2e-4。
两个forward kernels（partial+scalar reduction）、一个backward stencil kernel，保留逐plane mean与sum顺序、d1差分顺序。
N=0仍保留独立regularization autograd；query与regularization plane梯度累加通过。
首版metadata按值传递helper曾使backward=9.301ms，慢于旧版，未采用；已移除helper中的大结构复制。
最终独立forward+backward：legacy=6.61752ms，CUDA=0.107540ms；详见phase_d_benchmark.csv。
原始首版测量保存在phase_d_initial/。
Phase 3：54个backend/mode/N组合全部finite，100预热+500测量，phase_3.exit=0。
真实N=8491 total CUDA event ms（legacy/cuBLAS/warp32）：
core=4.90729/1.78115/1.73975；regularized=11.2229/1.88090/1.82194；preview=12.5850/3.06572/2.96946。
仍完全使用旧Adam/Preview；这回答了不改optimizer的收益，不能当作完整Mapping wall time。
Phase E Nsight：同机复现legacy=991 kernels/step；cuBLAS=301。
forward 81→17；backward 602→66；regularization NVTX 93→3；Adam仍208；loss仍7。
grid_sampler=0、CatArrayBatchedCopy=0；D2D copy 156→0/step。
小于10us比例96.87%→98.54%：剩余短kernel主要由旧Adam与loss/grad处理产生，比例上升不代表kernel数增加。
N=8491 regularized peak allocated：57,263,104→48,643,584 bytes；reserved包含allocator已有缓存，不作为单次独占内存。
CUDA query/MLP/output/regularization已消除原coordinate/grid/sample/product/cat/重复ReLU/quaternion临时链；
仅保留feature、四份必要ReLU activation、heads、输出与gradient buffers。
Nsight原始trace保存在服务器本目录；summary与资源用量下载保留。query backward=183 registers，warp MLP=48，无local/stack spill。
完整训练中warp32略快于cuBLAS，最终生产选择仍需custom optimizer之后按总成本比较。
Phase F1：CUDA Adam direct通过，phase_f1.exit=0；311条p/m/v检查最大绝对误差全部为0，逐tensor step一致。
覆盖1/10/37/100步、grad=None、zero grad、LR0、weight decay、特定plane state reset、eps1e-15/1e-8、独立normal/CD/DSD状态、非默认stream。
注册参数与原AdamParamState存储不变；按metadata批量遍历，fast全部26参数只需一个kernel，state保持逐tensor clock。
真实TF参数规模100预热+500测量：Adam direct legacy=0.995019ms，CUDA=0.065272ms。
Phase F2：prepare/candidate/commit/discard与混合group adapter通过，phase_f2.exit=0。
136条检查最大绝对误差全部为0；覆盖连续100步、原gradient moments、shrink/projection方向、reject、过期/非法candidate的整组写入前验证、非默认stream。
实际GradientConstraintManager accept/shrink/project/fallback/reject决策与旧版一致；原CPU Adam/Guard测试仍通过，F1重跑通过。
真实TF参数规模Adam preview/candidate/commit：legacy=2.18247ms，CUDA=0.239213ms。
TF prepare不写live参数/state；candidate只修改位移；所有groups共用一次质量决策和统一commit/discard。
Phase 5 完整成本复测：54组合全部finite，100预热+500测量，phase_5.exit=0。
N8491 legacy/cuBLAS/warp32 total：core=5.06172/0.824185/0.793326ms；regularized=11.3968/0.954118/0.902027ms；preview=12.6601/1.14448/1.08958ms。
Warp32在1k/5k/8491/10k完整训练成本较低，cuBLAS在20k/50k更快。生产H32/F16且N<=10000选择Warp32，其他使用cuBLAS。
最终Nsight同机：regularized Adam total kernels=991/94/81；preview=1043/96/83；CUDA Adam direct=1kernel，prepare/candidate/commit=3kernels。
CUDA两候选grid_sampler=0、CatArrayBatchedCopy=0、D2D copy=0。N8491 Warp32 direct短kernel比例92.47%，legacy96.92%。
API与特殊流程：1810条检查通过，max_abs=9.53674e-7；激活/对数scale、outside、重复索引、teacher、old load/new save-load、非默认stream。
既有完整ChangeDistill与DriftStability生命周期套件已在CUDA backend运行通过，混合group direct多步测试通过。
主工程rebuild通过，main_build.exit=0；完整CTest 41/41通过，包括真实Mapper Replay CUDA TF入口、Protection、CD/DSD、Guard与旧CPU测试。
构建配置首次曾将CUDA target误放入rasterizer sources，已纠正并重建；原失败日志保留，最终main_build.log记录95个构建步骤与全部CTest。
当前阶段：完整在线四场景A/B、最终报告；此阶段尚不能宣称完整项目验收。
