# 失败候选与保留边界

本表保留所有主要失败路径。早期筛选只用于决定继续研究方向；最终数值和速度结论仅引用冻结 revision 2 及其之后的验收。早期候选继承的 Tile coverage 表达式尚未固定原二进制 FMA 方向，不能用其初筛逐位通过代替最终正确性。

| 候选 | 结果与原因 | 证据 |
|---|---|---|
| Warp 寄存器属性预取、shuffle | 各 Warp 重复读取/广播，Full 出现回退，不保留 | micro01 |
| 提前计算 2/4 个 Alpha | 保留合成顺序但增加寄存器与无效工作；无普遍收益，TUM Full 回退 | micro07 |
| 4 像素/线程 | 指令并行增加同时拉低并发、延长长链；多个 Full 更慢 | micro09 |
| bbox/log 阈值元数据复用 | 省重复计算却新增读取；低 Ratio 多数慢 1–5 μs | micro10 |
| Tile 枚举与像素渲染融合 | 省一个 Kernel，实际增加 Block 内串行阶段和资源占用；Full 变慢 | micro11 |
| radix8 Block 排序 | 静态共享内存 0x10220 超过 48 KiB，编译失败；没有进入测试 | sort01 |
| radix6/radix7/Block merge | 逐位通过初筛，但没有普遍优于 radix4 512 项稳定排序的收益 | sort02、micro12 |
| 预处理＋Block 排序融合 | 减少一次发射但预处理内资源竞争；Full 慢 2–5%，低 Ratio 慢 5–13 μs | micro13 |
| 属性分块缓存 64/128 项 | 减少静态共享容量却增加同步、循环和重复装载，跨场景有反例 | micro14 |
| 一 Gaussian 一个 CTA 展开 Tile | 低 Ratio 部分改善，但小 footprint 产生空闲线程，Full 部分慢约 8% | micro15 |
| Graph 容量按二次幂填充 | 低 Ratio 有收益，Full 填充过大，改成所有规模统一的 512 粒度 | graph_micro01/02 |
| 每帧 GraphExecKernelNodeSetParams 更新输入节点 | API 更新成本抵消少一次 Host Kernel 发射的收益 | graph_micro04 |
| Kineto 捕获 CUDA Graph | Collection futex 等待、GPU 0%，无日志进度；仅停止已核验身份的本轮进程，保留现场；Graph 改用 Nsight Systems | PROFILE_INCIDENT.json |
| 原 Tile coverage 普通 C++ 表达式 | 极薄 Gaussian 的 FMA 融合方向随编译上下文改变，原 B 及初版 C/G 在少数像素不一致；初版正式测试立即停止，按原二进制 SASS 固定 __fmul_rn/__fadd_rn/__fmaf_rn 后重新验收 | V1_REJECTION.json、reproduce01–03、ordered01、coverage01/02 |
| Graph task_count 的旧 cudaMemsetAsync 清零 | initcheck 报未初始化读取；保留诊断，改为图内显式清零 Kernel，并重新执行 memcheck/initcheck | INITCHECK_INCIDENT.json、sanitizer_init04 |
| 8 行/4 行像素子 Tile | 新独立模块完成编译和逐位对照。拆分增加同一成员列表/属性装载，除少数 TUM 条件外普遍慢于冻结 C；不在保留场景继续调参、不纳入最终路径 | microtile01 |
| 空集合相对 B 更慢 | B 旧 primitive 返回零图；A 产品入口和新候选正确返回背景图。语义不同，不能将 B 的较低开销称同结果优势；保留全部 fixed0 计时，不纳入非空加速范围 | empty01、RASTERIZER_RESULTS.csv 的 fixed0 |
| formal03 受 Xorg 活动干扰 | 一次约 7% SM 活动，整轮无效，原始数据保留；完整重跑 formal04，无外部活动 | reports/jobs/formal03.json、formal04.json |
| profile_suite01 受 Xorg 活动干扰 | 一次约 3% SM 活动；不使用该批 Kernel 时间，完整重跑 profile_suite02 | profile_suite01/02_MONITOR.jsonl |
| async_final01 受 Xorg 活动干扰 | 两次约 1% SM 活动，整轮无效；核验boot_id/PID/start_ticks/命令后仅SIGTERM自有测速子进程，保存现场并完整重跑async_final03 | ASYNC_INTERFERENCE_INCIDENT.json、async_final01/03 |

后续相机执行图候选的 `sanitizer_camera_graph01` 约143秒无中间输出，但实际CPU主线程保持99%左右活动，最终正常完成1592项检查、memcheck 0错误。它不是失败候选，也未被终止。只读strace attach因系统ptrace设置被拒绝，未修改系统权限；该诊断等待不被计入推理性能。

动态计数、末级目录溢出、不同 Stream 和工作区上限仍需遵守原错误处理。没有采用跳过检查、近似 Alpha、降低分辨率、裁减 SH、改变 Gaussian 身份、改变严格 K 或使用旧视角结果的方案。NCU 因 ERR_NVGPUCTRPERM 无法读取硬件计数器，未更改系统驱动权限；Nsight Systems 的 CUDA/NVTX/Graph 节点追踪可用。
