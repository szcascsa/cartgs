# Selector / Compensator 冻结推理 CUDA 优化报告

日期：2026-11-03。代码基础：本地 `real-lod` 分支、HEAD `857909570760b61337e698af63b725524d164619`，保留原有未提交修改。新增代码未自动 commit/push。

## 结论与适用范围

实际实现并编译了可选 `cuda` / `cuda_graph` 推理后端。固定地图、相同视角、相同噪声与严格 K 下，热态预算准备加速 **2.54–3.07×**；含 Rasterizer 的服务端准备加渲染加速 **2.17–2.50×**。5,000 点预算准备从 **3.805 ms 降至 1.438 ms**，含 Rasterizer 从 **4.274 ms 降至 1.903 ms**。两轮独立交错测量方向一致。

最终代码通过 **9,467 项逐位检查**，其中数值边界与恢复检查覆盖 **13,793,082 个结果值**。这些不是重新训练结果；没有改变模型、训练调度、Loss、Adam 或 checkpoint。验证目标是冻结推理输出和选择完全一致。

GPU 为 RTX 3090，CUDA 11.8，PyTorch 2.3.1+cu118，FP32，TF32 关闭，不使用 fast-math。现有默认仍为 `pytorch`。服务器两个项目均未部署覆盖。

这里的“准备”包括可见 Anchor、候选压缩、预算、Selector、排序、Compensation、最终属性与 finite 检查；“含渲染”加上相同 Rasterizer。**不含 HTTP、图像封包、OpenGL/Moba 显示，不是完整在线 SLAM 加速或 Viewer 实际 FPS。**

## 1. 当前实际模型与基线

只读使用 checkpoint：

`/home/prmi/zwj/realLOD/results/fr1_xyz_frozen_master_20261103_01/r20_40_60_80/training/13052_shutdown/ply`

- 数据：TUM RGB-D freiburg1_xyz，640×480，792 个可匹配数据集视角。
- Master：13,350 Gaussian；Anchor：3,387。
- Selector：11→32→ReLU→32→ReLU→2，1,506 参数；无 Set Encoder、Context Residual 或 Selector Anchor Feature 分支。
- Compensator：Residual48，主输入 51D（offset3+Feature48），Ratio3 独立线性投影；64/64 主干，XYZ/Scale/Rotation/Other 四个独立 Head，共 25,230 参数。
- 当前 checkpoint 训练节点实际为约 0.2/0.4/0.6/0.8，因此为一份 Base 加三份 Residual，不是历史六节点模型的五份 Residual。实现支持 2–6 节点，另做六节点合成等价检查。
- 使用 checkpoint 自带 Softplus Scale schema，未擅自切回历史物理加法或指数版本；融合解码另验证物理加法的合法正 Scale 域。
- 固定地图推理没有 Master/Feature 反传；训练代码与训练 CUDA 后端未修改。

初始独立审计在服务器有其他评估任务期间运行，只用于热点定位；最终两轮性能测试前 GPU compute 进程列表为空。不能将初始审计中较高墙钟时间当作本轮收益基准。

## 2. 保留的实现

### 静态属性预计算

`FrozenResidualInference.rebuild()` 在冻结状态下计算 Selector 的 10D 物理输入、Compensation 归一化/asinh/clamp 后 offset3、activated Scale/Opacity、归一化 Quaternion、Anchor 引用尺度和 eligibility。

Selector 每次只做单 Kernel Gather 与 log-ratio 打包，然后运行原形状的三个 `at::linear`。没有将候选 n 换成整图 N，也没有 padding，因为这可能改变 GEMM 的算法和舍入。

### 原序 Residual48 输入融合

单 CUDA Kernel 直接写入 K×51 输入。按原 Python double 规则计算活跃层和插值系数，转 FP32 后在 CUDA 使用 `__fmul_rn`、`__fadd_rn` 逐层累加；未激活层不进行加零。Offset 来自精确静态缓存。四 Head 权重、调用顺序和输出语义不变。

Compensation 的 Ratio3 log/sin/cos 在控制向量构造时只计算一次，不逐点重复；Selector 的 FP32 log 在打包核内保留原结果。Ratio3 投影仍使用原矩阵形状和独立权重。

### 只处理最终 K 点

原路径会在 n 个候选属性上 clone、scatter 补偿，再用 mask compact。新路径直接 Gather 最终有序全局 IDs，对 K 点补偿，保持属性与行序一致。

CUDA `publish_selected` 融合 XYZ 加法、Opacity logit 加法及 sigmoid、Softplus/物理加法 Scale 解码及原 log→exp、SH DC 加法、SH Rest Gather 和最终 Buffer 写入。Quaternion 乘法和重复归一化仍走原 ATen 表达式，避免改变归约与叉乘运算顺序。最终一个连续、独立拥有存储的 FP32 Buffer 供 Rasterizer 使用。

### 候选压缩、稳定排序与 Workspace

- CUB `DeviceSelect::If` 按原 Gaussian 行序压缩可见成员。
- Anchor 可见数、候选数及 eligible 数合并为一次 24B DtoH 读取，替代原多次 scalar/nonzero 同步。
- CUB stable radix sort 对真实 softmax score 排序；同分按原候选行序优先；选出 K 后恢复原递增行序。
- 排序、压缩和计数 Workspace 随 rebuild 预分配，按当前 N 有界，单 Viewer 实例串行使用。
- 全图训练 ratio 模式仍沿用原分位数及同分扩张语义；没有偷偷改成严格 Top-K。随机对照继续匹配学习组实际 K。

### CUDA Graph

仅捕获固定实际 K 的 Compensation、属性解码和 Gather；Selector 候选 n 保持动态 eager 路径。每个 `(K, compensation开关, 解码开关)` 第三次出现才捕获，最多 6 个 Graph，计入分配/保留显存的预算为 512 MiB，形状访问记录最多 32 项。Graph 不跨训练更新或拓扑变更。

每次 replay 更新 IDs 与 ratio 控制向量。发布的属性复制到独立 Buffer，下一次 replay、LRU 淘汰和 rebuild 不会改写之前返回的结果。捕获前后处理非默认 Stream 依赖。

这不是完全脱离 LibTorch 的运行时：输入、选择与解码热点已采用 CUDA/CUB，GEMM 保留相同 FP32 cuBLAS/ATen 调用，Quaternion 使用原 ATen。完全重写 GEMM 暂未作为最终候选；收益已主要来自消除调度及重复工作，不能为“纯 CUDA”改变矩阵累加顺序。

## 3. 热态真实预算性能

每条件 5 个相机（0/198/396/594/791），各后端交错正反顺序运行 60 次，舍弃前 15 次，保留 45 个样本。完成两轮独立进程复测。以下为第二轮中位数；各 ratio 使用本视角候选自己的 N 计算 K，没有 same-K 跨场景替代。

| 条件 | 平均 K | PyTorch 准备 ms | CUDA Graph 准备 ms | 准备加速 | 含渲染 ms：原→新 |
|---|---:|---:|---:|---:|---:|
| 1% | 79.0 | 3.728 | 1.213 | 3.07× | 4.200 → 1.678 |
| 5% | 397.0 | 3.684 | 1.201 | 3.07× | 4.171 → 1.684 |
| 10% | 794.6 | 3.728 | 1.230 | 3.03× | 4.207 → 1.714 |
| 20% | 1589.6 | 3.792 | 1.289 | 2.94× | 4.269 → 1.767 |
| 40% | 3180.0 | 3.791 | 1.412 | 2.69× | 4.288 → 1.886 |
| 80% | 6360.8 | 3.866 | 1.523 | 2.54× | 4.351 → 2.003 |
| 5,000 点 | 5000.0 | 3.805 | 1.438 | 2.65× | 4.274 → 1.903 |

不启用 Graph 的 `cuda` 后端准备约 1.71–1.83 ms，也比原版稳定更快，适合形状频繁变化的场景。Graph 初次构建不免费：本轮冷请求最大约 43.5 ms，已单独保存在 `cold` 数据，不混入热态收益。

固定输入 CUDA Event 微测进一步分离了组件：Selector 输入+MLP 约 0.32→0.15 ms；K 点补偿至最终 Renderer 输入，在 1% 约 2.18→0.27 ms，在 40% 约 2.22→0.46 ms，在 80% 约 2.39→0.63 ms。Event 包含 Host 提交间隙，不能称为纯 Kernel 时间。

## 4. Nsight Systems 复测

使用 Nsight Systems 2022.4、NVTX、CUPTI correlation 和 `--cuda-graph-trace=node`，每条件 10 次热请求。真实 Kernel 数和 Host 发射数分开统计，Graph 节点仍计作 GPU Kernel。

| ratio | GPU Kernel 数 | Host 发射调用 | 累计纯 Kernel µs | 同步 API 次数 | DtoH 次数 |
|---|---:|---:|---:|---:|---:|
| 1% | 241 → 97 | 241 → 54 | 937.7 → 394.7 | 27 → 8 | 21 → 3 |
| 40% | 244 → 106 | 244 → 63 | 1138.1 → 611.1 | 27 → 8 | 21 → 3 |
| 80% | 250 → 106 | 250 → 63 | 1372.4 → 780.1 | 27 → 8 | 21 → 3 |

Profiler 有扰动，不能用其墙钟与无 Profiler 的表格混比。最初 `cudaProfilerApi` 范围采集只得到部分 NVTX、没有 CUPTI Kernel 数据，已弃用；最终采用整进程采集、NVTX 归属并导出 SQLite，确认实际存在 Kernel 和 Runtime 表。

最终 40% 准备阶段 GPU Kernel 数减少约 56.6%，Host 发射调用减少约 74.2%，同步 API 27→8，DtoH 21→3。保留必要 finite 检查和同步依赖，不通过延后异常处理换速度。纯 Rasterizer 工作不变，约 0.46–0.50 ms。

## 5. 数值、状态与边界验证

整数位模式比较 FP32 输出，包含 signed zero；不是使用容差 `allclose`。检查总数：分层 1695，完整流程 6485，生命周期 1158，数值边界/恢复 129。另有原 15 个预算单测、6 个本地 CLI 单测通过。

- Selector 输入、logits、Feature 插值、Compensation 输入、14D residual、解码属性及实际 640×480 图像逐 bit 一致。
- 训练节点、未见 ratio、节点两侧 epsilon、0/1/17/129/1,000/13,350 等 K/候选形状，完整四组 A/B/C/D、0 与超大预算。
- 非默认 Stream、旧 Graph 输出所有权、形状缓存淘汰、空地图、106,824 点合成负载。
- Feature 更新、几何修正、Gaussian 与 Anchor reorder、Clone/Split、Anchor growing/迁移、Prune 均触发过期检测；重建后与原路径一致。
- checkpoint 不保存推理缓存，原状态保存重载后重建，输出一致；2/6 节点 Feature 插值一致。
- 随机噪声、CPU/CUDA RNG、模型参数/Buffer 保持不变，参数 grad 为空；启用 GradMode 时 CUDA 冻结入口明确拒绝训练调用。
- 非有限 Feature 保持报错；没有删除 finite 检查。

没有进行训练后的 PSNR 对比，因为本轮只更换冻结推理路径，没有训练。图像逐位一致意味着本轮已检查视角在相同参考图上的 PSNR/SSIM/LPIPS不变，不外推到未验证 GPU/工具链。

## 6. 显存、失效与并发

实测 6 个 Graph 记录容量约 **70.11 MiB**。三后端同进程对照的峰值 allocated 约 **293.22 MiB**、reserved 约 **832 MiB**；这是整个对照进程，不能全部归于优化额外成本。静态缓存和 Workspace 随 N/A 线性增长，没有按相机无限缓存。

`FixedBudgetRenderer` 每次检查所有模型参数/Buffer 与 Master Tensor 的对象、地址、版本和形状，以及 ratio 节点；任何改变均要求 `rebuild()`，同时清空 Graph、全图输出 LRU、eligibility 与物理缓存。原全图输出 LRU 仍最多 8 项。拓扑迁移不能沿用旧行索引。

单实例 Scratch 与 Graph 输出在一个请求中使用，Viewer 已有锁；多线程或多 Stream 并行请求须各自实例/Workspace，不能并发重放同一份可写 Graph Buffer。

## 7. 未保留的前缀缓存候选

冻结 Anchor 完整 Residual 前缀按原序预计算，再查询“前缀+最后插值层”，逐位验证通过。但本轮实现没有获得稳定端到端收益，若干条件准备时间回退约 1–6%；同进程交错打包微测仅高 ratio 改善约 4%，低 ratio 无收益或略慢。该候选已从最终源码删除，快照留在 `rejected_prefix/`，证据留在 `PREFIX_*` 与 `BENCHMARK_prefix_*`。

不能把“少读取几张表”自动视为更快；额外 Kernel 参数和分支的具体影响未通过寄存器/指令级分析完全隔离，因此不武断归因。最终采用原序融合累加。

## 8. 使用方式与回退

最终实验源码与扩展：

`/home/prmi/zwj/experiments/inference_acceleration_20261103/source/`

在服务器使用已有 `zwj` 环境，并指定独立扩展目录：

```bash
export PATH=/home/prmi/anaconda3/envs/zwj/bin:/usr/local/cuda-11.8/bin:$PATH
export CUDA_HOME=/usr/local/cuda-11.8
export TORCH_CUDA_ARCH_LIST=8.6
export TORCH_EXTENSIONS_DIR=/home/prmi/zwj/experiments/inference_acceleration_20261103/build
python /home/prmi/zwj/experiments/inference_acceleration_20261103/source/scripts/view_budget.py \
  /home/prmi/zwj/realLOD/results/fr1_xyz_frozen_master_20261103_01/r20_40_60_80/training/13052_shutdown \
  --dataset /home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz \
  --viewer native --viewer-binary /home/prmi/zwj/realLOD/bin/view_budget \
  --inference-backend cuda_graph
```

原 Viewer 需要现有 X11 DISPLAY；沿用用户能看到的会话即可。回退只需 `--inference-backend pytorch`；也可选 `cuda` 禁用 Graph。不支持的网络布局明确报错，防止静默使用错误权重布局。没有修改共享环境或安装依赖。

代码入口：`python/frozen_inference_cuda.py` 管理缓存和 Graph，`.cpp` 保留相同 GEMM 调度，`.cu` 负责融合/选择/发布；`anchor_budget.py` 接入可选后端，`budget_viewer.py` 和 `scripts/view_budget.py` 暴露独立 CLI 开关。

## 9. 完整性与剩余瓶颈

- `production`：346 个受保护文件，哈希全部不变。
- `realLOD`：2581 个受保护文件，哈希全部不变。

最终源码 SHA 在 `SOURCE_MANIFEST.json`。保留 baseline 与关闭开关，未改变生产默认，未 commit/push。

下一步最有价值的是：

1. 高 ratio 下四 Head/GEMM 已成为最大累计 Kernel 开销；可以继续评估同形状 cuBLAS 调度或精确分块，但必须重新证明逐位等价，不能直接合并为更宽 GEMM。
2. Selector 候选 n 动态，仍有 Host 发射和 CUB 排序；有界 exact-n Graph 或 C++ 统一调用是后续候选。排序稳定性、同分与严格 K 仍是硬条件。
3. Quaternion 仍由多个 ATen 算子完成；融合需要复现 norm、3D sum、cross 的具体运算顺序，收益需和单 Graph 已消除的 Host 开销区分。
4. 原生窗口端到端较慢仍包含 FP32 HTTP 传输、参考图、PSNR 与软件 OpenGL，不由网络 Kernel 优化完全解决。本轮未改变显示和评价功能以制造网络加速。
5. 可对更大真实地图补充性能回归；当前 106,824 点为合成边界验证，不能称为第二个在线数据集性能结论。
