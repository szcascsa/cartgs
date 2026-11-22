# 第二阶段 CUDA 优化的复现实验

此目录只保存隔离诊断、构建与分析工具。生产算法仍保留 Legacy，所有 CUDA 开关默认关闭。

## 实验边界

- 源 HEAD：`4269d7e3bad6cf5af7e1df1763634b61e98e0b51`，以未提交源码 manifest 为准。
- 真实 checkpoint：上一轮 R3 `7465_shutdown/ply`，15,891 Gaussian、6,121 Anchor。
- 固定相机、Gaussian/Anchor、Cached uniform、选择预算及 Adam 初始条件。
- A=0，B=7，C=15，D=31，E_workspace=63，E=127；bit 1/2/4 是上一轮三项，8/16/32/64 是统计、采样、workspace、SSIM。
- 两轮顺序相反的交替重放；正式计时与 Nsight 分开。共享 GPU 的负载保存在 `gpu_load.csv`。
- `protocol.json` 是事先确定的误差门槛。固定梯度 Adam 与 Renderer atomic 波动分别分析。

## 工具

- `setup.py`：复用上一轮 source/build，按内容增量同步，只在本轮隔离目录编译。
- `build_probe.py`：单模块独立编译与 Gate，4/5/6/7 分别为 Anchor、Selector、workspace、SSIM；8 为人工 ULP pruning 探针，非零退出保留为未通过证据。
- `staged_mapping_benchmark.cpp`：实际生产算子重放，Full 调用 `GaussianModel::addAnchorPruningStats()`；保存输入、输出、梯度和优化器 fixture。
- `continuous_mapping_replay.cpp`：固定拓扑 Full→轮转 LoD 连续更新；可用半分辨率 LoD，Full 始终保持原分辨率。
- `topology_checkpoint_benchmark.cpp`：真实地图上的四类受控事件；人工设置梯度、窗口及 birth clock，不能称为真实在线拓扑分布。
- `anchor_stats_scaling.cpp`：人工满容量 owner 布局的规模压力测试，不能与真实 Anchor 分布混淆。
- `run_suite.py`：模块、拓扑、交替计时、连续训练、Nsight、完整 fr1_xyz 在线运行，任何子任务失败即停止。
- `analyze_phase2.py`：分别汇总 Event、kernel、API 同步、Parity、连续训练与真实在线结果。
- `quality_metrics.py`：固定帧的 PSNR/SSIM/LPIPS/photo；不用于选择参数或 checkpoint。

服务器环境是 CUDA 11.8、RTX3090、g++9、LibTorch 2.3.1 ABI1；Python Torch 为独立 ABI0 环境。不要把 C++ LibTorch 的动态库路径注入 Python 指标进程。

## 运行

在服务器隔离根目录 `/home/prmi/zwj/anchor-selector-r3-cuda-phase2-20261023`：

```bash
/home/prmi/anaconda3/envs/zwj/bin/python setup.py
export CARTGS_PHASE2_RESULTS=/home/prmi/zwj/anchor-selector-r3-cuda-phase2-20261023/new_replay
/home/prmi/anaconda3/envs/zwj/bin/python run_suite.py
/home/prmi/anaconda3/envs/zwj/bin/python analyze_phase2.py
/home/prmi/anaconda3/envs/zwj/bin/python quality_metrics.py
```

首次运行要求对应 checkpoint、原生 fixture、数据集和上一轮依赖构建已存在。在线输出目录有碰撞保护；重跑需使用新的实验根目录，不能覆盖已有模型。

`ReplayModel` 只补齐离线重放所需的新窗口，未声称 PLY 恢复了完整在线训练状态。Topology fixture 的 birth clock 是显式合成状态。完整线上运行使用原配置的自然停止规则，不强制 Mapping=10000。

瞬时半分辨率微基准先对uint8 RGB做OpenCV resize；连续半分辨率重放和真实在线使用FP32 bilinear。各候选内部输入完全相同，但两套预处理的绝对质量不可混用。原生全分辨率fixture不受该量化差异影响。

本地 `evidence_archive/results/` 中已提取B/E的原生fixture、连续快照和Topology状态；`delivery_checks.json`记录全部328文件SHA及45源码与服务器一致的检查。
