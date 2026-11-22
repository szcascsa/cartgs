# Ultrafast room 服务器性能诊断

日期：2026-10-12。服务器：`172.19.163.58`，用户 `prmi`，conda 环境 `zwj`。本轮只运行诊断，没有修改服务器源码、构建产物或训练配置。

## 结论

1. **首要瓶颈是每个 Mapping iteration 的 Streaming GI CPU snapshot，不是 Gaussian Render。** 不带 Nsight 的运行中，GI 共 34.35 s，占 Mapping wall time 的 60.00%；其中全量 KF/MapPoint snapshot 为 29.75 s，占 51.97%。
2. **全量 snapshot 还与 Tracking 争用地图锁。** Nsight OSRT 调用栈确认 `updateGlobalGIBank()` 的 mutex wait 为 7.975 s；Tracking::Track 的 mutex wait 为 8.705 s。等待时间已经包含在各自函数耗时里，不能再加到总 wall time 上。
3. **第二个排查重点是 TF 启动后 LoD 的固定非 Render 工作，尤其全量 grid 正则化和 dense Adam。** native 运行 TF 更新 841 次，但仅 15 步实际变换 Gaussian，累计变换 15 个点。仍会执行 12 个完整 plane 的正则化和 grid optimizer。不能把 LoD 与 Full 的耗时差直接称为纯 TF 时间。
4. **GI periodic refresh、TF 小规模 sampling/MLP 都不是当前主要 GPU 计算开销。** native GI 的两次 periodic refresh 合计仅 73.35 ms。CUDA-only trace 中全程序 grid_sampler 合计 2.25 ms，所有 GEMM 合计 47.80 ms。

## 实验与证据目录

原命令的程序、词表、两份 SLAM/Mapping 配置、数据集、association、`no_viewer` 和 `--selector-config ./cfg/selector/fast.yaml` 均保持不变。仅把结果路径改为独立子目录，保留已有结果。

服务器证据根目录：

```text
/home/prmi/zwj/mycartgs/results/newscheme/tum_rgbd/room/perf_check_20261012_214513/
```

| 运行 | 用途 | 正常 Mapping | Selector / TF 更新 | GI / Mapping wall | 应用日志 FPS | 退出码 |
|---|---|---:|---:|---:|---:|---:|
| 根目录，CUDA + OSRT trace | 锁调用栈与 CUDA API 证据 | 2077 | 0 / 0 | 70.86% | 22.3007 | 0 |
| `native/`，无 Nsight | 主要性能基准 | 2671 | 841 / 841 | 60.00% | 22.7822 | 0 |
| `cuda_only/`，CUDA trace | 核对 TF sampling / GEMM | 2462 | 306 / 306 | 57.73% | 22.7077 | 0 |

三轮均完整处理 1352 帧，均 `tail_training_steps=0`。异步 SLAM 会使 KF、正常训练步数、maturity 与 TF 启动时间不同；这不是受控消融，不能把运行间差值全部归因于 profiling。

服务器当前配置：rotation=1；Protection age=800、optimization=300、unique-KF=5；TF start=800、spatial=64、ratio=150、feature=8、multires=[1,2]；GI streaming、view_ratio=0.30、refresh_interval=1000；Replay、GradientConstraint、DriftStability 均关闭，ChangeDistill 开启。没有擅自改这些参数。

每轮目录保留 `run.log`、`run.exit_code`、`time.txt`、`model/`；两个 Nsight 目录另有 `trace.nsys-rep`、`trace.sqlite`、`nsys_*.csv`。native 和根目录另有逐秒 `gpu.csv`。

## Native 实测

| Phase | 步数 | 正常训练均值 ms | Render Event 均值 ms |
|---|---:|---:|---:|
| 0.01 | 535 | 20.1632 | 0.2862 |
| 0.05 | 534 | 20.1256 | 0.2788 |
| 0.10 | 534 | 20.3197 | 0.2961 |
| 0.15 | 534 | 20.4029 | 0.3061 |
| Full | 534 | 16.4797 | 0.7550 |

- 正常训练 mean / P95：19.4985 / 43.4853 ms。
- 外围 Operations、GI refresh 等也计入的完整 Mapping mean：21.4343 ms，累计约 57.251 s。
- 正常 Render count：Full=534，LoD=2137；正常 Render Event 总时间约 1.027 s。辅助 GI 和结束后的导出渲染不在这两个计数中。
- 最终 Gaussian 数：248281；mature=51。
- TF 首次初始化的既有 mapping clock=1620，首次 normal LoD apply iteration=1622；online_end_mapping_iteration=online_end_rotation_step=2671。
- `TrackingTime.txt`：TrackRGBD 均值 23.9378 ms、P95 40.2 ms，对应不含读图/限速/退出等待的 41.7749 FPS。
- 应用打印 FPS=22.7822，统计范围含读图、按输入时间戳 sleep 和 shutdown/join，**不是纯 Tracking FPS**。进程完整 wall time=65.23 s，含启动和退出。
- native GPU utilization 逐秒均值 16.47%、P95 24%、最大 29%；CPU time=132.41 s user + 15.98 s system，多线程 CPU 使用约 227%。

## GI 开销拆分

以下以 native 的完整 Mapping wall time 57.251 s 为分母，不是整进程耗时。

| 项目 | 累计 s | 占 Mapping wall |
|---|---:|---:|
| KF / MapPoint snapshot，含锁等待 | 29.7549 | 51.97% |
| Bank selection / 排序 / 去重 / 比较 | 3.3958 | 5.93% |
| Incremental render + aggregation | 0.9260 | 1.62% |
| GI topology 同步 | 0.1998 | 0.35% |
| 两次 periodic full refresh | 0.0734 | 0.13% |
| GI 合计 | 34.3498 | 60.00% |

真实路径：

```text
trainForOneIteration()
  → trainMappingIteration()
    → updateGlobalGIBank()                           每步执行，2671 次
      → 获取 mMutexMapUpdate
      → 遍历全部 ORB keyframes
      → 每 KF 调用 GetMapPoints()
        → 获取 mMutexFeatures
        → 遍历关联，isBad()，构造 std::set<MapPoint*>
      → 外层再次调用每个 MapPoint 的 isBad()
      → 提取 id，构造 snapshot
      → 释放 map lock
      → GlobalGIViewBank::update()
        → 每 KF 排序、去重、与旧集合比较
        → changed / membership 逻辑
```

关键点：未发生关联变化时可以跳过 bank selection 的后续工作，但仍已付出完整 snapshot 成本。`refresh_interval=1000` 只控制 periodic render refresh，**不会使 snapshot 变成每 1000 步一次**。目标 views=72 也不限制 snapshot 对全部 240 KF 的扫描。

源码位置：[updateGlobalGIBank](<H:/AIProjects/cartgs优化更细致版/mycartgs/src/gaussian_mapper.cpp:2475>)、[bank update](<H:/AIProjects/cartgs优化更细致版/mycartgs/include/global_gi_bank.h:47>)、[GetMapPoints](<H:/AIProjects/cartgs优化更细致版/mycartgs/third_party/ORB-SLAM3/src/KeyFrame.cc:405>)、[MapPoint::isBad](<H:/AIProjects/cartgs优化更细致版/mycartgs/third_party/ORB-SLAM3/src/MapPoint.cc:373>)。

OSRT 查询通过调用链 symbol 精确过滤 `GaussianMapper::updateGlobalGIBank()`：978 次 mutex wait，累计 7.97486782 s，最大 243.51 ms。该证据来自较重的首轮 trace；不能直接把它当成 native 的锁等待数值，也不能把 snapshot 的剩余时间全部视为纯 CPU 运算。

## TF 与非 Render 成本

native 中 841 次 TF optimizer update 只有 15 次实际 transformed_gaussians>0，每次恰为 1 个；其余 826 次没有实际 TF 变换。

这是现有逻辑决定的：只要 TF 存在且 mature_count>0，就加入 `regularizationLoss()`，不要求本次 transform_count>0。12 个 plane 共 1182720 个参数，正则化遍历整 plane，产生完整 grid gradients，再由共享 Adam 更新。正常 LoD 的 Master gradients / optimizer 同样不是随 compact population 自动缩小的 sparse optimizer。

TF-updated LoD 与同 cycle Full 的训练时间差均值=10.4075 ms、median=9.2434 ms。这同时混有 LoD photo/backward、Selector/GI teacher、grid regularization/Adam、不同 KF 和锁等待，**不是独立测出的纯 TF 开销**。全量正则化与 Tensor/API launch 是源码支持的优先候选，尚未单独给每个子阶段加标记或做等价实现对照。

CUDA-only trace：TF 更新 306 次，实际变换 14 个点；grid_sampler 336 次合计 2.2517 ms，全部 GEMM 10809 次合计 47.8025 ms，包含 Selector/其他矩阵运算。这排除了小规模 TF sampling/MLP 是主要 GPU 计算瓶颈的解释。

源码：[LoD regularization admission](<H:/AIProjects/cartgs优化更细致版/mycartgs/src/gaussian_mapper.cpp:2835>)、[TF regularization](<H:/AIProjects/cartgs优化更细致版/mycartgs/src/transform_field.cpp:85>)、[HexPlane](<H:/AIProjects/cartgs优化更细致版/mycartgs/src/hexplane_field.cpp>)。

## CUDA Trace 与计时限制

首轮全程序：919312 个 kernel，kernel duration 合计 6.4867 s；kernel/memcpy/memset 区间去重后的 GPU busy time=7.7534 s，GPU 工作时间窗约 60.64 s。Mapping 线程 cudaLaunchKernel 806309 次，API wall 合计 3.6675 s；cudaDeviceSynchronize + cudaEventSynchronize 全程序合计约 1.0702 s。这些 API 时间包含等待与 profiler 开销，不能与 kernel duration 直接相加，也不能称为删除计时后必然获得的收益。

CUDA-only trace：1262683 个 kernel，GPU kernel duration 合计 8.0282 s。GPU Event 区间会包含 CPU launch gaps，日志的 `cuda_elapsed_ms` 不是纯 kernel busy time。

`MappingProfiler` 每 20 个 iteration 采样一次，而固定 rotation 长度为 5，所以采样总落在 Full phase，`lod:NA` 是采样混叠，不表示 LoD 没有 GPU 工作。阶段间不能直接用这组 sampled CUDA 值公平比较；上面的 phase 表使用每步 RotationTimer 的同步计时。

首轮 `/usr/bin/time` 包住 nsys，88.49 s 包含 trace 处理，且 CPU/RSS 统计只反映 launcher，不能作为真实程序资源指标。native 的 65.23 s 才是本报告采用的进程 wall time。没有完成 PSNR/SSIM/LPIPS 或质量消融。

## 优化顺序建议

1. 推荐先处理 GI snapshot 的等价实现：评估用 `GetMapPointMatches()` 的 vector snapshot 替代 `GetMapPoints()` 的临时 std::set，保持外层 bad-point 过滤与 bank 中已有排序去重；核对 snapshot、bank membership、coverage 一致后再比较性能。这比首先重设计 GI bank 更小、更可逆。
2. 再考虑真实关联/剔除变化驱动的 KF dirty cache 与缩短地图锁持有时间。不能仅依赖 MapChangeIndex，因为原代码明确指出它未覆盖所有 association/culling 变化。
3. 给 LoD 子路径加独立范围，分离 Protection/budget、Selector/teacher、TF forward、grid regularization、backward、shared Adam。优先减少等价数学运算的 kernel launch、临时 tensor 和 dense grid 中间结果，不先改训练频率或 loss 权重。
4. 在 TF forward 稀疏而 regularization 更新密集的现状下，任何跳过 regularization 的方案都改变方法语义，需要另行批准和质量验证。本轮没有这样修改，也未改 Protection/readiness/GI 算法。

## 复现与校验

native 命令：

```bash
source /home/prmi/anaconda3/etc/profile.d/conda.sh
conda activate zwj
cd /home/prmi/zwj/mycartgs
./bin/tum_rgbd /home/prmi/zwj/mycartgs/third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt ./cfg/ORB_SLAM3/RGB-D/TUM/tum_freiburg1_room.yaml ./cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml /home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_room ./cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_freiburg1_room.txt /home/prmi/zwj/mycartgs/results/newscheme/tum_rgbd/room/perf_check_20261012_214513/native/model no_viewer --selector-config ./cfg/selector/fast.yaml
```

Nsight 使用 CUDA 11.8 附带的 `/usr/local/cuda-11.8/bin/nsys`，首轮 trace=cuda,nvtx,osrt，第三轮 trace=cuda,nvtx；均 sample=none、cpuctxsw=none、kill=none。CPU IP sampling 受 perf_event_open 权限限制；OSRT callchains 仍可用。

```bash
nsys stats --report cudaapisum,gpukernsum,gpumemtimesum,gpumemsizesum,osrtsum --format csv --output <evidence>/nsys <evidence>/trace.nsys-rep
```

服务器 SHA256：

```text
cfg/selector/fast.yaml       1f66ab767f64106427f88f5453793b9808789f9503da6ec54906919c0e72ac71
lib/libgaussian_mapper.so    37b4da143b3aa7e0bf0df65147b90439157b6785db76ba42652f67251a32434e
bin/tum_rgbd                356e3679e341854a79ba2c7f3afe4bd83bd25c5d026c6e275ffae4cb0f3bf105
```

诊断产物已保留。没有改动已有模型，没有启动额外尾优化，没有提交或推送 Git。
