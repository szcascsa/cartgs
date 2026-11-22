# Selector backward 专项实验

冻结基线：`4f056ec0096ba5cc4f6af3659638f3ae78be400e`。环境为 RTX3090 / CUDA11.8 / FP32 / TF32=false。默认 backend 仍为 legacy；本轮只改变选择 CUDA 时的 backward 实现。

`frozen/` 和 `baseline_sources.json` 保存起点源码/hash；`baseline_evidence.tar.gz` 包含最初独占八规模测量和四规模 Nsight trace。`prepare_profile.py` 从冻结副本生成 NVTX 诊断源码；加 `--candidate` 则生成当前实现的诊断副本，不修改产品源码。

## 验收与计时

`gate.py STRATEGY isolated|training|all` 复用 direct parity oracle，计算/目标/容差不变。每次使用新输出目录；可用 `SELECTOR_DIAGNOSTIC_TAG` 添加后缀。除了原 direct tolerance gate，本轮还要求 aggregate `all_exact=true`；warp tree 虽满足 tolerance，仍被拒绝。

`benchmark.cpp` 复用已验收 benchmark 的 CUDA event / synchronized host wall 定义。每阶段 100 warmup + 500 measurements；八个 M，forward、retained-graph backward、fresh-graph F+B。返回 XYZ/q/scale 与 18 参数梯度；完整 ratio/raw input 梯度由 correctness harness 验收。

`prepare_caller.py` 生成仅诊断 executable 使用的 autograd 包装，读取 `SELECTOR_DIAGNOSTIC_BACKWARD` 以便同 binary A/B；它没有进入生产源码或产品配置。`backward_caller` 定义自己的 network/backward，动态链接原有 mapper/Adam/Guard。日志与 `nm -D` 核验使用的 strategy。structural caller 包含 input activation、production probability、ST、Protection/compact/gather、ratio/photo-opacity VJP、Selector update；不包含 rasterizer、TF 网络、Gaussian optimizer 或完整 Mapping。

`run_final.sh 6` 依次重新构建、检查小 batch/尾部边界、Native CD/DSD/Replay/Guard、完整 direct parity 与 7×1000 steps、八规模 benchmark、M219969 两次额外反向顺序复测、caller 八规模/room、四规模 Nsight before/after 及 row-ratio profile。GPU 实验顺序执行，编译不与最终计时并行。已有输出目录会使脚本失败，避免混合 cohort；中断时从日志指定的位置恢复，不覆盖已完成证据。

## 重现环境

隔离目录：`/home/prmi/zwj/selector-cuda-20261015`；生产数据/LibTorch 从 `/home/prmi/zwj/mycartgs` 只读复用。Python Torch ABI=0，Native LibTorch ABI=1；分别编译，不能混用动态库。

```bash
export PATH=/home/prmi/anaconda3/envs/zwj/bin:/home/prmi/.local/bin:/usr/local/cuda-11.8/bin:$PATH
export CUDA_HOME=/usr/local/cuda-11.8 CXX=/usr/bin/g++-9 TORCH_CUDA_ARCH_LIST=8.6 MAX_JOBS=2
export TORCH_EXTENSIONS_DIR=/home/prmi/zwj/selector-cuda-20261015/python_extensions
python prepare_profile.py
python prepare_caller.py
```

CMake 指定 Native Torch、GCC9/CUDA11.8、第三方 OpenCV/Eigen/jsoncpp 的现有安装；参见 `run_baseline.sh` 和各 configure log。Nsight Compute 返回 `ERR_NVGPUCTRPERM`，没有可用的 DRAM/occupancy counter；Nsight Systems 有 kernel active/span/count/register/shared-memory 数据。估算 Tensor 字节数必须与实测 counter 分开。

## 候选与证据

- 1：原始生产 CUDA backward。
- 5：直接按分支生成 final dX，单独 ReLU；提交 `434c838`。
- 6：直接生成分支梯度并融合 ReLU；提交 `474e8e0`，后续加入 M≤16 的原实现分派。
- 7：warp tree 窄 dX，新增输入梯度误差；拒绝。
- 8：顺序 FMA warp 窄 dX，七快照/7000 步 exact，但速度回退；拒绝。
- 9：shared tile 三分支融合，七快照/7000 步 exact，但小 batch 不 exact，独占复测仍慢；拒绝。

7/8/9 不存在于产品 enum/hot path；补丁相对 `474e8e0`，见 `rejected_candidates.json`。原始探索候选归档与最终归档分开，不能用含 CPU 编译干扰的 `measure_9` 宣称性能。

显存按用户最新要求仅记录，不作优化目标或 gate。没有运行在线 SLAM/equal-time/多 seed 质量实验。

最终小 batch 修正为 `33ebe07`。`final_evidence.tar.gz` 包含最终318,118项 exact gate、7000步训练、1408项边界比较、八规模/重复测量、caller、Nsight与同进程交替paired A/B。压缩在GPU测量全部完成后执行。首次顺序F+B波动较大，因此追加 `run_paired.sh`；每个strategy同样100 warmup+500 measurements，主批原始数据未丢弃。

首次 `run_baseline.sh` 使用当时冻结的旧 binary，SHA保存在baseline证据中；不要把后来重新编译的revised/network_benchmark当成这个旧binary。新的对照通过 `benchmark.cpp` 显式选择strategy1与6，在同一binary中测量。生成最终汇总/报告使用 `summarize.py`、`write_report.py`，压缩归档用 `package_final.py`。Native构建与GPU计时不并行；输出目录冲突时保留既有证据并另起cohort。
