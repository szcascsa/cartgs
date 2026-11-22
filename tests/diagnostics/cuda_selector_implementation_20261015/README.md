# Selector CUDA 双后端验收

本目录的 `selector_parity` 使用原始生产 `GumbelNetwork` 为 oracle。候选只在显式实验构建中运行，生产默认实现保持 legacy。输入为 FP32、原始 18 个 parameter tensors，不修改 checkpoint 或 optimizer 语义。

## 构建与运行

服务器独立目录：`/home/prmi/zwj/selector-cuda-20261015`。第三方 LibTorch 只读引用原项目；不替换生产 `bin/lib`。

```bash
export PATH=/usr/local/cuda-11.8/bin:$PATH
CMAKE=/home/prmi/.local/lib/python3.13/site-packages/cmake/data/bin/cmake
"$CMAKE" -S tests/diagnostics/cuda_selector_implementation_20261015 \
  -B tests/diagnostics/cuda_selector_implementation_20261015/build-candidates \
  -DSELECTOR_CUDA_CANDIDATES=ON -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-9 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-11.8/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-9 -DCMAKE_CUDA_ARCHITECTURES=86 \
  -DTorch_DIR=/home/prmi/zwj/mycartgs/third_party/libtorch-cu118/libtorch/share/cmake/Torch
bash tests/diagnostics/cuda_selector_implementation_20261015/run_parity.sh
```

`SELECTOR_CUDA_CANDIDATES=OFF` 可只构建 legacy oracle。`--candidate legacy|cublas|warp` 显式选择诊断实现，默认 legacy。`--device cpu` 可用于 legacy harness 调试；CPU 自对照不构成 CUDA 验收。

初始 near-tie 夹具通过的保守候选为 `--candidate cublas_strict`：融合 branch bias/ReLU，保留完整 final GEMM 和 ATen bias reduction。最终已训练 desk checkpoint 的 row-ratio 固定 Gumbel noise 改变了 58663 个 hard class，因此该候选也未通过全面验收；strict 仅为候选名称。分块 final GEMM 与 warp 候选同样因 near-tie hard mask 不同被拒绝。`cublas_strict` 保留 `[M,128]` packed workspace，不能宣称该内存已消除。

主 CMake 通过 `CARTGS_EXPERIMENTAL_SELECTOR_CUDA=ON` 编译实验入口。该选项默认 OFF；普通构建显式 `Selector.backend=cuda` 会拒绝启动。实验构建仍默认 `Selector.backend=legacy`，A/B 配置显式设 cuda。cloneFrozen 保留 backend，但 parameter names/identity/archive 不变。

Python evaluation 需要显式 `CARTGS_EXPERIMENTAL_SELECTOR_CUDA=1` 和配置 `Selector.backend=cuda`。binding 使用当前 Python Torch ABI 独立编译，不混用 Native LibTorch ABI。`python_parity.py` 比较 inference、完整一阶梯度和 checkpoint keys。

`--fixture` 读取旧审计的真实属性和 Mature global IDs，只作为固定网络输入；不把旧 Mature 分类写回当前连续成熟度/elastic entry 生产逻辑。`--checkpoint` 读取原 C++ archive，18 tensor 名称和形状保持一致。

## 验收记录

- 每个 CSV 行包含 max abs、RMS、p99、worst global Gaussian ID、不同元素数和通过状态。
- `stage_0..7` 是四分支两层 ReLU 输出，`stage_8` 是 logits。
- hard、compact 和 RNG 按 exact 检查；其余 `atol=2e-6, rtol=2e-4`，不自动放宽。
- 同时记录 legacy 同图输入重复 VJP 的数值下限。
- fixed noise、不同温度、同 RNG state/next draw、scalar/row ratio、空输入、ReLU 零点均覆盖。
- 梯度夹具覆盖 logits VJP、ratio、GI、selected opacity VJP、combined、detach、Guard uniform、CD L1、DSD BCE、外层 normalize/exp 链。
- 这些是结构梯度测试，不能冒充真实 rasterizer/TF、完整 Guard 事务、Replay scheduler 或在线质量测试。
- 多步训练每步检查 hard、loss、logits、全部参数/梯度和 Adam m/v/step；显式区分 undefined grad 与 allocated-zero grad，遇到本步差异立即停止并保留证据。
- `near_tie` 使用固定 `noise=-legacy_logits`，确保 legacy 两类恰好打平，检查 argmax 第一类优先规则对数值变化的敏感性。

`phase1_verified_*` 是加入 near-tie 后的 legacy 自对照；`phase1_final_*` 没有 near-tie，不能作为其通过证据。`phase1_checkpoint_1000` 等更早文件属于初版 harness，不替代 verified 记录；早期训练输入为 synthetic，strict 记录使用实际 checkpoint 属性的前 257 个 Mature 点。

`analyze.py` 汇总 CSV 到 `acceptance_status.json`。生产验收标志固定为 false，不会依据部分检查自动开启 CUDA。

## 性能边界

`network_benchmark` 仅测 MLP forward、完整一阶 VJP、forward+backward，100 warmup + 500 measurement，分别记录 CUDA event 和同步 wall mean/p50/p95。没有测量的 caller、Guard、Mapping 项必须标为未测；不能用这些 MLP 数值替代。

未通过 parity 的 candidate 即使更快，也属于被拒绝的实验实现，不进入生产。所有正式测量应在没有其它 GPU 任务时执行，legacy 与 candidate 使用同一个 binary、checkpoint、输入和 math 设置。

## 当前分支调用链

用户已明确选择当前分支 `forwardProbability → elasticSelectorProbability → cached_uniform ST` 为 online A/B oracle。原始 Gumbel / Guard uniform 单独保留数学测试。不能把审计冻结版本的 Mature/Protection 或 Gumbel sampling 恢复到当前 Mapping。

`caller_replay` 使用当前 room checkpoint 的真实属性、连续 maturity、保存 uniform、三个相机和真实 GT。最终 `benchmark` 启用 Gaussian 六类参数的 requires_grad，计时内重建 quaternion normalization、scale/opacity activation，验证 XYZ/opacity/raw-scale/raw-quaternion gradient 均存在。包括概率混合、ST、compact、六类 gather、opacity VJP、ratio、完整输入 backward 和 Adam；legacy 使用原 Adam，CUDA 复用已有通用 CUDA Adam kernel。排除真实 TF/rasterizer/GI/Gaussian optimizer step，因此仍是 structural caller，不能称完整 Mapping。早期未覆盖全部 Gaussian 输入 gradient 的 caller 数字不作为最终结论。

`grid AUDIT_FIXTURE CHECKPOINT` 对 1k/5k/10k/20k/50k/100k/200k/219969 行分别运行相同 100+500 measurement。使用真实审计属性和 checkpoint、maturity=1、entry=0，包含完整 Gaussian 输入 gradient；Gumbel/ST 独立测量保持原实现与同 logits/RNG state。每阶段 peak 另存 `.memory.csv`。

`replay 1000` 冻结 Gaussian/TF，每步训练 Selector，使用同一真实 rasterizer opacity VJP 驱动两图，排除 rasterizer atomic reduction 的随机差异。每步 exact 检查 probability/hard/global IDs/grad/parameter/Adam；`fixed_map_selected_ids.bin` 保存每步 int64 step、int64 count、int64 global IDs。最终三帧×四 ratios 比较图像及 PSNR/SSIM/LPIPS。该范围不能替代 joint Gaussian/TF 长训练质量验收。

`joint 1000` 则实际更新 Gaussian、Selector、TF：固定拓扑、maturity/uniform、三帧/四 ratio 顺序，Gaussian/TF LR=1e-5，Selector LR=0.001，两后端相同。共享未修改 renderer/TF 的真实外部 VJP，独立计算两个 Selector 的参数/输入 VJP 并与外部贡献相加，隔离原有 atomic reduction 噪声。每步比较完整参数/梯度/Adam/IDs，最终另有 12 组 exact 图像/质量。该受控 VJP replay 不冒充两份独立 rasterizer backward 的确定性证明。

`optimizer_candidate` 覆盖 direct/preview、默认与非默认 betas/eps/weight decay，四种 mode 各 1000 步；`guard_transaction` 用真实 Selector 梯度及 Gaussian/TF 代理参数组覆盖六类联合事务。`operator_trace` 在 Native LibTorch 中检查 ATen hot-path 接管；`profile` 的 Nsight 范围仍是 structural caller。`parse_caller_nsight.py` 汇总每 scope 的 kernels、active/span/gap、copy 和同步 API。

`online_ab.py` 使用同一个 online-only 测试 binary，在 SLAM shutdown 时结束，不运行 post-SLAM tail；这与已有 TF acceptance driver 一致。每个场景先跑三次 legacy，再跑三次 CUDA，初始 seed0 相同，记录 SLAM 非确定性；两组只改 Selector.backend。隔离 evaluator 保持渲染和 metric 计算，使用 Agg 并跳过图像文件写入，避免数 GB 中间图片；模型和 metric 均保留。

最终在线目录为 `results/selector_online_ab_final_20261015`。先前 `selector_online_ab_20261015` 是尚未接入 fused Adam 的初步运行，不纳入最终同 binary A/B。`summarize_online.py` 输出原始指标及 legacy std/min/max、CUDA-Legacy delta/std，也保留日志里首次输入/输出分歧的可观测位置。

`online_progress.py` 只读队列和日志，不导入 Torch。`online_checkpoint_parity.py` 等待全部在线运行结束，再对 24 个实际 checkpoint×四 ratio 比较概率误差统计、当前 cached-uniform mask/compact IDs、独立 seed0 Bernoulli、RNG state/next draw 和双侧 near-tie。该测试使用同一 checkpoint 的两后端，不把不同在线输入产生的 checkpoint 相互比较。

`collect_online_evidence.py` 要求 24 次 train/eval 全成功、120 套质量指标齐全且有限，复核仅 backend 不同、binary/library 不变、原生产文件未变、legacy/OFF 默认，并保存原始指标、配置、日志摘录及完整文件哈希。大模型和完整日志保留服务器；`online_evidence/` 为可审查的小型证据副本。`update_report_online.py` 从完整 `online_summary.json` 生成中文报告和逐项质量对照 CSV，不自行放行生产。

`online_stage_probe.py/.cpp` 定位实际 checkpoint 的新失败：最早差异来自 row-ratio 第一层 one-column GEMM 的转置/布局；单层非转置参考误差为 0，同输入 final GEMM 误差为 0。该参考只用于定位，未替换冻结的实验实现。96 组 checkpoint 的实际 cached-uniform masks/compact IDs 和 RNG exact，但两个双侧 near-tie case 失败；原始 CSV 与报告保留失败状态。
