# Selector CUDA 修订验收报告

后续验收范围已调整为同输入的 Selector 直接数学/训练 parity，且已得到 `CUDA_SELECTOR_CORRECTNESS = PASS`，见 `CUDA_SELECTOR_DIRECT_PARITY_REPORT.md`。下文保留上一轮的实验与较宽生产验收计划；其 online 未完成项不再作为当前 Selector 算法 correctness 的否决条件。

分支：`cuda-selector`；本轮基线：`63a7bf9`。旧报告保留为历史记录，本轮以用户修订标准判断，不再以 all-tie exact 或三个 legacy 样本的 min/max containment 否决实现。

诊断目录：`tests/diagnostics/cuda_selector_revised_20261015`（D）；性能与验收实验在隔离目录 `/home/prmi/zwj/selector-cuda-20261015` 执行。随后按用户要求将当前源码同步至 `/home/prmi/zwj/mycartgs`，未执行编译，未替换 binary 或模型。

## 1. Numerical parity

ratio 第一层由转置 `CUBLAS_OP_T, lda=1` 修为非转置 `CUBLAS_OP_N, lda=32`，保持 `[32,1]` 参数、checkpoint、bias、ReLU 和 FP32/TF32=false。修复提交：`a5c8903`。

24 个实际 checkpoint × 四 ratio × scalar/row，共 192 组：8 层激活、logits、概率以及四个输入/18 参数 isolated VJP 共 8064 条记录，gate failure=0；本次结果实际全部 exact。forward max abs/RMS=0，gradient relative L2=0、cosine=1。正常容差合同仍为用户给定的数值门槛。

## 2. Real selection parity

相同 192 组输入，actual cached-uniform 与六个 seed（0/1/2/42/123/2026）的正常两类 Gumbel 全部 hard disagreement=0。随机生成继续使用 legacy exponential CUDA generator；stream、state 和 next draw 一致。D/selection_parity.json 保存 margin 分布、1e-7/5e-7/1e-6 敏感性和每组 count。

## 3. Adversarial stress test

all-tie 仅作为诊断，记录 flip、ULP、first differing layer，不能代替真实选择验收。修复后所测 scalar/row case 的 activations/logits/probability 全部 exact，all-tie flip=0；这不是对所有未来浮点边界输入的 bitwise 合同。

## 4. Deterministic training parity

修复后 Native 1000 步网络训练通过；room N=109564 的联合 Gaussian/Selector/TF 受控 VJP replay 在 1/10/100/1000 步保持参数、梯度、Adam、选择 exact，最终 12 组图像/PSNR/SSIM/LPIPS exact。

该 replay 共享未修改 renderer/TF 的外部 VJP，独立计算两个 Selector/input VJP。独立 renderer 的非确定性将按 legacy repeat baseline 单独判断，不再要求比 legacy 自身更一致。

## 5. Microbenchmark

最终采用 fused activation backward，合并 split 与第二层 ReLU VJP，并保留 legacy bias reduction。192 组真实 checkpoint 的 isolated VJP 再次全部 exact。batched dW/grouped branch 大幅变慢；fused bias 虽更快，但真实 checkpoint 有 19 项梯度超限，因此不采用。

本轮最终冻结源码、同一 benchmark binary，所有八个 M 均 100 warmup + 500 measurement。M=219969 的同步 wall mean/p50/p95：

| 阶段 | legacy ms | CUDA ms | mean 倍率 |
|---|---|---|---:|
| forward | 3.96841 / 3.96543 / 3.99407 | 2.10462 / 2.10347 / 2.11741 | 1.886× |
| backward | 2.30440 / 2.30120 / 2.32844 | 2.25704 / 2.25557 / 2.27367 | 1.021× |
| F+B | 6.20202 / 6.20322 / 6.22302 | 4.29233 / 4.29164 / 4.31267 | 1.445× |

room structural caller：4.30516→2.95658 ms，1.456×；M=219969 的统一网格 caller：7.19214→5.03062 ms，1.430×。caller 含 Gaussian 输入梯度和 Selector Adam，排除 rasterizer/TF/GI/Gaussian optimizer，不能当作完整 Mapping 收益。

最终 Native 网络 1000 步、M=219969、Adam 1000 步、六类联合 Guard 事务、room 受控联合 VJP 1000 步均通过；CD/DSD/Replay/historical renderer/GradientConstraint 集成 5/5。Nsight 网络 kernel 数：legacy 18+42，CUDA 18+43；CUDA backward active time 约 2.187 ms，legacy 约 2.210 ms。原始数据见 D/final_measurement 和 D/replay_room。

## 6. Equal-updates

已增加独立完整 renderer 的 Gaussian/Selector/TF 训练夹具；相同初始图、三相机、四 ratio、cached uniform，1000 Full+LoD updates，不做拓扑变化。与共享 renderer VJP 的确定性测试分开。

room 初次三方对照：legacy、legacy repeat、CUDA 的训练 wall 为 10.5814 / 10.5758 / 8.8612 s，CUDA 对初次 legacy 为 1.194×。legacy repeat 对初次 legacy 的平均 hard mismatch=0.1629%、p99=0.9705%；CUDA 为 0.1346%、p99=0.9274%。完整图的原生 reduction 噪声能在两个 legacy 训练之间累积；这些数据不替代 isolated Selector 的严格梯度/训练验收。

36 个质量比较中有 5 项超出 `max(engineering margin, 2×单次 legacy repeat difference)`，该 pilot **未通过**，不把质量改善方向的差异当作等价通过。不能由三个有噪声样本判断系统性 backend 漂移。后续扩大重复实验尚未进行；desk/xyz/long 独立 equal-updates 尚未运行。原始数据保留在隔离服务器 D/fixed_map/room。

## 7. Equal-work online

预注册目标 room=5000、desk=3000、xyz=15000、long=8000 committed Mapping updates。测试 driver 和 orchestrator 已建立，核对实际 Gaussian commit 数，不将尝试次数冒充 commit。因前一阶段未通过而未运行，无本轮结果。

## 8. Equal-time online

Mapping work 预算 room=50s、desk=22s、xyz=130s、long=75s，依据旧 baseline 取共同可达值；按完整 Mapping cycle 边界停止并记录 overshoot。尚未运行。

## 9. Statistical online quality

预注册每场景每 backend 五次完整 online，algorithm seeds=0/1/2/42/123；分别记录 run/process ID、输入/config/binary hash 和 stop，按 seed 配对并交替 backend 顺序。尚未运行。

报告 mean/std、raw difference、95% bootstrap CI、difference/legacy std 与 Hedges g；primary 为配对 bootstrap，另报独立 resampling 敏感性。non-inferiority margins 和方法已写入 D/protocol.json，不能根据结果临时扩大。三个旧样本的 min/max 不参与本轮 gate。

## 10. Memory

保存 packed feature 的第二层 view，避免重复持有属性分支 `[M,32]` 激活。M=219969 的普通 F+B measurement incremental peak：legacy=397195264 bytes，CUDA=381377024 bytes，CUDA 少约 15.09 MiB，满足 legacy+32MiB 目标。单独 profiler capture 的 allocator 基线不同，另保留原值，不与普通测量混用。

## 11. Remaining limitations

用户随后要求把源码更新到 `/home/prmi/zwj/mycartgs`，自行编译和测试，因此后续自动质量实验停止。没有把尚未运行的实验标为通过；新 gate 未全部通过，仍保留 legacy 默认。源码交接不替换已有 binary，不执行编译。

当前可确认：ratio 问题修复；正常随机选择 mismatch=0；无 mismatch 可用于判断边界归属；all-tie flip=0 且仅作为诊断；受控 1000 步无累积漂移；backward 已达到略快于 legacy；普通网络 F+B 显存下降。equal-work 节约时间、equal-work 最终质量、equal-time 额外更新及其质量影响、5+5 online non-inferiority 均无本轮有效结论，不能建议默认切换 CUDA。显式实验使用方式见 `SELECTOR_CUDA_USAGE.md`。
