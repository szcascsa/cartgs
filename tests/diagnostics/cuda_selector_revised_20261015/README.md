# Selector CUDA 修订验收

基线为 `63a7bf9`，沿用现有 CUDA forward/custom backward/Adam/Guard 实现。旧报告和旧实验数据保留，本目录记录修订后的独立结果。

执行顺序：ratio GEMM 修复 → 24 checkpoint×四 ratio×scalar/row 的正常数值/isolated VJP → 六个真实随机 seed 的 Gumbel 与实际 cached-uniform 选择 → 1000 步 replay → backward/workspace 候选 → 四场景 equal-updates → equal-work online → equal-time online → 每场景 5+5 次统计 non-inferiority。

正常 forward 使用 max abs≤5e-7、RMS≤5e-8；isolated gradient 使用 relative L2≤1e-4、cosine≥0.99999。实际选择的 count relative difference 和全人口 hard-ID disagreement 均≤0.05%，至少99%的 disagreement 在 `max(1e-6,4×observed probability max error)` 内；另存 1e-7/5e-7/1e-6 的敏感性统计。

`parity.py numerical` 比较八层激活、logits、概率、四类输入/18 参数 VJP 和 legacy 自重复。`parity.py selection` 使用实际 metadata/cached uniform、seed 0/1/2/42/123/2026 的 legacy exponential CUDA generator、相同 noise/state/next draw，并独立记录 all-tie 的 flip/ULP/首个不同层。all-tie 仅诊断，不参与 gate。

全程 FP32、TF32=false，保留 continuous maturity / elastic entry / cached-uniform 的当前调用链及 `Selector.backend=legacy|cuda`。默认实现需等待完整新验收。
