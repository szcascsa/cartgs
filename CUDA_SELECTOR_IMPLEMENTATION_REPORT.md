# Selector CUDA backend 实施与验收报告

日期：2026-10-15；目标：RTX 3090 / CUDA 11.8；分支：`cuda-selector`。

## 结论与验收边界

**保留 `legacy` 默认，CUDA 仅供显式实验构建和 A/B。** `CARTGS_EXPERIMENTAL_SELECTOR_CUDA` 默认 OFF；实验构建仍需配置 `Selector.backend: "cuda"`。普通构建拒绝启用尚未验收的 CUDA backend。

**本轮没有满足全部硬验收的 CUDA 候选。** strict cuBLAS 在初始审计输入、动态规模和受控 1000 步 replay 中通过，但最终 24 个在线 checkpoint 的 96 组检查中，两个 row-ratio case 的双侧 near-tie 失败。进一步固定 Gumbel noise 的定位实验实际改变 **58663 个 hard class**。该候选也不具备生产替换资格，不能把“strict”名称当成验收结论。实际 cached-uniform masks/compact IDs 在这 96 组中全部 exact，仍不能免除 fixed-noise 失败。

实现了 cuBLAS 分块 final、H32 warp 和 strict cuBLAS 三个候选。前两个候选在 near-tie 夹具中改变 hard IDs，均被拒绝。实验入口只接入 strict cuBLAS：融合 branch bias/ReLU，保留完整 128 项 final GEMM 与 legacy reduction 顺序；为此保留 `[M,128]` packed workspace。

真实审计 checkpoint、M=219969、本次同一 binary 的 wall mean：forward **3.97066 → 2.11151 ms，1.880×**；backward **2.30698 → 2.38871 ms，0.966×**；forward+backward **6.20176 → 4.42894 ms，1.400×**。大 M backward 略慢，不宣称其加速。

最终 caller、Nsight 与在线统计在下文独立报告，不能由上述 MLP 数字推算。尚未满足的验收项不因局部速度或部分 parity 通过而自动放行。

room/desk/xyz/long 各完成 3 次 legacy + 3 次 CUDA，共 24 次独立运行、120 套质量指标。完整 Mapping 平均周期的观测倍率依次为 **1.143× / 1.195× / 1.174× / 1.161×**，但更新数和人口不同；相同工作量的完整 Mapping 总 wall 收益尚未证明。60 项图像质量均值比较中 **26 项超出 legacy 三次重复的观测范围**，不能宣称最终质量等价。

## 版本、环境与复现

- 本地基线：`18ef466d4856cf27ba23a15061a96f6b33a85f59`。
- 审计参照：`ORIGINAL_SELECTOR_CUDA_AUDIT.md`，SHA-256 `a6b2432053499065aa96874d703975cb935836098db3d5f75880ba871ea68b4b`。旧审计耗时仅用于确定优化优先级，不代替本轮 A/B。
- 实现代码最后改变于 `04d2fe2`；代码/验收证据最终 commit：`6754a884a614a8a494f052d713b048775fafd310`。本报告单独提交，避免自引用 hash；全部为本地提交，未 push。
- 服务器原生产目录：`/home/prmi/zwj/mycartgs`；隔离实验：`/home/prmi/zwj/selector-cuda-20261015`。
- 诊断目录：`tests/diagnostics/cuda_selector_implementation_20261015`，以下简称 D。
- RTX 3090 / driver 535.230.02 / sm86；nvcc 11.8.89；GCC 9.4；Nsight Systems 2022.4.2.1。
- Native LibTorch 2.3.1 **ABI=1**；Python Torch 2.3.1+cu118 **ABI=0**，分别编译，不能混用动态库。
- 全 FP32，`allowTF32CuBLAS=false`，没有 FP16/BF16、TF32 切换、CUDA Graph、Top-K 或固定 K。
- 原生产源码、配置、bin/lib 的前后 SHA-256 保存在环境清单；本任务不替换生产目录，不推送远端 Git。

远端原项目没有可用的自身 Git HEAD；上述 baseline 是本地仓库基线，不能冒充服务器历史版本。远端对照的身份由冻结源码、配置和二进制 SHA-256 明确记录。

网络复测的 `round2_environment_before.json` / `round2_environment_after.json` 中，运行源码和 benchmark/parity binary hash 不变。诊断 `CMakeLists.txt` 在独立增加 optimizer target 时变化，清单保留该差异；该变化不重建正在计时的网络 binary。不能把整份清单称为完全未变。

`round2_*` 是最终冻结网络源码复测，替代较早同名测量；源码 `.cu` SHA-256 为 `f507737b36893bb423521f36e5ddb8471e45e7550e07d4b54f37fb7c12e62aee`。最终在线 binary 和 mapper library 的 hash 另由在线 `status.json` 固定。

复现脚本：D/`run_final_network.sh`、`build_integration.sh`、`run_final_integration.sh`、`online_ab.py`。大体积 `.nsys-rep`、SQLite、模型及 selected-ID trace 保留在隔离服务器；可读 CSV/JSON/日志保存到本地并提交。

## 修改范围和未改变的语义

| 文件 | 实际变化 |
|---|---|
| `include/selector_cuda.h`、`src/selector_cuda.cpp`、`src/selector_cuda.cu` | 三个 FP32 网络候选、thin autograd、custom VJP、cuBLAS dW |
| `improvements/selector/gumbel_network.h` | backend 路由、冻结 teacher 继承 backend，18 参数名称和注册顺序不变 |
| `include/gaussian_mapper.h`、`src/gaussian_mapper.cpp` | 读取/保存 `Selector.backend`；实验 CUDA 的 Selector optimizer group 加入既有 CUDA direct/preview 事务 |
| `src/change_distill.cpp`、`src/drift_stability.cpp` | CUDA Selector 的独立局部 Adam 使用同一个已验收数值候选，保持状态隔离 |
| `python/selector.py`、`selector_cuda_backend.py`、`selector_cuda_bindings.cpp`、`run.py` | Python ABI 独立 binding、实验开关、评估 backend 路由 |
| `CMakeLists.txt`、四个既有集成测试、D | 默认关闭的构建开关、A/B 夹具、性能/质量证据 |

按用户后续确认，当前分支的 online oracle 是 **continuous maturity / elastic entry / cached-uniform ST**。没有把旧审计的离散 Mature 分类或正常 Gumbel caller 恢复到当前 Mapping。原始两类 Gumbel ST、`forwardHardWithUniform` 分别单独验证。

四分支 `3/4/3/1 → 32 → ReLU → 32 → ReLU`、final `128 → 2`、4962 参数、18 tensors、输入 normalize/exp、ratio/probability、损失、detach、Protection/GI/Guard/Replay/CD/DSD 定义、LR/频率均保持原规则。ST/RNG/softmax 和 compact 调用继续使用 legacy，实现没有以 bool mask 替代 ST 梯度。

## CUDA 架构与候选淘汰

### cuBLAS 分块 final

直接 cuBLASLt branch GEMM + bias/ReLU epilogue；ratio MLP 一次计算；final 四段乘加，不生成 `[M,128]` concat。backward 由 activation VJP、cuBLAS dX/dW、custom bias reduction 组成，避免逐点 atomicAdd weights。

M=219969 forward 为 1.48011 ms，F+B 为 4.14685 ms，但 `noise=-legacy_logits` 使 legacy 恰好打平时，scalar ratio 有 **89492** 个 hard ID 不同，row ratio 有 **31355** 个不同。定位为 final 累加分组/舍入顺序变化；普通随机噪声或 allclose 通过不足以放行。该候选保留为诊断，生产路由不可选。

### H32 warp

一 warp/Gaussian，32 lanes 对应 hidden neurons；三个属性分支和 final 在同一 kernel 内，ratio 预计算；readonly weight loads、shuffle、显式 FMA。forward kernel 少，但大 M forward 为 **5.14176 ms**，比 legacy 慢，near-tie 也失败。没有继续添加 shared-memory tile 变体，因为当前候选已同时失去 parity 和大 M 性能优势。

### strict cuBLAS

branch 使用直接 cuBLASLt bias/ReLU，final 使用与 legacy 相同的完整 128 项 GEMM；custom float4 pack 替代 LibTorch cat 调度，**仍保留等大的 packed Tensor**。one-column ratio 第一层保留 legacy SGEMM 路径；bias reduction 保留 ATen sum 的树与布局。

thin autograd 保存原输入、18 参数及激活，不在 backward 重建 LibTorch MLP graph。返回 XYZ、activated quaternion、activated scale、ratio 和全部参数 VJP；上层 renderer/Gaussian/TF 继续走既有图。ReLU 零点单独验收。

Lt plan/workspace 按 device、stream、层布局、math mode 保留当前 M，避免在线每次 append/prune 都永久增加一个 1 MiB cache 项。当前实现仍会产生保存激活、packed/full-gradient 等 workspace，显存并未全面下降。

Normal/Replay 等传入 `[1,1]` ratio 的路径只计算一次 ratio MLP，再在 pack 中广播。CD 和 Python 既有接口仍可传入 `[M,1]`，严格候选保留对应逐行 GEMM；没有为了压缩这条路径而改动其梯度 reduction 顺序。因此“所有 invocation 的 ratio 均只计算一次”尚未全面实现。

### 阶段与提交

| 阶段 | 本地 commit | 结果 |
|---|---|---|
| Legacy oracle / near-tie 硬门槛 | `e79e342`、`d88ed61` | 建立 forward、noise、gradient、Adam 自动比较，禁止未验收默认切换 |
| cuBLAS / warp / custom VJP | `7f131e6` | 两个原始候选都实现；mask 失败，不能进入 caller |
| Strict GEMM / reduction | `29e580d` | 保留严格计算顺序的第三候选 |
| Native / Python 实验入口 | `ec672d2` | 双 backend、checkpoint 和编译/运行 gate |
| Selector Adam / Guard | `04d2fe2` | 复用通用 CUDA kernel，保持联合事务 |
| 冻结网络性能 / 规模网格 | `101e1f7` | 同 binary 的 100 warmup + 500 measurement |
| 固定图 / caller | `82d11aa`、`4ec7381` | 联合 VJP replay；纠正 caller 的 Gaussian 输入梯度覆盖，并补实 Replay optimizer 路由 |
| 独立 renderer 梯度 | `ab17be2` | 记录 legacy 自重复误差边界，保留失败状态 |
| 在线 A/B / 最终证据 | `2c0f53b`、`6754a88` | 四场景各 3+3 次；真实已训练 checkpoint fixed-noise 失败，见在线节 |

probability/Gumbel/ST、Protection merge、CUB compact 均未替换。它们的实测成本低，strict backward 仍是更重要的后续优化对象；没有引入 custom RNG、固定 K 或改变采样轨迹来削减 launch。CUDA Adam 的 kernel 数下降单独报告，不能归功于 MLP 融合。

## 数学、RNG、训练与事务 parity

| 项目 | 已验证范围与结果 |
|---|---|
| 初始 Forward | 审计 checkpoint 与随机初始化，8 branch activations/logits/probability；scalar/row ratio、空输入、ReLU 零点；这些初始夹具的 strict max abs/RMS/p99 均为 0，无 worst ID（-1）；最终在线 checkpoint 的失败见后文 |
| 初始 Fixed noise | soft、ST、argmax、hard、compact global IDs/count，包含初始 near-tie；这些夹具 strict exact，不外推所有已训练参数 |
| Production RNG | 从相同 CUDA generator state 调用 legacy 噪声生成；noise、hard IDs、state、next draw exact；没有新增 custom RNG |
| Gradient | 同一 upstream 的全输入/18 参数；ratio/GI/photo-opacity/combined/detach、原 Guard uniform、CD L1、DSD BCE、normalize/exp 外链，误差均为 0；独立 renderer 图的不同结论见下文 |
| 多步 | 实际 checkpoint 前 257 个 Mature 点及随机初始化；1/10/100/1000 步逐步 hard/loss/logits/gradient/parameter/Adam exact |
| 动态 M | 额外测试非对齐和 benchmark 规模；逐规模 CSV 保留 max abs/RMS/p99/global ID，汇总见 D/final_evidence_summary.json |
| Python | 0/1/257/219969 行，概率、输入/18 参数一阶梯度 exact；18 checkpoint keys 和 roundtrip 不变 |
| 集成 | CD、DSD、joint Replay、真实 renderer historical gradient、GradientConstraint 5/5 通过；这不是四条路径的长期线上质量结论 |

初始 float 容差为 `atol=2e-6, rtol=2e-4`；hard/IDs/RNG 强制 exact。没有因 candidate 失败放宽容差。每项 CSV 包含 max abs、RMS、p99、worst global ID、unequal count；不只记录 allclose。

### 独立完整 renderer 梯度与 legacy 自身误差下限

另用同一真实 room checkpoint、三帧×四 ratio 独立构建三份图：legacy、legacy-repeat、CUDA。六类 Gaussian 参数和 TF 均保留 gradient；按照实际 mode0 先 `Full photo.backward()`，再 `LoD photo + ratio + GI.backward()`，没有共享这三份图的 renderer/TF VJP。

| 对照与边界 | 失败项 / 检查项（初始容差） | max abs | 最大 RMS |
|---|---:|---:|---:|
| legacy 自重复：Selector 调用前 Full 图像/loss | 0 / 24 | 0 | 0 |
| legacy 自重复：Selector 调用前 Gaussian gradient | 34 / 72 | 8.91060e-5 | 1.76754e-7 |
| legacy 自重复：完整 Gaussian gradient | 32 / 72 | 1.40188e-4 | 2.82397e-7 |
| legacy 自重复：Selector 18 参数 gradient | 0 / 216 | 7.45058e-9 | 1.51843e-9 |
| legacy 自重复：TF gradient | 18 / 312 | 8.24854e-5 | 5.92816e-5 |
| CUDA：完整 Gaussian gradient | 35 / 72 | 8.56072e-5 | 1.67821e-7 |
| CUDA：Selector 18 参数 gradient | 0 / 216 | 3.72529e-9 | 9.80987e-10 |
| CUDA：TF gradient | 14 / 312 | 8.31895e-5 | 5.99623e-5 |

三份图的 probability、hard、compact IDs、Full/LoD 图像和 loss 全部 exact。首次非零差异已经出现在**尚未调用 Selector 的原始 Full backward**，与原有 renderer reduction 的非确定性一致；这是直接的边界记录，不依据运行快慢推断原因。完整图的 Gaussian/TF gradient 在 legacy 自重复中也不能满足初始容差。

该实验**不标为通过**，也没有把 atol 改成 1e-4 来放行。固定 upstream 的网络测试及受控 VJP replay 用于隔离 Selector 本身；它们不冒充独立完整图的通过结果。原始逐 Tensor、p99、worst Gaussian ID 和失败状态见 D/renderer_gradient_parity.csv、renderer_gradient_summary.json，退出码为 1。

### Selector Adam / Guard

复用已有通用 CUDA Adam kernel，仅将 Selector 的 group 加入显式 CUDA 分组；参数对象、名称、checkpoint、每 Tensor step/m/v、has_grad/undefined grad、optimizer groups 和局部状态均仍由原 optimizer 持有。allocated-zero grad 继续推进状态，undefined grad 完全跳过。

direct / preview 各 1000 步：LR schedule、zero LR、逐 Tensor undefined、allocated-zero、候选 shrink/projection/discard；另测非默认 betas/eps/weight decay。最终结果见 D/optimizer_candidate_parity.csv。实现继续 `prepare → candidate → Guard → commit/discard`，没有 live update 后 rollback。

六类联合事务用 Gaussian/TF 代理参数组和真实 Selector 网络梯度测试：accept、shrink、projection、reject、fallback、nonfinite；决策、candidate 最终值、梯度、Adam 状态 exact。reject/nonfinite 保持所有组原参数和空 state，防止 Selector 先 commit。该代理组测试验证事务机制，真实 renderer 由历史梯度集成测试另行覆盖。Guard 的 m/v 仍由原 gradient 形成，不随 alpha/projection 重算。四种 Adam mode 合计 215984 项检查，无不一致。

## 网络 microbenchmark

本次同一 binary、同 checkpoint/属性，全部 M 均 100 warmup + 500 measurement。forward 单项在 NoGrad 下测量；backward 在保留的训练图上返回 XYZ/quaternion/scale 和 18 参数 VJP；F+B 测量完整一阶 autograd 图，使用 `autograd::grad` 返回梯度而不含 optimizer。真实 caller 另用 `loss.backward()` 和 Adam 计时。以下是同步 wall **mean**；完整 event/wall mean/p50/p95 在 D/round2_network_benchmark.csv。

| M | Forward legacy / strict ms | 倍率 | Backward legacy / strict ms | 倍率 | F+B legacy / strict ms | 倍率 |
|---:|---:|---:|---:|---:|---:|---:|
| 1000 | 0.2245 / 0.0903 | 2.487× | 0.6832 / 0.3601 | 1.897× | 1.0486 / 0.5254 | 1.996× |
| 5000 | 0.2277 / 0.1171 | 1.944× | 0.7282 / 0.6336 | 1.149× | 1.1389 / 0.7427 | 1.533× |
| 10000 | 0.2840 / 0.1487 | 1.910× | 0.7174 / 0.3936 | 1.823× | 1.1404 / 0.5660 | 2.015× |
| 20000 | 0.4431 / 0.2320 | 1.910× | 0.7095 / 0.3836 | 1.850× | 1.1121 / 0.5793 | 1.920× |
| 50000 | 1.0151 / 0.5231 | 1.941× | 0.7113 / 0.7182 | 0.990× | 1.6597 / 1.1822 | 1.404× |
| 100000 | 1.8934 / 0.9989 | 1.895× | 1.1838 / 1.2350 | 0.959× | 3.0124 / 2.1602 | 1.394× |
| 200000 | 3.6190 / 1.9035 | 1.901× | 2.1073 / 2.1873 | 0.963× | 5.6508 / 4.0172 | 1.407× |
| 219969 | 3.9707 / 2.1115 | 1.880× | 2.3070 / 2.3887 | 0.966× | 6.2018 / 4.4289 | 1.400× |

M=219969 的 wall mean/p50/p95：

| Backend | Forward ms | Backward ms | F+B ms | 验收 |
|---|---|---|---|---|
| legacy | 3.97066 / 3.96927 / 3.98001 | 2.30698 / 2.30528 / 2.32702 | 6.20176 / 6.19746 / 6.22745 | oracle |
| split cuBLAS | 1.48011 / 1.47974 / 1.49008 | 2.72191 / 2.72108 / 2.73695 | 4.14685 / 4.14625 / 4.16964 | hard ID 失败 |
| warp | 5.14176 / 5.14069 / 5.15686 | 2.72279 / 2.71652 / 2.75498 | 7.76606 / 7.76369 / 7.78036 | parity、性能失败 |
| strict cuBLAS | 2.11151 / 2.11007 / 2.12263 | 2.38871 / 2.38801 / 2.39847 | 4.42894 / 4.42888 / 4.44496 | 初始夹具通过；最终真实 checkpoint fixed-noise 失败 |

## Selector caller 性能

当前 room checkpoint N=109564，保存真实 continuous maturity/uniform，六类 Gaussian 属性 gather。计时包含 raw quaternion normalization、scale/opacity activation、网络/elastic mix、ST、动态 compact、opacity VJP、ratio、全部 Selector 输入及 raw Gaussian 相关 backward、Selector Adam。每次 backward 显式检查 XYZ/opacity/log-scale/raw quaternion 的 gradient 存在；没有为速度关闭这些 gradient。

这是有 Gaussian 输入梯度的结构化 caller；仍**不含真实 rasterizer、TF、GI 和 Gaussian optimizer step**，不能等同完整 Mapping。100 warmup + 500 measurement，D/caller_benchmark.csv。以下均为 wall mean/p50/p95。

| 阶段 | legacy ms | CUDA ms | mean 倍率 |
|---|---|---|---:|
| 结构化 caller，含 Gaussian 输入梯度 | 4.311110 / 4.304930 / 4.369350 | 3.022600 / 3.018430 / 3.056260 | 1.426× |
| Adam | 0.674631 / 0.671713 / 0.715967 | 0.034183 / 0.033866 / 0.034593 | 19.736× |
| Guard preview+commit | 1.422600 / 1.417360 / 1.476730 | 0.186398 / 0.184219 / 0.194689 | 7.632× |
| Guard preview+discard | 1.349190 / 1.346170 / 1.407090 | 0.175784 / 0.172383 / 0.199868 | 7.675× |
| ST/mask/compact/gather | 0.145819 / 0.145510 / 0.147894 | 0.147020 / 0.142254 / 0.173309 | 0.992× |
| 原始两类 Gumbel/ST | 0.091588 / 0.088873 / 0.108083 | 0.089639 / 0.089300 / 0.090583 | 1.022× |

Guard 此项不含历史渲染、projection solve、质量检查，不能称完整 Guard speedup。两组 Gumbel/ST 使用同一原实现、相同 logits 和重置的同一 RNG state；小差异属于测量波动，没有采样优化。外围成本低，未引入 custom RNG、CUB scan 或融合 Protection/ST。

全部八个 M 也完成 caller/ST/selection/Adam/preview 网格，每项 100+500。该组使用真实审计 Gaussian 属性和 Selector checkpoint，maturity 固定为 1、entry=0、固定 cached uniform，完整保留 Gaussian 输入 gradient；用于统一规模比较，不能代替真实 maturity 的 room fixture。各阶段 event/wall mean/p50/p95 见 D/caller_grid_benchmark.csv。

| M | legacy caller mean ms | CUDA caller mean ms | 倍率 |
|---:|---:|---:|---:|
| 1000 | 2.61914 | 1.36769 | 1.915× |
| 5000 | 2.59478 | 1.47564 | 1.758× |
| 10000 | 2.64808 | 1.43266 | 1.848× |
| 20000 | 2.67483 | 1.48482 | 1.801× |
| 50000 | 3.42030 | 1.88696 | 1.813× |
| 100000 | 4.29662 | 2.90682 | 1.478× |
| 200000 | 6.82114 | 4.85344 | 1.405× |
| 219969 | 7.21411 | 5.17497 | 1.394× |

## Nsight、dispatch 与显存

真实审计 M=219969，10 次单独捕获的网络 F+B：

| Backend | Forward kernels / active ms | Backward kernels / active ms | 网络总 kernels |
|---|---|---|---:|
| legacy | 18 / 3.92707 | 42 / 2.21538 | 60 |
| split cuBLAS | 22 / 1.45548 | 53 / 2.61725 | 75 |
| warp | 5 / 5.10281 | 53 / 2.61552 | 58 |
| strict cuBLAS | 18 / 2.09010 | 47 / 2.31384 | 65 |

这不是旧审计 351-kernel caller 的同边界对照。strict forward kernel 数未减少，bias/ReLU epilogue 降低 active time；backward 增加 split/VJP launches，大 M 单独 backward 仍略慢。D/round2_nsight_summary.json 保留 kernel names、device span、active/gap、CUDA API、copy 和 launch gap。

最终有 Gaussian 输入梯度的 structural caller（N=109564，10 次捕获）：

| Scope | legacy kernels/iteration | CUDA kernels/iteration |
|---|---:|---:|
| Gaussian activation preparation | 5 | 5 |
| network/probability/elastic mix | 26 | 26 |
| ST/mask/compact/gather | 17 | 17 |
| loss | 7 | 7 |
| caller backward | 77 | 82 |
| Adam | 144 | 1 |
| 总计 | 276 | 138 |

legacy：active 3.87775 ms，device span 5.77451 ms，span 内 gap 1.89677 ms；copy 132 bytes/iteration，`cudaStreamSynchronize` host API 0.002585 ms。

cuda：active 2.61135 ms，device span 3.40064 ms，span 内 gap 0.78929 ms；copy 4 bytes/iteration，`cudaStreamSynchronize` host API 0.002470 ms。

legacy 每次 D2D=128 bytes、D2H=4 bytes；CUDA D2D=0、D2H=4 bytes；H2D=0。两组均有一次动态 count 的 stream synchronize。warmup 后捕获区间无 cudaMalloc/cudaFree，仍有 caching allocator 中间 Tensor 和每次 8 个 memset。不能把 nonzero 等待前序网络的 host 时间全算作 compact GPU 成本。D/caller_nsight_summary.json。

Native Kineto、M=219969：legacy 每次 `aten::addmm=9`、`relu=8`、`threshold_backward=8`、`cat=1`；CUDA 全部为 0，hot path 已接管。strict custom pack 仍占 `[M,128]`，不能称 concat workspace 已消除。D/operator_dispatch_summary.json。

网络 F+B allocator incremental peak（M=219969）：legacy **397195264 bytes**，strict **465846272 bytes**，增加约 **65.5 MiB**。这不是显存优化成功；packed、完整 dFeature 和保存激活是严格路径的代价。原始各阶段 allocated/reserved peak 见 D/round2_network_memory.csv，不能叠加各阶段 peak。

有 Gaussian 输入 gradient 的 room caller allocator 记录：

| Backend | allocated before bytes | allocated peak bytes | incremental peak bytes |
|---|---:|---:|---:|
| legacy | 98135040 | 316852224 | 218717184 |
| cuda | 110849024 | 355194368 | 244345344 |

以上包含 fixture 和既有 handle/workspace，前置分配不同；reserved peak 受同进程缓存复用影响，不是独立进程峰值。网格每阶段另存 `.memory.csv`。

## 固定地图质量

当前 room checkpoint N=109564，三个真实相机帧，ratio 0.01/0.05/0.10/0.15，固定 maturity、uniform、frame/ratio 顺序，1000 步 Selector 训练。每步 probability、ST、global selected IDs、loss、梯度、参数、Adam m/v/step exact；最终 12 组图像 exact，因此每组 PSNR/SSIM/LPIPS 也 exact。D/fixed_map_replay.csv、fixed_map_quality.json。

上述实验冻结 Gaussian/TF，使用同一真实 rasterizer opacity VJP 驱动两个 Selector 图，隔离 rasterizer atomic reduction 噪声。每步 selected IDs 的二进制 trace 保存在服务器 D/fixed_map_selected_ids.bin，SHA-256 为 `ca89e7f6e0becd00614904e128c763d586f3c208c1b0333063109a92cd0126c2`。

另完成 **1000 步联合 Gaussian/Selector/TF deterministic VJP replay**：相同 checkpoint、三帧/四 ratio 顺序、maturity 和 uniform，Gaussian/TF LR 固定 1e-5，Selector LR=0.001，两后端设置相同。Gaussian、Selector 和 TF 实际更新；每步 probability/ST/selected IDs、完整梯度、参数及每 Tensor Adam m/v/step exact。

为隔离未修改模块的 atomic reduction 噪声，该联合夹具共享真实 renderer/TF 产生的外部 VJP；两个 Selector 的参数和 Gaussian 输入 VJP 各自独立计算，再与同一外部贡献相加。没有切断 Selector→XYZ/quaternion/scale 路径。它是受控梯度 replay，不是两份完全独立 rasterizer backward 的确定性证明；固定拓扑、maturity、相机和实验 LR 也不等于完整 SLAM 的动态训练。

联合训练后另有三帧×四 ratio 的 **12 组图像及 PSNR/SSIM/LPIPS exact**，与 Selector-only 的 12 组分开保存。结果见 D/joint_fixed_map_replay.csv、joint_fixed_map_quality.json 和 selected-ID trace；后者 SHA-256 为 `9734f0dfeeae9c6fb798d35b8f8aec4dea9c3c72a952bd4bd48d3e554d082a04`。

## 完整在线 A/B 与 legacy 自身波动

最终网络+Adam 队列：`/home/prmi/zwj/selector-cuda-20261015/results/selector_online_ab_final_20261015`。room/desk/xyz/long，每场景先三次 legacy，再三次 CUDA，均为 seed0 的独立进程重复，不是三个不同 seed。各组使用同一 binary/library/dataset/SLAM/Selector/Protection/TF/ratio/初始化策略/stop，只改 `Selector.backend`。

使用既有 online-only driver，在 SLAM shutdown 时停止，两组都没有 post-SLAM tail。冻结 fast.yaml 的完整副本和 SHA 在在线目录；不能外推所有 Guard/CD/DSD 配置组合。评估 Full 和四个 ratio，保留当前 checkpoint cached-uniform Bernoulli；原始 seed0 draw 另作 sampler parity。图像落盘关闭，渲染/指标计算和模型保存保留。

冻结配置为 Protection=1、Replay=1、ChangeDistill=1、TF backend=cuda、GradientConstraint=0、DSD=0。Guard/DSD 的数学、状态和集成测试另有记录，不能称本轮完整线上启用了它们。

下列 ± 为三个独立运行的 sample std，原始每 run 数值、min/max、CUDA-Legacy delta/std 和首次日志输入分歧在 D/online_summary.json、online_quality_comparison.csv。

### Mapping 周期与累计工作时间

| 场景 | legacy 周期 mean ms | CUDA 周期 mean ms | mean 倍率 | legacy p95 ms | CUDA p95 ms | legacy 累计周期 wall s | CUDA 累计周期 wall s |
|---|---|---|---:|---|---|---|---|
| room | 9.1663 ± 0.0553 | 8.0208 ± 0.2138 | 1.143× | 20.5501 ± 0.4582 | 17.9425 ± 1.5591 | 56.5492 ± 0.2753 | 56.7094 ± 0.2719 |
| desk | 7.0567 ± 0.0364 | 5.9067 ± 0.1520 | 1.195× | 11.3002 ± 0.1921 | 9.7926 ± 0.5807 | 25.0179 ± 0.5296 | 24.6390 ± 0.2138 |
| xyz | 7.0543 ± 0.0880 | 6.0072 ± 0.0832 | 1.174× | 9.2799 ± 0.2903 | 7.7663 ± 0.2516 | 144.8716 ± 0.9978 | 143.9668 ± 0.3972 |
| long | 10.8680 ± 0.1242 | 9.3607 ± 0.0410 | 1.161× | 21.9637 ± 0.7921 | 19.2260 ± 0.6280 | 115.5140 ± 0.5530 | 116.2756 ± 0.6680 |

这里的倍率是完整 Mapping **平均优化周期**，不是整个 SLAM 进程 wall 的倍率。MappingProfiler 使用原有 begin/end 范围的 host wall，不额外同步 GPU；它不是 Nsight GPU active time 或严格 device span。两组人口和调度也不同，周期倍率不能单独归因于 Selector；固定输入下的归因由网络/caller benchmark 提供。固定 SLAM 停止窗口内，更快的周期允许更多更新；累计 Mapping 工作时间不一定下降。累计值由 MappingProfiler 的 mean×cycles 得到。外层队列为独立诊断暂停过协调器，但不中断正在运行的训练子进程；外层 subprocess wall 可能含延迟回收时间，故不用于性能结论。

### 实际更新和人口

| 场景/后端 | cycles | committed GS updates | Selector updates | TF normal updates | 最终 N=eligible | compatibility Mature |
|---|---|---|---|---|---|---|
| room/legacy | 6169.3 ± 26.6 | 6169.3 ± 26.6 | 4936.0 ± 20.8 | 2536.0 ± 20.8 | 119867.3 ± 6618.6 | 65347.7 ± 1345.8 |
| room/cuda | 7073.0 ± 152.4 | 7073.0 ± 152.4 | 5659.0 ± 122.1 | 3259.0 ± 122.1 | 122912.3 ± 18101.5 | 67694.7 ± 4341.4 |
| desk/legacy | 3545.3 ± 76.4 | 3545.3 ± 76.4 | 2836.7 ± 61.3 | 436.7 ± 61.3 | 62397.3 ± 9176.7 | 35175.7 ± 3180.0 |
| desk/cuda | 4173.7 ± 135.2 | 4173.7 ± 135.2 | 3339.3 ± 108.2 | 939.3 ± 108.2 | 54993.7 ± 3635.7 | 32598.0 ± 1335.9 |
| xyz/legacy | 20539.7 ± 368.4 | 20539.7 ± 368.4 | 16432.0 ± 294.6 | 14032.0 ± 294.6 | 32380.3 ± 839.8 | 29916.3 ± 514.8 |
| xyz/cuda | 23968.3 ± 271.1 | 23968.3 ± 271.1 | 19175.3 ± 216.8 | 16775.3 ± 216.8 | 33989.0 ± 953.9 | 30631.3 ± 764.5 |
| long/legacy | 10630.0 ± 160.2 | 10630.0 ± 160.2 | 8504.7 ± 128.0 | 6104.7 ± 128.0 | 122898.3 ± 5681.8 | 76523.0 ± 5400.3 |
| long/cuda | 12422.0 ± 125.8 | 12422.0 ± 125.8 | 9937.7 ± 100.6 | 7537.7 ± 100.6 | 112969.3 ± 2891.5 | 81400.3 ± 2558.8 |

本配置 entry=0，eligible 等于总 N。compatibility Mature 是连续 maturity 的兼容统计，不是恢复旧 Mature 分池规则。每 run 的 maturity mean/p10/p50/p90、四 ratio 的 selected count/实际 ratio、TF transformed count 和 CD/Replay counters 均保存在 JSON。

| 场景/后端 | LoD selected mean | LoD 实际 selected ratio | LoD TF transformed mean | TF 累计处理行数 |
|---|---|---|---|---|
| room/legacy | 5783.2 ± 154.4 | 0.07555 ± 0.00055 | 4231.7 ± 121.2 | 20888272 ± 644126 |
| room/cuda | 5433.0 ± 208.7 | 0.07578 ± 0.00034 | 4393.4 ± 162.1 | 24849627 ± 446953 |
| desk/legacy | 3476.0 ± 159.7 | 0.07489 ± 0.00069 | 765.7 ± 59.9 | 2174383 ± 212401 |
| desk/cuda | 3107.8 ± 239.3 | 0.07587 ± 0.00129 | 1274.6 ± 30.3 | 4258247 ± 236940 |
| xyz/legacy | 1775.4 ± 88.0 | 0.07519 ± 0.00203 | 1724.7 ± 85.4 | 28323266 ± 932702 |
| xyz/cuda | 1709.1 ± 50.9 | 0.07444 ± 0.00158 | 1664.5 ± 50.0 | 31911050 ± 639080 |
| long/legacy | 8156.7 ± 29.3 | 0.08535 ± 0.00045 | 7435.9 ± 34.7 | 63243119 ± 1238107 |
| long/cuda | 7358.4 ± 102.2 | 0.08287 ± 0.00075 | 6825.9 ± 100.6 | 67839578 ± 1639958 |

TF 累计处理行数是 normal LoD 调用的工作量，包含同一 Gaussian 多次处理，不是 unique Gaussian count。

本轮 24 次日志的 `replay_tf_updates` 均为 0，不能宣称线上覆盖了 Replay accepted-update；该路径由独立集成夹具覆盖。`cd_tf_steps` 观测范围为 0–60，各 run 原始计数保存在 JSON。

### Trajectory 与图像质量

trajectory 沿用原 evaluator 的 translation-part APE 与 rotation-part APE（后者不是角度单位）。

| 场景 | translation RMSE legacy / CUDA | rotation-part RMSE legacy / CUDA | CUDA translation 均值在 legacy min/max 内 |
|---|---|---|---|
| room | 0.058782 ± 0.017753 / 0.057891 ± 0.019653 | 0.066597 ± 0.010765 / 0.059440 ± 0.000552 | 是 |
| desk | 0.015681 ± 0.000250 / 0.015066 ± 0.000071 | 0.038921 ± 0.000650 / 0.037639 ± 0.002840 | 否 |
| xyz | 0.003161 ± 0.000310 / 0.002834 ± 0.000076 | 0.034128 ± 0.000349 / 0.034702 ± 0.000274 | 是 |
| long | 0.010104 ± 0.001290 / 0.011509 ± 0.001456 | 0.014505 ± 0.000427 / 0.015031 ± 0.000415 | 否 |

| 场景/ratio | PSNR legacy / CUDA | SSIM legacy / CUDA | LPIPS legacy / CUDA | 三项 CUDA 均值均在 legacy min/max 内 |
|---|---|---|---|---|
| room/full | 16.0478 ± 0.2654 / 16.0914 ± 0.5645 | 0.5942 ± 0.0035 / 0.5930 ± 0.0115 | 0.5786 ± 0.0077 / 0.5762 ± 0.0178 | 是 |
| room/ratio_0p01 | 14.4308 ± 0.2187 / 14.5693 ± 0.2609 | 0.5553 ± 0.0033 / 0.5589 ± 0.0039 | 0.7046 ± 0.0100 / 0.7022 ± 0.0034 | 是 |
| room/ratio_0p05 | 15.6971 ± 0.2185 / 15.7810 ± 0.4841 | 0.5851 ± 0.0023 / 0.5865 ± 0.0089 | 0.6379 ± 0.0078 / 0.6372 ± 0.0102 | 是 |
| room/ratio_0p1 | 15.9707 ± 0.2311 / 16.0181 ± 0.5476 | 0.5921 ± 0.0027 / 0.5923 ± 0.0110 | 0.6096 ± 0.0068 / 0.6088 ± 0.0154 | 是 |
| room/ratio_0p15 | 15.9978 ± 0.2378 / 16.0608 ± 0.5538 | 0.5927 ± 0.0027 / 0.5930 ± 0.0114 | 0.6008 ± 0.0060 / 0.5995 ± 0.0157 | 是 |
| desk/full | 18.1904 ± 0.0313 / 18.3335 ± 0.1884 | 0.6529 ± 0.0051 / 0.6594 ± 0.0052 | 0.4575 ± 0.0033 / 0.4451 ± 0.0046 | 否 |
| desk/ratio_0p01 | 16.0647 ± 0.3614 / 15.8449 ± 0.3454 | 0.5949 ± 0.0060 / 0.5899 ± 0.0096 | 0.6133 ± 0.0126 / 0.6198 ± 0.0074 | 是 |
| desk/ratio_0p05 | 17.4854 ± 0.1267 / 17.3735 ± 0.3505 | 0.6298 ± 0.0007 / 0.6287 ± 0.0084 | 0.5348 ± 0.0069 / 0.5346 ± 0.0063 | 否 |
| desk/ratio_0p1 | 17.9814 ± 0.0705 / 18.0358 ± 0.2554 | 0.6446 ± 0.0030 / 0.6479 ± 0.0068 | 0.4963 ± 0.0028 / 0.4906 ± 0.0052 | 否 |
| desk/ratio_0p15 | 18.0916 ± 0.0203 / 18.2265 ± 0.1782 | 0.6482 ± 0.0039 / 0.6535 ± 0.0053 | 0.4817 ± 0.0016 / 0.4711 ± 0.0026 | 否 |
| xyz/full | 21.4264 ± 0.2310 / 20.6313 ± 1.0558 | 0.7141 ± 0.0053 / 0.7069 ± 0.0074 | 0.2993 ± 0.0098 / 0.3056 ± 0.0144 | 否 |
| xyz/ratio_0p01 | 17.6808 ± 0.0686 / 17.3798 ± 0.1138 | 0.6012 ± 0.0012 / 0.5974 ± 0.0008 | 0.6390 ± 0.0187 / 0.6392 ± 0.0289 | 否 |
| xyz/ratio_0p05 | 19.2983 ± 0.2113 / 18.4658 ± 0.7678 | 0.6416 ± 0.0048 / 0.6327 ± 0.0072 | 0.4956 ± 0.0118 / 0.5097 ± 0.0357 | 否 |
| xyz/ratio_0p1 | 20.2854 ± 0.2369 / 19.4612 ± 1.0796 | 0.6700 ± 0.0051 / 0.6609 ± 0.0075 | 0.4204 ± 0.0105 / 0.4271 ± 0.0210 | 否 |
| xyz/ratio_0p15 | 20.5318 ± 0.1733 / 19.6607 ± 1.1733 | 0.6781 ± 0.0044 / 0.6684 ± 0.0096 | 0.3981 ± 0.0078 / 0.4069 ± 0.0253 | 否 |
| long/full | 20.8195 ± 0.9133 / 20.0634 ± 0.5190 | 0.7337 ± 0.0295 / 0.7155 ± 0.0109 | 0.2871 ± 0.0199 / 0.2986 ± 0.0074 | 是 |
| long/ratio_0p01 | 17.8674 ± 0.1730 / 17.6042 ± 0.2153 | 0.6486 ± 0.0052 / 0.6444 ± 0.0021 | 0.5762 ± 0.0051 / 0.5882 ± 0.0061 | 否 |
| long/ratio_0p05 | 19.8919 ± 0.5781 / 19.3773 ± 0.4175 | 0.6994 ± 0.0162 / 0.6888 ± 0.0049 | 0.4029 ± 0.0066 / 0.4175 ± 0.0031 | 否 |
| long/ratio_0p1 | 20.5262 ± 0.8373 / 19.8379 ± 0.5335 | 0.7199 ± 0.0247 / 0.7047 ± 0.0088 | 0.3359 ± 0.0151 / 0.3520 ± 0.0026 | 否 |
| long/ratio_0p15 | 20.6461 ± 0.8879 / 19.8920 ± 0.5410 | 0.7248 ± 0.0267 / 0.7076 ± 0.0095 | 0.3194 ± 0.0173 / 0.3357 ± 0.0042 | 是 |

共 60 项图像质量比较，其中 26 项 CUDA 均值超出三次 legacy 的观测 min/max；仅 5 项满足所有 CUDA 单次结果均在 legacy min/max 内。这是原始观察范围的报告，不是人为质量容差或显著性检验。三个样本不足以证明统计等价，均值落入范围也不自动放行。

日志可见在线输入在 Selector 更新前就不同：room 的 legacy 自重复在首个 training step 的 N 已分别为 1989 和 1940；CUDA 对照首 step 的 keyframe/N 也不同。因此不能把在线权重/ID 轨迹差异直接当成同输入的 CUDA 数值漂移，也不能据此声称 CUDA 质量提升。固定 checkpoint、固定输入/随机数的数学和 replay 对照用于单独隔离算法一致性。

尚未记录 live Mapping 内每一步 global selected-ID trace、独立 Selector forward/backward 的逐步 p50/p95；不能用 microbenchmark 冒充这些在线观测。最终 checkpoint 的同输入 inference/mask parity 另有独立记录。

### 实际在线 checkpoint 的同输入验收

24 个实际 checkpoint × 四 ratio，共 96 组，分别使用相同已训练参数、Gaussian 属性和 metadata。比较的是同一 checkpoint 的 legacy/CUDA，不是把不同在线运行的参数相互比较。

当前 cached-uniform mask、compact IDs、独立 seed0 Bernoulli、generator state/next draw、全局 CUDA RNG 不变检查均为 **96/96 exact**。概率均满足初始 float 容差，最大 abs=1.78813934e-07。

但是包含双侧 near-tie 后仅 **94/96 通过**。失败来自 `desk/legacy/repeat2`、N=71165 的 ratio=0.10/0.15；两个 case 存在 FP32 概率差异，不能用实际 cached uniform 未碰到阈值来放行。原始 CSV、逐层定位和摘要均保留。**strict 是候选名，不代表其已全面满足 hard-mask 门槛。**

首次排队的 checkpoint 测试未设置 Ninja 所在 PATH，在 extension loader 阶段失败；补齐 PATH 后完成上述 96 组。失败环境日志单独保留，没有把环境错误当成数学通过。

### 最终 fixed-noise 失败的逐层定位

`desk/legacy/repeat2/3465_shutdown/ply`，N=71165，FP32/TF32=false：三个属性分支无差异；最早差异出现在 `[M,1]` ratio 第一层（stage_6），而不是 final GEMM 或 softmax。

| Ratio / 边界 | max abs | RMS | p99 | 不同元素或 hard class 数 | worst global ID |
|---|---:|---:|---:|---:|---:|
| 0.1 ratio 第一层 | 2.98023224e-08 | 5.26835597e-09 | 2.98023224e-08 | 71165 | 0 |
| 0.1 probability | 1.19209290e-07 | 4.37587522e-09 | 0.00000000e+00 | 422 | 1282 |
| 0.1 固定 Gumbel noise hard class | 0.00000000e+00 | 0.00000000e+00 | 0.00000000e+00 | 0 | -1 |
| 0.15 ratio 第一层 | 5.96046448e-08 | 1.49011612e-08 | 5.96046448e-08 | 142330 | 0 |
| 0.15 probability | 1.78813934e-07 | 1.97710275e-08 | 8.94069672e-08 | 29732 | 67069 |
| 0.15 固定 Gumbel noise hard class | 1.00000000e+00 | 9.07922804e-01 | 1.00000000e+00 | 58663 | 1 |

固定 `noise=-legacy_logits` 令 legacy 两类恰好打平，按首类优先。ratio=0.15 的冻结候选实际改变 **58663 个 hard class**；首个不同 global Gaussian ID 为 1。scalar ratio 的四个 ratio 在同一 checkpoint 全部 exact，不能外推 row ratio。

诊断直接复现当前 `CUBLAS_OP_T, lda=1` 的 ratio 第一层差异；对同一组参数/输入，改用等价 `CUBLAS_OP_N, lda=32` 的单层参考调用后误差为 0。用完全相同 packed feature 调用 legacy/custom final 时误差也为 0。这把该 case 的原因定位到 one-column ratio GEMM 的转置/布局及对应 FP32 累加路径；没有 TF32 或 RNG 改动。

非转置参考仅用于单层定位，未替换冻结的实验实现，不能称已修复或继承本报告全部验收/性能结果。该候选按用户 hard-ID 硬门槛判定失败，保留为明确标注的实验，不进入 production。原始逐层证据：D/online_stage_probe.cpp、online_stage_probe.py、online_stage_probe.csv、online_stage_probe.json。

## 生产切换与尚未通过项

**目前不允许默认切换。** 24 次最终在线训练和评估已全部完成，但真实已训练 checkpoint 的 near-tie 暴露了失败，26/60 项在线图像质量均值超出 legacy 观测范围，desk/long 的 translation RMSE 均值也超出范围。独立完整图的 Gaussian/TF 梯度初始容差未通过，legacy 自身也有明确误差下限；其联合长训练不能由受控 VJP replay 代替。大 M backward 没有加速，显存增加。

| 生产门槛 | 状态与边界 |
|---|---|
| 1. Fixed-noise hard mask exact | **失败**：真实 desk checkpoint 的 ratio=0.15 固定 Gumbel noise 改变 58663 个 hard class；另有两个双侧 near-tie case 失败；split/warp 也失败 |
| 2. Compact IDs exact | strict 的固定输入、固定图 replay 通过；live online 未记录逐步 IDs |
| 3. Forward parity | 初始夹具 exact；实际在线 checkpoint 两组有非零概率误差，均在 float 容差内，仍会破坏 near-tie |
| 4. Backward parity | 同一 upstream 的 Selector VJP 通过；独立完整 renderer 图仍未通过初始容差 |
| 5. Optimizer 多步 | 1000 步参数、18 tensor gradient、Adam 状态 exact；undefined/zero 分开测试 |
| 6. Guard/Replay/CD/DSD | 已列的事务及集成测试通过；不外推所有在线配置组合 |
| 7. 固定地图无差异 | 两类受控 VJP replay 各 1000 步和 12 组图像/质量 exact；完全独立 renderer 长训练尚不能替代 |
| 8. 在线质量/trajectory | 未通过：26/60 项质量均值在 legacy 观测范围外，desk/long translation 均值也在范围外 |
| 9. 真实 checkpoint Selector wall | 网络 F+B 1.400×，structural caller 1.394×；不包含真实渲染/TF/GI |
| 10. 完整 Mapping wall | 在线平均周期有可测收益；固定停止窗口中工作量不同，相同工作量的完整 Mapping 总 wall 收益未证明 |

因此本次结论是**可复现实验实现与局部性能收益成立，生产替换验收不通过**。不通过项保留原始数据，不以改采样、关闭梯度或放宽容差消除差异。

## 最终九问

1. **CUDA Selector 真正更快吗？** 局部性能是：M=219969 网络 F+B 为 1.400×；真实 room structural caller 为 1.426×，均是本轮同 binary A/B。但本轮没有通过全部 correctness/quality 门槛的 CUDA 替代实现。
2. **warp 还是 cuBLAS？** cuBLAS 更快；raw split 最快但 mask 失败，strict cuBLAS 进入实验 caller 后也在实际 checkpoint fixed-noise 中失败。最终 production 选择保留 legacy。
3. **Forward/backward 各多少？** forward 1.880×；backward 0.966×，慢约 3.5%。
4. **Selector caller 整体？** 真实 room continuous-maturity fixture 为 1.426×；M=219969 的全成熟统一规模 fixture 为 1.394×；均不包含 renderer/TF/GI。
5. **完整 Mapping？** 平均周期观测 room/desk/xyz/long 为 1.143×/1.195×/1.174×/1.161×；工作量不同，不能声称相同工作量的总 wall 同倍下降。
6. **selected IDs 变化过吗？** 发生过：三个候选都在 fixed-noise/near-tie 测试中失败；strict 在真实已训练 row-ratio case 中改变 58663 个 hard class。其初始受控 1000 步和 96 组实际 cached-uniform masks/IDs 无变化；live online 未保存逐步 ID trace。
7. **参数/梯度/Adam 系统漂移？** 受控网络/1000 步 replay/事务中未观察到。独立完整图的 Selector 梯度通过初始容差；Gaussian/TF 梯度及 legacy 自重复均超限，完整图验收仍未通过，不能宣称全部梯度已一致。
8. **最终质量/trajectory 在 legacy 波动内吗？** 两类受控固定图实验各 12 组 exact；在线 26/60 项质量均值和 desk/long translation 均值超出 legacy 观测范围，不能判等价；三个重复也不足以证明统计等价。
9. **能安全替换 legacy 吗？** 不能。本轮三个候选均没有取得全部硬验收资格。保留 legacy 默认和实验 gate。
