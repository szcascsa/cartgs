# Anchor SI L2：实现与实际实验结果

日期：2026-10-24。分支：`anchor-si`。性能优化基线提交：`27b9439`。

## 1. 结论

已完成 Anchor 级 SI、CUDA 统计/VJP、Selector 私有分支 L2 监督，并完成两种优化顺序 seed 的冻结地图实验和四次真实 `fr1_xyz` SLAM。**当前版本不建议默认启用。**

- 冻结地图、相同 K：1% 的选中集合完全不变；5% 只有 +0.005/+0.024 dB，10%/15%小幅下降。没有稳定的整体质量收益。
- 真实 SLAM、相同 K：1% 提高 +2.020/+2.171 dB，但 5%～15% 下降 0.535～0.961 dB。
- 实际 Cached Bernoulli：1% 为 +0.156/−0.783 dB，其他三个档位均下降。不能把 same-K 的单档改善当作生产路径整体改善。
- SI 增加了可观训练成本；原生微基准中 L2 加定向 head VJP 的 CUDA Event 中位时间约 0.71～0.72 ms。未达到“质量稳定提高且开销足够低”的启用条件。
- 在线 OFF 自然停止于 6400 Mapping，Anchor SI 停止于 5400；地图、关键帧和 online/tail 分配也不同。在线结果是系统行为对照，不能作为严格隔离的 SI 排名因果估计。

保留七项既有 CUDA 优化默认开启。`AnchorSI.enabled: 0` 为默认；显式实验配置为 [`cfg/selector/anchor_si_l2.yaml`](cfg/selector/anchor_si_l2.yaml)。没有部署 SI 到服务器生产目录，没有 push。

## 2. 版本、边界与源码

先提交当前性能优化版 `27b9439`，再创建 `anchor-si`。本轮 SI 修改保留在该分支工作区，未追加提交。Git index 为空。

| 文件 | 实际改动 |
|---|---|
| `cuda_rasterizer/anchor_si_capture.h`、`rasterize_points.cu` | Host 线程局部 RAII 采集开关；不改变公开 Torch/Python wrapper 签名 |
| `cuda_rasterizer/forward.cu`、`forward.h`、`rasterizer.h`、`rasterizer_impl.cu` | 内部 importance 模式 0=关闭、1=旧统计、2=Anchor SI；旧模式保留 |
| `include/anchor_si.h`、`src/anchor_si.cpp` | EMA、访问窗口、UID 迁移、监督损失、仅 head 定向梯度及统计保存 |
| `src/anchor_si.cu` | Anchor 均值/eligible 数/身份位图归约、VJP、按预算校准 soft target |
| `include/gaussian_mapper.h`、`src/gaussian_mapper.cpp` | 正常 Full 采集、LoD SI 梯度接入、几何事件失效、配置与导出 |
| `tests/anchor_selector/anchor_si_test.cpp` | CUDA 归约、梯度、身份、窗口和真实合成语义单测 |
| `tests/diagnostics/anchor_si_20261024/` | 预注册协议、构建、冻结训练、真实在线、评价、微基准与原始证据 |

原有 Variable Anchor 1～10 成员、L0/L1/L2、Feature16、Input27、共享 Encoder、Selector、R3 四表、Dense Adam、Cached Bernoulli ST、Budget、clone/split/prune 和原 Full/LoD 顺序保持。没有新增 MLP、可学习参数或优化器组。

服务器实验隔离于 `/home/prmi/zwj/anchor-si-20261024`。最终源码包 325 个文件与隔离源码逐文件 SHA256 一致，详见 `evidence/source_verification.json`。生产核心源码按 LF 规范化后与基线一致；生产配置有两个已有差异：`anchor_only_pruning=0`、`lod_render_scale=1`。未覆盖这些服务器设置。本实验明确使用本地基线 `anchor_only_pruning=1`、online LoD scale=0.5，不能冒称与服务器当前配置完全相同。

## 3. 实际统计和 L2 定义

对于正常 Full Render 中真正通过 Renderer 截断、参与颜色合成的 Gaussian：

\[
c_i=\sum_{p\in\text{accepted pixels}}\alpha_{ip}T_{ip}.
\]

这里使用经过空间核衰减与现有截断后的 `alpha`；旧统计的 `opacity*T` 模式仍保留。新模式只写临时 `[N]` FP32 contribution，不分配旧的逐点 pixel count。

Anchor 当前 eligible 集合为 \(E_A\)，沿用现有 inactive 与 opacity 规则。采集值：

\[
s_A=\frac{1}{HW\,|E_A|}\sum_{i\in E_A}c_i.
\]

采用成员均值避免“大 Anchor 仅因成员多而得到更高 SI”。只在 \(s_A>0\) 且有限时更新 EMA 和访问次数；未观察不记零。每 4 次正常 Full 采集一次，不增加 Bank 重渲染；EMA decay=0.9，至少 4 次访问，最大年龄 128 Full。

ratio target：

\[
y_A(r)=\sigma\!\left(\frac{\log(\max(\bar s_A,10^{-12}))-\tau_r}{T}\right),\quad T=1.
\]

令 \(M\) 为所有 Anchor 当前 eligible 成员数总和，\(V\) 为满足访问和时效条件的 Anchor 集合，令 \(\rho_r=rN/M\)。CUDA 中对每个 ratio 二分求阈值，使：

\[
\sum_{A\in V}|E_A|y_A(r)\simeq\rho_r\sum_{A\in V}|E_A|.
\]

这样保留原 Budget 的 N 人口分母；未观察 Anchor 不当作负标签。若没有有效 teacher 或预算超出 eligible 容量，忽略该批 teacher。此校准是本轮新增 SI teacher 的定义，不修改原 Budget Loss。

\[
\bar p_A(r)=\frac1{|E_A|}\sum_{i\in E_A}p_i(r),\qquad
L_{SI}=0.01\,\frac{\sum_{A\in V}(\bar p_A(r)-y_A(r))^2}{\max(|V|,1)}.
\]

L2 作用在概率均值，**不是 hard/ST mask 的 L2，也不是概率 L1**。原 photo loss 仍为 `0.8 L1 + 0.2 (1-SSIM)`，按任务约束保留。

SI 梯度只累加到现有 `local_emd.2.*` 与 `soft_net.*` 四个参数 Tensor；通过定向 `autograd::grad` 保留 photo/Budget 的原有图。Encoder、Anchor Feature、Master、R3 不直接接收 SI 梯度；它们原有 photo/Budget 路由不变。推理不读取 SI 状态。

## 4. 生命周期与修复记录

- SI 以 Anchor stable UID 对齐；成员签名使用 Gaussian UID，不能把 row 或空间 key 当稳定身份。
- 新建、成员增删/迁入迁出造成签名变化的 Anchor 冷启动；纯 Gaussian row reorder 保留统计。
- eligible 成员身份位图按 **Gaussian UID 升序**编码，padding 位于末尾；活跃成员变化即使成员数量相同，也会重置访问窗口。
- 几何修正使历史贡献失效，重新观察。保存前同步当前 topology，导出独立 `anchor_si_state.pt`。
- 当前完整在线 optimizer resume 本来就未由 PLY 完整实现；新增统计保存不意味着已经支持完整训练恢复。

开发中捕获并修复三项接口/身份问题，没有放宽数值门槛：

1. 可选 inactive undefined Tensor 不可直接作为 C++ autograd Function 的输入，改为已定义的空 CUDA bool Tensor。
2. 定向梯度可能返回 view，直接赋 `.grad` 会使下一次 `zero_grad` 失败；首次持有梯度时 clone，增加连续清梯度测试。
3. 初版 active 位图按 row 次序构建，纯 reorder 可能错误重置窗口；改为 UID canonical 顺序。保留初版模型/指标于服务器 `v1_row_order/`，两组 Anchor 冻结实验和两组 Anchor 在线实验全部重跑。OFF/Gaussian 控制路径不受该修复影响，保留其已完成对照。

v1 与 v2 的在线 Bank 并集不同，最终 non-Bank 集也不同，不能直接把两版均值相减。最终结果全部来自 v2 评价；没有按质量挑选版本或 checkpoint。

## 5. 正确性与回归

| 验证 | 结果 |
|---|---|
| eligible 成员计数、空输入、stride=2、inactive 截断边界 | PASS；计数 exact |
| Anchor 均值与 ATen 参考 | max abs `1.19209e-7` |
| 均值 VJP | max abs `0` |
| 成员成本加权 target 预算、过期/未观察处理 | PASS |
| SI 仅 head 梯度、重复 zero_grad、Master/Feature/R3 隔离 | PASS |
| stable UID、成员变化失效、纯 row reorder、几何失效和统计保存 | PASS |
| 白色 Gaussian/黑背景：sum(alpha*T)=单通道像素和 | exact |
| 真实 15,891 点 Full capture 开关 RGB、radii | exact |
| 真实 Renderer 六组 VJP | 所有 bad 元素为 0；见下 |
| R3 生命周期、四档隔离、append/reorder/Adam、checkpoint | PASS |
| Variable Anchor 实际 clone/split/prune/grow、Feature 与 Adam 迁移 | PASS |
| 正确 split 尺度/旋转采样、闭环容量与加权 Feature 迁移 | PASS |
| 本地 launcher 回归 | 21/21 PASS |

真实 Renderer VJP 预设门槛：`abs(diff) <= 2e-5 + 2e-4*abs(reference)`。XYZ/DC/SH-rest/Opacity/Scale/Rotation 最大绝对误差分别为 `1.584e-7 / 5.821e-10 / 8.149e-10 / 1.490e-8 / 5.678e-7 / 9.802e-8`。小梯度相对误差不单独作为放宽门槛的理由。采集 atomic 改变执行次序，浮点原子归约仍存在原生非确定性。

最终在线两份 SI archive 的 Anchor UID、canonical Gaussian UID 签名、owner/member_counts 全部匹配保存模型，Gaussian UID 唯一。所有在线分支 `lod_nonfinite=0`。没有单独触发一场可完全控制的真实 Loop Closure，因此不将合成迁移回归冒称实测闭环质量证明。

原始结果：`evidence/validation/`、`evidence/micro/real_renderer_parity.csv`、`evidence/regression/results.json`。

## 6. 冻结地图协议

- 同一 `fr1_xyz/7465_shutdown/ply`：15,891 Gaussian，6,121 Anchor，72 Bank 视图。
- Master、Encoder、Anchor Feature、共享补偿、R3、身份与 cached uniform 完全冻结；只训练 Selector 现有私有 head。
- 三臂：OFF、Gaussian 级概率 L2 控制、Anchor 级概率 L2。控制臂也使用相同采集/EMA/阈值协议，**并非复刻历史旧 L1 SI**。
- 两个优化顺序 seed：2026/2027；相同头初始化，每臂 1600 次 LoD 更新，每 ratio 400 次；相同 head Adam LR=0.001、eps=1e-15。
- 每步都执行同一无梯度 Full，保证控制臂可比。数据和 frame 顺序固定，不按 non-Bank 指标调参、早停或挑 checkpoint。
- Cached 选择会随 head 训练变化，故另外评价 exact same-K；不能称训练期间选择集合固定。
- same-K 沿用 cached-margin 排序：`keep_logit-drop_logit-logit(cached_uniform)`，row ID 稳定打破并列；不是直接对 p Top-K。
- 训练四档，另评 2.5/7.5/12.5%。K 来自源 checkpoint；四训练档为 156/799/1582/2368。
- 评价：Bank72、预声明64 RGB 中排除 Bank 后 non-Bank59。该轨迹已被多轮研究使用，不是全新独立场景。

### 6.1 Anchor SI 对等步数 OFF 的 same-K PSNR 差（dB）

| ratio | K | Bank seed2026 / 2027 | non-Bank seed2026 / 2027 |
|---|---:|---:|---:|
| 1% | 156 | 0 / 0 | 0 / 0 |
| 2.5% 未训练 | 435 | +0.0423 / +0.0501 | +0.0086 / +0.0196 |
| 5% | 799 | +0.0031 / +0.0284 | +0.0047 / +0.0240 |
| 7.5% 未训练 | 1229 | −0.0136 / −0.0152 | −0.0227 / −0.0281 |
| 10% | 1582 | −0.0022 / −0.0023 | −0.0018 / −0.0017 |
| 12.5% 未训练 | 2051 | −0.0013 / −0.0049 | −0.0039 / −0.0074 |
| 15% | 2368 | −0.0064 / −0.0142 | −0.0091 / −0.0210 |

1% same-K 集合两个 seed 都零替换。5% 仅替换 3/5 个，10% 替换 10/13 个，15% 替换 15/14 个。质量变化没有体现持续的整体改进。Full 参数不变，评价 PSNR 两 seed 均 exact 相同（non-Bank 21.302615 dB）。

Gaussian L2 控制的 non-Bank same-K 5% 为 +0.0311/+0.0215 dB，1% 同样零变化，10% 约 +0.00003/+0.00084，15% +0.0110/−0.0031。Anchor 版本未稳定优于这个控制；这不构成“任何 Anchor SI 都无效”的证明。

完整 PSNR/SSIM/LPIPS/photo、visible 与 effective proxy、逐视图 mean/median/p10/p90、改善比例见 `evidence/frozen_quality_summary.csv`、`frozen_quality_per_view.csv`、`paired_quality.csv`。`effective_proxy` 是 `radii>0 && opacity>=1/255`，不能当作逐像素真实合成非零计数。

## 7. 真实 SLAM：最终结果与混杂因素

四次从头 `fr1_xyz` RGB-D SLAM，seed2026/2027 各 OFF/Anchor。沿用自然停止，不强制 10000、不改调度；online LoD 半分辨率，tail 原分辨率。选取固定 shutdown 终点。

预声明64 RGB，排除最终四模型 Bank 并集，得到共同 non-Bank51。各模型使用自己的最终轨迹；same-K 为共同最小地图规模预算 142/712/1424/2136。地图或位姿不同的问题没有因 K 相等而消失。

### 7.1 same-K non-Bank PSNR

| ratio | OFF 2026 | SI 2026 | 差值 | OFF 2027 | SI 2027 | 差值 |
|---|---:|---:|---:|---:|---:|---:|
| 1% | 14.6095 | 16.6292 | +2.0198 | 14.4415 | 16.6127 | +2.1712 |
| 5% | 19.5077 | 18.8990 | −0.6087 | 19.2674 | 18.4121 | −0.8553 |
| 10% | 20.0196 | 19.1470 | −0.8726 | 19.5141 | 18.9793 | −0.5349 |
| 15% | 20.2557 | 19.5562 | −0.6995 | 19.8242 | 18.8636 | −0.9606 |
| Full，各自完整地图 | 21.1785 | 21.1319 | −0.0466 | 21.2158 | 20.9598 | −0.2560 |

same-K 的 1% SSIM 差值 +0.00728/+0.00711，但 LPIPS 差值 −0.01046/+0.00305，后者不一致。5%～15% SSIM 均下降；LPIPS 除 seed2027 的5%外均变差。

### 7.2 实际 Cached Bernoulli

| ratio | OFF K → SI K，2026 | PSNR 差值2026 | OFF K → SI K，2027 | PSNR 差值2027 |
|---|---:|---:|---:|---:|
| 1% | 164 → 151 | +0.1560 | 160 → 140 | −0.7832 |
| 5% | 713 → 742 | −0.5761 | 784 → 685 | −0.9014 |
| 10% | 1406 → 1502 | −0.8749 | 1574 → 1366 | −0.6141 |
| 15% | 2122 → 2227 | −0.6951 | 2374 → 1995 | −0.9365 |

Cached 四档、两 seed 的 SSIM、LPIPS、photo 均整体朝差的方向变化：包括1% seed2026虽 PSNR小升，SSIM仍下降、LPIPS/photo仍变差。实际部署路径没有稳定的正收益证据。

### 7.3 地图与训练曝光

| 分支 | N / Anchor | L0/L1/L2 | online / tail Mapping | Full 总次数 | LoD/head 总次数 | Gaussian 总更新 |
|---|---:|---:|---:|---:|---:|---:|
| OFF2026 | 14426 / 5466 | 1456/3787/223 | 5749 / 651 | 1800 | 5251 | 7051 |
| SI2026 | 14900 / 5942 | 1622/4148/172 | 4832 / 568 | 1534 | 4434 | 5968 |
| OFF2027 | 16075 / 6111 | 1693/4188/230 | 5794 / 606 | 1764 | 5242 | 7006 |
| SI2027 | 14244 / 5663 | 1494/3968/201 | 4853 / 547 | 1517 | 4430 | 5947 |

OFF 为6400 Mapping，SI为5400。停止规则未改；源代码 `GaussianMapper::run` 的自然 tail 由 SLAM shutdown 时刻、densify interval及当前iteration共同决定，不能把总步数不同隐藏成“相同训练预算”。Topology round 最终 OFF=11、SI=9。R3、地图和分辨率训练曝光随此一起变化。没有用 zero-gradient step 伪造 optimizer 时钟。

四次 SE(3) 对齐、无 scale correction 的 ATE RMSE 约 0.983/0.981/0.977/0.971 cm，分别对应 OFF2026/SI2026/OFF2027/SI2027。微小差异不能抵消训练曝光和地图差异，也不能证明 SI 改善定位。

完整逐视图数据和时钟：`online_quality_per_view.csv`、`online_quality_summary.csv`、`online_phase_summary.csv`、`online_model_summary.csv`、`trajectory_summary.csv`，均位于证据目录。

## 8. 梯度与状态诊断

冻结 OFF2026 的1%概率并非常数，也不是 ReLU 全死：

- p 从0到1，std=0.09843；96.684% 的概率低于 `1e-5` 或高于 `1-1e-5`。
- mean `p*(1-p)` 仅0.0002027，logit 差绝对值 p90=34.00。
- 私有 head hidden 全零行比例为0，active分量比例约48.9%。
- Anchor SI/head(photo+Budget) 梯度范数比中位数仅0.00412/0.00409；Gaussian控制为0.00732/0.00812。

这些证据支持“概率级 L2 经高度饱和 softmax 后，有效驱动力弱”的候选解释，并与1%集合未变一致；**未完成概率 L2 与 logit L2 的单变量对照，不能声称已经证明唯一原因**。日志每16步采样，因 ratio轮转，梯度统计主要对应1%；其他ratio没有完整梯度轨迹，不外推。

最终在线 SI2026/SI2027 约49.68%/46.79%的 Anchor 满足访问次数、时效和eligible条件；约10.06%/11.03%从未得到有效访问。剩余不满足teacher条件的Anchor被忽略，没有当作低贡献负样本。详见 `si_state_summary.csv`。

另外，Full `alpha*T` 只是合成权重，未测量删除该Anchor后的 photo 边际变化；它没有包含 LoD/R3 对稀疏覆盖的需求。一个Anchor只有一个均值监督，也无法区分组内好坏成员。这些是当前teacher的明确能力边界，不是已经隔离完成的失败原因。

## 9. 轻量性实测

RTX3090 共享GPU，有外部任务，使用成对交替 Full 测试；绝不宣称独占性能。CUDA Event 区间包含当前stream内的等待和CPU提交间隙，并非硬件纯kernel时间。

| 实际 N=15891 / A=6121，640×480 | CUDA Event p50 ms | Host提交 p50 ms |
|---|---:|---:|
| Full Forward，关闭采集 | 0.5494 | 0.4777 |
| Full Forward，开启alpha*T采集 | 1.2580 | 0.4866 |
| observe + 四ratio target | 0.6968 | 0.2207 |
| L2 + 仅head VJP，1% | 0.7076 | 0.6931 |
| 同上，5% | 0.7214 | 0.6954 |
| 同上，10% | 0.7183 | 0.7029 |
| 同上，15% | 0.7112 | 0.7051 |

前两项每组160次成对测试，后五项每组100次，均排除warmup。归约/VJP分别是一项新CUDA kernel；target每ratio一个CTA；但 EMA、窗口、mask、MSE和定向autograd仍有多项ATen/CPU开销。没有以“只增加几个新kernel”掩盖整个接入的算子成本。

冻结每步 Full+LoD 的CUDA Event p50：OFF 3.652/3.619 ms，Anchor SI 4.798/4.880 ms；两个seed均变慢。端到端冻结运行wall time OFF 9.64/9.50秒，SI 11.79/12.01秒，约+22.3%/+26.5%，包含共享GPU与进程开销，不能当独占纯SI成本。

在线/尾部 Host Mapping 延迟如下。不同地图/训练分配下仅作运行观测，不是固定工作量加速比较：

| 分支 | online p50 / p95 / p99 ms | tail p50 / p95 / p99 ms |
|---|---:|---:|
| OFF2026 | 3.704 / 14.012 / 26.653 | 7.596 / 18.653 / 18.953 |
| SI2026 | 4.607 / 14.566 / 33.842 | 8.951 / 19.928 / 20.955 |
| OFF2027 | 3.651 / 13.856 / 25.663 | 7.578 / 18.752 / 19.078 |
| SI2027 | 4.562 / 14.447 / 28.944 | 8.840 / 20.652 / 21.339 |

SI2026 实际保存统计Tensor payload为689,280 bytes；另有运行时成员索引和4ratio labels，按当前Tensor布局约1.20 MiB常驻状态，加临时N个FP32贡献。后者是布局计算，**未测量GPU allocator峰值**。模型推理参数量与Forward不增加；训练成本仍明显，当前轻量性目标未通过。

## 10. 交付、复现和下一步

本地脚本及协议：`tests/diagnostics/anchor_si_20261024/`。小型证据222项已下载到 `evidence/` 并逐文件SHA256验证；原始完整模型保留服务器，不移动/删除。v1失败与重跑数据也保留，没有删除负面结果。

服务器命令（在隔离实验根目录、对应CUDA/LibTorch环境下）：

```bash
python setup.py                  # 从源码清单恢复隔离源码并构建四个目标
python run_experiments.py        # 按commands.json跳过已完成对照；重新实验需独立输出目录
python final_evaluation.py       # 顺序 micro → frozen评价 → online评价 → probe
python analyse.py
python regression.py
python verify_source.py
```

`run_experiments.py` 不覆盖既有实验目录；从头复现需复制脚本并使用新的独立实验根目录。C++使用LibTorch ABI1，Python评价使用Torch ABI0，Python步骤清除C++的 `LD_LIBRARY_PATH`，不可混载。

按证据优先级建议：

1. **保留当前默认关闭**。没有整体质量收益，实际Cached路径退化，开销未足够低。
2. 下一轮保持网络结构不变，预注册比较概率级 L2 与归一化 raw-logit 级 L2，并在Bank匹配监督梯度尺度，验证softmax饱和是否是主要可修复限制。不能依据当前non-Bank挑选权重。
3. 在同一固定地图下检查Full贡献teacher与LoD/R3的实际photo边际收益是否一致，避免继续优化一个与目标不一致的teacher；独立场景确认后再考虑上线。
4. 只有出现稳定质量收益才继续投入轻量化：复用私有head中间结果，融合MSE与定向head VJP；当前约0.7ms开销比继续优化几个微小归约kernel更值得关注。

本轮没有自动增加网络、调Loss权重、改变ratio调度或改R3；没有用非Bank结果选择新的模型版本。
