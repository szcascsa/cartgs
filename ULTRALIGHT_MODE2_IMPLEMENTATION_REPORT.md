# UltraLight Mode 2 实现与验收报告

> 本文保留历史实现与验收结果。当前 mode2 已取消整段 LoD 启动等待，Full 遵循 GausPyramid，并在有效 Full 提交后更新成熟度；旧报告中的 native Full、整体 warmup 和成熟度冻结描述不再代表当前实现。当前契约见 [训练模式](improvements/ultra_light_training.md)，历史验收不能替代修改后的运行验证。

## 1. 交付与验证范围

代码位于当前 `cuda-tf` 分支，起点为 `e9e9a1737817b9b571dfdf64d83b6c9fae5f2ad1`。实现依据是完整阅读并核对的 `ULTRALIGHT_MODE2_AUDIT.md`，不是历史 UltraLight 文档。生产实现版本为 `2167e5f13d0fbe7846349ca662451742b18ed0b8`。

Mode2 已实现在线 Mapping 内部的 **原 CaRtGS P 顺序 Full → 完整 detached LoD**。正常训练仍只有一个 shared Adam state owner，不创建平行 mapper、正常 Selector optimizer 或 TF optimizer。Full-only parity、真实 CUDA detach/state 隔离和完整 46 项 CTest 已通过；在线 A/B 结果见第 8 节。

重要边界：P 顺序保留原始 topology 替换 Parameter 并丢失当前 grad 的行为。因此“每轮 Full forward/backward”成立，但 topology 轮不保证发生有效 Gaussian Adam 参数组提交。报告将 Adam 调用次数与非空、grad defined、LR 非零的 Gaussian 参数组提交次数分开，不把空 step 当作成功 Full evidence；也不把后者解释为所有参数元素都发生 bitwise 数值变化。

## 2. 分阶段提交

| Commit | 实际内容 |
|---|---|
| `571208a` | mode0/1/2 解析、Mode2 两阶段 P 调度、完整 detach、normal ratio cursor、阶段隔离提交与计数 |
| `1ae5d9e` | 固定调度 Full-only oracle、CUDA 梯度/Adam/topology/clock/特殊路径测试、microbenchmark |
| `dfd8dc2` | Replay 测试显式选择 mode0，避免用户可变 fast 配置改变旧测试前提 |
| `16e7583` | 各模式阶段统计、online/tail 快照和 profiler 汇总、门控在线 A/B 驱动 |
| `b0410da` | 导出/打印实际生效的 Mode2 detach、native Full、LoD scale 和 warmup 语义 |
| `2167e5f` | Full warmup 入口清理尚未附着 shared Adam 的辅助参数 stale grad，并增加测试 |
| `cf2b96d` | 仅diagnostic：修正evo APE字段提取，支持batch结束后重读已有metrics |
| `1772546` | 仅测试：defined-grad辅助参数Adam step必须逐参数+1，Selector/TF planes/MLP必须实际改变参数值 |

所有提交均为本地提交，未 push。用户原有 `cfg/selector/fast.yaml` 本文件和旧 CUDA TF diagnostic 文件修改保留，不纳入上述提交；仅单独保留本批冻结输入副本 `tests/diagnostics/ultralight_mode2_20261014/input_fast.yaml` 用于复现。

## 3. 实际 iteration 流程

入口为 `src/gaussian_mapper.cpp:1219` 的 `trainForOneIteration`，主体为 `:1350` 的 `trainMappingIteration`。KF 只采样一次，两个阶段复用同一个 KF；主体持有同一个 render mutex，LoD 开始时 Full 的 `NoGradGuard` 已结束。

```text
normal budget/stop preflight → 单次 KF sampling → 原有 Replay（若独立配置触发）
→ shared/辅助 grad 清为 undefined
→ Protection.advance(mapping_iter + 1)，一次
→ Gaussian 原 LR / SH schedule
→ native Full render → 原 Full loss → Full backward
→ Full KF loss/EMA/record
→ Full radii + viewspace-gradient statistics
→ 原 densify/clone/split/prune/opacity-reset modulo（before step）
→ 原 training report / KF bookkeeping
→ Gaussian-only shared Adam step → clear gradients
→ Protection.finishFullMapping(实际 Gaussian 提交证据)
→ 退出 Full NoGrad，继续持有 render mutex
→ normal LoD preflight（Selector、budget、map、TF warmup）
→ ratio = target_ratios[normal_lod_branch_count % size]
→ 重读 topology 后 Master / Protection / indices，完整 detached view
→ 原 Protection budget + 原 Selector ST + 原 TF
→ native RGB cache resize → half-resolution differentiable renderer
→ 原 photometric/ratio/GI/TF regularization loss → LoD backward
→ S/TF-only mixed Adam commit → clear gradients
→ 实际 S/TF commit clocks；normal ratio attempt cursor +1
→ 主体返回 → mapping_iter +1 → 原 GI/AABB/DSD 周期维护
```

P 顺序不是 U 顺序：`densifyAndPrune`、`resetOpacity` 和原 bookkeeping 在 Gaussian commit 之前。原 topology 算法、参数替换规则、optimizer moment 搬运规则均未改。LoD 位于 Full statistics/topology/commit/clear/Protection finish 之后、外层 Mapping clock 递增之前。

Mode2 不启用 mode1 rotation，因此不继承五 slot、deferred topology 或 online-only 退出条件。原 CaRtGS 的 online/tail 停止逻辑与显式 cap/forced-tail 配置继续独立生效。tail 仍是正常 Mapping，不新增离线 LoD 后训练。

## 4. 配置、optimizer 与梯度隔离

### 4.1 配置

```yaml
Training.ultra_light_mode: 2
Selector.enabled: 1
Selector.target_ratios: [0.01, 0.05, 0.10, 0.15]
Selector.lod_render_scale: 0.5
TransformField.start_iter: 3000
GradientConstraint.enable: 0
```

显式 mode 优先，缺省使用旧 `Training.ultra_light_enabled` 的 0/1 映射。非法 mode 报错。Mode2 缺省 ratios/scale/start 分别为上述值；已有显式 Selector/TF 配置仍优先，不新增重复参数。`Selector.enabled: 0` 是 Full-only。

TF enabled 时，整个正常 LoD branch 等待 completed `mapping_iter_ >= start_iter`；3000 对应第 3001 个正常 Mapping。此前每步 Full，不消耗 ratio、不推进辅助 update clock。TF disabled 时，Selector-only 可从 start0 运行。

Mode2 + Guard=1 在 constructor 明确报错。第一版不实现 stage-aware auxiliary Guard，也不调用 joint `stepWithHistoricalGuard` 两次。Mode0/1 的原 Guard 语义保留。

### 4.2 唯一 shared Adam state owner

| 参数 | 正常 state owner | Full 阶段 | LoD 阶段 |
|---|---|---|---|
| XYZ、SH DC/rest、opacity、scaling、rotation，groups 0–5 | `gaussians_->optimizer_` | 原 loss/backward/step；topology 后 undefined 参数自然 skip | 六组 grad 必须 undefined，参数/m/v/step 不变 |
| Selector 附加 group | 同上 | grad undefined，参数/m/v/step 不变 | ST 图产生 grad，shared Adam 提交 |
| TF MLP、TF planes 附加 groups | 同上 | grad undefined，参数/m/v/step 不变 | CUDA mixed adapter 或 legacy Adam 提交 |

`commitMode2Stage`（`:1706`）检查跨阶段 defined grad 并报错。Full 调用原 shared Adam；LoD 调用原 `tfCudaAdamMixedStep`，CUDA TF kernel 与 LibTorch Adam 仍使用同一 state。无梯度时 Adam skip；不能用零 Tensor 或 LR=0 替代隔离。

两阶段各自提交并各自 clear。Full 入口也用已有辅助 optimizer handles 清梯度，仅用于覆盖尚未附着 shared Adam 的参数，不调用其 step。这样 CD 遗留梯度不会在 warmup 结束时误提交。

辅助 nonfinite loss 不 backward/commit；nonfinite auxiliary grad 不 commit；均 clear 并计为一次已执行 LoD attempt。保留原 Full 的有限值处理语义，没有借本任务改变 Full 数学。

### 4.3 完整 detach 与可微路径

`accumulateLodGradients`（函数`:2844`、normal flag处理`:2853`）仅新增 normal Mode2 调用点 flag，强制以下六类 view detach：XYZ、activated scale、normalized rotation、activated opacity、SH DC、SH rest。Selector input、TF input、gather 全从该 view 读取。

```text
Master attributes.detach() ─→ Selector input → score → hard Gumbel ST mask
detached opacity × differentiable ST mask ─→ renderer → LoD loss → Selector
detached XYZ/scale/rotation ─→ 原 TF residual/output ─→ renderer → LoD loss → TF
```

不 detach mask、residual 或 transformed output，不使用 inference renderer，不修改 TF XYZ/scale/rotation heads、数学 backend 或 CUDA kernel。Renderer input autograd 保持，Master `.grad` 全 undefined。

### 4.4 半分辨率与 loss

normal Mode2 始终从本轮 KF native `original_image_` resize，包括 tail；不从 pyramid 二次缩小。H/W 使用 `max(1, floor(native_size * scale))`。复用 `ResizedImageCache` 的 bilinear、`align_corners(false)`，只向同一 KF/FoV/projection 的 renderer 传新 H/W。

保留 rasterizer 的 integer pixel-center / half-pixel resize 对应关系，不复制 Camera、不修改 K。已用真实 CUDA renderer 验证偶数、奇数和非 tile 对齐尺寸。loss 均沿用原权重；mean photometric loss 不乘/除4，SSIM window 不变，不加入 depth/mask loss。

## 5. Protection、statistics、clocks 和特殊路径

Protection age 每正常 Mapping advance 一次。optimized_count 与 unique KF support 仅通过 Full transaction finish 的成功证据更新。topology 替换参数导致空 step 时 finish=false，不产生虚假的 Full 优化 evidence。LoD 不 begin/finish Full transaction，只保留已有 elastic supervision metadata 更新。

LoD 重新计算 post-topology Mature/protected pools、budget、population/indices。Full `RenderPackage` 的 viewspace gradient、radii、visibility、Full GI contribution 与 LoD package 不混用；LoD 不调用 Master densification statistics。原 append/prune callbacks 继续同步 Protection/GI/version/metadata。

| Clock | Mode2 推进事件 | 不推进的情况 |
|---|---|---|
| `mapping_iter_` | 一个正常 Mapping 主体完成，一次 | 不能因两个 backward/commit 加两次 |
| `iteration_` / Gaussian LR | 原 Mapping iteration 调度 | LoD commit 不再调用/推进 Gaussian LR |
| `normal_lod_branch_count_` | 实际进入一个 normal LoD attempt，一次 | warmup/disabled/cap preflight skip、Full、Replay/CD/DSD/GI/Guard objective |
| `selector_step_` | 实际正常 Selector 提交 | Protected-only noop、nonfinite、Full、特殊训练 |
| `normal_update_clock_.tf_updates` | 实际正常 TF 提交 | 同上；不与 ratio attempt 绑定 |

ratio 采用 `[.01,.05,.10,.15]` 循环，单轮至多一个；noop/nonfinite 仍推进 attempt cursor。Guard rejection 的 stage-aware 路径本版不支持，已 fail-closed，不伪造 reject 验收。

Replay 仍是原 joint Full+LoD/shared Adam，包括 Gaussian 更新；其调用不传 normal Mode2 flag。CD 保留 geometry event capture、独立 event optimizer 与即时 distill。DSD 保留原临时训练/恢复。没有向特殊模块传播全局 detach、half-resolution 或 normal ratio cursor。

## 6. Gates A/B/C：实际 CUDA 验收

目标机器：`172.19.163.58`，RTX 3090 24 GiB；conda `zwj`，Torch/LibTorch 2.3.1、CUDA 11.8，现有 GCC9/sm86 build。密码不保存到任何源码、日志或报告。最终验收记录：`tests/diagnostics/ultralight_mode2_20261014/acceptance.json`，`complete=true`、commit=`1772546`，全部命令 exit0；关键 GPU 测试未 skip。在线E准入时的原验收版本为`2167e5f`，保留在`online_admission_2167e5f.json`及`pre_strict_aux_gates.tar.gz`；之后只改测试断言/diagnostic，不改生产源码。E完成后重新编译并重跑全部A/B/C/D，验证最终测试代码。

| Gate | 实际内容与结果 |
|---|---|
| A，2 tests | mode解析/兼容/default/invalid/Guard；10轮固定 map/KF/RNG Full-only 原调度 oracle 全通过 |
| B，3 tests | 真实 CUDA renderer、legacy/CUDA TF；两阶段参数/m/v/step 隔离、ST/head 梯度、topology、ratio/start/cap/tail/nonfinite、Replay/CD/DSD probe 全通过 |
| C，46 tests | 所有 CTest 46/46，通过；含 mode1/forced-tail、Guard、CD、Replay、DSD recovery、Protection、compact LoD、TF optimizer/backend 测试 |

Full-only oracle 取自原始 baseline `547c9055f52529dd920cdb3e4211d8e1c0ee68e3` mapper 的原控制流，调用当前未修改的 renderer/Gaussian/topology 数学。它不是另外编译运行的历史 CaRtGS binary。fixture 固定 native camera、SH1000边界、LR、单KF/RNG和topology modulo，关闭额外 GI/Guard/Replay/CD/DSD/adaptive sampling，避免在线异步调度干扰。

逐轮比较 Gaussian 六组 raw attrs、Adam m/v/step、Full loss/EMA、KF use count、iteration/Mapping clock、densification buffers；覆盖 topology due 但零新增/删除仍替换 Tensor、实际 split、prune、opacity reset。数值 parity 使用 `allclose(rtol=1e-5, atol=1e-7)`；loss/EMA 误差阈值 `1e-6`。不宣称与两个独立在线 SLAM run 逐 bit 相同。

两阶段测试的 Master attrs/Adam 隔离使用 CPU byte comparison，要求 bitwise 不变；在 clear 之前检查六组 `.grad` 全 undefined。photo-only 测试关闭 ratio/GI/TF regularization，验证 Selector ST、planes/MLP 和 XYZ/scale/rotation heads 梯度非零，不以 regularization gradient 代替 photo gradient。每个defined-grad辅助参数必须Adam step恰好+1，undefined-grad参数不得推进；Selector、TF planes、TF MLP各至少一个参数值必须实际改变。重复两轮覆盖已有 Adam momentum，检查不存在第二次 Gaussian step。

额外边界：Full 清除 stale aux grad；未附着 shared Adam 的 warmup Selector stale grad清除；aux提交若有 stale Gaussian grad则报错；topology 后 split/prune新indices有效；Protected-only noop旋转ratio但不推进aux提交clock；nonfinite attempt保持state且clear；实际Replay修改Master但不推进normalcursor；实际geometry CD TF step>0；DSD mapper probe保留clock/参数，DSD训练恢复由原独立recovery tests覆盖。

尺寸验收：`640×480→320×240`、`641×479→320×239`、`53×37→26×18` 均实际 render/backward 通过。

## 7. Gate D：短性能测试

固定 2048 Gaussian、native640×480，15轮 warmup + 50轮 measured，无 topology；所有变体调用真实实现/原 Full oracle。只有 benchmark batch 边界 synchronize，无 production 每轮 device synchronization。此结果不能代替真实场景完整 Mapping throughput。

| 模式 | complete mean ms | host p95 ms | Full / 有效GS / S / TF（含warmup65轮） |
|---|---:|---:|---|
| 原 Full-only oracle | 2.5194 | 2.6978 | 65 / 65 / 0 / 0，oracle固定65轮，无topology |
| Mode1（null-SLAM fixture） | 4.6476 | 5.2766 | 13 / 65 / 52 / 52 |
| Mode2 Full-only | 2.3376 | 2.4324 | 65 / 65 / 0 / 0 |
| Mode2 half-LoD | 7.7757 | 8.0679 | 65 / 65 / 65 / 65 |
| Mode2 native-LoD | 7.9171 | 8.0588 | 65 / 65 / 65 / 65 |

2048点的小 fixture 中 half-resolution 收益较小：不减少 Selector/TF 的点数计算，仅降低 raster/image-loss/backward 像素成本。Full-only 间微小差异不构成速度收益结论；Mode1 原有同步 timer 保留，不将该差异归因于 TF kernel。另一个明确限制是fixture无SLAM对象：旧Mode1分辨率选择按`pSLAM_ && !isShutDown()`判断，故该micro的Mode1 LoD实际为native，而非线上half；不把它作为与Mode2 half-LoD完全等价的online速度比较。真实Mode1 half-resolution线上耗时由GateE报告。

Mode1的65次有效GS提交包括52次LoD对Master的joint更新，不能误解为65次Full。原oracle不调用生产stage计数器，其日志的stage-counter0是未instrumented而不是未训练；上表oracle次数来自固定65轮且无topology/finitecap的fixture。其每轮stage拆分不报告；其余变体的50轮measured host stage如下（ms/Mapping，包含关系非互斥）：

| 模式 | Full render | Full backward | Gaussian optimizer | LoD总段 | LoD render | LoD backward | LoD optimizer / 原shared optimizer |
|---|---:|---:|---:|---:|---:|---:|---:|
| Mode1 | 0.1033 | 0.1175 | — | 3.3286 | 0.2299 | 1.6990 | shared 0.8618 |
| Mode2 Full-only | 0.4659 | 0.5856 | 0.2471 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Mode2 half-LoD | 0.4805 | 0.5920 | 0.2495 | 3.4544 | 0.2239 | 1.6171 | 1.9248 |
| Mode2 native-LoD | 0.4763 | 0.5997 | 0.2498 | 3.5919 | 0.2600 | 1.6182 | 1.9310 |

Profiler 新增独立 `GaussianOptimizer` 和 `LodOptimizer` event pair，及 FullLoss/LodLoss/TF stage，不复用被覆盖的同一 Optimizer event。每19轮采样、query已完成event；online/tail host mean/p95分开。scope有包含关系，不能把Total和所有nested scopes简单相加。辅助 finite检查与已有 loss读取会产生标量同步，未新增每轮 `cudaDeviceSynchronize`。

## 8. Gate E：完整在线 A/B

<!-- ONLINE_RESULTS_BEGIN -->
12/12完整在线run及全部有效匹配帧评估完成，训练/评估命令均exit0。Mode2各场景满足Full render/backward次数=有效Mapping次数；正常LoD完全遵守warmup与单ratio attempt cursor。以下是观测结果，不代表质量验收阈值已统一改善。

### 8.1 Full质量、轨迹与Gaussian规模

每格箭头为online→final；Mode1无tail，二者相同。APE为translation RMSE，单位m。Full-only控制不提供LoD指标。

| 场景 | 模式 | PSNR dB | SSIM | LPIPS | APE RMSE m | Gaussian online→final | 帧数 |
|---|---|---|---|---|---|---|---|
| room | Full-only | 17.957→18.185 | 0.6412→0.6473 | 0.4614→0.4535 | 0.055824 | 77807→77807 | 1352 |
| room | Mode1 | 15.543→15.543 | 0.5809→0.5809 | 0.6083→0.6083 | 0.047039 | 134485→134485 | 1352 |
| room | Mode2 | 16.409→16.740 | 0.6077→0.6172 | 0.5537→0.5374 | 0.061276 | 117547→81735 | 1352 |
| desk | Full-only | 19.967→20.060 | 0.7121→0.7153 | 0.3403→0.3328 | 0.015252 | 37284→36041 | 592 |
| desk | Mode1 | 17.827→17.827 | 0.6460→0.6460 | 0.4701→0.4701 | 0.014814 | 65623→65623 | 592 |
| desk | Mode2 | 19.206→19.385 | 0.6904→0.7003 | 0.3840→0.3673 | 0.014705 | 52552→39419 | 592 |
| xyz | Full-only | 22.808→22.725 | 0.7518→0.7511 | 0.2292→0.2310 | 0.003078 | 41685→41974 | 3397 |
| xyz | Mode1 | 21.404→21.404 | 0.7111→0.7111 | 0.2918→0.2918 | 0.003037 | 33301→33301 | 3397 |
| xyz | Mode2 | 22.073→22.357 | 0.7352→0.7405 | 0.2505→0.2432 | 0.003398 | 30745→31057 | 3397 |
| long | Full-only | 18.264→18.313 | 0.6572→0.6569 | 0.3972→0.3963 | 0.031767 | 62479→62479 | 2515 |
| long | Mode1 | 17.681→17.681 | 0.6409→0.6409 | 0.4049→0.4049 | 0.030358 | 137618→137618 | 2515 |
| long | Mode2 | 19.615→19.919 | 0.6984→0.7072 | 0.3059→0.2947 | 0.018607 | 111833→106300 | 2515 |

### 8.2 各ratio与Full（ratio=1）质量及高斯数量

#### PSNR、高斯数量与成熟占比汇总（final）

每格为`PSNR（dB） / 高斯数量 / 成熟高斯数量（成熟占比）`。以下统一使用final checkpoint：Full-only和Mode2包含各自tail，Mode1无tail。所有渲染指标均在native分辨率评估，半分辨率仅用于Mode2正常LoD训练。

LoD高斯数量使用实测`selector_stats.active_gaussians`，成熟数量使用`mature_actual`，成熟占比为`mature_actual / active_gaussians`。它们是实际选择参与渲染的高斯数，不是可见数，也不是`ratio × Master总数`的理论值。ratio=1列为原有Full Master渲染指标，高斯数使用`gaussian_total`，成熟数来自该final checkpoint的`protection_state.json:mature_gaussians`，占比为`Master成熟数 / Master总数`；不是新增一次Selector ratio=1评估。Full-only未执行LoD评估，以`—`表示，但其Full成熟数已从checkpoint补取。

本轮配置的Mature条件为`age >= 100 && optimized_count >= 80 && unique_keyframe_count >= 4`；这里报告满足条件的点数比例，不是`mean_maturity_confidence`。同一Master在各ratio下的总成熟数相同，但每个LoD实际选中的成熟数不同，不能用Master总成熟数代替LoD的`mature_actual`。

| 场景 | 模式 | ratio 0.01 | ratio 0.05 | ratio 0.10 | ratio 0.15 | ratio 1（Full） |
|---|---|---|---|---|---|---|
| room | Full-only | — | — | — | — | 18.185 / 77,807 / 69,804（89.71%） |
| room | Mode1 | 14.111 / 1,421 / 937（65.94%） | 15.107 / 6,559 / 4,132（63.00%） | 15.256 / 13,455 / 8,597（63.89%） | 15.306 / 19,880 / 12,665（63.71%） | 15.543 / 134,485 / 44,318（32.95%） |
| room | Mode2 | 12.194 / 1,401 / 1,110（79.23%） | 13.899 / 3,825 / 2,383（62.30%） | 14.933 / 8,354 / 5,469（65.47%） | 15.328 / 12,187 / 7,920（64.99%） | 16.740 / 81,735 / 71,050（86.93%） |
| desk | Full-only | — | — | — | — | 20.060 / 36,041 / 35,896（99.60%） |
| desk | Mode1 | 15.909 / 1,011 / 785（77.65%） | 16.873 / 3,193 / 2,014（63.08%） | 17.398 / 6,547 / 4,278（65.34%） | 17.539 / 9,387 / 5,999（63.91%） | 17.827 / 65,623 / 43,786（66.72%） |
| desk | Mode2 | 12.357 / 1,713 / 1,618（94.45%） | 13.267 / 2,172 / 2,077（95.63%） | 14.320 / 2,888 / 2,793（96.71%） | 15.426 / 3,979 / 3,884（97.61%） | 19.385 / 39,419 / 39,096（99.18%） |
| xyz | Full-only | — | — | — | — | 22.725 / 41,974 / 41,756（99.48%） |
| xyz | Mode1 | 17.839 / 333 / 245（73.57%） | 19.713 / 1,642 / 1,100（66.99%） | 20.400 / 2,964 / 2,196（74.09%） | 20.518 / 3,623 / 2,855（78.80%） | 21.404 / 33,301 / 31,504（94.60%） |
| xyz | Mode2 | 14.559 / 344 / 308（89.53%） | 17.134 / 1,607 / 1,571（97.76%） | 17.945 / 3,145 / 3,109（98.86%） | 18.436 / 4,496 / 4,460（99.20%） | 22.357 / 31,057 / 30,978（99.75%） |
| long | Full-only | — | — | — | — | 18.313 / 62,479 / 62,443（99.94%） |
| long | Mode1 | 16.868 / 1,704 / 1,231（72.24%） | 17.492 / 6,967 / 4,601（66.04%） | 17.673 / 13,797 / 9,061（65.67%） | 17.693 / 18,194 / 11,753（64.60%） | 17.681 / 137,618 / 119,509（86.84%） |
| long | Mode2 | 15.179 / 1,152 / 1,135（98.52%） | 17.723 / 5,573 / 5,556（99.69%） | 18.376 / 10,866 / 10,849（99.84%） | 18.724 / 15,714 / 15,697（99.89%） | 19.919 / 106,300 / 106,248（99.95%） |

数据来源：`tests/diagnostics/ultralight_mode2_20261014/online_summary.json:quality.final`及服务器各final checkpoint的`protection_state.json`。补取路径、状态计数和未四舍五入比例保存在`tests/diagnostics/ultralight_mode2_20261014/final_mature_counts.json`。已核对12个checkpoint的`Mature + Protected = Gaussian总数`，并核对Mode1/Mode2四个ratio的Master成熟数一致。仅只读补取统计，没有重新训练或渲染。

#### LoD完整指标与online/final分离

每格为`PSNR / SSIM / LPIPS；active count`；native评估。Mode1只列online/final同一结果，Mode2分别列online和final。

| 场景 | 模式/阶段 | ratio 0.01 | ratio 0.05 | ratio 0.10 | ratio 0.15 |
|---|---|---|---|---|---|
| room | Mode1/online=final | 14.111 / 0.5532 / 0.7270；1421 | 15.107 / 0.5774 / 0.6565；6559 | 15.256 / 0.5807 / 0.6354；13455 | 15.306 / 0.5808 / 0.6286；19880 |
| room | Mode2/online | 12.265 / 0.5210 / 0.7434；1628 | 13.656 / 0.5478 / 0.7008；4543 | 14.793 / 0.5686 / 0.6613；10252 | 15.134 / 0.5758 / 0.6458；15098 |
| room | Mode2/final | 12.194 / 0.5218 / 0.7459；1401 | 13.899 / 0.5525 / 0.6969；3825 | 14.933 / 0.5726 / 0.6515；8354 | 15.328 / 0.5809 / 0.6343；12187 |
| desk | Mode1/online=final | 15.909 / 0.5879 / 0.6229；1011 | 16.873 / 0.6150 / 0.5553；3193 | 17.398 / 0.6317 / 0.5176；6547 | 17.539 / 0.6359 / 0.5025；9387 |
| desk | Mode2/online | 12.729 / 0.5222 / 0.5786；2274 | 12.882 / 0.5256 / 0.5728；2965 | 13.090 / 0.5308 / 0.5657；3100 | 13.231 / 0.5352 / 0.5612；3225 |
| desk | Mode2/final | 12.357 / 0.5191 / 0.5955；1713 | 13.267 / 0.5411 / 0.5681；2172 | 14.320 / 0.5679 / 0.5383；2888 | 15.426 / 0.5928 / 0.5033；3979 |
| xyz | Mode1/online=final | 17.839 / 0.6053 / 0.5949；333 | 19.713 / 0.6489 / 0.4630；1642 | 20.400 / 0.6697 / 0.4140；2964 | 20.518 / 0.6741 / 0.4039；3623 |
| xyz | Mode2/online | 14.489 / 0.5545 / 0.7189；304 | 16.804 / 0.5890 / 0.5856；1428 | 17.767 / 0.6106 / 0.4959；2917 | 18.358 / 0.6264 / 0.4493；4331 |
| xyz | Mode2/final | 14.559 / 0.5559 / 0.7137；344 | 17.134 / 0.5944 / 0.5804；1607 | 17.945 / 0.6165 / 0.4937；3145 | 18.436 / 0.6307 / 0.4415；4496 |
| long | Mode1/online=final | 16.868 / 0.6262 / 0.6100；1704 | 17.492 / 0.6390 / 0.4897；6967 | 17.673 / 0.6430 / 0.4463；13797 | 17.693 / 0.6427 / 0.4360；18194 |
| long | Mode2/online | 14.774 / 0.5901 / 0.6842；1172 | 17.601 / 0.6433 / 0.4954；5700 | 18.357 / 0.6628 / 0.4203；11640 | 18.745 / 0.6726 / 0.3887；17004 |
| long | Mode2/final | 15.179 / 0.5999 / 0.6706；1152 | 17.723 / 0.6467 / 0.4904；5573 | 18.376 / 0.6641 / 0.4199；10866 | 18.724 / 0.6734 / 0.3872；15714 |

### 8.3 更新量与ratio分布

GS栏为`Adam调用/有效Gaussian参数组提交`；Mode1有效GS包括LoD joint更新，不能当作Full更新。所有Mode1 tail均0，不另列空行。S/TF均为正常Mapping commits，不包含CD等event optimizer。ratio次数顺序为.01/.05/.10/.15。

| 场景/模式 | 阶段 | Mapping | Full | GS调用/有效 | S | TF | LoD attempts | ratio次数 |
|---|---|---:|---:|---|---:|---:|---:|---|
| room/Full-only | online | 16018 | 16018 | 16018/15987 | 0 | 0 | 0 | 0/0/0/0 |
| room/Full-only | tail | 400 | 400 | 400/400 | 0 | 0 | 0 | 0/0/0/0 |
| room/Mode1 | online | 6975 | 1395 | 6975/6975 | 5260 | 3180 | 5580 | 1395/1395/1395/1395 |
| room/Mode2 | online | 5925 | 5925 | 5925/5915 | 2925 | 2925 | 2925 | 732/731/731/731 |
| room/Mode2 | tail | 475 | 475 | 475/474 | 475 | 475 | 475 | 118/119/119/119 |
| desk/Full-only | online | 7134 | 7134 | 7134/7121 | 0 | 0 | 0 | 0/0/0/0 |
| desk/Full-only | tail | 766 | 766 | 766/765 | 0 | 0 | 0 | 0/0/0/0 |
| desk/Mode1 | online | 3673 | 734 | 3673/3673 | 2619 | 539 | 2939 | 735/735/735/734 |
| desk/Mode2 | online | 3952 | 3952 | 3952/3946 | 952 | 952 | 952 | 238/238/238/238 |
| desk/Mode2 | tail | 448 | 448 | 448/447 | 448 | 448 | 448 | 112/112/112/112 |
| xyz/Full-only | online | 38181 | 38181 | 38181/38106 | 0 | 0 | 0 | 0/0/0/0 |
| xyz/Full-only | tail | 719 | 719 | 719/718 | 0 | 0 | 0 | 0/0/0/0 |
| xyz/Mode1 | online | 21795 | 4359 | 21795/21795 | 17116 | 15036 | 17436 | 4359/4359/4359/4359 |
| xyz/Mode2 | online | 14286 | 14286 | 14286/14259 | 11286 | 11286 | 11286 | 2822/2822/2821/2821 |
| xyz/Mode2 | tail | 614 | 614 | 614/613 | 614 | 614 | 614 | 153/153/154/154 |
| long/Full-only | online | 28568 | 28568 | 28568/28512 | 0 | 0 | 0 | 0/0/0/0 |
| long/Full-only | tail | 400 | 400 | 400/400 | 0 | 0 | 0 | 0/0/0/0 |
| long/Mode1 | online | 11438 | 2287 | 11438/11438 | 8831 | 6751 | 9151 | 2288/2288/2288/2287 |
| long/Mode2 | online | 8603 | 8603 | 8603/8587 | 5603 | 5603 | 5603 | 1401/1401/1401/1400 |
| long/Mode2 | tail | 797 | 797 | 797/796 | 797 | 797 | 797 | 199/199/199/200 |

### 8.4 Mapping mean/p95与阶段开销

单位ms，全部为该阶段累计host wall；不是训练进程总wall、单kernel时间或各stage互斥之和。Full-only的原shared optimizer仅提交GS；Mode1是joint，无法拆分；Mode2分列GS和LoD。LoD总段不含随后LoD optimizer，TF在LoD总段内。

| 场景/模式 | 阶段 | Mapping mean/p95 | Full render/backward/loss | GS optimizer | LoD总段/render/backward | TF forward | LoD optimizer（Mode1为joint） |
|---|---|---|---|---:|---|---:|---:|
| room/Full-only | online | 3.540/4.101 | 0.728/0.721/0.183 | 0.293 | 0.001/0.000/0.000 | 0.000 | 0.000 |
| room/Full-only | tail | 3.392/3.823 | 0.887/0.602/0.161 | 0.266 | 0.000/0.000/0.000 | 0.000 | 0.000 |
| room/Mode1 | online | 8.119/18.117 | 0.136/0.147/0.038 | — | 4.901/0.256/2.591 | 0.217 | 0.943 |
| room/Mode2 | online | 9.513/22.779 | 0.616/0.744/0.183 | 0.289 | 3.640/0.163/1.858 | 0.164 | 1.246 |
| room/Mode2 | tail | 13.679/14.307 | 0.650/0.651/0.168 | 0.262 | 7.771/0.342/3.567 | 0.285 | 2.370 |
| desk/Full-only | online | 3.458/4.020 | 0.648/0.751/0.185 | 0.295 | 0.001/0.000/0.000 | 0.000 | 0.000 |
| desk/Full-only | tail | 3.255/4.034 | 0.696/0.673/0.168 | 0.280 | 0.000/0.000/0.000 | 0.000 | 0.000 |
| desk/Mode1 | online | 6.663/11.620 | 0.146/0.151/0.039 | — | 4.068/0.266/1.916 | 0.050 | 0.880 |
| desk/Mode2 | online | 6.107/15.394 | 0.615/0.757/0.190 | 0.295 | 1.529/0.077/0.739 | 0.081 | 0.626 |
| desk/Mode2 | tail | 11.720/12.386 | 0.676/0.646/0.165 | 0.269 | 5.980/0.292/3.081 | 0.274 | 2.364 |
| xyz/Full-only | online | 3.787/4.387 | 0.941/0.669/0.170 | 0.277 | 0.001/0.000/0.000 | 0.000 | 0.000 |
| xyz/Full-only | tail | 4.201/4.347 | 1.166/0.597/0.155 | 0.258 | 0.000/0.000/0.000 | 0.000 | 0.000 |
| xyz/Mode1 | online | 6.647/8.929 | 0.172/0.140/0.036 | — | 4.771/0.274/2.619 | 0.207 | 0.954 |
| xyz/Mode2 | online | 10.121/13.761 | 0.761/0.702/0.174 | 0.278 | 4.551/0.272/2.390 | 0.232 | 1.940 |
| xyz/Mode2 | tail | 11.339/12.169 | 0.900/0.621/0.159 | 0.257 | 5.384/0.349/2.749 | 0.258 | 2.284 |
| long/Full-only | online | 4.105/4.745 | 0.947/0.770/0.188 | 0.298 | 0.001/0.000/0.000 | 0.000 | 0.000 |
| long/Full-only | tail | 3.763/4.067 | 0.990/0.611/0.156 | 0.264 | 0.000/0.000/0.000 | 0.000 | 0.000 |
| long/Mode1 | online | 10.063/20.462 | 0.194/0.158/0.041 | — | 6.606/0.300/3.135 | 0.336 | 0.988 |
| long/Mode2 | online | 13.322/28.975 | 0.875/0.792/0.191 | 0.293 | 6.043/0.245/2.715 | 0.374 | 1.690 |
| long/Mode2 | tail | 14.712/15.801 | 0.938/0.636/0.164 | 0.253 | 8.237/0.362/3.127 | 0.534 | 2.335 |

### 8.5 明确结论

- 相对Mode1，Mode2 final Full PSNR在room/desk/xyz/long分别变化+1.197、+1.558、+0.953、+2.238 dB。Full render恢复为每Mapping一次，而非每五轮一次。
- 这不是比原Full-only控制全面更好：room/desk/xyz final Full仍低于控制，long高于控制。各run的Mapping cycles、KF/trajectory与附加模块配置不同。
- 低ratio LoD并未统一改善，尤其room/desk的0.01下降明显；不能宣称同时提升所有压缩档位。按需求未据此改动Selector/TF算法、warmup或loss权重。
- Mode2增加每轮LoD成本，在线总Mapping cycles少于便宜的Full-only控制；“每步Full”不等于同wall-clock下与Full-only相同Full次数。更多cycles不作为CUDA TF质量收益证据。
- 四场景Mode2合计23100次normal LoD attempt，所有normal LoD nonfinite=0；normal ratio/Selector/TF clocks分别计数。Special-module clock隔离由固定fixture验证，E统一禁用Replay/DSD。
- 本批共清理227824张生成PNG/JPG/JPEG，只清自有run内图像，模型/日志/指标保留。
<!-- ONLINE_RESULTS_END -->

### 协议与解释边界

场景：room/desk/xyz/long，逐场景运行原 Full-only控制、Mode1、Mode2，共12次完整在线输入。`original_full_only` 是当前未改数学/原调度的 mode0+Selector0/TF0/GI0/CD0 控制，不是独立历史二进制。Mode1/2沿用冻结的用户fast其他配置、CUDA TF、start3000、scale.5；本批统一Guard0、Replay0、DSD0、cap0、forced-tail0。具体输入config保留在每个run下。

评估使用原 `python/run.py`、全部有效trajectory/RGB匹配帧，不是仅KF，没有stride/随机抽样；实际evaluated_frames逐run报告，不把dataset中无法匹配trajectory的帧算成已评估。评估的Full与LoD均为native分辨率，与normal Mode2的half-resolution训练区分。Full及四ratio LoD都报告PSNR/SSIM/LPIPS和active count；active count是checkpoint场景级Selector选择数，不是每camera的visible数，也不保证等于ratio×Gaussian总数，Protection预算/ST软预算算法原样保留。原Full-only的LoD为N/A。online结束额外保存诊断快照，原tail之后保存final，两者分别评估；mode1没有tail，online=final。快照不增加optimizer/Mappingclock或新训练阶段。

APE沿用原evaluator：trajectory按0.08s匹配，先`traj_ref.align(traj_est, True)`，再调用align=True的evo APE；CLI未启用`--correct_scale`。本报告不将它改称另一套标准TUM指标。render使用该run的估计trajectory，而非固定GT camera，因此跨run图像指标也包含Tracking差异。

这里是独立SLAM run，线程/Tracking/KF/Mapping调度非确定。质量、Gaussian数、APE和周期数的变化是完整系统观察，不是固定schedule的CUDA TF质量因果证明。online→final差异只能描述tail对该run的贡献，不是跨mode的无混杂对照。固定schedule因果证据仅来自GateA/B。

输出按用户授权保存在服务器 `/home/prmi/zwj/mycartgs/results/newgi/ultralight_mode2_20261014`。评估成功后只清理这个run目录内部生成的PNG/JPG/JPEG，不清模型、日志、指标、已有用户结果或dataset。图像可由保留模型/配置重新render；被删除的PNG本身无保留副本。

## 9. 失败项、修正与保留限制

| 初次失败/风险 | 根因 | 处理与最终结果 |
|---|---|---|
| 空ratio配置未报错 | OpenCV empty sequence 的 `.empty()` 与预期不同 | 显式检查sequence和size==0；GateA通过 |
| 单KF parity fixture首次sampling异常 | 原 adaptive sampler在empty loss边界的既有行为 | 两arm固定关闭adaptive；不改变生产sampler |
| size断言失败 | canonical显式scale1优先于Mode2缺省.5 | fixture显式.5，保留真实配置优先级；GateB通过 |
| 旧Replay两项回归失败 | 测试依赖用户可变fast，旧joint断言却进入mode1 | 测试显式mode0；生产Replay未改，GateC46/46 |
| warmup未附着S可能残留CD梯度 | shared zero_grad不能覆盖未加入group的参数 | Full入口clear原辅助handles，新增边界测试通过 |
| evo headless评估风险 | 用户evo settings强制TkAgg，MPLBACKEND不足 | 本批isolated HOME/.evo使用Agg，保留Torch缓存；实际import/eval通过 |
| APE未进入首版summary | evo输出whitespace而不是colon字段 | diagnostic解析translation/rotation独立stats，并在batch结束后从原指标文件刷新summary |

未实现/不宣称：Mode2 auxiliary Guard（组合报错）；训练checkpoint的完整optimizer/clock resume（沿用此前限制）；另一台GPU/不同LibTorch版本parity；全分辨率LoD完整在线A/B（只有短microbenchmark对照）。没有修改Gaussian topology、Renderer、Protection公式、Selector/TF数学、CD/Replay/DSD算法。

## 10. 复现命令与证据路径

服务器已存在的source/runtime备份：`tests/diagnostics/ultralight_mode2_20261014/pre_mode2_source_runtime.tar.gz`。远端checkout原本没有有效GitHEAD，本任务不reset/init它；本地commit与远端source SHA256对应由 `acceptance.json`核验。启动E前再次核验accepted source hashes，任何差异禁止进入长序列。

```bash
cd /home/prmi/zwj/mycartgs
conda activate zwj
export PATH=/usr/local/cuda-11.8/bin:/usr/bin:/bin:$PATH
CMAKE=/home/prmi/.local/lib/python3.13/site-packages/cmake/data/bin/cmake
CTEST=/home/prmi/.local/lib/python3.13/site-packages/cmake/data/bin/ctest
$CMAKE --build build-cuda118-sm86-gcc9 -j2
$CTEST --test-dir build-cuda118-sm86-gcc9 \
  -R '^ultralight_mode2_(config|full_parity)$' --output-on-failure --timeout 120
$CTEST --test-dir build-cuda118-sm86-gcc9 \
  -R '^ultralight_mode2_(stages|clocks|special)$' --output-on-failure --timeout 120
$CTEST --test-dir build-cuda118-sm86-gcc9 --output-on-failure --timeout 120 -j1
bin/ultralight_mode2_mapping_test benchmark
python tests/diagnostics/ultralight_mode2_20261014/run_acceptance.py --commit 1772546
python tests/diagnostics/ultralight_mode2_20261014/run_online_ab.py --preflight-only
python tests/diagnostics/ultralight_mode2_20261014/run_online_ab.py
# 只在batch退出后：重读已有metrics，不训练、不render
python tests/diagnostics/ultralight_mode2_20261014/run_online_ab.py --refresh-metrics-only
```

以上在线driver跳过已complete run，遇到存在的incomplete目录fail-closed。需要重新完整实验时，先对精确输出目录作可恢复移动/备份，不能覆盖旧结果。每run新输入config、training命令、退出码、全部metrics命令和秒数由summary/driverlog保留；不改用户evo或生产fast.yaml。

单场景实际训练命令模板（`NAME=freiburg1_room`，其他为`freiburg1_desk/freiburg2_xyz/freiburg3_long_office_household`）：

```bash
NAME=freiburg1_room
RUN=/home/prmi/zwj/mycartgs/results/newgi/ultralight_mode2_20261014/room/mode2
bin/tum_rgbd third_party/ORB-SLAM3/Vocabulary/ORBvoc.txt \
  cfg/ORB_SLAM3/RGB-D/TUM/tum_${NAME}.yaml \
  cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml \
  /home/prmi/datasets/TUM_RGBD/rgbd_dataset_${NAME} \
  cfg/ORB_SLAM3/RGB-D/TUM/associations/tum_${NAME}.txt \
  "$RUN" no_viewer --selector-config "$RUN/input_selector.yaml"
```

完整评估命令从summary读取；`eval_source_online/final`是原evaluator支持的read-only checkpoint命名视图，不修改数据/相机/renderer。每ratio配置读取checkpoint保存的selector/TF/config。

## 11. 修改文件与最终检查

| 文件 | 改动范围 |
|---|---|
| `src/gaussian_mapper.cpp` | mode解析、两阶段顺序/提交、normal调用点隔离、native LoD resize、计数/导出/诊断快照 |
| `include/gaussian_mapper.h` | mode/attempt/count状态、stage结果声明 |
| `include/mapping_profiler.h` | 分离优化器event与online/tail汇总 |
| `cfg/selector/selector_config.yaml` | 仅兼容配置说明，不覆盖用户fast |
| `tests/ultralight_mode2_mapping_test.cpp`、`CMakeLists.txt` | 5个新CTest与microbenchmark |
| `tests/mapping_profiler_test.cpp`、`tests/joint_replay_test.cpp` | profiler和可变fixture回归 |
| `tests/diagnostics/ultralight_mode2_20261014/*.py` | 分阶段门控/完整在线评估和安全图像清理 |

<!-- FINAL_VERIFICATION_BEGIN -->
最终检查通过：本地、远端与`1772546`验收manifest的5个source SHA256全部匹配；其中生产mapper/header/profiler/CMake与在线准入`2167e5f`逐字相同，后续仅补强测试与diagnostic。`git diff --check`通过；用户原有3处tracked diff与任务入口保存的binary diff完全一致；未改TF数学、CUDA kernel、Renderer、Gaussian topology或Protection公式。

`online_verification.json`记录12组complete、质量有限值、四场景Full/backward每Mapping一次、topology空step排除、warmup/ratio/实际commit clock校验及图像清理结果。全部训练和评估进程已结束，最终source与binary门控已复核。文档无未完成结果占位；Markdown code fences与标题分隔检查通过。

本地证据位于`tests/diagnostics/ultralight_mode2_20261014/`：最终`acceptance.json`、Gate A/B/C/D与build日志、在线`online_summary.json`和`online_driver.log`、冻结`input_fast.yaml`、`online_verification.json`；历史准入及最初失败fixture日志单独保留。远端模型/完整日志/每帧指标仍保留于批准的results目录，未下载/提交大型模型或既有build zip。
<!-- FINAL_VERIFICATION_END -->
