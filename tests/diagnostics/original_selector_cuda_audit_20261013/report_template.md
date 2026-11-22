# 原始 Selector CUDA 加速可行性审计

审计日期：2026-10-13 至 2026-10-14（Asia/Hong_Kong）。目标机器：`prmi@172.19.163.58`，仓库 `/home/prmi/zwj/mycartgs`，Conda 环境 `zwj`。本文审计当前实现；本轮只新增诊断和报告，未实现 CUDA Selector。

## 结论与证据边界

**当前 Selector 是逐 Gaussian 的四分支小 MLP，4,962 个参数，输出两类 logits；没有 Gaussian Top-K，也没有按 Gaussian 全量排序。** 正常训练采用每行两类 Gumbel softmax + straight-through（ST）。ratio 同时是网络输入和期望预算目标，实际 selected count 不是精确 K。

在真实 checkpoint 的 `Ntotal=220084、Mature/network M=219969、ratio=0.05` 上，无 profiler 的 Native C++ caller 诊断路径 mean wall **8.678 ms**，event **8.614 ms**。该路径包含真实网络、Protection、mask、六类 Gaussian gather、opacity VJP、ratio loss、backward 和 Selector Adam，**不包含 LoD rasterizer、TF、GI teacher、GI bank、Full Mapping 或 Guard 的历史渲染**。不能将它称为完整 Mapping 时间。

最终 Nsight 捕获中，network forward **4.395 ms**、backward **2.855 ms** kernel active time；Gumbel **0.012 ms**，两类 argmax **0.011 ms**，mask + compact indices + gather 合计 **0.098 ms**。大 M 下主要空间是网络及其 backward；小 M 下调度、预算同步和 18 个小参数张量的 Adam 成本更突出。351 kernels/iteration 中 82.14% 短于 10 µs，但短 kernel 数多不等于它们占主要执行时间。

已有历史正常 Mapping 窗口的 GI host stage 占 59%–69%，整个 LoD host stage 占约 20%–23%。即使把整个 LoD stage 理想化为零，该日志口径的 Mapping speedup 也只有约 **1.25–1.29×**；Selector 网络只能消除 LoD 的一部分。GI 全量刷新、bank snapshot 和 contribution render 不应计入 Selector MLP 的收益空间。当前 UltrafastRotation 的 LoD/Full 轮换与这些历史正常窗口不同，不能沿用该上限预测当前全部运行。

已完成 Native C++ build/run、96 个规模/ratio/mode case 的 48,000 个基准样本、280 个独立阶段 case（各 100 warmup + 500 measurement）、11 条 loss/gradient 路径、同噪声 Native/Python 梯度对照、Adam preview commit/discard、Native Kineto 和 Nsight。没有运行长序列训练。未来 CUDA kernel 的真实速度、完整 Guard accept/reject/fallback 联合 parity、CUDA Graph capture、完整 GI bank 微基准和长期质量变化均尚未验证。

## 1. 版本、环境与产物

{{ENVIRONMENT}}

远端目录没有自己的 `.git`。`git rev-parse --show-toplevel` 返回 `/home/prmi`，其 branch 为 `master`，但 `rev-parse HEAD` 失败（空仓库），`status --short -- .` 为 `?? ./`。因此**远端没有可用 Git commit hash**；本地 HEAD 只作本地工作区记录，不能冒充远端版本。远端版本以 `source_hashes_before.json` 的逐文件 SHA-256、冻结源码以及二进制 SHA-256 为准。

{{PROVENANCE}}

最终逐文件核对 115 个清单文件，仅发现 `cfg/selector/fast.yaml` 的 `GradientConstraint.enable: 1 → 0`。该文件远端 mtime 为 `2026-10-14 00:11:36 +0800`，晚于最终基准结束（约 00:10）；本轮诊断脚本不写该文件。已纳入清单的 C++/CUDA/Python 生产源码哈希未变。保留 `source_snapshot/cfg/selector/fast.yaml` 和 `current_fast.yaml` 两版，不回退外部配置修改。所有本轮诊断使用已保存的 checkpoint 和显式参数，没有读取变化后的 fast.yaml。

配置口径：checkpoint 的 `Training.ultra_light_enabled=0`、`Selector.enabled=1`、ratios `[0.01,0.05,0.10,0.15]`、tau=1、Selector LR=0.001、render/GI/ratio weights=1/1/0.01、detach=0、LoD scale=0.5、Protection thresholds=100/80/4、stale=1000、budget fraction=0.35。该 checkpoint Guard disabled、DSD disabled、Replay enabled。初始 fast.yaml 也关闭 rotation，但启用 Guard；不能假设 checkpoint 曾用它训练。历史 UltrafastRotation 日志来自另外的运行目录。具体配置完整保存在诊断目录。

证据目录为 `tests/diagnostics/original_selector_cuda_audit_20261013/`（下文简称 D）。源码引用 `路径:行号` 均指**远端冻结源码**，可在 `D/source_snapshot/` 或 `D/complete_source_snapshot/` 查看；本地生产 checkout 可能属于并行工作，不作为远端版本依据。

## 2. 实际调用地图

| 文件与入口 | 调用者/条件 | 职责与训练边界 |
|---|---|---|
| `improvements/selector/gumbel_network.h:16,38,113,141` | Mapping/Replay | 网络、forwardHard、logitsFor、Gumbel ST |
| 同文件 `:53,65,80` | DSD、Guard、CD/Python 对照 | keep-logit、fixed-uniform ST、deterministic probability |
| `src/gaussian_mapper.cpp:1312,1389,1431` | 正常 Mapping | Full render/backward → 条件 LoD → 共享 optimizer |
| `src/gaussian_mapper.cpp:2673` | Mapping、trainJointReplay | `accumulateLodGradients`，完整 Mature/budget/ST/compact/TF/render/loss/backward |
| `src/gaussian_mapper.cpp:1778,1810` | enhanced optimizer step | 追加 Selector/TF 参数组；正常 step 或 Guard |
| `src/gaussian_mapper.cpp:1944` 与 `src/historical_gradient_render.cpp:24` | Guard baseline/candidate/history | 同 uniform 的历史 Full/LoD 目标；fresh autograd graph |
| `src/gaussian_mapper.cpp:2555,2600` | ConflictReplay 队列 | Full + LoD 联合梯度；共享 preview；**detach=false** |
| `src/change_distill.cpp:64,276` | 几何事件 | frozen teacher probability L1，独立临时 optimizer |
| `src/drift_stability.cpp:171` | DriftStability probe/distill | teacher probability + student keep-logit BCE；detached 状态 |
| `src/gaussian_mapper.cpp:2351` | stability/reference/probe 相关确定性路径 | `forwardProbability`，无 Gumbel；非普通每步网络调用 |
| `include/gaussian_protection.h`、`include/elastic_budget.h:38,46` | LoD/budget、Full evidence、append/prune | GPU 元数据分类；cached-uniform admission；软预算 |
| `src/protection_evidence.cu:53,63` | Full accepted update | 已有 CUDA clock/evidence kernel；LoD/Replay 不冒充 Full evidence |
| `src/gi_teacher.cpp:7`、`src/global_gi.cpp:149,216` | LoD teacher；bank 更新/周期刷新 | NoGrad GI label；贡献缓存、聚合、刷新 |
| `src/gaussian_renderer.cpp`、`include/gaussian_renderer.h` | Full 和 compact LoD render | compact ID 是 local；通过 global map 回写 supervision |
| `src/constrained_adam.cpp:6,156`、`src/gradient_constraint.cpp:60` | Guard/Replay | prepare/candidate/commit，状态与参数原子事务 |
| `src/gaussian_mapper.cpp:3822`、`python/selector.py:49,110` | 保存/离线评估 | Selector 参数 archive，加载和确定性 Bernoulli mask |
| `CMakeLists.txt:2,4,15,60` | 生产 build | CUDA 11.8/sm_86、LibTorch、已有 TF CUDA/cuBLAS 基础 |

源码搜索覆盖完整 `include/src/python/improvements/selector`。普通 `GumbelNetwork::forward()` 保留了两次独立 Gumbel 采样接口，但正常 Mapping 调用 `forwardHard()`，不使用第二次 soft draw。当前无 Selector Top-K；原有命名或论文中的 Top-K 不能作为现有实现证据。历史旧 Selector 的具体移除 commit 无法从远端空 Git 重建。

现有 CMake 有 Protection、GI、CD、DSD、TF、Guard 相关测试注册。本轮只构建 D 的独立 CMake targets；没有声称重新执行了全部历史测试。

## 3. 从 Mapping 到更新的执行链

正常非 rotation 的 enhanced iteration：更新/配置 GS LR、准备 GT → Full render（可采 GI contribution）→ GI cache/bank 维护 → Full photometric backward → LoD 下表 → 共享 optimizer/Guard → zero_grad → accepted Full evidence 更新及 topology 生命周期。Full loss 不直接连接 Selector，但 GS 的 Full 和 LoD 梯度在同一次共享更新累积。

`trainMappingIteration:1431` 只有 `enhanced && (!rotation || !full_step)` 才调用 LoD。普通 ratio index 为 `selector_step_ % ratios.size()`，每次一个 ratio；不是每个 Mapping 连跑四个网络。当前 rotation 是四个 LoD slot 后一个 Full slot，LoD 强制 detach=false，Full slot 不运行 Selector。启用/预算耗尽/空 Mature/事件路径均会改变实际调用次数。

下表以 N=全图数，M=Mature 数，K=最终 compact 数，J=TF subset 数。float32/boolean/long tensors 均在 CUDA；正常 contiguous `[rows,width]` stride 为 `[width,1]`，单列 view/广播例外见下表。计时、日志和 `.cpu()` 是 host 部分。

| 阶段 | 输入 → 输出、layout | 分配/kernel/同步 | autograd 与频率 |
|---|---|---|---|
| Mature preparation | birth/optimized/KF metadata `[N]` → bool mature `[N]` → int64 IDs `[M]` | 全 N 分类；`nonzero` 动态输出并等待 count；常驻 GPU 元数据 | NoGrad；每个 LoD |
| Protection population/plan | need、cached uniform `[N]` → probability/admitted `[N]`，host D/Pactual/targets | reduction、bool/float 临时量；D/Pactual stack 一次 D2H；并非 CPU 每点循环 | NoGrad；每个 LoD |
| Gaussian preparation | xyz `[N,3]`、raw q `[N,4]`、log scale `[N,3]`、raw opacity `[N,1]` → normalize/exp/sigmoid | 新激活 tensor；M<N 时 3 次 Mature index_select | 正常 detach 可配；当前=0 |
| 网络 | xyz/q/scale `[M,3/4/3]`、ratio `[1,1]` → 3×`[M,32]` + ratio `[1,32]` expanded stride `[0,1]` → cat `[M,128]` → logits `[M,2]` | 9 Linear、8 ReLU、cat；float32；参数见下一节 | M>0 的每个 LoD；跨 ratio 共享参数 |
| probability | logits `[M,2]` → softmax keep `[M]`，列 view stride=2 | 额外 softmax；复用同 logits | ratio loss 图 |
| Gumbel/ST | logits → noise/soft/onehot `[M,2]` → keep-ST `[M]` stride=2 | exponential/log/neg、div/softmax、row max、scatter、ST；无全局 sort/Top-K | soft 图保留；噪声/硬 ID 不导 |
| merge | ST `[M]` scatter 到 `[N]`，admitted float `[N]` → selector_mask `[N]` | index_copy + float masks；all-Mature 时不 scatter | 仅 Mature 有 Selector 梯度 |
| GI teacher/loss | combined GI `[N]`、valid/mature bool → supervised IDs、label `[N]` → scalar L1 | nonzero、kthvalue、index_put；masked indexing；同步和全量工作见 GI 节 | target NoGrad；L1 向 ST 回传 |
| compact | `nonzero(mask.detach()>0)` → int64 global IDs `[K]` | bool temp、动态 count 同步；升序 global ID 顺序 | ID 不导 |
| gather | 6 GS tensors → xyz/scale/q/opacity/SH `[K,...]`；compact ST `[K]` | index_select；opacity `alpha*ST` 新 tensor | 原属性 gather + opacity ST 图保留 |
| TF | selected mature∩inside-AABB → IDs `[J]` → transform `[J,3/3/4]` → index_copy `[K,...]` | 第二次非零索引及 gather/copy；TF 有独立 CUDA/torch backend | TF 和 GS 梯度；每个可用 TF LoD |
| render/loss | K 个 compact Gaussian → image `[3,Hlod,Wlod]` → photo + ratio + GI + TF reg | rasterizer、image cache、SSIM；online LoD scale0.5、post-SLAM scale1 | 一次 LoD backward；M=0/K=0 有分支 |
| optimizer/Guard | 累积 GS/Selector/TF grads → 新参数和 Adam state | 共享 optimizer；Guard 额外历史 forward/backward 与 host decisions | 正常 step 与特殊事务分开 |

stride 表是源码产生 layout 的规格，非生产所有张量实测 stride trace。参数 stride 由 checkpoint 实际记录；Kineto 记录真实 operator input shapes。SH rest `[N,15,3]`，DC `[N,1,3]`，激活 opacity `[N,1]`；SH 不进入 Selector。

## 4. 精确网络结构与参数

令 raw quaternion 为 q，log scale 为 ℓ，则网络输入：`x_i=XYZ_i`，`qhat_i=q_i/max(||q_i||2,epsilon)`，`s_i=exp(ℓ_i)`，`r=global target_ratio`。XYZ 没有 AABB normalization、positional encoding、邻域或 GI 编码。

对于 b∈{pos,rotation,scale,ratio}：

`h_b1 = ReLU(v_b W_b1^T + b_b1)`

`h_b2 = ReLU(h_b1 W_b2^T + b_b2)`

`z_i = concat(h_pos2,i,h_rotation2,i,h_scale2,i,h_ratio2) W_out^T + b_out ∈ R²`

`p_keep,i = softmax(z_i)[1]`。`time_emd` 名字表示 ratio 分支，不是时间坐标。每点独立；参数全图/所有 ratios 共享。无 normalization layer、skip connection 或全局 attention。opacity、SH、GI、age、optimized_count、unique KF evidence 都不是网络输入；Protection/GI 在 caller 外围生效。

| 分支 | 两层及激活 | 参数量 |
|---|---|---:|
| position | 3→32 ReLU→32 ReLU | 1,184 |
| rotation | 4→32 ReLU→32 ReLU | 1,216 |
| scale | 3→32 ReLU→32 ReLU | 1,184 |
| ratio/time | 1→32 ReLU→32 ReLU | 1,120 |
| output | 128→2 Linear | 258 |
| 合计 | 9 Linear、8 ReLU | **4,962** |

{{PARAMETERS}}

所有参数为 CUDA float32 contiguous、普通 Linear row-major `[out,in]`；上表是实测 checkpoint tensor。C++ float-ratio 路径只计算一次 ratio embedding 后 expand；Python eval 和 CD 的 tensor-ratio 路径传 `[M,1]`，会重复同值 embedding，这是可优化但需独立 parity 的差异。

## 5. Gumbel、Mask、ratio 与预算

正常 `forwardHard` 的确切规则：

`E_ic ~ Exp(1); g_ic = -log(E_ic)`（ATen CUDA 默认 generator）

`y_i = softmax((z_i+g_i)/tau); c_i=argmax_{c∈{0,1}} y_ic`

`h_i=onehot(c_i); st_i=h_i-y_i.detach()+y_i; keep_i=st_i[1]`

forward ST 数值为 hard 0/1；backward 的 Jacobian 是 soft relaxation，hard max/scatter 不提供 score 梯度。没有跨 Gaussian 的排序、Top-K、排名 buffer，也没有 `K=floor(r*M)` 用于强制选点。Gumbel 最大类的选择概率为 softmax(z)，与 tau 无关；tau 影响 soft 梯度。普通 ratio loss 使用同 logits 的无噪声 softmax，不是 noisy soft count。

普通 CUDA sampling 消费默认 device RNG，诊断固定 `torch::manual_seed(123)` 的 parity 单独保存；benchmark 连续消费随机流，不为每次训练重置相同噪声。Protection uniform 来自常驻元数据，append 时补充、prune 同步裁剪，跨 ratio 复用；它与 Gumbel RNG 是不同用途。生产本次运行的最初全局 seed 无法仅从 checkpoint 确认，报告不补造。

Guard 的 `forwardHardWithUniform` 是另一条明确定义的路径：`d=z1-z0`，clamped u 的 logistic threshold `t=log(u)-log(1-u)`，`soft=sigmoid((d-t)/tau)`，`hard=(original u < sigmoid(d))`，`ST=hard-soft.detach()+soft`。同一次 Guard baseline/candidate 共用 u，且 Scope 恢复 CPU/CUDA RNG；不能把普通两类 Gumbel backward 偷换为这条 relaxation，即使 hard sampling law 相同。

Python evaluation（`selector.py:109`）NoGrad，计算 deterministic p_keep（实际调用 tau=1），每次新建 seed0 CUDA generator 产生 `[N]` uniform，以 `u[eligible]<p_keep` 选择 Mature，再合并 Protection。对固定输入可重复，**仍是 Bernoulli，不是 deterministic Top-K**。函数 temperature 参数没有改变这一实际 score 调用。现有 C++ viewer/evaluation 主要走 Full render；本轮 `cpp_inference` 是以真实 C++ 网络复现 Python selection law 的基准，不能称为已有生产 C++ LoD evaluation。

Protection 精确公式（当前配置）：

`maturity_i=min(clamp(age/100,0,1),clamp(optimized_count/80,0,1),clamp(unique_KF/4,0,1))`

`Mature_i=(age>=100 && optimized_count>=80 && unique_KF>=4); need_i=1-maturity_i`

`T=r*N; D=sum(need_i*!Mature_i); Bp=min(D,0.35*T)`

`pi_i=need_i*!Mature_i * Bp/max(D,1e-12)`

`admitted_i=!Mature_i && cached_uniform_i<pi_i; Pactual=sum(admitted)`

`Tmature=clamp(T-Pactual,0,M); mature_ratio=(M>0 ? Tmature/M : 0)`

`mask_i=admitted_i+(1-admitted_i)*selector_hard_mask_i`

集合关系：network eligible=Mature；protected population=非 Mature；Protection 实际保留是其中 admitted subset。所有非 Mature 并不无条件保留。WARMUP、UNDER_OBSERVED、STALE、MATURE 均存在；stale threshold=1000，STALE 是不满足成熟 evidence 的老点，不能把全部老 Mature 判为 STALE。Protection disabled 时 caller eligible=全部 N；classify 仍可执行，但其 need 乘非 Mature 为零。

最终 `K=Pactual+sum(hard_keep over Mature)`，是采样结果，没有最小 K 保证；M=0/K=0 由 caller 分支处理，T 是浮点不先 rounding。GI teacher 的 `target_k=floor(mature_ratio*M)` 只是 teacher metadata，严格分位标签会因 ties 更少，并不锁定 renderer K。

固定 evidence/uniform 下 Protection admissions 对 ratio nested；**整个 Selector 不保证 ratio-monotonic**。网络本身依赖 ratio，普通训练每次重新采 Gumbel；Python 固定同 uniform 也无法保证不受约束的 p(r) 单调。

可复用的范围：同参数版本、同 geometry/topology/Mature set 内，pos/rotation/scale embeddings 与输出的 geometry 部分可跨 ratios 共用；ratio embedding/output contribution 必须重算。可将 output 按四个32宽 block 分拆，避免实体 `[M,128] cat`，但改变浮点累加顺序会改变近阈值 hard mask，不能直接宣称选择 bitwise 不变。普通 Mapping 更新参数/GS 后缓存即失效，跨 iteration 复用不成立；多个 Guard candidates 也改变参数，不能共用 baseline logits。fresh graphs 与正确累计参数梯度是复用条件。

## 6. GI、Protection 和共享准备的成本归属

**Selector core cost**：network/logits、Gumbel/probability/ST及其 VJP，参数 backward、Selector optimizer。**selection/index cost**：Mature scatter、merge、compact IDs和 compact mask；六类 GS gather 的共同调用准备单列归属，不能重复累加。**GI-associated cost**：teacher、GI loss，以及另列的 bank/snapshot/contribution renders/refresh。**Protection-associated cost**：evidence 生命周期、分类、need/budget/admission/readback。**Shared Mapping preparation cost**：GS activation、GT/image cache、camera、TF和 renderer 自身准备。

GI 不直接进入网络。Full render 根据当前 KF/RecentWindow 条件收集 frame importance 与 count；`refreshRecentWindowContribution` 更新近期贡献，并触发 GlobalGI bank batch/snapshot 维护。`global_gi.cpp:149` 流式成员增删缓存每 view contribution；`aggregate:79` 聚合贡献与有效标记；append/prune 要重映射所有 cached view tensors。`refreshIfDue:216` 默认每1000 Mapping iterations 对 bank members 重新 contribution render并聚合。配置 view_ratio=0.3、coverage anchor=0.8、sampling_seed=0，max_views=0；bank/snapshot 的 CPU对象遍历与 GPU贡献扫描是独立成本。

GI teacher（`gi_teacher.cpp:7`）在 NoGrad 下只监督 observed Mature 非 protected：`j=floor((1-r_mature)*(supervised_n-1))+1`，`threshold=kthvalue(scores,j)`，`label=(score>threshold)`；不是 sort 全数组，不是 Selector Top-K。normal GI L1/smooth-L1 作用于 ST keep，向 Selector及未detach GS xyz/q/scale回传；target 无 gradient。valid-count日志、`recordGuidance:279` 的 `.sum().item()`、supervised nonzero/masked index、周期刷新和 topology cache都会同步或扫描。

本轮独立 teacher 使用**真实单帧 Native frame_importance**，不是完整生产多 KF bank 的已保存快照。N219969、r0.05 的 teacher event mean1.207 ms/wall1.433 ms，只代表这份 GI fixture；不能推断 bank 全生命周期成本。历史日志 GI stage 达17–45ms说明它可能控制完整 Mapping，但没有本轮完整 GI stage profiler，不能为 GI 优化给可靠新倍数。

Protection 的 age/evidence 元数据常驻 GPU，已有 clock/evidence CUDA kernel；不需要每步把 CPU mask 搬上 GPU。分类/预算还是 ATen full-N 操作，`nonzero` + D/Pactual readback 是 dispatch/sync 小规模瓶颈。独立 protection_budget 用预先计算的 population，因而**不包含 classify/Mature nonzero**；最终 Nsight protection stage包括它们。两者不能互换。

## 7. Loss 公式、真实梯度和计算图

正常 LoD：

`L_photo=(1-lambda_dssim)*L1_or_smoothL1(I_selected,GTlod)+lambda_dssim*(1-fast_ssim(I_selected,GTlod))`

`L_ratio=abs(sum_{Mature} softmax(z_i)[1]-Tmature)/max(1,T)`

`L_GI=L1_or_smoothL1(ST_keep[supervised],label[supervised])`

`L_lod=w_render*L_photo+w_ratio*L_ratio+w_GI*w_ratio_aware*L_GI+L_TFreg`

checkpoint w_render=1、w_ratio=.01、w_GI=1；ratio-aware disabled，四个权重均1。TF regularization是已有 spatial/ratio smooth和ratio L1，**不是额外 Selector regularization**。普通 loss 没有凭空加入 entropy/sparsity 项。诊断真实 Native Renderer 使用 lambda_dssim=.2、普通 L1，即 `.8*L1+.2*(1-fast_ssim)`；生产 lambda 仍由运行 optimization config 决定。

`IDs=nonzero(mask.detach()>0)`不可导，但 `alpha_compact=alpha[IDs]*mask[IDs]` 可导。photo 对 Selector的梯度来自这个乘法；只有选中的 Mature rows承接 opacity photometric VJP，ratio loss覆盖所有 Mature，GI覆盖 observed supervised rows。未选点不应从 compact photometric 路径得到伪梯度。TF使用 detached的 subset decision，但变换本身保留图，并通过同一个 rendered photo给GS/TF梯度；不需要也不允许关闭 Selector autograd。

{{GRADIENTS}}

表中 “18/18” 表示全部18个参数张量有非零 norm，不代表每个元素非零；ReLU 造成稀疏元素梯度。`gradient_paths.csv`逐 tensor记录 norm、defined、非零比例、finite，`gradient_losses.csv`记录 N/M/K/visible/J/loss。所有已产生梯度有限。

各项单独建立 fresh graph，正常 Full和LoD各backward一次，不复用旧图，不用retain_graph。Guard history用`autograd::grad` fresh graph、不写 normal `.grad`；raw candidate rejected之后才计算history gradient。Replay Full和LoD是两张图的累计，随后一次共享事务。

真实图测试加载当前 `libgaussian_mapper.so/libcuda_rasterizer.so`，只更新私有诊断对象；GT 来自真实 TUM 图像、320×240、PIL bilinear resize。该 resize 并未证明与生产 resized_image_cache 数值完全相同。TF 使用 checkpoint 的同一组权重、AABB 和维度，明确选择 legacy LibTorch backend 验证联合计算图，不代表当前 TF CUDA backend 的数值 parity。GI 是单 view fixture，梯度测试的 observed mask 设为全真以覆盖监督路径，未复现生产多 KF validMask；独立 teacher benchmark 用 frame importance 构造 observed mask。CD/DSD 测试将 teacher target 轻微扰动以确认梯度，不是完整几何事件生命周期。

同噪声结构 oracle N257/K38：Native/Python hard mask exact，18 个参数和 x/q/s/alpha 的梯度全部通过 `atol=2e-6,rtol=2e-4`。同构 Python 图检查得到 70 个 autograd nodes、33 个 saved-tensor references、unique saved storage 364,968 bytes；没有将它标为 Native 内部 graph 节点直接计数。这是 compact opacity VJP + ratio 结构验证，**不是完整 Renderer 数值 parity，也不是 CUDA 候选后端 parity**。记录见 `gradient_parity_and_graph.json`。不把 N257 的图保存量线性推成生产总峰值。

## 8. 真实输入规模与工作负载

checkpoint：N=220084，M=219969；WARMUP0、UNDER_OBSERVED88、STALE27、非 Mature115，Protection demand43.425，所有四ratio的实际admission33。网络M与最终selected K不是同一个数。

{{CHECKPOINT_COUNTS}}

上述为Python seed0评估，一次普通Gumbel训练r=.05的gradient fixture K8569（visible8219，TF J8535）；Guard shared-uniform fixture K8672。差别符合不同采样路径，不能要求它们K相同。

历史日志保留13个path、源line，共36,228条LoD rows。下面分别列正常checkpoint日志和UltrafastRotation来源；重复GI基准日志和三scene轮换日志不作为单一在线统计分布。全来源合并量只用于规模覆盖，不代表现有Mapping概率分布。每条原始记录见`workload_rows.json`，每个source的全部字段分位见`workload_quantiles.json`。

{{WORKLOADS}}

历史M只接受**同iteration的Protection STATE**：四ratios可确认M的样本数仅4/2/1/5，不足以给可信Mature全程分布。不得用Ntotal或K倒推网络M。TF字段和真实K有较完整统计。日志没有普通每step完整forward/Guard repeat counts；每条LoD日志证明一次caller，正常网络M>0时一次forward；cycle频率由代码条件决定。Guard repeat数、Replay occurrence不能从这些LoD rows盲补。基准500个K样本/ratio单独保存，属于固定checkpoint随机采样分布。

大小1k–200k是这份真实checkpoint的prefix，未通过重复点扩成100k。额外52181是历史Ntotal median覆盖点，108688是normal iter1000的observed M，219969是checkpoint M；独立阶段fixture仍取checkpoint prefix，**不是对应时间点的原始Gaussian集合**。caller benchmark使用全N population的真实Mature subset，N220084时才与完整checkpoint一致。

## 9. Native C++ 基准口径与结果

RTX3090/CUDA11.8/LibTorch2.3.1，C++线程数1；100 warmup +500 measured，当前default CUDA stream；计时前device sync，event首尾在同stream，尾event synchronize；load和input准备排除。无CUDA Graph、多stream或CUDA_LAUNCH_BLOCKING。wall含host dispatch/等待；event是stream首尾跨度，**含GPU空闲、CPU提交滞后和同步等待，不是kernel active time**。

三mode：`caller_train`直接调用原生产forwardHard，最终训练基线；`cpp_inference`C++网络+Python seed0规则，无grad；`staged_train`拆分同数学流程用于stage观察。私有Selector更新在每次timed region后恢复checkpoint参数（18 D2D），moments继续推进，GS不step；其目的保持checkpoint分布，不是连续优化质量实验。

`native_benchmark.csv` 1,120行，`native_samples.csv` 48,000样本=8 sizes×4ratios×3modes×500。最终性能取无内部stage event的caller_train/cpp_inference。早期staged CSV在tau1省略了一次`/1.0`算子，数值同但graph/kernel不同；最终Nsight/Kineto已按显式division重跑，**staged CSV不可作最终kernel计数或与最终stage总耗时比较**。所有ratio完整mean/p50/p95见CSV/JSON。

{{BASELINE}}

补充独立阶段共10 sizes×4ratios×7stages=280 cases，每case100+500；如下均r=.05。inference_network包含无grad forwardProbability；training_forward含forwardHard及keep probability；training_backward在计时前构图，返回Selector参数和XYZ梯度，q/scale在这个独立fixture不requires_grad（完整caller和gradient fixture有），因此不能把该backward视为生产全部input VJP。feature_activations是exp/normalize外加一个诊断cat，不是caller原样feature stage。selection_mask_index使用从checkpoint概率重建的双logit，NoGrad单独测Gumbel/ST/nonzero，不包含Protection merge、完整GS gather；teacher用global ratio，不是完整budget-adjustedcaller。独立样本分配、CPU节奏与调用上下文不同，**不可简单相加重建caller总时间**。

{{ISOLATED}}

{{OPTIMIZER}}

optimizer结果仅包含18个Selector参数和固定1e-4 grads；preview包含candidate构造。没有GS/TF/full Guard evaluation的耗时。commit/discard validity已通过（discard不改params/state，commit各state step=1），不代表完整Guard策略的所有分支已动态验证。

## 10. Native Nsight、算子、分配与同步

最终Nsight捕获100 warmup后的10个staged iteration，N220084/M219969/r.05；没有Renderer/TF/GI。Native Kineto单独捕获相同10iteration并记录shapes/memory，二者kernel时间不是同一次测量。每stage有diagnostic event/NVTX，插桩会影响调度；以下active是Nsight实际kernel执行时间。

{{NSIGHT}}

无Top-K/sortkernel；row_argmax只是每行2类。network 19kernel=9GEMM+8ReLU+cat+ratio full填充。backward88包括MLP、ST/ratio、opacity及活跃GS输入VJP，并非全是纯MLP。optimizer144kernel对应18张小参数的ATen Adam操作；零梯度清理本轮没有GPUkernel。

{{SYNCHRONIZATION}}

3 D2H/iteration来自预算两scalar合包及Mature/compactnonzero count；2 D2D/iteration是timed stages复制，另有18 D2D/iteration的checkpoint恢复在计时外，已单列排除。没捕获H2D，不能据此推断完整Mapping不存在H2D。`cudaMemcpyAsync`可能因host pageable目标阻塞；nonzero的CPU时间会包含等待前序network完成，所以不能把compact host3.979ms当作compactkernel成本。

CPU launch gap均值/最大值是相邻launch API的间隔，可能包含同步、调度、依赖和插桩，不是可消除的纯PyTorch占比。各CPU operator inclusive时间存在嵌套，禁止直接相加。

{{OPERATORS}}

完整operator counts、每input shape、CPU inclusive、kernel数和active时间在`operator_breakdown.json`。本Native Kineto的GPU External id与CPU RecordFunction ID不一致；解析器通过runtime correlation及同thread最内层CPU scope归属，不直接把External id数字相等当对应关系，避免把GEMM误标为transpose。

{{TOP_KERNELS}}

{{MEMORY}}

allocator requests不是cudaMalloc次数，累计申请bytes不是同时占用；包括cache复用、profiler保留与fixture。saved graph要保留h/relu和cat等；M219969的`[M,128]cat`单张112.624MB（107.407MiB），每个`[M,32]`28.156MB，两类logits1.760MB。C++ratioexpand自身stride0不复制，但cat将其物化。只消掉cat不能证明整个训练峰值节省112MB：backward如何存储/重算决定最终峰值。

## 11. 未来 CUDA backward 的完整规格

边界推荐是可导network + ST outputs，不把hard IDs当有梯度Tensor。自定义autograd Function必须让所有18个Parameter成为明确Tensor inputs；返回`grad_x,grad_qhat,grad_scale`和各weight/bias梯度。normal float ratio无需grad，但tensor-ratio接口若requires_grad必须支持`grad_ratio`并正确处理标量广播sum；否则限制要显式且caller已证明适用。opacity/SH/GI不是network输入，其梯度仍由caller/renderer/index-select图产生。若融合raw activation/compact gather，则对应raw quaternion、log_scale、raw opacity和其他属性梯度也必须由新Function返回，不能截断。

对每行softmax y与上游v：

`dL/dz_c=(y_c/tau)*(v_c-sum_k(v_k*y_k))`。正常keep-only上游扩展到两类v；Gumbel noise是常量，hard onehot/argmax和IDs不导。ratio probability另走无噪声softmax Jacobian，`v_keep=w_ratio*sign(sum(p)-Tmature)/max(1,T)`；零差值对应abs约定须一致。GI普通L1的上游是`w_GI*w_ratio_aware*sign(ST-label)/supervised_n`，smoothL1配置走实际piecewise derivative；target和Protection预算均不导。

photo上游从Renderer对compact opacity的梯度产生：对选中非admitted Mature i，`dL/dST_i = alpha_i * dL/dalpha_compact_i`；未选中点此项为0。GS opacity梯度是`ST_i*dL/dalpha_compact_i`再接sigmoid；GS position/scale/q还接直接renderer、TF与Selector输入路径，需与Full gradient累计。

Guard uniform ST：`dL/dd=v*soft*(1-soft)/tau`，再`dL/dz1=+dL/dd,dL/dz0=-dL/dd`；hard使用原u，而threshold用clamp u，不能合并后改变端点语义。CD probability需tau对应softmaxJacobian；DSD BCE logits的`dL/dd=(sigmoid(d)-target)/batch_numel`再接`(z1-z0)/temperature`。

Linear用row batch：`dW=dY^T X; db=sum_rows(dY); dX=dY W`；ReLU `dA=dH*(A>0)`，0点导数保持ATen；ratio branchexpanded后的梯度先sum所有M再反传一次time MLP。输出层向四个branch拆分梯度，参数reduction必须覆盖全M而非K。raw activations：`dℓ=dscale*exp(ℓ)`；normal quaternion在norm>eps时`dq=(I-qhat*qhat^T)*dqhat/norm`，小norm采用当前normalize clamp的精确分支，禁止换成另一epsilon或归一化规则。

保存/重算：至少保留网络输入、参数版本、ReLU结果或bitmasks、soft y/无噪声p、Mature/compact映射；可重算线性activation以省显存，代价是重新网络计算。比特mask仅代替ReLUgate不能代替计算dW所需的h；也可重算h。Gumbel可保存noise，或保存forward取得的Philox seed/offset与确切counter consumption；backward不得再消费默认RNG。重算不允许用“再manual_seed一次”破坏后续随机流。PyTorchexponential的块/向量化draw与新uniform变换不保证同seedbitwise一致；同noiseparity、RNGconsumptionparity、统计parity应分别验。

参数梯度建议block partial reductions +第二次确定reduction，或cuBLAS SGEMM，避免对每点×4962参数大量atomicAdd。原图的分配/保存要求、undefined grad、空M、非contiguous输入和non-defaultstream都必须在Function契约中覆盖。可先限制float32并显式验证，不能无声cast改变生产精度。

## 12. Optimizer、Guard、Replay、CD、DSD、checkpoint

正常shared optimizer原前6groups为XYZ、SHdc、SHrest、opacity、logscale、rotation。Selector追加group6，TF MLP/grid随后；densify/prune依赖前6顺序，不能重排。`GaussianModel::trainingSetup:504`的Adam defaults eps=1e-15，betas=(.9,.999)，weight_decay0、amsgrad=false。Selectorgroup只改LR=.001，因此继承这个eps；standalone selector_optimizer_为Adam影子对象，defaults eps1e-8，**普通更新不调用该shadow step**。本轮optimizer基准按shared eps1e-15。

没有单独Selector衰减scheduler；正常group LR取selector_learning_rate，GS/TF另有现成scheduler/clock。共享对象是SparseGaussianAdam，但普通enhanced route经`tfCudaAdamMixedStep`调用denseAdam行为，TF CUDAgroups可特殊处理；可见性masked的`step(visible,N)`重载不是此普通Selector路径。未来不得将4962元素的Selector参数交给假设每Gaussian存储的sparsevisibility更新。

Adam每Parameter维护step、exp_avg、exp_avg_sq（amsgrad时max），undefinedgrad跳过，不等于零gradstep。zero_grad(true)释放grad引用；normalLoD在进入时用shadowzero_grad清相同参数对象，Full GS梯度不能误清。Adam公式顺序需匹配LibTorch：m/v更新、bias correction、sqrt(v)/sqrt(bc2)+eps、addcdiv原参数。eps、FMA、sqrt/div顺序与floatrounding影响candidate。

Guard：共享prepare建立proposedmoments/step/delta，不写live参数状态；历史Full/LoD baseline和候选forward以共同uniform测MSE。raw验证失败才freshhistory梯度，Gram/statistics合包CPU读出，dualprojection得到实际FP32direction，再验证alpha1/.5/.25（最多max_candidate_checks=3）。cancelfallback启用时用原delta*.2；代码检查finite/changed/evaluationfinite后可提交fallback，不能假设所有fallback都通过nonregression约束。最后commit一次live参数和proposedAdamstate；discard全部不动。SHRINK/PROJECT后state仍是原Adam准备态，不能把投影方向当新gradient重新计算moment。保留所有clock/rejection/TFwarmup判定；不能先原地step再rollback。

Guard每轮历史有V views、R ratios时objectives=V*(1+R)，其中LoDnetwork至少V*R次baseline，并按candidatechecks再重复；raw拒绝才再V*R次historygrad。实际数量取决于检查/skip/cancel/fallback，不能写固定每Mapping3倍。默认sourceview1/ratio1，但没有本轮完整Guardrun实测次数。GPUkernel优化不会消除历史rasterization。

Replay当前`trainJointReplay:2600`显式detach=false：Full+LoD同时更新GS/Selector/TF，以sharedAdampreview通过历史约束提交；normalLRschedule复用，不推进normal Mapping/Selector/TF clocks，只记录Replay专用计数，逐ParameterAdamstate按实际提交更新。不能沿用旧版“Replay detach=true”。CD和DSD则来自detachedGaussianAttrState与frozenteachers，临时独立Adam，只更新相应辅助模块；不得拿normalsharedmoments替代其临时state。CDselectorL1Probability在geometrychange事件按affected/unaffectedstrata/batch、多个ratios/epochs；DSDstudentkeep-logitBCE，probe/reference/anchors有独立生命周期和RNGScope；本轮只验证这些gradientAPI。

checkpoint保存`selector.pt`注册参数、selector_config.yaml、selector_metadata.ply/protectionstate及transform/clockJSON。**没保存Selector正常sharedoptimizerAdamstate或完整训练RNG**；`transform_optimizer.pt`保存独立shadow，不是正常sharedTFgroupstate。Pythonloader可加载archive并重建同名权重/metadata进行eval，但并非精确resume训练。未来CUDA后端应保留参数名/layout与现有eval兼容；训练state若新增需明确版本，不能假称现有checkpoint足以1/10/100stepresumeparity。

## 13. 可复用 CUDA 基础

生产已有 Protection clock/evidence CUDA、rasterizer CUB scan/sort、TF cuBLAS Linear 及 warp MLP、shared-memory reduction、atomicAdd grid grad、TF multi-tensor Adam prepare/commit。远端 `cuda_rasterizer/rasterizer_impl.cu:267,314,433,470` 可核查 CUB InclusiveSum/SortPairs；补充冻结文件保存在 `D/complete_source_snapshot/cuda_rasterizer_impl.cu`（未纳入开始时的 115 文件清单，不声称其前后哈希核对）。TF neural 使用 current BLAS handle/current stream、CUDAGuard、TORCH_CUDABLAS_CHECK；cuBLAS math/pointer mode 保存恢复，TF target 有 `--fmad=false`。Protection 与 TF launch 后用 C10_CUDA_KERNEL_LAUNCH_CHECK。

可复用stream/device/errorhandling、parameterpacking与transactionAPI设计，不能直接把TF层结构或optimizergroups硬套Selector。future`selector_cuda.h/.cu`可负责forward/backwardFunction；`selector_cuda_optimizer.cu`只有测得必要才增加。本轮没有创建这些生产文件。rasterizerCUB代码能作compactscan参考，但替换nonzero须保留升序IDs与动态K接口；不能使用排序网络为不存在的Top-K增加工作。

所有futurelaunch都需`CUDAGuard(input.device())`、`getCurrentCUDAStream(device)`；event也用该stream。临时tensor由LibTorchallocator持有，跨stream使用record/event保证生命周期；不能写死stream0。错误检查不能增加每kerneldevice-wide synchronize。cuBLASprecision/FMA/TF32设置属于parity契约，不能仅由FLOPs假定一样。

## 14. 候选方案与优先级

| 推荐顺序 | 方案与具体范围 | 依据/优点 | 复杂度、风险与保留项 |
|---|---|---|---|
| 1 | **networkforward + custombackward**：试32-widewarp/Gaussian分支融合；同时保留cuBLAS+fusedepilogue参照 | 大M主要active在网络及backward；可省多层中间读写、cat、ReLUlaunch | 复杂；weight/biasreduction与浮点顺序风险。生产总收益必须带Renderer/TF/GI复测 |
| 2 | **callerpreparation/selection**：classify合并、ST/probability/merge融合、共享一次compactID、gather+opacityVJP | 小M固定开销、nonzero/readback和全Ntemp；实现可分阶段回退 | 中等；动态K仍需count交host供现renderer，不能承诺零同步；必须保留ST |
| 3 | **18tensorfusedAdam事务**，接现有preview/commit | 4962params却144kernels；单独Adamwall~.955ms，preview~1.7ms | 中高；undefinedgrad/step/eps/candidateprojection/statecommit。大Mactive仅约4%，收益受限 |
| 独立项目 | GIbank/snapshot/teacher/cache优化 | 历史完整MappingGI占比高 | 不能计入SelectorMLP加速；需同配置完整stageprofiler，变更授权另行确认 |

fusedMLP候选比较：one-thread/Gaussian每thread持有多个32宽activation，registerpressure和重复weightloads大，不优先；one-warp/Gaussian自然映射32channels，用shuffle/sharedweighttiles跨多个warp复用，可融合3→32→32与最后2输出；多个warp/Gaussian可加大矩阵并行但此width32跨warp同步成本可能更高，不推荐直接作为第一版。大M也可能cuBLAS的矩阵tile复用胜过warp；**当前只有LibTorch/cuBLAS基线，没有候选kernel，不能宣布warp必胜**。推荐同N/ratios在FP32parity约束下比较warpforward+cuBLASweightgrad与cuBLASsurroundingfusion，再决定dispatch阈值；不预先添加未测阈值或复杂配置。

“fusedGumbel+Top-K”在本代码改为fusedrow-wiseGumbel/probability/ST：无需全局排序、smallK/largeK双Top-K策略。Gumbel3kernel仅12µs，单独重写优先级低；更合理是与ST/merge共同减少中间量和dispatch。严格RNGparity困难，应先用传入固定noise证明数学与梯度，再考虑generator集成。

compact优化可避免全量2列onehot和部分float/bool中间量，但仍要让所有Mature参与ratio/GIloss；不能只对K点计算网络/backward。可把photoopacityVJPscatter回Maturegradbuffer，而ratio/GIVJP另累积；保留全图Protectionset和globalID稳定顺序。跨ratio共享geometryembedding只适用于同参数版本的multi-ratioevaluation/CDteacher等，普通每步一个ratio并无四倍复用空间。

阶段完成标准：第一阶段fixed-noiseforward/backwardparity+Native无profilerbenchmark；第二阶段真caller/Renderer/TFgraphparity+动态topology/streamtests；第三阶段Adam/Guard/Replay事务parity；最后才做同配置短在线Mapping验证和长统计质量实验。本轮结束于设计，没有执行任何生产重写。

## 15. CUDA Graph 可行性

全路径存在：N/M随append/prune变，Mature和compactnonzero输出dynamicshape，budgetD/Pactual.cpu()、日志.item()、TFsubsetdynamicJ、Guardhostdecisions/preview以及candidatefreshgraph。Graphcapture内部不允许这些host同步/动态allocation确定shape操作；同ratio也因RNG使K变化。稳定临时地址和stream是额外要求。固定最大Npadding会额外计算且改变RNG消费，不建议为Graph强造固定maxshape。

较窄的network-onlyfixedMscope有研究价值：预先分配输入/outputs、ratio用devicebuffer、parameters地址稳定；topology/MatureIDs改变时重新构建或切换有限缓存，Gumbel若包含须用capture兼容generator/counter和正确replay随机性。Guard不同candidatevalues可填stablebuffer，但caller同步仍在Graph外；不会自动消除renderer/compact等待。**本轮未实际capture，结论为静态可行范围，不是已验证加速方案**。

## 16. 数值一致性验收设计（下一阶段）

| 范围 | 必须验的输入/输出 | 判定与特殊边界 |
|---|---|---|
| network/features | raw/activatedx/q/s、9层outputs、2logits、ratio scalar/tensor/expand | FP32初始atol2e-6/rtol2e-4仅作本轮oracle参考；先看maxerror和thresholdmargin，不能由allclose推maskexact |
| fixednoise | 用同[M,2]Gumbel分别跑参考/候选 | soft/ST tolerance +hard IDs严格相等；ties按ATenrowmax、NaN/Inf传播，不新增未授权guard |
| RNG | seed/state/offset、forward和backward后下一draw、GuardRNG恢复 | bitwise要求另验；若新RNG仅统计一致，应明示不同ID来自RNG而非算法 |
| budgets/masks | r=0,.01,.05,.1,.15,1；M=0/N；Protectionall/none | D/Bp/Pactual/Tmature、strict<、strict>、clamps、IDs顺序、K=0/1/N（固定mask构造） |
| GI | observedall/none、等GI、ties、teacherpercentile | label/targetk/supervisedset一致；没有SelectorTop-K可供验 |
| gradient | 18params、x/q/s、rawGS6fields、TFallparams | 分photo/ratio/GI/combined/Full/detach，selectedopacityST必须存在；复用本轮NativeRendererfixture |
| backwardstorage | recompute/savedcontext、retain-free多图累计 | 未selectedphoto梯度为0；ratio/GI全M，ratioexpandgradsum；不二次消费RNG |
| trainingsteps | 同inputs/noise序列1/10/100连续steps | params、moments、step、loss/IDs；本轮未执行候选连续steps，恢复checkpoint不能替代 |
| Adam | missinggrad、zeroLR、eps1e-15/1e-8、state未初始化/已有 | prepare不改live，discard完全不动，commitonce；biascorrection和操作顺序 |
| Guard | accept、rawreject、projection/shrink、cancel、finitefallback、no-op | 候选sameuniform/freshgraph，params/state/clocks/historyMSE与参考一致 |
| Replay/CD/DSD | Replayjointnodetach/不推进normalclock；CDprobL1；DSDBCE | 临时optimizer/teacher/affectedanchors/RNG/epoch语义保持；gradientAPI通过不等于event通过 |
| lifecycle/checkpoint | append/prune、Mature转移、save/load、旧archive | parametername/layout、Protectionuniformalignment；不能要求未保存state的精确resume |
| streams/layout | CUDA非默认stream、stride2/ST与stride0ratio、empty/非contiguous | 正确stream/device/lifetime；显式支持或拒绝layout；CUDAerrorcheck无globalSync |
| statisticalbehavior | 真实RNG多seed、ratio各自selected分布与训练质量 | mean/variance、ratio/GI/photo趋势及Mappingwall；不将nestedselection作为现有规格 |

本轮已执行的是Native原实现gradient/preview/固定noise结构对照，其结果不能代替以上futurecustombackendtests。单独实现的mathematical同构sigmoid可能与2类softmax在浮点/near-tie有差异；约束优先级应明确：同noisehard-IDexact、gradtolerance、RNGcounterpreserve，任何放宽另有证据。

## 17. 收益预测与真正的 Mapping 上限

只优化networkforward会保留Gumbel、ST、Protection、GIteacher、dynamicnonzero、GSgather、backward、optimizer、Renderer/TF。network+selection能省cat/onehot/merge中间量、多个小kernel，但**没有全局Top-K成本可消除**，dynamicKcount对现Renderer接口仍不可直接移除。完整Selectorhotpath还需MLPbackwardreduction、multi-tensorAdam和Protection数据准备；GIbankrender与FullMapping保留。

{{PREDICTIONS}}

这些是Nsightactive占比的代数情景，不是cudaevent/wall的预测倍数，更不是Mapping承诺。forward-only不能把backward收益一起领走；partstagehostspan不是额外可删时间。若候选只减少launch而不改善大M的SGEMM/activation读写，可能仅改善小M。

{{MAPPING_LIMITS}}

历史normal窗口LoD里还包含Renderer/TF/GIteacher，所以整LoD归零是更宽的条件上限；网络2×只能作用于其中更小的一部分。Optimizer作为另列共享stage，不全部属于Selector；不能将其整段归零。当前rotation/Guard运行应重采相同配置的NativeMappingwall分解后使用`speedup=1/((1-f)+f/s)`，本轮无当前在线新窗口，因此不报猜测的在线倍数。

## 18. 最终20问的明确答复

| # | 答复 |
|---:|---|
| 1 | 3/4/3/1四分支→32→32，concat128→2，4962params、18tensors、ReLU；逐Gaussian独立 |
| 2 | 条件Mature/budget→activatedGS→MLP→GumbelST+prob→GIteacher→merge/nonzero→6GSgather+opacityST→TF→LoDrender/loss/backward→sharedstep/Guard |
| 3 | network=Mature M；checkpoint219969而total220084；历史M只有稀疏同iteration证据，不能拿K或N代替 |
| 4 | 大M Nsightforward4.395ms/backward2.855ms；Gumbel.012ms/rowmax.011ms/mask+ID+gather.098ms；无Top-K |
| 5 | GIteacher/bank/refresh与Protectionclassify/budget/evidence分别归属；共享activation/gather单计，见6/10节 |
| 6 | 351kernels/iter、82.14%<10µs、真实launchgaps和sync；gap不能全归成可删framework开销 |
| 7 | fused32wideMLP与backward、ST/merge/compact准备、18tensor事务Adam；GI独立另测 |
| 8 | warp/Gaussian是合适候选，需与cuBLAS+fusion同口径对比；当前没有证据证明赢家 |
| 9 | 不重写不存在的Top-K；Gumbel单独收益小，适合作为ST融合部分，严格RNG有成本 |
| 10 | 可省cat/onehot/部分fullfloatmask；sameversion可共享geometryembedding，normal跨step不能缓存旧logits |
| 11 | hard-soft.detach+soft，向softJacobian回传；compactopacity仍乘ST，未selectedphotoVJP为0 |
| 12 | 18参数+XYZ/activatedq/scale；tensorratio按契约；opacity/SH/TF继续接原caller图，融合它们时额外返回对应grads |
| 13 | Guard共uniform/事务，Replayjoint不推进normalclock，CDprobL1、DSDBCE和独立临时optimizer；不能统一成normalGumbel |
| 14 | network-only固定M值得试；fullpath受dynamicnonzero、CPUreadback、Guard/topology限制；capture未实测 |
| 15 | networkactive约55%；其2×仅使active模型约1.375×，不是Mapping速度；完整表见17节 |
| 16 | 大Mselectionindexactive小；launch/sync改良更可能帮小M，全部surrounding都删也只是条件上限 |
| 17 | 独立rewrite12µsGumbel、ratio32wideembedding、zero_grad以及不存在的GaussianTop-K优先级低；不能从count判断收益 |
| 18 | 固定noiseFP32Function→warp/cuBLASforward+reductionbackward比较→callerST/compact融合→事务Adam→真实短Mapping验收 |
| 19 | 缺customkernel无profiler对照、当前同配置Mapping/Guard/GIbank细分、完整event/stream/Graph/连续stepparity及统计质量；不缺baseline源码/参数/梯度证据 |
| 20 | ①networkforward/backward；②Protection/feature/ST/compactcaller融合减同步与分配；③compatiblefusedAdam。GI可能更控制Mapping但属于独立生命周期项目 |

## 19. 交付与复现

{{ARTIFACTS}}

完整命令和产物解释见[D/README.md](tests/diagnostics/original_selector_cuda_audit_20261013/README.md)。所有测量使用私有诊断网络与已有checkpoint，未保存渲染RGB或新checkpoint；少量binaryfixture用于复现。没有删除数据集/历史结果，没有commit/push，没有执行生产CUDASelector重写。服务器密码不进入报告、脚本或环境快照。
