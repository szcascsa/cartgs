# Selector CUDA 直接 parity 验收报告

结论：**`CUDA_SELECTOR_CORRECTNESS = PASS`**。

本轮只判断同一 checkpoint、同一输入、同一随机数/target/upstream 下的直接计算是否一致。完整独立 SLAM 的 PSNR、trajectory、equal-time 和 5×5 质量统计不参与本轮 gate，也没有继续运行。

当前分支 `cuda-selector`，ratio 修复提交 `a5c8903`；fused activation backward/激活 view 复用实现提交 `40e88b6`。每个实现文件的 SHA-256 和 Native binary SHA-256 保存在 `tests/diagnostics/cuda_selector_direct_parity_20261015/environment.json`。本轮诊断目录简称 D。

## 相同输入与版本依据

RTX 3090、CUDA 11.8、Torch 2.3.1+cu118，FP32、TF32=false。七个快照每个只读取一份基础模型/Gaussian/metadata，再克隆为 legacy A、legacy B、CUDA；所有参数值一致。Native Adam/CUDA Adam、preview、GI loss 直接调用项目实现；Python Torch ABI=0 与 Native LibTorch ABI=1 分开使用，未混用动态库。

| 快照 | Mapping iteration | Gaussian 数 | entry | 参数 tensors / 参数数 |
|---|---:|---:|---:|---:|
| room_current | 7900 | 102643 | 0.0 | 18 / 4962 |
| desk | 3617 | 52860 | 0.0 | 18 / 4962 |
| xyz | 20961 | 31609 | 0.0 | 18 / 4962 |
| long | 10479 | 129459 | 0.0 | 18 / 4962 |
| earlier_5329 | 5329 | 160289 | 0.600000024 | 18 / 4962 |
| middle_6900 | 6900 | 109564 | 0.0 | 18 / 4962 |
| later_16400 | 16400 | 57489 | 0.0 | 18 / 4962 |

当前 room 是用户 `results/cudaselector/tum_rgbd/room/7900_shutdown/ply`。desk/xyz/long 为已冻结的真实在线快照；额外早/中/后快照按可用 Mapping iteration 选取，不声称它们是同一次 SLAM 的连续训练轨迹。每个快照只与自身克隆比较，不比较不同运行产生的两个模型。完整路径、checkpoint/Gaussian/metadata SHA-256 见 D/checkpoints.json 与 D/summary.json。

网络/backward/Adam/Guard/GI、handoff/elastic entry 及 Python oracle 的计算源码与部署目录 SHA-256 相同。`gumbel_network.h` 的原始文本仅在注释和未编译 CUDA 时的错误提示不同；CUDA 启用条件下的有效定义一致，校验记录单独保留。Native binary 在本轮测量前后 SHA-256 不变。

## Forward 与真实 selection

每个快照测试四个 ratio（.01/.05/.10/.15）和 scalar `[1,1]` / row `[M,1]`，共 56 组。输入准备保留原始 XYZ、quaternion normalize、scale exp、opacity sigmoid 和 metadata/elastic entry。

八层 branch activation、final logits、keep probability、production elastic probability 和 ST：本次全部 exact。max abs、mean abs、RMS、p99、relative L2 均为 0，cosine=1，finite 全部通过；worst Gaussian/index 为 -1 表示没有差异，scalar ratio 的全局 feature 本来没有逐 Gaussian 的唯一 ID。

cached-uniform：56 组、5,151,304 个逐点决定；hard mismatch=0、selected count difference=0、compact global IDs 全部 exact。

normal Gumbel：每组只生成一份 noise 并共享，seeds=0/1/2/42/123/2026；336 组、30,907,824 个逐点决定；soft、ST、class、IDs、count 全部 exact。

不存在 mismatch，因此其 margin 归属为“不适用”，没有把空集合表述成已观察到 99% near-boundary。完整 legacy margin 分布和实际 mismatch ID/margin 列表见 D/selection.jsonl。门槛仍是 hard/count difference≤0.01%，且≥99% mismatch 的 margin≤1e-6。

网络不消费 RNG；正常 Gumbel noise 继续使用 legacy CUDA `exponential_`，没有引入 custom RNG。all-tie 仅诊断：56 组 flip=0、logits ULP difference=0、first differing stage 无；此项不参与最终 pass/fail。

ST forward 的逐 Tensor 数值差异单独记录，hard 决定使用上述 selection 门槛，不额外要求 ST 的硬值差异满足 soft-output 容差。旧 Native 小规模 exact/all-tie harness 同样只作补充 diagnostic；本次也通过，但不参与本轮 gate。

首次执行还通过了这些更严格的额外检查。归档的 `direct_executed.py` 与 `executed_harness.json` 保留执行版本；发布 harness 去掉了额外 veto，数学计算和本次所有测量值没有改变。

## 固定 upstream 的 backward

每个 case 额外运行相同输入的 legacy B，使用完全相同的固定 `grad_logits`、`grad_probability` 和 `grad_ST`。逐 Tensor 检查 XYZ、activated quaternion、activated scale、ratio、raw quaternion、raw log-scale、raw opacity 以及全部 18 个 Selector 参数。

另构造共享 target/upstream 的 ratio loss、原生 GI loss、selected opacity×ST photo VJP、combined loss、Guard shared-uniform、Replay、CD teacher-probability L1、DSD BCE-with-logits。没有分别调用两个 renderer 来生成 upstream。

这些 loss/output/输入与参数 VJP 本次全部 exact：max abs/RMS/p99/relative L2=0，cosine=1；undefined gradient 的有无也相同。每个 Tensor 的 nonzero count、finite、worst index 见 D/tensor_metrics.jsonl（压缩归档同样保留）。未选择点的 photo-opacity 链按原图的 compact IDs 构造；没有把 hard mask 变成断开 ST 的 bool 训练路径。

forward gate 为 max≤1e-6、relative L2≤1e-6、cos≥.999999；gradient gate 为 relative L2≤1e-5、cos≥.99999。接近零的参考 Tensor 使用预注册 absolute RMS/max 判据。本次没有靠放宽 tolerance 通过。

## 1000-step 固定训练与 Adam/Guard

每个快照运行 1000 步，Gaussian 输入固定，仅更新 Selector 18 个参数；四 ratio 循环，scalar/row 交替，cached uniform/Gumbel 交替，外部 photo upstream 使用共享的三个固定 slot。slot 不代表独立 renderer 图或完整 SLAM 帧。三方使用相同 LR、targets、noise sequence，legacy 使用 Native Adam，CUDA 使用生产 CUDA Adam。

| 快照 | 步数 | parameter max relative L2 | Adam m/v max relative L2 | loss max relative difference | average hard mismatch |
|---|---:|---:|---:|---:|---:|
| room_current | 1000 | 0 | 0 / 0 | 0 | 0 |
| desk | 1000 | 0 | 0 / 0 | 0 | 0 |
| xyz | 1000 | 0 | 0 / 0 | 0 | 0 |
| long | 1000 | 0 | 0 / 0 | 0 | 0 |
| earlier_5329 | 1000 | 0 | 0 / 0 | 0 | 0 |
| middle_6900 | 1000 | 0 | 0 / 0 | 0 | 0 |
| later_16400 | 1000 | 0 | 0 / 0 | 0 | 0 |

1/10/100/1000 步有额外逐 Tensor gradient 记录；全部 7000 步都有 logits/probability/loss、gradient、参数、Adam m/v/step 与选择统计。所有检查 exact，无累积漂移；原始逐步 CSV 在 D/training.csv。

另外 direct/preview × 默认/非默认 Adam 参数，共四组各 1000 步，逐 Tensor 检查参数、m/v 和 step，全部 exact。覆盖 normal/zero/undefined grad、LR=0、LR schedule、不同 per-tensor step、beta/eps/weight decay、candidate、commit/discard、shrink 与 direction projection。undefined grad 保持无状态更新，allocated-zero grad 推进 state，二者没有合并。

原生联合 Guard 另测 accept/shrink/projection/reject/fallback/nonfinite 六种结果，output/gradient/candidate/state 与决策输入全部 exact，reject 不会单独 commit Selector。Native CD/DSD/Replay/GradientConstraint smoke test 4/4 通过。直接特殊路径的 loss/VJP 与一步 update 同样 exact；完整长期质量不属于这些测试的结论。

## Legacy 自重复与 renderer 边界

所有本轮 isolated 检查的 legacy A/B repeat error 均为 0；CUDA error 同样为 0，没有超过 legacy 自身误差。两倍 repeat-error 的规则因此没有扩大固定门槛。

本轮没有运行独立 renderer quality 或完整 online。旧完整 renderer 中 legacy 自身的 Gaussian/TF reduction 噪声仅保留为历史 integration diagnostic，不作为本轮 Selector failure，不用旧 room quality pilot 否决这里的同输入证据。

## 新一轮性能复测

仅在直接 correctness 与原生事务测试通过后测量；同一 Native binary、同 checkpoint/属性，100 warmup + 500 measurement。下面均为同步 wall mean，完整 mean/p50/p95、CUDA event 和 memory CSV 保留。

| M | forward legacy / CUDA ms（倍率） | backward legacy / CUDA ms（倍率） | F+B legacy / CUDA ms（倍率） | structural caller legacy / CUDA ms（倍率） |
|---:|---|---|---|---|
| 1000 | 0.22527 / 0.10002（2.252×） | 0.67355 / 0.35357（1.905×） | 1.07084 / 0.52999（2.020×） | 2.55793 / 1.36811（1.870×） |
| 5000 | 0.24441 / 0.11826（2.067×） | 0.71672 / 0.63131（1.135×） | 1.12424 / 0.74662（1.506×） | 2.57026 / 1.42948（1.798×） |
| 10000 | 0.28442 / 0.15192（1.872×） | 0.72368 / 0.39280（1.842×） | 1.13454 / 0.57175（1.984×） | 2.61702 / 1.42409（1.838×） |
| 20000 | 0.44713 / 0.23488（1.904×） | 0.72403 / 0.40219（1.800×） | 1.15542 / 0.57771（2.000×） | 2.70088 / 1.49617（1.805×） |
| 50000 | 1.01499 / 0.53047（1.913×） | 0.72276 / 0.67896（1.064×） | 1.67286 / 1.14798（1.457×） | 3.26884 / 1.79137（1.825×） |
| 100000 | 1.91307 / 1.00499（1.904×） | 1.19006 / 1.17311（1.014×） | 3.03107 / 2.10805（1.438×） | 4.15800 / 2.77609（1.498×） |
| 200000 | 3.63547 / 1.90967（1.904×） | 2.11265 / 2.07390（1.019×） | 5.66330 / 3.91006（1.448×） | 6.65525 / 4.62146（1.440×） |
| 219969 | 3.96797 / 2.11487（1.876×） | 2.31235 / 2.27032（1.019×） | 6.23027 / 4.31278（1.445×） | 7.22394 / 5.03310（1.435×） |

M=219969 的 mean / p50 / p95：

| 阶段 | legacy ms | CUDA ms | 倍率 |
|---|---|---|---:|
| forward | 3.967970 / 3.966240 / 3.976570 | 2.114870 / 2.113990 / 2.126150 | 1.876× |
| backward | 2.312350 / 2.310920 / 2.335740 | 2.270320 / 2.267900 / 2.285620 | 1.019× |
| F+B | 6.230270 / 6.227180 / 6.253290 | 4.312780 / 4.311980 / 4.331170 | 1.445× |
| structural caller | 7.223940 / 7.217360 / 7.275130 | 5.033100 / 5.025660 / 5.075570 | 1.435× |
| Selector Adam | 0.673769 / 0.670167 / 0.714386 | 0.034120 / 0.034046 / 0.034740 | 19.747× |

八个 M 的性能门槛全部通过：forward≥1.7×、F+B≥1.3×、caller≥1.3×，backward 未超过允许的 5% regression：**True**。原始各项判定见 D/acceptance.json；没有用旧历史耗时代替本轮 A/B。

structural caller 包括 Gaussian activation/input gradient、网络、elastic/ST、compact/gather、loss 与 Selector Adam，排除真实 rasterizer/TF/GI/Gaussian optimizer；不等于完整 Mapping。Adam 总参数数固定 4962，各 M 的结果分别记录。没有为性能关闭 Gaussian gradient 或改变 LR/update frequency。CUDA 网络采用 cuBLAS strict + fused activation VJP，warp candidate 不参与当前生产路径。

## 最终回答

1. 同 checkpoint/input 的 branch/logits forward 差异：0。
2. probability 差异：0。
3. cached-uniform hard mask/IDs/count 差异：0。
4. normal fixed-noise Gumbel hard mask/IDs/count 差异：0。
5. 没有 mismatch；near-boundary 归属不适用。
6. 18 参数 gradient 差异：0。
7. XYZ/q/scale/ratio 及直接 raw-input boundary gradient 差异：0。
8. 七个 checkpoint 的 1000 步后 parameter 差异：0。
9. Adam m/v 差异：0，step exact。
10. Guard/Replay/CD/DSD 直接数学和 update parity：通过。
11. legacy isolated repeat error：0。
12. CUDA error 没有超过 repeat error，两者均为0。
13. M=219969 forward/backward/F+B/caller 倍率：1.876× / 1.019× / 1.445× / 1.435×。
14. **`CUDA_SELECTOR_CORRECTNESS = PASS`**，这是相同输入下的算法实现验收；完整 SLAM 的独立运行质量波动不改变该结论。

本轮没有更改网络、loss、sampling、Protection/GI/Replay/CD/DSD、RNG 消费、optimizer 数学或默认 backend。`Selector.backend=legacy|cuda` 继续保留；本报告不自动修改用户正在测试的配置，也不将“直接 correctness 通过”写成未测的完整 Mapping/质量收益。
