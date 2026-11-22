# 消融实验重复波动分析

> 面向当前正在跑的 ChangeDistill / Selector / TF 消融。目标不是"解释噪声"，而是回答：
> **同一份配置重复跑，结果为什么不是同一个分布？哪些量没有真正被固定？**
>
> 结论建立在代码路径 + 本仓库已有实测数据上；代码行号对应当前工作树，量级取自仓库内已有审计报告。

---

## 0. 一句话结论

重复实验的差异不是单一噪声源，而是**三条互相独立的随机链**叠加：

1. **输入链**：ORB-SLAM3 多线程跟踪给出的轨迹 / 关键帧 / 点云每次不同 → 初始 Gaussian 与数据流不同。
2. **训练路径链**：视图顺序由 `std::random_device` 决定（与 seed 无关）；全局 torch RNG 被 split 抽样与保护掩码共用，采样位置随稠密化事件漂移。
3. **剂量链**：ChangeDistill 的"剂量" = 事件数 × epochs × batch 数，而**事件数由 SLAM 闭环/尺度修正触发**，不是配置项。在线训练步数同样由 SLAM 停止时刻和 tail 相位决定。

前两条让模型自己走岔，第三条让**消融的自变量本身不是固定的**——这才是"每次差很多"的主因，尤其对 epochs 这类只作用于事件上的参数。

---

## 1. 本仓库自己已经量到的波动（不是你这次才出现的）

以下数字来自仓库内已有的审计报告，协议都是 **seed=0 的独立进程重复**（和 `scripts/tum_rgbd.sh` 完全同款）：

| 来源 | 量 | 观测 |
|---|---|---|
| `CUDA_SELECTOR_IMPLEMENTATION_REPORT.md:246-252` | 协议 | room/desk/xyz/long，每场景先 3 次 legacy 再 3 次 CUDA，**同一 seed0，不是三个不同 seed** |
| 同上 `:258-261` | Mapping 周期数 | room 6169.3 ± 26.6；desk 3545.3 ± 76.4；xyz 20539.7 ± 368.4 |
| 同上 `:269-276` | 最终 Gaussian 数 | room 119867 ± 6619（5.5%）；desk 62397 ± 9177（**14.7%**）；xyz 32380 ± 840 |
| 同上 `:306-319` | PSNR（full） | room 16.0478 ± 0.2654；xyz 21.4264 ± 0.2310（CUDA 侧 20.6313 ± **1.0558**） |
| 同上 `:293` | CD 实际步数 | `cd_tf_steps` 观测范围 **0–60** |
| `CUDA_TF_IMPLEMENTATION_REPORT.md:282` | PSNR | "旧系统自身已有 **1.9151 dB** 在线波动" |
| `CUDA_SELECTOR_REVISED_ACCEPTANCE_REPORT.md:51` | 判据 | "36 个质量比较中有 5 项超出 max(工程裕度, 2×单次 legacy 重复差)" |
| `README.md:55` | 上游声明 | "We ran each sequence five times to lower the effect of the nondeterministic nature of the system." |

关键读数：**同一配置、同一 seed、同一台机器，最终模型规模能差 15%，PSNR 能差 1 dB，CD 剂量能差 0–60 步。** 这不是"GPU 抖动"，是模型走向了不同分支。

---

## 2. 三条随机链

### 2.1 输入链：SLAM 上游先分叉

`bin/tum_rgbd` 的结构是：ORB-SLAM3（Tracking / LocalMapping / LoopClosing 三线程）与 `GaussianMapper` 训练线程并发，中间靠 `Atlas::MappingOperation` 队列传递关键帧与局部 BA 结果。

- 训练用的初始点云 = `pMap->GetAllMapPoints()`（[`src/gaussian_mapper.cpp:945-963`](src/gaussian_mapper.cpp)），每个 run 的点数、位置、颜色都不同 → `createFromPcd()` 出来的初始 N 就不同。
- 每个 mapping step 训练哪张图，取决于**当时已存在哪些关键帧**；关键帧何时到达由 SLAM 线程调度决定。
- 跟踪节拍用 `usleep(T - ttrack)` 对齐数据集的 30 fps（`examples/tum_rgbd.cpp:225`）。机器一忙，tracking 变慢、mapping 与 tracking 的相对进度变化 → **每个 mapping step 看到的关键帧人口不同**。
- 若某次 tracking 丢失较重，Atlas 可能切图/尺度修正（`ScaleRefinement`，[`src/gaussian_mapper.cpp:3449-3454`](src/gaussian_mapper.cpp)），会使 Gaussian 群整体被变换，质量出现**双峰**。
- ORB-SLAM3 内部唯一用随机数的地方是 `Sim3Solver` / `MLPnPsolver` 的 RANSAC（`DUtils::Random::RandomInt` → 全局 `rand()`，`third_party/ORB-SLAM3/src/Sim3Solver.cc:174,243`）。它虽然被 `std::srand(0)` 固定了起点，但**抽签次数取决于调用次数**，所以闭环/重定位的结果仍会随线程交错漂移。

> 判断方法：比较各 run 的 `KeyFrameTrajectory_TUM.txt` 行数、`CameraTrajectory_TUM.txt` 行数、`TrackingTime.txt` 帧数。三者只要有一个不同，"同一份数据"这个前提就不成立。

### 2.2 训练路径链：视图顺序根本没被 seed 控制

```cpp
// src/gaussian_mapper.cpp:3579
void GaussianMapper::generateKfidRandomShuffle() {
  ...
  std::mt19937 g(rd_());                  // ← rd_ 是 std::random_device 成员
  std::shuffle(kfid_shuffle_.begin(), kfid_shuffle_.end(), g);
}
```

- `rd_` 是 `std::random_device`（`include/gaussian_mapper.h:542`），每次调用都取 OS 熵。
- `scene_->addKeyframe()` 会把 `kfid_shuffled_` 置回 false（`src/gaussian_scene.cpp:46`），于是**每来一个新关键帧就重新洗一次牌**。
- 每个 mapping step 都从这把洗好的牌里取下一个 KF（`useOneRandomSlidingWindowKeyframe()`，被 `trainForOneIteration()` 在 [`src/gaussian_mapper.cpp:1285`](src/gaussian_mapper.cpp) 调用）。

所以：**`std::srand(seed)` / `torch::manual_seed(seed)` 全部生效，仍然无法复现任何一次 run**。相同 KF 集合、相同次数，但"先训哪张、后训哪张"每次都不同。

顺序为什么重要？因为下一步的梯度、`xyz_gradient_accum_`、`max_radii2D_` 都是**逐视图累加**的，而稠密化是在固定 iteration 上用**阈值**判定 clone/split/prune。顺序换一下，第 500/1000/… 步的阈值判定就会翻转若干点 → 拓扑不同 → 之后的梯度、掩码、成熟度全部不同。

### 2.3 剂量链：消融的自变量是"事件数 × epochs"

CD 不在固定 iteration 上跑，而是被**几何事件**触发：

| 触发点 | 位置 | 事件来源 |
|---|---|---|
| `LOCAL_GEOMETRY` | [`src/gaussian_mapper.cpp:3345-3347`](src/gaussian_mapper.cpp) | ORB-SLAM3 闭环且位姿变化超阈值 |
| `GLOBAL_GEOMETRY` | `:3449-3454` | Scale refinement |
| `FIELD_DOMAIN` | `:2534-2535` | AABB 扩张 |

`finishGeometryChange()` 在 `capture.snapshot_count == 0` 时直接返回（`:2427-2441`），在某些 run 里**一次 CD 都不会发生**。

一次事件的优化步数：

```
selector_steps ≈ ceil(selector_count / batch_size) × selector_epochs
tf_steps       ≈ ceil(tf_count / batch_size)       × tf_epochs
```

所以真实剂量 = Σ(每次事件) × epochs。`cd_tf_steps` 实测 0–60 就是这件事的直接证据：**同样的 `epochs: 5`，有的 run 打进去 0 步，有的 60 步。**

这带来一个致命的推论：

> 如果某次 run 期间既没有发生闭环、也没有触发 AABB 扩张（`distill_tf_on_aabb_change: 1`），那么 `selector_epochs: 5 → 20` 这个消融在该 run 上**没有任何作用**，两臂差异 100% 来自其他噪声；如果发生了 2 次事件，差异被放大 2 倍。
> 于是"臂间差异"= 一个**点质量在 0、其余为随机放大量**的混合分布，用 3 次重复去估计它没有意义。

另外，CD 是**同步**执行的，占着 `mutex_render_`（`:2416`）跑在 mapping 线程里。epochs 翻 4 倍会让每次事件的墙钟时间翻几倍，而在线驱动器的总步数是由墙钟决定的（见 2.4）→ **"多蒸馏"会挤掉正常训练步数**，这是一个系统性混杂，不只是随机波动。

### 2.4 在线步数：不是预算，是"SLAM 几时停 + 相位"

`Optimization.max_num_iterations: -1`（`cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml:56`）→ 在线循环一直跑到 SLAM shutdown，步数 = ∫(mapping 速率)dt。之后还有一段 tail：

```cpp
// src/gaussian_mapper.cpp:1140-1149
int densify_interval = densifyInterval();     // 500
int n_delay_iters = densify_interval * 0.8;   // 400
while (... &&
       (getIteration() - SLAM_stop_iter < n_delay_iters ||
        getIteration() % densify_interval < n_delay_iters ||
        isKeepingTraining())) {
  trainForOneIteration();
}
```

`keep_training_` 只在 viewer 里被置 false，headless 运行始终为 false。于是 tail 长度 = **第一个满足 `i - SLAM_stop ≥ 400` 且 `i % 500 ≥ 400` 的 i**：令 `r = (SLAM_stop + 400) % 500`，则 `r ≥ 400` 时 tail = 400 步，`r < 400` 时 tail = 400 + (400 − r) 步。**范围 400–800 步，均值约 560，完全由 SLAM 停止时刻的相位决定**。这一段跑在全量最终关键帧上，且结果就是最后保存的模型。

也就是说：**最后 6.7%–13% 的训练量（短序列可达 20%+）是随机的。**

固定预算模式（`CARTGS_MATURITY_ABLATION`，`examples/tum_rgbd.cpp:110-134` + `include/gaussian_mapper.h:133-151`）能消掉这一项：`Experiment.mapping_steps` 让步数严格相等、tail 关闭。但**它消不掉 2.1 / 2.2 / 2.3**。

---

## 3. 放大器：为什么 1e-7 的差异能变成 1 dB

单纯"顺序不同"不会必然产生 1 dB 差异，中间有三个正反馈：

1. **离散阈值事件**：`densifyAndClone/Split/Prune` 用 `grad >= 0.001`、`opacity < 0.02`、`max_radii2D_ > max_screen_size` 判定（`src/gaussian_model.cpp:797-893`）。阈值附近的点，梯度差 1e-6 就可能翻边 → 拓扑变化 → 后续所有量级联变化。一次 run 里会经历 10+ 轮稠密化，每轮可能翻转数百个点。
2. **共享 RNG 流的位置漂移**：`densifyAndSplit` 画 `torch::normal(0,1,[2N,3])`（`src/gaussian_model.cpp:815`），`protection_.cached_uniform` 在 `reset()` / `append()` 时画 `torch::rand`（`include/gaussian_protection.h:62,97`），且**每个新关键帧整表重采样一次**（`src/gaussian_mapper.cpp:2777`）。这些都用全局 torch RNG：一旦某轮 split 多出 100 个点，之后所有采样位置整体错位。**"固定 seed"只固定了序列起点，没固定序列位置。**
3. **掩码本身是逐 run 重抽的**：`sampleSelectorST()` 用 `cached_uniform < probability` 做 hard mask（`include/selector_handoff.h:14-21`），而 `cached_uniform` 每次新关键帧、每次稠密化都会变 → **哪些 Gaussian 被选中，本质上是每次 run 重新抽一次的 Bernoulli 样本**。N=12 万、ratio=7.5% 时每次抽 9000 行，行集合不同 → 监督子集不同。

再加上 CUDA 侧的 `atomicAdd`：

- 反向：`cuda_rasterizer/backward.cu:643-650`、`:779-813`（dL_dmean2D / dL_dconic2D / dL_dopacity / dL_dcolors）
- 前向 GI 统计：`cuda_rasterizer/forward.cu:421-422`（`atomicAdd(frame_importance …)`）

这两个都是**浮点累加顺序依赖**的，同一输入每次 backward 的结果在 ULP 级别不同。单独看只有 1e-7，但经过上面的阈值放大器，会被放大到可见。

> 这一条不代表"CUDA 实现有 bug"：3DGS 家族普遍如此。它只意味着**逐位复现是不可能的**，也意味着不能靠"固定 seed"来消灭方差，只能靠统计设计。

---

## 4. 测量层：还有三处会让"同一模型"测出不同分

1. **评估帧集不同**：`python/run.py` 的关联帧来自**估计轨迹**（`associations = associate_frames(tstamp, gt_tstamp)`，`:476`）。某次跟踪少 30 帧，均值 PSNR 就少算 30 帧；跟踪丢失时更是直接截断。仓库里已经有 `--evaluation-frame-set`（`:180-182, 477-482`）就是为这个问题加的，**但 `scripts/tum_rgbd.sh` 没有传**。
2. **检查点选择**：`run.py:199-232` 按目录名数字取最大 iteration 的 `*_shutdown`。如果 tail 长短不同，两臂保存的 iteration 号与模型状态并不同步。
3. **LoD 的 `cached_uniform`**：评估侧用的是 checkpoint 里保存的 uniform（`python/selector.py:133`），对同一 checkpoint 是确定的——这点是好的。但不同 run 的 checkpoint 里的 uniform 是不同样本，"同 ratio 下的 active 数/行集合"本身就带随机性。

---

## 5. 还有一类"看起来像波动"的差异：配置漂移

仓库里两份权威 YAML 目前**并不一致**，且差异正好落在本次消融的旋钮上：

| 键 | `cfg/selector/fast.yaml` | `cfg/selector/selector_config.yaml` |
|---|---|---|
| `ChangeDistill.epochs` | **5** | **1** |
| `TransformField.start_iter` | 3000 | 5000 |
| `Selector.lod_render_scale` | 0.5 | 1.0 |
| `TransformField.feature_dim` / `hidden_dim` | 8 / 32 | 16 / 64 |
| `TransformField.multires` | [1, 2] | [1, 2, 4] |
| `TransformField.aabb_scale` | 1 | 1.3 |
| `Protection.performance_logging` | 0 | 1 |

如果 runner 对不同臂用了不同文件（或某次忘了指定 `--selector-config`，`scripts/tum_rgbd.sh:3` 默认是 `selector_config.yaml` 而不是 `fast.yaml`），那"消融差异"里就混进了一个 `epochs: 1 vs 5` 的整包差异。

另外：工作树当前有 **40 个文件未提交**（`git status`），`src/gaussian_mapper.cpp` 改了 290 行。服务端与本地是否同一二进制，需要每次记录 `git rev-parse HEAD` + binary SHA-256，否则"今天和昨天不一样"可能纯粹是代码变了。

---

## 6. 诊断协议：用现有产物定位主导项

### 6.1 先看四个数（每个 run 都要看，不只看 PSNR）

| 量 | 从哪读 | 期望 | 不等的含义 |
|---|---|---|---|
| `mapping_steps` | 日志 `N iterations completed.` | 固定（若用固定预算） | 步数不同 → 训练量不同 |
| `eval_frames` | `evaluation_frames.json` 的 `gt_indices` 长度 | 固定 | 评估像素集不同 → 均值不可比 |
| `cd_tf_steps` / `cd_selector_steps` | `<N>_shutdown/ply/transform_state.json` | 固定 | **CD 剂量不同 → epochs 消融自变量失效** |
| `total_gaussians` | `selector_stats.json` 或 `point_cloud.ply` 头 | 接近 | 拓扑分叉，已不是同一个模型 |

`transform_state.json` 里还有几个非常好用的"分支时间戳"：
`tf_initialization_iteration`、`first_ready_population_iteration`、`ready_count_when_initialized`、`tail_completed_steps`、`guard_accept/cancel`、`replay_commit`。它们相等，才说明两条 run 走的是同一分支。

### 6.2 再看事件流（决定 CD 是否真的发生）

日志里数这几行：

```
[Gaussian Mapper]Loop Closure Detected.
[Gaussian Mapper]Scale refinement Detected.
[CD BEGIN] event=… epochs=… selector_epochs=… tf_epochs=… count=…
[CD END]   event=… selector_steps=… tf_steps=…
[ChangeCoordinator] skip type=… reason=no_snapshot
[TF CD SKIP] / [SELECTOR CD SKIP]
```

若两臂的 `[CD BEGIN]` 次数不同，请先别比较 PSNR——**先比较"每次事件内"的 loss 轨迹**（`[SELECTOR CD EPOCH] loss=… delta_prev=…`、`[TF CD EPOCH]`）。那是这个机制真正作用的层级，也是唯一能把 epochs 的效果从事件噪声里剥出来的观测量。

### 6.3 一次出表

新增脚本 `scripts/analyze_ablation_variance.py`：

```bash
python scripts/analyze_ablation_variance.py \
  --arm baseline=results/change_distill_ablation/<ts>/<baseline_dir> \
  --arm epochs20=results/change_distill_ablation/<ts>/<epochs20_dir> \
  --target-delta 0.2 --json variance.json
```

它输出四块：臂内噪声底（mean ± std, min/max）、"因果变量是否真的固定"、逐臂判据（步数/帧集/剂量/人口）、以及**当前 n 能分辨多大的差异**。脚本只读已有产物，不改任何东西；嵌套的 `lod_eval/ratio_*/eval.txt` 不会被误当成重复。

> 想让脚本拿到逐 step 信息（第一次分歧出现在第几步），需要在诊断 run 里临时打开 `Mapping.iteration_logging: 1`（`cfg/selector/fast.yaml:8` 当前为 0）。注意它每步做一次 CUDA 同步，会改变墙钟节拍，所以**只用于诊断，不用于正式对比的臂**。

---

## 7. 修复清单（按投入产出排序）

### 7.1 立刻可做（不改代码，只改流程）

1. **冻结一份配置**，A/B 两臂只允许一个键不同；把实际读取的 `selector_config.yaml` 从 `_shutdown/ply/` 里拷回来归档（mapper 会自动写一份，`saveSelectorConfig()`）。
2. **配对评估帧集**：先跑一次，把 `evaluation_frames.json` 当公共帧集传给所有臂（`--evaluation-frame-set`）。
3. **串行 + 空载**：同一时刻只跑一个 run；开工前确认 `nvidia-smi --query-compute-apps` 为空（本仓库的性能审计已经在这么做：`tests/diagnostics/change_distill_performance_20261017/run_matrix.py:32-42`，并把被污染的 case 标成 `CONTAMINATED.json`）。
4. **记录指纹**：`git rev-parse HEAD`、二进制 SHA-256、`Experiment.seed`、`Experiment.mapping_steps`（若用固定预算）。没有指纹的两次 run 不能算重复。
5. **若做 epochs 消融**：确认用的是 `tum_rgbd_maturity_ablation`（固定步数、无 tail），而不是自由跑的 `tum_rgbd`。两者方差差一个量级。

### 7.2 小改代码（低风险，收益最大）

**(a) 视图顺序接回 seed**（`src/gaussian_mapper.cpp:3579-3589`）

```cpp
std::seed_seq seq{static_cast<std::uint32_t>(initialization_seed_),
                  static_cast<std::uint32_t>(scene_->keyframes().size()),
                  static_cast<std::uint32_t>(change_version_)};
std::mt19937 g(seq);
std::shuffle(kfid_shuffle_.begin(), kfid_shuffle_.end(), g);
```

这样"第几个关键帧到达"决定洗牌结果，可复现；不需要引入新配置。

**(b) 给两处全局 RNG 调用装独立 generator**

- `src/gaussian_model.cpp:815` 的 split 噪声：派生一个 `torch::Generator`，seed = `initialization_seed_ ^ (mapping_iter_ * 常数) ^ topology_version`，按当前人口建 generator 后传入 `torch::normal(..., gen)`。
- `include/gaussian_protection.h:62,97` 的 `cached_uniform`：不要从全局流抽，改成按行用哈希派生（仓库里 `include/global_gi_bank.h:182` 已有 `(id ^ sampling_seed) + 黄金常数` 的写法），这样"第几次抽样"不再改变掩码本身。

**(c) 拆开 CD 的两个随机流**（`src/change_distill.cpp:98`）

```cpp
// 现在：一个 rng_ 同时服务 selector shuffle 和 tf shuffle
std::shuffle(order.begin(), order.end(), rng_);
```

`epochs` 一变，selector 分支的抽取次数就变，TF 的 shuffle 位置随之漂移——`improvements/change_distill_epochs.md:17-19` 里已经写明这一点。若你的 runner 没有额外固定 TF 起点，把它拆成 `selector_rng_` / `tf_rng_`（或在每个事件开始时 `rng_.seed(config.seed ^ event.id)`），否则**消融 A 与 B 的差异里有一部分只是随机数位置差**。

### 7.3 结构性改造（想拿到可信结论的话）

把"epochs 效果"从端到端 PSNR 搬到**受控回放**上：冻结同一个 `event.json` + `snapshot.pt` + `selector.pt` + `transform_field.pt`（你们性能审计已经在用 `/…/change_distill_ablation/20261017-2038/events_supplement/local_loop/event_6207_1` 这套格式，`run_matrix.py:13`），在固定输入上只改 epochs，测：

- 每次事件的 pre/post loss（`[SELECTOR CD EPOCH]` / `[TF CD EPOCH]`）
- 事件后固定视角的 PSNR/SSIM（固定帧集、固定 uniform）
- 以及"是否破坏了 untouched anchor"（日志里已有 `untouched_pre/post`）

端到端 PSNR 只作为二级证据，且必须 n≥5 + 配对帧集。

---

## 8. 统计上：三次重复到底能分辨什么

两臂各 n 次、双侧 α=0.05、power=0.80 时的最小可检测效应（MDE）≈ `2.80 · σ · sqrt(2/n)`：

| 臂内 σ (dB) | n=3 | n=5 | n=10 |
|---|---|---|---|
| 0.23（xyz legacy 实测） | 0.53 | 0.41 | 0.29 |
| 0.27（room legacy 实测） | 0.62 | 0.48 | 0.34 |
| 0.50 | 1.14 | 0.89 | 0.63 |
| 1.06（xyz CUDA 实测） | 2.43 | 1.88 | 1.33 |
| 1.92（TF 审计实测） | 4.39 | 3.40 | 2.41 |

读法：**σ=0.27 dB 时，3 次重复只能判定 >0.62 dB 的差异**。你们的 pilot 里臂间差经常是 0.1–0.3 dB——那在这个噪声水平下本来就不可判，`CUDA_SELECTOR_REVISED_ACCEPTANCE_REPORT.md:51` 的"未通过"正是同一个问题。

反推：若目标效应是 0.2 dB，需要 σ ≲ 0.11 dB（n=5）或 σ ≲ 0.16 dB（n=10）。这只能靠**先砍噪声源**，不是靠加重复次数。

---

## 9. 需要你提供的信息（我可以直接接着做）

1. 这次消融的**臂**是什么：`ChangeDistill.selector_epochs` / `tf_epochs`、`maturity_ablation_mode`，还是别的？
2. 用的是 `bin/tum_rgbd`（自由跑）还是 `bin/tum_rgbd_maturity_ablation`（固定步数）？
3. 有没有一份 run 目录（或 `train.log`）可以给我？我可以直接跑第 6 节的表，并定位到"第一次出现分歧的 step"（本仓库已有同类做法：`tests/diagnostics/cuda_selector_implementation_20261015/summarize_online.py:86-91` 的 first-logged-input-difference）。
4. 每个臂准备跑几次？如果只有 3 次，建议先按 7.2(a)(c) 打两个小补丁再跑，否则结论会被噪声吃掉。

---

### 附：快速自检清单

- [ ] 两臂的 `mapping_steps` 一致（或都用固定预算）
- [ ] 两臂的 `eval_frames` 一致（或都传同一个 `--evaluation-frame-set`）
- [ ] 两臂的 `cd_tf_steps` / `cd_selector_steps` 同量级
- [ ] 两臂 `[CD BEGIN]` 次数相同，或改为逐事件比较
- [ ] `tf_initialization_iteration` / `ready_count_when_initialized` 一致
- [ ] `total_gaussians` 差异 < 5%
- [ ] `CameraTrajectory_TUM.txt` 行数一致
- [ ] 配置来自同一文件、同一二进制、同一 commit
- [ ] 运行期间 GPU 无其他进程
- [ ] 每个臂 n ≥ 5，且报告 mean ± std + min/max（不要只报均值）
