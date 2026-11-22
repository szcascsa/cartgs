# Continuous Cycle-Geometric Maturity — Current-Pipeline Audit

Read-only audit of the current repository (HEAD `f24124c`, working tree has
uncommitted docs/config only — **no source file was modified**).

Active online configuration in this checkout is `cfg/selector/fast.yaml`
(`Training.ultra_light_enabled: 1` → `ultra_light_mode_ = 1` →
`ultrafast_rotation_.enabled = true`), so the 5-step
`0.01 → 0.05 → 0.10 → 0.15 → Full` rotation is the live path.

---

## 1. Current ultrafast rotation pipeline

### 1.1 Entry and phase definition

* `GaussianMapper::trainForOneIteration()` — [gaussian_mapper.cpp:1219-1348](src/gaussian_mapper.cpp#L1219-L1348)
  is the only entry that advances the rotation. It refuses work when
  `!ultrafast_rotation_.allowsNormalStep(online, mapping_iter_, normalMappingLimit())`
  or when `opt_params_.iterations_` is exhausted ([gaussian_mapper.cpp:1221-1229](src/gaussian_mapper.cpp#L1221-L1229)).
* `GaussianMapper::trainMappingIteration(camera, enhanced, resumed_after_slam)` — [gaussian_mapper.cpp:1350-1704](src/gaussian_mapper.cpp#L1350-L1704)
  contains the whole body.
* Phase is stateless and derived from a single counter:
  `UltrafastRotation::phase() = step % 5`, `full() = phase() == 4`,
  `ratio() = full() ? 1.0f : ratios[phase()]` with
  `ratios = {0.01, 0.05, 0.10, 0.15}` — [ultrafast_rotation.h:12-28](include/ultrafast_rotation.h#L12-L28).
  `step` is advanced **once per completed mapping** by
  `UltrafastRotation::complete(...)` — [ultrafast_rotation.h:41-52](include/ultrafast_rotation.h#L41-L52).
* `full_step = !rotation || ultrafast_rotation_.full()` — [gaussian_mapper.cpp:1359](src/gaussian_mapper.cpp#L1359).

Ratios 0.01/0.05/0.10/0.15 for `ultra_light_mode_ == 1` are **hard-assigned**
from `UltrafastRotation::ratios` and cannot be overridden by
`Selector.target_ratios` ([gaussian_mapper.cpp:544-548](src/gaussian_mapper.cpp#L544-L548)).

### 1.2 Keyframe selection

`useOneRandomSlidingWindowKeyframe()` is called **once per mapping, before any
phase branch**, at [gaussian_mapper.cpp:1233-1234](src/gaussian_mapper.cpp#L1233-L1234);
the implementation is [gaussian_mapper.cpp:3427-3490](src/gaussian_mapper.cpp#L3427-L3490).
The chosen keyframe is therefore identical in role for LoD and Full steps — it is
not re-drawn per phase. The ratio rotation never influences KF selection.

### 1.3 Per-phase real call sequence

Common preamble (all 5 phases), lines [gaussian_mapper.cpp:1353-1430](src/gaussian_mapper.cpp#L1353-L1430):

1. `ultrafast_training_timer_->begin()`
2. `writeKeyframeUsedTimes(...)`
3. lock `mutex_render_` (held until the end of the body)
4. SH degree bump every 1000 iterations; `gaussians_->setShDegree`
5. `global_gi_->setMappingIteration`, `updateGlobalGIBank()`
6. `gaussians_->optimizer_->zero_grad(true)` (rotation or mode2)
7. `selector_optimizer_->zero_grad(true)` / `transform_optimizer_->zero_grad(true)` **only when `full_step`**
8. `protection_.profiler->beginIteration()`, `protection_.advance(mapping_iter_ + 1)`
9. `updateLearningRate(...)`, `setFeatureLearningRate`, `setOpacityLearningRate`,
   `setScalingLearningRate`, `setRotationLearningRate`

#### Phase 0 — ratio 0.01 (`phase()==0`, `full_step==false`), lines 1478-1493

| step | where |
|---|---|
| LoD branch entered | [1478](src/gaussian_mapper.cpp#L1478) `!mode2 && enhanced && (!rotation \|\| !full_step)` |
| `ratio_index = phase` | [1479](src/gaussian_mapper.cpp#L1479) |
| **render** | inside `accumulateLodGradients` → `GaussianRenderer::render` [3035-3043](src/gaussian_mapper.cpp#L3035-L3043) |
| **loss** | `selector_loss = render_weight*photo + ratio_weight*ratio_loss (+ gi_loss + tf_reg)` [3058-3077](src/gaussian_mapper.cpp#L3058-L3077) |
| **backward** | `selector_loss.backward()` [3084-3087](src/gaussian_mapper.cpp#L3084-L3087) |
| **Gaussian `optimizer_->step()`** | [1551](src/gaussian_mapper.cpp#L1551) via `commit_optimizer()` → `stepWithHistoricalGuard` [1533](src/gaussian_mapper.cpp#L1533) |
| **Selector `optimizer_->step()`** | **the same** `step()` (shared optimizer group `shared_selector_group_index_`) [1551](src/gaussian_mapper.cpp#L1551) |
| **TF `optimizer_->step()`** | **the same** `step()` (groups `shared_transform_mlp_group_index_`, `shared_transform_grid_group_index_`) [1551](src/gaussian_mapper.cpp#L1551) |
| **zero_grad** | [1547](src/gaussian_mapper.cpp#L1547) plus [2889-2893](src/gaussian_mapper.cpp#L2889-L2893) |
| `kfs_loss_` | **not touched** (Full only) [1552-1556](src/gaussian_mapper.cpp#L1552-L1556) |
| densify stats | **not touched** (`full_step` only) [1575-1586](src/gaussian_mapper.cpp#L1575-L1586) |
| densify/clone/split/prune | **not executed** (`full_step` only) [1575-1612](src/gaussian_mapper.cpp#L1575-L1612) |
| Selector/TF `zero_grad` | [2889-2893](src/gaussian_mapper.cpp#L2889-L2893) *inside* `accumulateLodGradients` |

**Critical ordering fact:** in a LoD step the optimizer commit at line 1551 happens
**before** the LoD render (which is produced inside `accumulateLodGradients`,
called at line 1487, i.e. after the forward full-render block but *before*
`commit_optimizer()`). For a LoD phase there is no Full render, so the LoD render
at line 1487 already uses the parameters left by the **previous** phase, and the
step at line 1551 then applies the *LoD* gradients. Verified by
[tests/ultrafast_mapping_test.cpp:156-162](tests/ultrafast_mapping_test.cpp#L156-L162).

#### Phases 1,2,3 — ratios 0.05 / 0.10 / 0.15

Byte-for-byte identical code path to phase 0, with
`ratio_index = 1, 2, 3` → `selector_target_ratios_[ratio_index]`
([gaussian_mapper.cpp:1479-1490](src/gaussian_mapper.cpp#L1479-L1490)).
Only the ratio fed to the Selector, the demand-driven budget, and the TF
`ratio` argument change.

#### Phase 4 — Full (`full_step == true`), lines 1442-1476 + 1552-1664

| step | where |
|---|---|
| **render** | `GaussianRenderer::render(...)` [1445-1452](src/gaussian_mapper.cpp#L1445-L1452) (only place `collect_importance` can be true) |
| protection Full transaction opens | `protection_.beginFullMapping(render_pkg.visibility_indices)` [1462](src/gaussian_mapper.cpp#L1462) |
| **loss** | `full_loss = (1-λ)·L1 + λ·(1-SSIM) (+ opacity_reg)` [1463-1472](src/gaussian_mapper.cpp#L1463-L1472) |
| **backward** | `full_loss.backward()` [1473-1475](src/gaussian_mapper.cpp#L1473-L1475) |
| LoD branch | **skipped** (`!full_step` false) |
| **Gaussian/Selector/TF `step()`** | `commit_optimizer()` at [1551](src/gaussian_mapper.cpp#L1551) — shared Adam, all three sub-systems |
| `kfs_loss_[fid]` + `ema_loss_for_log_` | [1552-1556](src/gaussian_mapper.cpp#L1552-L1556) |
| optional image record | [1558-1561](src/gaussian_mapper.cpp#L1558-L1561) |
| densify deadline observation | [1568-1574](src/gaussian_mapper.cpp#L1568-L1574) |
| max_radii2D + `addDensificationStats` | [1578-1586](src/gaussian_mapper.cpp#L1578-L1586) |
| **densify/clone/split/prune** | `densifyAndPrune(...)` [1591-1598](src/gaussian_mapper.cpp#L1591-L1598) |
| opacity reset | [1605-1611](src/gaussian_mapper.cpp#L1605-L1611) |
| `protection_.finishFullMapping(committed, kfid)` | [1635-1636](src/gaussian_mapper.cpp#L1635-L1636) |
| `ultrafast_rotation_.complete(...)` | [1658-1659](src/gaussian_mapper.cpp#L1658-L1659) |

`zero_grad` for Full happens in the preamble (line 1413) and after the step
(line 1547); note that Gaussian gradients **accumulate across all five phases**
because there is no `zero_grad` between phases 0→4 for the three LoD phases
(line 1413 fires at the top of every iteration, so in practice each phase starts
clean — but the four LoD `backward()` calls and the Full `backward()` are
separated by that per-iteration `zero_grad`).

### 1.4 Where one complete cycle ends

```
step 0: ratio 0.01  (LOD)
step 1: ratio 0.05  (LOD)
step 2: ratio 0.10  (LOD)
step 3: ratio 0.15  (LOD)
step 4: Full         (FULL)  <-- cycle ends here
step 5: ratio 0.01  (LOD)   <- next cycle
```

The counter is advanced at the **last statement of `trainMappingIteration`**:

* `ultrafast_rotation_.complete(ms, render_ms, rendered, selector_committed, tf_committed)`
  — [gaussian_mapper.cpp:1658-1659](src/gaussian_mapper.cpp#L1658-L1659)
  → `++step` and simultaneously `phase()` changes for the next call.

So the ABI-level cycle boundary is: **end of `trainMappingIteration()` in the
`rotation && full_step` execution, after line 1659, before the function returns
at line 1704.** Everything before that within the same call (optimizer step,
densify, prune, protection commit) belongs to the *previous* cycle; everything
after belongs to the next.

`safe point` recommendation is in section 9.

### 1.5 Adjacent path (`ultra_light_mode_ == 2`) — different structure

`mode2` is **not** a 5-phase rotation: [gaussian_mapper.cpp:1668-1702](src/gaussian_mapper.cpp#L1668-L1702)
runs a **native Full step for every mapping** (lines 1442-1476, `rotation==false`) and
additionally a detached LoD branch with
`ratio_index = normal_lod_branch_count_ % selector_target_ratios_.size()`
([1675](src/gaussian_mapper.cpp#L1675), [1686](src/gaussian_mapper.cpp#L1686)). Its
cycle definition is therefore "one mapping = Full + one LoD ratio", which is
**not** Full-to-Full. Any cycle-geometric maturity must special-case or gate
this mode.

### 1.6 Observable / test evidence of the 5-step cadence

* `tests/ultra_light_schedule_test.cpp:12-22` asserts `phase() == mapping % 5`.
* `tests/ultrafast_mapping_test.cpp:184-186` asserts exactly
  `full_render_count == 1 && lod_render_count == 4` after 5 mappings.

---

## 2. Exact Gaussian geometry representation

All tensors live on `GaussianModel` ([include/gaussian_model.h:190-201](include/gaussian_model.h#L190-L201))
and are `torch::TensorOptions().dtype(torch::kFloat).device(device_type_)`
created via the `GAUSSIAN_MODEL_INIT_TENSORS` macro
([include/gaussian_model.h:53-72](include/gaussian_model.h#L53-L72)).
`device_type_` is `kCUDA` when available ([gaussian_model.cpp:55-57](src/gaussian_model.cpp#L55-L57)).
Default dtype is float32 unless the config overrides it — nothing in
`cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml` sets a dtype, so **float32 is
actual**.

| quantity | tensor | shape | dtype | raw/internal | device |
|---|---|---|---|---|---|
| XYZ | `xyz_` | `[N,3]` | float32 | world metres, directly usable | CUDA |
| scale | `scaling_` | `[N,3]` | float32 | **log-scale** (`log σ` per axis) | CUDA |
| rotation | `rotation_` | `[N,4]` | float32 | **unnormalized** quaternion, order `w,x,y,z` | CUDA |
| opacity | `opacity_` | `[N,1]` | float32 | pre-sigmoid logit | CUDA |
| existence | `exist_since_iter_` | `[N]` | int32 | iteration of creation | CUDA |
| last elastic supervision | `last_elastic_supervised_iter_` | `[N]` | int64 | mapping iter of last LoD supervision | CUDA |
| densify accum | `xyz_gradient_accum_` | `[N,1]` | float32 | Σ ‖∇screen‖ | CUDA |
| densify count | `denom_` | `[N,1]` | float32 | visibility count | CUDA |
| max screen radius | `max_radii2D_` | `[N]` | float32 | px | CUDA |

Answers to the specific questions:

* **Scale is stored as raw log-scale.** Conversion to physical: `getScalingActivation()`
  → `torch::exp(this->scaling_)` — [gaussian_model.cpp:62-64](src/gaussian_model.cpp#L62-L64).
  Inverse: `torch::log(σ)` is written directly in `densifyAndSplit`
  ([gaussian_model.cpp:818-820](src/gaussian_model.cpp#L818-L820)).
* **Rotation is stored unnormalized** and normalized *lazily* whenever an
  activation is requested: `getRotationActivation() = torch::nn::functional::normalize(rotation_)`
  — [gaussian_model.cpp:66-68](src/gaussian_model.cpp#L66-L68). Other
  normalizations: `general_utils::build_rotation` normalizes internally
  ([include/general_utils.h:31-38](include/general_utils.h#L31-L38)),
  `quaternion_utils::normalize` ([include/quaternion_utils.h:7-11](include/quaternion_utils.h#L7-L11)),
  and `GuardGaussianParameters::input` normalizes on the fly
  ([historical_gradient_render.cpp:9-11](src/historical_gradient_render.cpp#L9-L11)).
  `rotation_` itself is never rewritten in place by a normalization step; it is
  only replaced by `replaceTensorToOptimizer` after a correction.
* **R / covariance utility already exists**: `getCovarianceActivation(int scaling_modifier)`
  builds `L = R·diag(s)`, `Σ = L·Lᵀ` — [gaussian_model.cpp:81-99](src/gaussian_model.cpp#L81-L99);
  `general_utils::build_rotation(r)` returns `[N,3,3]`
  ([include/general_utils.h:31-55](include/general_utils.h#L31-L55)).
  **Do not use either for maturity drift** — covariance construction is the
  expensive path (matmuls + a `[N,3,3]` allocation).
* **Cheap physical scale without covariance**: `torch::exp(scaling_)` (elementwise,
  `[N,3]`). Even cheaper for drift: work directly in **log-scale space** and take
  `|Δlog σ|`, which needs no `exp` at all.
* **Cheapest drift representation already used elsewhere in the repo** (reusable
  pattern): `GaussianAttrState{xyz, log_scaling, rotation}`
  ([include/gaussian_transform_ops.h:15-27](include/gaussian_transform_ops.h#L15-L27)) and the
  reference metric in `DriftStabilityManager::tfLossTensor`
  ([drift_stability.cpp:209-213](src/drift_stability.cpp#L209-L213)):
  ```
  position = |Δxyz|.mean() / max(diagonal, eps)
  scale    = |Δlog_scaling|.mean()
  rotation = 1 - |<q_a, q_b>|.clamp(0,1)
  ```
  That is exactly position / log-scale / quaternion-angular drift, and it
  already exists at zero new-code cost — but it is currently used only for
  teacher/student *function* drift of TF, not for per-Gaussian maturity.
* **No per-Gaussian geometry snapshot exists today.** `ChangeCapture.old_state`
  ([include/change_management.h:84](include/change_management.h#L84)) is a
  whole-population snapshot taken only around loop-closure / scale-refinement
  events, not per cycle. `FrozenTeacherTargetCache` /
  `HistoricalProbeCache` cache network outputs, not Gaussian geometry.

---

## 3. Current Protection / Mature implementation

### 3.1 Per-Gaussian Protection tensors

`GaussianProtection` ([include/gaussian_protection.h:18-46](include/gaussian_protection.h#L18-L46)):

| tensor | shape | dtype | meaning |
|---|---|---|---|
| `birth_iter` | `[N]` | int64 | mapping iteration of creation |
| `optimized_count` | `[N]` | int32 | number of Full Mapping evidence commits |
| `unique_keyframe_count` | `[N]` | int32 | saturated distinct-KF support count |
| `supporting_keyframe_ids` | `[N, unique_keyframe_ready]` | int64 | the actual distinct KF ids, `-1` padding |
| `cached_uniform` | `[N]` | float32 | per-row frozen Bernoulli uniform |
| `transitions` | `[5]` | int64 | state-transition counters |
| `pending_visibility_` (private) | `[V]` | int64 | Full-render visible indices inside a transaction |

`ElasticBudgetManager::uniform_` ([include/elastic_budget.h:79](include/elastic_budget.h#L79)) is a
*second* uniform buffer, but it is aliased to `protection_.cached_uniform` via
`useUniform(...)` at every call site ([gaussian_mapper.cpp:2879](src/gaussian_mapper.cpp#L2879),
[2138](src/gaussian_mapper.cpp#L2138), [2413](src/gaussian_mapper.cpp#L2413)).

### 3.2 Age / birth, seen counters, unique-KF metadata

* Age: `age = (iteration - birth_iter).clamp_min(0)` — [gaussian_protection.h:78](include/gaussian_protection.h#L78).
* Seen counter: `optimized_count`, incremented **only** in
  `finishFullMapping(committed=true, ...)` for the rows in the pending Full
  visibility ([gaussian_protection.h:141-159](include/gaussian_protection.h#L141-L159) CPU,
  [protection_evidence.cu:30-51](src/protection_evidence.cu#L30-L51) CUDA).
  LoD/auxiliary optimizers cannot increment it (test:
  [gaussian_protection_test.cpp:27-32](tests/gaussian_protection_test.cpp#L27-L32)).
* **Unique-KF metadata is still very much present**: `unique_keyframe_count`,
  `supporting_keyframe_ids`, and the CUDA kernel that maintains them
  ([protection_evidence.cu:39-43](src/protection_evidence.cu#L39-L43)). It is
  persisted into the PLY as `supporting_keyframe_id_<k>_low/high`
  ([gaussian_model.cpp:1135-1173](src/gaussian_model.cpp#L1135-L1173)) and read
  back by the Python evaluator ([python/selector.py:89-99](python/selector.py#L89-L99)).

### 3.3 Hard / Cold / Mature masks and the maturity condition

There is **no "Hard" or "Cold" mask in the current code** (a grep for
`hard_mask`/`cold` over `python/`, `src/`, `include/` finds only a negative
assertion in [tests/elastic_budget_python_test.py:76](tests/elastic_budget_python_test.py#L76)
and third-party OpenCV). The current equivalent is the 4-state enum
`GaussianProtectionState { WARMUP, UNDER_OBSERVED, STALE, MATURE }`
([include/gaussian_protection.h:9](include/gaussian_protection.h#L9)). "Protected"
is exactly `~mature`.

Maturity and the hard mask ([gaussian_protection.h:74-97](include/gaussian_protection.h#L74-L97)):

```cpp
age_confidence        = (age / age_ready).clamp(0,1)
optimization_confidence = (optimized_count / optimization_ready).clamp(0,1)
keyframe_confidence   = (unique_keyframe_count / unique_keyframe_ready).clamp(0,1)
maturity = min(min(age_conf, optimization_conf), keyframe_conf)   // continuous but input-driven
need     = 1 - maturity
mature   = (age >= age_ready) & (optimized_count >= optimization_ready)
                             & (unique_keyframe_count >= unique_keyframe_ready)
state: default UNDER_OBSERVED; age >= stale_age -> STALE; age < age_ready -> WARMUP; mature -> MATURE
```

Defaults ([include/elastic_budget_allocation.h:7-20](include/elastic_budget_allocation.h#L7-L20), mirrored in
[python/elastic_budget.py:5-6](python/elastic_budget.py#L5-L6)): `age_ready=100`,
`optimization_ready=80`, `unique_keyframe_ready=4`, `stale_age=1000`,
`max_budget_fraction=0.35`.

**This is precisely the age-based + optimization-count + unique-KF maturity the
future design wants to remove.** Note `maturity` already exists as a continuous
`[0,1]` tensor, so the *naming* is reusable but its three inputs all violate the
target design.

### 3.4 Stale state, cached random state, budget allocation

* Stale: diagnostic only — `state == STALE` feeds reporting
  ([gaussian_protection.h:177-193](include/gaussian_protection.h#L177-L193)) and nothing branches on it for
  behaviour; STALE rows are simply `~mature` and therefore Protected.
* Cached random state: `cached_uniform` (float32, `[N]`, resampled only on
  `reset`/`append`) [gaussian_protection.h:36](include/gaussian_protection.h#L36), [58](include/gaussian_protection.h#L58).
  Uniform-based admission gives exact **ratio nesting**:
  `ratio₁ < ratio₂ ⟹ admitted(ratio₁) ⊆ admitted(ratio₂)`
  (test: [elastic_budget_test.cpp:26-35](tests/elastic_budget_test.cpp#L26-L35)).
* Budget allocation: see section 5.

### 3.5 Exact code path: Gaussian state → … → final LoD mask

```
GaussianModel::xyz_/scaling_/rotation_                    (geometry)
   │
   ├─ protection_.advance(mapping_iter+1)                 gaussian_mapper.cpp:1420
   │     → advanceProtectionClockCUDA  (transitions only) protection_evidence.cu:14-29
   │
Full step only:
   ├─ protection_.beginFullMapping(render_pkg.visibility_indices)      gaussian_mapper.cpp:1462
   └─ protection_.finishFullMapping(gaussian_committed, kfid)          gaussian_mapper.cpp:1635
         → commitProtectionEvidenceCUDA → optimized_count++, unique_keyframe_count++,
           supporting_keyframe_ids[], transitions          protection_evidence.cu:30-51
   │
LoD step:
   ├─ GaussianModel::getSelectorMatureMask()  == protection_.matureMask()   gaussian_model.cpp:678-680
   │     = (mapping_iteration - birth_iter >= age_ready)
   │       & (optimized_count >= optimization_ready)
   │       & (unique_keyframe_count >= unique_keyframe_ready)               gaussian_protection.h:93-97
   │
   ├─ protection classification:  classify() -> {maturity, need, state, mature}   gaussian_protection.h:74-92
   │
   ├─ protection admission (budget):  elastic_budget_manager_.plan(pop, ratio, cfg)
   │     selected = ~mature & (cached_uniform < probability)                elastic_budget.h:46-77
   │
   ├─ Selector on mature rows only:  selector_network_->forwardHard(
   │        xyz[mature_indices], rotation[mature_indices], scaling[mature_indices], ratio, tau, &keep_probability)
   │                                                                       gaussian_mapper.cpp:2914-2919
   │     hard mask scattered to full N: zeros[N].index_copy(0, mature_indices, hard_selected)
   │                                                                       gaussian_mapper.cpp:2922-2923
   │
   └─ final mask feeding the compact LoD render:
        selector_mask = protected_f + (1 - protected_f) * selector_hard_mask
        compact_to_global = nonzero(selector_mask.detach() > 0)            gaussian_mapper.cpp:2948-2977
```

The Python evaluator reproduces a simplified version of this path:
`classify_protection` → `protection_budget` → `SelectorNetwork.forward`
([python/selector.py:109-142](python/selector.py#L109-L142),
[python/elastic_budget.py:17-59](python/elastic_budget.py#L17-L59)). It does **not**
reproduce the ratio-loss / GI-loss coupling.

### 3.6 Tensors/classes that become obsolete under a continuous maturity

Would become dead or trivially derivable:

* `GaussianProtectionState` enum + `ProtectionClassification::state`
  ([include/gaussian_protection.h:9,13](include/gaussian_protection.h#L9)) and the whole
  transitions reporting block ([gaussian_protection.h:177-196](include/gaussian_protection.h#L177-L196)).
* `optimized_count`, `unique_keyframe_count`, `supporting_keyframe_ids`
  ([gaussian_protection.h:22](include/gaussian_protection.h#L22)) and the CUDA evidence
  kernel [src/protection_evidence.cu](src/protection_evidence.cu) entirely;
  `protection_.advance/beginFullMapping/finishFullMapping`
  ([gaussian_protection.h:98-161](include/gaussian_protection.h#L98-L161)).
* `birth_iter` + `mapping_iteration` (age) — the whole clock
  ([gaussian_protection.h:20-21](include/gaussian_protection.h#L20-L21)).
* `ProtectionConfig::{age_ready, optimization_ready, unique_keyframe_ready, stale_age}`
  ([include/elastic_budget_allocation.h:8-11](include/elastic_budget_allocation.h#L8-L11)).
* The corresponding PLY/JSON schema: `selector_metadata.ply` fields
  `optimized_count`, `unique_keyframe_count`, `supporting_keyframe_id_*`,
  `birth_iter_*`, and the `protection_state.json` transition keys read in
  [gaussian_model.cpp:1075-1133](src/gaussian_model.cpp#L1075-L1133) and
  [python/selector.py:80-107](python/selector.py#L80-L107).
* `elastic_budget_allocation_test`, parts of `gaussian_protection_test`,
  `elastic_budget_python_test`.
* `exist_since_iter_` is already near-dead (only used by
  `scaledTransformVisiblePointsOfKeyframe`'s stability check,
  [gaussian_model.cpp:442-449](src/gaussian_model.cpp#L442-L449)) — a second
  age-like signal worth knowing about.
* `last_elastic_supervised_iter_` is **only** defensively re-initialized
  ([gaussian_mapper.cpp:2449-2454](src/gaussian_mapper.cpp#L2449-L2454)); it is written
  every LoD step but never consumed.

---

## 4. Selector interface

### 4.1 What it produces before hard sampling

`GumbelNetworkImpl` ([improvements/selector/gumbel_network.h](improvements/selector/gumbel_network.h)) is a
4-branch MLP (pos 3, rotation 4, scale 3, ratio 1 → 4×32 → Linear(128,2)).
**Two-class logits** are the primitive; a `[N,2]` categorical with class 1 = keep.

Public API:

| function | returns | line |
|---|---|---|
| `forward(pos,rot,scale,target_ratio,tau)` | tuple `{logits[:,1], hard_st[:,1], soft[:,1]}` | [23-35](improvements/selector/gumbel_network.h#L23-L35) |
| `forwardHard(pos,rot,scale,float ratio,tau,keep_probability*)` | `[N]` hard straight-through keep sample; optionally writes `[N]` `p = softmax(logits)[:,1]` | [38-49](improvements/selector/gumbel_network.h#L38-L49) |
| `forwardKeepLogit(...)` | `[N]` scalar keep **logit** = `(logit₁ − logit₀)/tau` | [53-60](improvements/selector/gumbel_network.h#L53-L60) |
| `forwardHardWithUniform(...)` | `[N]` hard sample from a **supplied** uniform, same Bernoulli law, ST backward | [65-78](improvements/selector/gumbel_network.h#L65-L78) |
| `forwardProbability(pos,rot,scale,ratio,tau)` | `[N]` **scalar keep probability** `softmax(logits/tau)[:,1]` | [80-95](improvements/selector/gumbel_network.h#L80-L95) |

`gumbelSoftmax` hard branch = `one_hot(argmax) − soft.detach() + soft`
([gumbel_network.h:127-139](improvements/selector/gumbel_network.h#L127-L139)).

Shapes in the normal training call: `hard_selected` is `[mature_count]` (or `[N]`
in the all-mature case), `keep_probability` `[mature_count]`, `selector_hard_mask`
`[N]`, `selector_mask` `[N]`, `compact_to_global` `[K]`
([gaussian_mapper.cpp:2912-2971](src/gaussian_mapper.cpp#L2912-L2971)).

Keyframe-level entry for composition: `GaussianRenderInput::indexSelect(indices)`
([include/gaussian_renderer.h:53-59](include/gaussian_renderer.h#L53-L59)).

### 4.2 How the target ratio flows

```
selector_target_ratios_[ratio_index]
   ├─ Selector input:  time_emd(ratio)  -> logitsFor()          gumbel_network.h:106-118
   ├─ Gumbel sampling: gumbelSoftmax(logits, tau, hard=true)   gumbel_network.h:127-139
   ├─ Budget:          elastic_budget_manager_.plan(pop, ratio, protection_config_)
   │                     target.total = ratio * mature_count    elastic_budget.h:53
   │                     target.mature = clamp(total - protected_count, 0, mature_count)
   │                     ratio_loss = |E[keep] - target.mature| / max(1,total)
   │                                                             gaussian_mapper.cpp:2951-2952
   └─ Final mask:      compact_to_global = nonzero(selector_mask.detach() > 0)
                                                             gaussian_mapper.cpp:2969-2970
```

The ratio itself is the *only* conditioning signal for the network, and it is
also the *only* thing shaping the target count. Both go through the same
`selector_target_ratios_` vector. In `ultrafast_rotation` mode there is a second,
independent ratio consumer: the TF `applyResidual(..., target_ratio)`
([gaussian_mapper.cpp:3001-3005](src/gaussian_mapper.cpp#L3001-L3005)).

### 4.3 The five answers

1. **Is there already a scalar keep probability `p_selector` per Gaussian?**
   **Yes**, but only on the rows the Selector is evaluated on. In normal
   training `keep_probability` is `[mature_count]`
   ([gaussian_mapper.cpp:2918](src/gaussian_mapper.cpp#L2918)) and `selector_all`
   ([2912](src/gaussian_mapper.cpp#L2912)) is true only when
   `mature_count == N`. `forwardProbability` produces the same quantity on demand
   for an arbitrary index set ([historical_gradient_render.cpp:24-28](src/historical_gradient_render.cpp#L24-L28)
   shows the pattern).
2. **If not, closest equivalent?** `selector_hard_mask` (the straight-through
   `[N]` scatter, [gaussian_mapper.cpp:2922-2923](src/gaussian_mapper.cpp#L2922-L2923))
   and, through the *same* network in one extra forward, `forwardProbability`.
   The literal answer is: `p_selector` exists for the mature subset; for
   immature rows the current code has no probability at all (it hard-codes
   `0`).
3. **Exact insertion point for a maturity blend without rewriting the Selector.**
   One line, immediately after the scatter and before `protected_f` is built:
   ```
   // 2923  selector_hard_mask = zeros[N].index_copy(mature_indices, hard_selected)
   // 2924  expected_mature   = keep_probability.sum()
   // NEW:  p_sel_all = zeros[N].index_copy(0, mature_indices, keep_probability)  // [N] in [0,1]
   // NEW:  p_mix     = (1 - maturity) * target_ratio + maturity * p_sel_all
   // 2948  protected_f = budget.admitted.float()
   // 2949  selector_mask = protected_f + (1-protected_f) * selector_hard_mask
   ```
   Replace `selector_hard_mask` with a **hard sample drawn from `p_mix`** (e.g.
   `(cached_uniform < p_mix)` or `forwardHardWithUniform` with `p_mix` as the
   logit source) and change `expected_mature` from
   `keep_probability.sum()` to `p_mix.sum()`. Everything downstream
   (`compact_to_global`, `GaussianRenderInput::indexSelect`, TF mask, opacity
   multiply) already consumes whatever `[N]` mask arrives. **No Selector
   rewrite is needed.**
   Cheapest variant that avoids a new probability forward: evaluate the network
   on *all* N rows (`selector_all = true` already supported,
   [gaussian_mapper.cpp:2912-2919](src/gaussian_mapper.cpp#L2912-L2919)), i.e. drop the
   `index_select(0, mature_indices)` restriction. Cost: `N - mature_count` extra
   MLP rows per LoD step (32-wide, 4 branch nets) — negligible next to the render.
4. **Would blending interfere with the straight-through / Gumbel path?**
   Partially, and it must be done deliberately:
   * `p_selector` is produced from `logits.softmax(-1)` with **no** Gumbel noise
     ([gumbel_network.h:45-48](improvements/selector/gumbel_network.h#L45-L48)), so using it inside a
     blend does **not** break the `one_hot − soft.detach() + soft` gradient: the
     gradient of `p_mix` w.r.t. `soft_net` weights flows through `.softmax()`,
     which is a valid differentiable path.
   * The current ST guarantee is "forward value ∈ {0,1}, backward = soft
     relaxation". If `selector_mask` becomes a *continuous* `p_mix` and it is
     then multiplied into opacity ([gaussian_mapper.cpp:2977-2980](src/gaussian_mapper.cpp#L2977-L2980)),
     the forward value stops being binary and immature rows start contributing
     partial opacity. That is a **behavioural change**, not a gradient break, and
     it is exactly the intended handoff.
   * Recommended to keep the ST contract: sample a **hard** `{0,1}` from `p_mix`
     with a detached mask plus an ST correction
     (`hard − p_mix.detach() + p_mix`), mirroring
     `forwardHardWithUniform` ([gumbel_network.h:65-78](improvements/selector/gumbel_network.h#L65-L78)).
     Then `compact_to_global = nonzero(mask.detach() > 0)` still yields a clean
     discrete compact set and the render graph is unchanged in structure.
   * `maturity` and `target_ratio` must be **detached** in the blend — otherwise
     the ratio becomes a differentiable input to a graph that is also
     supervised by `ratio_loss`, creating a feedback loop.
   * The `ratio_loss` numerator must be updated in the same edit, or the loss
     will fight the blend (see section 5).
5. **Does the Selector require a Mature-only compact tensor?**
   **No.** Two facts settle this:
   * `selector_all = mature_count == num_gaussians`
     ([gaussian_mapper.cpp:2912](src/gaussian_mapper.cpp#L2912)) — the code already
     supports all-N input, and `logitsFor` broadcasts a single ratio embedding to
     any batch size ([gumbel_network.h:110-112](improvements/selector/gumbel_network.h#L110-L112)).
   * `historicalLoDInput` and `makeHistoricalEvaluation`
     ([historical_gradient_render.cpp:14-51](src/historical_gradient_render.cpp#L14-L51),
     [gaussian_mapper.cpp:2134-2155](src/gaussian_mapper.cpp#L2134-L2155)) show the same pattern
     parameterized by `mature_mask`/`mature_indices`, so a continuous path can be
     threaded through by passing `torch::ones`-like inputs.
   What **does** assume mature-only is the *scatter* (`index_copy` into a
   zero-initialized `[N]`) and `protected_mask = ~mature_mask`
   ([gaussian_mapper.cpp:2927](src/gaussian_mapper.cpp#L2927)) — that is the minimal
   compatibility surface.

---

## 5. Current budget control

Target: `active_gaussians ≈ ratio * N`.

```
elastic_budget_manager_.useUniform(gaussians_->protection_.cached_uniform)   gaussian_mapper.cpp:2879
population    = { mature_mask, need = classify().need * ~mature, mature_count }
                                                                             elastic_budget.h:38-45
target.total  = ratio * population.mature.numel()                            elastic_budget.h:53
demand        = sum(need)                                                    elastic_budget.h:57
protection_budget = min(demand, max_budget_fraction * target.total)          elastic_budget.h:60
probability   = need * (protection_budget / max(demand, 1e-12))              elastic_budget.h:61
admitted      = ~mature & (cached_uniform < probability)                     elastic_budget.h:65
target.protection = min(demand, frac * total)                                elastic_budget_allocation.h:26-31
target.mature     = clamp(total - protected_count, 0, mature_count)          elastic_budget_allocation.h:32-34
```

* **Target computation**: `ratio * N` (whole population, *not* mature count) at
  [elastic_budget.h:53](include/elastic_budget.h#L53).
* **Protection budget**: demand-driven, capped by `max_budget_fraction = 0.35`
  ([elastic_budget.h:60](include/elastic_budget.h#L60)). With `P = need` the exact
  solution needs no bisection — the comment at [elastic_budget.h:58-59](include/elastic_budget.h#L58-L59)
  records the `Bp ≤ D ⟹ λ ≤ 1` identity, verified by
  [elastic_budget_test.cpp:16-18](tests/elastic_budget_test.cpp#L16-L18).
* **Mature budget**: `clamp(total − actual_protected, 0, mature_count)`
  ([elastic_budget_allocation.h:32-34](include/elastic_budget_allocation.h#L32-L34)). It uses the **actual**
  Bernoulli sample, so admission noise cancels exactly
  ([elastic_budget_test.cpp:36-39](tests/elastic_budget_test.cpp#L36-L39)).
* **Count / ratio loss**: `ratio_loss = |expected_mature − target.mature| / max(1, target.total)`
  ([gaussian_mapper.cpp:2951-2952](src/gaussian_mapper.cpp#L2951-L2952)); weighted by
  `selector_ratio_loss_weight_ = 0.01` ([gaussian_mapper.cpp:3064-3067](src/gaussian_mapper.cpp#L3064-L3067)).
  It supervises **mature rows only** — proven by
  [elastic_budget_test.cpp:40-44](tests/elastic_budget_test.cpp#L40-L44).
* **Bernoulli admission**: yes — `uniform_ < probability`, one frozen uniform per
  row. This is what gives exact nesting across `0.01/0.05/0.10/0.15`.
* **Probability calibration**: `probability = need · (Bp / D)` — a single global
  scalar rescale that makes `Σpᵢ = Bp` exactly.
* **Top-K / Gumbel**: no Top-K anywhere. Gumbel is only the Selector's sampler
  ([gumbel_network.h:127-139](improvements/selector/gumbel_network.h#L127-L139)); the
  budget is enforced by φ + count loss, not by ranking. `torch::kthvalue` appears
  only in the GI teacher ([gi_teacher.cpp:64](src/gi_teacher.cpp#L64)).

**Is there a reusable scalar-bias / threshold / normalization mechanism that can
enforce `Σ pᵢ ≈ ratio * N` after maturity blending?**

**Yes — `ElasticBudgetPlan` is exactly that mechanism, and it is already generic
in `ratio`.** `plan(population, ratio, config)` takes a demand vector and returns
per-row `probability` with `Σ probability == min(demand, frac·ratio·N)` and a
matching `admitted` mask. To reuse it after blending, feed the blended
probability as the demand: `population` becomes
`{mature_mask, p_mix, mature_count}` instead of `{mature_mask, need·~mature, ...}`.
The same `plan()` is already called from three places — main LoD
([gaussian_mapper.cpp:2884](src/gaussian_mapper.cpp#L2884)), historical guard
([2153-2155](src/gaussian_mapper.cpp#L2153-L2155)), drift probe
([2417-2419](src/gaussian_mapper.cpp#L2417-L2419)) — so the normalization exists once
and is shared.

Caveat: `max_budget_fraction = 0.35` would then cap the *total* keep probability
at 0.35·ratio·N, which is not what a blended `p_mix` wants. The first version
should leave Protection admission untouched and only renormalize the Selector
side (see section P).

---

## 6. TF coupling to Mature / Protection

Configuration: `TransformFieldConfig::mature_only = true` and
`outside_identity = true` by default ([include/transform_field.h:28-29](include/transform_field.h#L28-L29)),
set from `TransformField.mature_only` / `TransformField.outside_identity`
([gaussian_mapper.cpp:648-650](src/gaussian_mapper.cpp#L648-L650),
[4330-4331](src/gaussian_mapper.cpp#L4330-L4331)). `fast.yaml` sets both to `1`.

**Which subset can enter TF (normal LoD path)** — [gaussian_mapper.cpp:2982-3013](src/gaussian_mapper.cpp#L2982-L3013):

```cpp
transform_mask = selector_mask.detach().index_select(0, compact_to_global) > 0;   // 2985-2986
if (mature_only)       transform_mask &= mature_mask.index_select(0, compact_to_global);   // 2987-2989
if (outside_identity)  transform_mask &= transform_field_->insideMask(selected_input.xyz);// 2990-2992
tf_local_indices = nonzero(transform_mask);                                       // 2993
transformed = applyResidual(xyz[tf_local], scaling[tf_local], rotation[tf_local], ratio);  // 3001-3005
selected_input.{xyz,scaling,rotation}.index_copy_(0, tf_local_indices, ...)       // 3006-3011
```

* `mature_only` is checked at **exactly one place** in the normal path,
  [gaussian_mapper.cpp:2987-2989](src/gaussian_mapper.cpp#L2987-L2989), plus three mirror
  sites: [historical_gradient_render.cpp:38-39](src/historical_gradient_render.cpp#L38-L39),
  [gaussian_mapper.cpp:2537](src/gaussian_mapper.cpp#L2537) (drift probe),
  and the config parse at [648-650](src/gaussian_mapper.cpp#L648-L650).
* **Protected Gaussians receive identity transform**: yes, effectively — they are
  excluded from `transform_mask`, so `applyResidual` is never called on them and
  their `xyz/scaling/rotation` in `selected_input` are the untouched master
  values. There is no explicit "apply identity" call; identity is the absence of
  a call. The same convention holds for `outside_identity`
  (`insideMask` excludes them from the copy).
* **Selected indices are constructed as compact-local indices**:
  `tf_local_indices = nonzero(mask over the compact population)`
  ([2993](src/gaussian_mapper.cpp#L2993)) — **not** global row ids. They index
  into `selected_input`, which came from
  `full_input.indexSelect(compact_to_global)` ([2976](src/gaussian_mapper.cpp#L2976)).
* **XYZ / scale / rotation compensation**:
  `TransformFieldImpl::applyResidual(xyz, world_scaling, world_rotation, ratio)`
  ([include/transform_field.h:52-56](include/transform_field.h#L52-L56)) returns
  `{xyz, scaling, rotation}` already in *activation* space, and the caller
  writes them back with `index_copy` on the compact input. The residual itself is
  `{delta_xyz, delta_log_scale, delta_rotation_raw}`
  ([include/transform_field.h:32-36](include/transform_field.h#L32-L36)) — so scale is
  added in log space.
* **Compact vs full-N**: TF operates on the **compact** population end to end.
  The whole `selected_render_pkg` render is over `selected_input` (compact), and
  `transform_count` is reported as `tf_transformed_gaussians` in the
  `[UltrafastRotation]` log line ([gaussian_mapper.cpp:1651](src/gaussian_mapper.cpp#L1651)).
  `GaussianModel`'s own TF-related transform ops
  (`applyScaledTransformation`, `scaledTransformVisiblePointsOfKeyframe`) are a
  *different* mechanism — SLAM loop/scale corrections, not TF (section 11).

### What happens if hard Mature / Protected classification is weakened

`mature_mask` enters TF only through the `&` at
[2989](src/gaussian_mapper.cpp#L2989). If hard classification is replaced by a
continuous `m_i`, the two viable minimal compatibility choices are:

* `mature_mask := (m_i > τ)` with `τ` = the existing hard threshold's analogue —
  **one line**, TF behaviour preserved bit-for-bit for a suitable τ.
* `mature_mask := ones` (all Gaussians TF-eligible) — changes TF's domain and
  therefore needs re-tuning; **not** the first-version choice.

Additionally, `mature_mask` is used by TF-adjacent code outside the LoD render:

* TF AABB initialization and domain: [gaussian_mapper.cpp:1903-1913](src/gaussian_mapper.cpp#L1903-L1913)
  (`initializeTransformFieldIfNeeded` computes the AABB from `mature_xyz` only).
* Dynamic AABB pressure evaluation: [gaussian_mapper.cpp:2332-2359](src/gaussian_mapper.cpp#L2332-L2359).
* Change Distill capture: [gaussian_mapper.cpp:2239-2255](src/gaussian_mapper.cpp#L2239-L2255).
* Drift-stability probe transform mask: [gaussian_mapper.cpp:2537](src/gaussian_mapper.cpp#L2537).
* Historical guard LoD composition: [historical_gradient_render.cpp:38-39](src/historical_gradient_render.cpp#L38-L39).

**Minimum compatibility layer for a first continuous-maturity implementation:**

> Keep a function with the *exact* current signature and dtype
> `torch::Tensor GaussianModel::getSelectorMatureMask() const` returning
> `[N] bool`, but derive it from the new maturity scalar:
> `return maturity_ >= kMatureThreshold;` (bool, `.detach()`, no grad).
>
> That single shim keeps **all seven** call sites above compiling and behaving,
> including TF's `mature_only`, without touching TF, AABB, Change Distill, the
> historical guard, or the drift probe. `mature_indices = nonzero(...)` and the
> AABB-from-mature-points logic continue to work unchanged.

---

## 7. GI coupling to Mature

GI = `GlobalGIManager` (`streaming` mode, `fast.yaml: GlobalGI.mode: "streaming"`).

* **GI valid mask**: `GlobalGIManager::valid_` `[N]` bool, built in `aggregate()`
  and `updateBankBatch` ([global_gi.cpp:61](src/global_gi.cpp#L61), [86](src/global_gi.cpp#L86),
  [200](src/global_gi.cpp#L200), [243](src/global_gi.cpp#L243)); exposed by
  `validMask()` ([include/global_gi.h:50](include/global_gi.h#L50)). Semantics: finite
  importance from at least one view — **completely independent of maturity**.
* **GI teacher construction**: `GITeacherBuilder::build(combined_gi, mature_mask,
  protected_mask, ratio, gi_version, topology_version, known_mature_count,
  observed_mask)` — [include/gi_teacher.h:32-38](include/gi_teacher.h#L32-L38),
  [gi_teacher.cpp:6-67](src/gi_teacher.cpp#L6-L67). Inside:
  ```
  supervised = mature & ~protected & observed_mask      gi_teacher.cpp:31-37
  target_k   = floor(ratio * mature_count)              gi_teacher.cpp:46-47
  percentile on supervised scores (strict >)            gi_teacher.cpp:59-66
  label      = [N] float32, 0 outside supervised        gi_teacher.cpp:51-52,66
  ```
  `ratio` here is `budget.mature_ratio = target.mature / mature_count`
  ([elastic_budget.h:75](include/elastic_budget.h#L75)) — **it is derived from the
  hard mature pool**.
* **GI loss**: `gi_loss = l1(selector_hard_mask[supervised_mask],
  teacher.label[supervised_mask])`, added as
  `selector_gi_loss_weight_ * ratio_weight * gi_loss`
  ([gaussian_mapper.cpp:2957-2965](src/gaussian_mapper.cpp#L2957-L2965),
  [3069-3075](src/gaussian_mapper.cpp#L3069-L3075)). Weight
  `Selector.gi_loss_weight = 1.0` ([fast.yaml:28](cfg/selector/fast.yaml#L28)),
  `Importance.ratio_aware_weight = 0` so `ratio_weight == 1.0f`.
* **Mature-only filtering**: `supervised = mature ∧ ¬protected ∧ observed`
  ([gi_teacher.cpp:31-37](src/gi_teacher.cpp#L31-L37)). GI labels exist *only* on
  mature, non-protected, observed rows; `protected_mask = ~mature_mask`
  ([gaussian_mapper.cpp:2927](src/gaussian_mapper.cpp#L2927)), so in fact
  `supervised = mature ∧ observed` today.
* **Tensor shapes**: `combined_gi`/`importance_` `[N]` float32; `mature_mask`,
  `protected_mask`, `observed_mask`/`validMask()` `[N]` bool; `teacher.label` `[N]`
  float32; `supervised_mask` `[N]` bool. All validated in
  [gi_teacher.cpp:13-38](src/gi_teacher.cpp#L13-L38) and
  [global_gi.cpp:261-265](src/global_gi.cpp#L261-L265).
* **Other mature consumers in the GI path**:
  `global_gi_->recordGuidance(mature, loss_enabled)` at
  [gaussian_mapper.cpp:1338-1342](src/gaussian_mapper.cpp#L1338-L1342) and
  [3940-3944](src/gaussian_mapper.cpp#L3940-L3944) (statistics only —
  [global_gi.cpp:279-284](src/global_gi.cpp#L279-L284)); `refreshRecentWindowContribution`
  uses the recent-window `observedMask`, not maturity
  ([gaussian_mapper.cpp:1453-1461](src/gaussian_mapper.cpp#L1453-L1461)).

**Would removing the old Mature pool break GI?**

* `GITeacherBuilder::build` **throws** on mismatched lengths or non-1-D inputs
  ([gi_teacher.cpp:13-22](src/gi_teacher.cpp#L13-L22)) and on a ratio outside
  `[0,1]` ([23-24](src/gi_teacher.cpp#L23-L24)). So `mature_mask` must remain a
  `[N]` bool tensor on the same device as `importance_`.
* `budget.mature_ratio` is `0` when `mature_count == 0`
  ([elastic_budget.h:75](include/elastic_budget.h#L75)), and `target_k` is
  `floor(0 * 0) = 0`; `supervised_k == 0` short-circuits at
  [gi_teacher.cpp:54](src/gi_teacher.cpp#L54), so an all-immature population is
  already a handled degenerate case.
* **Minimal compatibility mask required temporarily: exactly the same
  `mature_mask` shim as section 6** — a `[N]` bool derived from `m_i > τ`. No GI
  code changes are needed for the first version. The one behavioural note: with
  a continuous `m_i`, the *effective* GI supervision population becomes a
  thresholded subset of the maturity field, so `target_k` and the strict
  percentile ([gi_teacher.cpp:59-66](src/gi_teacher.cpp#L59-L66)) will follow the
  threshold, not the continuum. That is acceptable while GI behaviour is
  intentionally left unchanged.

---

## 8. Available "touched / valid update" signal

| candidate | available in LoD step? | available in Full step? | notes |
|---|---|---|---|
| `RenderPackage::visibility_indices` | yes, but **compact-local** | yes, **global** | [gaussian_renderer.cpp:177-183](src/gaussian_renderer.cpp#L177-L183): `(radii > 0).nonzero()`, empty in `Inference` mode |
| `radii` | yes (compact) | yes (full N) | `[N]` float, `radii > 0` ⇔ visible |
| `render_pkg.viewspace_points` | yes (compact) | yes (full N) | `[N,3]` with `.grad()` |
| gradient-valid mask | implicit | implicit | `param.grad().defined()` per group, used at [gaussian_mapper.cpp:1512-1524](src/gaussian_mapper.cpp#L1512-L1524) |
| optimizer state (`exp_avg`) | yes | yes | `optimizer_->state()[impl]`, not a mask |
| selected LoD indices | yes | n/a | `compact_to_global` [2969-2970](src/gaussian_mapper.cpp#L2969-L2970); `result.supervised_indices` = compact→global mapped visible set [3050-3053](src/gaussian_mapper.cpp#L3050-L3053) |
| `last_elastic_supervised_iter_` | written every LoD step | no | `updateElasticSupervision` [gaussian_model.cpp:682-692](src/gaussian_model.cpp#L682-L692); stored as int64, written only for supervised compact rows, never read |
| Conflict-Replay LoD supervision | — | — | `trainJointReplay` also calls `updateElasticSupervision(lod.supervised_indices, ...)` at [gaussian_mapper.cpp:2825-2826](src/gaussian_mapper.cpp#L2825-L2826); that path can move Gaussians too, so a touched mark is needed there as well |
| `protection_.pending_visibility_` | no | yes | private, consumed and cleared by `finishFullMapping` |
| `max_radii2D_` / `xyz_gradient_accum_` / `denom_` | not updated in LoD | updated in Full | densification statistics, Full only by design (test [ultrafast_mapping_test.cpp:158-159](tests/ultrafast_mapping_test.cpp#L158-L159)) |
| `exist_since_iter_` | no | no | only used for the loop-closure stability check |

Answers:

* **Can `touched_in_cycle[N]` be updated without new rendering work?**
  **Yes.** The Full render already produces `render_pkg.visibility_indices` in
  global row ids, and it is already handed to
  `protection_.beginFullMapping(...)` at
  [gaussian_mapper.cpp:1462](src/gaussian_mapper.cpp#L1462). A three-line
  `touched.index_fill_(0, render_pkg.visibility_indices, true)` (or a `bool`
  scatter) is the entire cost. It is the **same tensor** that already feeds the
  evidence transaction, so no extra render, no extra sync.
* **Can it stay on GPU?** **Yes.** `visibility_indices` is a CUDA int64 tensor
  ([gaussian_renderer.cpp:180](src/gaussian_renderer.cpp#L180)) and index_fill is a
  device op. The only reason to touch the host is the existing
  `full_loss.item()` at [1553](src/gaussian_mapper.cpp#L1553), which already
  synchronizes — a `touched.any()` read is unnecessary.
* **Exact mask to reuse:** `render_pkg.visibility_indices` from the Full render
  ([gaussian_mapper.cpp:1447-1450](src/gaussian_mapper.cpp#L1447-L1450), produced at
  [gaussian_renderer.cpp:179-180](src/gaussian_renderer.cpp#L179-L180)). For LoD steps the
  equivalent is `lod.supervised_indices`
  ([gaussian_mapper.cpp:3050-3053](src/gaussian_mapper.cpp#L3050-L3053)), already computed and
  already passed to `updateElasticSupervision` — modulo the
  `record_supervision` flag.
* **Does Full visibility represent all meaningful Gaussian updates?**
  **No, not by itself.** Two gaps:
  1. **LoD optimizer steps do change Gaussian attributes in the running config**
     — `Selector.detach_gaussian_parameters: 0` ([fast.yaml:34](cfg/selector/fast.yaml#L34)), and
     `accumulateLodGradients` is called with `rotation ? false : selector_detach...`
     ([gaussian_mapper.cpp:1490](src/gaussian_mapper.cpp#L1490)), so for mode 1 the
     Gaussian parameters are *not* detached and the LoD `backward()` accumulates
     real Gaussian gradients ([2895-2911](src/gaussian_mapper.cpp#L2895-L2911)). Those rows
     are moved by the LoD commit at [1551](src/gaussian_mapper.cpp#L1551).
  2. A row can be modified by an LoD step and be frustum-culled in the very next
     Full render, in which case Full visibility would mark it untouched even
     though it moved.
  → **Recommendation: mark `touched = true` from both.**
  `touched.index_fill_(0, lod.supervised_indices, true)` on LoD phases
  (cheap: the tensor already exists) **and**
  `touched.index_fill_(0, render_pkg.visibility_indices, true)` on Full (free).
  Clearing happens once per cycle at the Full boundary (section 9), which is the
  same place the drift is read.
* Note that visibility is strictly weaker than "received gradient":
  `visibility_indices = (radii > 0).nonzero()` and the opacity multiply at
  [gaussian_mapper.cpp:2980](src/gaussian_mapper.cpp#L2980) /
  [gaussian_renderer.cpp:60-61](src/gaussian_renderer.cpp#L60-L61) mean a masked-out row
  has zeroed opacity. Using `radii > 0` as the touch signal therefore slightly
  over-counts rows that were in-frustum but contributed nothing to the loss. The
  `protect`-free alternative with tighter semantics is
  `p.grad().defined()` per row, which costs a `[N]` bool per attribute group and
  is not currently materialized.

---

## 9. Full-to-Full snapshot placement

### 9.1 Is the desired order compatible?

Desired:
```
previous Full snapshot → 4 LoD updates → Full render → Full optimizer step
   → compute cycle drift → update maturity → save new snapshot → densify / prune
```

**Yes, and the code already has exactly that shape.** The Full branch runs the
optimizer commit at [gaussian_mapper.cpp:1551](src/gaussian_mapper.cpp#L1551) and
densify/prune immediately afterwards at [1575-1612](src/gaussian_mapper.cpp#L1575-L1612),
with nothing in between except bookkeeping that does not touch Gaussian topology
(kfs_loss / EMA at [1552-1556](src/gaussian_mapper.cpp#L1552-L1556), optional image
record at [1558-1561](src/gaussian_mapper.cpp#L1558-L1561), densify deadline observation
at [1568-1574](src/gaussian_mapper.cpp#L1568-L1574)).

### 9.2 Specific determinations

* **Drift before or after `optimizer.step()`?** **After.** The desired ordering
  is the correct one and the only one that makes the quantity meaningful: the
  snapshot must be taken at a fully-committed parameter state. Taking it before
  the step would measure half-applied Adam updates and would double-count the
  optimizer step into the drift. Concretely: snapshot in cycle *k* is written at
  the Full step of cycle *k*, after line 1551; drift in cycle *k+1* is measured
  at the Full step of cycle *k+1*, after line 1551, against that snapshot.
* **Does densify/prune happen immediately after `optimizer.step()`?**
  **Yes**, at [1591-1598](src/gaussian_mapper.cpp#L1591-L1598) (`gaussians_->densifyAndPrune`),
  gated by `full_step && topology_window` and the deferred
  `ultrafast_densify_task_`. Opacity reset follows at
  [1605-1611](src/gaussian_mapper.cpp#L1605-L1611).
* **Can the maturity update be inserted before topology change?**
  **Yes, and it must be.** Both `densifyAndPrune` and `resetOpacity` eventually
  call `prunePoints` / `densificationPostfix`
  ([gaussian_model.cpp:608-676](src/gaussian_model.cpp#L608-L676),
  [694-782](src/gaussian_model.cpp#L694-L782)) which **replace every parameter tensor
  identity** and **reindex every `[N]`-shaped auxiliary tensor**. Any `[N]` tensor
  written before topology changes would be reindexed by
  `protection_.prune` / `protection_.append` — which is fine if the new
  maturity tensors are added to those two functions, and catastrophic if they are
  not.
* **Are parameters still valid and index-aligned at that point?** **Yes.**
  After `optimizer_->step()` (or `stepWithHistoricalGuard`, which commits
  through `tfCudaAdamMixedStep` / `GradientConstraintManager::step`), all six
  master tensors are freshly written `[N]` tensors and no index set has changed
  since `beginFullMapping` at [1462](src/gaussian_mapper.cpp#L1462). Index
  alignment is guaranteed until the first `prunePoints`/`densificationPostfix`.
* **Other topology-changing operations between Full steps?** Only these:
  * `evaluateDynamicAABB` → `captureGeometryChange`/`finishGeometryChange`
    ([gaussian_mapper.cpp:1321-1326](src/gaussian_mapper.cpp#L1321-L1326),
    [2321-2387](src/gaussian_mapper.cpp#L2321-L2387)) — **does not** change Gaussian
    count or order; it resamples the TF grid domain and may run Change Distill.
  * `evaluateDriftStability` ([1327-1332](src/gaussian_mapper.cpp#L1327-L1332)) —
    explicitly does not change master Gaussians
    ([drift_stability.h:15-16](include/drift_stability.h#L15-L16)).
  * `processConflictReplay` ([1293-1296](src/gaussian_mapper.cpp#L1293-L1296)) — can
    train, but is gated on `ConflictReplay` and runs only through
    `trainJointReplay`.
  * `increasePcd` from `LocalMappingBA`/`LoopClosingBA`
    ([gaussian_mapper.cpp:3174-3178](src/gaussian_mapper.cpp#L3174-L3178), [3273-3277](src/gaussian_mapper.cpp#L3273-L3277)) —
    **appends rows and can happen between Full steps**, i.e. asynchronously to the
    rotation. This is the one real hazard for a naive `[N]` snapshot.
  * The Full step's own densify/prune.
  In `ultrafast_rotation` mode, `combineMappingOperations` is not driven from
  `trainMappingIteration`, so the SLAM-side append can interleave with any phase.

### 9.3 Exact recommended insertion point

**Primary (recommended): inside the `commit_optimizer` lambda, immediately after
`gaussian_committed` is finalized and before `zero_grad`
([gaussian_mapper.cpp:1539-1547](src/gaussian_mapper.cpp#L1539-L1547)).**

```
1539   gaussian_committed &= has_gaussian_gradient;
1540   selector_committed = ...
1541   tf_committed = ...
1542   ...                          <-- INSERT: if (full_step && gaussian_committed)
1543   mode2_counts.gaussian_updates += ...
1544   mode2_counts.selector_commits += ...
1545   mode2_counts.tf_commits += ...
1546   if (rotation ? selector_committed : has_selector_update) ++selector_step_;
1547   gaussians_->optimizer_->zero_grad(true);
```

Why here and not at line 1658:

* `full_step` is in scope and is *the same expression* `ultrafast_rotation_.full()`
  used everywhere else, so the guard is exactly "this is a Full commit".
* `gaussian_committed` is in scope *only* here (it is a lambda-local at
  [1497](src/gaussian_mapper.cpp#L1497)); it tells us the step actually landed,
  which matters because `stepWithHistoricalGuard` can return `false` under
  `GradientConstraint` and because `hasNormalOptimizationBudget()` can silently
  skip the step ([1509](src/gaussian_mapper.cpp#L1509)).
* `zero_grad` at 1547 and `kfs_loss_` at 1552 do not touch parameters.
* `mutex_render_` is still held ([1398](src/gaussian_mapper.cpp#L1398)).
* Everything before the first topology call ([1591](src/gaussian_mapper.cpp#L1591)) is
  still ahead, so `[N]` is stable and `prev_*` can be written with a plain
  assignment/clone.

**Secondary (pure cycle-boundary bookkeeping, if you prefer to keep the lambda
untouched): at the very end of `trainMappingIteration`, right after
`ultrafast_rotation_.complete(...)` ([gaussian_mapper.cpp:1658-1659](src/gaussian_mapper.cpp#L1658-L1659)),
before line 1664.** At that instant the step counter has already flipped to the
next cycle and `gaussian_committed` is recoverable from
`normal_update_clock_` / `mode2_counts.gaussian_updates`. This is the true
"cycle ended" point and is where a `touched_in_cycle` **clear** naturally belongs.

**Recommended split:**

| action | where | why |
|---|---|---|
| compute drift, update `maturity`/`drift_ema`, write `prev_*` snapshot | after line 1545, guarded by `full_step && gaussian_committed` | pre-topology, `[N]` stable, commit flag in scope, mutex held |
| clear `touched_in_cycle` for the new cycle | after line 1659 | exact cycle boundary; the following cycle's LoD/Full marks will repopulate it |
| register `maturity`/`drift_ema`/`prev_*` in append/prune | `GaussianProtection::append/prune` or `GaussianModel::densificationPostfix/prunePoints` | see section 10 |

**Do not** put the measurement at line 1658 *if* you also want to avoid reading
parameters that densify may have already reindexed — at 1658 `[N]` may already be
the post-prune size. Drift math must run *before* line 1591.

---

## 10. Topology operations

### 10.1 `append from new keyframe`

* Entry: `GaussianModel::increasePcd(...)` — [gaussian_model.cpp:201-292](src/gaussian_model.cpp#L201-L292)
  (vector overload) and [293-397](src/gaussian_model.cpp#L293-L397) (tensor overload).
* It builds new rows and calls `densificationPostfix(..., AppendKind::Append, {})`
  ([gaussian_model.cpp:390-396](src/gaussian_model.cpp#L390-L396) area) → `protection_.append(count)` with
  no parents ([gaussian_protection.h:47-59](include/gaussian_protection.h#L47-L59)).
* Index change: **children appended at the tail**; existing indices unchanged.
* Parent survives: n/a.
* Driven from SLAM mapping operations at
  [gaussian_mapper.cpp:3174-3178](src/gaussian_mapper.cpp#L3174-L3178) and
  [3273-3277](src/gaussian_mapper.cpp#L3273-L3277).

### 10.2 `clone`

* `GaussianModel::densifyAndClone(grads, grad_threshold, scene_extent)` —
  [gaussian_model.cpp:843-867](src/gaussian_model.cpp#L843-L867).
* Selection: `‖grads‖ ≥ th` **and** `max(scale) ≤ percent_dense * extent`
  ([847-852](src/gaussian_model.cpp#L847-L852)).
* Appends `new_xyz = xyz_[mask]` (copies of the parents) at the tail, with
  `parent_indices = nonzero(selected_pts_mask)` ([854-866](src/gaussian_model.cpp#L854-L866)).
* Parent survives (no prune of the parent in clone).
* Index change: append only.

### 10.3 `split`

* `GaussianModel::densifyAndSplit(grads, grad_threshold, scene_extent, N=2)` —
  [gaussian_model.cpp:794-841](src/gaussian_model.cpp#L794-L841).
* Selection: `padded_grad ≥ th` **and** `max(scale) > percent_dense * extent`
  ([804-809](src/gaussian_model.cpp#L804-L809)).
* Children: `new_xyz = xyz_[mask].repeat({N,1})` plus `Σ^(1/2)·N(0,1)` isotropic
  jitter built from the full covariance
  ([811-817](src/gaussian_model.cpp#L811-L817)); scale is `log(σ/(0.8N))`
  ([818-820](src/gaussian_model.cpp#L818-L820)); rotation, features, opacity and
  `exist_since_iter` are inherited by repeat ([821-829](src/gaussian_model.cpp#L821-L829)).
* `parent_indices = nonzero(mask).repeat({N})` — so child *j* of parent *p* maps to
  `parent_indices[j] == p`, in **repeat order** (all 2 copies of parent 0, then
  all 2 of parent 1, …) ([830](src/gaussian_model.cpp#L830)). This matters for
  per-parent inheritance.
* **Parent is then removed**: `prune_filter = cat({selected_pts_mask, zeros(N·k)})`
  and `prunePoints(prune_filter)` ([835-840](src/gaussian_model.cpp#L835-L840)).
* Index change: append `N·k` children at the tail, then remove `k` parents →
  **arbitrary reindexing of the whole array**.

### 10.4 `prune`

* `GaussianModel::prunePoints(torch::Tensor& mask)` — [gaussian_model.cpp:608-676](src/gaussian_model.cpp#L608-L676).
  **Takes a removal mask** (`valid_points_mask = ~mask`), not survivor indices.
* `GaussianModel::densifyAndPrune(max_grad, min_opacity, extent, max_screen_size)` —
  [gaussian_model.cpp:869-890](src/gaussian_model.cpp#L869-L890) composes clone → split → build
  `prune_mask` from `opacity < min_opacity` and optionally
  `max_radii2D_ > max_screen_size` / `max(scale) > 0.1·extent`, then calls
  `prunePoints`. It ends with `c10::cuda::CUDACachingAllocator::emptyCache()`.
* Reindexed `[N]` tensors in `prunePoints`: all six optimizer params + their Adam
  state, `exist_since_iter_`, `protection_*` (via `protection_.prune`),
  `last_elastic_supervised_iter_`, `xyz_gradient_accum_`, `denom_`, `max_radii2D_`
  ([653-673](src/gaussian_model.cpp#L653-L673)). `protection_.prune(survivors)` internally takes
  `valid_points_mask` ([gaussian_protection.h:60-73](include/gaussian_protection.h#L60-L73)).
* Reindexed `[N]` tensors in `densificationPostfix` (append side): the same set
  plus a reset of `xyz_gradient_accum_`/`denom_`/`max_radii2D_` to zeros
  ([770-775](src/gaussian_model.cpp#L770-L775)) — **note: densification postfix zeroes the
  densification statistics entirely**, which is easy to miss.

### 10.5 Existing common topology synchronization hook

**Yes, and it is the right place.** `GaussianModel::densificationPostfix` and
`GaussianModel::prunePoints` are the *only* two functions that mutate Gaussian
row count, and both already:

1. call `protection_.append(count, parent_indices)`
   ([gaussian_model.cpp:765](src/gaussian_model.cpp#L765)) /
   `protection_.prune(valid_points_mask)`
   ([gaussian_model.cpp:663](src/gaussian_model.cpp#L663));
2. call `protection_.assertAligned(this->xyz_.size(0))`
   ([776](src/gaussian_model.cpp#L776), [672](src/gaussian_model.cpp#L672));
3. fire the GI callbacks
   ([778-781](src/gaussian_model.cpp#L778-L781), [674-675](src/gaussian_model.cpp#L674-L675)).

`GaussianProtection::append/prune` already implements **exactly** the three
parent-inheritance semantics you need for maturity:

* `append(count, parents)` — `unique_keyframe_count`/`supporting_keyframe_ids` are
  **inherited from parents** when `parents` is defined, reset to
  `0`/`-1` when it is not ([gaussian_protection.h:47-59](include/gaussian_protection.h#L47-L59)).
  For clone/split children that is the "region-level evidence" propagation; for
  new-KF points it is a clean birth.
  `cached_uniform` is always freshly randomized per appended row
  ([58](include/gaussian_protection.h#L58)).
* `prune(survivors)` — index-selects every metadata tensor
  ([60-73](include/gaussian_protection.h#L60-L73)) and additionally remaps
  `pending_visibility_` through the survivor mapping
  ([68-72](include/gaussian_protection.h#L68-L72)) — a ready-made pattern for
  remapping any pending index set.

**Cleanest existing place to synchronize the new tensors**
(`maturity[N]`, `drift_ema[N]`, `touched_in_cycle[N]`, `snapshot_valid[N]`,
`prev_full_*`):

> Add them as members of `GaussianProtection` and extend
> `GaussianProtection::reset()` ([gaussian_protection.h:27-39](include/gaussian_protection.h#L27-L39)),
> `append()` ([47-59](include/gaussian_protection.h#L47-L59)) and `prune()` ([60-73](include/gaussian_protection.h#L60-L73)).
> `GaussianModel::densificationPostfix` / `prunePoints` already call all three,
> so **no mapper-side topology hook is needed** and the alignment assertion
> `assertAligned` ([40-46](include/gaussian_protection.h#L40-L46)) can be extended to
> cover them in the same edit.
>
> **Sharp edge:** a split child is *not* a continuation of the parent's motion
> history. Children are jittered by `Σ^(1/2)·N(0,1)`
> ([gaussian_model.cpp:811-817](src/gaussian_model.cpp#L811-L817)), which is a large
> artificial "drift". Inherit `maturity` (via `parents`) but **reset
> `drift_ema`/`touched` for split children**, or mark them `snapshot_valid =
> false` so their first measured drift is discarded. Extending
> `AppendKind`-aware reset logic is required; `append()` already knows
> `parents` but not the kind, and `densificationPostfix` knows the kind but the
> kind is not currently forwarded to `protection_.append`
> ([gaussian_model.cpp:765](src/gaussian_model.cpp#L765)). The one-line fix is to forward
> `append_kind` and branch.
>
> **Second sharp edge:** `prunePoints` may be called *after* a snapshot was taken
> in the same Full step? No — in the current Full order the step (1551), the
> snapshot point, and densify (1591) are sequential, and prune only happens at
> 1591. But `combineMappingOperations` can call `increasePcd` **between** Full
> steps (section 11), which appends rows mid-cycle. `snapshot_valid[N]` is the
> mechanism to handle that: set `snapshot_valid = false` for appended rows
> (they have no previous Full state) and skip their drift for one cycle.

---

## 11. Loop closure / Scale refinement interaction

Both are driven from `GaussianMapper::combineMappingOperations()`
([gaussian_mapper.cpp:3134-3336](src/gaussian_mapper.cpp#L3134-L3336)), which drains
`pSLAM_->getAtlas()->hasMappingOperation()`. They run **outside** the normal
optimizer and **between** Full cycles (asynchronously to `ultrafast_rotation_`).

### Loop closing BA — [gaussian_mapper.cpp:3181-3283](src/gaussian_mapper.cpp#L3181-L3283)

* Entry: `captureGeometryChange(ChangeFlag::LOCAL_GEOMETRY)` ([3183](src/gaussian_mapper.cpp#L3183)).
* Per keyframe with a large pose delta:
  `gaussians_->scaledTransformVisiblePointsOfKeyframe(point_not_transformed_flags,
  diff_pose, world_view, full_proj, creation_iter, stable_num_iter_existence,
  num_transformed, loop_kf_scale)` — [gaussian_model.cpp:425-494](src/gaussian_model.cpp#L425-L494).
* What it modifies: **XYZ** (`points.index_copy_` then
  `replaceTensorToOptimizer(points, 0)`, [460](src/gaussian_model.cpp#L460), [490-492](src/gaussian_model.cpp#L490-L492)),
  **rotation** (`rotation_.index_copy_` + `replaceTensorToOptimizer(...,5)`, [461-466](src/gaussian_model.cpp#L461-L466)),
  **scale/log-scale** (`scaling_.index_copy_` + `replaceTensorToOptimizer(...,4)`, [467-471](src/gaussian_model.cpp#L467-L471)).
  So **yes to all three**.
* Inside `GaussianModel`, outside the optimizer's gradient path
  (`torch::NoGradGuard`, [436](src/gaussian_model.cpp#L436)), but it **does** rebuild
  optimizer param tensors and resets their Adam moments via
  `replaceTensorToOptimizer` ([586-606](src/gaussian_model.cpp#L586-L606)).
* Affected subset is decided by `markVisibleUntransformedPoints(points,
  point_not_transformed_flags, point_unstable_flags, world_view, full_proj)` —
  [gaussian_model.cpp:447-450](src/gaussian_model.cpp#L447-L450), implemented in
  [src/operate_points.cu](src/operate_points.cu). `point_unstable_flags` comes from
  `exist_since_iter_` ([442-445](src/gaussian_model.cpp#L442-L445)) — i.e. **young
  points are excluded from loop correction**, which is itself an age-based rule
  worth knowing about.
* New points from the loop are appended via `increasePcd` ([3273-3277](src/gaussian_mapper.cpp#L3273-L3277)).

### Scale refinement — [gaussian_mapper.cpp:3285-3329](src/gaussian_mapper.cpp#L3285-L3329)

* `captureGeometryChange(ChangeFlag::GLOBAL_GEOMETRY)` ([3289-3291](src/gaussian_mapper.cpp#L3289-L3291)).
* `gaussians_->applyScaledTransformation(s, T)` — [gaussian_model.cpp:398-423](src/gaussian_model.cpp#L398-L423):
  `correction.affects_all = true` ([404](src/gaussian_model.cpp#L404)), then
  `applyGaussianTransform({xyz_, scaling_, getRotationActivation()}, correction)`
  ([410-411](src/gaussian_model.cpp#L410-L411)) and `replaceTensorToOptimizer` for groups 0, 4, 5.
  **Modifies XYZ, scale, rotation for the whole population**
  (`affects_all = true`, `population_count = N`).
* `applyGaussianTransform` semantics — [include/gaussian_transform_ops.h:44-61](include/gaussian_transform_ops.h#L44-L61):
  ```
  xyz         = (xyz * scale) @ Rᵀ + t
  log_scaling = log_scaling + log(scale)
  rotation    = normalize(quat_mul(correction.rotation_wxyz, rotation))
  ```

### ChangeEvent / ChangeCoordinator exposure

* `ChangeCapture` already snapshots the **whole pre-correction state**:
  `old_state` (`GaussianAttrState`), `mature_mask_before`,
  `mature_indices_before`, `old_aabb`, `topology_version`, `correspondence_version`,
  `mapping_iter` — [include/change_management.h:78-90](include/change_management.h#L78-L90).
* `ChangeCapture.corrections` is `std::vector<CorrectionGroup>` where
  `CorrectionGroup : GaussianTransformResult` + `keyframe_id`
  ([include/change_management.h:64-90](include/change_management.h#L64-L90)). For each
  correction you therefore already have:
  * **transformed indices** — `GaussianTransformResult::indices`
    ([include/gaussian_transform_ops.h:30](include/gaussian_transform_ops.h#L30)) (`[K]`, or empty
    when `affects_all`), with `affects_all` for the global case;
  * **old/new transform** — `transform_4x4` (transposed SE3, row-vector convention)
    and `rotation_wxyz`, `scale`
    ([include/gaussian_transform_ops.h:37-40](include/gaussian_transform_ops.h#L37-L40));
  * **which attributes changed** — `changed_attributes` bitmask
    (`XYZ`/`SCALE`/`ROTATION`, [include/gaussian_transform_ops.h:8-13](include/gaussian_transform_ops.h#L8-L13));
  * **global vs local** — via `ChangeFlag::GLOBAL_GEOMETRY` vs
    `LOCAL_GEOMETRY` on `capture.flags` and via `affects_all`;
  * a change flag stream — `ChangeMapper::change_version_` and
    `correspondence_version_` ([include/change_management.h:86](include/change_management.h#L86),
    [gaussian_mapper.h:355-357](include/gaussian_mapper.h#L355-L357)).
* `finishGeometryChange` publishes CD/DSD side effects and bumps
  `change_version_` ([gaussian_mapper.cpp:2267-2319](src/gaussian_mapper.cpp#L2267-L2319)).

**Yes — there is already enough exposed state to transform a previous maturity
snapshot into the new coordinate frame.** For a `GaussianTransformResult` `C`:

```
xyz_new         = (xyz_old * C.scale) @ R(C)ᵀ + t(C)
log_scaling_new = log_scaling_old + log(C.scale)
rotation_new    = normalize(C.rotation_wxyz ⊗ rotation_old)
```
which is literally `applyGaussianTransform` ([gaussian_transform_ops.h:44-61](include/gaussian_transform_ops.h#L44-L61)).
So the correct handling of a known SLAM correction is: **apply the same
correction to `prev_full_xyz/log_scale/rot` and invalidate
`prev_full_valid` only where the correction is not exactly reproducible** —
instead of treating the correction as Gaussian instability. `affects_all` plus
`transform_4x4`/`rotation_wxyz`/`scale` is precisely the information needed, and
`old_state` in `ChangeCapture` is available as a fallback for the case where the
correction list is incomplete.

Additional safety mechanism already present: the TF grid domain is resampled and
Change Distill runs on the corrected population
([change_management.cpp](src/change_management.cpp), `finishGeometryChange`), and
`drift_stability_manager_->refreshReference(..., "geometry_change")` is called
when geometry mutated ([gaussian_mapper.cpp:2310-2318](src/gaussian_mapper.cpp#L2310-L2318)) —
i.e. the codebase already has a "a known global change invalidates the drift
baseline" convention that the new maturity system should mirror by resetting
`prev_*`/`drift_ema` for the corrected indices.

---

## 12. Learning rate and optimizer information

* **Optimizer type**: `SparseGaussianAdam : public torch::optim::Adam`
  ([include/gaussian_rasterizer.h:138-145](include/gaussian_rasterizer.h#L138-L145)), constructed in
  `GaussianModel::trainingSetup` ([gaussian_model.cpp:496-535](src/gaussian_model.cpp#L496-L535))
  as `new SparseGaussianAdam(Tensor_vec_xyz_, adam_options)` with
  `lr = 0.0` initially ([504-512](src/gaussian_model.cpp#L504-L512)), `eps = 1e-15`.
  The class only adds a special `step(visibility, N)` overload; normal training
  calls the inherited `step()` ([gaussian_mapper.cpp:1535](src/gaussian_mapper.cpp#L1535)).
* **Parameter groups** (0-5, fixed and hard-coded throughout the codebase):

  | idx | tensor | LR source | fast.yaml / tum_rgbd.yaml |
  |---|---|---|---|
  | 0 | `xyz_` | `position_lr_init * spatial_lr_scale_`, then `exponLrFunc(step)` | `0.00016 · S` |
  | 1 | `features_dc_` | `feature_lr` | `0.0025` |
  | 2 | `features_rest_` | `feature_lr / 20` | `0.000125` |
  | 3 | `opacity_` | `opacity_lr` | `0.05` |
  | 4 | `scaling_` | `scaling_lr` | `0.005` |
  | 5 | `rotation_` | `rotation_lr` | `0.001` |

  ([gaussian_model.cpp:511-528](src/gaussian_model.cpp#L511-L528); config values from
  [cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml:57-64](cfg/gaussian_mapper/RGB-D/TUM/tum_rgbd.yaml#L57-L64)).
* **LR schedule**: only group 0 is scheduled —
  `updateLearningRate(step)` → `exponLrFunc(step)` → `param_groups()[0].options().set_lr(lr)`
  ([gaussian_model.cpp:537-548](src/gaussian_model.cpp#L537-L548),
  [1336+](src/gaussian_model.cpp#L1336)). Groups 1-5 are **constant**, refreshed each
  mapping iteration from the mapper getters
  ([gaussian_mapper.cpp:1422-1429](src/gaussian_mapper.cpp#L1422-L1429),
  [4578-4596](src/gaussian_mapper.cpp#L4578-L4596)).
* **How current LR is retrieved**: directly from the optimizer:
  `static_cast<const torch::optim::AdamOptions&>(optimizer.param_groups()[g].options()).lr()`
  — see [gaussian_mapper.cpp:1514](src/gaussian_mapper.cpp#L1514),
  [1522](src/gaussian_mapper.cpp#L1522), [1530](src/gaussian_mapper.cpp#L1530),
  [1718](src/gaussian_mapper.cpp#L1718), [1990](src/gaussian_mapper.cpp#L1990). There is **no**
  `getLastLR()` accessor; the pattern is read-the-group-options.
  The Gaussian step index is `normal_update_clock_.gaussianLRStep(getIteration())`,
  which subtracts guarded-rejected updates
  ([include/gradient_constraint_clock.h:17-19](include/gradient_constraint_clock.h#L17-L19)).
* **Do different attributes have different LR?** **Yes** — six distinct values
  above, spanning `0.00016·S` (xyz) … `0.05` (opacity) at init: a ~300× spread.
  Any per-Gaussian drift→maturity threshold must therefore be either
  LR-normalized or empirically calibrated per attribute, since `scale`/`rotation`
  move far faster in parameter space than `xyz` for the same loss gradient.
* **Do Full and LoD share the same Gaussian optimizer and state?**
  **Yes — one optimizer, one state map, all sub-systems.** Evidence:
  * `gaussians_->optimizer_->zero_grad(true)` is called before both
    ([gaussian_mapper.cpp:1413](src/gaussian_mapper.cpp#L1413)) and after both
    ([1547](src/gaussian_mapper.cpp#L1547), [1697](src/gaussian_mapper.cpp#L1697)).
  * `ensureSharedMappingOptimizer()` **appends the Selector's and TF's parameters
    into the Gaussian optimizer's param group list** as groups
    `shared_selector_group_index_`, `shared_transform_mlp_group_index_`,
    `shared_transform_grid_group_index_`
    ([gaussian_mapper.cpp:1949-1979](src/gaussian_mapper.cpp#L1949-L1979)). `commitMode2Stage`
    asserts `gaussian == (g < 6)` ([gaussian_mapper.cpp:1713-1716](src/gaussian_mapper.cpp#L1713-L1716)).
  * So a single `optimizer_->step()` in a LoD phase advances Gaussian **Adam
    moments**, Selector moments, and TF moments **together** — see the comment
    and assertions in [tests/ultra_light_schedule_test.cpp:60-81](tests/ultra_light_schedule_test.cpp#L60-L81)
    and the Full-side freeze check in
    [tests/ultrafast_mapping_test.cpp:170-181](tests/ultrafast_mapping_test.cpp#L170-L181).
  * Consequence for maturity: a per-cycle drift already **includes the Adam
    first/second-moment effect** of the squeezed LR schedule. There is no
    natural-units drift available without explicitly dividing by the current
    per-group LR (available cheaply as shown above) or by the Adam update
    magnitude.
  * **Full does not update Selector/TF** in `ultrafast_rotation` mode: the
    preamble zeroes those gradients only on `full_step`
    ([gaussian_mapper.cpp:1414-1417](src/gaussian_mapper.cpp#L1414-L1417)) and the LoD
    branch is skipped, so groups ≥6 have no gradient and Adam skips them. This is
    verified by [tests/ultrafast_mapping_test.cpp:167-178](tests/ultrafast_mapping_test.cpp#L167-L178).
    A maturity update that runs on Full therefore cannot accidentally feed the
    Selector optimizer.

---

## 13. Memory cost of a Full snapshot

Per-row sizes for `N` Gaussians:

| tensor | dtype | bytes/row |
|---|---|---|
| `prev_xyz [N,3]` | float32 | 12 |
| `prev_scale [N,3]` | float32 | 12 |
| `prev_rotation [N,4]` | float32 | 16 |
| `maturity [N]` | float32 | 4 |
| `drift_ema [N]` | float32 | 4 |
| `touched [N]` | bool | 1 |
| `valid [N]` | bool | 1 |
| **total** | | **50 B/row** |

**Option A (all float32 where numeric):**

| N | bytes | MiB | MB |
|---|---|---|---|
| 100,000 | 5,000,000 | 4.77 | 5.00 |
| 250,000 | 12,500,000 | 11.92 | 12.50 |
| 500,000 | 25,000,000 | 23.84 | 25.00 |
| 134,485 (actual mode1 run) | 6,724,250 | 6.41 | 6.72 |
| 77,807 (actual full-only run) | 3,890,350 | 3.71 | 3.89 |

**Option B (snapshot tensors in float16, flags stay bool, maturity/EMA stay float32):**

| N | bytes | MiB | MB | saving |
|---|---|---|---|---|
| 100,000 | 3,000,000 | 2.86 | 3.00 | 40.0% |
| 250,000 | 7,500,000 | 7.15 | 7.50 | 40.0% |
| 500,000 | 15,000,000 | 14.31 | 15.00 | 40.0% |

Representative Gaussian counts are taken from
[tests/diagnostics/ultralight_mode2_20261014/final_mature_counts.json](tests/diagnostics/ultralight_mode2_20261014/final_mature_counts.json):
`total_gaussians = 134485` (mode1) and `77807` (original full-only).

**Context**: the current `GaussianProtection` metadata is
`birth_iter(8) + optimized_count(4) + unique_keyframe_count(4) + supporting_keyframe_ids(4×8=32) + cached_uniform(4) = 52 B/row`
([gaussian_protection.h:162-167](include/gaussian_protection.h#L162-L167) `metadataMB()`). So Option A
is essentially **size-neutral versus what the existing Protection metadata already
costs** (50 vs 52 B/row), and Option B is a 40% reduction on top of that.

**Would float16 materially harm drift measurement?**

* **Log-scale** — safe. `scaling_` holds `log σ`; for Gaussian splats in
  metre-scale scenes that is roughly `[-6, 0]`. float16 has ~10-bit mantissa, so
  the absolute quantum near 1.0 is `2⁻¹¹ ≈ 4.9e-4` and near 0.1 is `~6e-5`.
  A `|Δlog σ|` per cycle of interest is well above `1e-3`. Fine.
  **But do not store *physical* scale in float16**: near `σ ≈ 0.005-0.05 m`,
  float16's quantum is `~4e-6` at 0.005 (`relative ~8e-4`) and `~3e-5` at 0.05
  (`relative ~6e-4`) — comparable to a small but real per-cycle scale change, and
  the `exp()` round-trip adds error on top. Store `log σ`, not `σ`.
* **Rotation quaternion** — acceptable. Components live in `[-1,1]`; float16
  quantization gives an angular error of order `1e-3` rad ≈ `0.06°`, so a
  rotation-drift threshold must not be finer than ~`5e-3` rad. The current
  reference-weight idiom `1 - |<q_a,q_b>|` ([drift_stability.cpp:212](src/drift_stability.cpp#L212))
  is second-order in angle, so a `1e-3` rad error becomes `~5e-7` in the metric —
  harmless.
* **XYZ** — **the risky one**. Scenes here are ~1-10 m across (the TF AABB is
  built from mature XYZ extents, [gaussian_mapper.cpp:1912-1920](src/gaussian_mapper.cpp#L1912-L1920)).
  float16's absolute quantum near 1.0 is `~4.9e-4`, near 4.0 it is `~2e-3`, and
  near 0.1 it is `~6e-5`. A per-cycle position drift of interest is plausibly
  `1e-4 … 1e-3 m`. **float16 therefore risks the quantization noise being the
  same order as the signal**, worst for points far from the origin. Mitigations,
  in order of preference: (a) keep `prev_xyz` in float32 (Option A costs only 8 MB
  more than Option B at 500k); (b) store `prev_xyz - scene_center` in float16,
  which makes the values small and the quantization relative to the scene extent;
  (c) normalize drift by the scene AABB diagonal, which is needed anyway to make
  thresholds scene-independent.
* Recommendation for v1: **Option A** (float32 snapshot). The absolute cost at a
  realistic 134k Gaussians is **6.4 MiB** — negligible against the renderer,
  Adam moments (24 B/row × 6 groups ≈ 144 B/row) and the rasterizer's
  per-frame `[P, ...]` buffers. Optimize later, and only `prev_xyz` is genuinely
  at risk.

---

## 14. Existing tests / best place for tests

`enable_testing()` block: [CMakeLists.txt:243-450](CMakeLists.txt#L243-L450).

| area | existing test | file |
|---|---|---|
| **Protection** (state machine, evidence transaction, clone/split inheritance, prune correspondence, GI-invalid-mature) | `gaussian_protection` / `gaussian_protection_cuda` | [tests/gaussian_protection_test.cpp](tests/gaussian_protection_test.cpp) (registered [439-443](CMakeLists.txt#L439-L443)) |
| **elastic budget** (demand-weighted allocation, ratio nesting, actual-vs-expected mature quota, gradient locality, topology) | `elastic_budget` / `elastic_budget_cuda` | [tests/elastic_budget_test.cpp](tests/elastic_budget_test.cpp) ([435-444](CMakeLists.txt#L435-L444)) |
| **elastic budget (pure formula + Python parity)** | `elastic_budget_allocation`, `elastic_budget_python_test` | [tests/elastic_budget_allocation_test.cpp](tests/elastic_budget_allocation_test.cpp), [tests/elastic_budget_python_test.py](tests/elastic_budget_python_test.py) |
| **topology synchronization** | `gaussian_model_lifecycle` | [tests/gaussian_model_lifecycle_test.cpp](tests/gaussian_model_lifecycle_test.cpp) ([335-340](CMakeLists.txt#L335-L340)) |
| **ultrafast ratio scheduler** | `ultra_light_schedule`, `ultrafast_rotation` | [tests/ultra_light_schedule_test.cpp](tests/ultra_light_schedule_test.cpp), [tests/ultrafast_rotation_test.cpp](tests/ultrafast_rotation_test.cpp) ([299-307](CMakeLists.txt#L299-L307)) |
| **ultrafast 5-step integration** | `ultrafast_mapping_cuda` | [tests/ultrafast_mapping_test.cpp](tests/ultrafast_mapping_test.cpp) ([308-312](CMakeLists.txt#L308-L312)) |
| **mode2 split-ratio pipeline** | `ultralight_mode2_{config,full_parity,stages,clocks,special}` | [tests/ultralight_mode2_mapping_test.cpp](tests/ultralight_mode2_mapping_test.cpp) ([313-321](CMakeLists.txt#L313-L321)) |
| **GaussianModel append / prune / clone / split** | `gaussian_model_lifecycle` (+ the zero/split/prune section of mode2 parity) | as above |
| **Selector sampling / ST contract** | `compact_lod` / `compact_lod_cuda`, `historical_gradient_render` | [tests/compact_lod_test.cpp](tests/compact_lod_test.cpp), [tests/historical_gradient_render_test.cpp](tests/historical_gradient_render_test.cpp) |
| **GI teacher + mature coupling** | `gi_teacher`, `global_gi`, `gi_association_state` | [tests/gi_teacher_test.cpp](tests/gi_teacher_test.cpp), [tests/global_gi_test.cpp](tests/global_gi_test.cpp), [tests/gi_association_state_test.cpp](tests/gi_association_state_test.cpp) |
| **TF mature_only composition** | `transform_field`, `change_distill`, `drift_stability` | [tests/transform_field_test.cpp](tests/transform_field_test.cpp), [tests/change_distill_test.cpp](tests/change_distill_test.cpp), [tests/drift_stability_test.cpp](tests/drift_stability_test.cpp) |
| **LR / shared Adam** | `ultra_light_schedule` (`sharedAdamSkipsAuxiliaryState`) | [tests/ultra_light_schedule_test.cpp:39-82](tests/ultra_light_schedule_test.cpp#L39-L82) |

**Files that should be extended for the future implementation:**

1. **[tests/gaussian_protection_test.cpp](tests/gaussian_protection_test.cpp)** — the
   primary home for the new maturity state: reset/append/prune alignment,
   touched gating, slow-up / fast-down asymmetry, no-increase-without-touch.
   Its current assertions on `optimized_count`/`unique_keyframe_count` will need
   to be replaced, not deleted, since those tensors are the ones being removed.
2. **[tests/ultrafast_mapping_test.cpp](tests/ultrafast_mapping_test.cpp)** — the only
   test that exercises a real 5-phase `0.01 → 0.05 → 0.10 → 0.15 → Full` loop with
   real shared Adam. Add: exactly one maturity update per cycle; drift measured
   against the previous Full; a *new* value is written only after `step()`;
   densify/prune reindexes `maturity`/`drift_ema`/`touched` correctly. The
   existing checks at lines 184-186 and 156-182 are the template.
3. **[tests/elastic_budget_test.cpp](tests/elastic_budget_test.cpp)** and
   **[tests/elastic_budget_python_test.py](tests/elastic_budget_python_test.py)** —
   extend for `Σ p_mix ≈ ratio·N` after blending, and for the nesting property
   `ratio₁ < ratio₂ ⟹ keep(ratio₁) ⊆ keep(ratio₂)` still holding.
4. **[tests/gi_teacher_test.cpp](tests/gi_teacher_test.cpp)** — assert the
   thresholded compatibility mask keeps GI supervision on a subset and that a
   `[N]`-bool contract is still enforced.
5. **[tests/ultra_light_schedule_test.cpp](tests/ultra_light_schedule_test.cpp)** — the
   natural place for a pure unit test of the EMA / slow-up-fast-down recurrence
   if it is factored out of the mapper.
6. **[tests/historical_gradient_render_test.cpp](tests/historical_gradient_render_test.cpp)** —
   extend if `historicalLoDInput`'s `mature_mask` argument is re-parameterized.
7. **[tests/ultralight_mode2_mapping_test.cpp](tests/ultralight_mode2_mapping_test.cpp)** —
   extend to assert that mode 2 either participates in the cycle definition or is
   explicitly excluded (the mode2 loop is *not* Full-to-Full).

Also worth adding to **[CMakeLists.txt](CMakeLists.txt)** if a new
`tests/continuous_maturity_test.cpp` is created: follow the
`add_executable` + `target_link_libraries(... gaussian_mapper)` +
`target_compile_definitions(... CARTGS_TEST_SOURCE_DIR=...)` pattern used at
[CMakeLists.txt:308-315](CMakeLists.txt#L308-L315).

---

## A. Current ultrafast 5-step execution chain

```
trainForOneIteration()                                   gaussian_mapper.cpp:1219
 ├─ allowsNormalStep guard                               1221-1229
 ├─ increaseIteration(1)                                 1232
 ├─ useOneRandomSlidingWindowKeyframe()                  1233  (ONE KF for all 5 phases)
 ├─ replay / online-end / tail guards                    1246-1304
 ├─ trainMappingIteration(cam, enhanced=true, false)     1306
 │   ├─ lock mutex_render_                               1398
 │   ├─ SH + GI bank updates                             1401-1410
 │   ├─ zero_grad (Gaussian 1413; Sel/TF only if Full 1415-1416)
 │   ├─ protection_.advance(mapping_iter+1)              1420
 │   ├─ LR refresh                                       1422-1429
 │   ├─ if full_step:                                    1442
 │   │    render → visibility_indices                    1447-1452
 │   │    beginFullMapping(visibility_indices)           1462
 │   │    loss = L1/SSIM + opacity_reg                   1463-1472
 │   │    full_loss.backward()                           1474
 │   ├─ else: accumulateLodGradients(ratio_index=phase)  1487
 │   │    mature_mask / budget plan                      2868-2885
 │   │    Sel+TF zero_grad                               2889-2893
 │   │    forwardHard → selector_hard_mask               2914-2923
 │   │    GI teacher                                     2929-2946
 │   │    selector_mask                                  2948-2950
 │   │    compact_to_global                              2969-2970
 │   │    TF applyResidual on compact local indices      2982-3013
 │   │    LoD render                                     3039-3041
 │   │    updateElasticSupervision                       3054-3056
 │   │    selector_loss (+GI/ratio/TF-reg)               3058-3077
 │   │    selector_loss.backward()                       3084-3087
 │   ├─ commit_optimizer()                               1551  ONE shared Adam step
 │   │    stepWithHistoricalGuard / optimizer_->step     1533-1538
 │   │    zero_grad(true)                                1547
 │   ├─ if full_step: kfs_loss_ + ema_loss              1552-1556
 │   ├─ if full_step: record image                       1558-1561
 │   ├─ if full_step: densify stats                      1578-1586
 │   ├─ if full_step: densifyAndPrune / resetOpacity     1591-1611
 │   ├─ if full_step: finishFullMapping                  1635-1636
 │   └─ ultrafast_rotation_.complete(...)   <-- ++step   1658-1659
 └─ ++mapping_iter_, GI refresh, AABB, drift probes      1307-1346
```

Cycle: `step 0=0.01, 1=0.05, 2=0.10, 3=0.15, 4=Full`, then `step 5 = 0.01` of the
next cycle.

## B. Current Protection data flow

```
Full render visibility_indices
   → protection_.beginFullMapping        (pending_visibility_)
   → protection_.finishFullMapping(committed, kfid)
        optimized_count++ ; unique_keyframe_count++ (saturating)
        supporting_keyframe_ids[row, count] = kfid
   → protection_.advance(mapping_iter+1)  (transitions only)
   → matureMask()  = age≥100 ∧ opt≥80 ∧ uniqKF≥4
   → classify()    = {age_conf, opt_conf, kf_conf, maturity=min, need=1-m, state, mature}
   → ElasticBudgetManager::plan(mature, need·~mature, ratio, cfg)
        target = ratio·N ; Bp = min(D, 0.35·target) ; p = need·Bp/D
        admitted = ~mature ∧ (cached_uniform < p)
   → Selector forwardHard on mature rows → hard_selected + keep_probability
   → selector_hard_mask = zeros[N].index_copy(mature_indices, hard_selected)
   → selector_mask = admitted_f + (1-admitted_f)·selector_hard_mask
   → compact_to_global = nonzero(selector_mask > 0)   → render / TF / loss
```

## C. Exact Gaussian geometry tensor representations

| | tensor | shape | dtype | device | raw meaning | physical conversion |
|---|---|---|---|---|---|---|
| XYZ | `GaussianModel::xyz_` | `[N,3]` | float32 | CUDA | world metres | — (already physical) |
| scale | `GaussianModel::scaling_` | `[N,3]` | float32 | CUDA | **log σ** | `getScalingActivation() = exp(scaling_)` |
| rotation | `GaussianModel::rotation_` | `[N,4]` | float32 | CUDA | **unnormalized** `w,x,y,z` | `getRotationActivation() = normalize(rotation_)` |

* R/covariance utilities already exist: `getCovarianceActivation()` ([gaussian_model.cpp:81-99](src/gaussian_model.cpp#L81-L99)), `general_utils::build_rotation` ([general_utils.h:31](include/general_utils.h#L31)).
* Cheap physical scale without covariance: `exp(scaling_)`; cheaper still, use log-space directly.
* Existing cheap drift idiom: `GaussianAttrState{xyz, log_scaling, rotation}` + `{‖Δxyz‖/diag, |Δlog σ|, 1-|<q,q>|}` ([gaussian_transform_ops.h:15-27](include/gaussian_transform_ops.h#L15-L27), [drift_stability.cpp:209-213](src/drift_stability.cpp#L209-L213)).

## D. Selector probability / logit / Gumbel interface

* Primitive: `[N,2]` logits from `logitsFor(...)`; `soft_net_: Linear(128,2)`.
* `forwardHard` → `[N]` hard ST keep sample; optionally writes `[N]` `keep_probability = softmax(logits)[:,1]`.
* `forwardProbability` → `[N]` scalar keep probability (no Gumbel).
* `forwardKeepLogit` → `[N]` scalar keep logit.
* `forwardHardWithUniform` → `[N]` hard sample from a supplied uniform, ST backward.
* Gumbel hard = `one_hot(argmax(logits+noise)) − soft.detach() + soft`.
* **No Top-K anywhere.** Ratio enters twice: as the `time_emd` input and as
  `target.mature` for `ratio_loss`.

## E. Current budget mechanism

`ratio · N → target.total`; `D = Σ need`; `Bp = min(D, 0.35·target.total)`;
`p = need·Bp/D` (Σp ≡ Bp, no bisection); `admitted = ~mature ∧ (u < p)`;
`target.mature = clamp(total − actual_admitted, 0, mature_count)`;
`ratio_loss = |Σ keep_probability − target.mature| / max(1, total)`.
Frozen per-row uniform ⇒ exact nesting across 0.01/0.05/0.10/0.15.
**A reusable normalization mechanism exists**: `ElasticBudgetManager::plan()`
already converts an arbitrary demand vector into a per-row probability whose sum
equals a target, and it is already called from three sites with different ratios.

## F. TF dependence on Mature mask

One real check ([gaussian_mapper.cpp:2987-2989](src/gaussian_mapper.cpp#L2987-L2989)) plus three mirrors
([historical_gradient_render.cpp:38-39](src/historical_gradient_render.cpp#L38-L39),
[gaussian_mapper.cpp:2537](src/gaussian_mapper.cpp#L2537), AABB at [1903-1913](src/gaussian_mapper.cpp#L1903-L1913) and
[2332-2339](src/gaussian_mapper.cpp#L2332-L2339), CD at [2239-2255](src/gaussian_mapper.cpp#L2239-L2255)).
Protected rows get identity by *exclusion*, never an explicit identity call.
TF works on **compact local indices** throughout.
Minimum compatibility layer: keep `bool getSelectorMatureMask()` as `m_i > τ`.

## G. GI dependence on Mature mask

`supervised = mature ∧ ¬protected ∧ observed` ([gi_teacher.cpp:31-37](src/gi_teacher.cpp#L31-L37));
`target_k = floor(mature_ratio · mature_count)` with
`mature_ratio = target.mature/mature_count` ([elastic_budget.h:75](include/elastic_budget.h#L75));
strict-percentile labels ([gi_teacher.cpp:59-66](src/gi_teacher.cpp#L59-L66));
loss on mature∪observed only ([gaussian_mapper.cpp:2957-2965](src/gaussian_mapper.cpp#L2957-L2965)).
`build()` throws on shape/dtype mismatch ⇒ the `[N]` bool shim is mandatory.
`validMask()` (`global_gi.cpp`) is independent of maturity.
Minimum compatibility mask: the same `m_i > τ` bool.

## H. Best existing touched/visibility signal

* Full: `render_pkg.visibility_indices` (global int64, `(radii>0).nonzero()`), already
  on GPU and already consumed by `beginFullMapping`.
* LoD: `lod.supervised_indices` (compact→global), already on GPU and already
  consumed by `updateElasticSupervision`.
* **Use both**, because `detach_gaussian_parameters = 0` means LoD steps really do
  move Gaussian parameters, and a row can be LoD-touched then Full-culled.
  A third producer exists: `trainJointReplay` → `updateElasticSupervision`
  ([gaussian_mapper.cpp:2825-2826](src/gaussian_mapper.cpp#L2825-L2826)).
* Cost: one `index_fill_` per phase. No new render, no host sync.

## I. Exact recommended Full-boundary insertion point

* **Drift / maturity / `drift_ema` / snapshot write:** inside `commit_optimizer`,
  after [gaussian_mapper.cpp:1545](src/gaussian_mapper.cpp#L1545), guarded by
  `full_step && gaussian_committed`. Pre-topology, `[N]` stable, mutex held,
  commit flag in scope.
* **`touched_in_cycle` clear for the next cycle:** after
  `ultrafast_rotation_.complete(...)` at [gaussian_mapper.cpp:1659](src/gaussian_mapper.cpp#L1659).
* **Never** at line 1658 for anything that reads `[N]`-sized parameters, because
  densify/prune at [1591](src/gaussian_mapper.cpp#L1591) may already have changed `N`.

## J. Topology synchronization hooks

`GaussianProtection::reset/append/prune` ([gaussian_protection.h:27-73](include/gaussian_protection.h#L27-L73)) are
already invoked from the only two row-count-mutating functions,
`densificationPostfix` ([gaussian_model.cpp:765](src/gaussian_model.cpp#L765)) and
`prunePoints` ([gaussian_model.cpp:663](src/gaussian_model.cpp#L663)). Add the new tensors
there. Forward `AppendKind` to distinguish clone/split so split children can reset
`drift_ema`/`snapshot_valid`. `protection_.assertAligned` is the ready-made
alignment guard.

## K. Loop / Scale correction hooks

`captureGeometryChange`/`finishGeometryChange` + `ChangeCoordinator` already
expose `old_state`, per-correction `indices`, `affects_all`, `transform_4x4`,
`rotation_wxyz`, `scale`, `changed_attributes`, plus `change_version_`.
`applyGaussianTransform` is the exact snapshot-transform primitive. `drift_stability_manager_->refreshReference(..., "geometry_change")`
is the existing "known global change invalidates the baseline" convention.

## L. Optimizer / LR information

`SparseGaussianAdam : torch::optim::Adam`, six fixed groups
(`xyz, f_dc, f_rest, opacity, scaling, rotation`) plus appended groups for
Selector/TF by `ensureSharedMappingOptimizer`. Only group 0 is scheduled
(`exponLrFunc`); groups 1-5 are constant and refreshed every iteration. Current
LR is read from `param_groups()[g].options().lr()`. **Full and LoD share one
optimizer and one Adam state map**, including for Selector and TF; Full does not
produce gradients for groups ≥6, so Adam skips them.

## M. Snapshot memory estimate

Option A (float32) = **50 B/row**: 4.77 MiB @100k, 11.92 MiB @250k,
23.84 MiB @500k, 6.41 MiB @134,485 (this repo's actual mode1 run).
Option B (float16 snapshot) = **30 B/row**: 2.86 / 7.15 / 14.31 MiB (40% less).
The existing Protection metadata is 52 B/row, so Option A is **size-neutral**
versus today. float16 is safe for `log σ` and quaternion components but **not**
for raw XYZ or raw physical scale. Recommend Option A for v1.

## N. Files that must be modified for a continuous maturity implementation

Minimum set:

1. `include/gaussian_protection.h` (+ `src/protection_evidence.cu`) — add
   `maturity`, `drift_ema`, `prev_full_xyz/log_scale/rotation`, `touched_in_cycle`,
   `snapshot_valid`; extend `reset`/`append`/`prune`/`assertAligned`; replace or
   bypass `matureMask()`/`classify()`.
2. `include/gaussian_model.h` + `src/gaussian_model.cpp` — `getSelectorMatureMask()`
   shim; forward `AppendKind` into `protection_.append`; the append/prune/
   `updateElasticSupervision` writes for the new tensors.
3. `src/gaussian_mapper.cpp` — the Full-boundary measurement/update block, the
   `touched` marks in the LoD branch and Full branch, the `selector_mask` blend,
   and the `expected_mature`/`ratio_loss` update.
4. `include/elastic_budget.h` + `include/elastic_budget_allocation.h` —
   `population`/`plan` must accept a blended demand (or a parallel
   `planFromProbability`) and the maturity threshold must replace
   `age_ready`/`optimization_ready`/`unique_keyframe_ready`.
5. `cfg/selector/*.yaml` + `src/gaussian_mapper.cpp` config parsing
   ([541-572](src/gaussian_mapper.cpp#L541-L572)) — new keys, removal of the four
   Protection threshold keys (keeping them as no-ops avoids breaking existing
   checkpoints).
6. `python/elastic_budget.py` + `python/selector.py` — the evaluator's
   `classify_protection`/`protection_budget`/`load_selector_metadata` reproduce
   the hard scheme and `load_selector_metadata` **hard-fails** on
   `protection_schema != "evidence_aware"` ([python/selector.py:86-87](python/selector.py#L86-L87)).
   Any new metadata schema must update this or eval breaks.
7. `src/gaussian_model.cpp` `saveSelectorMetadataPly`/`loadSelectorMetadataPly` —
   PLY schema, or keep the old fields write-only for compatibility.
8. Tests: `tests/gaussian_protection_test.cpp`,
   `tests/ultrafast_mapping_test.cpp`, `tests/elastic_budget_test.cpp`,
   `tests/ultralight_mode2_mapping_test.cpp`, `tests/gi_teacher_test.cpp`,
   `tests/ultra_light_schedule_test.cpp`.

## O. Risks / hidden dependencies

1. **`getSelectorMatureMask()` has 9 call sites** across TF, AABB, Change
   Distill, the historical guard, the drift probe, and GI reporting — verified by
   grep: [gaussian_mapper.cpp:1339](src/gaussian_mapper.cpp#L1339), [1904](src/gaussian_mapper.cpp#L1904), [2135](src/gaussian_mapper.cpp#L2135),
   [2240](src/gaussian_mapper.cpp#L2240), [2333](src/gaussian_mapper.cpp#L2333), [2404](src/gaussian_mapper.cpp#L2404), [2872](src/gaussian_mapper.cpp#L2872),
   [3941](src/gaussian_mapper.cpp#L3941), plus `GaussianModel::getSelectorMatureMask`
   itself ([gaussian_model.cpp:678-680](src/gaussian_model.cpp#L678-L680)). Removing the hard mask
   without a shim breaks all of them at once.
2. **`GITeacherBuilder::build` throws on shape/dtype mismatch** and requires
   same-device `[N]` bool ([gi_teacher.cpp:13-38](src/gi_teacher.cpp#L13-L38)). The shim must
   return exactly `[N] bool` on `importance_.device()`.
3. **`finishFullMapping` must be called after topology** — the comment at
   [gaussian_mapper.cpp:1633](src/gaussian_mapper.cpp#L1633) states that pending visibility is
   remapped through prune/append before the commit. So the Full evidence commit is
   already "topology-aware"; a maturity snapshot taken at 1545 is **not**.
   Any append/prune must therefore extend `protection_.append/prune` or the
   snapshot desynchronizes.
4. **`combineMappingOperations` can append rows between Full steps**
   ([gaussian_mapper.cpp:3174-3178](src/gaussian_mapper.cpp#L3174-L3178), [3273-3277](src/gaussian_mapper.cpp#L3273-L3277)) — an
   `[N]` snapshot taken at Full can become stale mid-cycle. `snapshot_valid` is
   required.
5. **Split children get a covariance-sized jitter** ([gaussian_model.cpp:811-817](src/gaussian_model.cpp#L811-L817))
   which is not optimizer drift. Inheriting `drift_ema` would mark freshly split
   Gaussians as unstable; inheriting `maturity` while resetting `drift_ema` is the
   correct combination.
6. **The LoD optimizer step at line 1551 happens before the LoD render** — for
   phases 0-3 the parameters that produced the LoD image are the *pre-step*
   parameters, and the step applies the *LoD* gradients. A naive
   "snapshot at the start of the phase" would pair the wrong parameters with the
   wrong loss; use the Full boundary only.
7. **Gradients accumulate across the 4 LoD phases plus Full** within one
   iteration; `zero_grad` fires at the top of each iteration and after the step.
   A maturity update that samples gradients must be explicit about which phase it
   is in. Conflict Replay
   ([gaussian_mapper.cpp:2726-2843](src/gaussian_mapper.cpp#L2726-L2843)) is a third
   Gaussian-parameter producer that runs *outside* the rotation and uses its own
   replay ratio cursor ([2845](src/gaussian_mapper.cpp#L2845) area,
   `replay_ratio_cursor_`), so it neither belongs to a phase nor advances
   `ultrafast_rotation_.step`.
8. **`mode2` is not Full-to-Full** ([gaussian_mapper.cpp:1668-1702](src/gaussian_mapper.cpp#L1668-L1702)) — one
   native Full per mapping plus one detached LoD. A cycle-based design must
   either gate on `ultrafast_rotation_.enabled` or define a different cadence.
9. **`densificationPostfix` zeroes `xyz_gradient_accum_`/`denom_`/`max_radii2D_`**
   ([gaussian_model.cpp:770-775](src/gaussian_model.cpp#L770-L775)) — do not place maturity state
   in those tensors.
10. **`Selector.protection_enabled: 0` disables the whole classification** and
    forces `mature = ones` ([gaussian_mapper.cpp:2871-2873](src/gaussian_mapper.cpp#L2871-L2873),
    [gaussian_protection.h:171-176](include/gaussian_protection.h#L171-L176)). A continuous maturity
    must define what happens in that mode.
11. **Python evaluator schema coupling:** [python/selector.py:86-87](python/selector.py#L86-L87)
    raises unless `protection_schema == "evidence_aware"` and
    `unique_keyframe_ready` matches the checkpoint width; changing the metadata
    without updating the evaluator breaks `python/run.py` evaluation entirely.
12. **LR spread of ~300× across groups** (section 12) means one global drift
    threshold is not scale-free across attributes; either normalize by the group
    LR or calibrate per attribute.
13. **`protection_.mapping_iteration` is the maturity clock** and is advanced
    every mapping iteration ([gaussian_mapper.cpp:1420](src/gaussian_mapper.cpp#L1420)); a
    cycle-indexed maturity must use `ultrafast_rotation_.step / 5`, not
    `mapping_iter_`.

## P. Minimal implementation path

The smallest set of changes that delivers `m_i∈[0,1]` from Full-to-Full geometric
drift with a touched gate, an EMA, and a continuous Selector handoff. No new
algorithm, no new subsystem.

**Step 1 — state (new tensors on `GaussianProtection`, ~6 lines each in `reset`)**

```
torch::Tensor maturity, drift_ema;                 // [N] float32, in [0,1]
torch::Tensor prev_full_xyz, prev_full_log_scale, prev_full_rot;   // [N,3]/[N,3]/[N,4]
torch::Tensor touched_in_cycle, snapshot_valid;    // [N] bool
```
`reset` → `zeros` for `maturity`/`touched`/`snapshot_valid`,
`ones` for `drift_ema` (or `full_like(drift_ema, +inf)`);
`append` → inherit `maturity` from `parents` (like
`unique_keyframe_count` already does), but `drift_ema = 1`,
`snapshot_valid = false`, `touched = false`;
`prune` → `index_select` all six exactly like the existing five.

**Step 2 — compatibility shim (2 lines, replaces `matureMask()`)**

```
torch::Tensor GaussianModel::getSelectorMatureMask() const {
  return (protection_.maturity >= kMatureThreshold);   // [N] bool, no grad
}
```
with `kMatureThreshold` set so the initial population reproduces today's behaviour
(e.g. `1.0f` with a hard cutoff, or `0.999f`). This alone keeps TF, GI, AABB,
Change Distill, the historical guard and the drift probe compiling and behaving.

**Step 3 — `protection_.need` redefinition (1 line)**

`classify().need = 1 - maturity` instead of `min(age_conf, opt_conf, kf_conf)`.
Nothing else in `elastic_budget.h` changes; the demand-driven admission keeps
working with a continuous need.

**Step 4 — the cycle measurement (one block, ~25 lines), at the insertion point of section 9.3**

```
if (full_step && gaussian_committed) {
  const auto xyz  = gaussians_->getXYZ().detach();          // [N,3]
  const auto logs = gaussians_->scaling_.detach();          // [N,3] log-scale
  const auto rot  = gaussians_->getRotationActivation().detach();
  touched.index_fill_(0, render_pkg.visibility_indices, true);   // Full mark
  if (prev_full_xyz.defined() && prev_full_xyz.size(0) == xyz.size(0)) {
    const auto d_xyz   = (xyz - prev_full_xyz).norm(2, 1) / scene_diagonal;
    const auto d_scale = (logs - prev_full_log_scale).abs().mean(1);
    const auto d_rot   = 1 - (rot * prev_full_rot).sum(1).abs().clamp(0, 1);
    const auto drift   = (d_xyz + d_scale + d_rot) * snapshot_valid.to(kFloat);
    drift_ema = drift_ema.defined() ? 0.5f*drift_ema + 0.5f*drift : drift;   // EMA
    const auto t = (touched_in_cycle & snapshot_valid).to(kFloat);           // gate
    const auto grow  = (drift_ema < kDriftTol).to(kFloat) * t * kGrowRate;   // slow up
    const auto decay = (drift_ema >= kDriftTol).to(kFloat) * kDecayRate;     // fast down
    maturity = (maturity + (1 - maturity) * grow).clamp(0, 1) * (1 - decay);
  }
  prev_full_xyz = xyz.clone(); prev_full_log_scale = logs.clone();
  prev_full_rot = rot.clone(); snapshot_valid = true; touched_in_cycle = false;
}
```
Two deliberate properties: `grow` is multiplied by `t`, so an untouched row can
never gain maturity; and `decay` is not, so an untouched *unstable* row still
loses maturity (the "fast decrease" is unconditional).

**Step 5 — touched marks in the LoD branch (1 line)**

In the LoD phase, after `lod` is returned from `accumulateLodGradients`:
`touched.index_fill_(0, lod.supervised_indices, true);`
(`lod.supervised_indices` is already computed at
[gaussian_mapper.cpp:3050-3053](src/gaussian_mapper.cpp#L3050-L3053) and already on GPU.)

**Step 6 — the Selector handoff (3 lines around [gaussian_mapper.cpp:2922-2952](src/gaussian_mapper.cpp#L2922-L2952))**

```
auto p_sel_all = torch::zeros({num_gaussians}, xyz.options())
                     .index_copy(0, mature_indices, keep_probability);   // [N]
const auto m   = gaussians_->protection_.maturity.detach();               // [N]
const auto p_mix = (1 - m) * target_ratio + m * p_sel_all;               // [N]
// hard sample preserving the straight-through contract:
const auto u = gaussians_->protection_.cached_uniform;
const auto st = p_mix;                                                    // soft
const auto hd = (u < p_mix).to(st.scalar_type());                          // hard
selector_hard_mask = hd - st.detach() + st;
expected_mature    = p_mix.sum();                                          // was keep_probability.sum()
```
`protected_mask = ~mature_mask` and the `protected_f + (1-protected_f)·mask`
combination at [2948-2950](src/gaussian_mapper.cpp#L2948-L2950) are **left exactly as they are** —
`mature_mask` is now `m_i > τ`, Protection admission is unchanged, and the blend
only reshapes the Selector's contribution. `ratio_loss`
([2951-2952](src/gaussian_mapper.cpp#L2951-L2952)) automatically follows `expected_mature`.

**Step 7 — teardown of the old scheme (only once steps 1-6 are validated)**

Delete `birth_iter`, `optimized_count`, `unique_keyframe_count`,
`supporting_keyframe_ids`, `transitions`, `GaussianProtectionState`,
`advance/beginFullMapping/finishFullMapping`, `src/protection_evidence.cu`, the
four `Protection.*_ready/stale_age` config keys, the corresponding
`selector_metadata.ply` fields and `protection_state.json` keys, and the matching
Python code. Keep `cached_uniform` — it is the Bernoulli source the new handoff
reuses.

**Explicitly not in scope for v1** (per the stated design): age, optimization
count, unique-KF count, Hard/Cold/Mature pools, 65/25/10 quotas,
photometric-residual maturity, opacity maturity, Fisher/Hessian, exposure/parallax
models, and any TF change.
