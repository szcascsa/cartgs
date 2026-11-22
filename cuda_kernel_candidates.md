# CUDA候选接口与任务拆分

2026-10-22，基线HEAD `4269d7e3bad6cf5af7e1df1763634b61e98e0b51`。以下均为**建议接口，尚未实现**；测量及边界见 `ANCHOR_SELECTOR_R3_PERFORMANCE_AUDIT.md`。GPU共享负载下的耗时只作定位，不能作为已承诺收益。

## K1：多Tensor dense Adam，首项

生产入口：`src/gaussian_mapper.cpp:2313 ensureSharedMappingOptimizer`、`stepWithHistoricalGuard`→`src/transform_field_cuda_optimizer.cpp:97 tfCudaAdamMixedStep`。当前Anchor路径cuda groups为空，最终执行base Adam。实际LoD有14个活跃parameter，约112 kernels；Full六组Master约48 kernels。

建议C++接口：

```cpp
struct DenseAdamLeaf {
  Tensor parameter, gradient, exp_avg, exp_avg_sq; // FP32 CUDA，独立leaf
  double lr, beta1, beta2, eps, weight_decay;
  int64_t step_before;
};
void dense_adam_many(std::vector<DenseAdamLeaf>& active,
                     cudaStream_t stream);
```

首版只支持当前真实配置：FP32、AMSGrad关闭、连续参数/gradient。宿主收集定义了grad的leaf，逐叶推进step；CUDA按Tensor分块处理。已有TF optimizer实现可作公式/接口参考，但禁止因复用而把R3加入错误schema。

必须保留：eps1e-15、bias correction位置、每叶LR、全人口dense moment衰减、grad=None跳过、零grad仍可动量更新；保持ATen版本计数、optimizer alias与state。不能将四张表合成一个普通Adam leaf，也不能仅更新ids行。

Backward不变；只接管实际step，不聚合ratio或跨step。验收使用 `native1/*_parity.pt` 的同一gradient/m/v/step，另构造“上一轮选中、本轮未选中”的moment样例。性能比较完整Adam scope及完整Mapping，不只一张R3表。

预期减少：每叶约8个elementwise launch、中间sqrt/div Tensor。是否受内存带宽限制需新profile，当前没有NCU证据。

## K2：14D属性decode与VJP

生产入口：`improvements/selector/gumbel_network.h:121 applyCompensationResidual/applyAppearanceResidual`、`src/gaussian_mapper.cpp:3419`。当前Forward约34 kernels，Backward是多个slice/reduce/quaternion/normalize/scatter。

```cpp
Decoded decode14(Tensor xyz_K3, Tensor activated_scale_K3,
                 Tensor normalized_rotation_K4, Tensor opacity_logit_K1,
                 Tensor sh_dc_K13, Tensor delta_K14, Tensor st_mask_K,
                 cudaStream_t stream);
DecodeGrad decode14_vjp(DecodedContext context, AttributeGrad output_grad,
                       cudaStream_t stream);
```

输出仍为Renderer需要的SoA：XYZ[K,3]、Scale[K,3]、Rotation[K,4]、Opacity[K,1]、DC[K,1,3]。SH rest不补偿，沿原gather传递。

语义：XYZ加法；activated scale乘exp(delta)；`normalize(normalize(identity+delta_q)*base_q)`；opacity先加logit后sigmoid、最后乘ST；DC加法。normalize的clamp与当前代码相同，顺序/乘法方向不改变。若base quaternion已归一化，不再偷偷去掉生产存在的相关导数路径。

VJP输出各base属性梯度、delta[K,14]梯度、ST梯度。第一版仍交给ATen把K行梯度scatter到Master/R3；通过后才考虑融合scatter。保持ST对选中opacity的photo梯度；Budget梯度仍独立作用全N概率。

当前 `state.log_scaling+delta` 的中间结果不是Mapping实际scale输出路径，不能用错误的log-space导数替代activated-scale输入的链式梯度。测试极小/极大scale、四元数范数clamp、opacity饱和、K=0/1与任意strides。

预期减少：属性Forward/VJP的launch及临时量；先独立decode，再逐步融合gather，不一次改Renderer。

## K3：Anchor Full统计与完整性校验

生产入口：`src/gaussian_model.cpp:155 addAnchorPruningStats`、`improvements/selector/explicit_anchor_blocks.h:20 assertBlocks`。真实布局检查约0.5ms；完整窗口统计每次Full执行。

```cpp
AnchorStatsResult update_anchor_window(
    Tensor owner_N_i64, Tensor member_counts_A_i64,
    Tensor levels_A_i64, Tensor centers_A3, Tensor scales_A1,
    Tensor visible_ids_V_i64, Tensor opacity_N1,
    MutableAnchorWindow state, cudaStream_t stream);
// result包括设备error flags；宿主在既有安全边界检查。
```

状态包括Anchor FP64 opacity累积、int64 visits，以及N行slot opacity累积/低于0.02与1/255的计数。某Anchor任一成员可见则该Anchor访问一次，其所有成员使用本次真实opacity；未观察不记零。不引入按成员失败比例删除等新条件。

建议分2～3 kernels：标记可见Anchor；并行成员统计/owner归约并完成校验；必要的结果归并。能否按Anchor分段，需要先有可靠CSR，不能假定成员连续。RTX3090的FP64 atomic/规约成本需要测试，不能为速度改FP32。

保留owner范围、bincount一致、1..10成员、层级、center finite及s>0全部检查。错误flag不能延迟到非法索引访问之后才检查。宿主一次读取取代多次scalar读取是候选，不能直接删检查。

无Backward；prune仍按原T/O严格不等式与原重置窗口。验证整组删除mask、未充分观察累积、slot诊断、重复visible、异常owner/NaN中心。

## K4：Anchor-aware input pack与VJP

生产入口：`anchor_bank.h:323 inputs`→`gumbel_network.h:329 anchorInputs`。lookup5+input pack4 kernels。

```cpp
Tensor pack27(Tensor owner_N, Tensor feature_A16,
              Tensor center_A3, Tensor reference_scale_A1,
              Tensor xyz_N3, Tensor activated_scale_N3,
              Tensor sh_dc_N13, Tensor activated_opacity_N1,
              float ratio, cudaStream_t stream);
PackGrad pack27_vjp(Tensor d_input_N27, PackContext context,
                   cudaStream_t stream);
```

输入27的顺序与原代码逐列一致；Offset=`(XYZ-center[owner])/s[owner]`；ratio为log(r)，不新增globalXYZ，不使用camera/Rotation/SI。中心/尺度是空间参考buffer，不额外创建可独立漂移几何。

VJP对Master XYZ、activated Scale/DC/Opacity和Anchor Feature回传；Feature按重复owner相加，中心/尺度保持原不可训练边界。候选按Gaussian并行写连续N27；Feature缓存可warp/shared试验，但按真实owner分布测试。

输出给现有cuBLAS GEMM；不要首版就手写整个MLP。若随后加入Linear epilogue，cuBLAS/cuBLASLt/自定义kernel需三方比较，尤其小K与N=10K/50K/100K形状。

不能缓存训练step之间的Offset/Feature值；仅owner/分段索引按topology epoch缓存。几何修正、owner迁移、Feature optimizer step均为对应缓存失效源。

## K5：Cached ST / Budget / Stable Compaction

生产入口：`include/selector_handoff.h:16`、`gaussian_mapper.cpp:3341/3402`；当前约18 kernels和动态nonzero同步。

```cpp
SelectionResult select_cached(Tensor probability_N, Tensor uniform_N,
                              float ratio, int64_t population_N,
                              Workspace scan_scratch, cudaStream_t stream);
// hard/ST mask[N], ids[K]升序, Budget scalar, K
```

正向严格`uniform < p`，K取原hard结果；ST正向精确0/1、梯度沿原`hard+(p-p.detach())`；Budget为`abs(sum(p)-r*N)/max(1,r*N)`，N含存储人口，不能改为eligible数量。slot masking保持原顺序和梯度。

采用稳定scan/select；ids按原Gaussian行升序。两类softmax改sigmoid差值属于另一个数值敏感候选，不与首版compaction混在一起。无额外随机数、不重新采样uniform。

当前Tensor shape与Renderer CPU P依赖K；首版允许一次明确count同步。若要全部device-driven，必须另设计容量buffer+device count的Renderer接口并单独证明语义，不能通过固定K或padding改变算法。

测试p/u相等和相邻浮点、饱和概率、K0/全选、abs零点、非连续p stride2、slot eligibility、expected/actual budget。集合与K必须精确，不使用allclose替代。

## 后续融合路径及前置条件

1. K行Gaussian gather→R3 gather/add→K2 decode：当前6+2+34 launches，先通过K2再融合，避免把多个错误混在一个变更里。
2. K4 pack→现有GEMM/epilogue：保留N/K及原h，检查Forward和所有输入梯度，不预设custom胜过cuBLAS。
3. K3 validation→窗口统计：保留完整检查，减少device-host round-trip。
4. Renderer workspace池：Forward上下文必须活到对应Backward；不能用同一scratch覆盖另一个在途渲染。已有CUDA的scan/sort/桶数读取不可直接删除。
5. Keyframe CPU准备/Pinned H2D预取：只预取只读输入，CUDA Event交付，保持KF顺序；不并行写Master/Anchor。

SSIM backward在本次5% Nsight中单kernel约0.80ms，应与Renderer一起继续硬件profile；它已经CUDA化，不应被遗漏，也不应该在没有NCU和等价loss验证时直接重写。

## Stream与发布约束

- 所有新kernel显式使用LibTorch当前stream，禁止默认stream硬编码。
- 当前Rasterizer与SSIM含隐式default-stream launch；在其stream协议修正及两streamparity前，不把整个Mapping迁到非默认stream。
- 同一个optimizer step及Topology事务不并发写共享状态；Full→step→topology→LoD读取顺序保留。
- 先隔离编译开关和原实现对照；通过fixture、same-K渲染、短程训练和独占GPU计时后，才申请生产替换。
