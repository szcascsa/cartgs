# Compact LOD 渲染接入设计

## 目标与接入边界

将 `GaussianMapper::trainSelectorUpdate()` 的 LOD rasterizer 输入从全局
`N` 个 Gaussian 改成实际保留的 `K` 个 Gaussian。Master Gaussian 不删除、
不重排，optimizer 及其状态仍对应全局 population。

Selector、GI loss 和 ratio loss 仍在全局 population 上计算。Compact 只改变
LOD 的临时输入表示，适用于在线和后优化两个阶段；原有在线降分辨率、
SLAM 结束后恢复原图分辨率的逻辑继续生效。

`GaussianRenderer` 的 tensor-input 接口已经按输入行数工作，CUDA wrapper
也使用 `means3D.size(0)` 分配前向和反向缓冲区，因此无需修改 CUDA kernel。
第一版由 Mapper 完成 gather、TF 局部更新及索引回映；Renderer 保持输入局部
索引的统一语义。

## 两个索引空间

全局保留集合为：

```text
protected ∪ selected mature
```

使用本次已经生成的有效 Selector mask，创建唯一的映射：

```cpp
auto compact_to_global =
    torch::nonzero(selector_mask.detach() > 0.0f).reshape({-1});
const int64_t compact_count = compact_to_global.numel();
```

`compact_to_global[j]` 表示 compact 第 `j` 行对应的 master 行号。
该 Tensor 是当前 device 上的 int64 Tensor，行号唯一且按全局索引递增。
本次所有属性、掩码和统计必须使用同一张映射表。

例如：

| compact 行号 | master 行号 |
|---:|---:|
| 0 | 1 |
| 1 | 4 |
| 2 | 7 |
| 3 | 9 |

Renderer 返回局部可见索引 `[0, 2]` 时，对应全局索引是 `[1, 7]`。

第一版不分配长度为 `N` 的 `global_to_compact`。需要投影全局掩码时，直接
使用 `global_mask.index_select(0, compact_to_global)`；TF eligibility 由此
在 compact 空间内重新表达，不需要搜索或反向映射旧的 TF 全局索引。

## 在 TF 之前 compact

先保留全局 Selector/GI/ratio loss 的现有计算，再 gather 六类渲染属性：

```cpp
auto xyz_c = xyz.index_select(0, compact_to_global);
auto scale_c = scaling.index_select(0, compact_to_global);
auto rotation_c = rotation.index_select(0, compact_to_global);
auto opacity_c = opacity.index_select(0, compact_to_global);
auto dc_c = features_dc.index_select(0, compact_to_global);
auto rest_c = features_rest.index_select(0, compact_to_global);
auto mask_c = selector_mask.index_select(0, compact_to_global);
```

属性来源继续遵守 `detach_gaussian_parameters` 的现有选择。正常 no-detach
训练中，上述 gather 不放入 `NoGradGuard`，不对属性或 `mask_c` 再 detach。

TF mask 逐项沿用现有规则，但在 compact 空间构造：

```cpp
auto tf_mask_c = selector_hard_mask.detach()
    .index_select(0, compact_to_global) > 0.0f;
if (transform_field_config_.mature_only)
  tf_mask_c = torch::logical_and(
      tf_mask_c, mature_mask.index_select(0, compact_to_global));
if (transform_field_config_.outside_identity)
  tf_mask_c = torch::logical_and(
      tf_mask_c, transform_field_->insideMask(xyz_c));
auto tf_local_indices = torch::nonzero(tf_mask_c).reshape({-1});
```

TF 输入是 compact 属性中 `tf_local_indices` 对应的行；返回结果使用
out-of-place `index_copy()` 写回长度为 `K` 的 `xyz_c/scale_c/rotation_c`。
不再创建并写回长度为 `N` 的三个 elastic 属性 Tensor。

当前 fast/slower 配置的 `mature_only=1`，protected Gaussian 保持 keep 和
TF identity。Compact 不另行改变 `mature_only/outside_identity` 的配置语义；
关闭 mature_only 时涉及的 protected TF 策略应单独审阅。

最后构造的渲染输入为：

```cpp
GaussianRenderInput input{
    xyz_c,
    mask_c.unsqueeze(1) * opacity_c,
    scale_c,
    rotation_c,
    dc_c,
    rest_c,
    gaussians_->active_sh_degree_,
    gaussians_->max_sh_degree_};
```

必须保留这里的 `mask_c * opacity_c`。尽管被保留点的 mask 前向值约为 1，
将它替换为常量 1 会切断 mature selected 点的 Selector photometric 梯度。

## 可见性和统计回映

`RenderPackage` 的统计仍以 renderer 输入行为索引：

| 数据 | Compact 渲染中的索引空间 | 消费方式 |
|---|---|---|
| image | 图像像素 | 直接计算 photometric loss |
| visibility_indices | compact 局部行号 | 用 compact_to_global 回映后更新 master |
| radii | K 行 | 保持局部；需要全局诊断时才显式 scatter |
| viewspace_points / grad | K 行 | 保持局部，不手工加入 master XYZ grad |
| frame_importance / contribution_count | K 行（若开启） | 当前 LOD 不采集，Recent Window 仍用完整渲染数据 |

```cpp
auto global_visible = compact_to_global.index_select(
    0, render_pkg.visibility_indices);
gaussians_->updateElasticSupervision(global_visible, mapping_iter_);
```

当前 LOD 中针对 `selector_mask > 0` 的二次可见性筛选可以删除，因为所有
compact 行在构造映射时已经满足这个条件。

若消费者只需要 visible mature，则使用
`mature_mask.index_select(0, global_visible)` 进一步过滤全局索引。
沿用当前 `radii > 0` 的可见性定义；它不等于逐像素实际贡献计数。

普通 mapping 的 seen count、GI、max_radii2D、densification statistics 继续
使用完整渲染的全局结果，不混用 compact radii 或 viewspace gradients。

## 梯度、optimizer 和生命周期

`index_select` 的 backward 会把局部属性梯度累加回对应的全局源 Tensor，
再通过 scaling/rotation/opacity activation 等原有路径回到 master 参数。
因此不需要手工 scatter Gaussian 参数梯度，也不创建 compact optimizer。

保留当前完整渲染 backward、LOD backward、共享 optimizer step 的执行方式。
未被 LOD 选中的 Gaussian 仍可能从完整渲染及全局 Selector 的 GI/ratio loss
获得梯度，不能清零其全局梯度。即使当前 LOD photometric 梯度为零，Adam
也可能通过已有 momentum 更新参数，不能据 compact 集合裁剪 optimizer 更新。

CUDA 当前在 forward 和 backward 中都跳过 `alpha < 1/255`，原有零-opacity
未选中点已经没有 LOD 像素贡献及对应 photometric 梯度。这提供了等价改造的
基础；具体图像和梯度仍需在目标 GPU 上验证。

映射只属于当前一次 LOD forward/backward，使用已有 `mutex_render_` 保护
的训练区间，完成 backward 和全局统计更新后即弃用。之后 densify/prune 可以
照常执行；不缓存映射、不跨 iteration 使用、不增加 topology/SLAM 同步锁。

`K=0` 时沿用现有不执行 LOD photometric render 的分支，仍计算适用的 GI、
ratio loss 和 TF regularization。`N=0` 或无 mature 时沿用现有提前返回规则。

## 不应加入第一版的筛选

不以完整分支的 visibility 提前裁剪 Selector 或 compact population。
TF 可以把原先不可见的点移入当前视野，而大 Gaussian 也可能从视场边缘产生
有效贡献。第一版仅按有效 hard mask 与 protection 规则筛选，继续由 LOD
rasterizer 对变换后的属性做视场与屏幕覆盖判断。

保留 Gumbel hard selection 及全局 ratio loss，不改成严格 Top-K。
历史 QUALITY 使用 soft probability 的全量输入，不套用 hard compact 规则。
Change Distill、DSD、GI 缓存和 Master Gaussian identity 均不受此次临时 gather
影响。Compact 在 SLAM 结束后也继续启用，因为它本身不是降低画质的近似操作。

## 验证标准与观测指标

1. 索引测试：非连续行号、protected/selected 混合、局部 TF 行号与局部可见
   行号均正确回映；append/prune 后下一次 forward 重新构建映射。
2. 梯度测试：gather 后的 Student 梯度回到正确全局行，保留 mask 的
   straight-through 梯度；detach 模式仍不更新 Gaussian 参数。
3. CUDA 对照：固定相同 Gaussian、网络、ratio 和 Gumbel 采样，比较原来的
   N 点零-opacity渲染与 K 点 compact 渲染。图像和 Gaussian/Selector/TF 梯度
   必须在设定数值容差内一致，不能比较两次不同随机采样。
4. 覆盖混合保护点、空/全保留集合、TF 开关、TF 跨视场边界、在线半分辨率和
   后优化原图分辨率，并检查当前使用的 covariance/SH renderer 分支。
5. 数量日志记录 N、K、protected_count、selected_mature_count、TF_count、
   visible count。局部/全局 visible count 相同，只是行号不同。
   日志按现有报告频率输出，不新增逐步文件写入。实际 gather、TF、LOD
   forward/backward GPU 耗时需在目标机做 profiling，不用 CPU 异步提交耗时
   代替，也不在每次训练中增加额外 CUDA synchronize。

实际输入量是 `K = protected_count + selected_mature_count`，不是固定的
`ratio * N`。例如 N=100000、mature=80000，若 mature 实际保留 10%，则
K=20000+8000=28000。Protection 占比很高时，compact 的收益也相应受限。

本次降低的是 LOD preprocessing、tile 排序/遍历、CUDA 缓冲区、SH/covariance
计算和部分属性 Tensor 操作成本。全局 Selector、完整 mapping 渲染及全局
optimizer 仍有 N 规模开销，不能据 K/N 直接推算整个系统的加速倍数。

## 实现位置与验证状态

- `GaussianRenderInput::indexSelect()` 使用同一索引 gather 六类属性，保留
  autograd 和 SH degree，供 Mapper 和回归测试共用。
- `GaussianMapper::trainSelectorUpdate()` 每次重建 `compact_to_global`，在 TF
  之前 gather，TF 用 compact 局部索引写回，渲染可见性回映到 master。
  Renderer 实现、CUDA kernel 和 optimizer step 不变；不新增配置开关。
- `[LOD Compact]` 按主配置 `Record.training_report_interval` 打印，例如：

```text
[LOD Compact] iter=6000 ratio=0.1000 total=100000 rasterized=28000 protected=20000 selected_mature=8000 tf=7600 visible=9200
```

`rasterized` 是真正传入 rasterizer 的 K，不是屏幕最终可见数量；`visible`
沿用 radii > 0 的定义。报告间隔为 0 时不打印该日志。日志在本次 densify/prune
之前生成，数量对应本次 LOD 的输入，不一定等于稍后训练报告的 Gaussian 数量。

新增 `tests/compact_lod_test.cpp`，注册两个 CTest：

- `compact_lod`：CPU tensor 测试，检查共同 gather、TF 局部索引、可见性
  回映、空/全保留集合、append/prune 后重新索引；检查六类 master 梯度、
  Selector straight-through 梯度、TF 写回梯度及 detach 边界，并确认完整
  渲染已有的 master 梯度不会被覆盖。
- `compact_lod_cuda`：固定同一份 hard/soft straight-through mask 和 TF 初始化，
  对照原 N 行零-opacity输入与 K 行输入。覆盖 TF 开/关、Gaussian detach
  开/关、32/64 图像尺寸、CUDA/预计算 covariance、SH 预计算及 separate SH。
  TF 测试包含从原视场外移入的点。比较图像、保留点 radii、回映后的 visibility、
  master/Selector/TF 梯度；CUDA 不可用时返回 77，CTest 标记 skipped。

目标 CUDA 工程需已启用 `-DCARTGS_BUILD_TESTS=ON`，再执行：

```text
cmake --build <目标构建目录> --target compact_lod_test
ctest --test-dir <目标构建目录> -R "^compact_lod(_cuda)?$" --output-on-failure
```

当前本机没有 CUDA/LibTorch，按要求未构建、未运行上述 C++/CUDA 测试。
索引/调用链静态检查不能替代 GPU 等价性和速度测量。
