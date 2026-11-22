# LoD Selector 名义Anchor视锥候选

## 功能与开关

`Selector.anchor_frustum_culling: 1`为新默认。设为`0`恢复原全图Selector路径。该开关只在显式Anchor架构生效，Full训练保持原有全图渲染。

每次LoD使用当前Mapping相机的`full_proj_transform_`，对每个Anchor的名义AABB

\[
[c_a-h_a/2,\;c_a+h_a/2],\qquad h_a=\mathrm{offsetScale}_a
\]

执行六个裁剪平面的支持半径测试。盒子与视锥边界接触也通过。与既有Python推理一致，采用行向量矩阵和`0≤z_clip≤w_clip`约定。没有padding，不使用成员包围盒，不添加补偿位移扩张。

通过Anchor的**全部现存成员**形成候选池；不检查每个成员自身是否位于视锥。Selector仍逐Gaussian预测、采样，不是整Anchor选择。Compensator仅处理最终选中点。

## 预算、采样与梯度

1. 全图人口为`N`，网络输入ratio仍为`r`。
2. 候选成员以递增全局row组成`rows`，紧凑属性进入原11D Selector。CUDA输入打包新增`source_rows`参数，按全局row查询正确Anchor owner。紧凑人口不符合已校准CUDA执行范围时，使用原LibTorch MLP。
3. 保留原FlexGS Gumbel直通采样。采样结果通过`index_copy`恢复为长度`N`的全图Mask；非候选为零，无Selector前向和直通梯度。
4. 原预算Loss继续使用全图Mask，归一化版本仍为

   \[
   L_b=\frac{|\sum_{i=1}^{N}m_i-rN|}{\max(1,rN)}.
   \]

   未归一化版本仍为`abs(mask.mean()-r)`，其中`mask.numel()==N`。因此不会误用`r×候选点数`。
5. 训练保持概率采样，**不承诺每步实际K严格等于rN**；推理保持既有全图分位数选点。候选不足时预算可能不可达，不从视锥外补点，不改变Loss权重、不切换为训练Top-K。
   新路径只为候选生成Gumbel随机数，随机数消费量随候选数变化；相同seed不代表与历史全图训练具有逐位相同轨迹。
6. Maturity相对Loss的pool使用`候选Mask ∩ 原slot资格Mask`，仍使用实际采样的可微soft权重。成熟度detach、Selector私有VJP与Photo/Budget反向的原累加顺序不变。
   小人口回退ATen Linear时，私有VJP可能返回view。写入首次`.grad`前仅对view调用`detach()`，保留原存储、数值与stride，避免当前LibTorch在下一步`zero_grad`调用`detach_()`时报错。不合并双VJP，不复制或重算梯度。
7. Gaussian/Anchor的UID、owner、优化器和Feature表不变。每次读取当前Anchor几何及成员映射，没有跨步视锥缓存，Clone/Split/Prune/重排/Anchor迁移和闭环后不会复用旧候选。

## 保存、推理和诊断

- `saveSelectorConfig`和`anchor_state.json`保存开关。
- Python评价默认读取checkpoint的该开关；缺项的历史checkpoint仍采用原评价路径。`--selector-anchor-frustum`显式启用，`--no-selector-anchor-frustum`显式关闭。
- 既有推理视锥几何和全图分位数规则保持不变。
- 开启Mapping日志后，`[LOD Compact]`增加`selector_candidates`、`anchor_frustum`、`visible_anchors`和`budget_shortfall`。候选数也进入现有逐ratio统计。

## 验证范围与状态

验证代码和服务器独立构建位于：

- 本地：`research/anchor_frustum_training_20261102/`。
- 服务器：`/home/prmi/zwj/experiments/anchor_frustum_training_20261102/`。

验证结果：

| 检查 | 结果 |
|---|---|
| 独立CMake构建 | `tum_rgbd`、`tum_mono`、新测试及回归测试编译通过 |
| C++ CPU / CUDA几何与预算单测 | 六平面、边界、空集合、全局row、Anchor重排、预算VJP、Maturity detach通过；view包装解除前后存储地址和数值一致 |
| 实际`accumulateLodGradients` | 合成地图实际初始化为16点、候选4点；六档训练ratio的预算Loss均匹配`r×16`，包含候选不足的40%档 |
| 实际渲染/反向/Adam | 候选外Master梯度为零；仅Maturity项时Selector有梯度、Master/Compensation/Feature为零；Photo/Budget联合反向和Adam更新通过 |
| 开关与成员边界 | 关闭恢复16点输入；空视锥为0；成员移至视锥外但owner未变仍进入4点候选；inactive候选保持排除；配置保存读回为1 |
| CUDA owner查询 | 非连续候选row的logits/shared与相同全局成员逐值一致 |
| 既有Selector handoff回归 | CPU和CUDA均通过 |
| Residual48输入融合回归 | 97项Forward/VJP bitwise检查通过 |
| 既有Python推理视锥回归 | 7项测试通过 |
| 服务器主项目完整性 | 334个受保护源码/配置/二进制/库等文件前后SHA256一致 |
| 暂停状态 | 原60项和24项两批`paused=true`，均未恢复 |

这是生产函数上的合成地图渲染、反向与更新验证，**不是完整数据集在线SLAM质量实验**。本轮未声称PSNR提升或端到端加速。服务器仅构建到独立目录，未替换主项目运行库或默认配置；本地生产默认YAML已启用新开关。未commit/push。

最终原始日志、返回码、SHA与队列核验见`research/anchor_frustum_training_20261102/evidence/`。

## 方法边界

名义格子不保证包含优化后成员，更不保证包含无界XYZ补偿后的位置。因此这项功能改变LoD候选集合与训练结果，不属于严格无损Rasterizer裁剪。Full仍然监督全图；画外点没有当前LoD直接监督时，其共享网络预测仍可能因其他点的训练而变化。
