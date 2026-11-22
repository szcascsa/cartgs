# 首层复用的单层 Anchor 上下文残差

日期：2026-11-02。

## 方案定义与配置

本方案实现上一轮确定的Mean/Max集合残差：复用原11D Selector首层隐藏特征，通过每Anchor一个单层投影为第二层提供上下文。没有添加独立成员Encoder，也没有引入Q/K/V或softmax attention。

独立配置：`cfg/selector/anchor_context_residual48.yaml`。

```yaml
Selector.anchor_set_encoder: 0
Selector.anchor_set_statistics: 0
Selector.anchor_context_residual: 1
CUDA.fused_anchor_input_pack: 1
CUDA.fused_flexgs_selector: 1
```

三个上下文配置互斥，省略`anchor_context_residual`时默认关闭。保留当前Residual48补偿、Maturity V2及贡献修正、slot规则、Anchor名义格子视锥筛选、全图N预算、1000 Full预热和tail Full→LoD。

## 数学形式

原11D输入完整保留：

\[
u_i=[(x_i-c_{a(i)})/s_{a(i)},\operatorname{Scale}_i,DC_i,\alpha_i,\log r]\in\mathbb R^{11}.
\]

其中`c`为名义格子中心，`s=offsetScale`，Scale和Opacity使用原activated属性。原第一层：

\[
h_i=\operatorname{ReLU}(W_0u_i+b_0)\in\mathbb R^{32},\qquad v_i=h_i[0:8].
\]

对当前slot-eligible成员集合`E_a`聚合：

\[
g_a=[\operatorname{Mean}_{j\in E_a}v_j,\operatorname{Max}_{j\in E_a}v_j,|E_a|/10]\in\mathbb R^{17}.
\]

共享无偏置单层投影`P∈R^(32×17)`每Anchor计算一次，然后广播到成员：

\[
b_a^{context}=Pg_a,\quad
h'_i=\operatorname{ReLU}(W_1h_i+b_1+b_{a(i)}^{context}),\qquad
\ell_i=W_2h'_i+b_2.
\]

原11D输入和两个32宽隐藏层保持原权重。新增分支每Anchor只投影一次，没有每Gaussian额外MLP、私有特征或新增Loss。上下文包含当前ratio条件下的可学习几何/外观表示；每次LoD forward重新聚合，不能跨ratio复用旧值。

## 参数与初始化

| 模块 | 参数量 |
|---|---:|
| 原 `local_emd.0`，11→32 | 384 |
| 原 `local_emd.2`，32→32 | 1,056 |
| 原 `soft_net`，32→2 | 66 |
| 新 `selector_context_residual.projection.weight`，17→32无bias | 544 |
| 总计 | **2,050** |

投影权重初始化为零，不修改原权重。加载已有11D模型后，可调用C++ `enableSelectorContextResidual()`或Python `enable_context_residual()`接入；必须在建立/重建含新参数的Optimizer前完成。直接从配置从头训练时，Mapper在建立Selector优化器前创建该模块。

只将投影置零，没有再叠加一个零门控。若原`soft_net`非零，投影在第一次Backward即可获得有效梯度；从头训练时沿用原零`soft_net`，第一步仅输出Head学习，之后投影开放梯度。不得用修改Head初始化或增加训练步数掩盖接入差异。

模块创建保留外部初始化RNG时钟。原11D模型不被静默升级；新模型通过独立`selector_context_residual_schema=1`保存，原`selector_set_encoder_schema`不用于该方案。C++/Python加载和Frozen克隆按实际schema恢复。

## slot、梯度与生命周期

资格沿用`not(slot_inactive and opacity<1/255)`，没有slot deactivation时全员eligible。只在上下文聚合中排除不可选成员，保留原逐点主干行为；最终选点仍由原slot mask限制。全无效或空Anchor的17D上下文为零，无偏置投影保证其残差为零。

Photo/Budget保持原11D输入梯度，并新增eligible成员经上下文参与邻居评分的梯度。不可选成员不能通过上下文改变邻居，也不会从其他eligible成员的评分收到上下文输入梯度。未额外detach隐藏特征或Master。

Maturity Loss的私有参数名单加入`selector_context_residual.*`，继续采用原双VJP及累加顺序，只更新Selector私有参数，不更新Master、Compensator或Anchor补偿Feature。

每次forward读取当前owners、Opacity及slot状态；compact输入通过全局source_rows查询。没有上下文跨步缓存，不在查询中改变slot停用/恢复历史。Clone/Split/Prune、重排和Anchor迁移后使用当前成员重新聚合。

## CUDA路径

`src/anchor_context_residual_cuda.cpp/.cu`提供共享前向和自定义Backward：

1. 复用原11D输入打包，计算原首层32D激活。
2. GPU按当前owners建立最多10成员/Anchor的有界分组。排序组内行，避免分组原子到达顺序改变池化顺序。
3. 每Anchor一个Warp求Mean8/Max8/Count1；`A×17`上下文只投影一次，然后广播注入第二层pre-ReLU。
4. 第二层和输出层使用融合kernel。Backward汇总每Anchor残差梯度，保留Mean及Max并列最大值的均分VJP，再叠加原主干/共享输出梯度。

所有分组、上下文、激活由当前Autograd图私有保存，不覆盖其他未完成图。新增缓冲规模有界为`O(A×(10+17+32))`，无新Host计数同步。CUDA执行遵循原Stream；不改变FlexGS随机数生成、采样及K预算定义。

`CUDA.fused_flexgs_selector=0`回退LibTorch残差计算；`CUDA.fused_anchor_input_pack`控制原输入打包。不支持原Dispatcher设备/人口条件时沿用回退入口。Python提供数学一致的参考实现，未将新CUDA kernel绑定到Python。

## 验证状态

独立目录：`research/anchor_context_residual_20261102/`；服务器：`/home/prmi/zwj/experiments/anchor_context_residual_20261102/`。

首批Python4项测试通过：初始化函数、输入VJP与RNG保持；eligible聚合与FP64 gradcheck；邻居/compact/slot隔离；零Head情况下后续投影梯度与Adam更新。

原生CPU/CUDA、checkpoint、Maturity隔离、实际LoD Mapping与性能微测均已完成。本次不启动完整场景训练，不将功能通过或子模块加速等同于在线质量提高。

## 最终验证结果

| 验证 | 结果 |
|---|---|
| 独立构建 | CUDA 11.8 / RTX3090；核心库、`tum_rgbd`、`tum_mono`及测试编译通过 |
| 原生集合与模型 | eligible Mean8/Max8/Count，2050参数，原权重保持，schema加载与Frozen clone通过 |
| 零投影初始化 | Python原11D输出/输入VJP逐位一致，RNG不前进；CUDA 1031/20000行原融合11D输出及原参数/输入VJP逐位一致 |
| 非零残差CUDA前向 | 相对参考实现最大绝对差`2.98023e-8` |
| 非零残差CUDA VJP | 参数/输入梯度相对参考实现最大绝对差`4.76837e-7`；未宣称非零残差与ATen逐位一致 |
| Python | 7个测试、60项数值对照通过，包括FP64 gradcheck、slot污染隔离、Clone/Prune/迁移/重排及checkpoint |
| 空/全inactive/Max并列 | 0/1/1031/20000行边界、全inactive零投影梯度、Max并列值均分VJP通过 |
| 实际LoD Mapping函数 | 合成地图`N=16`、视锥候选`C=4`；六档全图预算、私有Maturity VJP、Joint Adam实际更新投影、空候选及slot排除通过 |
| 原路径 | 原11D模型30项跨运行库逐位回归、Residual48输入Forward/VJP 97项逐位回归、原集合和21D统计回归通过 |
| 生产保护 | 334个受保护文件SHA256无变化；两批原队列仍暂停 |
| 本地与服务器版本 | 15份patch源文件及67份证据文件哈希核验通过 |

实际LoD Mapping验证使用合成地图，并人工设置输出Head以确保触发选点和梯度路径；它验证接口、梯度隔离和Adam更新，不替代默认初始化后的完整在线训练。

## 性能微测

相同RTX3090、FP32、每Anchor四成员的合成输入；5次预热，每组20次、5组取中位数，CUDA Event计时。投影使用非零权重，Backward包含Master输入与各方案可训练参数；不包括Renderer、Maturity双VJP、选点、拓扑或完整Mapping。

单位：毫秒；单元格为Forward / Forward+Backward。

| 候选Gaussian数 | 原11D融合 | 集合Encoder 50D | 21D统计融合 | 本方案参考实现 | 本方案融合 |
|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.029 / 0.266 | 0.424 / 1.480 | 0.036 / 0.295 | 0.309 / 1.188 | **0.048 / 0.392** |
| 50,000 | 0.116 / 0.366 | 0.653 / 1.522 | 0.148 / 0.448 | 0.468 / 1.235 | **0.144 / 0.420** |
| 100,000 | 0.271 / 0.669 | 1.292 / 2.547 | 0.268 / 0.683 | 0.939 / 1.574 | **0.278 / 0.721** |

本轮融合方案相对集合Encoder的Forward+Backward加速约`3.78 / 3.63 / 3.53倍`。相对原11D的Forward+Backward仍增加`47.3% / 14.7% / 7.7%`，对应绝对增量约`0.126 / 0.054 / 0.052 ms`，不能用大规模结果掩盖小图额外开销。

以上为同机子路径微测，测前GPU已有约6.1GB占用；未锁定时钟，测前/测后时钟分别约1755/1920MHz。跨方案执行顺序和共享GPU负载仍可能影响数字，不能据此认定端到端SLAM加速。真实视锥候选还可能对应更大的全局Anchor表，目前池化和投影按全局Anchor行分配，这一规模差异需在实际Mapping中测量。

## 交付与后续限制

- 独立配置：`cfg/selector/anchor_context_residual48.yaml`，日志应显示`context_residual=1 input_dim=11`。
- 保存模型记录`selector_context_residual_schema=1`、`selector_parameters=2050`、`anchor_context_residual=true`。
- 服务器独立运行库位于`/home/prmi/zwj/experiments/anchor_context_residual_20261102/source/lib/`；主项目未部署新库，不应使用旧生产二进制运行新配置并认为已经启用。
- 完整证据见`research/anchor_context_residual_20261102/evidence/`，下载包SHA256为`09d0321d628478b5a1908d5dd4fc7b9ec89a113be7a726e0e083e9d4064b5220`；本地核验见`LOCAL_VERIFICATION.json`。
- 没有完成场景PSNR/SSIM/LPIPS验证，尚不能声称优于11D或集合Encoder。下一步应使用匹配seed、帧序、调度与训练曝光的在线对照，重点观察1%/5%质量、实际K、投影梯度及Mapping耗时。
- 本轮未切换生产默认、未恢复暂停队列、未commit/push。
