# Anchor集合上下文Selector实现

## 启用方式

新配置：`cfg/selector/anchor_set_residual48.yaml`。

```yaml
Selector.anchor_set_encoder: 1
Selector.anchor_frustum_culling: 1
```

集合编码配置以当前Residual48基线为基础，仅开启新的Selector表示。原`selector_config.yaml`和`spatial_offset3_residual48.yaml`中的`anchor_set_encoder`为0，便于保留原11D对照；Anchor名义格子视锥筛选保持开启。本轮没有将尚未完成质量实验的网络替换成服务器生产默认。

## 数学定义与尺寸

一个LoD步骤先选择与相机视锥相交的Anchor，取其全部现存成员；集合聚合仅使用现有slot规则判定为eligible的成员，之后沿用原FlexGS采样。成员不按各自视锥位置二次过滤。

对于Anchor `a`，现存成员集合为`A_a`、其中slot-eligible集合为`E_a`、有效数量为`m_a=|E_a|`、名义格子中心为`c_a`、参考格子边长为`h_a=offsetScale_a`：

\[
 d_i=(x_i-c_a)/h_a,
\quad u_i=[d_i^3,s_i^3,c_i^3,\alpha_i^1]\in\mathbb R^{10}.
\]

`s`使用现有activated Scale，`c`使用SH DC，`α`使用activated Opacity。没有改变这些属性的激活方式，没有加入Fourier、Rotation、相机方向或新的私有Embedding。

\[
h_i=\phi_\theta(u_i),\quad i\in E_a,\qquad\phi:\ 10\to16\to16,
\]

两层Linear后均为ReLU。不可选成员在网络输入和编码结果处均置零，不参与Mean/Max/Count。按Anchor聚合：

\[
g_a=[\operatorname{Mean}_{j\in E_a}(h_j),\operatorname{Max}_{j\in E_a}(h_j),m_a/10]\in\mathbb R^{33}.
\]

\[
\ell_i=\psi_\omega([h_i,g_{a(i)},\log r]),
\quad\psi:\ 50\to32\to32\to2.
\]

评分主干仍使用两层ReLU，输出仍为FlexGS keep/drop logits。ratio只作为原`log(r)`标量进入评分MLP；集合编码及33D上下文本身与ratio无关。

名义格子中心offset只作为共享成员Encoder的内部输入，评分网络读取编码后的成员与集合特征，没有平铺全部成员offset。单成员Anchor仍保留局部位置。全无效Anchor的Mean、Max和Count均为零；其成员由原slot规则排除。slot资格是纯查询：`not(slot_inactive and opacity<1/255)`，不因低Opacity本身额外排除成员，不在Encoder前向中更新停用/恢复历史。

## 参数与初始化

| 参数模块 | 尺寸 | 参数量 |
|---|---|---:|
| `selector_set_encoder.network.0` | 10→16 | 176 |
| `selector_set_encoder.network.2` | 16→16 | 272 |
| `local_emd.0` | 50→32 | 1,632 |
| `local_emd.2` | 32→32 | 1,056 |
| `soft_net` | 32→2 | 66 |
| 总计 | 全场景共享 | **3,202** |

原11D评分网络为1,506参数，新Selector增加1,696参数。Compensator及既有Anchor补偿Feature表保持原结构；没有新增Anchor/Gaussian私有可训练参数。

新Encoder和50D评分主干使用LibTorch Linear默认初始化。保留原零Feature Selector的零初始化`soft_net`，使step0 logits为0；第一步更新soft_net后，Encoder逐步获得学习信号。模块构造通过`ScopedTorchRNG`保留外部初始化随机数时钟。

新Selector第一层从11D变为50D，不能把旧11D权重视为同函数初始化。新结构应从头训练或显式重新初始化Selector；旧checkpoint使用关闭开关的旧路径。

## 梯度与优化器

- Encoder属于Selector私有模块，沿用Selector学习率及共享Adam中的Selector参数组，无新学习率或Loss。
- Photo/Budget保留Joint输入梯度：eligible成员的局部坐标、Scale、SH DC和Opacity参与autograd。一个eligible成员可通过Anchor上下文收到其他成员评分的梯度；未detach上下文输入。不可选成员的Encoder输入梯度为零。
- Maturity相对Loss继续仅对Selector私有参数求VJP，私有名单新增`selector_set_encoder.*`。成熟度detach；Maturity不会更新Master、Compensator或Anchor补偿Feature表。
- 两次VJP及其累加顺序沿用原实现。保留前一轮小候选MLP返回view梯度时的包装解除修复。
- 训练与推理的FlexGS规则不变。训练预算仍从全图长度N的Mask计算`|sum(mask)-rN|/max(1,rN)`，并非`r×候选数`；不保证概率采样每步严格命中目标K。

## 索引、生命周期及兼容性

- compact输入必须传全局`source_rows`，用其取得当前`owners`。上下文仅在输入包含的完整Anchor成员集合内计算。
- 当前视锥筛选以Anchor为单位且`Elastic.entry_maturity=0`，保证主训练/逐视图推理输入包含通过Anchor的全部成员。新配置禁止同时使用逐成员成熟度准入，以免产生不完整集合。
- slot资格在集合聚合与最终采样中保持一致；compact输入通过全局`source_rows`查询当前slot状态。
- Clone/Split/Prune/重排/Anchor迁移后每次forward重新聚合当前成员；无成员特征或质心跨步缓存，无新增拓扑优化器状态。
- 新模型使用`selector_set_encoder_schema=2`：名义格子相对坐标、eligible聚合。C++和Python读取schema 1时保留旧质心坐标及全部成员聚合语义，不静默升级；Frozen teacher沿用源模型schema。原11D checkpoint不受影响。
- 配置保存、`anchor_selector_state.json`记录开关、50D输入、3,202参数及集合定义。旧checkpoint保持11D加载路径。

## 性能与适用边界

当前原型使用LibTorch `index_add`计算数量、质心和Mean，`scatter_reduce(amax,include_self=false)`计算Max。没有逐Anchor Python/C++循环，没有CPU成员遍历或逐步Host计数同步。

额外临时量规模为`O(N_candidates×16 + Anchor数×16)`，总计算对成员数线性增长。1～10成员均产生固定33D上下文；额外空Anchor行不影响已占用Anchor。

新的50D评分路径自动避开仅支持11D的专用CUDA Selector MLP。FlexGS采样、Compensator融合、渲染和其他既有优化继续使用原开关。没有声称新网络与原网络数值或速度相同。

Mean的FP32累加可能因成员重排或CUDA原子顺序产生微小舍入差异，数学上满足排列不变性，验证应报告实际误差而非宣称重排后bitwise必然一致。

后续性能优化可聚焦按Anchor一次计算质心/Mean/Max并广播，且必须保留质心反向与Max并列值的梯度语义。当前阶段不重写未测量的CUDA路径。

## 原schema 1的验证证据

验证目录：`research/anchor_set_selector_20261102/`；服务器独立目录为`/home/prmi/zwj/experiments/anchor_set_selector_20261102/`。原两批训练保持暂停，服务器主项目不在构建输出范围。

| 验证 | 实测结果 |
|---|---|
| 独立CUDA 11.8 / RTX3090构建 | 新旧Selector、`tum_rgbd`、`tum_mono`编译通过 |
| C++ CPU / CUDA集合测试 | 1/2/3/4/10成员、Mean/Max/数量、空集合、成员排列等变性、质心平移不变性、邻居属性影响、Anchor重排、输入/参数梯度、3,202参数检查通过 |
| 原生模型生命周期 | 新模块schema保存加载、Frozen teacher克隆；实际Mapping中保存的含成员/Compensator模型由C++重载、Python加载后输出对照通过 |
| 实际LoD Mapping | 全图16点、4个候选成员；多成员Anchor上下文、六档全图预算、候选外Master零梯度、空候选、inactive slot、名义格子成员规则、配置读回通过 |
| Maturity隔离 | 仅Maturity项时新Encoder及评分网络获得梯度，Master/Compensator/既有Anchor Feature梯度为零 |
| Joint Adam | Photo/Budget联合反向有限，Master保留梯度，新Encoder首层权重实际更新 |
| Python CPU / CUDA | Clone/Prune/成员迁移、ratio无关上下文、邻居梯度、空Anchor表、新旧checkpoint、完整Anchor子集、逐点重排测试通过 |
| C++ / Python数值对照 | 66项对照通过；当前实测最大绝对误差`5.96e-8`，包括前向、输入VJP、Encoder参数VJP和加载后logits |
| 原路径回归 | 原11D Mapping、Selector handoff CPU/CUDA、Residual48的97项Forward/VJP逐位检查、原Python视锥7项测试通过 |

本轮CUDA Pooling诊断：相同输入重复前向最大差`2.98e-8`；只改变一个Anchor后其他Anchor最大差`2.98e-8`；Anchor重排最大差`5.96e-8`。CPU对应三项均为0。CUDA浮点原子求和顺序导致这些微小差异，未把数学排列不变性写成bitwise承诺；跨Anchor反向梯度隔离另行验证为零。

测试中发现的C++ const接口编译问题已修正。最初用bitwise检查CUDA Pooling重排的断言失败，保留失败日志，增加重复前向误差对照后改为有记录的FP32误差验证；没有改动网络计算来掩盖失败。

服务器主项目334个受保护文件SHA256前后一致，两批队列仍为`paused=true`。最终核验见`research/anchor_set_selector_20261102/evidence/production_integrity.json`、`queues_after.json`；全部原始日志和数值对照保存在同目录。

此次交付为功能实现与正确性验证，不包含完整场景质量实验或端到端性能Benchmark，不据此声称改善PSNR或提速。未替换服务器生产运行库，未恢复暂停队列，未commit/push。

## 名义格子与slot资格修正

当前默认创建的集合网络为schema 2。修正验证独立保存在`research/anchor_set_selector_nominal_eligible_20261102/`，与上文schema 1历史证据分开；最终结果见`ANCHOR_SET_SELECTOR_NOMINAL_ELIGIBLE_REPORT.md`。参数量和输入维度保持不变。
