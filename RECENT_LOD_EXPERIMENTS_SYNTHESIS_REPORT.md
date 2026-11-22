# Anchor LoD近期实验总报告：质量瓶颈、证据与下一步

日期：2026-10-22。项目：mycartgs。源码基线：`02cd92ff4fcab299008ba094be6966b70e842061`。

本报告汇总近期Feature容量、输入长尾、split修复、预算梯度、Selection/Compensation Oracle、Joint、Gaussian-private补偿、ratio调制和属性拆头实验。**本轮只整理与复核已有证据，新增此报告，没有重新训练或修改生产算法、配置、Git暂存区。**

## 1. 核心结论

**目前最有依据的判断是：固定地图下存在明显的补偿可恢复空间，但现有“共享特征→共享补偿函数”的参数化和优化方式，没有有效利用这部分空间。现有证据不支持将问题简单归结为MLP太浅，也不支持直接加深、拆头或移除共享梯度。**

| 核心问题 | 当前判断 | 主要证据 | 证据边界 |
| --- | --- | --- | --- |
| 地图本身是否已没有提升空间？ | 否，仍有明显可恢复空间 | 固定集合的Independent14D补偿使旧non-Bank平均PSNR提高约0.806 dB | 补偿允许移动实际渲染几何；不是严格固定成像几何的上界 |
| Selector排序是否是最主要问题？ | 尚未证明 | 有限换点搜索平均仅约+0.023 dB，1%约+0.067 dB | 搜索覆盖很小，不能宣称Selector已最优 |
| 补偿是否受限？ | 是，但限制尚未被唯一定位 | 独立14D显著优于等步数原头，4D/8D只恢复约15% | 参数共享、输入信息、有效方向、优化条件同时变化 |
| 主要原因是否是网络层数不够？ | 没有支持 | 加一层、扩大Feature、统一头加宽、四属性拆头均无稳定收益 | 只否定已测结构和训练协议，不能证明所有更大网络都无效 |
| 负梯度或ReLU死区是否足以解释失败？ | 不足以解释 | 死区真实存在，但bypass恢复梯度后质量更差；若干detach干预也失败 | 梯度指标是诊断，不是质量指标 |
| Ratio-conditioned私有投影是否有效？ | 本次线性形式不支持 | 新确认集训练档仅+0.021 dB，未见档−0.116 dB，明显弱于Constant控制 | 仍不能据此否定所有ratio条件化方法 |
| Joint是否存在Full竞争？ | 在已测离线协议下存在 | 同Full更新次数下，Joint Full比Full-only低约0.550 dB | Adam时钟、fresh状态与在线训练不同，尚未隔离全部原因 |

**当前建议：保留生产16维Anchor Feature、27维输入、原统一14维补偿头及现有梯度路由。下一轮优先做诊断矩阵，暂不把预算移出共享层、不减弱约束、不加深网络、不上线Ratio-conditioned或四分支补偿。**

## 2. 当前真实系统与实验版本

本轮读取生产配置和源码，确认当前默认路径：

| 模块 | 当前实现 |
| --- | --- |
| 地图 | Variable Anchor，每Anchor 1～10个显式Gaussian；多层Anchor粗细共存 |
| 基础尺度与层级 | h0=0.1，L0/L1/L2 |
| 地图增长 | Gaussian clone/split与Anchor growing交替，保留容量与归属管理 |
| Pruning | 仅Anchor窗口质量pruning；Gaussian split父替换仍执行；保留slot停用机制 |
| 输入 | Feature16 + Offset3 + activated Scale3 + SH DC3 + activated Opacity1 + log(ratio)1，共27维 |
| 共享Encoder | 27→32，ReLU |
| Selector | 32→32→2，ratio-aware、视角无关，Cached Bernoulli |
| 补偿 | 选中后，32→32→14；XYZ3、Scale3、Rotation4、Opacity1、SH DC3 |
| Joint | LoD更新Master、Selector、Anchor Feature、Compensation；Full更新Master |
| 默认关闭 | TF、ChangeDistill、SI/GI、ratio-feature表、global XYZ输入、串行Full/LoD、额外Full尾优化 |
| 默认停止 | max_mapping_iterations=0，保留原自然停止，不强制10000 |
| split采样 | 两个split入口均已使用尺度旋转因子L，而非协方差Σ |

最近冻结地图实验共用源checkpoint：

`/home/prmi/zwj/budget-shared-audit-20261022/baseline_seed2026/6900_shutdown/ply`

该地图有 **13,919 Gaussian、5,177 Anchor**，不是固定十点地图。L0/L1/L2分别为1,410/3,565/202个Anchor，归属Gaussian分别4,611/9,059/249；全部Anchor非空。2692个Anchor仅1成员，满10成员为356个。slot停用244个，当前eligible为13,675个。

不要把该源地图与早期固定十点、7,099点可变成员地图、10,000步11,974点地图或split修复的7,400步地图直接混作同一组质量对照。相同scene名称不代表相同几何、位姿、优化次数或点数。

证据：[当前配置](H:/AIProjects/cartgs优化更细致版/mycartgs/cfg/selector/selector_config.yaml)、[生产补偿头](H:/AIProjects/cartgs优化更细致版/mycartgs/improvements/selector/gumbel_network.h)、[Gaussian split](H:/AIProjects/cartgs优化更细致版/mycartgs/src/gaussian_model.cpp)、[架构与源checkpoint审计](H:/AIProjects/cartgs优化更细致版/mycartgs/RENDER_QUALITY_BOTTLENECK_AUDIT.md)。

## 3. 必须统一的统计口径

### 3.1 三类实验分别说明了什么

| 类型 | 可控制项 | 可以回答 | 不能直接回答 |
| --- | --- | --- | --- |
| 在线SLAM重新训练 | 配置、seed、规则；最终地图和异步轨迹仍可不同 | 方法在真实流程中的综合表现 | 单独某一网络因素的严格因果收益 |
| 冻结地图、固定集合补偿实验 | Master、Anchor、Encoder/Selector、相机、选中ID和K | 给定地图与集合时，补偿参数化的可恢复质量 | 新结构在线Joint、增长和闭环后的完整表现 |
| 固定拓扑离线Joint续训 | 初态、帧顺序、Full事件次数、拓扑 | 梯度路由及Full/LoD取舍 | 原始在线Adam/轨迹/拓扑全复现 |

### 3.2 Same-K不等于统一的概率Top-K

生产选择为`S={i:p_i(r)>u_i}`，u为缓存uniform。冻结地图主实验固定原Cached实际集合，训练档K为：

| ratio | 1% | 2.5%（未见） | 5% | 7.5%（未见） | 10% | 12.5%（未见） | 15% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 实际K | 138 | 343 | 697 | 1,052 | 1,365 | 1,749 | 2,021 |

相同K的诊断排序使用原始logit差减`logit(u)`，非eligible置−∞，tie按行号；不能用已饱和的FP32概率反推logit。原Cached的实际K下，该排序精确复现原集合。跨不同在线地图的same-K可能还用K/N作为网络ratio输入，须按各轮协议解释，不能混同固定原ratio实验。

`selected`是选中的存储点；`visible`是radii有效点；`contributing`是至少实际参与一个像素累积的点。K−contributing包含视野外、遮挡、截断等因素，**不能全叫低opacity预算浪费**。

### 3.3 视图与seed

- 冻结地图主实验：70 Bank训练，320×240；正式图像质量评价640×480。
- V0：旧44 non-Bank，之后多次用于结构研究，现属于复用诊断集。
- V1：Ratio-conditioned实验前锁定的32个新确认视图，排除旧评价和历史Bank；属性拆头实验也复用它。
- V2：属性拆头实验前另锁定的32个确认视图，再排除V1和此前集合。**D/E没有在V2上评价，不能宣称它们在V2重复成功或失败。**
- V1/V2均来自同一场景、相邻轨迹，并非独立场景或独立训练地图；评价图像也可能在SLAM建图过程中被观测过。non-Bank主要表示未用于本轮离线优化，不等于地图训练全程未见图像。
- 2026/2027在近期冻结实验中只改变优化视图顺序，不是两份独立SLAM地图。在线重新训练的两个seed与这两个优化顺序seed应分开表述。
- 原Shared头在每个ratio独立续训，与单个Shared头联合四ratio训练是两种不同控制。约0.609 dB是Oracle相对前者的差距；近期Recovery以联合Shared为分母，不能直接拿0.609套用。
- 多批结果包含复用模型、复用Oracle和复用评价记录；记录总数不能当成独立样本数量相加。

## 4. 已完成实验全景

Δ均按各轮自己的匹配控制计算；表内不同轮的数值不能相加。

| 实验 | 场景/类型 | 主要结果 | 当前决策 |
| --- | --- | --- | --- |
| Anchor Feature16→32 | 在线，fr1_xyz单seed、两组10000 Mapping | same-K平均PSNR−0.233 dB，2/4预算PSNR/photo同时改善 | 不以增加Feature宽度作为质量修复 |
| 共享8+每ratio8特征 | 在线，两seed | same-K−0.223 dB；插值ratio名义结果−0.691 dB，同时点数变化 | 不认定专属表/特征插值有效；当前默认已回16维 |
| 补偿头加一层32 | 旧地图冻结、固定集合、两优化顺序 | 平均+0.000038 dB；一seed正、一seed负 | 没有加深收益证据 |
| split尺度采样修复 | 在线两seed、四次训练 | Full平均+0.394 dB；same-K PSNR+0.165 dB，但SSIM/LPIPS等混合 | 保留数学修复，不宣称LoD四指标全面提升 |
| 预算只更新选择私有层 | 在线两seed | same-K平均−5.082 dB，ratio响应塌缩 | 不采用 |
| 预算权重0.01→0.003 | 在线两seed | 名义1%部分改善伴随超预算；same-K平均−0.660 dB | 不采用 |
| 补偿不回传共享层 | 在线两seed | same-K平均−0.099 dB，seed冲突 | 不采用 |
| 预算私有Feature Oracle | 固定地图、五臂×两顺序 | 私有Feature可恢复预算响应，但主要目标比等步数原版低0.151 dB | 能承接约束不等于更高质量，HOLD |
| Selection Oracle | 固定图、真实render接受swap | 四档平均约+0.023 dB；1%+0.067 dB | 排序仍待研究，但本搜索证据弱于补偿 |
| Independent14D Compensation Oracle | 固定图、固定K、两顺序 | 旧non-Bank相对源MLP+0.806 dB；比单ratio续训头高0.609 dB | 最强的可恢复质量证据 |
| Private4D/8D | 固定图，单模型联合四ratio | 旧non-Bank+0.119/+0.117 dB，恢复约15.1%/14.2% | 小收益真实，未接近Oracle，未见ratio不稳定 |
| Private4D ReLU bypass | 同参数、同协议 | 取消死区；旧non-Bank训练档仅+0.032 dB，未见档−0.028 dB | 梯度修复没有转化为质量突破 |
| D Constant4 | 固定图，自由14×4投影 | V1训练档+0.119 dB，未见档+0.094 dB，恢复14.46% | 有限正面控制，未突破旧4D水平，HOLD生产接入 |
| E Ratio-conditioned4 | 固定图，线性log-ratio投影 | V1训练档+0.021 dB，未见档−0.116 dB，恢复1.38% | 本形式NO-GO |
| 四属性分支 | 固定图，六次训练含控制 | V2训练档−0.027 dB，未见档−0.044 dB，seed冲突 | 不替换原统一头 |
| 相近参数量统一宽头100 | 与四分支匹配 | V2训练档+0.004 dB、未见档−0.005 dB | 无稳定增容收益 |
| Joint / Full-only / Frozen-Master | 固定拓扑离线续训 | Joint比Full-only的Full低0.550 dB；比Frozen-Master的LoD高0.339 dB | 存在取舍，不能简单关闭LoD→Master |

早期固定十点→可变成员、Anchor-only pruning等属于地图生命周期背景；它们确实改变人口和低opacity组成，但不是当前冻结地图补偿实验的匹配控制。其质量证据也不能合并归因于MLP。

## 5. 问题具体发生在哪里

### 5.1 已确定的工程问题：split尺度语义错误，现已修复

旧split使用`Δx=Σz`，其中`Σ=LLᵀ`，因此扰动协方差与Σ²成比例；正确尺度因子形式是`Δx=Lz`。保留原3σ置零噪声后，比例常数κ约0.970709：旧版κΣ²，修复版κΣ。

历史10,000步模型出现Offset分量210.6，1%位移/Anchor边长p90为3.37；短程出生快照实际捕获到子点从正常范围跳到极大Offset，支持真实生成机制，而非仅统计或推理错误。

修复的两seed对照中，极端新生Offset得到抑制，原一组368降到68，Offset>100点从8降到0；但一般尾部并未整体收窄。修复不保证所有Gaussian留在Anchor cell内，普通XYZ优化和继承归属也会继续产生漂移。

**不能直接clamp或删除长尾。** 在旧模型上，1%中20个长尾点被选中，占K的16.95%；移除它们使non-Bank PSNR下降3.655 dB。此操作改变有效K，只是条件贡献诊断；它说明这些点有用，不证明极大Offset是理想表示。

当前代码已经使用L采样。后续实验不能继续把已修复问题当成所有补偿不足的唯一解释；历史checkpoint也不会因代码修复自动重新定位。

证据：[输入长尾与容量审计](H:/AIProjects/cartgs优化更细致版/mycartgs/MLP_GRADIENT_EXTRAPOLATION_CAPACITY_AUDIT.md)、[split修复独立对照](H:/AIProjects/cartgs优化更细致版/mycartgs/SPLIT_SCALE_SAMPLING_FIX_REPORT.md)。

### 5.2 预算与共享表征真实耦合，但当前耦合也承担有效功能

当前LoD目标：

```text
L = L_photo + 0.01 |Σ p_i − rN| / max(1,rN)
```

预算经选择网络影响共享Encoder、Anchor Feature，也能通过Offset/Scale/SH DC/Opacity输入影响Master；补偿头虽然不直接接收预算loss，读取的共享h仍会变化。

对同一档位，预算对logit施加同方向的概率增减压力；与选择photo梯度出现负cosine可能是在履行数量约束，而非反向传播错误。跨ratio预算误差符号不同也会自然产生相反方向。

三个实际干预表明：直接移走预算共享梯度会破坏ratio响应，降低预算权重容易通过多选点改善名义ratio，切断补偿共享梯度也没有稳定收益。预算私有Feature虽然把预计K平均误差恢复到约0.4%～0.6%，质量仍低于等步数原版。

**已证明“有牵制”，尚未证明“删掉牵制会更好”。** 不能仅凭梯度负cosine就决定detach、减预算、做梯度手术或加深MLP。此前提出的expected-K implicit校准仍是另一个未验证方案；它改变训练/推理数值，不能在本报告中视为已有效。

证据：[预算共享梯度对照](H:/AIProjects/cartgs优化更细致版/mycartgs/BUDGET_SHARED_GRADIENT_COUPLING_AUDIT.md)、[预算私有Feature Oracle](H:/AIProjects/cartgs优化更细致版/mycartgs/PRIVATE_BUDGET_FEATURE_ORACLE_AUDIT.md)。

### 5.3 固定地图下，最明显的可恢复质量在补偿，但并非“14种属性不够”

旧44 non-Bank、固定当前集合和相同K：

| ratio | Independent14D−源MLP | 单ratio原头续训−源MLP | Independent14D−等步数单ratio头 |
| --- | --- | --- | --- |
| 1% | +1.1851 dB | +0.3706 dB | +0.8144 dB |
| 5% | +0.8430 dB | +0.2395 dB | +0.6035 dB |
| 10% | +0.5607 dB | +0.0819 dB | +0.4788 dB |
| 15% | +0.6342 dB | +0.0934 dB | +0.5408 dB |

独立14D沿用完全相同的属性语义/变换与photo，四指标和两顺序均改善。因此无需新增属性类型就能恢复显著质量；限制更接近“怎么从共享表示得到每Gaussian、每ratio的补偿”。

但Oracle同时放宽了Gaussian共享和ratio共享，又改变优化条件，不能据此唯一认定输入特征、维度或层数。源Master冻结也不代表实际渲染几何冻结：Oracle可改变XYZ/Scale/Rotation，1%位移/h p90约3.9～4.1，几何覆盖会随之重排。

条件属性恢复中XYZ、Rotation影响最大，Scale次之；这些值包含协同，不能把XYZ、Rotation损失相加作为总可恢复收益。1% Oracle仍比源Full低约4 dB，说明Oracle也没有解决所有稀疏表示限制。

### 5.4 4D特征“有梯度”不等于“有足够的有效个体方向”

旧私有注入形式：

```text
δ = W2 ReLU(W1h(r)+b1+Uz_i)+b2
J_z = W2 diag(ReLU_active) U
```

15%初始有1,653个点存在直接14D photo梯度，其中549个被私有ReLU死区挡住；训练后部分高ratio的J_z秩中位数仅约1。增加z到8维不自动增加有效补偿方向。

但bypass取消死区后，J_z恢复rank4、真实上游梯度可传到z，质量仍较差；因此死区不是剩余Oracle差距的充分解释。训练union为2,021点，原4D约82%～83%曾获得有效梯度；从未选中的行保持零是预期，不应强行要求全N都有补偿梯度。

**还需区分私有Jacobian与整个模型能力。** rank4只限制单点z的局部自由度，C(h)的共享权重仍可更新；不能由私有rank4直接宣称整个14D输出只有四维能力。

证据：[Gaussian-private补偿审计](H:/AIProjects/cartgs优化更细致版/mycartgs/GAUSSIAN_PRIVATE_COMPENSATION_AUDIT.md)。

### 5.5 线性ratio调制实际学到了变化，却降低了泛化

定义原共享头`C(h(r))`，每Gaussian一份`z_i∈R4`：

```text
D Constant4:         δ_i(r) = C(h_i(r)) + V0 z_i
E Ratio-conditioned: δ_i(r) = C(h_i(r)) + [V0 + φ(log r)V1] z_i
φ(r) = 2[log(r)−log(0.01)]/[log(0.15)−log(0.01)]−1
```

D的“Constant”仅指私有残差不随ratio变，原共享头仍ratio-aware。旧bypass则为`C(h)+W2Uz`，其投影与共享输出层耦合，不等同于自由V0。

V1=0时E精确包含D。两组初始函数相同，实际z梯度正常；E最终`||V1||/||V0||`约1.11/1.47，显著使用了调制，不是“没有学出ratio”。但E在V1确认集5%/10%训练节点下降，三个未见ratio两seed均下降。

固定同一个E模型、同一真实photo上游梯度，仅比较1%与15%的代数VJP：通过V0回传时全局cosine约0.284、负方向行20.5%；通过V(r)回传时约0.035、38.9%。这比直接比较不同模型的cosine更能隔离投影影响，支持调制加大共享z跨ratio牵制；仍非质量退化唯一原因的因果证明。

直接清零训练后E的V1会再降低质量，说明C/z/V0/V1已经共同适配；**不代表E优于独立训练的D**。E包含D但在当前训练协议下更差，至少说明“容量不足”不是解释E弱于D的充分理由，优化条件和泛化必须一起研究。

### 5.6 Oracle低秩分析有启发，但不能直接决定网络维度

四档共同选中138点上，采用XYZ/Anchor尺度、log-scale、SO(3)相对旋转、激活opacity、SH DC进行物理分组，避免将原始四元数欧氏距离当图像误差。

- 分属性rank4解释约83%～96%的跨ratio变化能量；将各属性按RMS归一化联合后，rank4仅解释约48%。
- 任意逐点线性log-ratio拟合解释约69%～71%的归一化跨ratio变化；再要求共同Gaussian潜变量rank4，数值近似误差仍明显。
- 独立Oracle跨优化seed也有差异；这些目标不是唯一、必然平滑的函数。
- 该分析仅覆盖共同点和特定物理参数化，不是新模型`C(h)+Vz`的严格能力上界，更未证明48%的参数能量等于48%的质量收益。

因此它支持“多属性共同4D可能有结构限制”，但不足以直接要求增加维度或更复杂调制。

证据：[Ratio-conditioned审计](H:/AIProjects/cartgs优化更细致版/mycartgs/RATIO_CONDITIONED_PRIVATE_COMPENSATION_AUDIT.md)。

### 5.7 属性分支拆分缓和部分梯度，却没有质量收益

四个独立`32→32→输出`分支分别预测XYZ3、Scale3、Rotation4、Opacity+SH DC4。不是只把同一输出Linear拆开；各自隐藏层也独立。参数从1,518增至4,686；相近参数量控制`32→100→14`有4,714参数。

公共h上的XYZ/Rotation梯度cosine在1%从约−0.228变为−0.010，在15%从约−0.182变为−0.050，但新确认集质量没有提升，Scale/Rotation等其他冲突仍存在。h本轮被冻结，不能由该结果断言生产共享Encoder上的冲突已经被解决。

Split在两个seed间方向相反；宽头增益也接近重复波动量级。拆头与加宽并未证明“所有容量方案无效”，但已经削弱“再堆一点容量就能解决”的解释。

证据：[属性分支审计](H:/AIProjects/cartgs优化更细致版/mycartgs/SPLIT_COMPENSATION_HEAD_AUDIT.md)。

## 6. 近期补偿模型的可比质量表

### 6.1 V0旧44帧：训练ratio四档的四指标

以下为两seed×四ratio等权均值。前半组使用同一批联合Shared；后半组使用本轮重新训练Shared32。**两种Shared控制不完全逐位重复，不能把跨批次差值当成纯结构因果结果。**

| 模型 | 控制 | ΔPSNR↑ | ΔSSIM↑ | ΔLPIPS↓ | Δphoto↓ |
| --- | --- | --- | --- | --- | --- |
| Private4D | 联合Shared | +0.1188 | +0.001934 | −0.003567 | −0.001099 |
| Private8D | 联合Shared | +0.1170 | +0.001932 | −0.004154 | −0.001061 |
| Private4D bypass | 联合Shared | +0.0318 | +0.001460 | −0.004023 | −0.000715 |
| D Constant4 | 联合Shared | +0.1341 | +0.002149 | −0.006370 | −0.001237 |
| E Ratio-conditioned4 | 联合Shared | +0.0286 | +0.000547 | −0.002457 | −0.000340 |
| 四属性分支 | 新训练Shared32 | −0.0162 | +0.000396 | +0.000388 | +0.000036 |
| 统一宽头100 | 新训练Shared32 | +0.0068 | +0.000491 | +0.000377 | −0.000100 |

### 6.2 V1上一轮32帧：私有方案的训练与未见ratio

| 模型 | 训练4档ΔPSNR | 未见3档ΔPSNR | 训练档Oracle恢复率 |
| --- | --- | --- | --- |
| Private4D | +0.1116 dB | +0.0271 dB | 14.04% |
| Private4D bypass | +0.0361 dB | −0.0289 dB | 3.03% |
| D Constant4 | +0.1187 dB | +0.0940 dB | 14.46% |
| E Ratio-conditioned4 | +0.0211 dB | −0.1156 dB | 1.38% |

D表现较稳，但训练档比原4D仅高约0.0071 dB，部分四指标并非同步胜出；没有达到“接近Oracle”的目标，不应称为突破。恢复率按每seed/ratio先算再平均，分母接近零不算、负值保留；不能用平均增益直接相除代替原定义。

### 6.3 V2本轮新32帧：拆头与统一宽头

| 模型 | 训练4档ΔPSNR | 未见3档ΔPSNR | 训练档Δphoto | 未见档Δphoto |
| --- | --- | --- | --- | --- |
| 四属性分支 | −0.0266 dB | −0.0435 dB | +0.000148 | +0.000272 |
| 统一宽头100 | +0.0043 dB | −0.0053 dB | −0.000083 | −0.000076 |

Split seed2026训练/未见均值−0.0836/−0.1042 dB，seed2027为+0.0304/+0.0171 dB。重新训练Shared与此前Shared在单ratio/seed的PSNR均值差最大约0.0224 dB；这说明微小改进必须配合基线重复与更多独立地图判断，不能从+0.004 dB得出结构有效结论。

## 7. Joint的限制是质量取舍，不是已证明全局梯度反向

固定拓扑、同初始状态、同视图序列、每seed400次Full更新；Joint与Frozen-Master各有1,600次LoD辅助更新：

| 分支 | Full PSNR | 1% PSNR | 5% PSNR | 10% PSNR | 15% PSNR |
| --- | --- | --- | --- | --- | --- |
| 源checkpoint | 21.3272 | 16.1384 | 19.2877 | 20.0505 | 20.1638 |
| Full-only，辅助冻结 | 21.6411 | 15.5678 | 19.0598 | 19.8272 | 19.9590 |
| 当前Joint | 21.0913 | 16.2885 | 18.8791 | 19.5354 | 19.6135 |
| Full更新Master，LoD仅辅助 | 21.6408 | 15.6973 | 18.5605 | 19.2540 | 19.4471 |

Joint Full比Full-only低0.5498 dB，两seed均出现；Frozen-Master保护了Full，但LoD四档比Joint平均低0.3394 dB。Joint相对源状态也并非四档PSNR均提升。

同分辨率Full/LoD的初始整体梯度cosine四档中位数约0.173/0.416/0.529/0.544，32个测量均正；部分属性可负。因此“Full与LoD有质量竞争”不能简化为“所有梯度互相抵消”。

关键局限：源导出没有原Adam moments与完整训练时钟；使用fresh Adam及固定XYZ spatial scale代理。Joint的Master Adam实际推进2,000次，其他分支400次。**处理差异包含额外LoD梯度和Adam时钟，尚未独立隔离。** 资格mask也可随opacity恢复变化。

结论是值得进一步隔离Master/辅助优化取舍，而不是立刻永久冻结Master或删除Joint。

## 8. 额外时间和显存是否值得

同一RTX3090、相同N/K、缓存K行h的补偿头微基准。数值含Python eager启动间隙；不含Encoder、Selector、Renderer、optimizer step或完整Mapping。不同实验批次的绝对时间不可过度精确横比。

| 方案 | 测试批次 | 15% Forward | 15% Backward | 参数量 | 质量判断 |
| --- | --- | --- | --- | --- | --- |
| 原Shared | 私有调制批次 | 57 μs | 230 μs | 1,518 | 控制 |
| D Constant4 | 同批 | 115 μs | 353 μs | 57,250（含N行z） | 小幅增益，未接近Oracle |
| E Ratio-conditioned4 | 同批 | 147 μs | 399 μs | 57,306 | 比D更慢、质量更差 |
| 原Shared32 | 拆头批次 | 60 μs | 224 μs | 1,518 | 控制 |
| 四属性分支 | 同批 | 249 μs | 687 μs | 4,686 | 未见稳定收益 |
| 统一宽头100 | 同批 | 66 μs | 238 μs | 4,714 | 近零收益 |

4D全N私有特征222,704 bytes，Adam m/v为445,408 bytes，合计约0.637 MiB，另有头部/投影与激活。E相对Shared的Torch峰值约增加0.85 MiB，拆头相对其控制约增加0.81 MiB；这些是驻留实验数据上的allocated差值，不是整卡或完整SLAM显存。

四档独立Oracle实际仅存选中行：14×(138+697+1365+2021)=59,094个参数，和完整N行E的57,306并没有数量级差距。若私有表只存训练union，可明显减少参数，但需要身份映射与在线生命周期；此前有固定地图精确压缩验证，未完成生产集成。**不能将理论稀疏存储收益当作已测在线性能。**

## 9. 哪些解释已经排除，哪些仍需验证

| 解释 | 当前状态 | 原因 |
| --- | --- | --- |
| 没有学习信号，所有补偿都不更新 | 已排除这种笼统说法 | 真实梯度、参数更新和Oracle泛化均存在 |
| 所有质量差异都来自选点数量 | 冻结补偿实验已排除 | Gaussian身份与K精确相同 |
| 14维属性语义本身不够 | 不支持作为主要解释 | 同14维Independent Oracle有明显收益 |
| ReLU阻断就是主要剩余原因 | 不足以成立 | bypass取消阻断但质量不改善 |
| 私有特征越大越好 | 未支持 | 8D没有稳定优于4D；有效Jacobian也未随维度增长 |
| 网络更深/更宽/拆属性就会好 | 未支持 | 多个受控候选收益近零或不稳定 |
| ratio输出数值连续就能保证泛化 | 已被实验否定 | E数值连续且finite，未见ratio仍退化 |
| Joint整体梯度普遍反向 | 当前样本不支持 | 整体Full/LoD多数同向，但长期质量存在取舍 |
| 共享C与z/V联合优化发生可辨识性或竞争 | 有迹象，尚未直接隔离 | E依赖V1却弱于其子空间D；Bank也较弱，需冻结/联合对照 |
| 4D的Gaussian身份自由度不足 | 有迹象，未证明是唯一限制 | Oracle联合轨迹不显著低秩，direct14D优于低维；同时存在优化差异 |
| 不同ratio必须有不同个体修正 | 有迹象，尚无直接最简对照 | Oracle方向/幅度差异明显，但解不唯一且依赖独立优化 |
| 冻结共享Encoder缺少补偿所需信息 | 未确定，当前重要盲区 | 多轮只训练头和z/V，没有充分检验单独适配补偿Encoder |
| 固定地图已经没有剩余质量 | 已排除 | Oracle还能恢复约0.6～1.2 dB，按ratio而变 |

**当前主要盲区不是“还没试够大网络”，而是没有把“个体自由度、跨ratio共享约束、共享表示、优化路径”分别拆清。**

## 10. 下一步：按信息价值排序，而非继续堆结构

以下均为**建议，尚未执行**。每项启动前锁定协议；不自动修改生产默认。先完成P1，根据结果决定P2/P3，避免同时展开多个结构。

### P1：先补上最关键的缺失控制——跨ratio共用一份独立14D修正

固定源地图、原Selector/Cached集合、原补偿输出δ0_i(r)、全部相机。统一使用当前四ratio、同一Bank顺序与每档曝光，至少两个优化顺序；先不让原C/Encoder变化。

| 臂 | 输出 | 目的 |
| --- | --- | --- |
| R0 | δ0_i(r)，完全冻结 | 源函数基线 |
| R1 | δ0_i(r)+V z_i，z_i为4D | C冻结下的Constant4控制 |
| R2 | δ0_i(r)+b_i，b_i为14D，所有ratio共用一份 | 放宽个体属性自由度，但不引入ratio专属表 |
| R3 | δ0_i(r)+b_i,r，各训练ratio独立14D | 同协议的独立Oracle参考；不当作未见ratio模型 |

所有新增参数零输出初始化；R1用非零V/零z，避免双零死路径。R2只是一份14维**附加修正**，原δ0仍随ratio变化，不是强迫所有ratio使用完全相同最终补偿。R3每ratio独立参数/Adam时钟，明确记录总更新次数。

应如何判读：

- **R2接近R3、显著优于R1**：同一个Gaussian的共同修正已能恢复大部分收益，优先怀疑4D子空间/双线性优化，而非缺少复杂ratio调制。
- **R2仍明显弱于R3**：个体修正确有较强ratio依赖，继续研究条件化才有更直接依据；不能只从旧Oracle轨迹差异作结论。
- **R1与R2都主要在Bank改善**：优先检查优化/过拟合和评价位姿，不推进生产结构。
- **R1/R2的未见ratio稳定改善**：再评估存储成本及独立场景，不把单地图结果直接上线。

R1与R2还同时改变了秩约束和双线性/直接参数优化条件，因此不能仅凭该差异唯一认定“维度不够”。但这组小矩阵比继续尝试更复杂V(r)能更直接缩小原因范围。

### P2：隔离共享头C与私有残差的共同适配

在同一D结构、相同初始化/选择/优化曝光下，只比较：

1. C冻结，只训练z/V；
2. C与z/V联合训练，复现现有D。

除最终四指标外，记录C输出变化、私有残差变化、两者的方向/幅度抵消，以及Adam实际预条件更新，不再只看原始gradient cosine。

若冻结C明显更好，才有证据研究参数组优化或训练阶段安排；若没有收益，不能将共同适配当作主因。若P1的R1已经覆盖冻结C情况，可直接复用该臂，避免重复训练；联合控制仍须使用相同协议。

### P3：检验冻结Encoder是否限制补偿输入

仅当P1/P2仍无法解释差距时，给补偿复制一份原`27→32` Encoder，初始权重完全相同；Selector保留原Encoder且冻结，Master/Anchor Feature也冻结。比较“只训练原头”与“训练补偿专用Encoder+原头”，其他条件不变。

该项新增约896个Encoder参数，直接测试补偿是否需要调整输入表征，信息价值高于盲目加深输出头。它与此前“补偿detach共享h”不同：以前detach只是切断梯度，h仍由选择/预算路径塑造，并没有允许补偿学习专用表示。

风险：可能只记忆Bank，或提高推理开销。只有独立视图和未见ratio保持收益才继续；不据Bank loss下降认定有效。

### P4：补偿问题明确后，再复查Joint的优化器时钟与Master竞争

需要保存Master/辅助模块完整Adam状态、step和LR时钟，固定拓扑/视图，分别记录Full和LoD在每组参数上的真实更新。先复现现有Full/LoD取舍，再做单一梯度路由对照。

Full更新次数、辅助更新次数应匹配；Master是否接收LoD更新会自然改变其Adam推进次数，不能声称所有时钟已自动相同。如果要控制时钟，必须另列专门实验并明确optimizer语义，不能暗中插入零梯度step冒充无干预控制。

目标是找到Full与LoD的可接受取舍，而不是仅靠保护Full就宣布成功。Full-only辅助冻结只能作为地图控制，不能和受训LoD方法称为同等辅助训练量。

### 各候选方案的收益与风险

| 方案 | 优点 | 风险/代价 | 优先级 |
| --- | --- | --- | --- |
| P1 共同14D修正矩阵 | 最直接区分个体自由度与跨ratio专属修正，仍复用现Renderer | 参数较多，仍属诊断，R1/R2优化条件不同 | 最高 |
| P2 冻结C vs联合C | 结构/存储不变，容易定位共同适配 | 冻结C也可能减少有效学习能力，不预设一定改善 | 紧随P1，可复用臂 |
| P3 补偿专用Encoder | 直接补足目前冻结h的盲区，初始函数不变 | 可能过拟合，增加前向和存储 | 条件执行 |
| 更复杂ratio调制、增加z维度/层数 | 可能增加表示自由度 | 当前失败机制未分清，搜索成本与选择偏差高 | 暂缓 |
| 预算校准/梯度手术 | 可能改变数量约束与共享梯度关系 | 改训练/推理语义，尚无质量依据，不能与补偿改造同时做 | 单列未来课题 |
| 全部切断LoD→Master | 能保护Full | 已测LoD明显损失，改变Joint取舍 | 不直接采用 |

## 11. 后续实验统一验收与停止规则

1. **先过正确性**：step0等价、K和ID一致、冻结状态SHA、finite、唯一训练参数清单，所有分量语义保持一致。
2. **Bank-only预先锁定**：LR、步数、顺序、初始化、checkpoint规则；不根据non-Bank调参。每个候选报告两个seed，保留全部失败结果。
3. **质量四指标同时看**：PSNR为主结果，但必须并列SSIM/LPIPS/photo；每ratio、每seed、逐视图mean/median/p10/p90及改善比例完整输出。
4. **区分数值变化与有效改进**：加做同seed基线重复，收益应超过重复波动并在两个优化顺序同向；不从当前有限重复估计可靠总体方差或宣称统计显著。
5. **明确泛化边界**：旧V0/V1/V2可用于复现对比，新的设计反复依据这些结果选择后，它们就不是全新独立验证。进入生产前应锁定未用于结构选择的场景或独立地图，而非无限追加同轨迹邻近帧。
6. **比真实成本**：同N/K测K行forward/backward、gather、参数/Adam、峰值；生产接入后另测整个Mapping，不能用头部微基准代替端到端结论。
7. **正面结果再进入在线Joint**：一个固定地图的正面结果不足以直接替换生产。候选需在新地图保留收益、未见ratio不系统退化，且Full质量与成本可接受。
8. **失败就定位并停止扩容**：P1/P2无稳定收益时，不自动扩维、加层、加Feature或改Loss；依据对照决定是输入、优化还是跨ratio结构问题。

## 12. 证据索引与本轮复核范围

### 12.1 原始报告

| 主题 | 报告 |
| --- | --- |
| 早期地图/人口背景 | [可变成员Joint质量](H:/AIProjects/cartgs优化更细致版/mycartgs/VARIABLE_ANCHOR_JOINT_QUALITY_REPORT.md)、[Anchor-only pruning](H:/AIProjects/cartgs优化更细致版/mycartgs/ANCHOR_ONLY_PRUNING_ABLATION_REPORT.md) |
| Anchor特征容量与ratio特征 | [Feature32](H:/AIProjects/cartgs优化更细致版/mycartgs/ANCHOR_FEATURE32_10000_REPORT.md)、[Ratio-aware特征](H:/AIProjects/cartgs优化更细致版/mycartgs/RATIO_AWARE_ANCHOR_FEATURE_REPORT.md) |
| 长尾/深度与split | [输入/梯度/容量](H:/AIProjects/cartgs优化更细致版/mycartgs/MLP_GRADIENT_EXTRAPOLATION_CAPACITY_AUDIT.md)、[split修复](H:/AIProjects/cartgs优化更细致版/mycartgs/SPLIT_SCALE_SAMPLING_FIX_REPORT.md) |
| 预算路径 | [共享梯度对照](H:/AIProjects/cartgs优化更细致版/mycartgs/BUDGET_SHARED_GRADIENT_COUPLING_AUDIT.md)、[私有预算Feature](H:/AIProjects/cartgs优化更细致版/mycartgs/PRIVATE_BUDGET_FEATURE_ORACLE_AUDIT.md) |
| Selection/Compensation/Joint Oracle | [渲染质量瓶颈审计](H:/AIProjects/cartgs优化更细致版/mycartgs/RENDER_QUALITY_BOTTLENECK_AUDIT.md) |
| Gaussian-private4D/8D/bypass | [私有补偿审计](H:/AIProjects/cartgs优化更细致版/mycartgs/GAUSSIAN_PRIVATE_COMPENSATION_AUDIT.md) |
| Constant/Ratio-conditioned | [Ratio-conditioned审计](H:/AIProjects/cartgs优化更细致版/mycartgs/RATIO_CONDITIONED_PRIVATE_COMPENSATION_AUDIT.md) |
| 属性拆头/统一加宽 | [分支补偿审计](H:/AIProjects/cartgs优化更细致版/mycartgs/SPLIT_COMPENSATION_HEAD_AUDIT.md) |

### 12.2 关键原始数据

- Oracle与Joint：[质量汇总](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/render_quality_bottleneck_20261022/evidence/results/quality_summary.csv)、[配对统计](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/render_quality_bottleneck_20261022/evidence/results/paired_statistics.csv)、[梯度汇总](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/render_quality_bottleneck_20261022/evidence/results/gradient_summary.csv)。
- 私有补偿：[四指标汇总](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/gaussian_private_compensation_20261022/evidence/results/quality_summary.csv)、[完整逐视图](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/gaussian_private_compensation_20261022/evidence/results/all_quality_per_view.csv)。
- Ratio-conditioned：[逐视图及真实贡献点](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/ratio_conditioned_private_compensation_20261022/evidence/results/quality_per_view_with_counts.csv)、[恢复率](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/ratio_conditioned_private_compensation_20261022/evidence/results/oracle_recovery.csv)、[同上游投影控制](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/ratio_conditioned_private_compensation_20261022/evidence/results/projection_same_upstream_control.csv)、[交付核验](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/ratio_conditioned_private_compensation_20261022/verification.json)。
- 分支头：[逐视图及真实贡献点](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/split_compensation_head_20261022/evidence/results/quality_per_view_with_counts.csv)、[配对统计](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/split_compensation_head_20261022/evidence/results/paired_statistics.csv)、[微基准](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/split_compensation_head_20261022/evidence/results/efficiency.csv)、[交付核验](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/split_compensation_head_20261022/verification.json)。

本次整理独立复算Ratio-conditioned的11,388条和拆头的8,900条逐视图记录，对234/200个摘要组的四指标逐组核对；复算Oracle逐ratio增益、Joint Full/LoD关键差值及同上游投影cosine；重新核对当前12个核心文件SHA、HEAD与tracked工作区。其余历史结果引用相应冻结报告与原有验证归档，没有把未重新跑过的实验写成新执行。

**最终研究决策：生产结构暂不变；优先完成“共同14D修正 vs 独立ratio14D修正”的缺失控制，再隔离C/z/V共同优化，随后才决定是否需要补偿专用Encoder。当前没有足够证据支持继续直接加深MLP。**
