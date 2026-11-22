# Anchor LoD Compensation：近期证据与下一步决策依据

日期：2026-10-26。项目：mycartgs。用途：确定下一轮研究问题、复用正确基线，并避免重复已完成的低价值对照。

本次仅整理本地源码、25份报告和对应指标；没有新训练、服务器作业、生产算法或默认配置修改。源码快照：分支 `anchor-si`，HEAD `27b943956894367632b1a50562fd16e90d51ae8e`；工作区另有17个既存 tracked 改动，**HEAD不能单独代表实际源码**。文件SHA和工作区diff摘要见[证据清单][manifest]。

## 1. 供决策直接使用的结论

1. **保留原11维输入、浅层Encoder和59维四Head，作为下一轮神经网络实验基线。** 原11维是 Offset3、Scale3、SH DC3、Opacity1、log(ratio)1；不要把其中的16维补偿latent误认为Selector的共享Anchor Feature。
2. **四Head是近期较明确的正向方向。** 同源地图、两seed下，相对统一32宽头平均提高0.621 dB；相对近似等参数量的统一80宽头提高0.311 dB。存在同seed重跑波动，所以这是优先复验候选，尚不是跨场景或在线最优结构。
3. **不建议继续泛化地加深Encoder或增加输入。** 深32相对浅32仅+0.002 dB且seed方向相反；专用Encoder的匹配增量仅+0.003 dB且seed方向相反；新增基础Rotation、原始全局XYZ均降低近期两seed平均质量。
4. **E8扩大到E50没有形成接近R3的证据。** 两场景各8seed的均值约+0.159 dB，但置信区间跨0，未见ratio均值略降；比单seed“+0.633 dB”更适合用于总体判断。
5. **接近R3的关键差距仍未在最新四Head上完成归因。** 旧固定集合Oracle确认：个体补偿自由度、跨ratio绑定和优化粒度均有作用；最近多数神经网络实验仍训练Selector，选中身份可变，不能把全部差距只归给Compensator。
6. **下一轮推荐优先做固定集合下的补偿共享约束诊断，而不是继续输入/深度扫参。** 具体建议见第8节；如果目标是尽快落地四Head，则先做独立场景确认，再进入在线Joint。

以上结论均限定于已测表示、初始化、训练预算和数据。没有证据证明所有更大网络、所有Rotation表示或所有坐标编码都无效。

## 2. 必须分清的三个“Baseline”

| 名称 | 当前含义 | 用途 |
| --- | --- | --- |
| 生产默认 | `S00R00E50D12`，统一补偿头 | 当前源码默认运行路径；本次不替换 |
| 近期实验Baseline | 原11维输入 + 浅层Encoder + latent16 + 四个两层Head | 最近Rotation、XYZ和输入消融的参照 |
| R3系统目标 | 原共享Anchor16的Selector + Gaussian/ratio独立14D补偿残差 | 系统质量目标；与近期Baseline的Selector也不同 |

### 2.1 生产默认：以当前配置为准

核对[配置][config]及[C++网络][cpp]：

```text
Variable Anchor，显式Gaussian，每Anchor 1～10成员
Selector输入：Offset3 + activated Scale3 + SH DC3 + sigmoid Opacity1 + log(ratio)1
共享Encoder：11 → 32 → ReLU
选择Head：32 → 32 → ReLU → 2
选后补偿：[h32, 每Anchor/ratio latent50] → 32 → ReLU → 32 → ReLU → 14
```

关键默认：`anchor_feature_dim=0`、`anchor_ratio_features=0`、`anchor_global_xyz=0`、`compensation_method=s00r00e50d12`。`anchor_ratio_features=0`只关闭Selector的ratio特征表，**不关闭Compensation的ratio latent表**。

- Variable Anchor开启；Gaussian clone/split和Anchor growing仍存在。基础尺度0.1，多层组织沿用现有实现。
- 当前 `anchor_only_pruning=0`，是Gaussian/Anchor拓扑轮换路径，**不能沿用旧报告“当前只进行Anchor pruning”的描述**；`anchor_refine_requires_full=0`，slot deactivation开启。本文只记录，不修改。
- TF、SI/GI、ChangeDistill关闭；成熟度固定1。LoD joint更新Master/Selector/辅助补偿路径，Full只更新Master。
- `compensation_start_iter=3000`；非串行joint；`max_mapping_iterations=0`，保留自然停止，不强制10000。
- 本地两个split入口已使用 `R diag(scale)` 采样，而不是协方差矩阵；本次是源码核对，没有重新运行在线split验证。

这些是本地静态事实；本次没有重查服务器默认配置、重建二进制或确认新的在线运行。

### 2.2 最近使用的59维四Head实验Baseline

定义原始输入：

```text
u_i(r) = [Offset3, activated Scale3, SH DC3, sigmoid Opacity1, log(r)] ∈ R11
Offset = (Gaussian XYZ − Anchor center) / Anchor reference scale
h_i(r) = ReLU(W_enc u_i(r) + b_enc) ∈ R32
p_i(r) = softmax(SelectorHead(h_i(r)))_keep
v_i(r) = [u_i(r), e_a(i)(r), h_i(r)] ∈ R59
```

其中 `e_a(r)∈R16` 是**每Anchor、每训练ratio一份，仅供补偿使用**的latent。Selector没有可学习Anchor latent输入。四个训练节点1/5/10/15%分别存表；未见ratio先按raw ratio线性插值latent，再解码，同时输入真实 `log(r)`。

| Head | 结构 | 输出维数 |
| --- | --- | --- |
| XYZ | 59→32→ReLU→32→ReLU→3 | 3 |
| Scale | 59→32→ReLU→32→ReLU→3 | 3 |
| Rotation | 59→32→ReLU→32→ReLU→4 | 4 |
| Other | 59→32→ReLU→32→ReLU→4 | Opacity1 + SH DC3 |

四Head隐藏参数独立；输入latent和h32共享，梯度在共享输入处相加。Other不是额外的全14维残差头。先计算选择，再仅对选中的K行解码补偿。

在N=6838/M=2066地图上，补偿Head共12,366参数，整个实验网络146,096参数，其中四张Anchor/ratio latent表占132,224参数。源码入口：[原Baseline构造][baseline_run]、[四Head构造][split_run]。

14D输出的变换始终一致：

```text
xyz'     = xyz + delta[0:3]
scale'   = activated_scale * exp(delta[3:6])
q_corr   = normalize([1,0,0,0] + delta[6:10])
rotation'= normalize(q_corr ⊗ base_rotation)
opacity' = sigmoid(base_opacity_logit + delta[10])
SH_DC'   = SH_DC + delta[11:14]
```

其他SH系数不变。Quaternion四个系数不是四个独立旋转自由度；不能将原14维系数的欧氏距离直接视为物理误差或图像质量差距。

### 2.3 最新地图上的R3目标与尚未归因的差距

从[矩阵逐seed指标][r3_metrics]提取同一fr1源地图、同12000步、seed3100/3101的R3参照：七档same-K分别为18.01770/17.94501 dB，平均17.98136 dB。最近锁定的四Head Baseline为17.24435 dB，**系统差距约0.73701 dB**。

该差距不等于“最新Compensation Oracle还能恢复0.737 dB”：R3同时保留Selector共享Anchor16输入、使用Gaussian/ratio独立14D，而四Head的Selector没有Anchor latent，最终集合也不同。这正是需要第8节固定选择诊断的原因。

R3的14D直接用于属性补偿，不再由MLP解码为另一个14D；已接入的R3系统对未见ratio插值其14D残差。早期固定集合Oracle则没有定义未见ratio输出，两者的泛化协议不能混同。R3不是理论质量上界。

## 3. 两套源地图和三种实验边界

### 3.1 源地图不能混合

| 实验家族 | Gaussian / Anchor | 训练/评价 | 主要用途 |
| --- | --- | --- | --- |
| 2026-10-22 Oracle/根因研究 | 13919 / 5177 | 70 Bank，44历史non-Bank；部分另有同场景确认视图 | 固定集合的补偿、ratio绑定和优化器诊断 |
| 旧根因研究第二地图 | 14616 / 5342 | 72 Bank，同协议44 non-Bank | 同一fr1场景的另一在线地图确认；不是新场景 |
| 2026-10-26近期结构研究 | 6838 / 2066 | 54 Bank训练，13内部Bank留出，45历史non-Bank | 特征维度、Head、Encoder与输入对照 |
| 矩阵/E50的fr2确认 | 9275 / 2724 | 47 Bank训练，12留出，64 non-Bank | 另一场景的锁定候选确认 |

近期fr1源checkpoint：

```text
/home/prmi/zwj/anchor-shared16-ratio8-20261026/
  r3_independent14_seed2026/6400_shutdown/ply
```

旧Oracle源checkpoint：

```text
/home/prmi/zwj/budget-shared-audit-20261022/
  baseline_seed2026/6900_shutdown/ply
```

同为fr1_xyz、同名ratio、同为Variable Anchor，不代表几何、位姿、K、输入或优化状态相同。旧报告中的“当前默认”只代表该报告当时的版本；[2026-10-22总报告][old_summary]适合查历史，不适合直接恢复今天的默认。

### 3.2 最近多数实验的统一协议

- Master所有属性、Gaussian身份和数量、Anchor归属/中心/尺度、相机、eligible和cached uniform冻结；网络从预定初始化训练。
- **Encoder、Selector、Compensation和当前ratio补偿latent可训练，因此选中身份会变化。** Master冻结也不代表补偿后的渲染几何固定。
- 两seed3100/3101同时影响网络初始化和Bank顺序；它们是同一地图重复，不是两个独立SLAM地图。
- 每模型12000真实LoD步，每ratio3000曝光；同cycle使用同一Bank帧，按1/5/10/15%轮转。训练320×240，评价640×480。
- fresh fused Adam，lr=0.001，eps=1e-15；Loss为 `0.8 L1 + 0.2(1-fast_ssim) + 0.01 relative expected-budget L1`，没有新增Loss。
- 正式结果统一取最终12000步；4k/8k只看收敛，不按non-Bank选checkpoint、LR或模型。
- PSNR/SSIM/LPIPS使用clamp到[0,1]的图像；photo使用原render。54个实际训练视图以 `bank_train` 标记，不能把全部67个Bank都当训练集。
- 历史45个non-Bank已经反复用于结构研究，不是新盲测。独立场景确认仍是上线前必需证据。

训练使用 Cached Bernoulli：`S={i:p_i(r)>u_i}`。正式same-K按**原logit差减logit(cached uniform)**排序，而非纯概率Top-K；应直接读logits，避免饱和概率反推logit。

| ratio | 1% | 2.5% | 5% | 7.5% | 10% | 12.5% | 15% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| same-K | 68 | 170 | 341 | 512 | 683 | 854 | 1025 |
| 类型 | 训练 | 未见 | 训练 | 未见 | 训练 | 未见 | 训练 |

same-K固定预算，不固定身份。原Cached实际K另存；`visible`是radii>0，近期 `effective_proxy`是可见且补偿后opacity≥1/255，**不是逐像素积分的真实贡献计数**。旧原生Oracle报告中的contributing使用另一实际像素累积统计，不可混称。

deep32/seed3100出现2次1%空选择：Head实际11998步、1%表2998步；其余近期输入模型无该例外。不能用伪造零梯度step补齐时钟。

### 3.3 专用Encoder是另一个协议

该对照从已训练12000步的48维输入四Head模型继续训练**额外12000步**，冻结原Selector/Encoder和七档选中身份。A冻结补偿编码，B训练同容量补偿专用Encoder；两者Head/latent均继续训练。

它的主结果是固定源Cached集合；辅助结果才是固定floor-ratio same-K集合。不能拿其17.44 dB直接减从头训练Baseline的17.24 dB，声称获得0.20 dB Encoder收益。

## 4. 最近特征容量与Head实验：能证明什么

下表仅使用各轮自己的匹配参照。不同报告的增益不能相加。

| 研究 | 匹配结果：non-Bank PSNR | 证据范围 | 当前判断 |
| --- | --- | --- | --- |
| Selector维度/Head矩阵 | 锁定候选S00R00E08D12相对S16R00E14D11，fr1 +0.579、fr2 +1.400 dB | 77种实测组合，复验8seed；fr2匹配确认 | 支持这个配置家族，不证明S=0普适最优 [E01] |
| E8→E50 | fr1/fr2均约+0.159 dB；95%CI均跨0 | 每场景8seed；未见ratio均值−0.041/−0.079 | 不支持单纯增维接近R3；生产E50按既定选择保留 [E02] |
| 补偿绕过h32 | E8时Offset直连−0.294、raw11直连−0.147 dB | fr1单seed | 没有证明h32是主要坏表示 [E03] |
| 直接latent8→16 | Offset路径+0.500、raw路径+0.415 dB | fr1单seed | 有容量信号，未验证稳定性 [E03] |
| DirectRaw latent16→32 | +0.041 dB；训练+0.100，未见−0.039 | fr1单seed | 增维收益很小且存在权衡 [E04] |
| raw11+latent16加回h32 | +0.135 dB | fr1单seed；同时恢复补偿→Encoder梯度 | 保留h32有正向信号，不能只归于信息恢复 [E04] |
| 统一补偿头2→3隐藏层 | +0.010 dB；训练−0.141、未见+0.211 | fr1单seed | 无明确总体增益 [E05] |
| 59维输入统一头→四Head | 初次+0.552 dB；两seed复验+0.621 dB | fr1，复验有重复波动 | 近期最值得保留的结构方向 [E06][E07] |
| 统一32→统一80宽头 | 两seed+0.310 dB | 与四Head近似等参数量 | 单纯增容也有效，但低于四Head [E07] |
| 四Head−统一80 | 两seed+0.311 dB | 共享输入相同，参数化/初始化不完全相同 | 支持属性专用隐藏层，但不是纯梯度解耦因果 [E07] |
| 四Head删raw11直连，59→48 | −0.147 dB，两个seed均负 | 参照重跑敏感性可缩小到约−0.013 | 质量优先暂留raw11，尚未证明其不可替代 [E08] |
| 48维四Head加专用Encoder | 主固定Cached集合+0.003 dB，seed +0.072/−0.067 | 同初态额外12000步；辅助same-K +0.017 | 没有稳定专用Encoder收益 [E09] |

E50单seed3100曾相对E8提高0.633 dB，但两场景各8seed总体只约+0.159 dB。**单seed快筛不能覆盖多seed结论。** 矩阵候选也不是全组合穷举：720种理论组合中实际测77种，效应模型配置外推R²仅0.138，不能给未跑配置指定“最优数值”。

## 5. 最近输入与Encoder实验：完整链路

以下均为N=6838/M=2066地图、两seed、七档non-Bank same-K等权。原Baseline固定参照为17.24435 dB。

| 方案 | Encoder输入/结构 | PSNR | 相对原Baseline | 直接能回答的问题 |
| --- | --- | --- | --- | --- |
| 原Baseline | Offset+Scale+DC+Opacity+log(r)，11→32 ReLU | 17.24435 | 0 | 当前研究参照 |
| XYZ-only | 世界XYZ+log(r)，4→32 ReLU | 15.90709 | −1.33726 | 一并移除原属性后退化 [E10] |
| XYZ+Scale+Opacity | 上述加Scale3/Opacity1，8→32 ReLU | 16.46470 | −0.77965 | 比XYZ-only +0.55761；不能拆分Scale/Opacity贡献 [E11] |
| 再加Offset，浅层增量对照 | 世界XYZ+Scale+Opacity+Offset+log(r)，11→32 ReLU | 16.94428 | −0.30007 | 比8维参照+0.47958 [E12] |
| 改加DC，浅层等容量对照 | 世界XYZ+Scale+Opacity+DC+log(r)，11→32 ReLU | 15.86399 | −1.38036 | 比8维参照−0.60071；Offset优于该DC方案 [E12] |
| 加Offset同时改16瓶颈 | 11→16 ReLU→32 ReLU→32 ReLU | 16.27296 | −0.97139 | 同时改输入和结构，不能单独归因Offset [E13] |
| 重建浅32Encoder | 固定世界XYZ+Scale+Opacity+Offset+log(r)，11→32 ReLU | 16.75815 | −0.48619 | 比16瓶颈+0.48520 [E14] |
| 深32Encoder | 同输入，11→32 ReLU→32 ReLU→32 ReLU | 16.75976 | −0.48459 | 比浅层+0.00160，seed方向相反 [E14] |
| 残差32Encoder | 同输入，32维残差块 | 16.56445 | −0.67989 | 比浅层−0.19370，两seed均负 [E14] |
| 上述浅32加DC | 世界XYZ+Scale+Opacity+Offset+DC+log(r)，14→32 ReLU | 16.81184 | −0.43251 | 比无DC +0.05368，seed +0.206/−0.098 [E15] |
| 原Baseline直接加世界XYZ | 保留原11维和初始化，追加XYZ3，14→32 ReLU | 16.86764 | −0.37671 | 排除旧属性初始化差异后仍退化 [E16] |
| 原Baseline同时加Quaternion | 原11维+规范化/统一符号Quaternion4，15→32 ReLU | 16.99088 | −0.25347 | 两seed平均均负，1%平均−1.297 [E17] |

### 5.1 这些数字不能被解释成什么

- “浅DC”与原Baseline不仅差世界XYZ：浅DC沿逐步加回链路继承权重，Scale/Opacity/Offset/DC新增连接从零开始；原Baseline原属性列有随机初始化连接。
- 最后一组“原Baseline直接加XYZ”专门修正该混淆：原属性、bias、latent/h32和后续层原样复制，仅XYZ列为零；两seed step0隐藏编码和非零输出探针误差均为0，真实photo梯度有效。
- 因此目前可以说“原始世界XYZ接入当前Encoder及Head，没有在该预算下改善质量”，不能说“世界坐标在所有表示中都有害”。尚未测试坐标归一化，也未分开Encoder与Head接入效果。
- 两个语义相近的“XYZ+Scale+Opacity+Offset、浅32”记录分别为16.94428和16.75815，它们来自不同实验构建/重复，不是同一个checkpoint。本文保留两条，不挑高值、不将其差值记为结构增益；要作进一步归因，必须再核对初始化、帧序和数值重现性。
- 深32改善16瓶颈，不等于超过原11维Baseline。不同11维输入的内容也不同，不能仅看维数相同就认定信息相同。
- 删除网络输入不等于删除Renderer中的真实属性。Renderer仍使用完整Gaussian Scale、Rotation、Opacity和SH。
- Rotation输入已经做归一化和q/−q符号统一；不能把退化简单归结为“没归一化”。这仍未检验其他连续旋转表示，也没有证明需要新增Rotation输入。

### 5.2 最新XYZ严格对照的关键数值

| 指标 | 原Baseline | 加XYZ |
| --- | --- | --- |
| PSNR↑ | 17.24435 | 16.86764 |
| SSIM↑ | 0.60185 | 0.59068 |
| LPIPS↓ | 0.60022 | 0.60538 |
| photo↓ | 0.15107 | 0.15697 |

两seed ΔPSNR为−0.26614/−0.48728；训练ratio平均−0.42850、未见ratio−0.30766；Bank训练视图也下降0.3537 dB。原Cached实际K评价下降0.2959 dB。退化不是只出现在non-Bank或插值档。

| ratio | 1% | 2.5% | 5% | 7.5% | 10% | 12.5% | 15% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 加XYZ ΔPSNR | −0.4373 | −0.1095 | −0.4358 | −0.1989 | −0.3494 | −0.6147 | −0.4915 |

## 6. 旧Oracle与根因研究：仍然有用，但不能搬用绝对分数

### 6.1 旧固定集合实验已确认的内容

旧13,919点地图，Master/Anchor/Encoder/Selector/选中身份全冻结，采用原photo：

| 证据 | 匹配结果 | 可以得出的结论 |
| --- | --- | --- |
| Selection Oracle有限swap | 四档平均约+0.023 dB，1%约+0.067 | 已搜索范围收益小；不能证明Selector最优 [E18] |
| Independent14D | 相对冻结原MLP约+0.806 dB，超过等曝光单ratio头续训约0.609 | 原14D属性空间内有可恢复质量；不是理论上界 [E18] |
| Gaussian-private4/8 | 相对联合Shared训练档约+0.119/+0.117 dB，恢复约15% | 低维私有表达有小收益，未接近Oracle [E19] |
| ReLU bypass | 梯度覆盖改善，质量未突破 | 死区真实存在，但不是主要差距的充分解释 [E19] |
| Constant4自由投影 | 确认视图训练/未见约+0.119/+0.094 dB | 维持小收益，没有突破 [E20] |
| 线性ratio-conditioned4 | 对应约+0.021/−0.116 dB | 已测线性调制无上线依据；不等于ratio条件化无价值 [E20] |

### 6.2 匹配优化粒度后，个体自由度与ratio绑定都有效

以冻结共享输出 `delta0_i(r)` 为起点：

```text
R1：delta = delta0 + V z_i，Gaussian共享一份z4
R2：delta = delta0 + b_i，Gaussian共享一份b14
R3：delta = delta0 + b_i,r，Gaussian每ratio独立b14
```

根因研究额外对齐了四photo聚合、SparseAdam、LR/eps和表step；这些是**离线诊断控制，不是生产调度变更**。

| 曝光/ratio | R1_cycle增益 | R2_cycle增益 | R3_matched增益 | R2恢复R3比例 |
| --- | --- | --- | --- | --- |
| 2000 | +0.2200 | +0.5610 | +0.7993 | 约70.2% |
| 6000 | +0.2675 | +0.6716 | +1.1043 | 约60.8% |

2000曝光，14D直接残差比匹配4D因子化多0.3411 dB；解除ratio绑定再多0.2382 dB。6000曝光两个差距约为0.4041和0.4327 dB。第二张同场景地图也确认独立ratio比共享14D多约0.2136 dB。[E21]

这些结果支持“个体参数化、ratio自由度和更新方式都值得研究”，尚未将纯秩不足与双线性优化完全拆开。学V优于一个随机固定V，不证明所有固定4D子空间都差；PCA保留能量也不保证真实render质量。

**优化器警戒线：**旧隔离P1沿用Adam/SparseAdam eps=1e-8；当前冻结结构实验fused Adam eps=1e-15。旧生产joint当时的有效eps也是1e-15。旧数值优化归因不能不加控制地移植到今天的新Baseline。

### 6.3 共享头、Encoder和Joint还不能简单定罪

- 旧P2中C与私有残差联合训练短期较差，长程跨seed方向混合；输出抵消和负cosine不构成系统性损害的证明。[E21]
- 新专用Encoder对照的真正增量只有+0.003 dB；两组继续训练均在训练ratio提高约0.65 dB，但未见ratio平均下降约0.08～0.09 dB。需要区分训练充分性与泛化，不能把续训收益归给Encoder。[E09]
- 旧Joint离线对照中Full比Full-only低约0.550 dB，而LoD比Frozen-Master高约0.339 dB；Master Adam推进次数为2000 vs 400，且没有恢复原在线moments。它提示Full/LoD取舍，尚未隔离公平优化器时钟下的在线因果。[E18]
- 早期“属性拆头无收益”使用13,919点地图、冻结27→32编码和另一种Head；近期59维四Head有收益。两者不是同一个实验，也不构成直接矛盾。

## 7. 证据强度与当前剩余问题

### 7.1 相对可靠的工程与实验事实

- 可恢复补偿质量真实存在：旧独立14D复现通过，且长程仍能提高。
- 当前四Head比统一32和近似等参数统一80有一致正向信号。
- 原始XYZ和当前Quaternion同时接入Encoder/Head，在近期两seed下无收益；新增连接正常训练。
- 最近加深Encoder、同容量专用Encoder均没有稳定收益。
- 不同ratio的私有自由度确实有价值，但单Gaussian共享14D也能恢复多数旧Oracle收益。

### 7.2 仍未确定，不能用猜测填补

| 问题 | 缺少的证据 |
| --- | --- |
| 最新四Head离R3主要差在哪里？ | 同一最新源模型、固定身份下的独立14D复现及匹配共享控制 |
| Anchor共享latent是否限制成员差异？ | 固定选择下Anchor共享与Gaussian私有补偿的匹配对照；raw/Offset本来就允许MLP区分成员，不能仅凭共享表判断 |
| 输入有用信息被h32丢失？ | 原始直连/专用编码结果没有给出稳定主因证据 |
| 四Head优势来自属性解耦还是优化几何？ | 虽有统一80控制，初始化嵌入和选择轨迹仍不同 |
| XYZ退化发生在选择还是补偿？ | 目前两条路径同时接入，未固定集合分别测量 |
| 更长训练是否接近R3？ | 新Baseline无匹配长程固定集合收敛对照；不能拿专用Encoder续训或旧Oracle时长替代 |
| 插值latent是否损害低ratio？ | 网络/集合/ratio变化同时存在；三个未见节点不足以保证任意ratio |
| 生产Joint是否损害Full？ | 缺完整moments、相同Full预算、可比较真实Adam时钟的控制 |

### 7.3 重复性必须进入下一轮预算

四Head同seed3100、相同初始权重和帧序的重复曾出现约0.268 dB七档平均差异；早期第3步photo仅约1.49e-8差异，长程选择/训练轨迹随后分化。没有单独定位全部CUDA非确定性来源，不能称完全确定性。

把旧重复也纳入敏感性分析，四Head相对统一80的平均优势约0.177～0.311 dB；这是敏感性范围，不是置信区间。Rotation最新参照的重跑波动约+0.011/+0.041 dB。**不存在适用于所有实验的统一“0.268 dB噪声阈值”。** 应对当前对照测量配对波动。

## 8. 推荐的下一步：有顺序、有停止条件

以下全部为**尚未执行的计划建议**。不据此自动启动、改网络、改Loss或改默认。

### P0：锁定来源与重复噪声，作为所有新实验的前置门槛

固定原11维、59维四Head、latent16；选用第3节同一fr1冻结地图和同一源训练模型。封存checkpoint、完整Adam状态、ratio表、cached uniform、相机、eligible和Gaussian身份。基线需标识具体run/seed/重复编号，不能只写“Baseline”。

建议正式比较至少4个配对优化seed，另有一个相同seed重复检查；如先按成本做两seed快筛，结果只能标探索性，不能据其微小均值上线。正式预算预先固定，所有臂使用同一每ratio曝光，不按non-Bank延长个别臂。

**通过标准：**源SHA/K/帧序/初始输出一致；数值有限；预定可训练项获得有效梯度；Master/选点冻结项无变化；实际optimizer clock与空集合/稀疏访问记录一致。任一关键检查失败，停止质量归因。

### P1：先验证最新Baseline的“补偿自由度”，推荐优先于继续扫MLP

从同一已训练59维四Head模型取得 `delta0_i(r)`，固定Master、原Head/Encoder/Selector、Anchor表、相机及七档选中ID；原Cached集合为主，固定floor same-K集合为辅助。只用原photo。

| 诊断臂 | 输出 | 目的 |
| --- | --- | --- |
| A0 | `delta0_i(r)` | 冻结基线 |
| A_anchor | `delta0_i(r) + b_a(i),r`，Anchor/ratio直接14D | 测同Anchor成员被约束为相同附加残差的代价 |
| A_gaussian | `delta0_i(r) + b_i,r`，Gaussian/ratio直接14D | 在最新模型上重建独立14D可恢复空间 |

两类新增残差从零开始，step0与A0一致。使用相同Bank帧、每ratio曝光、LR/eps、optimizer类型和每ratio表step；只gather选中行，记录每个Gaussian/Anchor参与次数、非零梯度、实际参数移动与成员数。不能用伪造step匹配。

先完成训练ratio比较，再将两类残差都按相同raw-ratio线性插值评未见节点；明确这是训练节点间的残差插值诊断，不是“未见ratio独立优化Oracle”。不要为未见ratio另训再混称零样本。

**判读与停止：**

- 若A_gaussian在新Baseline上也没有明显可恢复收益，先检查地图、初值、收敛、K与旧Oracle差异，暂停进一步feature设计。
- 若A_gaussian明显改善、A_anchor落后，优先研究成员级补偿差异；按occupancy=1/2～3/4～10分组及1%/5%检验。**这个差距只证明Anchor常量残差约束有代价，不严格证明共享latent+非线性MLP一定不足。**
- 若A_anchor接近A_gaussian，Anchor共享组织未被该诊断否定，下一步更应查共享Head的解码/优化，而不是立刻增加每Gaussian表。
- 每臂需同预算保留收敛曲线；尚在显著下降时，不把固定步数差距称为表达上限。属性条件替换可能破坏共同适配，不能把XYZ/Rotation分量收益直接相加。

### P2：根据P1结果只选择一个后续机制对照

| P1结果 | 推荐继续的一个问题 | 暂不做的扩展 |
| --- | --- | --- |
| 个体差异明显 | 固定选择下，低维Gaussian私有表达能否恢复最新A_gaussian收益；若研究4D秩/优化，优先复用Bank学习投影再冻结、重置z的控制 | 直接扩大多个维度、增加层数、上复杂ratio网络 |
| Anchor常量残差已接近 | 相同59维输入下冻结选择，Head/latent续训是否已充分；固定预算对照训练Head或latent的贡献 | 根据旧non-Bank挑最优训练步数 |
| 主要差在未见ratio | 固定选择下区分训练节点质量与插值节点质量，密集ratio只检查稳定性；共享14D与独立ratio14D作机制参照 | 将数值平滑当作PSNR保证、用未见ratio调参 |

若采用Bank学习投影，额外预训练预算必须单独报告。旧“固定随机V失败”不等于“所有固定子空间失败”。本阶段仍不改变生产调度或Loss。

### P3：四Head的独立场景确认，随后才考虑在线Joint

若目标是快速决定能否部署，而不是解释R3差距，可将本阶段放在P1之前：在另一独立地图/场景只比较已锁定的59维四Head、统一80控制和生产E50，保持原训练预算；不在该场景重新选择输入、深度或latent维度。

建议沿用已有矩阵的0.1 dB实用阈值，但在新协议中预先声明；报告配对seed与连续视图块的区间、每ratio指标、1%/5%结果及未见ratio。区间跨0不是等价；大量视图不能替代独立seed/场景。多臂主比较需预先登记并校正，不以试过的全部结果挑赢家。

只有独立场景也保持实用收益、未见ratio无不可接受退化，才进入真实在线Joint。在线阶段同时报告Full质量、实际点数、拓扑、Mapping/Full/LoD/补偿更新次数；不能将自然停止差异算作纯网络收益。独立保存Full与Selector100%结果。

### P4：Joint竞争单独研究，后置

需要保存并恢复完整Master/辅助optimizer moments、step和LR时钟，在相同拓扑及Full更新预算下比较LoD→Master的独立影响。不能伪造零梯度step补时钟，不能仅凭负cosine关闭joint，也不与上述特征实验同时改变。

## 9. 下一轮必须带走的信息与交付格式

| 项目 | 必须记录 |
| --- | --- |
| 版本 | HEAD + tracked diff SHA + 关键源码SHA；本地/服务器分开 |
| 地图 | 源checkpoint、N/M、Variable Anchor、Gaussian UID/row映射、Anchor层级/中心/尺度、eligible |
| 网络 | S/R/E含义、原始输入字段顺序、激活/单位、Encoder和Head结构；初始化列映射 |
| 训练 | 每ratio曝光、实际每参数/每表Adam step、Bank帧序、空选择、稀疏行访问/移动次数、optimizer状态 |
| 选择 | cached uniform、每ratio固定IDs和SHA、实际K、same-K排序语义 |
| 评价 | Bank训练/留出/non-Bank分开，逐视图四指标、selected/visible/effective_proxy；若用真实贡献计数需注明来源 |
| 泛化 | 训练1/5/10/15%，未见2.5/7.5/12.5%；特征插值还是输出插值、区间外策略 |
| 判断 | 预定最终步、收敛曲线、配对mean/median/p10/p90和改善比例；不按non-Bank选模型 |
| 结论边界 | 冻结地图或在线、仅same-K或固定身份、是否复用模型、同地图seed或独立地图 |

近期代码/证据入口：

- [原Baseline及Rotation对照][baseline_run]：定义近期原始11维和59维四Head。
- [四Head/统一80实现][split_run]：属性分组、独立隐藏层和容量控制。
- [固定地图runtime][runtime]：Scene、Cached训练、same-K评价和真实Adam时钟。
- [最新XYZ单变量原型][xyz_run]、[核验记录][xyz_proof]：严格复制原初始化的可复用范例。
- [最新源地图manifest][scene]：checkpoint/相机/地图SHA、N/M、K及视图列表。
- 各实验目录的 `evidence/analysis/quality_per_view.csv`、`paired_statistics.csv`、`fairness_checks.csv`、`selected_set_overlap.csv`及最终checkpoint保留原样。

服务器既有路径只供定位，本次未重新连接核验：项目 `/home/prmi/zwj/mycartgs`，近期实验 `/home/prmi/zwj/s00-...-20261026`；已使用的Python为 `/home/prmi/anaconda3/envs/zwj/bin/python`。不要用系统默认Python代替。认证信息不放入文档、脚本或实验归档。

## 10. 当前状态

- **已完成并有本地证据：**矩阵/E50、直接输入、Head容量/拆分、48维消融、专用Encoder、Rotation、XYZ逐步加回、Encoder深度、浅DC、Baseline严格加XYZ；旧Oracle和根因研究。
- **局部验证：**四Head在fr1固定地图上的正向结果；未见三个ratio的历史评价；旧Oracle上的个体/ratio约束。均不等于全新独立场景与在线质量保证。
- **尚未执行：**最新59维四Head固定集合下的Anchor/ratio vs Gaussian/ratio 14D诊断、四Head独立场景确认、公平optimizer状态的新版Joint归因、生产四Head集成。
- **本轮新增：**本中文决策文档及46个来源文件的SHA/指标核对清单。没有修改生产配置、启动训练、stage、commit或push。

[manifest]: H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/compensation_research_brief_20261026/evidence_manifest.json
[config]: H:/AIProjects/cartgs优化更细致版/mycartgs/cfg/selector/selector_config.yaml
[cpp]: H:/AIProjects/cartgs优化更细致版/mycartgs/improvements/selector/gumbel_network.h
[baseline_run]: H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/s00_rotation_encoder_20261026/run.py
[split_run]: H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/s00_split_wide_rotation_20261026/run.py
[runtime]: H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/s00_baseline_globalxyz_20261026/matrix_runtime.py
[xyz_run]: H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/s00_baseline_globalxyz_20261026/run.py
[xyz_proof]: H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/s00_baseline_globalxyz_20261026/delivery_verification.json
[scene]: H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/s00_baseline_globalxyz_20261026/evidence/results/fr1_xyz/scene_manifest.json
[r3_metrics]: H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/selector_capacity_matrix_20261026/evidence/analysis/fr1_xyz_confirm_test_seed_metrics.csv
[old_summary]: H:/AIProjects/cartgs优化更细致版/mycartgs/RECENT_LOD_EXPERIMENTS_SYNTHESIS_REPORT.md
[E01]: H:/AIProjects/cartgs优化更细致版/mycartgs/SELECTOR_CAPACITY_MATRIX_AUDIT.md
[E02]: H:/AIProjects/cartgs优化更细致版/mycartgs/S00_E50_COMPENSATION_AUDIT.md
[E03]: H:/AIProjects/cartgs优化更细致版/mycartgs/S00_DIRECT_COMPENSATION_AUDIT.md
[E04]: H:/AIProjects/cartgs优化更细致版/mycartgs/S00_DIRECT_COMPENSATION_EXTRA_AUDIT.md
[E05]: H:/AIProjects/cartgs优化更细致版/mycartgs/S00_RAW_SHARED_DEPTH_AUDIT.md
[E06]: H:/AIProjects/cartgs优化更细致版/mycartgs/S00_RAW_SHARED_SPLIT_AUDIT.md
[E07]: H:/AIProjects/cartgs优化更细致版/mycartgs/SPLIT_WIDE_ROTATION_COMPENSATION_AUDIT.md
[E08]: H:/AIProjects/cartgs优化更细致版/mycartgs/SPLIT_HEAD_NO_RAW_COMPENSATION_AUDIT.md
[E09]: H:/AIProjects/cartgs优化更细致版/mycartgs/COMPENSATION_PRIVATE_ENCODER_AUDIT.md
[E10]: H:/AIProjects/cartgs优化更细致版/mycartgs/XYZ_ONLY_COMPENSATION_AUDIT.md
[E11]: H:/AIProjects/cartgs优化更细致版/mycartgs/XYZ_SCALE_OPACITY_COMPENSATION_AUDIT.md
[E12]: H:/AIProjects/cartgs优化更细致版/mycartgs/OFFSET_SHDC_INPUT_AUDIT.md
[E13]: H:/AIProjects/cartgs优化更细致版/mycartgs/OFFSET_ENCODER16_RELU_COMPENSATION_AUDIT.md
[E14]: H:/AIProjects/cartgs优化更细致版/mycartgs/ENCODER_OPTIMIZATION_AUDIT.md
[E15]: H:/AIProjects/cartgs优化更细致版/mycartgs/SHALLOW_DC_COMPENSATION_AUDIT.md
[E16]: H:/AIProjects/cartgs优化更细致版/mycartgs/BASELINE_GLOBAL_XYZ_AUDIT.md
[E17]: H:/AIProjects/cartgs优化更细致版/mycartgs/ROTATION_ENCODER_COMPENSATION_AUDIT.md
[E18]: H:/AIProjects/cartgs优化更细致版/mycartgs/RENDER_QUALITY_BOTTLENECK_AUDIT.md
[E19]: H:/AIProjects/cartgs优化更细致版/mycartgs/GAUSSIAN_PRIVATE_COMPENSATION_AUDIT.md
[E20]: H:/AIProjects/cartgs优化更细致版/mycartgs/RATIO_CONDITIONED_PRIVATE_COMPENSATION_AUDIT.md
[E21]: H:/AIProjects/cartgs优化更细致版/mycartgs/COMPENSATION_ROOT_CAUSE_AUDIT.md
