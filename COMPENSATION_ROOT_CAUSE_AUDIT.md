# Compensation质量瓶颈归因研究

日期：2026-10-22。源码基线：`02cd92ff4fcab299008ba094be6966b70e842061`。状态：**隔离诊断完成；生产算法、Loss、默认配置均未修改，未stage/commit/push。**

## 1. 结论与证据边界

Independent14D的优势同时包含**更自由的个体补偿参数化、解除跨ratio绑定，以及更合适的优化条件**。现有证据不支持把约0.8 dB差距笼统归为“MLP不够深”。

在同一fr1_xyz地图、相同Cached选择集合、每ratio 2000次曝光、两个优化顺序seed下：

| 方案 | 顺序seed2026 | 顺序seed2027 | 平均ΔPSNR |
| --- | --- | --- | --- |
| R1_cycle | +0.2019 | +0.2380 | +0.2200 |
| R2 | +0.0797 | +0.4140 | +0.2468 |
| R2_cycle | +0.5618 | +0.5603 | +0.5610 |
| R3_matched | +0.7846 | +0.8139 | +0.7993 |

上述数值均为44个non-Bank视图、四训练ratio等权的PSNR增益，相对冻结R0，不是不同地图绝对PSNR相减。

- 相同4D/14D同帧聚合控制中，直接14D比可学习4D投影高 **0.3411 dB**。这是“低秩因子化参数化整体”的差距，尚不能全部归为纯秩不足。
- 相同SparseAdam、聚合方式、LR、epsilon、每表step和初始化下，仅解除跨ratio的同一Gaussian残差绑定，独立ratio14D再高 **0.2382 dB**。另一独立SLAM地图也得到 **0.2136 dB**。
- 一张共享14D表在上述匹配控制中恢复同期独立ratio14D收益的 **70.2%**；到6000次曝光约 **60.8%**。它可以恢复多数收益，但不能称为已接近完整Oracle。
- 共享头C联合训练短期较差、长程跨seed结论冲突；不能判定存在系统性有害干扰。Encoder是否主限制**未确定**，本轮未加深/加宽网络。

![质量与收敛](tests/anchor_selector/compensation_root_cause_20261022/quality_and_convergence.png)

## 2. 当前生产架构与源checkpoint核对

### 2.1 实际拓扑与输入

生产与源checkpoint均为 **Variable Anchor，1～10个显式Gaussian/Anchor**，不是固定十点；Feature16、原27维输入，无ratio私有Feature表、无全局XYZ输入。

| 源地图run seed | Gaussian | 有效Anchor | 空Anchor | 停用成员 | L0/L1/L2 |
| --- | --- | --- | --- | --- | --- |
| 2026 | 13919 | 5177 | 0 | 244 | 1410 / 3565 / 202 |
| 2027 | 14616 | 5342 | 0 | 275 | 1504 / 3682 / 156 |

对保存的`anchors.rows`重新bincount，与`member_counts`逐元素exact一致，所有Gaussian恰有有效归属，成员数在1～10；没有把预分配空节点当作有效Anchor。主地图各occupancy的Anchor数为：

| 成员数 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Anchor数 | 2692 | 844 | 484 | 271 | 172 | 127 | 88 | 82 | 61 | 356 |

输入与网络：

```text
Anchor Feature16 + local offset3 + activated scale3 + SH DC3
                 + activated opacity1 + log(ratio)1 = 27
Encoder: 27 → 32 ReLU
Selector: 32 → 32 ReLU → 2
Compensation: selected K rows, 32 → 32 ReLU → 14
```

默认TF关闭、SI/GI disabled、ChangeDistill关闭；maturity固定1；Cached-uniform Bernoulli；joint LoD更新Master/Selector/Anchor/Compensation，Full只更新Master。仅Anchor质量pruning，Gaussian clone/split与Anchor growing交替。`max_mapping_iterations=0`保持原样。本轮冻结地图，不执行增删/重排/几何修正。

局部输入`offset=(Gaussian XYZ − Anchor center)/Anchor reference scale`即时从显式XYZ计算，Anchor center/归属是组织状态，没有第二套独立漂移的几何变量。默认LoD不detach Gaussian输入，所以Master既有直接render梯度，也可能通过XYZ/scale/SH DC/opacity输入接收Selector与补偿回传。

生产预算项为`|sum(p)−rN|/max(1,rN)`，默认权重0.01，photo权重1。它直接回传Selection Head、共享Encoder、Anchor Feature，并可沿未detach的Gaussian输入回传Master；**不经过私有Compensation Head参数**。LoD photo同时经过补偿头、选中K行的共享表征和原render属性；Full不调用Selector。默认GI与TF均关闭，因此相应监督/正则不生效。本轮概率与集合固定，只对原photo优化指定补偿参数，不将生产预算梯度混入P1/P2。

源码证据：`improvements/selector/gumbel_network.h`、`anchor_bank.h`、`variable_anchor_blocks.h`、`include/selector_handoff.h`、`src/gaussian_mapper.cpp`、`python/selector.py`。12个生产文件SHA及全部相关默认项见[source_audit.json](tests/anchor_selector/compensation_root_cause_20261022/source_audit.json)；历史四份报告也以SHA存档，未将其推测直接当结论。

服务器既有shadow源码保留split/budget只读诊断hook；其与当前本地源码的比较在排除这些hook、统一换行后`mismatches=[]`，不是声称所有C++文件的原始SHA逐位相同。对应构建验证、源文件真实SHA与本轮源模型SHA分别保留，不把脚本兼容等同于已验证新的在线训练。

### 2.2 split采样

`src/gaussian_model.cpp`的`getScalingRotationActivation`返回`R diag(activated scale)`；legacy及variable成员split均使用：

`x_child = x_parent + R diag(s) ξ`。

仍保留`abs(ξ)>3 → 0`的既有处理，因此不是完全无截断正态。协方差接口继续返回`LLᵀ`，采样没有再误用协方差矩阵。历史独立修复已有native CPU/CUDA证据；本轮只做源码一致性核对，没有借冻结实验声称重新验证在线split质量。

### 2.3 14D属性语义与渲染

```text
xyz       = base_xyz + δ[0:3]
scale     = activated_base_scale * exp(δ[3:6])
rotation  = normalize(multiply(normalize([1,0,0,0] + δ[6:10]), base_rotation))
opacity   = sigmoid(base_opacity_logit + δ[10])
SH_DC     = base_SH_DC + δ[11:14]
```

其余SH不变，使用原Renderer、原photo=`0.8 L1 + 0.2(1-fast_ssim)`。Quaternion四个存储分量只有三个旋转自由度，并有缩放/正负非唯一性；不能把14个原始系数的总欧氏误差解释为物理误差或PSNR。

R2共享的是**变换前加性系数**`b_i`；由于δ0本身依赖ratio、旋转归一化和opacity sigmoid非线性，同一b并不意味着各ratio最终旋转/opacity增量完全相同。

### 2.4 必须保留的生产优化器边界

`GaussianModel::trainingSetup`创建`SparseGaussianAdam`子类，默认eps=`1e-15`；`ensureSharedMappingOptimizer()`添加Selector组时继承该默认，仅设置LR。默认joint路径通过基类零参数`Adam::step()`更新，不是Python `SparseAdam`，也不是子类带visibility的重载。

本轮独立LibTorch测试输出：`shared_selector_eps=1e-15`，standalone Selector构造器为`1e-8`。**standalone构造器不能代表生产joint的有效epsilon。** 离线实验为复现历史协议，保留Python Adam/SparseAdam eps=`1e-8`；因此优化器数值归因不能直接等同于线上生产问题。测试代码、binary及输出均已存证。

### 2.5 历史尝试在本轮如何重新解释

核对历史脚本后，Private4/8将`Uz`注入原ReLU内部，并继续训练C；Bypass避开ReLU，但有效投影仍受原输出权重W2约束。Constant4改为自由`V0z`；Ratio-conditioned4使用`[V0+φ(log r)V1]z`，同时继续训练C。它们改变的不只一个因素，不能把各轮恢复率直接当作“私有维度不足”的证明。

本轮P1固定C，直接比较共享14D、独立ratio14D与自由4D投影，去掉这些混杂；P2再单独放开C。旧线性ratio模型虽然为V增加连续调制，却仍用一份z4耦合所有属性与ratio，并未达到每ratio独立14D自由度。它失败不能反推ratio条件化无用，亦不能证明更复杂调制必然有效。

Split Compensation Head只把XYZ/Scale/Rotation/Other分别解码，仍基于同一个冻结32维Encoder、没有Gaussian身份参数；它没有解除个体共享和跨ratio约束。历史“梯度cosine变好”没有形成稳定质量收益，只能作描述。当前生产仍是统一14D头；本轮没有拿Split实验checkpoint替换源网络，也没有把其新增确认帧混入本轮44帧协议。

## 3. 受控协议与原Oracle复现

主地图来自`baseline_seed2026/6900_shutdown/ply`；N=13919、M=5177，70个Bank训练视图、既有44个non-Bank视图。第二张地图来自独立在线重建run seed2027，N=14616、M=5342，72个Bank、同协议44个非Bank视图，确认Bank/eval无交集。

两个优化顺序seed2026/2027用于每个正式对照；它们不代表两张地图。第二张地图是额外拓扑确认，仍是fr1_xyz，**不是第二场景**。不新增同轨迹确认帧，不将反复使用的44视图视为新盲测。

训练320×240，正式评价640×480。PSNR/SSIM/LPIPS按原协议使用clamp到[0,1]的图像；photo保持原raw render。除指定补偿副本/私有残差外，Master、Anchor Feature、Encoder、Selector、相机与Gaussian身份全冻结，源文件及tensor SHA前后exact。

所有方案直接复用原Cached Bernoulli集合，不改为Top-K，也不重新抽样；各方案在同一ratio下的source row与K完全一致：

| 地图 | 1% | 2.5%未见 | 5% | 7.5%未见 | 10% | 12.5%未见 | 15% |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 主地图2026 | 138 | 343 | 697 | 1052 | 1365 | 1749 | 2021 |
| 第二地图2027 | 142 | 361 | 727 | 1078 | 1424 | 1832 | 2117 |

主地图冻结R0的non-Bank绝对质量，作为所有增量的共同起点：

| ratio | PSNR | SSIM | LPIPS | photo |
| --- | --- | --- | --- | --- |
| 1% | 16.138353 | 0.591735 | 0.691091 | 0.161320 |
| 5% | 19.287746 | 0.655382 | 0.526824 | 0.122267 |
| 10% | 20.050544 | 0.677976 | 0.479734 | 0.112757 |
| 15% | 20.163840 | 0.682933 | 0.465649 | 0.110934 |

主地图训练集合union=2021；私有表仅分配该union并只gather选中K行。未见ratio没有独立训练表；主地图的未见集合均被训练union覆盖。R3没有未见ratio输出，不插值、不为未见ratio另训Oracle冒充零样本结果。

### 3.1 优化预算

固定LR=0.001，Adam/SparseAdam新建零moments，无新增loss/regularizer。每cycle用相同Bank帧，ratio按1/5/10/15顺序，各曝光一次。主对照2000次曝光/ratio；统一继续至4000/6000诊断收敛，不按non-Bank挑checkpoint或停训。所有后续控制先写单独protocol，再执行；P1独立阶段结论保存在[P1_A_FINDINGS.md](tests/anchor_selector/compensation_root_cause_20261022/P1_A_FINDINGS.md)。

同帧聚合控制仅对四个原photo梯度取均值后做一次真实更新，不加新损失、不伪造零梯度step，也未接入生产ratio调度。

### 3.2 原Independent14D复现

旧Oracle的N行forward初始化与本轮K行compact初始化在1%/5%有最大`1.192e-7`浮点差，10%/15%exact。分别保留“旧保存initial exact replay”和“current compact initial”两个absolute控制。

旧initial复现相对旧Oracle，四档non-Bank PSNR均值差为−0.00031/−0.00857 dB；加性R3相对current-compact absolute控制为−0.00970/+0.00713 dB。原约0.8 dB收益可复现，但CUDA backward轨迹不要求逐位相同。小于这些重复波动的差异不做稳定因果判断。

## 4. P1-A：跨ratio绑定与优化条件

模型全部从相同冻结δ0开始：

```text
R0:          δ(r) = δ0(r)
R2:          δ(r) = δ0(r) + b_i                  一张共享14D表
R3:          δ(r) = δ0(r) + b_i,r                四张独立14D表
R2_cycle:    R2结构不变，同帧四photo均值后一次SparseAdam更新
R3_matched:  R3结构不变，使用与R2_cycle相同SparseAdam和同帧聚合
```

初始R2逐ratio SparseAdam与R3独立DenseAdam，时钟不同，不能直接将差距当表达极限。因此补了R2_cycle，再以R3_matched消除Dense/Sparse、梯度聚合点、均值尺度、LR、eps与每表时钟差异。**R2_cycle vs R3_matched唯一结构干预是ratio是否绑定同一Gaussian参数。**

### 4.1 2000次曝光的四指标

| 方案 | Bank ΔPSNR | non-Bank ΔPSNR | ΔSSIM↑ | ΔLPIPS↓ | Δphoto↓ |
| --- | --- | --- | --- | --- | --- |
| R2 | +0.2466 | +0.2468 | +0.00465 | -0.01088 | -0.00425 |
| R2_cycle | +0.5894 | +0.5610 | +0.01070 | -0.02311 | -0.00672 |
| R3 | +0.8618 | +0.8039 | +0.01628 | -0.02808 | -0.00954 |
| R3_matched | +0.8554 | +0.7993 | +0.01614 | -0.02817 | -0.00949 |

### 4.2 6000次曝光与seed敏感性

| 方案 | 顺序seed2026 | 顺序seed2027 | 平均ΔPSNR |
| --- | --- | --- | --- |
| R2 | +0.7266 | +0.0112 | +0.3689 |
| R2_cycle | +0.6564 | +0.6867 | +0.6716 |
| R3 | +1.1130 | +1.1052 | +1.1091 |
| R3_matched | +1.1129 | +1.0957 | +1.1043 |

| 方案 | Bank ΔPSNR | non-Bank ΔPSNR | ΔSSIM↑ | ΔLPIPS↓ | Δphoto↓ |
| --- | --- | --- | --- | --- | --- |
| R2_cycle | +0.6989 | +0.6716 | +0.01658 | -0.03360 | -0.00881 |
| R3_matched | +1.1264 | +1.1043 | +0.02561 | -0.04583 | -0.01335 |

R2逐ratio更新在6000时两个seed相差很大；不能将其均值当稳定容量。R2_cycle没有换表达结构，却显著减少此波动。聚合控制同时改变聚合位置、频率和梯度尺度，尚未单独隔离“Adam标量时钟”。

### 4.3 各ratio与Oracle恢复率


| 曝光/ratio | ratio | R2_cycle ΔPSNR | R3_matched ΔPSNR | 差距 | 恢复率 |
| --- | --- | --- | --- | --- | --- |
| 2000 | 1% | +0.8468 | +1.1833 | +0.3365 | 71.6% |
| 2000 | 5% | +0.5590 | +0.8418 | +0.2829 | 66.4% |
| 2000 | 10% | +0.4272 | +0.5462 | +0.1190 | 78.2% |
| 2000 | 15% | +0.4111 | +0.6257 | +0.2146 | 65.7% |
| 6000 | 1% | +0.9896 | +1.4656 | +0.4760 | 67.5% |
| 6000 | 5% | +0.5537 | +1.1042 | +0.5504 | 50.1% |
| 6000 | 10% | +0.5505 | +0.9002 | +0.3497 | 61.2% |
| 6000 | 15% | +0.5924 | +0.9472 | +0.3549 | 62.5% |



恢复率使用同地图、同曝光和同seed控制的PSNR增益；上述合并率是“双seed平均增益之比”，不等同于每ratio比例的算术均值。不能拿6000次共享表收益去除2000次Oracle收益。

### 4.4 第二张独立地图确认（2000次曝光）

| 方案 | 顺序seed2026 | 顺序seed2027 | 平均ΔPSNR |
| --- | --- | --- | --- |
| R2 | +0.3727 | +0.4736 | +0.4232 |
| R2_cycle | +0.5338 | +0.4699 | +0.5018 |
| R3_matched | +0.7296 | +0.7012 | +0.7154 |

独立ratio相对共享cycle平均多恢复0.2136 dB，两个顺序seed同向。共享cycle恢复约70.1%，与主地图一致。

但cycle相对逐ratio R2的收益并非每张地图、每个seed都显著；第二地图seed2027约−0.0038 dB，在重复噪声范围内。不能声称聚合策略普遍提高线上质量。

### 4.5 Adam更新与每Gaussian曝光

| 方法 | 每ratio曝光 | 共享/每表Adam step | 一身份每cycle参数参与 | 备注 |
| --- | --- | --- | --- | --- |
| R2 | 6000 | 24000 | 被几个训练ratio选中，就参与几次 | SparseAdam表全局step；行只在gather时更新 |
| R3 | 6000 | 每ratio表6000 | 每个所属ratio参数一次 | inactive表grad=None |
| R2_cycle | 6000 | 6000 | 一份共享参数一次 | 四photo真实梯度均值 |
| R3_matched | 6000 | 每ratio表6000 | 每张被用表一次 | 与R2_cycle同SparseAdam/聚合 |
| R1_learned / C_joint | 6000 | 24000 | z按所属ratio参与；V/C每次参与 | dense V/C + sparse z |
| R1_cycle | 6000 | 6000 | 共享z/V每cycle一次 | 与R2_cycle对齐 |

`per_gaussian_update_counts.csv`分开记录每ratio photo曝光、非零gradient、optimizer参与与非零权重移动。R2/R3的非零gradient有perratio计数；R3_matched还保留perratio实际移动。R2/R3部分权重移动只保存identity跨ratio合计，不能冒充perratio精确移动次数；cycle的共同step也不能给四档分别计作四次更新。78个checkpoint中的真实optimizer state已逐项复核。

### 4.6 物理属性与冲突

6000 checkpoint、两个seed、固定8个Bank视图测量跨ratio VJP。XYZ以Anchor边长归一化，Scale用log坐标，Rotation用SO(3)切空间及q正负等价，opacity梯度保留logit坐标以避免饱和逆除，SH DC用系数坐标；不将不同单位拼成一个“总物理范数”。

R2_cycle在1%与15%的全局梯度cosine及共同非零身份中的反向比例：


| 属性 | R2_cycle全局gradient cosine | 局部身份反向gradient比例 | R3跨ratio补偿反向比例 |
| --- | --- | --- | --- |
| xyz | 0.294 | 25.0% | 30.5% |
| scale | 0.551 | 22.9% | 32.0% |
| rotation | 0.433 | 26.1% | 42.3% |
| opacity | 0.625 | 26.2% | 33.8% |
| sh_dc | 0.499 | 18.7% | 40.4% |



最后一列来自原Dense R3的物理补偿增量方向，属于辅助诊断，非R3_matched质量控制。局部冲突存在，而全局cosine仍可为正；两者都不能单独证明一次LoD更新有害。独立ratio表允许不同身份、不同属性采用不同方向，但这些有限步Oracle不构成唯一目标函数。

将R2_cycle的一个属性块替换成R3的对应块，在固定8个Bank+8个non-Bank probe实际render：整体换成R3提高约0.476 dB；只换XYZ/Rotation/Scale分别下降约2.214/1.663/0.923 dB。**不能将它理解为这些属性不重要或R3该属性较差**；它说明各属性/身份强烈共同适配，破坏一个块的配合会损害图像。逐块/按身份出现频率的条件替换结果不具可加性，不能分配成“各属性贡献百分比”。原始物理fields、VJP与全部probe指标均保存。

## 5. P1-B：4D个体子空间与双线性优化

R1_fixed=`δ0+Vz`，V为预先固定的14×4随机QR正交列，init seed6102，只训练一张z4。R1_learned同初始V/z，训练V与z；C均冻结。z=0使step0与R0 exact一致，V初始非零使z第一步即可得到梯度。

使用单位列而非微小随机V，避免把“固定投影步幅太小”误作4D表达失败；只测试一个预注册随机子空间，没有用non-Bank挑V。R1_cycle进一步与R2_cycle对齐四photo聚合和真实更新频率。


| 方案 | Bank ΔPSNR | non-Bank ΔPSNR | ΔSSIM↑ | ΔLPIPS↓ | Δphoto↓ |
| --- | --- | --- | --- | --- | --- |
| R1_fixed | -0.0000 | -0.0463 | -0.00432 | -0.00225 | +0.00090 |
| R1_learned | +0.3155 | +0.2751 | +0.00372 | -0.01013 | -0.00277 |
| R1_cycle | +0.2533 | +0.2200 | +0.00278 | -0.00814 | -0.00220 |
| R2_cycle | +0.5894 | +0.5610 | +0.01070 | -0.02311 | -0.00672 |



6000次曝光：

| 方案 | 顺序seed2026 | 顺序seed2027 | 平均ΔPSNR |
| --- | --- | --- | --- |
| R1_fixed | +0.0192 | -0.2782 | -0.1295 |
| R1_learned | +0.3996 | +0.4100 | +0.4048 |
| R1_cycle | +0.2738 | +0.2611 | +0.2675 |
| R2_cycle | +0.6564 | +0.6867 | +0.6716 |



可学习V明显优于这个随机固定V，但不能推断所有固定4D子空间都差。R1_learned逐ratio在2000时甚至优于不稳定R2逐ratio；只有对齐聚合控制后，才能量化4D因子化与直接14D的差距：2000为0.3411 dB，6000为0.4041 dB。该差同时包含秩约束与双线性优化，尚未严格拆开。

`Vz`存在`V→VA, z→A⁻¹z`的参数化不唯一性。实际R1_learned的V到6000仍为rank4，条件数约9.0～12.3，V范数4.74～4.81、z范数20.4～20.9；没有观察到rank坍缩或数值爆炸。范数增长本身不能证明病态优化；同帧聚合改善14D却降低4D收益，说明优化粒度对不同参数化的作用不同。

### 5.1 真实渲染检验低秩压缩

对已用Bank photo训练的R2_cycle残差做两种预定rank4 SVD（raw系数、按属性block RMS归一化），不重新训练、不用于正式模型初始化，也不读取non-Bank做拟合。真实same-K render：

| 方案 | 顺序seed2026 | 顺序seed2027 | 平均ΔPSNR |
| --- | --- | --- | --- |
| R2_cycle | +0.6564 | +0.6867 | +0.6716 |
| SVD4_raw | -1.2785 | -1.0397 | -1.1591 |
| SVD4_block_rms | -4.3278 | -4.5452 | -4.4365 |

raw rank4保留约76.4%/76.8%的系数能量，却造成明显PSNR损失；block归一化同样不能保证图像。这只说明**这些已训练补偿的数值低秩压缩不保渲染质量**，不是所有4D子空间的最优PSNR界。Quaternion非唯一性也使raw系数SVD具有坐标依赖。纯photo优化的4D仍有正收益，不能被一次SVD失败否定。

raw rank4在6000时相对系数RMSE约0.486/0.481；分属性归一化后的物理误差、真实旋转geodesic与参数误差另列于`lowrank_statistics.csv`，不混合单位。高能量保留率与图像下降同时存在，进一步否定“数值PCA保留大部分能量即可保PSNR”的推断。

本阶段确认：当前4D方案存在可恢复质量损失；**纯秩不足 vs 双线性优化尚未完全隔离**。不据此增加维度/层数。

## 6. P2：共享头C是否与私有补偿互相干扰

三个分支同初始输出、同V/z和每ratio曝光：C_only只训练原C；R1_learned冻结C训练V/z；C_joint同时训练C/V/z。Encoder、Anchor、Master与Selector仍冻结，未改变选点。

2000次曝光：

| 方案 | 顺序seed2026 | 顺序seed2027 | 平均ΔPSNR |
| --- | --- | --- | --- |
| C_only | -0.1065 | +0.1396 | +0.0166 |
| R1_learned | +0.2332 | +0.3169 | +0.2751 |
| C_joint | +0.0601 | +0.2675 | +0.1638 |

6000次曝光：

| 方案 | 顺序seed2026 | 顺序seed2027 | 平均ΔPSNR |
| --- | --- | --- | --- |
| C_only | +0.1755 | -0.1613 | +0.0071 |
| R1_learned | +0.3996 | +0.4100 | +0.4048 |
| C_joint | +0.5372 | +0.1365 | +0.3369 |

| 方案 | Bank ΔPSNR | non-Bank ΔPSNR | ΔSSIM↑ | ΔLPIPS↓ | Δphoto↓ |
| --- | --- | --- | --- | --- | --- |
| C_only | -0.0971 | +0.0071 | -0.00389 | -0.00050 | +0.00035 |
| R1_learned | +0.3915 | +0.4048 | +0.00631 | -0.01391 | -0.00408 |
| C_joint | +0.1950 | +0.3369 | +0.00234 | -0.01272 | -0.00295 |

2000时C_joint相对冻结C两个seed均差；6000时一个seed改善、另一个退化，不能认定“C联合训练必然有害”。C_only也没有稳定恢复Oracle。长程共同适配不能只看Bank某次下降判断成功。

记录了C输出变化、Vz变化、逐参数组跨ratio gradient范数/cosine、Adam一二阶moment与共同输出坐标上的实际更新。不同维度的C/V/z参数向量之间不直接计算cosine。

6000时同一参数组的1%/15%跨ratio梯度统计，固定8个Bank视图、两个seed均值；范数只在同一参数化内部比较：

| 模型 | 参数组 | 1% gradient norm | 15% gradient norm | 跨ratio cosine |
| --- | --- | --- | --- | --- |
| C_only | C | 0.6260 | 2.1144 | 0.7300 |
| R1_learned | V | 0.1126 | 0.1923 | 0.4170 |
| R1_learned | z | 0.0471 | 0.0579 | 0.3647 |
| C_joint | C | 0.6034 | 2.3850 | 0.8054 |
| C_joint | V | 0.0856 | 0.1401 | 0.3491 |
| C_joint | z | 0.0450 | 0.0533 | 0.3653 |

6000的C_joint输出中，XYZ的ΔC与Vz方向相反比例约58.8%，Scale约54.2%；但实际同一次Adam产生的共同输出步长，XYZ平均cosine约+0.055，只有约13.0%的probe为负。最终分解相互抵消不等于更新过程持续相互对抗，更不自动构成质量退化因果。Rotation采用SO(3)增量，只报告方向，**不使用加性抵消比例**。

因此不增加梯度手术、独立loss或复杂阶段；P2证据不足以将C联合优化列为首要已证实瓶颈。

## 7. 收敛、未见ratio及逐视图统计

### 7.1 收敛曲线

下表是全部Bank、四档、双seed的半分辨率photo均值，与正式全分辨率Bank指标分开：


| 方法 | 2000 | 4000 | 5000 | 6000 |
| --- | --- | --- | --- | --- |
| R2 | 0.116983 | 0.114867 | 0.115865 | 0.116255 |
| R2_cycle | 0.113466 | 0.111688 | 0.111155 | 0.110826 |
| R3_matched | 0.110002 | 0.106416 | 0.105905 | 0.105246 |
| R1_fixed | 0.123186 | 0.125043 | 0.125062 | 0.124564 |
| R1_learned | 0.118335 | 0.117362 | 0.117564 | 0.117262 |
| C_only | 0.122425 | 0.121620 | 0.124418 | 0.124314 |
| C_joint | 0.119310 | 0.117445 | 0.120456 | 0.119844 |



R2_cycle趋势更平稳；R3仍持续改善，原“0.8 dB”不是理论上限。R2逐ratio、C_only、C_joint存在后期波动；R1_learned后期改善很小，但不能以有限步曲线证明其表达能力达到最优。统一保留2000/4000/6000，不选最好checkpoint。

### 7.2 未见ratio：零额外训练


| 方法 | 曝光/ratio | 2.5% ΔPSNR | 7.5% ΔPSNR | 12.5% ΔPSNR | 三档平均 |
| --- | --- | --- | --- | --- | --- |
| R2 | 2000 | +0.0728 | -0.1082 | +0.0023 | -0.0110 |
| R2_cycle | 2000 | +0.1460 | +0.3760 | +0.4201 | +0.3140 |
| R1_learned | 2000 | +0.1514 | +0.1128 | +0.1781 | +0.1474 |
| R1_cycle | 2000 | +0.1563 | +0.1225 | +0.1131 | +0.1306 |
| C_only | 2000 | -0.0217 | -0.0478 | -0.1108 | -0.0601 |
| C_joint | 2000 | +0.1810 | +0.0279 | -0.0337 | +0.0584 |
| R2 | 6000 | +0.3994 | +0.0465 | +0.1062 | +0.1840 |
| R2_cycle | 6000 | +0.3643 | +0.3481 | +0.5619 | +0.4248 |
| R1_learned | 6000 | +0.3057 | +0.1903 | +0.2803 | +0.2587 |
| R1_cycle | 6000 | +0.1523 | +0.0941 | +0.2049 | +0.1504 |
| C_only | 6000 | -0.0256 | -0.1508 | -0.1381 | -0.1048 |
| C_joint | 6000 | +0.2251 | +0.1053 | +0.1182 | +0.1495 |



R2_cycle在这三个未见ratio有正收益，说明一张共享表可以承接部分跨ratio信息，但它不证明任意连续ratio质量保证。R3无天然未见输出，表中不包含R3，也没有为这些ratio拟合专属参数。完整四指标/逐视图结果保存在统一CSV中。

### 7.3 逐视图配对

下表合并两个优化顺序seed的44个同名non-Bank视图，共88个配对；它们不是88个独立场景。2000曝光，R3_matched减R2_cycle：


| ratio | mean ΔPSNR | median | p10 | p90 | 改善比例 |
| --- | --- | --- | --- | --- | --- |
| 1% | +0.3365 | +0.3468 | +0.1478 | +0.4995 | 100.0% |
| 5% | +0.2829 | +0.2777 | +0.1215 | +0.4484 | 100.0% |
| 10% | +0.1190 | +0.1367 | -0.0812 | +0.2709 | 78.4% |
| 15% | +0.2146 | +0.2329 | +0.0027 | +0.3911 | 90.9% |



全部PSNR/SSIM/LPIPS/photo的mean、median、p10、p90、改善比例同时保存；未将图像数当独立随机地图数推断统计显著性。只有两个优化顺序seed，重复噪声较小的端点不能独立证明普遍稳定性。

### 7.4 实际与有效渲染数量

selected是固定K；visible是radii>0；contributing使用已有native像素贡献计数器，统计真正通过原渲染路径像素参与判断的Gaussian，而非仅用opacity≥1/255作代理。所有方法逐视图记录selected/visible/contributing及pixel contribution count。固定K不意味着优化后有效参与数不变。

主地图2000曝光、non-Bank均值：


| ratio | K | R0 visible / contributing | R2_cycle visible / contributing | R3_matched visible / contributing |
| --- | --- | --- | --- | --- |
| 1% | 138 | 128.0 / 117.5 | 127.8 / 116.6 | 128.4 / 116.5 |
| 5% | 697 | 556.9 / 509.4 | 555.3 / 505.8 | 556.6 / 508.2 |
| 10% | 1365 | 985.6 / 867.5 | 985.1 / 865.9 | 986.5 / 867.4 |
| 15% | 2021 | 1335.9 / 1100.7 | 1336.3 / 1101.1 | 1337.3 / 1101.0 |



每case抽检一个Bank及一个non-Bank视图，native计数器像素输出与正式Python Renderer一致、radii exact；第二地图有72 Bank，已显式保证non-Bank parity样本不是第71个Bank。没有用有效点数改变预算或筛选结果。

## 8. 优化器数值机制复核

同样LR/eps不等于DenseAdam与SparseAdam相同。对fresh state第一步，已用真实photo梯度与本地CPU公式独立复核：

```text
DenseAdam:  Δ = -lr g / (|g| + eps)
SparseAdam: Δ = -lr g / (|g| + eps / sqrt(1-β2))
β2=0.999 → Sparse第一步的有效epsilon约31.6倍
```

旧Private4/8位于ReLU内部时，在非零梯度坐标中Sparse/Dense实际步幅比中位数约0.456/0.377；direct14约0.984；本轮正交4D约0.999。原Private确有更易受小梯度数值尺度影响的机制，且ReLU零梯度仍须另外区分。

但是R3_matched换成同Sparse协议后，Oracle质量基本保持；Bypass历史上恢复梯度也未自动恢复质量。因此**epsilon或梯度覆盖不是约0.8 dB优势的充分解释**，这些first-step局部数值指标不能代替长期渲染对照。生产joint有效eps又是1e-15，本轮未更改它；不能用离线1e-8观察声称生产必须改epsilon。

## 9. P3 / P4的执行边界

### P3：未执行，Encoder主因未确定

P1已量化个体参数化、ratio绑定与更新粒度的主要效应，P2未给出稳定共享头干扰证据。按用户“阶段完成后停止低价值扩展”的要求，本轮没有复制/训练Compensation-private Encoder，也未增加输入、深度、宽度。不能宣称Encoder没有瓶颈，亦不能根据本轮结果建议直接加深MLP。

### P4：只审计旧Joint，不重跑

读取了旧Full-only/Joint/Frozen-Master六个最终checkpoint的真实optimizer state：Full-only与Frozen-Master的六组Master step均400，Joint均2000；三者Full更新均400，Joint另有1600 LoD更新。初始moment为fresh zero，非原在线完整优化状态；还存在空间LR尺度代理等协议限制。

因此历史约−0.55 dB Full差距不能作为“相同Master优化时钟下LoD梯度独立损害”的公平因果估计。本轮不将它与补偿参数化实验混合，不用负cosine单独判断伤害，不做伪造零梯度step。后续若研究，必须保存完整在线Master/辅助optimizer、step和LR时钟，并明确选择可比较的真实更新协议。

## 10. 因果判断、优先级与最终回答

| 问题 | 结论 | 证据/限制 |
| --- | --- | --- |
| Independent14D优势主要来自哪一项？ | 个体参数化与ratio独立均有作用 | 2000：对齐4D→14D +0.3411；共享14D→独立ratio +0.2382 dB；不能只归因一个因素 |
| 一份共享14D是否足够？ | 能恢复多数，但不足以接近全部 | 两独立地图2000均约70%；主图6000约61%，后者仍有约0.433 dB差距 |
| 4D是秩不足还是双线性优化？ | 尚未完全拆开 | 直接14D优于匹配4D；学V优于一个随机固定V；低秩SVD不保图像；未测最佳学习子空间固定后的重新优化 |
| C与私有残差联合训练是否导致退化？ | 短期负，长程MIXED | 2000两个seed均差，6000一正一负；抵消与cosine不是质量因果 |
| 当前Encoder是否主要限制？ | 未确定 | P3未执行；本轮不支持直接加深网络 |
| 哪个因素恢复PSNR最多？ | 依曝光预算而变 | 2000匹配4D→14D +0.341；6000独立ratio增益 +0.433略大于4D→14D +0.404 |
| 下一项具体策略？ | 先做已学习子空间固定后的z重新优化诊断 | 只改变V是否参与优化，去掉随机固定子空间差的混杂；不直接更换生产网络 |

上述端点差存在交互，不能把几个不同控制的收益当作互不相关的因果百分比相加。直接14D也只在本次有限photo优化中较好，不构成理论最优。

**下一项优先诊断（尚未执行）**：取仅用Bank训练得到的R1_learned最终V并固定，将z重新置零，保持δ0与初始输出一致，使用相同视图/曝光/Adam协议仅优化z；与同初始V、z=0且继续学习V/z的分支比较。这样比继续测试随机V、增大维数或加深MLP更直接地区分“可用4D子空间不足”与“寻找/共同优化该子空间困难”。需要把Bank学习V的额外预训练计算预算单独列出，不能冒充与fresh训练完全等成本；不把此建议自动扩展为生产两阶段算法。

在该区分前，不推荐新的复杂ratio调制、loss或梯度手术。若进入生产决策，必须在不同场景及生产实际optimizer路径确认；本次另一张地图只验证了同一fr1_xyz重建拓扑下P1结论，不满足跨场景最终模型选择要求。

## 11. 可复核交付与完成状态

- **TESTED**：两地图拓扑、源文件/tensor冻结、step0一致、固定Cached集合/K、Bank/eval分离、Oracle复现、78个checkpoint真实时钟、Dense/Sparse公式、生产optimizer options、native贡献计数parity。
- **TESTED**：P1-A/B、P2的两个优化顺序seed、2000/4000/6000统一曲线、主地图逐视图四指标；第二地图P1至2000。
- **NOT EXECUTED**：P3 Encoder训练、P4新Joint公平实验、跨场景确认、生产集成。
- **UNCHANGED**：生产Selector/Compensation/Master/Anchor/Loss/采样器/TF/ratio调度/默认配置；未stage/commit/push。

核心入口与证据：

- [本地独立验证](tests/anchor_selector/compensation_root_cause_20261022/verification.json)、[源码审计](tests/anchor_selector/compensation_root_cause_20261022/source_audit.json)、[拓扑核验](tests/anchor_selector/compensation_root_cause_20261022/topology_verification.json)。
- [机器摘要](tests/anchor_selector/compensation_root_cause_20261022/audit_summary.json)、[报告汇总四指标](tests/anchor_selector/compensation_root_cause_20261022/report_summary.csv)、[报告配对统计](tests/anchor_selector/compensation_root_cause_20261022/report_paired_statistics.csv)。
- [统一逐视图结果](tests/anchor_selector/compensation_root_cause_20261022/evidence/results/all_quality_per_view.csv)、[原始分组均值](tests/anchor_selector/compensation_root_cause_20261022/evidence/results/quality_summary.csv)、[全部配对统计](tests/anchor_selector/compensation_root_cause_20261022/evidence/results/paired_statistics.csv)、[Oracle恢复率](tests/anchor_selector/compensation_root_cause_20261022/evidence/results/oracle_recovery.csv)。
- [Bank收敛曲线](tests/anchor_selector/compensation_root_cause_20261022/evidence/results/bank_convergence.csv)、[逐Gaussian更新记录](tests/anchor_selector/compensation_root_cause_20261022/evidence/results/per_gaussian_update_counts.csv)、[参数化数值状态](tests/anchor_selector/compensation_root_cause_20261022/parameterization_statistics.csv)。
- P1/P2物理场、梯度、条件替换、optimizer状态、全部模型权重、源checkpoint与脚本在同目录`evidence/`；`delivery_manifest.json`逐文件SHA可核验。
- 远程实验根：`/home/prmi/zwj/compensation-root-cause-20261022`；本地隔离脚本与完整证据归档保留，没有覆盖生产目录。

验证规模：262个SHA清单文件，54822条逐视图记录，960个质量分组，8784条原配对统计，1088条恢复率记录。原始证据已在本地独立复算；这些是同源受控诊断证据，不是全新场景的线上质量保证。
