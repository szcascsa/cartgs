# Gaussian-private Compensation审计

**结论：HOLD。原注入式4D/8D带来稳定但较小的训练档non-Bank收益，未接近Independent14D Oracle，也没有在全部未见ratio上取得稳定收益。保持生产结构，不进入在线Joint测试。**

4D/8D相对同样8000步联合训练的共享头，训练档non-Bank PSNR分别提高平均 **0.119 / 0.117 dB**，恢复当前基线到Oracle差距的平均 **15.1% / 14.2%**。4D整体效率更合适，8D只在1%明显更好。此次结果不支持单纯加大私有维度或加深网络。

新发现是补偿头ReLU会切断大量Gaussian的私有梯度，并把可用个体方向压到低秩；但额外的等参数ReLU-bypass虽恢复这些梯度，质量仍未改善。这表明**梯度死区真实存在，却不是剩余质量差距的充分解释**。下一步更值得研究私有补偿的连续ratio条件表达与参数共享，同时检查补偿优化条件。

## 1. 源码、版本与冻结边界

HEAD `02cd92ff4fcab299008ba094be6966b70e842061`，分支`redundancy-aware-si`；12个核心文件SHA、HEAD、tracked状态与上一轮一致。源码审查入口为 `improvements/selector/gumbel_network.h:87`、`python/selector.py:151`、`src/gaussian_mapper.cpp:3315`、`improvements/selector/variable_anchor_blocks.h`。

源地图与 `RENDER_QUALITY_BOTTLENECK_AUDIT.md`完全相同：fr1_xyz、13919 Gaussian / 5177 Variable Anchor、每Anchor1..10成员；27维输入、Feature16、共享Encoder27→32，选择头32→32→2，补偿头32→32→14。ratio_features/global_xyz/TF/SI/GI/ChangeDistill关闭。原clone/split与Anchor交替拓扑仍保留在生产源码，本轮冻结地图中不执行。

本轮只复制并训练补偿头及其私有分支。Master六组属性、Anchor Feature/归属、Encoder、Selector、相机、cached uniform和各ratio集合均固定；训练前后逐Tensor SHA完全一致。生产网络、Loss、采样器、Joint、默认配置均未改，没有stage、commit或push。

| ratio | 类型 | 固定Cached实际K |
|---|---|---|
| 1% | 训练 | 138 |
| 2.5% | 零样本未见 | 343 |
| 5% | 训练 | 697 |
| 7.5% | 零样本未见 | 1052 |
| 10% | 训练 | 1365 |
| 12.5% | 零样本未见 | 1749 |
| 15% | 训练 | 2021 |

所有方法使用相同集合，不按新模型重选点。70 Bank训练，44个固定且不与Bank重叠的non-Bank评价；训练320×240，正式评价640×480。选点语义仍为原Cached Bernoulli，等同于以该实际K截取原logit-margin排序。本轮没有改变K或网络输入ratio。

## 2. 原型结构、初始化和对照

原补偿头 `a=W1 h(r)+b1`，`delta=W2 ReLU(a)+b2`。私有特征 `z_i`绑定冻结Gaussian source row，与Anchor无关；一张`[13919,d]`表，不存按ratio分离的表。

主设计：`delta_i(r)=W2 ReLU(a_i(r)+U z_i)+b2`，d=4或8。原权重直接复制，z精确为零，U按固定seed6102以std=0.01非零初始化，独立matmul相加。原头和U使用Adam，z使用SparseAdam；lr统一0.001，无weight decay，无新增loss。

step0在全部7个正式ratio输出与原头逐项精确相同。首个真实photo backward中，138个索引行有127行私有梯度非零且finite；U的首步梯度为零符合z=0的链式法则，z可立即更新。仅对K个选中行Embedding gather，sparse梯度/Adam更新不遍历全人口；全N参数/Adam状态的零初始化只发生一次。

| 方法 | 训练对象/限制 |
|---|---|
| A Shared | 原32→32→14头；同一个头联合四ratio |
| B Private4 | 一张Gaussian-private4D表＋原共享头＋32×4投影 |
| C Private8 | 一张Gaussian-private8D表＋原共享头＋32×8投影 |
| D Oracle14参考 | 上一轮每ratio独立[K,14]；不具备未见ratio零样本输出 |
| E Shared-capacity控制 | 原头＋32→1187→14共享残差适配器；额外55803参数，与4D额外55804只差1个；非候选生产设计 |
| SECONDARY 4D bypass | 仅将ReLU(a+Uz)换为ReLU(a)+Uz；其他参数、初始化、训练协议与4D相同 |

E的残差最后层零初始化，保证相同step0；只测试这一预定匹配参数量控制，未进行容量或LR搜索。SECONDARY由Bank-only梯度诊断中的死区导出，单独登记于`bypass_protocol.json`，保留所有主实验结果；不修改学习率、步数或按non-Bank挑seed。

所有补偿仍是XYZ3、log-scale3、quaternion4、opacity-logit1、SH DC3；严格复用原加法、exp、identity+quaternion normalize/compose、sigmoid和SH DC变换。训练仅photo=`0.8 L1+0.2(1-fast_ssim)`，无额外正则、蒸馏或预算梯度。

## 3. 训练曝光与Oracle可比性

两个优化顺序seed2026/2027，初始化投影seed固定；均来自同一个SLAM seed2026地图，不是两个独立场景。每seed2000个打乱Bank相机循环，每循环同一视图固定按1/5/10/15%更新，**每个联合模型共8000步，每档2000次曝光**；评价最终步数，不早停或选最优checkpoint。

Oracle沿用上一轮每档2000步。源文件SHA、K、选中ID、初始残差（跨GEMM batch数值误差≤1e-6）、photo、Bank序列逐项核对；每ratio的视图子序列与新联合模型精确相同。不同参数化的Adam时钟不能完全相同：共享头8000步，旧Oracle各自2000步；SparseAdam全局clock8000，具体行访问次数另记。

上一轮“约0.61 dB”是Oracle相对**每ratio单独续训补偿头**的优势；本轮A要求单个头支持四ratio，不能直接将两种基线混同。旧单ratio头仅作参考，所有正式Recovery以本轮A为分母。

## 4. 训练ratio的真实质量与Oracle恢复率

下表为44 non-Bank视图、两个优化seed均值，Δ相对本轮联合Shared。Recovery先对每seed/ratio计算，再取均值；分母≤0.05 dB时不计算，本轮正式训练档均有正分母。

| ratio | 方法 | ΔPSNR↑ | ΔSSIM↑ | ΔLPIPS↓ | Δphoto↓ | Recovery |
|---|---|---|---|---|---|---|
| 1% | private4 | +0.1520 | +0.00248 | -0.00500 | -0.002050 | 17.6% |
| 1% | private8 | +0.2325 | +0.00375 | -0.00982 | -0.003050 | 26.2% |
| 5% | private4 | +0.1346 | +0.00162 | -0.00243 | -0.000994 | 16.3% |
| 5% | private8 | +0.1076 | +0.00127 | -0.00256 | -0.000617 | 12.7% |
| 10% | private4 | +0.0775 | +0.00185 | -0.00266 | -0.000660 | 12.2% |
| 10% | private8 | +0.0552 | +0.00131 | -0.00176 | -0.000293 | 8.6% |
| 15% | private4 | +0.1113 | +0.00178 | -0.00418 | -0.000693 | 14.1% |
| 15% | private8 | +0.0728 | +0.00140 | -0.00248 | -0.000285 | 9.3% |

训练档每个seed的PSNR/photo均优于A，SSIM/LPIPS也同步改善；8D在1%更强，但5/10/15%均值不如4D，没有稳定的“维度越高越好”。仍保留逐seed和逐view的mean/median/p10/p90/improved fraction。

| split | 方法 | 四档ΔPSNR | 四档ΔSSIM | 四档ΔLPIPS | 四档Δphoto | 平均Recovery |
|---|---|---|---|---|---|---|
| bank | private4 | +0.108736 | +0.001895 | -0.003795 | -0.001221 | 12.9% |
| bank | private8 | +0.100722 | +0.002084 | -0.004111 | -0.001191 | 11.7% |
| nonbank | private4 | +0.118840 | +0.001934 | -0.003567 | -0.001099 | 15.1% |
| nonbank | private8 | +0.117020 | +0.001932 | -0.004154 | -0.001061 | 14.2% |

并非仅Bank改善，但Oracle差距仍大。正式绝对质量如下；四指标、Bank/non-Bank、每seed/ratio的完整值见`quality_summary.csv`和`all_quality_per_view.csv`。

| ratio | A Shared PSNR | 4D PSNR | 8D PSNR | Oracle14 PSNR |
|---|---|---|---|---|
| 1% | 16.4174 | 16.5694 | 16.6499 | 17.3234 |
| 5% | 19.3019 | 19.4365 | 19.4095 | 20.1308 |
| 10% | 19.9497 | 20.0272 | 20.0049 | 20.6112 |
| 15% | 19.9923 | 20.1037 | 20.0651 | 20.7980 |

原4D/8D相对上一轮每ratio单独头，训练档non-Bank平均PSNR仍约低0.072/0.074 dB，没有恢复之前约0.61 dB优势。跨ratio共享造成的限制不能通过此处简单增加身份特征完全消除。

共享容量控制E在本统一LR下明显退化：训练档non-Bank平均−1.427 dB，位移/Anchor边长p90最高187、最大约499，虽finite但属于优化失稳。它只能否定这个具体扩容/初始化/学习率组合，**不能证明任何共享扩容都无效，也不能据它严格排除“仅增加参数”的解释**。

## 5. 未见ratio与连续输出

2.5/7.5/12.5%直接输入相同冻结Encoder和同一个已训练补偿网络，使用相同源Selector集合和实际K。没有ratio专属z、插值表、新训练或未见ratio Oracle。三档选中身份100%属于训练15%集合，故退化不能归因为遇到从未训练过的Gaussian身份。

| 未见ratio | 方法 | ΔPSNR | ΔSSIM | ΔLPIPS | Δphoto |
|---|---|---|---|---|---|
| 2.5% | private4 | -0.014207 | -0.001156 | -0.001086 | +0.000656 |
| 2.5% | private8 | -0.005751 | +0.001108 | -0.001968 | +0.000123 |
| 7.5% | private4 | +0.051068 | +0.001214 | -0.001406 | -0.000429 |
| 7.5% | private8 | +0.018755 | +0.000555 | -0.000124 | +0.000007 |
| 12.5% | private4 | +0.066144 | +0.001444 | -0.003097 | -0.000417 |
| 12.5% | private8 | +0.032927 | +0.000790 | -0.001232 | +0.000102 |

2.5%的PSNR存在seed冲突：4D为−0.060/+0.032 dB，8D为−0.090/+0.079 dB；两种方法的平均photo均恶化。4D在7.5/12.5%保持小幅同步收益；8D虽然PSNR略升，photo没有同步改善。不能宣称未见ratio稳定受益。

57个预定ratio（1%～15%等距）对固定256 Gaussian身份探测，全部14维输出finite，私有/压缩版输出精确一致。结构由log(r)、Linear、ReLU组成，数学上在r>0连续，但可有ReLU导数拐点；数值连续不代表渲染质量插值正确。未验证区间外ratio，也不能凭三个未见ratio给所有连续ratio质量保证。

所有模型的XYZ/Scale/Rotation/Opacity/SH分量、差分斜率和曲率保存于`ratio_smoothness.csv`/npz及补偿幅度CSV。原8D位移p90最高达Anchor边长3.93倍，却仍远低于Oracle质量；因此不能把结果解释为补偿完全不更新或只需放大幅度。

## 6. 私有梯度、补偿分量与可达方向

训练选中身份并集为2021，所有未选中行和其Adam moments精确为零。原4D有1660/1670个身份曾获得梯度并实际更新，8D为1680/1676，约占选中并集82%～83%；私有特征有效更新，非零初始化死路径已排除。

| 训练ratio出现档数 | 身份数 | 每身份索引次数 | 4D曾获梯度比例（两seed） | 8D曾获梯度比例（两seed） |
|---|---|---|---|---|
| 1 | 656 | 2000 | 62.2% / 63.9% | 64.6% / 64.2% |
| 2 | 668 | 4000 | 85.2% / 85.0% | 85.8% / 85.3% |
| 3 | 559 | 6000 | 97.5% / 97.5% | 97.5% / 97.9% |
| 4 | 138 | 8000 | 100% / 100% | 100% / 100% |

频次与梯度覆盖/代码幅度相关，但也与选择层级、可见性及opacity相关，不能直接解释为访问次数造成质量差异。按频次组将z归零的固定8个non-Bank视图条件对照，保存在`frequency_conditional_delta.csv`；不是把每个点的质量贡献线性相加。

### 私有特征被ReLU压缩

对同一Bank photo求直接14维残差梯度g，私有路径Jacobian为 `J_i=W2 diag(a_i+Uz_i>0) U`。J的列空间投影只分析单点z可提供的局部方向，不包含共享头参数的全局更新。欧氏范数混合14维不同物理单位，是局部诊断，非质量上界。解析VJP用固定同一上游g与autograd核对通过。

| ratio | 初始存在直接14D梯度的点 | 初始原注入式可传到z的点 | 被全ReLU死区挡住的点 |
|---|---|---|---|
| 1% | 137 | 137 | 0 |
| 5% | 682 | 672 | 10 |
| 10% | 1237 | 1038 | 199 |
| 15% | 1653 | 1104 | 549 |

5/10/15%原4D/8D的最终局部Jacobian秩中位数均约1；8个代码维度不等于8个有效补偿方向。15%最终仍有805～879个点具有直接14D梯度，却被私有路径死区阻断；部分点无直接梯度则属于当前渲染可见性/截断限制，两者需要区分。

### 分量条件贡献

在固定8 Bank＋8 non-Bank视图中，把一个私有模型补偿分量恢复为本轮A输出，其他分量保持原值。下表是恢复该分量导致的non-Bank PSNR损失（四训练档、两seed均值），仅表示条件贡献，不能相加或作为单分量拟合上限。

| 方法 | XYZ | Scale | Rotation | Opacity | SH DC |
|---|---|---|---|---|---|
| private4 | +0.1820 | +0.1996 | +0.0979 | +0.0098 | +0.0005 |
| private8 | +0.2500 | +0.3134 | +0.1282 | -0.0063 | -0.0101 |
| private4_bypass | +0.5880 | +0.6317 | +0.2387 | +0.0628 | +0.0195 |

主要适配仍在XYZ/Scale/Rotation；8D的Opacity、SH DC条件差值为负，不能视为全部属性都得到改进。相对Oracle的大幅XYZ/Rotation可恢复收益，当前模型仍有明显限制；较大条件差值也可能来自属性间抵消和协同。

## 7. SECONDARY：等参数ReLU-bypass验证

单变量结构：`delta=W2(ReLU(W1h+b1)+Uz)+b2`。同一4D表、同一U、同一参数数、同一初始化及8000步；只移动私有注入位置。它取消了私有路径的ReLU阻断，但其私有项`W2Uz`也变为与ratio无关的加法残差，属于该变更的表达权衡。

初始15%全部1653个有直接梯度的点都能传到z；最终两seed分别1680/1680个可传，J秩中位数为4。可达性得到实证恢复，但质量没有相应恢复。

| ratio | bypass−A ΔPSNR | bypass−原4D ΔPSNR | bypass−A Δphoto |
|---|---|---|---|
| 1% | +0.1491 | -0.0028 | -0.002590 |
| 2.5% | +0.0218 | +0.0360 | -0.001635 |
| 5% | +0.0197 | -0.1148 | -0.000382 |
| 7.5% | -0.0533 | -0.1044 | +0.000397 |
| 10% | -0.0411 | -0.1187 | +0.000059 |
| 12.5% | -0.0524 | -0.1186 | +0.000372 |
| 15% | -0.0004 | -0.1117 | +0.000053 |

bypass训练档Bank平均+0.123 dB，non-Bank仅+0.032 dB，弱于原4D的+0.119 dB；未见档PSNR平均−0.028 dB，seed差异明显。它没有通过质量门槛，不作为修复方案接入生产。**仅恢复梯度覆盖不足以逼近Oracle。**

私有跨ratio梯度也不是普遍反向：原4D的1%/15%全局cosine中位数+0.105，个体负方向占比约34%；8D相应为+0.123、约29%。bypass后全局cosine约+0.452、个体负比例约21%，质量仍较差。不同参数化改变了Jacobian和cosine度量，不能把梯度更同向当作质量改进。

## 8. 参数、显存、K行计算和精确压缩

FP32参数核算必须同时考虑N和实际K。四个独立Oracle合计`14×(138+697+1365+2021)=59094`个残差参数；它不是全N×14×4表，也没有连续ratio零样本能力。

| 方法 | 完整训练参数 | 私有Feature参数字节 | 私有Adam字节 | 压缩推理权重参数 |
|---|---|---|---|---|
| A Shared | 1518 | 0 | 0 | 1518 |
| Private4 | 57322 | 222704 | 445408 | 9734 |
| Private8 | 113126 | 445408 | 890816 | 17950 |
| 4D bypass | 57322 | 222704 | 445408 | 9734 |
| Oracle14四档 | 59094 | 按各档K×14 | 旧独立Adam | 不适用于未见ratio |

完整N表的4D仅略少于稀疏四档Oracle，8D反而更多。训练后将从未被选中的全零身份折叠为共享零行，保留2021个已选身份及一个零行；4D/8D压缩后分别9734/17950个权重参数，另需N×int32映射55676字节。审计用Gaussian ID列表另有16168字节，不计入映射；它不参与forward。

全N逐行重建私有表与原表精确一致，正式7档及57档固定身份probe输出精确一致。此压缩只验证冻结模型推理，不宣称实现了在线按需分配、压缩训练Adam或新点生命周期。压缩减少存储，但多一次映射查找/类型转换，测得更慢。

RTX3090 / PyTorch2.3.1+cu118；固定seed2026权重，32次warmup，5组×64次CUDA-event重复。缓存h，不含Encoder、Selector、Renderer或optimizer step；backward使用固定上游14维梯度，含Embedding稀疏梯度。表中单位µs，forward已包含私有gather。

| ratio | 方法 | K | Forward | Backward | Feature gather | 压缩Forward |
|---|---|---|---|---|---|---|
| 1% | shared | 138 | 59.3 | 214.9 | 0.0 | — |
| 1% | private4 | 138 | 110.0 | 321.9 | 14.4 | 141.1 |
| 1% | private8 | 138 | 112.6 | 313.7 | 14.4 | 146.0 |
| 1% | private4_bypass | 138 | 117.5 | 345.6 | 13.8 | 150.6 |
| 15% | shared | 2021 | 59.2 | 216.2 | 0.0 | — |
| 15% | private4 | 2021 | 111.2 | 315.7 | 14.6 | 143.9 |
| 15% | private8 | 2021 | 112.0 | 312.4 | 14.6 | 145.7 |
| 15% | private4_bypass | 2021 | 115.3 | 319.1 | 13.9 | 150.5 |

15%时，4D/8D相对A的Forward约+52/+53µs，Backward约+100/+96µs；单独Compensation的相对开销明显，绝对值约0.11ms前向、0.31ms反向。不能把它等同于完整Mapping增幅。完整表Feature+Adam额外约0.637/1.274 MiB。

本harness的Torch峰值allocated（15%）约A503.85MiB、4D504.61MiB、8D505.31MiB；减去固定环境基线的模型/工作峰值约1.59/2.25/2.95MiB。私有方法峰值还包含一份压缩推理副本，且包含PyTorch eager launch间隙；不是整卡占用或严格生产显存基准。完整逐ratio与p90见`efficiency.csv`，完整在线Mapping耗时未测。

原型正式forward只执行K行gather/projection/MLP，无全N浮点forward或CPU/GPU同步。全人口统计、Tensor SHA、SVD和逐行诊断在离线边界完成；训练harness中的finite检查、统计和计时开销不代表生产Mapping。可进一步融合K行映射/gather/小矩阵操作，但本轮未改CUDA。

## 9. 在线生命周期接入审计（未实现生产集成）

| 事件 | 必要工作与当前状态 |
|---|---|
| append新观测 | 为新Gaussian建立对应零z行、零Adam状态；保留已有行及状态。未实现 |
| clone/split | 必须沿parents/source_row映射同步创建代码与状态；继承或零初始化会改变补偿连续性，需独立验证，未擅自确定 |
| prune / reorder | 与Master使用完全相同survivor mask/permutation，同时处理z、Adam、compact映射与诊断计数。未实现 |
| 几何修正/闭环 | z随Gaussian身份迁移，不按Anchor Feature成员数平均；Master坐标与h改变后需检验补偿几何一致性。未实现 |
| 身份管理 | 本原型用冻结source row；当前稳定Anchor UID不等于Gaussian UID，不能在重排后直接沿用旧行号 |
| optimizer所有权 | 现有共享Gaussian Adam不宜直接接收稀疏Embedding梯度；需独立私有优化器，并保持Full不更新辅助参数的时钟语义。未实现 |
| checkpoint | 原型已保存head/U/z、source SHA、source N、ID映射、两优化器及访问/更新计数；生产serializer/schema尚未接入 |
| 连续ratio/压缩 | 一个模型直接接受任意合法r的编码h；7档与57档probe已测，压缩训练和动态身份增删未测 |

## 10. 验收、限制与下一步

源码边界、step0、首步梯度、稀疏行更新、Gaussian身份、冻结对象SHA、K、view分离、压缩表重建、数值finite和原生像素贡献计数parity均通过。所有方法保留两个seed及最终checkpoint，不按质量挑选。

本地独立核验：136个文件SHA、10602行正式逐视图、186组摘要、2016行paired统计、64行Recovery，以及10个最终训练模型和6个压缩模型。条件分量、频次消融和逐Gaussian梯度另存CSV/npz。

限制：一个冻结地图和两个优化顺序seed；共享头与旧单ratio Oracle的优化器clock不同；SparseAdam采用全局clock而行访问频次不同；固定8000步未证明全部模型收敛或排除其他优化条件。参数匹配的大头控制已出现优化失稳。secondary为Bank梯度诊断后的探索性验证，使用相同既有non-Bank集合，不构成新的独立场景重复。

| 最终问题 | 回答 |
|---|---|
| 1. 4D/8D恢复多少Oracle收益？ | non-Bank平均约15.1% / 14.2%；1%约17.6% / 26.2%，其余档约9%～16%；未接近Oracle |
| 2. 是否提高等步数原头的non-Bank质量？ | 训练ratio YES，平均+0.119/+0.117dB；未见ratio MIXED |
| 3. 个体补偿还是仅增加参数？ | 个体特征确实更新且消融有贡献；匹配参数控制失稳，不能完整排除容量/优化条件；没有“参数一加就恢复Oracle”的证据 |
| 4. 未见ratio是否保持收益？ | 不稳定；2.5%seed冲突，8D部分档photo退化。无任意ratio质量保证 |
| 5. XYZ/Rotation等是否仍受限？ | YES；主要适配发生在几何分量，仍远低于独立14D，且有激活低秩和跨属性协同 |
| 6. 开销增加多少？ | 15% Comp前向约+52～53µs、反向+96～100µs；Feature+Adam约+0.637/1.274MiB；全Mapping未测 |
| 7. 是否接入生产Joint？ | HOLD：未通过接近Oracle和稳定未见ratio质量门槛 |
| 8. 下一步研究什么？ | 优先私有补偿的连续ratio参数共享/条件表达与优化机制；不能只加d或恢复梯度覆盖，更不支持直接加深整个Selector |

后续可单独研究 `delta_i(r)=C(h_i(r))+V(r)z_i` 一类连续ratio调制的低秩个体补偿：保持一张z表，允许个体修正方向/幅度随ratio变化，同时避免私有路径被全ReLU关闭。**此建议未实现或测试**，不能作为已验证改进。需要新预注册与独立验证，避免继续在同一non-Bank集合上反复挑结构。

停止于本审计。生产仍使用原结构；未自动调Loss、采样、阈值、Feature维度或进一步加深网络；未stage、commit、push。证据：`tests/anchor_selector/gaussian_private_compensation_20261022/evidence/`。
