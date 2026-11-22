# Ratio-conditioned Gaussian-private Compensation 审计

日期：2026-10-22。状态：**隔离原型 IMPLEMENTED；双seed冻结地图质量/梯度/性能 TESTED；生产集成 NOT IMPLEMENTED。**

**结论：本次线性 Ratio-conditioned Private4D 不支持进入生产Joint。** 相比等步数Shared，新确认集训练四档仅提升 **+0.0211 dB**，三个未见ratio平均下降 **−0.1156 dB**，Oracle恢复率仅 **1.38%**。独立Constant投影控制更好：对应 **+0.1187 / +0.0940 dB**、恢复 **14.46%**。这没有突破旧Private4D约15%的恢复水平。

没有更改生产Selector、Master、Anchor、Loss、TF、采样器、ratio调度或默认配置；没有stage、commit、push。源码HEAD为 `02cd92ff4fcab299008ba094be6966b70e842061`。

## 1. 当前真实架构与冻结边界

- 默认采用Variable Anchor，成员数1～10，并非强制十点。源地图 **13,919 Gaussian / 5,177 Anchor**；各层均有实际成员。Feature16、输入27；生产ratio-feature表关闭。
- 输入为 `Anchor16 + local offset3 + activated scale3 + SH DC3 + activated opacity1 + log(r)1`。Encoder `27→32 ReLU`；Selector私有分支保持原样。原Compensation是 `32→32 ReLU→14`。
- TF、SI/GI、ChangeDistill保持关闭。生产Joint语义保持；本轮是完全冻结地图诊断，没有在线clone/split/prune、SLAM轨迹更新或Joint训练。
- 源checkpoint为服务器 `budget-shared-audit-20261022/baseline_seed2026/6900_shutdown/ply`；不是重新训练的地图。源文件SHA、冻结张量SHA及当前本地12个核心文件SHA全部核对。训练只更新各分支的补偿头、私有z和对应投影。
- 四个训练ratio的Cached实际K为 **138 / 697 / 1,365 / 2,021**；未见2.5%/7.5%/12.5%为 **343 / 1,052 / 1,749**。沿用原Cached集合；原始logit差减logit(cached uniform)在这个实际K下精确复现集合。未替换成概率Top-K或重新采样。
- 训练union共2,021个Gaussian；未见ratio选中点100%来自该union。本轮考察的是已有Gaussian的ratio泛化，不是新Gaussian身份泛化。

## 2. 原型与预注册协议

原头记为 `C(h)=W2 ReLU(W1h+b1)+b2`。A/B/C来自已完成旧实验，但逐项核对源SHA、集合/初始化、Bank顺序、LR、优化器及曝光次数后复用；在新增确认视图上全部重新评价。B/C没有重跑挑选更好seed。

| 组 | 唯一结构差别 | 训练模型数量 |
| --- | --- | --- |
| A | 原Shared头继续训练 | 复用2 |
| B | `W2 ReLU(W1h+b1+Uz)+b2` | 复用2 |
| C | `W2[ReLU(W1h+b1)+Uz]+b2`；有效投影受W2约束 | 复用2 |
| D | `C(h)+V0 z`，自由14×4投影 | 新训2 |
| E | `C(h)+[V0+φ(log r)V1]z` | 新训2 |
| F | 每档选中Gaussian独立14D；仅训练ratio的参考 | 复用8个表 |

每个Gaussian只有一份4D `z`，不绑定Anchor、不按ratio存表，Selector不读取它。D/E只gather选中K行。连续条件为：

```text
φ(r) = 2 [log(r) − log(0.01)] / [log(0.15) − log(0.01)] − 1
δ_D = C(h(r)) + V0 z_i
δ_E = C(h(r)) + [V0 + φ(r)V1] z_i
```

不clamp、不查表；正ratio可连续计算，区间外不作质量保证。训练节点和未见节点均输入同一模型，无插值专属z。

- 两组复制同一原头，`z=0`、相同 `V0~N(0,0.01)`（初始化seed6102）、E的`V1=0`。七个ratio的step0输出均与原头**exact相等**；两组首步有效z梯度范数均约0.003438，127行非零。初始V梯度为零是z=0的预期行为，z先获得梯度，随后V学习。
- Dense Adam更新原头和V，SparseAdam更新z；LR均0.001、无weight decay、fresh状态。z使用完整[N,4]存储，只有K个索引参与每步稀疏更新。没有附加loss或regularization。
- 70个Bank，320×240训练，640×480正式评价；原photo不变：`0.8 L1 + 0.2(1-fast_ssim)`。每seed2,000次Bank帧循环，每帧依次四档；单模型共8,000步，每档2,000次曝光。F每档2,000步，与对应曝光/Bank序列一致。
- seed2026/2027只改变优化视图顺序，**不是两个独立场景或地图seed**。只评价最终checkpoint，不按non-Bank调参、挑步数或挑seed。
- 旧44 non-Bank已用于之前结构研究，只作复用诊断。新增32确认视图在任何新质量读取前，按RGB行号等间隔固定，排除旧64评价帧及历史Bank union243条路径；使用现有轨迹最近时间位姿（误差<0.08秒）。其与Bank/旧评价不重合，但仍是同一场景相近轨迹，不能当作独立场景复现。

完整协议见 `evidence/results/preregistration.json`；选择集、视图、pose源文件和所有模型均保留。

## 3. Oracle跨ratio结构

分析8个已有Independent14D结果。四档共同选中138点；另分析至少在后2/3档共同出现的1,365/697点。以下主要结论来自138点，不能代表只在高ratio出现的全部Gaussian。

属性表示使用：XYZ增量/Anchor边长、log-scale增量、相对初始补偿旋转的SO(3) rotvec、**激活后opacity变化**、SH DC增量。四元数先归一化并处理正负双覆盖，另记录真实旋转geodesic；canonical等价检查通过。不是直接对原始14维做统一欧氏距离。

- 1%到15%的XYZ补偿方向cosine中位约0.525/0.542，约33.1%/30.9%有效行方向相反。实际旋转差中位约0.191/0.186 rad，p90约0.526/0.619 rad。不同ratio需要的变化并非单纯统一缩放。
- 去除每Gaussian的跨ratio均值后，分别对XYZ/Scale/Rotation/SH DC做Gaussian×(ratio×属性) SVD，rank4分别解释约83～86% / 84～87% / 87～88% / 96%的变化能量。
- 五类属性各自按RMS归一化，再联合SVD，rank4仅解释 **48.28% / 48.08%**；未去跨ratio均值时为42.57% /42.90%。标量opacity四列本来rank≤4，其100%解释率不构成“四维足够”的证据。
- 任意每点线性log-ratio拟合解释约 **71.30% /69.09%** 的归一化跨ratio变化；再限制共同Gaussian潜变量rank4，整体相对数值RMSE约 **77.95% /78.04%**。该低秩系数近似不是直接图像优化，也不是原共享头加残差的严格能力上界。
- Oracle不同优化seed的差异并非零：XYZ跨seed RMS约0.270 Anchor边长，对比其跨ratio RMS0.529。Oracle结果是非唯一局部优化解，不能当成平滑且唯一的监督函数。

判断：**有局部、分属性可利用结构；一份4D解释所有属性/ratio的证据不足。** 本轮没有以Oracle拟合监督任何正式模型，也没有把低秩数值误差换算成PSNR。

## 4. 正式重建质量

下表全部相对等步数A；PSNR/SSIM越高越好，LPIPS/photo越低越好。按ratio和两个seed等权汇总。恢复率先按seed/ratio分别计算，再平均：`(candidate−A)/(F−A)`；分母≤0.05 dB不计算。本次训练节点均超过阈值，负恢复率保留，不截成零。

### Bank 70帧：训练ratio四档

Shared绝对均值：PSNR 19.2492、SSIM 0.672160、LPIPS 0.516417、photo 0.119219。

| 方案 | ΔPSNR ↑ | ΔSSIM ↑ | ΔLPIPS ↓ | Δphoto ↓ | Oracle恢复率 |
| --- | --- | --- | --- | --- | --- |
| B Private4/ReLU | +0.1087 | +0.001895 | -0.003795 | -0.001221 | 12.94% |
| C Private4/Bypass | +0.1230 | +0.002424 | -0.005463 | -0.001514 | 14.68% |
| D Constant4 | +0.2015 | +0.002964 | -0.007126 | -0.001970 | 24.04% |
| E Ratio-conditioned4 | +0.0882 | +0.001487 | -0.003635 | -0.001058 | 10.39% |
| F Independent14 | +0.8336 | +0.017793 | -0.027996 | -0.009369 | 100% |

### 复用non-Bank 44帧：训练ratio四档

Shared绝对均值：PSNR 18.9154、SSIM 0.650087、LPIPS 0.539738、photo 0.126707。

| 方案 | ΔPSNR ↑ | ΔSSIM ↑ | ΔLPIPS ↓ | Δphoto ↓ | Oracle恢复率 |
| --- | --- | --- | --- | --- | --- |
| B Private4/ReLU | +0.1188 | +0.001934 | -0.003567 | -0.001099 | 15.05% |
| C Private4/Bypass | +0.0318 | +0.001460 | -0.004023 | -0.000715 | 2.47% |
| D Constant4 | +0.1341 | +0.002149 | -0.006370 | -0.001237 | 16.76% |
| E Ratio-conditioned4 | +0.0286 | +0.000547 | -0.002457 | -0.000340 | 2.38% |
| F Independent14 | +0.8005 | +0.018199 | -0.027282 | -0.009433 | 100% |

### 新增确认32帧：训练ratio四档

Shared绝对均值：PSNR 18.7842、SSIM 0.641261、LPIPS 0.549501、photo 0.129012。

| 方案 | ΔPSNR ↑ | ΔSSIM ↑ | ΔLPIPS ↓ | Δphoto ↓ | Oracle恢复率 |
| --- | --- | --- | --- | --- | --- |
| B Private4/ReLU | +0.1116 | +0.001988 | -0.003665 | -0.001095 | 14.04% |
| C Private4/Bypass | +0.0361 | +0.001395 | -0.004127 | -0.000659 | 3.03% |
| D Constant4 | +0.1187 | +0.001912 | -0.006100 | -0.001081 | 14.46% |
| E Ratio-conditioned4 | +0.0211 | +0.000430 | -0.002288 | -0.000200 | 1.38% |
| F Independent14 | +0.7948 | +0.018182 | -0.027201 | -0.009447 | 100% |

### 新确认集逐ratio的PSNR变化

| ratio | 实际K | B−A | C−A | D−A | E−A |
| --- | --- | --- | --- | --- | --- |
| 1% | 138 | +0.1434 | +0.1504 | +0.2217 | +0.1842 |
| 2.5%（未见） | 343 | -0.0187 | +0.0070 | +0.1967 | -0.2287 |
| 5% | 697 | +0.1260 | +0.0264 | +0.1200 | -0.1014 |
| 7.5%（未见） | 1052 | +0.0370 | -0.0493 | +0.0432 | -0.1031 |
| 10% | 1365 | +0.0717 | -0.0348 | +0.0524 | -0.0480 |
| 12.5%（未见） | 1749 | +0.0631 | -0.0445 | +0.0420 | -0.0152 |
| 15% | 2021 | +0.1053 | +0.0023 | +0.0807 | +0.0497 |

### 三个未见ratio合计：四项指标

| 视图集合 | 方案 | ΔPSNR ↑ | ΔSSIM ↑ | ΔLPIPS ↓ | Δphoto ↓ |
| --- | --- | --- | --- | --- | --- |
| Bank | B Private4/ReLU | +0.0388 | +0.000371 | -0.001567 | -0.000284 |
| Bank | C Private4/Bypass | +0.0666 | +0.001937 | -0.003412 | -0.001104 |
| Bank | D Constant4 | +0.1701 | +0.002954 | -0.005853 | -0.001860 |
| Bank | E Ratio-conditioned4 | -0.0573 | -0.000662 | -0.001244 | +0.000604 |
| 旧non-Bank | B Private4/ReLU | +0.0343 | +0.000500 | -0.001863 | -0.000064 |
| 旧non-Bank | C Private4/Bypass | -0.0280 | +0.001042 | -0.002225 | -0.000289 |
| 旧non-Bank | D Constant4 | +0.1147 | +0.002139 | -0.005261 | -0.001179 |
| 旧non-Bank | E Ratio-conditioned4 | -0.1121 | -0.001306 | +0.000333 | +0.001308 |
| 新确认集 | B Private4/ReLU | +0.0271 | +0.000413 | -0.001668 | +0.000045 |
| 新确认集 | C Private4/Bypass | -0.0289 | +0.000813 | -0.001815 | -0.000086 |
| 新确认集 | D Constant4 | +0.0940 | +0.001835 | -0.004700 | -0.000934 |
| 新确认集 | E Ratio-conditioned4 | -0.1156 | -0.001466 | +0.000465 | +0.001481 |

### 新确认集逐ratio Oracle恢复率

| ratio | B | C | D | E |
| --- | --- | --- | --- | --- |
| 1% | 16.85% | 17.70% | 25.30% | 20.73% |
| 5% | 15.15% | 1.95% | 14.69% | -13.84% |
| 10% | 11.00% | -7.01% | 7.85% | -7.80% |
| 15% | 13.14% | -0.51% | 10.02% | 6.43% |

### 两个seed完整保留

| seed | 方案 | 新确认训练ratio ΔPSNR | 新确认未见ratio ΔPSNR |
| --- | --- | --- | --- |
| 2026 | D Constant4 | +0.1277 | +0.1196 |
| 2026 | E Ratio-conditioned4 | +0.0472 | -0.0742 |
| 2027 | D Constant4 | +0.1097 | +0.0684 |
| 2027 | E Ratio-conditioned4 | -0.0049 | -0.1571 |

E在2.5%出现最明显退化，而且两个seed一致；5%和10%训练节点也下降。D在两个seed的七档新确认视图PSNR均优于A，但训练节点相对B仅+0.0071 dB，SSIM/photo并未全面优于B，不能称为新的高质量突破。

![逐ratio质量及seed范围](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/ratio_conditioned_private_compensation_20261022/quality_delta.png)

## 5. 实际渲染贡献与补偿分量

正式结果包含11,388条逐视图记录，覆盖234个method/seed/ratio/split组合，四项指标完整。`contributing`为原生Renderer中真正参与像素累积的Gaussian数，不是简单的opacity阈值代理；另保留visible与pixel contribution count。156个代表视图Python/原生图像及radii一致性检查通过，全部计数满足`contributing≤visible≤K`。

新确认集真实贡献点均值（两seed）：

| ratio | 相同存储/选择K | A贡献点 | D贡献点 | E贡献点 |
| --- | --- | --- | --- | --- |
| 1% | 138 | 117.7 | 118.0 | 117.7 |
| 5% | 697 | 507.7 | 509.0 | 506.3 |
| 10% | 1365 | 872.8 | 877.8 | 874.2 |
| 15% | 2021 | 1109.0 | 1117.5 | 1114.0 |

固定8 Bank、8旧non-Bank、8新确认视图做属性条件替换：把候选模型某一输出块替换为已训练A对应输出，其他块保留，不重训。新确认集训练节点上，D的XYZ/Rotation块替换后PSNR分别下降约 **0.597/0.275 dB**，E约 **0.346/0.376 dB**。Scale也有显著条件贡献。数值不可相加：补偿属性和各模块已共同适配，替换破坏协同不等于该模块的独立可恢复收益。

与Oracle实际属性对照，XYZ的RMS误差（Anchor边长单位）从A约0.770降为D/E约0.717/0.718；旋转geodesic RMS从A约0.301 rad变为D/E约0.305/0.309 rad。**没有证据表明E改善了Rotation限制。** Oracle属性误差仅作辅助诊断，不是图像质量的等价指标。

1%补偿XYZ/Anchor边长p90：A约2.44～2.61，D约3.06～3.12，E约2.76～3.15。长幅度补偿仍存在；本轮没有引入clamp或新约束。

## 6. 梯度、连续性与V1实际贡献

- 两个新模型在全部七档的私有Jacobian均为rank4；128个真实photo/VJP检查中，所测非零14D梯度均获得非零z梯度，没有出现旧ReLU完全阻断。rank4仍不是14维全方向可达的保证。
- D最终曾有梯度的行数为1,721/1,717；E为1,720/1,715，占训练union约85%。剩余未覆盖行的原因没有逐步分解，不能仅凭覆盖率归因。11,898个从未被选中的z及Adam m/v均保持exact零。覆盖不是全population100%，也不应要求从未参与渲染的身份获得梯度。
- `||V1||/||V0||`为1.11/1.47；1%与15%投影之差为V0范数的2.22/2.93倍。**E没有退化成近似常数投影。** 在端点，V1项约占最终私有残差范数的70%～80%。
- 在E已训练模型内直接清零V1，新确认8帧探针的训练ratio均值再下降约0.590 dB；分组清零的条件影响主要来自XYZ约0.354、Rotation约0.198、Scale约0.155 dB。说明E依赖V1，并不说明E优于独立训练的D；其共享头、z和V0已经对V1发生共同适配。
- 57个密集ratio×固定256 Gaussian×双seed输出均finite；训练节点两侧相对ratio扰动1e-5时，所有属性块最大输出差约2.67e-5。没有观察到表切换跳变。**输出连续没有带来未见ratio质量稳定性。** 仅测试了训练区间内密集采样，未证明所有合法ratio的质量。

### 同一模型、同一真实上游梯度的投影控制

正式质量读取后增加了机制诊断，明确标为探索性：固定E训练后状态和真实渲染的`gδ`，只比较代数回传`V(r)^T gδ`与`V0^T gδ`；使用同一z坐标和共同有效行。不是另一个训练分支，不能把该VJP替换当作新的可用算法。

| seed | 1% vs15% 回传 | 全局gradient cosine | 负方向Gaussian比例 |
| --- | --- | --- | --- |
| 2026 | 仅V0回传 | 0.2922 | 21.4% |
| 2026 | V(r)回传 | 0.1133 | 28.7% |
| 2027 | 仅V0回传 | 0.2751 | 19.5% |
| 2027 | V(r)回传 | -0.0424 | 49.2% |

两seed平均，V1调制把同一上游的cosine从约0.284降至0.035，负方向比例从约20.5%升至38.9%。这比不同模型之间直接比较cosine更能隔离投影本身的作用；原始上游梯度和权重均保存，可CPU复算。它支持“当前调制加大共享z的跨ratio梯度冲突”，**仍不是质量退化唯一原因的因果证明**。

关键关系：`∂L/∂z=V(r)^T gδ`，`∂L/∂V0=Σgδ zᵀ`，`∂L/∂V1=φ(r)Σgδ zᵀ`。同一z同时控制不同属性和ratio；E的矩阵变化会旋转/拉伸各档回传方向。

## 7. 计算与显存

RTX3090，CUDA events，32 warmup，5×64重复；同一N/K、缓存好的K行h，包含私有gather和Python eager kernel launch间隙。Backward使用固定14D上游；不含Encoder、Selector、Renderer、optimizer step或完整Mapping。模型分别测量，不并行争用GPU。峰值是Torch allocated，约502 MiB为已加载视图/冻结模型背景。

| 方案 | ratio | Forward μs | Backward μs | gather μs | Torch峰值 MiB |
| --- | --- | --- | --- | --- | --- |
| A Shared | 1% | 60.9 | 244.5 | 0.0 | 503.70 |
| A Shared | 15% | 57.0 | 229.5 | 0.0 | 504.23 |
| B Private4/ReLU | 1% | 109.8 | 348.4 | 14.0 | 504.55 |
| B Private4/ReLU | 15% | 117.8 | 351.4 | 15.2 | 505.11 |
| C Private4/Bypass | 1% | 119.7 | 350.4 | 14.7 | 504.57 |
| C Private4/Bypass | 15% | 118.7 | 350.0 | 14.6 | 505.36 |
| D Constant4 | 1% | 117.0 | 355.9 | 14.8 | 504.55 |
| D Constant4 | 15% | 115.4 | 353.3 | 14.7 | 505.08 |
| E Ratio-conditioned4 | 1% | 148.2 | 399.3 | 14.5 | 504.55 |
| E Ratio-conditioned4 | 15% | 147.3 | 399.1 | 14.7 | 505.08 |

- 15%时E比D约多31.9 μs forward /45.8 μs backward；E比A约多90.3/169.6 μs，峰值约多0.85 MiB。未进行CUDA化或推理缓存优化，不代表完整在线Mapping延迟。
- 可训练参数：A1,518，B/C57,322，D57,250，E57,306；E只比D多56个参数。全population z占222,704 bytes，z的Adam m/v占445,408 bytes，二者合计约0.637 MiB。E全部Adam状态458,472 bytes，D458,020 bytes。
- **本次实际参数并不比Oracle少很多**：四档Oracle实际只存选中行，共14×(138+697+1365+2021)=59,094个参数；E完整N行表加头共57,306。若仅存训练union，E理论上为8,084+1,518+112=9,714个参数，另需身份映射，但本轮没有实现这种生产压缩/在线生命周期管理，不能把理论压缩作为已测性能。
- D双seed每模型8,000步约35.6秒，E约36.4～36.8秒；这是冻结地图、半分辨率Bank的单机诊断耗时，不是在线SLAM速度。

## 8. 瓶颈判断与下一步

1. **当前线性ratio调制没有质量价值，E NO-GO。** 它显著学到V1、获得rank4梯度、数值连续，却弱于D并损害全部三个未见ratio。不能再把主要问题归结为ReLU阻断或ratio输入缺失。
2. E包含D的精确子空间（V1=0），起点也完全一致；因此E比D更差，不能用“E容量更小”解释。Bank训练节点photo改善也较弱，说明不只是non-Bank过拟合，更需要检查C/z/V联合优化与跨ratio共享耦合。
3. Oracle低秩结果提示全部属性共同4D的结构限制，但它不是渲染质量上界；目前仍不能区分线性V形式本身的最优能力、有限步数优化及C与私有残差之间的可辨识性。增加维度/层数没有本轮支持。
4. **建议的下一项独立诊断**：保持D或E结构、固定地图/集合，比较“冻结C、只优化z/V”与“联合优化C/z/V”，保持所有训练预算一致，检测共同适配是否吞掉私有收益；再决定是否研究参数组优化，而非马上加深MLP。此建议本轮未执行，也不加入新loss。
5. 当前Encoder一直冻结，不能声称已经证明Encoder与私有补偿的训练梯度互相干扰。D仅有有限冻结地图正面结果，需独立场景/地图验证后才考虑受控在线Joint；生产结构保持原样。

## 9. 验收回答

| 问题 | 回答 |
| --- | --- |
| 1. Oracle存在可利用低维结构？ | PARTIAL。分属性较低秩；全部属性共同4D不足。线性log-ratio有部分结构，但不是可直接监督的唯一真值。 |
| 2. E明显优于原4D及Constant？ | NO。新确认集训练节点E相对Shared仅+0.0211 dB，低于B的+0.1116及D的+0.1187 dB。 |
| 3. 恢复多少Oracle收益？ | E：旧non-Bank 2.38%、新确认1.38%；D：16.76%、14.46%。没有实质突破此前约15%。 |
| 4. 未见ratio泛化改善？ | NO。E三个未见ratio在两个seed均退化；新确认集平均−0.1156 dB。D为+0.0940 dB。 |
| 5. 主要剩余瓶颈？ | E相对D更像联合优化/跨ratio梯度耦合；强共享4D仍可能限制绝对Oracle恢复。Encoder被冻结，其自身是否是主因仍未确定。 |
| 6. 时间和显存？ | 15%：E相对A forward约+90.3 μs、backward约+169.6 μs，Torch峰值约+0.85 MiB；仅补偿头微基准。 |
| 7. 进入生产Joint？ | E：NO-GO。本轮保留生产结构；D只有单场景冻结地图的有限正面证据，HOLD生产接入。 |

## 10. 证据与复现

本地实验目录：`tests/anchor_selector/ratio_conditioned_private_compensation_20261022/`；服务器隔离目录：`/home/prmi/zwj/ratio-conditioned-private-compensation-20261022`。四个新模型、六个旧控制权重、八个Oracle表、源checkpoint/相机/轨迹、schedule、逐视图指标、原生计数、原始梯度及数值诊断均在`evidence/`，`delivery_manifest.json`逐文件SHA。

主要文件：

- `results/oracle_low_rank.csv`、`oracle_linear_log_fit.csv`、`oracle_cross_ratio.csv`、`oracle_seed_stability.csv`。
- `quality_per_view_with_counts.csv`、`quality_summary.csv`、`paired_statistics.csv`、`oracle_recovery.csv`、`overall_deltas.csv`。
- `training_curve.csv`、`feature_coverage.csv`、`private_jacobian.csv`、`projection_statistics.csv`、`cross_ratio_gradients.csv`、`gradient_vjp_checks.csv`。
- `conditional_quality.csv`、`conditional_deltas.csv`、`oracle_physical_error.csv`、`compensation_amplitude.csv`、`ratio_smoothness.csv`及原始NPZ。
- `projection_same_upstream_control.csv`、`projection_photo_upstream.npz`；该项是质量读取后的机制分析，不是预注册质量对照。
- `efficiency.csv`、`benchmark_protocol.json`、`correctness.json`、本地`verification.json`。

运行顺序：`oracle_structure.py` → `ratio_experiment.py train` → `ratio_experiment.py evaluate` → `diagnostics.py` → `native_counts.py` → `postprocess.py`。机制补充为`projection_vjp_control.py`；最终证据由`export_evidence.py`导出。

本地使用CPU Torch2.4.1独立复算逐视图聚合、1,704组配对统计、96行恢复率、Oracle SVD、稀疏Adam未使用行、代码SHA及Git边界。Windows该环境须设`MKL_THREADING_LAYER=SEQUENTIAL`避免重复OpenMP运行库，然后运行`verify_delivery.py`。生产tracked工作区保持干净。
