# SpatialLocalXYZ + Anchor Feature 研究报告

## 结论范围

本轮基于当前生产默认 **SpatialLocalXYZ**，没有改回59D或E50。生产版本 `c49c0222d34c0104fadc7fa4d4e11473fe152b2c`；生产源码/配置哈希核对见 `evidence_validation.json`。本轮仅新增独立研究目录，未stage、commit或push。

首次测量起点 `2026-10-27T08:01:54+00:00`；报告生成时实际经过 **99.9 分钟**。完成 57 个训练尝试，总计 628,000 次真实LoD photo backward。预算约180分钟，没有为凑时长扩大维度或网络层数。

**测试前按Bank低ratio配对收益锁定的研究候选：FrozenBaseHead8**。该选择记录在 `BANK_LOCKED_CANDIDATE.json`，全部正负对照均评价预定终点。本轮non-Bank视图此前已被历史研究使用，属于独立于本轮训练的留出数据，不能称为全新盲测。continuation两seed仅改变优化帧序；scratch的seed同时决定原网络初始化和帧序。两地图各使用两个seed，并非四次独立SLAM建图。

### 当前最重要的判断

**Anchor Feature可以提高固定地图的PSNR，但本轮没有获得足以直接接入生产的稳定低ratio、多指标收益。** 最值得保留的是“已有非零SpatialLocalXYZ保持固定，8D Anchor Feature在四个Head隐藏层融合”的研究候选；普通联合训练和从零训练的收益明显较弱。生产算法和配置保持原样。

以下为两个顺序seed的non-Bank均值，均使用同一源地图、同K、同帧序和新增12k真实LoD曝光：

| 地图 | 等步数继续训练Spatial | FrozenBaseHead8 | 七档ΔPSNR | 1%ΔPSNR | 5%ΔPSNR | 未见ratio均值ΔPSNR | ΔSSIM | ΔLPIPS（正数更差） |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fr1 | 17.8065 | 18.0865 | +0.2799 | +0.1765 | +0.1734 | +0.3824 | +0.0098 | +0.0009 |
| fr2 | 18.4216 | 18.4725 | +0.0509 | +0.0384 | −0.0013 | +0.0984 | +0.0009 | +0.0032 |

fr2去掉2.5%后，其余六档的平均收益只有 **+0.0116dB**。fr2的1%/5%收益很小，LPIPS变差，不能认定普遍改善。普通联合Head8在fr1约+0.1669dB、fr2约−0.0163dB；fr1同seed重跑配对收益变为−0.0144dB。从零训练Head8的两地图七档收益仅+0.0320/+0.0565dB，且低ratio不一致。

### Feature究竟提供了什么

**已证实：** 冻结网络下，完整Anchor表比全局单一Feature有更多可恢复PSNR：fr1约+0.1240dB、fr2约+0.1111dB。它有实际作用，并非所有收益都能由全局Head偏置的冗余参数化解释。

**尚未证实真实空间分组的特殊价值：** 精确匹配成员数及四ratio访问次数的随机分组，non-Bank达到18.1006/18.4653dB，与真实分组18.0865/18.4725dB接近。训练后再打乱Feature会大幅退化，只说明Feature与成员发生了适配，不能反推真实Anchor几何关系是收益来源。

**较符合证据的解释：** Feature提供了分组可直接优化的抽象自由度，帮助共享神经函数吸收局部的重建误差历史。它不等于新增了某种已经辨认出的几何或外观信息，也没有证明55D物理输入缺少不可恢复的信息。latent基底本身不唯一，不能将某维解释为特定物理属性。

收益并非仅由单成员Anchor承担：冻结网络后禁用4～10成员组Feature，PSNR分别下降约0.283/0.080dB；禁用单成员组下降约0.022/0.058dB。普通Joint的MultiOnlyHead8也接近完整Head8。这些最终函数反事实不相加，不代表各组独立因果贡献。

### 优化与属性作用

FrozenBaseHead8的fr1同seed重复为18.0837→18.0823dB，明显比普通联合Head的配对收益稳定；两地图均优于同结构联合Head的均值。**支持“共享网络更新与新Feature共同适配存在优化耦合”这一判断，但尚不能断言系统性的梯度冲突或相互抵消。** FreshAdam控制没有带来一致改善，历史Adam状态不能单独解释主要结果。没有继续添加梯度手术、Loss或正则项。

Feature对几何分支的影响较大：冻结模型中禁用Scale/Rotation/XYZ投影，fr1平均下降约0.128/0.132/0.085dB，fr2约0.172/0.041/0.115dB；Other分支约0.021/0.040dB。这里是真实渲染PSNR反事实，不能当成属性独立贡献的加和。物理输出分析采用世界位移/Anchor单位位移、log-scale、Quaternion旋转测地角、activated opacity和DC变化，没有用原14D欧氏距离替代图像误差。

fr1访问328/2066个Anchor，全部有过非零Feature梯度；fr2访问619/2724个，其中585个有过非零梯度。未访问表行保持零。当前评价Gaussian全部在某个训练ratio被选过，因此**本轮没有检验对全新成员或全新Anchor的Feature迁移能力**。

### 未见ratio的新发现

2.5%虽然处于全局训练ratio范围内，但fr1有102/170、fr2有139/231个选中Gaussian，目标ratio低于该Gaussian实际被选入训练的最低ratio，约占60%。7.5%约33%，12.5%约20%。这是**按Gaussian条件的ratio外推**，不等于新Gaussian，也不等于纯数值插值。

这与2.5%的质量低谷相符，但不能仅凭覆盖统计断言它是唯一原因。后验固定K反事实也没有支持“把外推成员Feature简单禁用即可统一修复”：fr1在2.5%禁用该组Feature提高约0.052dB，fr2反而下降约0.048dB；只保留该组Feature又会产生不同交互。它们只做归因，不成为新生产屏蔽规则。

### 与R3和历史数值的准确关系

本轮源Spatial checkpoint为fr1 seed5100、fr2 seed6100，原始same-K记录与重评分别为17.7454/18.3061dB。用户给出的17.6109/18.2605是上一轮两个seed均值，不能直接相减。

R3同图同集合快照为18.0223/18.2871dB。FrozenBaseHead8七档数值高约0.0642/0.1855dB，但已有Spatial 12k训练后又增加12k曝光，与R3历史训练预算、参数化不同；不能称为公平超越。四个训练ratio仍平均低约0.1132/0.1168dB，5%分别低0.1344/0.1467dB。对低ratio目标仍有差距。

### 成本与后续决策

8D Anchor表及四Head投影新增18576/23840个参数；其权重、梯度、Adam双moment合计约 **0.283/0.364MiB**。PyTorch冻结输入原型中，普通Joint Head8的Forward相较基线增加约0.163/0.178ms，F+B增加约0.492/0.526ms。FrozenBase的Backward更便宜是因为冻结了共享权重，不能当成同样Joint算法的加速；当前结果也不包含生产实时55D/Master梯度或完整Mapping。

**下一轮优先验证等总预算的训练阶段，而不是增加Feature维度。** 用独立地图预注册“先训练非零Spatial，再固定共享权重训练U/e”的切换，与全程Joint及原Spatial等曝光比较。生产零输出初始化不能立即冻结base，否则Feature梯度会永久为零。只有独立地图的1%/5%、未见ratio及LPIPS均给出稳定证据后，再进入在线Joint原型。当前结论为：保留研究候选，暂不接入生产。


## 1. 当前生产架构与实验边界

生产Compensation为55D物理/空间输入、55→64→64、连续log-ratio FiLM、XYZ/Scale/Rotation/Other四Head、Anchor单位XYZ输出，共33806个共享参数。生产Selector独立使用11D输入，不读取本轮Feature。Variable Anchor真实成员数为1～10；源码split采样读取activated Scale与Rotation；本轮固定地图未运行densification。TF/SI/GI/ChangeDistill默认关闭。

实验使用生产Python同函数的只读快照。`production_input_gate.json`直接调用生产实时physical55并核对缓存55D及最终14D；14组两地图七ratio通过原阈值。`correctness_gates.json`覆盖历史非零网络bitwise输出、函数保持初始化、真实photo Feature梯度及UID/Adam迁移。

本轮冻结Master、Anchor身份/几何、Selector、相机和Cached same-K集合。训练时仅更新Compensation相关参数。实时生产CUDA与Joint Master梯度不在本轮质量实验内；缓存输入只适用于该冻结诊断。归一化沿用历史冻结全地图统计，生产统计则在首次在线补偿时固定，二者建立时点不同。

| 地图 | Gaussian | Anchor | Bank训练视图 | Bank留出 | non-Bank |
| --- | ---: | ---: | ---: | ---: | ---: |
| fr1_xyz | 6838 | 2066 | 54 | 13 | 45 |
| fr2_xyz | 9275 | 2724 | 47 | 12 | 64 |

这些是本轮明确锁定的两张Variable Anchor地图，并非早期13,919 Gaussian/5,177 Anchor地图；没有将不同拓扑的历史质量直接混合。训练ratio1/5/10/15%，未见2.5/7.5/12.5%。选点身份直接复用历史R3终点的固定集合，不重算当前Selector11输出；这是Compensation能力对照，不能称为当前生产Selector11全系统质量。same-K精确排序为 `logit(p)-logit(cached_uniform)`，并非单纯p排序；保留原eligibility和身份。K见逐ratio表。损失 `.8L1+.2(1-fusedSSIM)`，训练320×240，评价640×480。PSNR使用clamp图像，SSIM torchmetrics，LPIPS Alex normalize=True，photo使用原始Renderer输出。

## 2. 发现驱动的实验路线

详见 `RESEARCH_LOG.md`。4k Bank筛选比较Early/Mid/FiLM/Head及Context/Hybrid；Head8与AnchorFiLM8短程约+0.20dB且投影参数都为2048，进入匹配12k验证。fr2反例和同seed重跑反转，促使研究转向优化耦合与信息控制，没有机械扩大Feature维度。

Head8：每Anchor共享8D抽象e，四Head隐藏层分别加共享投影U_t e；e不按ratio存表，ratio仍由原FiLM连续条件化。AnchorFiLM8：在第二层用 `[e,phi(log r)e]` 产生64D gain/bias。Context将成员physical19的mean/max/std与log occupancy（58D）通过共享58→32→8编码，减去冻结初始Encoder输出以保持step0。Hybrid连接learned/context。所有最终14D均由共享神经Head生成，无Gaussian私有物理补偿参数。

继续训练从同一12k终点恢复原base完整Adam，各分支再训练12k；新U与e的Adam从0开始。scratch从头训练12k，同一seed的基线与新模型base初始化完全相同，新分支RNG不扰动base。每ratio均3000次曝光，同seed帧序SHA完全匹配，最终终点预先指定。Dense Adam对M×8整表计时，未访问行梯度为0，但已访问行仍可能因动量更新；访问步数和实际更新数分别保存在feature_diagnostics.pt。

FrozenBaseHead8冻结已经训练为非零的原SpatialLocalXYZ，仅训练U和e；若从生产零输出初始化立刻冻结base，Feature梯度会被零末层永久阻断，不能直接冷启动部署。FrozenBaseContext对应共享Context与U。FreshAdam控制只清空base历史状态，作为优化器诊断，不能与原Adam的差异直接归因为网络。

## 3. non-Bank完整结果：两seed均值

scratch、continuation、冻结base与重复分别列出，不跨协议合并：

| tier | scene | config | seeds | psnr | ssim | lpips | photo |
| --- | --- | --- | --- | --- | --- | --- | --- |
| confirm_continue | fr1_xyz | AnchorFiLM8 | 2 | 17.9267 | 0.6268 | 0.5841 | 0.1384 |
| confirm_continue | fr1_xyz | Baseline | 2 | 17.8065 | 0.6248 | 0.5867 | 0.1403 |
| confirm_continue | fr1_xyz | Head8 | 2 | 17.9735 | 0.6281 | 0.5830 | 0.1379 |
| confirm_continue | fr2_xyz | AnchorFiLM8 | 2 | 18.4471 | 0.6230 | 0.5466 | 0.1372 |
| confirm_continue | fr2_xyz | Baseline | 2 | 18.4216 | 0.6220 | 0.5494 | 0.1374 |
| confirm_continue | fr2_xyz | Head8 | 2 | 18.4054 | 0.6235 | 0.5436 | 0.1373 |
| frozen_mechanism | fr1_xyz | FrozenBaseContextHead8 | 2 | 18.0305 | 0.6332 | 0.5902 | 0.1352 |
| frozen_mechanism | fr1_xyz | FrozenBaseFakeHead8 | 2 | 18.1006 | 0.6348 | 0.5861 | 0.1343 |
| frozen_mechanism | fr1_xyz | FrozenBaseGlobalHead8 | 2 | 17.9625 | 0.6308 | 0.5923 | 0.1362 |
| frozen_mechanism | fr2_xyz | FrozenBaseContextHead8 | 2 | 18.4186 | 0.6212 | 0.5590 | 0.1370 |
| frozen_mechanism | fr2_xyz | FrozenBaseFakeHead8 | 2 | 18.4653 | 0.6227 | 0.5525 | 0.1363 |
| frozen_mechanism | fr2_xyz | FrozenBaseGlobalHead8 | 2 | 18.3614 | 0.6199 | 0.5626 | 0.1377 |
| frozen_mechanism | fr2_xyz | FrozenBaseHead8 | 2 | 18.4725 | 0.6229 | 0.5527 | 0.1362 |
| frozen_same_seed_repeat | fr1_xyz | FrozenBaseGlobalHead8 | 1 | 17.9712 | 0.6307 | 0.5917 | 0.1361 |
| frozen_same_seed_repeat | fr1_xyz | FrozenBaseHead8 | 1 | 18.0823 | 0.6346 | 0.5876 | 0.1346 |
| mechanism_continue | fr1_xyz | FakeMatchedHead8 | 2 | 17.9558 | 0.6320 | 0.5829 | 0.1375 |
| mechanism_continue | fr1_xyz | FrozenBaseHead8 | 2 | 18.0865 | 0.6346 | 0.5876 | 0.1345 |
| mechanism_continue | fr1_xyz | GlobalHead8 | 2 | 17.9088 | 0.6273 | 0.5876 | 0.1384 |
| mechanism_continue | fr1_xyz | MultiOnlyHead8 | 2 | 17.9715 | 0.6298 | 0.5852 | 0.1375 |
| optimizer_control | fr1_xyz | FreshAdamBaseline | 2 | 17.8294 | 0.6280 | 0.5867 | 0.1394 |
| optimizer_control | fr1_xyz | FreshAdamHead8 | 2 | 17.9555 | 0.6314 | 0.5833 | 0.1376 |
| same_seed_repeat | fr1_xyz | Baseline | 1 | 18.0266 | 0.6274 | 0.5824 | 0.1384 |
| same_seed_repeat | fr1_xyz | Head8 | 1 | 18.0122 | 0.6320 | 0.5815 | 0.1384 |
| scratch_confirm | fr1_xyz | Baseline | 2 | 17.8621 | 0.6264 | 0.5943 | 0.1393 |
| scratch_confirm | fr1_xyz | Head8 | 2 | 17.8941 | 0.6272 | 0.5926 | 0.1389 |
| scratch_confirm | fr2_xyz | Baseline | 2 | 18.1912 | 0.6164 | 0.5608 | 0.1406 |
| scratch_confirm | fr2_xyz | Head8 | 2 | 18.2477 | 0.6185 | 0.5563 | 0.1397 |

历史源Spatial12k和R3快照在同地图、同集合、同评价视图重评：

| scene | reference | psnr | ssim | lpips | photo |
| --- | --- | --- | --- | --- | --- |
| fr1_xyz | R3_snapshot | 18.0223 | 0.6285 | 0.5917 | 0.1370 |
| fr1_xyz | Spatial_initial12k | 17.7454 | 0.6263 | 0.5965 | 0.1388 |
| fr2_xyz | R3_snapshot | 18.2871 | 0.6184 | 0.5553 | 0.1392 |
| fr2_xyz | Spatial_initial12k | 18.3061 | 0.6188 | 0.5640 | 0.1382 |

R3快照只作质量参照，其训练参数化及历史曝光不同；不能把差距解释为Anchor Feature单独可恢复的理论上限。

## 4. 每ratio质量与实际预算

以下为主要continuation研究臂的两seed均值；scratch和所有控制的完整逐ratio表见 `quality_per_ratio.csv`。实际有效点数为 **radii>0且opacity≥1/255的代理**，不等于经过像素积分测得的真实渲染贡献。

| scene | config | ratio | psnr | ssim | lpips | photo | K | visible | effective |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fr1_xyz | AnchorFiLM8 | 0.0100 | 16.2970 | 0.5860 | 0.7022 | 0.1617 | 68.0000 | 57.6778 | 57.6778 |
| fr1_xyz | AnchorFiLM8 | 0.0250 | 14.8946 | 0.5597 | 0.6688 | 0.1786 | 170.0000 | 146.1556 | 146.1556 |
| fr1_xyz | AnchorFiLM8 | 0.0500 | 18.4566 | 0.6358 | 0.5940 | 0.1312 | 341.0000 | 281.3556 | 281.3556 |
| fr1_xyz | AnchorFiLM8 | 0.0750 | 18.5159 | 0.6429 | 0.5627 | 0.1291 | 512.0000 | 408.4000 | 408.4000 |
| fr1_xyz | AnchorFiLM8 | 0.1000 | 18.9914 | 0.6527 | 0.5313 | 0.1239 | 683.0000 | 530.7000 | 530.7000 |
| fr1_xyz | AnchorFiLM8 | 0.1250 | 19.1267 | 0.6541 | 0.5211 | 0.1228 | 854.0000 | 640.9889 | 640.9111 |
| fr1_xyz | AnchorFiLM8 | 0.1500 | 19.2050 | 0.6563 | 0.5083 | 0.1218 | 1025.0000 | 761.6556 | 761.5000 |
| fr1_xyz | Baseline | 0.0100 | 16.2475 | 0.5864 | 0.7052 | 0.1624 | 68.0000 | 57.9222 | 57.9222 |
| fr1_xyz | Baseline | 0.0250 | 14.7546 | 0.5688 | 0.6721 | 0.1789 | 170.0000 | 146.3556 | 146.3556 |
| fr1_xyz | Baseline | 0.0500 | 18.3319 | 0.6338 | 0.5958 | 0.1331 | 341.0000 | 281.9222 | 281.9222 |
| fr1_xyz | Baseline | 0.0750 | 18.3990 | 0.6387 | 0.5642 | 0.1314 | 512.0000 | 408.9111 | 408.9111 |
| fr1_xyz | Baseline | 0.1000 | 18.8644 | 0.6483 | 0.5318 | 0.1262 | 683.0000 | 530.3778 | 530.3778 |
| fr1_xyz | Baseline | 0.1250 | 19.0053 | 0.6480 | 0.5244 | 0.1252 | 854.0000 | 640.4444 | 639.9333 |
| fr1_xyz | Baseline | 0.1500 | 19.0431 | 0.6499 | 0.5133 | 0.1245 | 1025.0000 | 760.7667 | 759.2222 |
| fr1_xyz | FrozenBaseContextHead8 | 0.0100 | 16.3989 | 0.5905 | 0.7036 | 0.1588 | 68.0000 | 58.9333 | 58.9333 |
| fr1_xyz | FrozenBaseContextHead8 | 0.0250 | 15.4104 | 0.5843 | 0.6744 | 0.1675 | 170.0000 | 145.5889 | 145.5889 |
| fr1_xyz | FrozenBaseContextHead8 | 0.0500 | 18.4664 | 0.6371 | 0.6033 | 0.1300 | 341.0000 | 280.4000 | 280.4000 |
| fr1_xyz | FrozenBaseContextHead8 | 0.0750 | 18.6583 | 0.6453 | 0.5689 | 0.1267 | 512.0000 | 407.7000 | 407.7000 |
| fr1_xyz | FrozenBaseContextHead8 | 0.1000 | 18.9924 | 0.6550 | 0.5370 | 0.1223 | 683.0000 | 530.1111 | 530.1111 |
| fr1_xyz | FrozenBaseContextHead8 | 0.1250 | 19.1027 | 0.6583 | 0.5293 | 0.1210 | 854.0000 | 640.9444 | 640.8556 |
| fr1_xyz | FrozenBaseContextHead8 | 0.1500 | 19.1842 | 0.6618 | 0.5151 | 0.1197 | 1025.0000 | 761.0778 | 760.9000 |
| fr1_xyz | FrozenBaseHead8 | 0.0100 | 16.4240 | 0.5909 | 0.7039 | 0.1585 | 68.0000 | 59.1000 | 59.1000 |
| fr1_xyz | FrozenBaseHead8 | 0.0250 | 15.3632 | 0.5844 | 0.6744 | 0.1676 | 170.0000 | 146.4222 | 146.4222 |
| fr1_xyz | FrozenBaseHead8 | 0.0500 | 18.5053 | 0.6380 | 0.5996 | 0.1295 | 341.0000 | 281.6000 | 281.6000 |
| fr1_xyz | FrozenBaseHead8 | 0.0750 | 18.7443 | 0.6470 | 0.5640 | 0.1258 | 512.0000 | 409.2444 | 409.2444 |
| fr1_xyz | FrozenBaseHead8 | 0.1000 | 19.0758 | 0.6571 | 0.5339 | 0.1213 | 683.0000 | 531.8444 | 531.8444 |
| fr1_xyz | FrozenBaseHead8 | 0.1250 | 19.1985 | 0.6604 | 0.5254 | 0.1200 | 854.0000 | 642.8778 | 642.8000 |
| fr1_xyz | FrozenBaseHead8 | 0.1500 | 19.2942 | 0.6644 | 0.5117 | 0.1186 | 1025.0000 | 763.2889 | 763.1333 |
| fr1_xyz | Head8 | 0.0100 | 16.3008 | 0.5870 | 0.7039 | 0.1613 | 68.0000 | 58.3556 | 58.3556 |
| fr1_xyz | Head8 | 0.0250 | 15.0061 | 0.5631 | 0.6686 | 0.1773 | 170.0000 | 147.0111 | 147.0111 |
| fr1_xyz | Head8 | 0.0500 | 18.4362 | 0.6359 | 0.5946 | 0.1314 | 341.0000 | 284.0556 | 284.0556 |
| fr1_xyz | Head8 | 0.0750 | 18.5913 | 0.6425 | 0.5583 | 0.1287 | 512.0000 | 412.7111 | 412.7111 |
| fr1_xyz | Head8 | 0.1000 | 19.0662 | 0.6541 | 0.5282 | 0.1231 | 683.0000 | 535.1333 | 535.1333 |
| fr1_xyz | Head8 | 0.1250 | 19.1749 | 0.6562 | 0.5200 | 0.1221 | 854.0000 | 646.0000 | 645.9111 |
| fr1_xyz | Head8 | 0.1500 | 19.2387 | 0.6577 | 0.5077 | 0.1213 | 1025.0000 | 767.4222 | 767.2556 |
| fr2_xyz | AnchorFiLM8 | 0.0100 | 17.1327 | 0.5943 | 0.6953 | 0.1535 | 92.0000 | 87.6641 | 87.6641 |
| fr2_xyz | AnchorFiLM8 | 0.0250 | 16.3582 | 0.5849 | 0.6175 | 0.1626 | 231.0000 | 218.9375 | 218.9375 |
| fr2_xyz | AnchorFiLM8 | 0.0500 | 18.8361 | 0.6270 | 0.5420 | 0.1329 | 463.0000 | 431.6328 | 431.6328 |
| fr2_xyz | AnchorFiLM8 | 0.0750 | 18.9970 | 0.6321 | 0.5139 | 0.1308 | 695.0000 | 629.1719 | 626.3359 |
| fr2_xyz | AnchorFiLM8 | 0.1000 | 19.2199 | 0.6391 | 0.4970 | 0.1275 | 927.0000 | 827.8594 | 818.2891 |
| fr2_xyz | AnchorFiLM8 | 0.1250 | 19.2527 | 0.6401 | 0.4883 | 0.1271 | 1159.0000 | 1008.5234 | 984.0469 |
| fr2_xyz | AnchorFiLM8 | 0.1500 | 19.3327 | 0.6431 | 0.4726 | 0.1259 | 1391.0000 | 1200.1094 | 1166.7578 |
| fr2_xyz | Baseline | 0.0100 | 17.1576 | 0.5943 | 0.6957 | 0.1531 | 92.0000 | 87.5312 | 87.5312 |
| fr2_xyz | Baseline | 0.0250 | 16.3845 | 0.5856 | 0.6163 | 0.1621 | 231.0000 | 218.6953 | 218.6953 |
| fr2_xyz | Baseline | 0.0500 | 18.8196 | 0.6264 | 0.5434 | 0.1330 | 463.0000 | 431.4062 | 431.4062 |
| fr2_xyz | Baseline | 0.0750 | 18.9756 | 0.6314 | 0.5173 | 0.1310 | 695.0000 | 627.2109 | 621.9922 |
| fr2_xyz | Baseline | 0.1000 | 19.1768 | 0.6376 | 0.5003 | 0.1280 | 927.0000 | 825.6641 | 813.6797 |
| fr2_xyz | Baseline | 0.1250 | 19.1946 | 0.6383 | 0.4933 | 0.1277 | 1159.0000 | 1004.8906 | 977.9062 |
| fr2_xyz | Baseline | 0.1500 | 19.2428 | 0.6404 | 0.4799 | 0.1269 | 1391.0000 | 1195.3359 | 1159.2109 |
| fr2_xyz | FrozenBaseContextHead8 | 0.0100 | 17.1845 | 0.5945 | 0.6980 | 0.1526 | 92.0000 | 86.8516 | 86.8516 |
| fr2_xyz | FrozenBaseContextHead8 | 0.0250 | 16.7438 | 0.5906 | 0.6189 | 0.1579 | 231.0000 | 218.0625 | 218.0625 |
| fr2_xyz | FrozenBaseContextHead8 | 0.0500 | 18.7364 | 0.6252 | 0.5527 | 0.1332 | 463.0000 | 429.5938 | 429.5938 |
| fr2_xyz | FrozenBaseContextHead8 | 0.0750 | 18.8890 | 0.6295 | 0.5274 | 0.1311 | 695.0000 | 624.6953 | 620.1094 |
| fr2_xyz | FrozenBaseContextHead8 | 0.1000 | 19.0984 | 0.6354 | 0.5128 | 0.1284 | 927.0000 | 823.2109 | 811.7109 |
| fr2_xyz | FrozenBaseContextHead8 | 0.1250 | 19.1096 | 0.6359 | 0.5068 | 0.1282 | 1159.0000 | 1001.6406 | 975.2891 |
| fr2_xyz | FrozenBaseContextHead8 | 0.1500 | 19.1688 | 0.6375 | 0.4965 | 0.1274 | 1391.0000 | 1190.3828 | 1156.0000 |
| fr2_xyz | FrozenBaseHead8 | 0.0100 | 17.1960 | 0.5947 | 0.6986 | 0.1525 | 92.0000 | 86.8750 | 86.8750 |
| fr2_xyz | FrozenBaseHead8 | 0.0250 | 16.6711 | 0.5892 | 0.6187 | 0.1589 | 231.0000 | 218.5781 | 218.5781 |
| fr2_xyz | FrozenBaseHead8 | 0.0500 | 18.8184 | 0.6266 | 0.5482 | 0.1323 | 463.0000 | 430.1016 | 430.1016 |
| fr2_xyz | FrozenBaseHead8 | 0.0750 | 18.9856 | 0.6320 | 0.5191 | 0.1299 | 695.0000 | 626.9531 | 622.4141 |
| fr2_xyz | FrozenBaseHead8 | 0.1000 | 19.1687 | 0.6379 | 0.5033 | 0.1272 | 927.0000 | 825.4062 | 814.4609 |
| fr2_xyz | FrozenBaseHead8 | 0.1250 | 19.1931 | 0.6387 | 0.4963 | 0.1269 | 1159.0000 | 1006.3125 | 980.4062 |
| fr2_xyz | FrozenBaseHead8 | 0.1500 | 19.2748 | 0.6415 | 0.4845 | 0.1258 | 1391.0000 | 1196.4766 | 1163.3125 |
| fr2_xyz | Head8 | 0.0100 | 17.1318 | 0.5948 | 0.6945 | 0.1533 | 92.0000 | 87.9453 | 87.9453 |
| fr2_xyz | Head8 | 0.0250 | 16.2238 | 0.5844 | 0.6144 | 0.1639 | 231.0000 | 219.0391 | 219.0391 |
| fr2_xyz | Head8 | 0.0500 | 18.8090 | 0.6275 | 0.5370 | 0.1329 | 463.0000 | 432.1016 | 432.1016 |
| fr2_xyz | Head8 | 0.0750 | 18.9802 | 0.6333 | 0.5088 | 0.1306 | 695.0000 | 630.0156 | 627.4844 |
| fr2_xyz | Head8 | 0.1000 | 19.1838 | 0.6397 | 0.4942 | 0.1275 | 927.0000 | 828.6953 | 822.1641 |
| fr2_xyz | Head8 | 0.1250 | 19.2185 | 0.6408 | 0.4849 | 0.1271 | 1159.0000 | 1009.5078 | 987.7656 |
| fr2_xyz | Head8 | 0.1500 | 19.2904 | 0.6438 | 0.4717 | 0.1260 | 1391.0000 | 1200.8750 | 1171.9922 |

![两地图七ratio质量](figures/primary_quality.png)

## 5. 配对差值、波动与训练曲线

每个seed及重复均保留：

| tier | scene | model | seed | ratio | psnr_delta_mean | psnr_delta_median | psnr_delta_p10 | psnr_delta_p90 | psnr_improved_fraction | ssim_delta_mean | lpips_delta_mean | photo_delta_mean | all_metrics_winner |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| confirm_continue | fr1_xyz | AnchorFiLM8 | 7200 | seven | 0.1574 | 0.1441 | -0.0657 | 0.3906 | 0.8032 | 0.0033 | -0.0021 | -0.0027 | candidate |
| confirm_continue | fr1_xyz | AnchorFiLM8 | 7201 | seven | 0.0830 | 0.0604 | -0.0122 | 0.2272 | 0.8730 | 0.0006 | -0.0032 | -0.0009 | candidate |
| confirm_continue | fr1_xyz | Head8 | 7200 | seven | 0.1703 | 0.1679 | -0.0731 | 0.4643 | 0.8032 | 0.0033 | -0.0043 | -0.0027 | candidate |
| confirm_continue | fr1_xyz | Head8 | 7201 | seven | 0.1635 | 0.1306 | -0.0158 | 0.3610 | 0.8762 | 0.0032 | -0.0030 | -0.0021 | candidate |
| confirm_continue | fr2_xyz | AnchorFiLM8 | 7200 | seven | 0.0263 | 0.0414 | -0.1211 | 0.1363 | 0.7054 | 0.0012 | -0.0052 | -0.0003 | candidate |
| confirm_continue | fr2_xyz | AnchorFiLM8 | 7201 | seven | 0.0245 | 0.0146 | -0.0759 | 0.1545 | 0.5915 | 0.0007 | -0.0004 | -0.0001 | candidate |
| confirm_continue | fr2_xyz | Head8 | 7200 | seven | -0.0348 | -0.0081 | -0.2118 | 0.0979 | 0.4576 | 0.0015 | -0.0067 | -0.0001 | mixed |
| confirm_continue | fr2_xyz | Head8 | 7201 | seven | 0.0022 | -0.0003 | -0.1059 | 0.1202 | 0.4978 | 0.0014 | -0.0049 | -0.0000 | candidate |
| frozen_mechanism | fr1_xyz | FrozenBaseContextHead8 | 7200 | seven | 0.1529 | 0.0955 | -0.4234 | 0.7863 | 0.6159 | 0.0102 | 0.0053 | -0.0061 | mixed |
| frozen_mechanism | fr1_xyz | FrozenBaseContextHead8 | 7201 | seven | 0.2950 | 0.2893 | -0.1033 | 0.7088 | 0.8349 | 0.0064 | 0.0017 | -0.0041 | mixed |
| frozen_mechanism | fr1_xyz | FrozenBaseFakeHead8 | 7200 | seven | 0.2022 | 0.1839 | -0.4421 | 0.8572 | 0.6508 | 0.0115 | 0.0021 | -0.0068 | mixed |
| frozen_mechanism | fr1_xyz | FrozenBaseFakeHead8 | 7201 | seven | 0.3860 | 0.4062 | -0.0797 | 0.7831 | 0.8730 | 0.0084 | -0.0033 | -0.0052 | candidate |
| frozen_mechanism | fr1_xyz | FrozenBaseGlobalHead8 | 7200 | seven | 0.0672 | -0.0067 | -0.5597 | 0.8141 | 0.4952 | 0.0075 | 0.0080 | -0.0049 | mixed |
| frozen_mechanism | fr1_xyz | FrozenBaseGlobalHead8 | 7201 | seven | 0.2448 | 0.2649 | -0.2363 | 0.6976 | 0.7587 | 0.0043 | 0.0032 | -0.0033 | mixed |
| frozen_mechanism | fr2_xyz | FrozenBaseContextHead8 | 7200 | seven | -0.0261 | -0.0243 | -0.3811 | 0.3356 | 0.4665 | -0.0006 | 0.0118 | -0.0002 | baseline |
| frozen_mechanism | fr2_xyz | FrozenBaseContextHead8 | 7201 | seven | 0.0200 | 0.0505 | -0.4624 | 0.4211 | 0.5201 | -0.0010 | 0.0073 | -0.0007 | mixed |
| frozen_mechanism | fr2_xyz | FrozenBaseFakeHead8 | 7200 | seven | 0.0353 | 0.0504 | -0.2582 | 0.3036 | 0.5692 | 0.0010 | 0.0049 | -0.0011 | mixed |
| frozen_mechanism | fr2_xyz | FrozenBaseFakeHead8 | 7201 | seven | 0.0519 | 0.0764 | -0.3846 | 0.4501 | 0.5759 | 0.0003 | 0.0012 | -0.0012 | mixed |
| frozen_mechanism | fr2_xyz | FrozenBaseGlobalHead8 | 7200 | seven | -0.0685 | -0.0671 | -0.5093 | 0.3695 | 0.4196 | -0.0017 | 0.0151 | 0.0003 | baseline |
| frozen_mechanism | fr2_xyz | FrozenBaseGlobalHead8 | 7201 | seven | -0.0520 | -0.0252 | -0.6049 | 0.4032 | 0.4777 | -0.0026 | 0.0112 | 0.0002 | baseline |
| frozen_mechanism | fr2_xyz | FrozenBaseHead8 | 7200 | seven | 0.0352 | 0.0410 | -0.2560 | 0.3379 | 0.5603 | 0.0013 | 0.0052 | -0.0010 | mixed |
| frozen_mechanism | fr2_xyz | FrozenBaseHead8 | 7200 | low | -0.0478 | -0.0422 | -0.2749 | 0.1581 | 0.3984 | 0.0003 | 0.0046 | 0.0001 | mixed |
| frozen_mechanism | fr2_xyz | FrozenBaseHead8 | 7200 | unseen | 0.1252 | 0.1274 | -0.2070 | 0.4675 | 0.6771 | 0.0022 | 0.0045 | -0.0020 | mixed |
| frozen_mechanism | fr2_xyz | FrozenBaseHead8 | 7201 | seven | 0.0666 | 0.0986 | -0.3593 | 0.4529 | 0.6205 | 0.0006 | 0.0013 | -0.0013 | mixed |
| frozen_mechanism | fr2_xyz | FrozenBaseHead8 | 7201 | low | 0.0849 | 0.1143 | -0.3718 | 0.4945 | 0.6406 | 0.0003 | 0.0031 | -0.0013 | mixed |
| frozen_mechanism | fr2_xyz | FrozenBaseHead8 | 7201 | unseen | 0.0716 | 0.0986 | -0.3409 | 0.4248 | 0.6615 | 0.0008 | 0.0004 | -0.0014 | mixed |
| frozen_same_seed_repeat | fr1_xyz | FrozenBaseGlobalHead8 | 7200 | seven | -0.0553 | -0.0878 | -0.6541 | 0.6365 | 0.4413 | 0.0034 | 0.0093 | -0.0022 | mixed |
| frozen_same_seed_repeat | fr1_xyz | FrozenBaseHead8 | 7200 | seven | 0.0558 | 0.0882 | -0.5558 | 0.6077 | 0.5746 | 0.0072 | 0.0052 | -0.0038 | mixed |
| frozen_same_seed_repeat | fr1_xyz | FrozenBaseHead8 | 7200 | low | -0.0110 | 0.0324 | -0.4040 | 0.3346 | 0.5333 | 0.0024 | 0.0040 | -0.0022 | mixed |
| frozen_same_seed_repeat | fr1_xyz | FrozenBaseHead8 | 7200 | unseen | 0.1729 | 0.2501 | -0.4626 | 0.6788 | 0.6519 | 0.0133 | 0.0062 | -0.0062 | mixed |
| mechanism_continue | fr1_xyz | FakeMatchedHead8 | 7200 | seven | 0.1615 | 0.1449 | -0.0276 | 0.4547 | 0.8667 | 0.0111 | -0.0043 | -0.0035 | candidate |
| mechanism_continue | fr1_xyz | FakeMatchedHead8 | 7201 | seven | 0.1371 | 0.1354 | -0.0104 | 0.2894 | 0.8698 | 0.0032 | -0.0033 | -0.0021 | candidate |
| mechanism_continue | fr1_xyz | FrozenBaseHead8 | 7200 | seven | 0.1797 | 0.1650 | -0.4457 | 0.7834 | 0.6635 | 0.0113 | 0.0037 | -0.0065 | mixed |
| mechanism_continue | fr1_xyz | FrozenBaseHead8 | 7200 | low | 0.0441 | 0.0575 | -0.5007 | 0.5455 | 0.5889 | 0.0034 | 0.0018 | -0.0038 | mixed |
| mechanism_continue | fr1_xyz | FrozenBaseHead8 | 7200 | unseen | 0.3176 | 0.3228 | -0.3214 | 0.9233 | 0.7481 | 0.0136 | 0.0045 | -0.0082 | mixed |
| mechanism_continue | fr1_xyz | FrozenBaseHead8 | 7201 | seven | 0.3801 | 0.4002 | -0.0733 | 0.7666 | 0.8603 | 0.0082 | -0.0020 | -0.0051 | candidate |
| mechanism_continue | fr1_xyz | FrozenBaseHead8 | 7201 | low | 0.3057 | 0.3133 | -0.1172 | 0.7055 | 0.8222 | 0.0053 | 0.0007 | -0.0037 | mixed |
| mechanism_continue | fr1_xyz | FrozenBaseHead8 | 7201 | unseen | 0.4471 | 0.4950 | -0.0076 | 0.8289 | 0.8963 | 0.0107 | -0.0024 | -0.0065 | candidate |
| mechanism_continue | fr1_xyz | GlobalHead8 | 7200 | seven | 0.1178 | 0.1065 | -0.0861 | 0.3426 | 0.7397 | 0.0033 | 0.0007 | -0.0024 | mixed |
| mechanism_continue | fr1_xyz | GlobalHead8 | 7201 | seven | 0.0868 | 0.0746 | -0.0433 | 0.2489 | 0.7937 | 0.0017 | 0.0011 | -0.0014 | mixed |
| mechanism_continue | fr1_xyz | MultiOnlyHead8 | 7200 | seven | 0.2098 | 0.1979 | -0.0064 | 0.4500 | 0.8921 | 0.0074 | -0.0016 | -0.0039 | candidate |
| mechanism_continue | fr1_xyz | MultiOnlyHead8 | 7201 | seven | 0.1200 | 0.0976 | -0.0305 | 0.2912 | 0.8222 | 0.0026 | -0.0013 | -0.0017 | candidate |
| optimizer_control | fr1_xyz | FreshAdamBaseline | 7200 | seven | -0.0396 | -0.0295 | -0.2308 | 0.2395 | 0.3905 | 0.0040 | 0.0006 | -0.0001 | mixed |
| optimizer_control | fr1_xyz | FreshAdamBaseline | 7201 | seven | 0.0853 | 0.0633 | -0.0361 | 0.2326 | 0.7683 | 0.0023 | -0.0006 | -0.0015 | candidate |
| optimizer_control | fr1_xyz | FreshAdamHead8 | 7200 | seven | 0.2158 | 0.1899 | 0.0504 | 0.4129 | 0.9524 | 0.0064 | -0.0046 | -0.0034 | candidate |
| optimizer_control | fr1_xyz | FreshAdamHead8 | 7201 | seven | 0.0364 | 0.0473 | -0.0690 | 0.1383 | 0.7143 | 0.0005 | -0.0022 | -0.0002 | candidate |
| same_seed_repeat | fr1_xyz | Head8 | 7200 | seven | -0.0144 | -0.0008 | -0.1895 | 0.1499 | 0.4952 | 0.0046 | -0.0009 | -0.0000 | mixed |
| scratch_confirm | fr1_xyz | Head8 | 7200 | seven | 0.0550 | 0.0491 | -0.1665 | 0.2923 | 0.5905 | 0.0006 | 0.0006 | -0.0005 | mixed |
| scratch_confirm | fr1_xyz | Head8 | 7201 | seven | 0.0091 | 0.0263 | -0.1841 | 0.2166 | 0.5714 | 0.0008 | -0.0041 | -0.0004 | candidate |
| scratch_confirm | fr2_xyz | Head8 | 7200 | seven | -0.0188 | 0.0102 | -0.2221 | 0.1395 | 0.5379 | 0.0003 | -0.0017 | 0.0002 | mixed |
| scratch_confirm | fr2_xyz | Head8 | 7201 | seven | 0.1319 | 0.1229 | -0.0828 | 0.3747 | 0.7612 | 0.0040 | -0.0072 | -0.0021 | candidate |

`paired_deltas.csv`另含逐视图聚类bootstrap区间；相邻时序视图相关，视图区间不能替代独立地图seed不确定性。fr1同seed重跑已出现约0.15dB级别的基线变化及普通Head配对收益反转，因此小幅均值提升不认定稳定收益。

![Bank收敛曲线](figures/bank_curves.png)

1000/6000/12000终点评价均为Bank，未据non-Bank挑步数或学习率。曲线仍变化的模型不能称为表达能力极限。本轮不做学习率搜索。

## 6. Anchor信息机制和occupancy

`feature_statistics.csv`分别记录1成员、2～3、4～10组的访问、非零梯度覆盖、有效秩与实际Adam更新。fr1只有328/2066个Anchor被四个固定训练集合访问，不能将全表平均梯度作为有效学习比例。单成员Anchor的身份记忆作用与多成员共享必须区别。

FakeMatchedHead8在“真实occupancy×四ratio Gaussian选中签名”内打乱Feature归属，精确保留每表行成员数与每档训练访问次数，改变97.7% Gaussian的Feature分组。它保留真实physical55/中心/位移尺度，仅破坏Feature的局部共成员关系。这个控制不等同于仅重命名embedding行；单成员本来就没有共成员关系可破坏。未见ratio访问频次未被强制匹配，解释泛化控制时需保留此限制。

GlobalHead8只有全局一份8D，与原四Head偏置存在冗余参数化；它测优化/全局容量效应，参数量未与完整Anchor表相等。MultiOnly只给2～10成员Anchor启用Feature，避免将收益全部归因于单成员身份。

`mechanism_ablation_summary.csv`给出Feature置零、同occupancy及四ratio Anchor访问签名内打乱、按occupancy禁用、按属性Head禁用、对训练从未选中成员禁用Feature的真实渲染反事实。它们使用已经共同适配的base和U，不能把置零损害等同于新增Feature的因果质量增益，分组贡献也不相加。

每个模型的dense_ratio_continuity.csv对固定1%身份密采141个ratio，检查有限值、相邻输出变化及Anchor单位位移。ratio_gradient.json测一个Bank相机的跨ratio Anchor梯度方向，只作局部诊断；平滑输出及负cosine都不能单独证明泛化或损害。

## 7. 参数、显存与性能

8D表参数8M：fr1为16528、fr2为21792；四Head投影各8×64，总2048。相较33806参数base，learned Head8总量分别52382/57646。Feature权重+梯度+Adam双Moment合计128M bytes，fr1约0.252MiB、fr2约0.333MiB；投影另约0.031MiB。没有ratio乘数。只Gather选中K行，Dense Adam仍扫M行。

以下为三块交错CUDA Events、块内median后平均、七ratio等权：

| scene | model | stage | gpu_ms |
| --- | --- | --- | --- |
| fr1_xyz | AnchorFiLM8 | backward_only | 1.3463 |
| fr1_xyz | AnchorFiLM8 | feature_gather | 0.0225 |
| fr1_xyz | AnchorFiLM8 | forward | 0.7974 |
| fr1_xyz | AnchorFiLM8 | forward_backward | 2.3844 |
| fr1_xyz | AnchorFiLM8 | gather55 | 0.0111 |
| fr1_xyz | AnchorFiLM8 | render_photo_adam | 5.3734 |
| fr1_xyz | Baseline | backward_only | 1.1716 |
| fr1_xyz | Baseline | forward | 0.6301 |
| fr1_xyz | Baseline | forward_backward | 2.0804 |
| fr1_xyz | Baseline | gather55 | 0.0135 |
| fr1_xyz | Baseline | render_photo_adam | 4.9005 |
| fr1_xyz | FrozenBaseContextHead8 | backward_only | 0.7285 |
| fr1_xyz | FrozenBaseContextHead8 | feature_gather | 0.0177 |
| fr1_xyz | FrozenBaseContextHead8 | forward | 0.9620 |
| fr1_xyz | FrozenBaseContextHead8 | forward_backward | 1.8925 |
| fr1_xyz | FrozenBaseContextHead8 | gather55 | 0.0121 |
| fr1_xyz | FrozenBaseContextHead8 | render_photo_adam | 4.7247 |
| fr1_xyz | FrozenBaseHead8 | backward_only | 0.6625 |
| fr1_xyz | FrozenBaseHead8 | feature_gather | 0.0233 |
| fr1_xyz | FrozenBaseHead8 | forward | 0.8124 |
| fr1_xyz | FrozenBaseHead8 | forward_backward | 1.6868 |
| fr1_xyz | FrozenBaseHead8 | gather55 | 0.0128 |
| fr1_xyz | FrozenBaseHead8 | render_photo_adam | 4.4929 |
| fr1_xyz | Head8 | backward_only | 1.4504 |
| fr1_xyz | Head8 | feature_gather | 0.0236 |
| fr1_xyz | Head8 | forward | 0.7927 |
| fr1_xyz | Head8 | forward_backward | 2.5721 |
| fr1_xyz | Head8 | gather55 | 0.0120 |
| fr1_xyz | Head8 | render_photo_adam | 5.6048 |
| fr2_xyz | AnchorFiLM8 | backward_only | 1.3700 |
| fr2_xyz | AnchorFiLM8 | feature_gather | 0.0206 |
| fr2_xyz | AnchorFiLM8 | forward | 0.7956 |
| fr2_xyz | AnchorFiLM8 | forward_backward | 2.4885 |
| fr2_xyz | AnchorFiLM8 | gather55 | 0.0111 |
| fr2_xyz | AnchorFiLM8 | render_photo_adam | 5.5169 |
| fr2_xyz | Baseline | backward_only | 1.1822 |
| fr2_xyz | Baseline | forward | 0.6377 |
| fr2_xyz | Baseline | forward_backward | 2.1050 |
| fr2_xyz | Baseline | gather55 | 0.0142 |
| fr2_xyz | Baseline | render_photo_adam | 4.9849 |
| fr2_xyz | FrozenBaseContextHead8 | backward_only | 0.7349 |
| fr2_xyz | FrozenBaseContextHead8 | feature_gather | 0.0227 |
| fr2_xyz | FrozenBaseContextHead8 | forward | 0.9345 |
| fr2_xyz | FrozenBaseContextHead8 | forward_backward | 1.9121 |
| fr2_xyz | FrozenBaseContextHead8 | gather55 | 0.0153 |
| fr2_xyz | FrozenBaseContextHead8 | render_photo_adam | 4.7932 |
| fr2_xyz | FrozenBaseHead8 | backward_only | 0.6657 |
| fr2_xyz | FrozenBaseHead8 | feature_gather | 0.0181 |
| fr2_xyz | FrozenBaseHead8 | forward | 0.8349 |
| fr2_xyz | FrozenBaseHead8 | forward_backward | 1.6872 |
| fr2_xyz | FrozenBaseHead8 | gather55 | 0.0111 |
| fr2_xyz | FrozenBaseHead8 | render_photo_adam | 4.6088 |
| fr2_xyz | Head8 | backward_only | 1.4622 |
| fr2_xyz | Head8 | feature_gather | 0.0256 |
| fr2_xyz | Head8 | forward | 0.8155 |
| fr2_xyz | Head8 | forward_backward | 2.6310 |
| fr2_xyz | Head8 | gather55 | 0.0153 |
| fr2_xyz | Head8 | render_photo_adam | 5.6997 |

这是冻结55D输入的PyTorch原型成本；Context的58D成员统计也已预计算，聚合成本未计入。生产55D实时CUDA、Master反传和完整Mapping均未计入，不能与生产性能报告中的在线范围直接相除。服务器另有用户GPU进程，未锁时钟或抢占；性能数字是共享负载下的初步测量。精确参数、Adam与显存计数见performance_*.json/csv。FrozenBase减少base梯度，不能把其F+B优势当成同样Joint算法的纯Kernel加速。

## 8. 生命周期与CUDA路径

生产原AnchorBank已有按UID/存活行迁移参数和Adam的机制；本轮独立helper实际验证reorder/prune/append的Feature与moment、优化器alias替换、新行零初始化和旧tensor级step保留；lifecycle_followup.json进一步通过迁移后真实Adam更新及checkpoint恢复后的下一步更新。没有完成生产新Feature接入。

建议独立 `compensation_anchor_features[M,8]`，与Selector旧表分开注册；新Anchor默认e=0、U小非零，初始恢复原Spatial函数。clone/split留在同Anchor时共享原e；新Anchor zero；grow不迁移已有成员。prune/reorder按稳定UID重排行；空间key碰撞不能充当身份。

局部闭环成员迁移可沿用当前成员数加权Feature聚合并清接收行moments，但**抽象latent的线性加权不保证神经补偿连续或质量保持**，仍需事件级验证。整Anchor刚体移动保留UID/e，物理55D和Context按最新几何重算。新行使用共享tensor Adam时钟并不等于独立年龄，需要在后续在线研究明确，而不能伪造零梯度step。

CUDA接入方案：保留现有live55D Kernel，在K行读取owner→e；四Head GEMM前融合U_t e相加，Backward计算U梯度并对重复owner Scatter/Add。Context需在成员/属性变化后重算有版本的聚合，不能读取过期缓存；Dense Adam可先复用，Sparse更新改变moment时钟，不应作为等价优化暗中替换。

## 9. 验证状态与局限

- **IMPLEMENTED / TESTED**：独立Feature原型、函数保持初始化、真实photo训练、两地图两顺序seed、从零对照、访问匹配假分组、冻结base、全局和Context控制、同seed重复、固定K、逐视图PSNR/SSIM/LPIPS/photo、梯度/更新/occupancy统计、输入同函数gate和UID helper。
- **PARTIALLY VERIFIED**：连续ratio在1%～15%七档及固定身份密采；不代表所有ratio质量保证。当前性能是共享GPU冻结输入原型；生命周期helper未覆盖真实SLAM事件。
- **NOT IMPLEMENTED**：生产Feature集成、在线Joint质量、闭环Feature混合质量、Sparse Adam和新CUDA Feature Kernel、独立第三地图盲测。

本轮模型都由共享网络生成14D；允许的Anchor抽象表与禁止的Gaussian直接物理残差严格区分。单成员Anchor可能发挥Gaussian身份记忆，不能隐去这一边界。

## 10. 逐项因果判断

逐项判断已在文首人工核验结论中列出。完整原始证据保留于模型目录，推荐方案仅为下一轮研究候选，不修改生产默认。57个尝试中，7个为4k筛选，其余50个为12k正式/机制对照；关键发现已完成两地图、两seed、从零与同seed重复。因低ratio与LPIPS尚不足以进入生产，停止扩大结构搜索，提前整理交付，没有继续用剩余预算增加维度或选择性重跑。
