# Anchor LoD 渲染质量瓶颈诊断

结论：**本轮固定地图下，可恢复质量主要出现在补偿参数化；有限换点搜索只找到小幅收益。Joint 存在 Full/LoD 质量权衡，但没有发现初始整体梯度普遍反向。**

独立14维补偿在当前集合上带来平均 **+0.806 dB non-Bank PSNR**；原补偿头等步数续训只能恢复 **+0.196 dB**。独立补偿仍领先等步数补偿头 **+0.609 dB**。这支持补偿共享参数化/可学习性受限，尚不能将原因进一步唯一归结为网络深度、宽度或共享编码器。

所有训练和搜索只使用 Bank；non-Bank 不参与选学习率、步数或 checkpoint。生产源码、Loss、MLP、采样器、TF、Anchor、默认配置均未修改；没有 stage、commit、push。

## 1. 当前真实代码架构与版本核对

本地 HEAD：`02cd92ff4fcab299008ba094be6966b70e842061`，分支 `redundancy-aware-si`。12个核心文件 SHA 与上一轮基线相同，tracked 工作区保持干净。Python 推理通过当前源码的严格 checkpoint schema/owner 校验；源渲染 PSNR/photo 与此前正式 evaluator 通过一致性 gate（0.005 dB / 5e-5）。

统一源 checkpoint：`/home/prmi/zwj/budget-shared-audit-20261022/baseline_seed2026/6900_shutdown/ply`。**13919 Gaussian / 5177 Anchor 是 Variable Anchor 地图，绝不是固定十点地图。**

| 项目 | 实际实现 |
|---|---|
| Anchor | 每个非空 Anchor 1..10 点；h0=0.1，L0/L1/L2；粗细节点各自持有显式 Gaussian |
| 初始化/增长 | SLAM真实观测去重，初始化L0/L1多尺度 occupied cells；不足10不补点；L2由增长产生 |
| Topology | Gaussian 回合 clone/split（容量10约束）与 Anchor growing/pruning 回合交替；topology_round=12 |
| Pruning | 仅 Anchor opacity窗口质量删除；传统逐点opacity/size质量删除关闭，split父替换仍执行 |
| Selector | Feature16 + Offset3 + Scale3 + SH DC3 + Opacity1 + log(r)1 = 27；27→32共享，32→32→2选择私有分支 |
| Compensation | 选中后解码共享32维；32→32→14；从iteration3000启用；最后层零初始化 |
| Joint | LoD更新Master、Selector、Anchor Feature、补偿；Full只更新Master；共享Adam各组按梯度是否存在推进 |
| 关闭项 | TF、SI/GI、ChangeDistill、ratio_features、global_xyz、adaptive occupancy refinement均关闭 |
| 保护/预算 | maturity=1；slot停用244，当前eligible=13675；预算loss以全部N=13919为分母 |
| 停止规则 | 默认max_mapping_iterations=0，保留自然停止；本轮不重跑异步SLAM或修改停止规则 |

各层成员统计：

| 层 | Anchor | Gaussian | 当前 eligible | 空Anchor | occupancy中位数/最大 |
|---|---|---|---|---|---|
| 0 | 1410 | 4611 | 4520 | 0 | 2.0/10 |
| 1 | 3565 | 9059 | 8906 | 0 | 1.0/10 |
| 2 | 202 | 249 | 249 | 0 | 1.0/9 |

成员数1/2/3的Anchor分别为2692/844/484，满10点仅356个。全体节点都持有有效归属成员；这里“有效归属”不等于每帧都可见或都有像素贡献。固定十点历史实现仍在代码中，仅用于兼容；本轮不混用其结果。

源码位置：`cfg/selector/selector_config.yaml`；`improvements/selector/gumbel_network.h:87`（共享forward）、`:99`（14维变换）；`src/gaussian_model.cpp:1151`（可变成员clone/split）、`:1207`（交替拓扑）、`:223`（Anchor prune）；`src/gaussian_mapper.cpp:1590`、`:2305`、`:3236`（优化器/梯度路由）。

### 统一数据与 same-K

一个 fr1_xyz 源地图（SLAM seed2026），70 Bank训练视图、44个预先锁定且与Bank无交集的non-Bank视图。优化顺序seed2026/2027是同地图重复，不是两个独立SLAM地图。LoD训练320×240，正式Bank/non-Bank评价640×480；Full续训640×480。

生产策略是 Cached Bernoulli：`S={i: u_i<p_i}`。当前离线same-K按 `m_i=(logit_keep−logit_drop)/temperature−logit(u_i)` 降序排列，tie按Gaussian行号；非eligible置−∞。必须读取softmax前logit差，不能由已饱和的FP32概率反推logit。生产日志另有 `p.topk(floor(rN))` 诊断，它不是本轮same-K规则。之前跨不同地图的same-K实验还曾将网络输入ratio改为K/N；本轮始终保留原始ratio。

| ratio | rN | floor(rN) | 本轮固定K=当前Cached实际K |
|---|---|---|---|
| 1% | 139.19 | 139 | 138 |
| 5% | 695.95 | 695 | 697 |
| 10% | 1391.90 | 1391 | 1365 |
| 15% | 2087.85 | 2087 | 2021 |

以当前实际Cached K构造margin Top-K后，与Cached集合**逐行精确相同**。因此起点既保留原采样结果，又可在明确的same-K排序语义下进行交换。C最终同时保存原Cached及同一固定K结果，质量主结论仅用same-K。

有效渲染点数使用现有原生 `RasterizeGaussiansCUDAWithImportance` 的 `contribution_count>0`：至少一个像素通过原alpha/transmittance条件并参与累积。单独保存radii可见数和opacity截断proxy，不混称。该统计只在训练后读取，从未用于选点、训练或接受swap。原生LibTorch ABI1与Python ABI0通过独立进程和TorchScript张量fixture隔离；未修改Renderer。

## 2. Selection Oracle结果与可恢复质量

固定全部Master/网络/Feature/资格mask/相机/u；交换点使用同一冻结MLP产生补偿。每ratio16轮，每轮8个移出候选×8个移入候选：各4个来自margin边界池、4个均匀随机。4个Bank视图筛选，最佳2候选在全部70 Bank上确认；仅接受photo均值严格下降的单点交换。始终K不变，最多16次接受。

最初试跑采用“首个失败轮即停”，只覆盖1～3轮；在未读取non-Bank质量的情况下，改为完整16轮以保证预定覆盖，原结果与修改理由均保留。最终每档1024次筛查、6406次真实render，约6.4秒/档；耗时仅是此离线harness观察。

| ratio | K | 接受swap | 不同候选对 | 对数/单状态K(Neligible−K) | non-Bank ΔPSNR | non-Bank Δphoto |
|---|---|---|---|---|---|---|
| 1% | 138 | 8 | 1024 | 0.0548% | +0.0671 | -0.000832 |
| 5% | 697 | 13 | 1016 | 0.0112% | +0.0197 | -0.000291 |
| 10% | 1365 | 13 | 1023 | 0.0061% | +0.0068 | -0.000114 |
| 15% | 2021 | 6 | 1022 | 0.0043% | -0.0020 | -0.000038 |

搜索找到的收益在1%最明显，但仍仅+0.067 dB。5%小幅泛化；15%的PSNR微降而photo/SSIM/LPIPS略改善。**有限搜索没有找到显著选点收益，不等于证明Selector最优或地图不存在更好集合。**不同尝试对数相对单状态全部可交换对数仅约0.004%～0.055%，它不是全部可能集合的覆盖比例；且每轮只确认筛选最佳2个候选。

所有移入/移出Gaussian的source row、Anchor、level、XYZ和接受前后Bank photo在 `selection_swaps.csv`；集合在 `source_sets.npz`、`oracle_sets.npz`。搜索跨全体eligible候选，但均匀与边界抽样不构成完整空间搜索。

| ratio | 最终新增行数 | 新增点L0/L1/L2 | 新增点占用h=.1区域数 | 同Anchor交换次数 | 新增点世界坐标范围 |
|---|---|---|---|---|---|
| 1% | 7 | 1/6/0 | 7 | 0 | X[-0.83,0.69]; Y[-0.39,0.44]; Z[0.94,1.73] |
| 5% | 13 | 7/6/0 | 13 | 0 | X[-1.94,0.91]; Y[-0.45,0.65]; Z[0.83,3.05] |
| 10% | 13 | 4/8/1 | 13 | 0 | X[-0.74,1.48]; Y[-0.97,0.38]; Z[0.95,2.44] |
| 15% | 6 | 3/3/0 | 6 | 0 | X[-2.52,1.28]; Y[-0.56,0.54]; Z[0.80,3.15] |

## 3. Compensation Oracle结果与可恢复质量

14维严格沿用现代码：XYZ3加到world位置；log-scale3加到基础log-scale；quaternion4先与identity相加并normalize，再左乘基础rotation并normalize；opacity-logit1相加后sigmoid；SH DC3相加。没有新增clamp、loss或约束。

每ratio/集合独立建立 `[K,14]` 参数，从当前MLP残差复制初始化；只有delta进入Adam，lr=0.001，固定2000步。两个seed使用同样70 Bank的打乱循环；Master/Selector/Feature/相机/集合不更新，最终step2000评价。

附加等步数控制只训练原补偿头32→32→14，冻结相同共享h；每个ratio/集合/seed各自复制一份头，使用同一集合、schedule、初始化、photo、lr和fresh Adam。该控制也放宽了不同ratio之间的头共享；即便如此仍未追平独立参数。它用于排除“只因多训练2000步”的简单解释，未改生产网络。

| ratio | 独立补偿−当前MLP ΔPSNR | 原补偿头续训−当前MLP | 独立补偿−等步数补偿头 | 独立补偿 Δphoto |
|---|---|---|---|---|
| 1% | +1.1851 | +0.3706 | +0.8144 | -0.016212 |
| 5% | +0.8430 | +0.2395 | +0.6035 | -0.008667 |
| 10% | +0.5607 | +0.0819 | +0.4788 | -0.006405 |
| 15% | +0.6342 | +0.0934 | +0.5408 | -0.006898 |

当前集合的Bank PSNR收益为+1.117/+0.932/+0.677/+0.721 dB；non-Bank为+1.185/+0.843/+0.561/+0.634 dB，SSIM、LPIPS、photo全部同步改善，两个优化seed均成立。收益不是只出现在Bank。

独立补偿超过等步数头续训，支持共享残差函数/冻结共享特征下的拟合限制；但两种参数化的优化条件不同，2000步也未证明收敛到全局解。因此仍不能严格拆分“表达能力”和“优化条件”，更不能由此直接断言加深网络有效。14种属性语义本身足以恢复上述收益。

### 哪些分量最重要

在最终独立补偿上，将一个分量组恢复为初始MLP值，其他分量保持已优化值。下表是恢复该分量导致的non-Bank PSNR损失，即其**条件贡献**；不是单独训练该属性的容量收益，不能相加。

| ratio | XYZ | Scale | Rotation | Opacity | SH DC |
|---|---|---|---|---|---|
| 1% | 2.514 | 0.743 | 1.410 | 0.077 | 0.201 |
| 5% | 1.739 | 0.584 | 1.088 | 0.057 | 0.190 |
| 10% | 1.261 | 0.579 | 0.955 | 0.034 | 0.184 |
| 15% | 1.429 | 0.591 | 0.878 | 0.041 | 0.197 |

XYZ、Rotation的条件影响最大，Scale次之，Opacity与SH DC较小。恢复XYZ时损失可超过总收益，表明各属性有明显协同适配；不能把这个数当作纯XYZ瓶颈的可加上界。

1%当前MLP位移/Anchor边长p90为1.49，独立补偿后为3.93/4.09；5%从0.37变为2.01/1.98。Master虽冻结，补偿后的实际几何可以明显移动。因此本轮收益同时包含对现有几何覆盖的重新分配，不能解释成在严格固定渲染几何上只改善颜色。补偿幅度、区域占用和alpha覆盖全部保存。

| ratio | 当前alpha≥0.5像素覆盖 | 独立补偿后覆盖 |
|---|---|---|
| 1% | 93.87% | 95.16% |
| 5% | 95.90% | 97.07% |
| 10% | 97.80% | 97.92% |
| 15% | 97.71% | 97.90% |

alpha覆盖只是成像平面透射率覆盖，不能证明表面/纹理/深度覆盖充分。1%即使独立补偿后仍比源Full低约4 dB；低ratio稀疏表示仍受限，但本实验不能将剩余差距唯一归因于“缺少空间覆盖”。

## 4. Selection + Compensation联合收益

以下为44 non-Bank视图，涉及训练的组合对两个seed取均值；每档四组合K完全一致。

| ratio | 组合 | PSNR↑ | SSIM↑ | LPIPS↓ | photo↓ | 真正贡献点数均值 |
|---|---|---|---|---|---|---|
| 1% | 当前集合＋当前MLP | 16.1384 | 0.5917 | 0.6911 | 0.1613 | 117.5 |
| 1% | 当前集合＋独立补偿 | 17.3234 | 0.6107 | 0.6687 | 0.1451 | 116.6 |
| 1% | Oracle集合＋当前MLP | 16.2055 | 0.5928 | 0.6875 | 0.1605 | 118.9 |
| 1% | Oracle集合＋独立补偿 | 17.4830 | 0.6130 | 0.6649 | 0.1434 | 117.2 |
| 5% | 当前集合＋当前MLP | 19.2877 | 0.6554 | 0.5268 | 0.1223 | 509.4 |
| 5% | 当前集合＋独立补偿 | 20.1308 | 0.6721 | 0.4997 | 0.1136 | 507.9 |
| 5% | Oracle集合＋当前MLP | 19.3074 | 0.6559 | 0.5216 | 0.1220 | 510.7 |
| 5% | Oracle集合＋独立补偿 | 20.1687 | 0.6731 | 0.4938 | 0.1132 | 508.9 |
| 10% | 当前集合＋当前MLP | 20.0505 | 0.6780 | 0.4797 | 0.1128 | 867.5 |
| 10% | 当前集合＋独立补偿 | 20.6112 | 0.6919 | 0.4485 | 0.1064 | 866.2 |
| 10% | Oracle集合＋当前MLP | 20.0573 | 0.6784 | 0.4783 | 0.1126 | 870.4 |
| 10% | Oracle集合＋独立补偿 | 20.5854 | 0.6917 | 0.4487 | 0.1065 | 870.0 |
| 15% | 当前集合＋当前MLP | 20.1638 | 0.6829 | 0.4656 | 0.1109 | 1100.7 |
| 15% | 当前集合＋独立补偿 | 20.7980 | 0.6984 | 0.4329 | 0.1040 | 1099.8 |
| 15% | Oracle集合＋当前MLP | 20.1618 | 0.6831 | 0.4652 | 0.1109 | 1099.6 |
| 15% | Oracle集合＋独立补偿 | 20.7943 | 0.6987 | 0.4325 | 0.1040 | 1101.0 |

联合组合相对当前集合＋当前MLP，四档PSNR提升约+1.345/+0.881/+0.535/+0.630 dB。独立补偿下，Oracle集合相对当前集合的增益为+0.160/+0.038/−0.026/−0.004 dB：低ratio有部分协同，高ratio没有一致协同。差中差完整保存于 `selection_compensation_interaction.csv`。

源集合每帧实际贡献点数均值为117.5/509.4/867.5/1100.7，对应选择K=138/697/1365/2021。其差额同时包含视野外、遮挡及alpha截断，不能全部归因于低opacity或无效Gaussian。

## 5. Joint / Full-only / Frozen-Master对照

### 实际梯度流向

| loss | Master | 共享编码器/Anchor Feature | 选择头 | 补偿头 |
|---|---|---|---|---|
| Full photo | 直接更新XYZ/Scale/Rotation/Opacity/SH | 无 | 无 | 无 |
| LoD photo | 选中点直接渲染路径＋网络属性输入的间接路径 | 有，选择/补偿两条路径 | 有，selected opacity ST | 有，仅选中点 |
| 预算loss | 经Offset/activatedScale/SH DC/activatedOpacity输入；Rotation/SH rest无此路径 | 有 | 有 | 无 |
| 已有regularization | Full的0.001×mean(opacity)，只到opacity | 无 | 无 | 无 |
| TF/GI/CD等 | 本配置关闭，无额外梯度 | 无 | 无 | 无 |

公式：`L_F=0.8L1+0.2(1−fast_ssim)+0.001 mean(sigmoid(opacity_logit))`；`L_LoD=photo+0.01|Σp−rN|/max(1,rN)`。预算监督不以eligible N重新定义分母。

同源状态、8个固定Bank视图逐档测量；同时保留native Full640/LoD320及Full/LoD均320的梯度，避免只把分辨率差异当作冲突。

| ratio | Full/LoD同320整体cosine中位数 | 负cosine视图占比 |
|---|---|---|
| 1% | 0.173 | 0.000 |
| 5% | 0.416 | 0.000 |
| 10% | 0.529 | 0.000 |
| 15% | 0.544 | 0.000 |

整体32/32个测量为正；部分属性/视图会负向，不能泛化为系统性全局梯度反向。native整体cosine中位数为0.187/0.419/0.515/0.534。分组norm、cosine、budget与regularization在 `joint_gradients.csv`、`gradient_summary.csv`。梯度方向诊断与完整续训结果需要一起解释。

续训后Joint的两个seed合计64个整体测量仅1个为负，出现在1%档。最终Joint四档整体cosine中位数约0.127/0.408/0.481/0.487：1%对齐较弱，但仍不支持“训练后普遍反向”的解释。

### 匹配续训条件及差异

每seed400个同帧循环，固定1%→5%→10%→15%→Full顺序；所有分支Full更新400次。Joint与Frozen-Master各LoD辅助更新1600次。Full-only不训练辅助网络，评价使用冻结源辅助网络，不能把它当作训练量相同的LoD方法。

同一初始权重和fresh零状态Adam。导出未保存原Adam moments或初始化时的camera extent，无法严格恢复在线优化器。离线沿用六组学习率及衰减形式，XYZ的spatial scale明确固定1；六组eps=1e-15，辅助lr=0.001。Full事件的XYZ学习率和虚拟事件时钟完全匹配。Joint Master Adam推进2000次，其余400次；这是LoD更新带来的处理差异，同时耦合了Adam时钟，不能将结果只归因于负cosine。

无SLAM异步、densify/prune、位姿修改、opacity reset；Anchor归属和坐标参考固定。本实验是固定拓扑的确定性离线续训，不能直接代替在线完整质量结论。

C保留现有slot恢复规则，Master opacity改变可以使原停用点恢复；因此eligible mask可能随分支轨迹变化。固定的是population和归属，A/B的资格mask则完全冻结。

| 分支 | Full PSNR | Full SSIM | Full LPIPS | Full photo | 1% PSNR | 5% PSNR | 10% PSNR | 15% PSNR |
|---|---|---|---|---|---|---|---|---|
| 源checkpoint | 21.3272 | 0.7378 | 0.3208 | 0.0923 | 16.1384 | 19.2877 | 20.0505 | 20.1638 |
| Full-only＋冻结辅助 | 21.6411 | 0.7460 | 0.3084 | 0.0891 | 15.5678 | 19.0598 | 19.8272 | 19.9590 |
| Joint | 21.0913 | 0.7368 | 0.3200 | 0.0948 | 16.2885 | 18.8791 | 19.5354 | 19.6135 |
| Full更新Master＋LoD仅辅助 | 21.6408 | 0.7460 | 0.3084 | 0.0891 | 15.6973 | 18.5605 | 19.2540 | 19.4471 |

Joint相对Full-only，Full损失0.550 dB、photo恶化0.005668；两个seed均出现。Frozen-Master的Full与Full-only相差仅−0.000289 dB（CUDA数值差异），证明辅助更新没有实质改变Master的Full轨迹。

但Frozen-Master相对Joint，四档LoD PSNR均值低0.339 dB，photo高0.005417；因此简单切断LoD→Master会保护Full，却损害当前稀疏渲染。Joint相对源checkpoint仅1% PSNR提高约0.150 dB，其余档下降约0.409/0.515/0.550 dB，同时LPIPS改善；不能将其称为四指标全面收益。

完整四指标/seed/Cached实际K/有效贡献点数见CSV。Master各属性变化保存在 `master_changes.csv`；各组实际Adam次数在 `joint_updates.csv` 和最终优化器权重中可核验。

| 优化seed | 1%独立补偿−源MLP | 5%独立补偿−源MLP | 四档独立补偿−等步数头 | Full Joint−Full-only | 四档LoD Joint−Frozen-Master |
|---|---|---|---|---|---|
| 2026 | +1.195 | +0.822 | +0.631 | -0.800 | +0.347 |
| 2027 | +1.175 | +0.864 | +0.588 | -0.299 | +0.332 |

## 6. 三个瓶颈的证据强度与局限

| 假设 | 本轮证据 | 强度/限制 |
|---|---|---|
| Selector排序是主要瓶颈 | 有限swap平均仅+0.023 dB；低ratio略有泛化 | 弱到中；搜索覆盖极小，不能排除大规模组合换点 |
| 补偿共享参数化受限 | 固定当前集合独立14维+0.806 dB；超过等步数原头+0.609 dB；四指标、两seed一致 | 本轮最强；仍仅一个地图、2000步、冻结共享编码，未隔离深度/宽度/特征/优化条件 |
| Joint有Full竞争 | 同Full更新预算下Joint Full低0.550 dB；切断Master可恢复Full但LoD差 | 离线协议下明确；fresh Adam、spatial scale代理、Adam时钟和固定拓扑限制在线外推 |
| 低ratio主要缺空间覆盖 | 独立XYZ/Rotation/Scale适配有收益，1%残余Full gap仍大 | 尚未唯一确定；alpha覆盖不等于几何/纹理覆盖，未做完整覆盖上界搜索 |

本轮不是理论上界：Selection只搜索有限候选；独立补偿只放宽共享约束并优化2000步；头续训未证明收敛；两个seed不是独立地图。分量恢复有协同效应，不能据其数值拼接总收益。源非Bank44帧来自既有锁定集合，结论不覆盖其他scene或全数据集。

正确性与交付：本地独立复算全部19950行逐视图、350组质量摘要、1600行配对统计；核验38份最终权重/Adam步数、source SHA、K、Bank分离及119个文件SHA。原生计数在每个case的Bank和non-Bank首帧与Python渲染比较，image最大误差<5e-6、radii逐项精确一致。

## 7. 下一步优化建议（本轮不实施）

1. **优先补偿私有表示与参数共享。** 保留选择、预算、Master和14维属性语义，单变量比较补偿私有输入/分支容量；XYZ、Rotation优先测量。证据是等步数补偿头仍落后独立参数0.61 dB，而非仅凭网络小就扩容。加深MLP可以作为后续单变量候选，当前没有实测证据保证其胜过补偿私有特征或更好的优化条件。

2. **随后研究Master更新竞争的控制。** 保持同Full更新预算、相同初始Adam状态，分开验证LoD Master更新强度、更新次数/时钟与补偿适配；同时看Full和same-K，不能直接把detach当成已验证改进。

3. **Selector保持原版，补偿稳定后扩大受控搜索。** 现有swap收益小，但覆盖不充分；若继续，预先扩大候选/交换预算，仍只用Bank接受，避免根据测试质量改排序算法。

| 最终问题 | 回答 |
|---|---|
| 当前是否主要受Selector排序限制？ | 未确定；本轮有限搜索未支持它是首要瓶颈 |
| 当前是否主要受Compensation表达限制？ | 共享补偿参数化/可学习性限制证据最强；尚未证明单纯加深网络就能解决 |
| Joint Training是否正在损害Full或LoD？ | 本轮离线Joint损害Full；对LoD存在权衡，冻结Master反而更差；当前在线系统的因果结论仍未确定 |
| 固定地图是否还有明显可恢复PSNR？ | YES：当前集合独立补偿四档约+1.19/+0.84/+0.56/+0.63 dB，non-Bank可恢复 |
| 下一步最值得修改哪个模块？ | 补偿的私有表征/共享参数化；其可恢复收益显著大于当前有限选点收益，且等步数控制仍存在差距 |

证据目录：`tests/anchor_selector/render_quality_bottleneck_20261022/evidence/`；源代码/配置保持不变，实验脚本独立保存。状态：SOURCE AUDIT / A / B / C / EQUAL-STEP CONTROL / NATIVE COUNT / LOCAL VERIFICATION均已完成；生产算法修改与在线多scene验证未实施。
