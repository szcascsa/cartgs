# 下一轮研究建议：用本轮证据决定是否推进

最终数值结论在主报告文首及 `paired_macro_deltas.csv`，本文件不预先宣布某Feature有效。生产仍保持本轮开始时的 SpatialLocalXYZ。

本轮实际结果：FrozenBaseHead8相对匹配继续训练基线七档PSNR为+0.2799/+0.0509dB，但1%/5%均值仅+0.1749/+0.0186dB，LPIPS为+0.0009/+0.0032（变差）。随机分组接近真实Anchor，Context未稳定超过learned表。因此优先级为：**等总预算的训练耦合与按Gaussian的ratio监督支持 → 独立地图确认 → 动态生命周期 → 在线性能**；暂不直接替换生产，也不扩大Feature维度。

## 1. 第一优先级：区分局部自由度与真实局部信息

本轮的关键控制是：真实Anchor、成员数/训练访问次数完全匹配的随机分组、全局一份Feature，以及由成员统计生成的Context。需要依照这四者的non-Bank差值判断后续路线。

如果真实分组与随机分组相当，不能声称Embedding学习了Anchor局部几何规律。更准确的解释是：小表为共享网络提供了分组可调的隐变量，使当前物理特征相似但需要不同补偿的样本更容易区分；其局部性价值尚未被证明。后续应验证真实分组是否在未选过的成员、成员增删和新区域中更可靠，而不是继续增加8D到32D。

如果Context在低ratio达到接近learned表的收益，应优先检验成员统计是否足够，以及是否需要显式描述保留/删除关系。当前Context聚合完整Anchor的19D统计，不知道这个ratio实际留下哪些成员；这是具体、可检验的信息缺口，不代表所有55D输入都不足。

## 2. 优化耦合优先于维度搜索

普通联合Head曾出现同seed重复收益反转，而冻结原网络的早期Bank结果较一致。FreshAdam控制没有证明单独重置旧状态就能解释全部差异。下一轮若最终non-Bank也支持冻结方案，应以相同Feature和相同每ratio曝光比较：

1. 原共享权重固定，只训练新U/e。
2. 共享权重与U/e联合训练。
3. 先固定共享权重训练U/e，再在预先指定时点联合训练；总真实photo步数保持一致。

第三项只是待验证训练机制，不是本轮已经有效的算法。必须预注册切换时点，不根据non-Bank挑选。保存base/e/U完整Adam状态及各自时钟；不能用零梯度step凑公平。先在固定地图上隔离，再考虑在线Master。

当前生产零输出初始化下不能从step0冻结base：末层零权重会永久切断Feature的梯度。本轮正向信号来自已有非零12k网络，不具备直接冷启动在线替换的证据。最具体的后续实验是：同一总12k预算内，先让原Spatial学习到非零输出，再按预定Bank-only时点冻结共享权重并训练Feature；与全程联合及原Spatial等曝光比较。不要用额外12k预训练隐瞒总预算。

如果冻结方案的non-Bank收益也不稳定，应停止生产接入，保留原SpatialLocalXYZ；不能仅凭Bank收敛或Feature梯度覆盖更高继续扩大网络。

## 3. 新成员泛化是比同轨迹多测ratio更有价值的验证

本轮许多Anchor和Gaussian从未被训练ratio访问。建议下一轮在Bank-only协议预先留出部分真实Anchor成员，让其仅在评价ratio出现，比较真实learned分组、匹配随机分组、Context和原Spatial。必须控制Anchor访问次数、occupancy和K。

这个实验直接检验同Anchor的Feature能否转移给未监督成员，比把单成员Anchor上的身份记忆称为“局部几何表示”更有解释力。不得为留出成员单独优化新的物理残差。

本轮现有七档选中Gaussian都曾在某训练ratio出现，所以尚未检验真正的新成员迁移。但在2.5%下约60%的点低于自身最低选中训练ratio；7.5%约33%，12.5%约20%。这是明确的支持范围缺口。下一轮可先在原网络和同一Anchor Feature上，预注册总曝光不变的ratio采样对照，检查每点的ratio覆盖是否改善；不能把更多采样曝光或按点私有补偿混进该因素。当前禁用外推组Feature的反事实两地图方向相反，不建议直接屏蔽或加入新Loss。

## 4. 真正独立地图确认

fr1/fr2的当前non-Bank在多轮历史结构研究中被使用。最终模型决策需要一个未用于结构选择的独立场景/建图checkpoint，预先固定协议与最终步数，至少两个优化顺序seed。不能无限追加同轨迹视图并称为新的盲测。

只有在独立地图、1%/5%与未见ratio收益一致、且超过重复波动时，才进入小规模在线Joint验证。在线时即使冻结共享网络权重，也必须保留其对Master输入的梯度，不能整个forward包no_grad。在线收益与固定地图收益分别报告。

## 5. 生命周期与CUDA的最低接入门槛

在新增Feature生产化前验证：UID重排、append/prune后Adamalias、clone/split归属、Anchor合并Feature加权后的输出变化、完整checkpoint恢复，以及冷AnchorFeature的训练延迟。现有独立UID helper只验证了参数/状态搬运，不能替代真实闭环质量。

性能方面优先保留当前live55D；只加K行8D Gather与Head前投影，Backward Scatter/Add。先测真实额外Kernel与dense Adam成本，再决定是否融合。Sparse Adam会改变行时钟和moment衰减，应作为单独优化算法实验，不作为无损性能替换。

## 6. 不建议继续的工作

- 不依据短程Bank最高值扩大Feature维度、Head宽度或Attention。
- 不将抽象Anchor Feature的维度直接解释为XYZ/Scale/Rotation补偿表。
- 不把Feature置零后的损害当成新增Feature相对重训基线的收益。
- 不用负梯度cosine、较高PCA秩或较低训练loss单独证明因果瓶颈。
- 不将不同Adam状态、不同Master拓扑或不同选择集合的PSNR差直接归因于融合结构。
