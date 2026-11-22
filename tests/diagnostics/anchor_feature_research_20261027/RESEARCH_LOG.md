# Anchor Feature 研究决策记录

研究起点：2026-10-27 08:01:54 UTC。生产基线为 `c49c0222d34c0104fadc7fa4d4e11473fe152b2c` 的 SpatialLocalXYZ；本目录之外不改算法或配置。

1. **真实边界核对**：Variable Anchor，55D Compensation 输入、连续 log-ratio FiLM、四 Head、Anchor 单位 XYZ 输出。Selector 是独立 11D 网络。本轮冻结历史地图与选点集合，以隔离补偿能力；生产在线固定统计的建立时点与历史冻结地图不同，报告必须区分。
2. **第一轮假设**：局部抽象 Feature 可能帮助已有物理编码。8D Early/Mid/AnchorFiLM/Head、Context 和 Hybrid 从同一个 12k checkpoint 零函数增量初始化，各继续 4k；只读 Bank 留出。Head 与 AnchorFiLM 约 +0.20 dB，且额外投影参数同为 2048，因此进入两地图、两顺序 seed、各 12k 的正式匹配对照。
3. **反例改变研究方向**：fr1 的 Head8 两 seed 约 +0.15 dB，但 fr2 基本无收益。不能从一个地图推广为有效架构；不扩大维度。
4. **访问覆盖发现**：fr1 只有 328/2066 个 Anchor 被训练 ratio 访问，110 个是单成员。加入成员数和四 ratio 访问次数都完全匹配的随机分组控制；另加 GlobalHead8 和 MultiOnlyHead8，区分真实局部关系、冗余参数化与单成员身份记忆。
5. **优化机制新发现**：fr1 首个 GlobalHead8 已恢复普通 Head8 的大部分 Bank 收益；随机成员分组与真实分组接近。增加 FreshAdamBaseline / FreshAdamHead8 控制，单独诊断历史 Adam 状态，不能将此差异算成 Feature 架构收益。
6. **冻结主网络的新信号**：FrozenBaseHead8 两 seed 的 Bank 七档约 18.14/18.16，高于普通联合 Head。起点 Bank 为 17.8723，因而不能只用“继续训练 Baseline 变差”解释全部收益。增加冻结网络的 Global/Fake 分组控制及 fr2 复验；所有候选均在读取本轮 non-Bank 指标前确定。
7. **从零与重复验证**：原网络及 Head8 从零各训 12k，两地图、两顺序 seed；另保留 fr1 同 seed 重跑，区分训练顺序效应和重复波动。正式评价始终使用预定终点，不选择最好 checkpoint。
8. **评价锁定**：`formal_plan_*.json` 在本轮 non-Bank 评价前锁定。后续置零、访问匹配的 Feature 打乱、occupancy 分组和属性分支禁用，只测最终模型的反事实依赖；不据此改变本轮训练配置。

各次原始 Bank 数字、帧序 SHA、实际训练步数、Adam 时钟与完整权重保存在各运行目录。后续汇总必须保留负结果、每个 seed 和各 ratio，不把 Bank 筛选收益等同于独立视图泛化。

9. **Context 的公平复查**：原 Context 只做了短程 FiLM 筛选，而冻结主网络出现收益，因此在相同冻结 C、相同 Head 融合和 12k 预算下增加 ContextHead8 两地图两 seed，区分信息不足与早先融合/优化混淆。此决定仍在本轮 non-Bank 评价前作出。

10. **重复波动反例**：fr1 相同 seed 普通 Head8 配对 Bank 增益从首次约 +0.15 dB 反转为约 -0.04 dB。该发现发生在 non-Bank 评价前；保留两次，不筛选重跑，增加冻结网络及 Global 控制同 seed 重复。正式候选依据两地图两 seed 的 Bank 低 ratio 均值锁定，全部候选仍完整评价。

11. **锁定后的评价与归因**：57次训练共628000步完成，正式逐视图指标40586条。FrozenBaseHead8的PSNR提高但LPIPS未同步改善；fr2除2.5%外六档增益约0.012dB。后验ratio支持统计发现2.5%约60%成员低于自身最低选中训练ratio；增加固定K、不训练的Feature条件消融，明确这是归因而非新模型选择。未追加网络或训练。
12. **提前收尾原因**：已完成两地图两seed、完整12k、从零、同seed重复、Global/Fake/Context和Adam机制控制。没有稳定低ratio、多指标证据支持进入生产；剩余更有价值的问题是等总预算的阶段训练与独立场景，保留为下一轮预注册实验，不为了使用满3小时而扩大维度或选取性重跑。
