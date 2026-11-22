# Phase 0：当前源码路径审计

本地 HEAD：f7e2ce52a0ba63bbfbd1ab2e63fb74ceb0beb016。tracked worktree clean；既有 STREAMING_GI_CURRENT_REPORT.md 和 pycache 均保留。服务器项目不是独立有效Git checkout，git向上找到home的空仓库，HEAD不可用；以实际文件SHA核对，不初始化或修复Git。关键SGI/teacher/mapper源码SHA与现有离线artifact对应的源码一致。

服务器Mapper与本地HEAD存在此前Selector集成留下的差异，已下载原文到source_before保留审计证据。用户随后明确要求“直接将服务器同步为本项目的代码”，因此最终以本地项目为权威同步1831个源码/配置文件，并重新构建；不沿用旧服务器专有改动。数据、模型、结果、依赖和构建缓存保留。当前fast配置使用默认原width32 Selector，两arm一致，不研究Wide或切换架构。source_after是同步指示之前的暂存版本，不是最终实验源码；最终版本以code_sync_manifest和project_code.tar.gz为准。

1. **Raw更新**：`GlobalGIManager::updateBankBatch`维护多视角Bank，增加成员只渲染新view，移除成员减去缓存contribution；sum除以当前view数得到importance。`checkedContribution`把非finite/负值记为unknown；不改变renderer统计定义。
2. **现有Marginal路径**：production不存在。`GlobalGI.mode`只接受disabled/recent/streaming。不得把旧offline结果称为已接入online。
3. **offline代码**：`redundancy-si-ranking-audit-20261019/marginal_si.py`及`covariance-redundancy-audit-20261019`；原h8、radius、原greedy/global lexsort可复用为权威输入与输出。
4. **view similarity**：只在offline `similarities`中，contribution按列归一化后做cosine，再与geo相乘。新的geometry模块不接受contribution vector；历史audit保持原样供复核。
5. **per-view保存**：production `GIViewCache::contribution`和`GlobalGIManager::views_`是Raw SGI必需缓存，不能删除。offline完整矩阵位于冻结`contributions.pt`与`ranking_inputs.npz`。
6. **teacher接口**：Mapper `accumulateLodGradients`把importance、valid、observed和当前maturity传入`GITeacherBuilder::build`；它按有效集合的percentile和严格大于threshold产生binary label，以原elastic maturity加权L1/SmoothL1。不是importance回归，不是Selector probability。
7. **cadence**：每个正常mapping boundary增量维护Bank；现有`refresh_interval=1000`在mapping提交后、同render锁内full refresh。新增Marginal只在同一full refresh发布，Raw policy不变。
8. **失效**：append/clone/split新增row，prune改变row映射；现有callback明确传入append count或exact survivor mask。xyz、scale、loop closure和scale refinement改变geometry。首次实现只在full refresh复制geometry+Raw I并重建grid/cache；两个refresh间冻结已发布score。append新增row无score且无监督，prune同步按同一survivor mask重映射，reset清空发布与固定h。不把row号当跨topology稳定ID，不跨snapshot重用geometry cache。
9. **线程位置**：Bank增量位于`trainMappingIteration`的render锁内、Full/LoD之前。全量refresh在`trainForOneIteration`提交mapping后重新持render锁完成。Marginal同一锁内构建并发布，可能延长锁持有时间，必须测refresh spike和tracking stall，不能从CPU ranker微测声称实时性。

## 冻结集成选择

新config mode为`streaming_marginal_geo`；另加默认关闭的`GlobalGI.marginal_shadow`和`GlobalGI.marginal_diagnostics`。Raw `streaming`保持可运行；Shadow仍用Raw teacher。teacher builder、Selector、TF、GI loss、maturity/protection、Bank和renderer不改。

在线h初始化：第一个既有full refresh且valid/observed候选>=8时，计算activated scale几何均值radius，d50=2×median(radius)，h=8d50，world origin=0；本map lifetime固定h，reset才清除。初始化前明确记录warmup并继续Raw teacher，不silent fallback。每次full refresh同步Raw I与当前xyz/scale；发布之后按snapshot age记录几何陈旧，不把历史teacher误称实时geometry。

现有teacher会在动态maturity/有效集合上重取percentile。为避免截断15%后tail同分影响监督，production发布完整marginal gain；15% lazy-global前缀单独验证与profile，同时报告online完整发布成本。这是安全的exact实现选择，没有用近似tail冒充完整ranking。

在线A/B：当前原Selector结构，两arm相同；fr1_room和fr1_desk、seeds2026/2027。训练所有production超参数沿用当前fast配置，仅GI mode不同。诊断seed入口默认不启用，不修改普通启动语义。固定mapping预算使用已有vanilla预算，评价1%/5%及Full；不重新选checkpoint或按质量重跑。
