# 实现前源码审计

审计基线：`40751d07e194aaaa997d06bdea14d90f813b1222`，分支 `redundancy-aware-si`。tracked clean；既有未跟踪报告和 pycache 完整记录于 `audit_before.json`，不纳入本轮修改。

- `cuda_rasterizer/forward.cu:419` 在 alpha/T 的 production 筛选后累加 `con_o.w*T`，不是文档中简单的 alpha*T。本轮不改 kernel、筛选或归一化。
- `src/gaussian_renderer.cpp` 返回 rasterizer 的 `frame_importance`；`GaussianMapper::renderGlobalGIContribution` 以完整 Master、Inference 模式请求原始统计，不调用 TF 来重新定义 GI。
- `src/global_gi.cpp` 的 `updateBankBatch` 保留/增删 `GIViewCache::contribution`；`aggregate` 或 `refreshIfDue` 把 Bank contribution 求和为 `sum_`，均值为 `importance_`。`valid_` 表示缓存有效，`observedMask()` 为原 importance>0。
- `GaussianMapper::updateGlobalGIBank` 在 Mapping boundary 读取 ORB association 快照；既有 full refresh 每1000次 Mapping 全 Bank 重渲染。本轮不改 Bank、cache 或 cadence。
- `GaussianMapper::accumulateLodGradients` 中的 `gi_teacher_builder_.build` 是 teacher score 入口；`src/gi_teacher.cpp` 先取 valid&observed，再按 score percentile 生成 label。本轮只替换输入 score；用户追加要求新路径 effective maturity=1，不使用旧成熟度/entry 权重。
- `GaussianModel::getXYZ()` 为当前 xyz；`getScalingActivation()` 为 exp(log-scale)。`densificationPostfix` 按 old+new 顺序 append，clone/split 走 append callback；split append 后再 prune parent；`prunePoints` 用精确 survivor mask 同时压缩全部状态；reset 替换 map。row 只作为当前 snapshot 的确定性 tie-break，不作为跨 topology 身份。
- 当前没有统一覆盖所有 geometry/scale revision 的可靠 hierarchy generation token。因此第一版不持久缓存：每次开启路径读取 teacher 时用当前 snapshot 重建。append/prune/reorder/参数原位更新或替换后，不存在可复用的旧树。关闭时不读取 geometry，不分配 hierarchy，也不改原 score/valid/observed/Selector 路径。
- 配置通过 `loadSelectorConfig`/`saveSelectorConfig` 中平铺的 dotted keys 读写。新增 enable 和 diagnostics 两个默认关闭 bool；2048/3/0.8/L3 为代码常量，不提供 sweep 参数。enable 要求 Raw `streaming`，与旧 marginal mode/shadow 互斥。

审计完成后才开始新增实现。完整源码 SHA 见 `audit_before.json`。

用户随后将初始 grid 从总数256改为总数2048，并明确不是最短边划分2048格；2048版本为最终验收对象。已完成的256版本结果保留为变更前工程记录，不用于挑参数或选择最终质量。
