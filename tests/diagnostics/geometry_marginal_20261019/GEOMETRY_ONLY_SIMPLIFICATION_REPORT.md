# Geometry-only Marginal SI简化报告

结论：删除pairwise view项后，准确复现旧IsoOnly；Optimized Exact全部质量与Reference逐值相同。此结论属于offline，不替代online结论。

| ratio | PSNR | SSIM | LPIPS | photo |
|---|---|---|---|---|
| 1% | 13.882451 | 0.562086 | 0.736327 | 0.203461 |
| 5% | 16.980360 | 0.618974 | 0.630773 | 0.150699 |
| 10% | 17.802212 | 0.637135 | 0.576634 | 0.138246 |
| 15% | 18.225545 | 0.648584 | 0.542243 | 0.131947 |

相对Raw与旧combined的完整对照：

| ratio | Raw SI | Geo×View | Geo-only | Geo-only−Raw | Geo-only−combined |
|---|---|---|---|---|---|
| 1% | 12.9883 | 13.8381 | 13.8825 | +0.8942 | +0.0444 |
| 5% | 16.6531 | 17.0118 | 16.9804 | +0.3272 | -0.0315 |
| 10% | 17.6804 | 17.8309 | 17.8022 | +0.1219 | -0.0287 |
| 15% | 18.0547 | 18.2583 | 18.2255 | +0.1708 | -0.0328 |

删除view后保持了主要offline same-K收益，并非宣称每档都胜过combined。

完整50330点ranking、四档selected set与旧GeometryOnly8 exact相等；Reference和Optimized各644个view-ratio的photo/PSNR/SSIM/LPIPS最大差均为0。maturity=1、原h8、固定Master/TF/Bank/GT，contribution未重算。

此前production没有pairwise view redundancy可删。新独立模块API只接受xyz、scalar radius、Raw I和snapshot row ID，完全不接收contribution vector、不计算cosine、不保存view similarity cache。旧audit历史文件保持原样；Raw SGI的GIViewCache、多视角Bank、均值贡献、incremental/full refresh和renderer统计全部保留。只新增可选mode `streaming_marginal_geo`；默认fast仍为Raw streaming，shadow/diagnostics默认false。

radius=cbrt(sx sy sz)，kappa=exp(−d²/(2 sigma²+1e−12))、diag=1，Fixed h8、无neighbor、无adaptive、无covariance。逐次条件marginal coverage不变，不是静态density penalty。生产实现见include/geometry_marginal_ranker.h和geometry_marginal_importance.h/.cpp。

已有teacher按动态valid/observed/maturity集合取percentile。为保留其接口与权重，online发布完整顺序，并编码为unique FP32 order score；实际marginal gain完整参与ranking和exact tests。前15%提前停止的优化独立测试；production不使用虚构tail score。append新row valid=false，prune按exact survivor mask同步映射，reset清空并重建；不使用row index作为跨topology稳定ID。首次full refresh发布之前明确使用Raw warmup。

用户追加要求后，服务器已按本地项目同步1831个源码/配置文件；没有删服务器数据、模型、结果、依赖。code_sync_result与source before/after SHA保存。原库在runtime_before保留，offline复渲染链接原库。所有production修改未commit、未push。
