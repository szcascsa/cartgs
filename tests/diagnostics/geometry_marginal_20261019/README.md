# Geometry-only Marginal SI：实现与验证归档

本目录保存 2026-10-19 已完成实验的脚本、三份报告和小型复核数据。本次提交将其与 production 实现一并纳入 `main`，不启动新的实验。

## 结论与采用边界

- 离线完整排名、四档 selected rows 和渲染指标与 GeometryOnlyReference exact 一致。
- 原 Python Reference 完整排名中位耗时 1209.342 ms；Optimized Exact 前 15% 为 50.889 ms，完整排名为 77.958 ms。总加速包含 Python 到 C++ 的实现差异，各项优化对照见性能报告。
- 两场景、两个 seed 的在线实验出现低预算质量正向信号。fr1_room 两 seed 平均 PSNR 增益：1% 为 +0.354 dB，5% 为 +0.182 dB。
- 预注册门控为受限的 Online GO；全面 production-ready 尚未通过。fr1_room seed2027 的 Full PSNR 下降 0.5917 dB，ATE 从 0.05103 m 增至 0.07520 m；新模式 full refresh 最大 297.5 ms。Raw 保持默认。
- 固定 normal Mapping 预算，但异步 SLAM 的 online/tail 分配不同，实际 optimizer commits 不完全相同。不得把在线结果描述为严格相同更新次数下的纯 ranking 因果结论。
- 仅验证 fresh-start 连续运行；checkpoint/resume 的固定 h 和 published state 持久化尚未验证。

## 报告

- [代码路径审计](PHASE0_CODE_PATH_AUDIT.md)
- [简化与语义验证](GEOMETRY_ONLY_SIMPLIFICATION_REPORT.md)
- [Reference 复现](GEOMETRY_ONLY_REFERENCE_VALIDATION.md)
- [Exact 优化与性能](GEOMETRY_MARGINAL_PERFORMANCE_REPORT.md)
- [在线质量、机制、ATE 与开销](GEOMETRY_MARGINAL_ONLINE_REPORT.md)

报告按实验完成时的内容原样保存，其中“未 commit、未 push”描述的是实验交付时点，不代表本归档提交后的 Git 状态。`archive_manifest.json` 保存归档文件原始 SHA。

## Production 入口

新增模块位于项目 `include/geometry_marginal_ranker.h`、`include/geometry_marginal_importance.h` 和 `src/geometry_marginal_importance.cpp`。

显式启用新 teacher：

```yaml
GlobalGI.mode: "streaming_marginal_geo"
```

默认配置继续使用 `streaming`。Raw Streaming GI 的多视角 Bank、原始贡献统计和缓存保留；pairwise redundancy 仅使用 xyz 与 activated scalar radius。新模式在既有 full refresh 重建固定 h8 membership 和 marginal ranking。为保持已有动态 percentile teacher 语义，production 发布完整顺序；前 15% 提前终止仅作为独立 exact 优化实验。

`marginal_shadow`、`marginal_diagnostics`、`CARTGS_AUDIT_SEED` 和 `CARTGS_MAPPING_TRACE` 为默认关闭的诊断入口。合成排名测试保存在本目录 `test_ranker.cpp`，topology lifecycle 测试位于项目 `tests/geometry_marginal_test.cpp`，CMake 测试名为 `geometry_marginal_lifecycle`。

## 数据与复核

`online_protocol.json`、`analysis_preregistration.json` 保存预注册协议；`online_quality.csv`、`online_paired_deltas.csv`、`online_training_budget.csv` 和各阶段 profile 保存结果与实际训练预算；`independent_verification.json` 和 `independent_local_checks.json` 保存交付前的核验结果。

脚本是原实验环境的冻结记录，使用 `/home/prmi/zwj` 下已有 checkpoint、相机归档、renderer 和构建依赖；本目录不是自包含数据集。不要直接在本目录执行 `online.py`、`deploy_build.py` 或 `sync_code.py`，它们会调度实验或部署源码。需要复验时应先恢复完整实验包到独立工作目录，核对外部输入 SHA。

大型模型、逐 snapshot 二进制、完整逐 view 数据和运行依赖不进入 Git，原文件保持在：

- 本地：`H:/AIProjects/cartgs优化更细致版/geometry-marginal-online-20261019`
- 服务器：`/home/prmi/zwj/geometry-marginal-online-20261019`
- 完整包：`geometry_marginal_validation_bundle.tar.gz`
- 完整包 SHA256：`87b3e5d30a9fa8fc2b3952d3ac770668227f62ed7886c9878f6eecaf202b902d`

输入、源码同步和大型外部 artifact 的路径/SHA 见 `input_manifest.json`、`code_sync_manifest.json`、`geometry_marginal_manifest.json`、`external_artifacts.json`。归档 manifest 的文件列表仅涵盖原样复制的实验产物；本 README 和 `.gitattributes` 为提交时新增说明。
