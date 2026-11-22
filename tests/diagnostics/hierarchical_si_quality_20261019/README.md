# SI 审计证据归档

本目录提交复现脚本、统计CSV、JSON、图表及小型验证记录。完整源码快照、冻结输入/输出、runtime库、可执行文件和超过512 KiB的明细CSV继续保留在外部完整包中，不纳入Git。

- 服务器：`/home/prmi/zwj/hierarchical-si-quality-audit-20261019`
- 本地：`H:\AIProjects\cartgs优化更细致版\hierarchical-si-quality-audit-20261019`
- 完整包：`hierarchical_si_quality_evidence.tar.gz`

`git_archive_manifest.json`列出本次提交的证据及未提交完整产物的SHA；原artifact manifest仍描述完整包，不能将其视为Git子集的文件清单。复现脚本应在恢复完整包后的目录执行，部分脚本含原服务器路径。

报告和验证JSON保留实验时的基线HEAD与未commit状态，属于历史溯源；本次Git归档不重写它们。质量结论以最终 `mechanism_cases.json` 和项目根质量报告为准，早期判读记录仅供审计。
