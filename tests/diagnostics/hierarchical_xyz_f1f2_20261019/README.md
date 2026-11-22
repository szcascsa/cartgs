# fr1_xyz / fr2_xyz F1/F2 机制审计

- 结论：两scene的F1、F2都未通过完整支持gate；保留当前C作为对照，不改生产实现。
- 顶层summary.json按scene保存六arm完整指标和机制结论。
- quality_summary.csv / quality_per_view.csv为六arm汇总和逐视角数据。
- 各scene目录保存完整ranking、tail/unlock trace、统一重复度、空间coverage、paired统计和验证。
- 本轮新F1/F2共744次render，baseline复现1488次，总2232次。
- 本地独立验证：local_delivery_validation.json。
- bundle_manifest.json核对原归档210个文件；本README和本地验证为归档后新增记录。
- 服务器完整目录：/home/prmi/zwj/hierarchical-xyz-f1f2-mechanism-20261019。
- 未commit、未push、未更新任何production代码。
