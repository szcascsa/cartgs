# fr1_xyz Frozen Same-K 实验证据

- 主结果：summary.json、quality_summary.csv、quality_per_view.csv。
- 当前C相对Raw低budget收益复现；F3在1%改善，5/10/15%回退，Combined gate未通过。
- 本地独立核对：local_delivery_validation.json。
- 服务器完整实验：/home/prmi/zwj/hierarchical-fr1-xyz-same-k-20261019。
- 所有数学/renderer/metric沿用上一轮；只新增离线诊断。
- bundle_manifest.json核对原归档96个文件；本README和本地验证为归档后新增记录。
- 大型cameras.pt未打包，保留服务器原件及SHA；camera_prepare_source.py可从相同输入重建。
- 未commit、未push、未改生产路径。
