# D50归一化Hierarchy审计

- 唯一新增实验参数：lambda=32；L1/L2/L3=32/16/8*d50。
- 结论：SCALE DRIFT HYPOTHESIS NOT SUPPORTED；KEEP TARGET-2048。
- Root尺度漂移确实存在，但相对Progressive改善伴随C baseline退化，并失去fr1_xyz 1%旧优势。
- summary.json和quality_summary.csv为三scene六arm完整结果。
- quality_deltas.csv、paired_statistics.csv、各scene结构/composition/unlock/redundancy/overlap数据完整保留。
- local_delivery_validation.json为Windows独立核对。
- 共6096次真实render：baseline4064次，新C32/P32为2032次；无F1/F3新render。
- bundle_manifest.json核对281个原归档文件；本README和本地验证为归档后新增记录。
- source/为隔离实验backend；production源码未改。create_backend.py为一次性生成脚本，快照已生成；直接build.py即可构建，勿对已补丁快照重复运行生成脚本。
- 大型Master/TF/camera/GT保留SHA验证的服务器原路径。
- 服务器完整目录：/home/prmi/zwj/d50-normalized-hierarchy-20261019。
- 未stage、未commit、未push。
