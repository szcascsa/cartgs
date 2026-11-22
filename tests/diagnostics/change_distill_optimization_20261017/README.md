# ChangeDistill 优化验证

主报告：仓库根目录 `CHANGE_DISTILL_OPTIMIZATION_REPORT.md`。
原审计 `../change_distill_performance_20261017/` 保持原样。

- `baseline/`：与原审计 manifest 完全一致的初始工作树，不等于 Git HEAD。
- `stage_a/`：被拒绝的逐点 BMM transport 实验，保留失败证据。
- `stage_a_grouped/`：保留原算式/组内顺序、改用 CPU group ID 表的 transport。
- `stage_b/`：在上一阶段上增加有 coverage 证明的 Teacher cache 直接 gather。
- `stage_c/`：再增加 batch 内 Selector 属性 embedding 复用；保留逐 ratio backward。
- 各阶段 `source/`、`raw/source_manifest.json`：对应源码与 SHA-256。
- `baseline/working_tree.diff`、`staged.diff`、`status.txt`：用户初始修改。
- `production_optimization.diff`：仅本任务相对初始工作树的生产源码修改。
- `final_working_tree.diff`：包含用户原有修改的完整最终 Git diff。
- `timings.csv/json`、`components.csv/json`：native 与插桩统计分开。
- `nsys_summary.json`：kernel 数、GPU kernel 总时长和 tiny kernel 计数。
- `profiling_outputs.tar.gz`：所有新 raw 结果、PT 校验数据、CSV/JSON、Nsight、日志、源码和构建记录。

服务器目录：`/home/prmi/zwj/mycartgs/results/change_distill_optimization/20261017`。
不覆盖服务器正式 build/lib，不启动 SLAM，不保存 SSH 凭据。

复现（服务器，Python=`/home/prmi/anaconda3/envs/zwj/bin/python`）：

```bash
python stage_c/build_native.py
python build_verify.py stage_c
python run_cases.py stage_c:selector:new_selector stage_c:tf:new_tf stage_c:both:new_both
python compare.py baseline stage_c selector
python build_profile.py stage_c
python run_cases.py stage_c:selector:new_profile:1:1
python summarize.py
```

`run_cases.py` 拒绝覆盖已有案例；每次 run 都恢复模型、新建 Adam/cache，
并检查 Gaussian、常规 optimizer 和 step 数；native 与插桩使用独立 executable。
GPU 进程监控在 `monitor.json`；正反顺序对照由 `run_interleaved.py` 定义。

若 CUDA 源码改变，必须删除**该隔离阶段**对应旧 `.cu.o` 后重建；
不得从 baseline 复制已改变源码的 object。复用记录见 `cuda_objects.json`。
native 计时包含完整 coordinator finalize、pre/post/audit evaluation；
CUDA Event span 包含 stream 空闲，不能当作纯 kernel busy time。
组件 timer 嵌套且 inclusive，禁止直接相加。
