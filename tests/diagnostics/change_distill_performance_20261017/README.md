# ChangeDistill 性能审计证据

本目录为独立诊断，不修改生产实现。主报告位于仓库根目录
`CHANGE_DISTILL_PERFORMANCE_AUDIT.md`，本目录也保留一份可独立阅读的副本。

- `source/`：本地源码冻结副本，哈希见 `raw/source_manifest.json`。
- `instrumented/`、`instrumentation.diff`：只增加观察 hooks 的副本。
- `benchmark.cpp`：同一真实snapshot重放、完整恢复Student、全新Adam、step断言。
- `build_native.py`：项目LibTorch 2.3.1/ABI1、CUDA11.8、sm86的隔离构建。
- `run_matrix.py`：10次warm重复，epoch/batch/ratio矩阵和并发GPU进程检查。
- `run_kernel_profiles.py`：单epoch Nsight与有限热点NCU；不修改驱动权限。
- `validate_results.py`：Gaussian/normal Adam隔离、实际step数和插桩参数一致性。
- `analyze.py`：CUDA launch关联CPU operator，统计CPU/CUDA跨度及kernel忙碌时间。
- `make_report.py`、`audit_static.md`：中文报告生成与静态审计。
- `raw/`：所有原始CSV/JSON/日志/trace/NSight输出与派生统计。
- `plots/`：基于原始统计的PNG/SVG图。

运行目标路径固定为本次用户指定的服务器工作目录；不能在Windows直接执行CUDA
runner。首先恢复服务器访问，再按报告顺序重放。`ssh_session.py`只从隐藏终端输入
读取密码并保留在进程内存中；不保存凭据，也不是benchmark的一部分。

注意：CUDA Event span包含stream空闲；nested inclusive组件不可相加。原始总性能用
无细粒度插桩的mode0，mode1只用于归因。TF的batch scaling会改变现有诊断audit大小，
训练点集保持不变；报告同时提供只含训练阶段的统计。

NCU硬件counter被`ERR_NVGPUCTRPERM`拒绝。原始错误与首轮不兼容报告名的输出均保留；
不把工具退出码0或估计occupancy当作硬件counter结果。
