# SGI 实测性能报告

所有数值来自输入日志；未记录项不作推算。

测试条件：Replica room0 前 400 帧，三模式各 2100 次正常 Mapping，Selector 启用，OMP_NUM_THREADS=1，TF.start_iter=5000（本次未进入 TF 训练）。

性能限制：Recent/Streaming 期间服务器存在用户运行的 TUM RGB-D 实验，且在线 SLAM 调度导致各运行的 KF/点数不同。下表仅记录该次运行，不能用于公平的速度或显存增减结论。没有重启或中断用户实验。

| GI 模式 | Mapping 周期数 | 平均周期 ms | iterations/s |
|---|---:|---:|---:|
| Disabled | 2000 | 11.0698 | 90.3363 |
| Recent | 2000 | 17.7873 | 56.2199 |
| Streaming | 2000 | 19.7263 | 50.6939 |

三组速度采用相同的 `[Mapping PERF]` 周期统计；不足 1000 次时该统计可能为空。
仅可比较相同数据、硬件、Selector/TF/Guard/训练参数及停止条件的运行。

## 拓扑同步与全量刷新

| 指标 | 实测值 |
|---|---:|
| 拓扑同步次数（日志事件） | 30.0000 |
| 拓扑同步累计 ms | 11.2707 |
| 拓扑同步平均 ms | 0.3757 |
| 拓扑同步 P95 ms | 0.6104 |
| 拓扑同步最大 ms | 1.9599 |
| 完整刷新次数（日志事件） | 2.0000 |
| 完整刷新累计 ms | 47.9766 |
| 完整刷新平均 ms | 23.9883 |
| 完整刷新最大 ms | 25.4762 |

## 累计指标与有效性

| 日志指标 | 最后记录值 |
|---|---:|
| `global_gi_total_kfs` | 26 |
| `global_gi_target_views` | 8 |
| `global_gi_coverage_ratio` | 0.8687 |
| `global_gi_incremental_total_ms` | 178.4926 |
| `global_gi_bank_total_ms` | 10475.7341 |
| `global_gi_topology_cumulative_ms` | 11.2707 |
| `global_gi_full_refresh_total_ms` | 47.9766 |
| `global_gi_overhead_fraction` | 0.2611 |
| `global_gi_cache_memory_mb` | 0.7001 |
| `global_gi_cache_peak_memory_mb` | 1.4002 |
| `global_gi_valid_gaussians` | 17905 |
| `global_gi_invalid_gaussians` | 0 |
| `global_gi_valid_mature_gaussians` | 15559 |
| `global_gi_valid_mature_fraction` | 1.0000 |
| `global_gi_guidance_enabled` | 1 |

缓存内存为实际 Tensor 的元素字节数；峰值为缓存及已跟踪替换临时 Tensor 的逻辑占用。
它们不等于 CUDA allocator reserved 显存，也不包含 renderer 临时缓冲。
CUDA 事件时间单独记录；操作 wall time 等待 GI CUDA 事件完成后取得。
GI 额外耗时比例采用 Bank、增量、拓扑、刷新累计操作耗时除以 Mapping wall time。
日志写入和未单独计时的统计开销包含在 Mapping 时间中。
该日志无法单独证明 Tracking 是否掉帧，应同时检查 Tracking 日志或轨迹时间。

## CUDA 功能验证

第 1000/2000 次 Mapping 分别用全部 8 个 Bank 视角刷新，耗时 25.4762/22.5004 ms；刷新后 invalid=0，valid Mature 比例均为 1。累计 Append/Clone/Split/Prune 为 18/3/3/6，共 30 次。

缓存最终 0.7001 MiB，逻辑峰值 1.4002 MiB；外部每 2 秒采样的 Streaming 进程显存最大 1022 MiB，可能遗漏短暂峰值。
