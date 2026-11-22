"""从三组 CaRtGS 日志生成 SGI 成本报告及数量/显存时间序列；不启动训练。"""
import argparse
import csv
import math
import re
import statistics
from pathlib import Path

KEY_VALUE = re.compile(r"([a-zA-Z_]+)=([^\s]+)")
MAPPING = re.compile(r"cycles=(\d+).*?wall_ms\(avg/p95/max\)=([\d.eE+-]+)/")


def read_log(path):
    rows, topology, full, blocks = [], [], [], []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = MAPPING.search(line) if "[Mapping PERF]" in line else None
        if match:
            blocks.append((int(match[1]), float(match[2])))
        if "[GlobalGI]" not in line:
            continue
        row = dict(KEY_VALUE.findall(line))
        rows.append(row)
        if row.get("event") == "topology":
            topology.append(float(row["global_gi_topology_total_ms"]))
        if row.get("event") == "full_refresh":
            full.append(float(row["global_gi_full_refresh_ms"]))
    return rows, topology, full, blocks


def nearest_p95(values):
    return sorted(values)[math.ceil(0.95 * len(values)) - 1] if values else None


def number(value):
    return "未记录" if value is None else f"{value:.4f}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disabled-log", type=Path)
    parser.add_argument("--recent-log", type=Path)
    parser.add_argument("--streaming-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lines = ["# SGI 实测性能报告", "", "所有数值来自输入日志；未记录项不作推算。", "",
             "| GI 模式 | Mapping 周期数 | 平均周期 ms | iterations/s |",
             "|---|---:|---:|---:|"]
    streaming = None
    for name, path in (("Disabled", args.disabled_log), ("Recent", args.recent_log),
                       ("Streaming", args.streaming_log)):
        if path is None:
            lines.append(f"| {name} | 未提供日志 | 未记录 | 未记录 |")
            continue
        data = read_log(path)
        count = sum(n for n, _ in data[3])
        mean = sum(n * ms for n, ms in data[3]) / count if count else None
        fps = 1000 / mean if mean and mean > 0 else None
        lines.append(f"| {name} | {count} | {number(mean)} | {number(fps)} |")
        if name == "Streaming":
            streaming = data
    rows, topology, full, _ = streaming
    if not rows:
        parser.error("Streaming 日志没有 [GlobalGI] 数据，无法生成实际 SGI 成本报告")
    latest = {}
    for row in rows:
        latest.update(row)
    lines += ["", "三组速度采用相同的 `[Mapping PERF]` 周期统计；不足 1000 次时该统计可能为空。",
              "仅可比较相同数据、硬件、Selector/TF/Guard/训练参数及停止条件的运行。", "",
              "## 拓扑同步与全量刷新", "",
              "| 指标 | 实测值 |", "|---|---:|"]
    for label, value in (("拓扑同步次数（日志事件）", len(topology)),
                         ("拓扑同步累计 ms", sum(topology)),
                         ("拓扑同步平均 ms", statistics.mean(topology) if topology else None),
                         ("拓扑同步 P95 ms", nearest_p95(topology)),
                         ("拓扑同步最大 ms", max(topology) if topology else None),
                         ("完整刷新次数（日志事件）", len(full)),
                         ("完整刷新累计 ms", sum(full)),
                         ("完整刷新平均 ms", statistics.mean(full) if full else None),
                         ("完整刷新最大 ms", max(full) if full else None)):
        lines.append(f"| {label} | {number(value)} |")
    lines += ["", "## 累计指标与有效性", "", "| 日志指标 | 最后记录值 |", "|---|---:|"]
    for key in ("global_gi_total_kfs", "global_gi_target_views", "global_gi_coverage_ratio",
                "global_gi_incremental_total_ms", "global_gi_bank_total_ms",
                "global_gi_topology_cumulative_ms", "global_gi_full_refresh_total_ms",
                "global_gi_overhead_fraction", "global_gi_cache_memory_mb",
                "global_gi_cache_peak_memory_mb", "global_gi_valid_gaussians",
                "global_gi_invalid_gaussians", "global_gi_valid_mature_gaussians",
                "global_gi_valid_mature_fraction", "global_gi_guidance_enabled"):
        lines.append(f"| `{key}` | {latest.get(key, '未记录')} |")
    lines += ["", "缓存内存为实际 Tensor 的元素字节数；峰值为缓存及已跟踪替换临时 Tensor 的逻辑占用。",
              "它们不等于 CUDA allocator reserved 显存，也不包含 renderer 临时缓冲。",
              "CUDA 事件时间单独记录；操作 wall time 等待 GI CUDA 事件完成后取得。",
              "GI 额外耗时比例采用 Bank、增量、拓扑、刷新累计操作耗时除以 Mapping wall time。",
              "日志写入和未单独计时的统计开销包含在 Mapping 时间中。",
              "该日志无法单独证明 Tracking 是否掉帧，应同时检查 Tracking 日志或轨迹时间。", ""]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"报告：{args.output.resolve()}\n时间序列：{csv_path.resolve()}")


if __name__ == "__main__":
    main()
