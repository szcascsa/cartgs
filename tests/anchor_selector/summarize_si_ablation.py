"""汇总预注册对照；保留两个 seed、逐视图及实际训练/点数差异。"""
import csv
import json
from pathlib import Path
import re
import sys
import numpy as np

root = Path(sys.argv[1]).resolve()
output = root/"validation"
cases = json.loads((output/"quality_summary.json").read_text())
plan = json.loads((output/"evaluation_plan.json").read_text())
frames = json.loads((output/"common_frames.json").read_text())["gt_indices"]
assert len(cases) == 36 and len(frames) == plan["frames"]
quality, per_view, paired, runtime = [], [], [], []
for case in cases:
    directory = Path(case["output"])
    assert json.loads((directory/"evaluation_frames.json").read_text())["gt_indices"] == frames
    stats = case["stats"]
    assert stats["render_gaussians"] == stats["active_gaussians"]
    row = dict(arm=case["arm"], seed=case["seed"], policy=case["policy"], level=case["level"],
               ratio=case["ratio"], population=stats["total_gaussians"], active=stats["active_gaussians"],
               retained_fraction=stats["active_gaussians"]/stats["total_gaussians"], views=len(frames), **case["metrics"])
    if case["policy"] == "same_k":
        assert row["active"] == plan["common_k"][str(case["ratio"])]
    quality.append(row)
    values = {metric: np.atleast_1d(np.loadtxt(directory/(metric+".txt"))) for metric in ("psnr", "ssim", "lpips")}
    assert all(len(v) == len(frames) and np.isfinite(v).all() for v in values.values())
    for i, frame in enumerate(frames):
        per_view.append(dict(arm=case["arm"], seed=case["seed"], policy=case["policy"], level=case["level"],
                             gt_index=frame, **{metric: float(value[i]) for metric, value in values.items()}))
for seed in (2026, 2027):
    for policy in ("cached_bernoulli", "same_k"):
        for level in (("Full",) if policy == "cached_bernoulli" else ())+("LOD_1", "LOD_5", "LOD_10", "LOD_15"):
            for metric, sign in (("psnr", 1), ("ssim", 1), ("lpips", -1)):
                values = {arm: np.array([r[metric] for r in per_view if r["arm"] == arm and r["seed"] == seed
                                        and r["policy"] == policy and r["level"] == level]) for arm in ("si", "no_si")}
                delta = values["si"]-values["no_si"]
                paired.append(dict(seed=seed, policy=policy, level=level, metric=metric, views=len(delta),
                    mean=float(delta.mean()), median=float(np.median(delta)), p10=float(np.percentile(delta, 10)),
                    p90=float(np.percentile(delta, 90)), improved_view_fraction=float((sign*delta > 0).mean())))
for job in json.loads((output/"training_jobs.json").read_text()):
    folder = Path(job["output"])
    log = (folder/"training.log").read_text()
    summary = [line for line in log.splitlines() if "[UltrafastRotation SUMMARY] event=training_end" in line][-1]
    counts = dict(re.findall(r"(\w+)=([0-9.e+-]+)", summary))
    assert int(counts["ultrafast_total_normal_steps"]) == 10000 and int(counts["balanced"]) == 1
    gi_counts = re.findall(r"gi_supervised_count=([0-9.e+-]+)", log)
    gi_weight = re.findall(r"gi_effective_weight_sum=([0-9.e+-]+)", log)
    supervised = sum(map(float, gi_counts[-4:]))
    weight = sum(map(float, gi_weight[-4:]))
    assert (supervised > 0 and weight > 0) if job["arm"] == "si" else supervised == 0
    row = dict(arm=job["arm"], seed=job["seed"], wall_seconds=job["wall_seconds"],
               gi_supervised_row_steps=supervised, gi_effective_weight_sum=weight,
               **{k: int(counts[k]) for k in ("ultrafast_total_normal_steps", "full_render_count", "lod_render_count",
                     "online_end_mapping_iteration", "tail_training_steps", "selector_online_lod_update_count",
                     "selector_tail_lod_update_count", "tf_online_lod_update_count", "tf_tail_lod_update_count")})
    memory = (folder/"GpuPeakUsageMB.txt").read_text()
    row.update({"peak_"+name.lower()+"_mb": float(value) for name, value in re.findall(r"Peak (\w+) \(MB\): ([\d.]+)", memory)})
    runtime.append(row)
for filename, rows in (("quality_summary.csv", quality), ("quality_per_view.csv", per_view),
                       ("paired_statistics.csv", paired), ("runtime.csv", runtime)):
    with (output/filename).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
outcomes = {}
for policy in ("cached_bernoulli", "same_k"):
    outcomes[policy] = {}
    for metric, sign in (("psnr", 1), ("ssim", 1), ("lpips", -1)):
        records = [r for r in paired if r["policy"] == policy and r["metric"] == metric and r["level"] != "Full"]
        outcomes[policy][metric] = dict(improved_cases=sum(sign*r["mean"] > 0 for r in records), cases=len(records),
            mean_delta=float(np.mean([r["mean"] for r in records])),
            mean_improved_view_fraction=float(np.mean([r["improved_view_fraction"] for r in records])))
report = ["# Anchor Selector 加入 Streaming SI 对照", "",
    f"场景 fr1_xyz；两个配对 seed（2026/2027）；每次 10,000 次 Mapping；全部 {len(frames)} 个共同 GT 图像帧。没有选择最好 seed/checkpoint，没有调整 SI 权重。", "",
    "## 方法和边界", "",
    "- Anchor 网络、输入（feature16 + offset3 + log ratio）、h=0.1、Cached Bernoulli、ratio loss、photo loss、TF 和 ChangeDistill 均不变，成熟度固定为 1。",
    "- 唯一算法差异为启用现有 Streaming SI Bank 和原 GITeacher loss，权重 1。SI 不作为网络输入或 score 基底；推理只运行 Anchor 网络。",
    "- 默认配置保持无 SI。启用方式为 GlobalGI.mode=streaming 且 Selector.gi_loss_weight=1。",
    "- 使用已有 fixed Mapping 总预算，关闭 forced-tail20，保留现有在线四 LoD/Full 轮转和 post-SLAM Full+单 LoD。异步 SLAM 导致在线/尾部比例、Full/LoD 次数、地图规模及估计姿态仍不完全相同。",
    "- 主结果为生产 Cached Bernoulli；same-K 仅使用最终概率排序进行点数控制诊断，不改变训练或生产导出策略。",
    "- 相同的是 GT 图像索引，各运行使用自己的估计相机外参及 TF。不是冻结地图/固定外参因果消融。", "",
    "## 生产 Cached Bernoulli 质量", "",
    "| seed | 档位 | 点数 无SI / 有SI | PSNR 无SI / 有SI | ΔPSNR | ΔSSIM | ΔLPIPS |", "|---|---|---:|---:|---:|---:|---:|"]
def delta_table(policy):
    result = []
    for seed in (2026, 2027):
        for level in (("Full",) if policy == "cached_bernoulli" else ())+("LOD_1", "LOD_5", "LOD_10", "LOD_15"):
            a, b = [next(r for r in quality if r["arm"] == arm and r["seed"] == seed and r["policy"] == policy and r["level"] == level) for arm in ("no_si", "si")]
            result.append(f'| {seed} | {level} | {a["active"]:,} / {b["active"]:,} | {a["psnr"]:.4f} / {b["psnr"]:.4f} | {b["psnr"]-a["psnr"]:+.4f} | {b["ssim"]-a["ssim"]:+.5f} | {b["lpips"]-a["lpips"]:+.5f} |')
    return result
report += delta_table("cached_bernoulli")
report += ["", "Δ=有SI−无SI；PSNR/SSIM 正数较好，LPIPS 负数较好。", "", "## 相同 K 概率排序诊断", "",
           "K 在质量评价前由四个 Master population 的最小值决定，floor(min_N×ratio)，source-row ascending 打破 tie。", "",
           "| seed | 档位 | 点数 无SI / 有SI | PSNR 无SI / 有SI | ΔPSNR | ΔSSIM | ΔLPIPS |", "|---|---|---:|---:|---:|---:|---:|"]
report += delta_table("same_k")
report += ["", "## 实际训练量与开销", "", "| seed | 方法 | 总Mapping | Full renders | LoD renders | 在线Mapping | 尾部Mapping | wall(s) | peak allocated(MiB) |", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
for r in runtime:
    report.append(f'| {r["seed"]} | {r["arm"]} | 10000 | {r["full_render_count"]} | {r["lod_render_count"]} | {r["online_end_mapping_iteration"]} | {r["tail_training_steps"]} | {r["wall_seconds"]:.2f} | {r["peak_allocated_mb"]:.2f} |')
report += ["", "运行时间包含在线 SLAM、Mapping 和保存，不能称 Selector latency benchmark。逐视图 mean/median/p10/p90/improved fraction 见 paired_statistics.csv。", "",
           "## 验证与产物", "", "源码、配置、build/gates、四个最终 checkpoint 和全部逐视图指标保存在隔离目录 anchor-si-ablation-20261020。SI 组必须存在非零 GI 监督和权重日志，无SI组 GI监督为零。same-K计数逐组验证；评价前后 checkpoint SHA 不变。", "",
           "两个 seed 和一个 scene 仅提供初步证据。不能据此宣称跨 scene 稳定有效。没有 stage、commit 或 push。"]
production = outcomes["cached_bernoulli"]
diagnostic = outcomes["same_k"]
conclusions = ["## 结论", "",
    f'当前 Cached Bernoulli 路径加入原版 Streaming SI 后，LoD PSNR {production["psnr"]["improved_cases"]}/8、SSIM {production["ssim"]["improved_cases"]}/8 个配对 case 改善，PSNR 平均差 {production["psnr"]["mean_delta"]:+.4f} dB。LPIPS 仅 {production["lpips"]["improved_cases"]}/8 case 改善，感知指标不稳定。',
    f'相同 K 的概率 Top-K 诊断中，PSNR {diagnostic["psnr"]["improved_cases"]}/8 case 改善；seed2027 的 1% 档退化，不能将生产 Cached Bernoulli 的收益外推到 Hard Top-K。',
    "Full 指标跨 seed 有分歧，不能把 LoD 改善表述为完整地图全面改善。",
    "建议：保留可选 SI，作为当前 Cached Bernoulli Anchor 路径的候选配置；默认配置暂不切换。继续验证跨 scene 的生产收益和 LPIPS 退化原因；本轮不调权重、不改采样器，也不自动继续实验。", "",
    "## 逐视图改善覆盖", "",
    "| seed | 选择策略 | 档位 | PSNR 改善视图比例 | SSIM 改善视图比例 | LPIPS 改善视图比例 |", "|---|---|---|---:|---:|---:|"]
for seed in (2026, 2027):
    for policy in ("cached_bernoulli", "same_k"):
        for level in (("Full",) if policy == "cached_bernoulli" else ())+("LOD_1", "LOD_5", "LOD_10", "LOD_15"):
            values = [next(r["improved_view_fraction"] for r in paired if r["seed"] == seed and r["policy"] == policy
                           and r["level"] == level and r["metric"] == metric) for metric in ("psnr", "ssim", "lpips")]
            conclusions.append(f'| {seed} | {policy} | {level} | {values[0]:.1%} | {values[1]:.1%} | {values[2]:.1%} |')
report[2:2] = conclusions+[""]
(output/"ANCHOR_SI_ABLATION_REPORT.md").write_text("\n".join(report)+"\n", encoding="utf-8")
summary = dict(scene="fr1_xyz", seeds=[2026, 2027], frames=len(frames), cases=len(cases),
               common_k=plan["common_k"], runtime=runtime, paired=paired, outcomes=outcomes)
(output/"audit_summary.json").write_text(json.dumps(summary, indent=2))
print("\n".join(report[:29]))
