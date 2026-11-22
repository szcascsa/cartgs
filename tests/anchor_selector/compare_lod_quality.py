"""汇总 Anchor v2 与旧生产 Legacy-SI 的共同视图 LoD 对照。"""
import csv
import json
from pathlib import Path
import sys
import numpy as np

repo = Path(sys.argv[1]).resolve()
evidence = repo/"tests/anchor_selector/validation/evidence"
output = repo/"tests/anchor_selector/validation/lod_comparison"
output.mkdir(exist_ok=True)
read = lambda path: json.loads(path.read_text())
frames = read(evidence/"validation/evaluation_frames.json")["gt_indices"]
rows, per_view, paired = [], [], []
summaries = {}
for method, folder, filenames in (
        ("Anchor", "quality_smoke", ("quality_smoke.json",)),
        ("Legacy-SI", "quality_legacy", ("legacy_full_quality.json", "legacy_lod_quality.json"))):
    arms = [arm for name in filenames for arm in read(evidence/"validation"/name)]
    summaries[method] = {arm["arm"]: arm for arm in arms}
    for arm in arms:
        directory = evidence/folder/arm["arm"]
        assert read(directory/"evaluation_frames.json")["gt_indices"] == frames
        stats = arm["stats"]
        n, total = stats["active_gaussians"], stats["total_gaussians"]
        assert stats["render_gaussians"] == n and 0 < n <= total
        if arm["ratio"] is not None:
            manifest = read(directory/"ply_manifest.json")
            assert manifest["active_gaussians"] == n and manifest["total_gaussians"] == total
            assert manifest["ratio"] == arm["ratio"] and manifest["transform_applied"]
            expected_checkpoint = "6828_shutdown" if method == "Anchor" else "5874_shutdown"
            assert expected_checkpoint in manifest["selector_checkpoint"]
            assert expected_checkpoint in manifest["source_ply"]
            assert expected_checkpoint in manifest["transform_checkpoint"]
        row = dict(method=method, arm=arm["arm"], ratio=arm["ratio"], gaussians=n,
                   master_gaussians=total, retained_fraction=n/total, views=len(frames))
        values = {metric: np.atleast_1d(np.loadtxt(directory/(metric+".txt")))
                  for metric in ("psnr", "ssim", "lpips")}
        assert all(len(v) == len(frames) and np.isfinite(v).all() for v in values.values())
        for metric, value in values.items():
            assert abs(value.mean()-float(arm["metrics"][metric])) < 1e-10
            row[metric] = float(value.mean())
        rows.append(row)
        for i, frame in enumerate(frames):
            per_view.append(dict(method=method, arm=arm["arm"], gt_index=frame,
                                 gaussians=n, **{metric: float(v[i]) for metric, v in values.items()}))

for arm in ("Full", "LOD_1", "LOD_5", "LOD_10", "LOD_15"):
    for metric, sign in (("psnr", 1), ("ssim", 1), ("lpips", -1)):
        a = np.array([r[metric] for r in per_view if r["method"] == "Anchor" and r["arm"] == arm])
        b = np.array([r[metric] for r in per_view if r["method"] == "Legacy-SI" and r["arm"] == arm])
        delta = a-b
        paired.append(dict(arm=arm, metric=metric, comparison="Anchor-Legacy-SI", views=len(delta),
                           mean=float(delta.mean()), median=float(np.median(delta)),
                           p10=float(np.percentile(delta, 10)), p90=float(np.percentile(delta, 90)),
                           improved_views=int((sign*delta > 0).sum()),
                           improved_view_fraction=float((sign*delta > 0).mean())))

for filename, data in (("quality_comparison.csv", rows), ("quality_per_view.csv", per_view),
                       ("paired_statistics.csv", paired)):
    with (output/filename).open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(data[0]))
        writer.writeheader()
        writer.writerows(data)

table = ["| 档位 | Anchor 点数 | Legacy-SI 点数 | Anchor PSNR | Legacy PSNR | ΔPSNR | Anchor SSIM | Legacy SSIM | Anchor LPIPS | Legacy LPIPS |",
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
for arm, label in (("LOD_1", "1%"), ("LOD_5", "5%"), ("LOD_10", "10%"), ("LOD_15", "15%"), ("Full", "Full")):
    a = next(r for r in rows if r["method"] == "Anchor" and r["arm"] == arm)
    b = next(r for r in rows if r["method"] == "Legacy-SI" and r["arm"] == arm)
    table.append(f'| {label} | {a["gaussians"]:,} | {b["gaussians"]:,} | {a["psnr"]:.4f} | {b["psnr"]:.4f} | {a["psnr"]-b["psnr"]:+.4f} | {a["ssim"]:.5f} | {b["ssim"]:.5f} | {a["lpips"]:.5f} | {b["lpips"]:.5f} |')
summary = dict(scene="fr1_xyz", frames=frames, seed=0, selection="cached Bernoulli",
               exact_same_k=False, checkpoint_anchor="6828_shutdown", checkpoint_legacy="5874_shutdown",
               baseline="old production SI-guided Selector, not Raw SI ranking or PG-ST",
               paired=paired, quality=rows)
(output/"comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
report = """# Anchor LoD 与 Legacy-SI 对照

在当前 fr1_xyz 单次运行、8 个共同评价帧上，Anchor 四档 LoD 均弱于 Legacy-SI：PSNR 低 1.80–2.05 dB，SSIM 更低、LPIPS 更高。四档共 32 个视图对中，三项质量指标均由 Legacy-SI 胜出。Full PSNR 也低 0.4079 dB。

本次补齐旧生产 Legacy-SI 的四档 LoD；没有重新训练，也没有修改 Selector、预算或损失。
Anchor 使用最终 joint ratio-conditioning v2；Legacy-SI 指原生产 SI-guided Selector，不是 Raw SI ranking，也不是 PG-ST。

## 实测质量

PSNR、SSIM 越高越好，LPIPS 越低越好；ΔPSNR = Anchor − Legacy-SI。

"""+"\n".join(table)+"""

## 协议与适用范围

- 场景为 TUM fr1_xyz，两次完整在线运行均使用 792 条 RGB-D association、seed 0、尾优化 20 步。
- 两者使用同一组 8 个时间均匀评价视图：84、175、265、355、446、536、626、717；没有按质量挑视图。
- 共同的是 GT 图像索引；渲染姿态沿用各自运行的估计轨迹，不是同一组冻结相机外参。
- 使用各自冻结 checkpoint、各自 TF 和原有 Cached Bernoulli 导出；实际点数不保证等于 nominal ratio × Master 点数。
- Anchor Master 为 24,561 点、iteration 6828；Legacy-SI Master 为 27,206 点、iteration 5874。在线 Mapping 次数不同，地图不是同一个冻结状态。
- Anchor 成熟度固定为 1、关闭 SI；Legacy 保留原成熟度与 SI 训练。该结果包含整个在线系统差异，不能单独归因于 Anchor MLP。
- 本表是相同 nominal ratio、相同视图对照，不是 exact same-K 或固定地图消融；只覆盖一个场景、一个 seed、8 个视图。
- 1% 档 Anchor 选了更多点（284 对 266），质量仍较低；5%/10%/15% 档 Anchor 点数更少，因此不能把全部差距归因于 Selector 的选点能力。
- 使用既有 evaluator 的 PSNR/SSIM/LPIPS；本次未新增 photo 指标，也未进行可信的独立速度 benchmark。

## 逐视图证据

"""
report += "| 档位 | PSNR 改善视图 | SSIM 改善视图 | LPIPS 改善视图 |\n|---|---:|---:|---:|\n"
for arm, label in (("LOD_1", "1%"), ("LOD_5", "5%"), ("LOD_10", "10%"), ("LOD_15", "15%"), ("Full", "Full")):
    counts = [next(r["improved_views"] for r in paired if r["arm"] == arm and r["metric"] == m) for m in ("psnr", "ssim", "lpips")]
    report += f"| {label} | {counts[0]}/8 | {counts[1]}/8 | {counts[2]}/8 |\n"
report += """

原始逐视图指标、实际点数、paired mean/median/p10/p90 均保存在 `tests/anchor_selector/validation/lod_comparison/`。
Legacy 四档原始 evaluator 输出保存在 `tests/anchor_selector/validation/evidence/quality_legacy/`。
没有 stage、commit 或 push。
"""
(repo/"ANCHOR_LOD_VS_LEGACY_SI_COMPARISON.md").write_text(report, encoding="utf-8")
print("\n".join(table))
print(json.dumps(paired, indent=2))
