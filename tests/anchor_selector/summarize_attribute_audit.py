"""单次27维无SI实验与既有相同seed基线比较；不调参、不挑结果。"""
import csv
import json
from pathlib import Path
import re
import sys
import numpy as np

root = Path(sys.argv[1]).resolve()
out = root/"validation"
protocol = json.loads((root/"source/tests/anchor_selector/attribute_protocol.json").read_text())
reference = Path(protocol["reference"])
cases = json.loads((out/"quality_summary.json").read_text())
assert len(cases) == 9
new = [dict(case, method="Anchor-27 noSI") for case in cases]
old = [dict(case, method="Anchor-20 "+("SI" if case["arm"] == "si" else "noSI")) for case in
       json.loads((reference/"validation/quality_summary.json").read_text()) if case["seed"] == protocol["seed"]]
all_cases = new+old
quality, per_view, paired = [], [], []
for case in all_cases:
    directory = Path(case["output"])
    frames = json.loads((directory/"evaluation_frames.json").read_text())["gt_indices"]
    assert frames == json.loads((reference/"validation/common_frames.json").read_text())["gt_indices"]
    stats = case["stats"]
    assert stats["active_gaussians"] == stats["render_gaussians"]
    values = {m:np.atleast_1d(np.loadtxt(directory/(m+".txt"))) for m in ("psnr","ssim","lpips")}
    assert all(len(v)==792 and np.isfinite(v).all() for v in values.values())
    quality.append(dict(method=case["method"],policy=case["policy"],level=case["level"],ratio=case["ratio"],
        population=stats["total_gaussians"],active=stats["active_gaussians"],**case["metrics"]))
    for i, frame in enumerate(frames):
        per_view.append(dict(method=case["method"],policy=case["policy"],level=case["level"],gt_index=frame,
                             **{m:float(v[i]) for m,v in values.items()}))
    if case["ratio"] is not None:
        selected = np.load(directory/"selected_rows.npz")["rows"]
        assert len(selected)==len(np.unique(selected))==stats["active_gaussians"]
for case in new:
    for baseline in ("Anchor-20 noSI","Anchor-20 SI"):
        for metric, sign in (("psnr",1),("ssim",1),("lpips",-1)):
            values = {method:np.array([r[metric] for r in per_view if r["method"]==method and
                r["policy"]==case["policy"] and r["level"]==case["level"]]) for method in (case["method"],baseline)}
            delta = values[case["method"]]-values[baseline]
            paired.append(dict(baseline=baseline,policy=case["policy"],level=case["level"],metric=metric,
                mean=float(delta.mean()),median=float(np.median(delta)),p10=float(np.percentile(delta,10)),
                p90=float(np.percentile(delta,90)),improved_view_fraction=float((sign*delta>0).mean())))
for filename, rows in (("quality_summary.csv",quality),("quality_per_view.csv",per_view),("paired_statistics.csv",paired)):
    with (out/filename).open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
job=json.loads((out/"training_job.json").read_text())
training=(Path(job["output"])/"training.log").read_text()
line=[line for line in training.splitlines() if "[UltrafastRotation SUMMARY] event=training_end" in line][-1]
counts=dict(re.findall(r"(\w+)=([0-9.e+-]+)",line))
assert int(counts["ultrafast_total_normal_steps"])==10000 and counts["balanced"]=="1"
assert all(float(v)==0 for v in re.findall(r"gi_supervised_count=([0-9.e+-]+)",training))
runtime=dict(job,counts=counts)
(out/"runtime.json").write_text(json.dumps(runtime,indent=2))
report=["# Anchor Selector 新增 Gaussian 属性输入：单次无 SI 训练", "",
    "fr1_xyz / seed2026 / 10,000 次 Mapping / 792 个共同图像帧。仅新训练一次；对照为此前相同 seed/预算的 Anchor-20 无 SI，另列 Anchor-20 + SI 作为参考。", "",
    "输入按顺序为 Anchor feature16、local offset3、activated scale3、SH DC RGB3、activated opacity1、log ratio1，共27维。SH DC 直接取第0阶三通道系数，不转成 RGB；scale/opacity 为 TF 前显式 Gaussian 的 exp/sigmoid 激活。网络27→32→32→2，ReLU，共2018个共享参数（原1794），另有每Anchor16维特征。", "",
    "保留 joint LoD：新增属性输入不额外detach，Gaussian、Selector/Anchor、TF 都可接收 LoD 梯度；Full 仍只更新 Master Gaussian。Change Distill 使用旧/新属性的 detached 快照，不修改 Gaussian。", "",
    "SI/GI mode=disabled、SI loss权重0；原photo/ratio/TF loss、LR、h=0.1、Cached Bernoulli、ratio轮转和maturity=1保持。默认新增属性开关为0，实验单独开启1；旧19/20维和新27维Python checkpoint显式识别。", "",
    "## 生产 Cached Bernoulli 质量", "",
    "| 档位 | 原20维无SI 点数 / PSNR | 新27维无SI 点数 / PSNR | ΔPSNR | ΔSSIM | ΔLPIPS | 原20维+SI PSNR |",
    "|---|---:|---:|---:|---:|---:|---:|"]
def table(policy):
    lines=[]
    for level in (("Full",) if policy=="cached_bernoulli" else ())+("LOD_1","LOD_5","LOD_10","LOD_15"):
        rows={method:next(r for r in quality if r["method"]==method and r["policy"]==policy and r["level"]==level)
              for method in ("Anchor-20 noSI","Anchor-27 noSI","Anchor-20 SI")}
        a,b,c=[rows[m] for m in ("Anchor-20 noSI","Anchor-27 noSI","Anchor-20 SI")]
        lines.append(f'| {level} | {a["active"]:,} / {a["psnr"]:.4f} | {b["active"]:,} / {b["psnr"]:.4f} | {b["psnr"]-a["psnr"]:+.4f} | {b["ssim"]-a["ssim"]:+.5f} | {b["lpips"]-a["lpips"]:+.5f} | {c["psnr"]:.4f} |')
    return lines
report+=table("cached_bernoulli")
report += ["", "Δ=新27维无SI−原20维无SI；PSNR/SSIM正数较好，LPIPS负数较好。", "",
    "## 相同 K 概率排序诊断", "", "沿用此前固定K：202 / 1010 / 2020 / 3031。该诊断不改变生产 Cached Bernoulli 选择。", "",
    "| 档位 | 原20维无SI 点数 / PSNR | 新27维无SI 点数 / PSNR | ΔPSNR | ΔSSIM | ΔLPIPS | 原20维+SI PSNR |",
    "|---|---:|---:|---:|---:|---:|---:|"]
report+=table("same_k")
outcomes={}
for policy in ("cached_bernoulli","same_k"):
    outcomes[policy]={}
    for metric,sign in (("psnr",1),("ssim",1),("lpips",-1)):
        rows=[r for r in paired if r["baseline"]=="Anchor-20 noSI" and r["policy"]==policy and r["level"]!="Full" and r["metric"]==metric]
        outcomes[policy][metric]=dict(improved_cases=sum(sign*r["mean"]>0 for r in rows),mean_delta=float(np.mean([r["mean"] for r in rows])),
            improved_view_fraction=float(np.mean([r["improved_view_fraction"] for r in rows])))
report += ["", "## 训练与适用范围", "",
    f'新训练 wall={job["wall_seconds"]:.2f}s；Full renders={counts["full_render_count"]}，LoD renders={counts["lod_render_count"]}，online Mapping={counts["online_end_mapping_iteration"]}，tail Mapping={counts["tail_training_steps"]}。',
    "同一总Mapping预算不意味着相同Full/LoD次数；异步SLAM在线结束时刻、地图规模、Gaussian初始化RNG消耗、相机估计和densification轨迹都可能不同。不是冻结地图因果消融；一次训练不能证明跨seed/scene稳定收益。", "",
    "native/Python概率及cached mask、输入依赖与梯度、compact source-row、Full/LoD梯度分流、全局/局部几何变化蒸馏、冻结teacher和旧模型加载均通过专项检查。逐视图差异保存mean/median/p10/p90/improved fraction。评价前后checkpoint SHA不变。", "",
    "本轮没有调LR、扩大网络、加入SI或其他loss，没有stage、commit、push。"]
intro=["## 结果概览", ""]
for policy in ("cached_bernoulli","same_k"):
    s=outcomes[policy]
    intro.append(f'{policy}：相对原20维无SI，四档LoD的PSNR {s["psnr"]["improved_cases"]}/4 改善（平均{s["psnr"]["mean_delta"]:+.4f} dB），SSIM {s["ssim"]["improved_cases"]}/4，LPIPS {s["lpips"]["improved_cases"]}/4。')
report[2:2]=intro+[""]
(out/"ANCHOR_GAUSSIAN_ATTRIBUTE_AUDIT.md").write_text("\n".join(report)+"\n",encoding="utf-8")
(out/"audit_summary.json").write_text(json.dumps(dict(scene="fr1_xyz",seed=2026,training_runs=1,new_cases=9,views=792,
    outcomes=outcomes,paired=paired,runtime=runtime),indent=2))
print("\n".join(report[:12]))
