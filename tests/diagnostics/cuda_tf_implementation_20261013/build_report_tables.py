"""从保留的CSV/JSON生成中文完整表，不从日志估计缺失数值。"""
import argparse
import csv
import json
from pathlib import Path

d = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--revision", choices=("initial", "reciprocal_fix"), default="reciprocal_fix")
args = parser.parse_args()
fixed = args.revision == "reciprocal_fix"
prefix = "guard_off_reciprocal_fix_" if fixed else "guard_off_"
commit = "462d53f" if fixed else "839a1e1"
benchmark_files = ["phase_a_benchmark.csv", "phase_b_after_c/phase_b_benchmark.csv",
                   "phase_c_benchmark.csv", "mlp_backward_benchmark.csv",
                   "regularization_reciprocal_fix/near_zero_adam/phase_d_benchmark.csv" if fixed else "phase_d_benchmark.csv",
                   "phase_f1_after_api/phase_f1_benchmark.csv", "phase_f2_benchmark.csv",
                   "regularization_reciprocal_fix/phase_3_benchmark.csv" if fixed else "phase_3_benchmark.csv",
                   "regularization_reciprocal_fix/phase_5_benchmark.csv" if fixed else "phase_5_benchmark.csv"]
text = ["# CUDA TF 独立benchmark完整表", "", f"最终实现提交：{commit}。",
        "RTX3090 / CUDA11.8 / LibTorch2.3.1+cu118，全部FP32；100次预热、500次测量。",
        "数值单位ms，独立阶段不可相加替代完整训练total。",
        "query/MLP/output/backward/Adam代码未被正则修正改变，保留其原独立阶段测量。",
        "phase_3为所选revision的旧Adam/Preview基线，phase_5为同一revision的完整CUDA Adam/Preview；对应legacy同机测量。",
        "probe的wall行包含每轮显式event等待，不是完整Mapping wall time；正则与optimizer参数规模固定。"]
for name in benchmark_files:
    rows = list(csv.DictReader((d/name).open(encoding="utf-8")))
    text += ["", "## "+name, "", "| Backend | 口径 | N | 阶段 | mean | p50 | p95 |",
             "|---|---|---:|---|---:|---:|---:|"]
    for row in rows:
        values = [row.get(key, "固定" if key == "N" else "独立")
                  for key in ("backend", "mode", "N", "stage", "mean_ms", "p50_ms", "p95_ms")]
        text.append("| "+" | ".join(values)+" |")
(d / "BENCHMARK_TABLES.md").write_text("\n".join(text)+"\n", encoding="utf-8")

text = ["# GradientConstraint关闭的完整在线结果", "", f"实验对应实现：{commit}。",
        "10次Mapping与50个完整Full/LoD评估全部退出码0，每组使用全部输入关联帧；不抽帧、不训练测试帧。",
        "同组仅backend不同，desk补充两边共同start=1000；post_slam_mappings=tail_completed_steps=0。",
        "Mapping mean/p95为原验收executable的host cycle wall time，未用独立TF诊断executable替代。",
        "显存为nvidia-smi每秒进程观察峰，包含SLAM，不是Torch allocator peak。"]
for title, name in [("原fast，TF start=3000", prefix+"online_summary.json"),
                    ("授权desk补充，TF start=1000", prefix+"desk_start1000_online_summary.json")]:
    rows = json.loads((d/name).read_text(encoding="utf-8"))
    text += ["", "## "+title, "",
             "| 场景 | Backend | cycles | committed TF updates | CD TF steps | Gaussian total | Mapping mean/p95 ms | 进程wall s | 显存峰MiB |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        c, m = row["counters"], row["mapping_profile"]
        values = [row["scene"], row["backend"], c["mapping_cycles"], c["tf_updates"], c["cd_tf_steps"],
                  c["gaussian_total"], m["complete_mapping_mean_ms"]+"/"+m["complete_mapping_p95_ms"],
                  f'{row["wall_seconds"]:.3f}', str(row["gpu_memory"]["observed_peak_mib"])]
        text.append("| "+" | ".join(values)+" |")
    text += ["", "| 场景 | Backend | 平移APE RMSE m | 旋转APE RMSE unit-less |",
             "|---|---|---:|---:|"]
    for row in rows:
        t = row["trajectory"]
        text.append(f'| {row["scene"]} | {row["backend"]} | {t["translation_ape_rmse_m"]:.6f} | {t["rotation_ape_rmse_unitless"]:.6f} |')
    text += ["", "APE使用SE(3) Umeyama alignment，无scale alignment。", "",
             "| 场景 | Backend | 口径 | 帧数 | PSNR dB | SSIM | LPIPS | active count | TF applied |",
             "|---|---|---|---:|---:|---:|---:|---:|---|"]
    for row in rows:
        assert len(row["quality"]) == 5
        for q in row["quality"]:
            assert q["status"] == "complete" and q["exit_code"] == 0
            m, s = q["metrics"], q.get("selector_stats", {})
            active = s.get("active_gaussians", row["counters"]["gaussian_total"])
            applied = str(s.get("transform_applied", False)).lower()
            text.append(f'| {row["scene"]} | {row["backend"]} | {q["label"]} | {q["evaluated_frames"]} | {m["psnr"]:.6f} | {m["ssim"]:.6f} | {m["lpips"]:.6f} | {active} | {applied} |')
    text += ["", "### 原生产Mapping分阶段采样", "",
             "既有MappingProfiler每19轮的current-stream event均值；范围嵌套，不能逐列相加。", "",
             "| 场景 | Backend | Full render | Full backward | LoD总范围 | LoD render | LoD backward范围 | 全optimizer | 总cycle CUDA ms | 样本数 |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in rows:
        m = row["mapping_profile"]
        values = [m.get(k+"_cuda_elapsed_mean_ms", "无")
                  for k in ("full_render", "full_backward", "lod", "lod_render", "lod_backward", "optimizer", "total")]
        text.append("| "+" | ".join([row["scene"], row["backend"]]+values+[m["total_cuda_samples"]])+" |")
    text += ["", "### 逐1000轮TF population观察", "",
             "原生产日志周期观察，不是逐轮mean N；ratio为该轮抽到的0.01。", "",
             "| 场景 | Backend | mapping_iter:TF N |", "|---|---|---|"]
    for row in rows:
        observations = ", ".join(o["mapping_iter"]+":"+o["tf"] for o in row["periodic_lod_observations"])
        text.append(f'| {row["scene"]} | {row["backend"]} | {observations} |')
(d / "ONLINE_RESULTS.md").write_text("\n".join(text)+"\n", encoding="utf-8")
