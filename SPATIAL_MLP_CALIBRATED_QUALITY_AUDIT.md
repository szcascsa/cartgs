# 修复版 Spatial MLP 正式质量测试

**结论：修复明显减轻了旧 MLP 的退化，但当前 Spatial MLP 仍没有稳定的质量优势。** non-Bank 相对旧 MLP 的 PSNR/photo 均为 23/24 改善，平均 ΔPSNR=+2.28307 dB；相对 PG-ST 的 PSNR 仅 1/24 改善、photo 0/24；相对 Raw 的 PSNR/photo 均为 7 改善、1 相同、16 变差，平均 ΔPSNR=-0.10241 dB。相对 Fixed/C 的平均 ΔPSNR 分别为 -0.48990/-0.49688 dB，LPIPS 均为 24/24 更差。

因此：已证实的梯度缺陷得到修正，但空间共享的质量假设仍未被支持。当前不进入 shared grid 或 split，保留 Current C；本轮停止。失败不能仅归因于 generalization：Bank hard photo 也只有 8/24 改善，sampled photo 只有 7/24 改善。残余的 optimization、ST objective 与空间参数化限制尚未被单独隔离，不能唯一归因于表达能力或平滑性。

本轮固定三个 scene × 四个 ratio × 两个 seed，完成 24 个独立模型、各 2000 steps，共 48,000 次更新。统一 LR=1e-4，固定 cached-uniform ST Bernoulli、Bank schedule、photo loss 与 Adam；完整 population differentiable centering + implicit expected-K VJP + 无输出 bias。正式结果均为 step2000 deterministic hard same-K，未用 non-Bank 选择 LR、step 或模型。

本轮相对 legacy MLP 同时修正了公共输出方向与完整预算校准导数，因此不能从本轮单独归因两者各自的质量贡献。

**主要结果：每项均为 24 个 scene/ratio/seed case，改善方向使用未舍入数值**

| 对照 | split | PSNR 改善/相同/变差 | SSIM 改善 | LPIPS 改善 | photo 改善 | 平均 ΔPSNR | ΔPSNR 范围 |
|---|---|---:|---:|---:|---:|---:|---:|
| MLP-ST-legacy | Bank | 23/0/1 | 24 | 18 | 23 | +1.79134 | -0.01252 … +7.17320 |
| MLP-ST-legacy | non-Bank | 23/0/1 | 24 | 22 | 23 | +2.28307 | -0.00669 … +9.49176 |
| PG-ST | Bank | 1/0/23 | 0 | 2 | 1 | -0.26196 | -1.17807 … +0.01425 |
| PG-ST | non-Bank | 1/0/23 | 2 | 1 | 0 | -0.28771 | -1.32111 … +0.00639 |
| Raw | Bank | 5/1/18 | 7 | 6 | 8 | -0.07759 | -0.55101 … +0.20976 |
| Raw | non-Bank | 7/1/16 | 7 | 3 | 7 | -0.10241 | -0.56321 … +0.28075 |
| Fixed | Bank | 3/0/21 | 10 | 0 | 7 | -0.46049 | -1.87007 … +0.24077 |
| Fixed | non-Bank | 3/0/21 | 9 | 0 | 8 | -0.48990 | -2.08190 … +0.20640 |
| C | Bank | 3/0/21 | 7 | 0 | 7 | -0.46726 | -1.63104 … +0.26249 |
| C | non-Bank | 3/0/21 | 8 | 0 | 9 | -0.49688 | -1.70310 … +0.17306 |

改善数目不能单独表示收益大小，必须结合下面的 delta、paired-view 分布及最终集合改动。四指标的完整 aggregate 与 per-view 数值均保存在 CSV。

**non-Bank：全部 24 个模型**

| scene | ratio | seed | PSNR | SSIM | LPIPS | photo | ΔPSNR legacy MLP | ΔPSNR PG | ΔPSNR Raw | ΔPSNR Fixed | ΔPSNR C |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fr1_room | 1% | 2026 | 13.01290 | 0.553012 | 0.751109 | 0.219048 | +0.19569 | -0.22720 | +0.02463 | -0.86955 | -0.82161 |
| fr1_room | 1% | 2027 | 12.99440 | 0.552501 | 0.752920 | 0.219695 | +0.15382 | -0.37942 | +0.00613 | -0.88805 | -0.84012 |
| fr1_room | 5% | 2026 | 16.65264 | 0.616853 | 0.648514 | 0.153131 | +0.10610 | -0.00165 | -0.00050 | -0.32772 | -0.39430 |
| fr1_room | 5% | 2027 | 16.64945 | 0.616862 | 0.648516 | 0.153134 | +0.08067 | -0.01123 | -0.00369 | -0.33091 | -0.39749 |
| fr1_room | 10% | 2026 | 17.68029 | 0.640307 | 0.599725 | 0.137313 | +0.06336 | -0.01382 | -0.00006 | -0.12192 | -0.07979 |
| fr1_room | 10% | 2027 | 17.67935 | 0.640314 | 0.599724 | 0.137312 | +0.02989 | -0.02712 | -0.00100 | -0.12286 | -0.08073 |
| fr1_room | 15% | 2026 | 18.05491 | 0.649811 | 0.571331 | 0.131865 | +0.00551 | -0.02725 | +0.00020 | -0.17064 | -0.12522 |
| fr1_room | 15% | 2027 | 18.05491 | 0.649811 | 0.571331 | 0.131865 | -0.00669 | -0.04292 | +0.00020 | -0.17064 | -0.12522 |
| fr1_xyz | 1% | 2026 | 12.50291 | 0.560557 | 0.708051 | 0.206320 | +3.92282 | -1.32111 | -0.51776 | -2.08190 | -1.70310 |
| fr1_xyz | 1% | 2027 | 13.30142 | 0.570773 | 0.702235 | 0.193464 | +1.57344 | -0.43873 | +0.28075 | -1.28339 | -0.90459 |
| fr1_xyz | 5% | 2026 | 17.62424 | 0.655728 | 0.548512 | 0.127290 | +7.56995 | -0.70255 | -0.19038 | -1.15314 | -1.19783 |
| fr1_xyz | 5% | 2027 | 17.88867 | 0.658554 | 0.544015 | 0.125386 | +6.47598 | -0.17073 | +0.07404 | -0.88871 | -0.93340 |
| fr1_xyz | 10% | 2026 | 19.65571 | 0.701729 | 0.443603 | 0.105255 | +8.80443 | -0.39884 | -0.15639 | -0.60549 | -0.64626 |
| fr1_xyz | 10% | 2027 | 19.39363 | 0.696860 | 0.450909 | 0.107636 | +6.83964 | -0.61535 | -0.41847 | -0.86757 | -0.90834 |
| fr1_xyz | 15% | 2026 | 20.79320 | 0.724409 | 0.393663 | 0.095658 | +9.49176 | -0.13776 | -0.09694 | -0.02434 | -0.10702 |
| fr1_xyz | 15% | 2027 | 20.40651 | 0.717725 | 0.402234 | 0.098839 | +5.99387 | -0.57662 | -0.48363 | -0.41104 | -0.49372 |
| fr2_xyz | 1% | 2026 | 16.05456 | 0.588707 | 0.730783 | 0.161971 | +0.56040 | -0.44573 | -0.17945 | -0.27514 | -0.67222 |
| fr2_xyz | 1% | 2027 | 15.67080 | 0.583672 | 0.730091 | 0.166729 | +0.04579 | -0.82858 | -0.56321 | -0.65890 | -1.05598 |
| fr2_xyz | 5% | 2026 | 19.90089 | 0.657683 | 0.507006 | 0.114412 | +0.67299 | -0.04799 | +0.01744 | -0.35325 | -0.33194 |
| fr2_xyz | 5% | 2027 | 19.87244 | 0.657614 | 0.507297 | 0.114506 | +0.39043 | -0.07779 | -0.01101 | -0.38170 | -0.36039 |
| fr2_xyz | 10% | 2026 | 21.19833 | 0.699069 | 0.407170 | 0.099393 | +0.69972 | +0.00639 | -0.04038 | +0.16602 | +0.13267 |
| fr2_xyz | 10% | 2027 | 21.23871 | 0.699281 | 0.406361 | 0.099144 | +0.57433 | -0.05603 | +0.00000 | +0.20640 | +0.17306 |
| fr2_xyz | 15% | 2026 | 21.30850 | 0.710317 | 0.374254 | 0.096729 | +0.19800 | -0.27408 | -0.17145 | -0.14391 | -0.09805 |
| fr2_xyz | 15% | 2027 | 21.45312 | 0.711740 | 0.366567 | 0.095411 | +0.35181 | -0.08902 | -0.02683 | +0.00071 | +0.04657 |

**Paired-view：non-Bank PSNR**

| 对照 | 正 median 的 case | 多数 view 改善的 case | case improved-view fraction 中位数 | case improved-view fraction 范围 |
|---|---:|---:|---:|---:|
| MLP-ST-legacy | 23/24 | 23/24 | 0.806 | 0.385 … 1.000 |
| PG-ST | 0/24 | 0/24 | 0.115 | 0.000 … 0.487 |
| Raw | 4/24 | 4/24 | 0.222 | 0.000 … 0.926 |
| Fixed | 2/24 | 2/24 | 0.111 | 0.000 … 0.821 |
| C | 4/24 | 4/24 | 0.130 | 0.000 … 0.795 |

每个 case 的 mean、median、p10、p90、improved/equal view fraction，以及四指标完整统计见 paired_statistics.csv；没有使用新的 dB 阈值。

**训练动态与真实 reordering**

Bank hard photo 比初始化降低：8/24。按每个 Bank frame 首末各五次访问等权比较，sampled photo 降低：7/24。最终相对 Raw 存在 boundary crossing：23/24。

| scene | ratio | seed | Bank hard Δphoto | Bank hard ΔPSNR | final correction std | Raw added | Jaccard vs Raw |
|---|---:|---:|---:|---:|---:|---:|---:|
| fr1_room | 1% | 2026 | -0.0001883 | +0.01252 | 0.0141157 | 7 | 0.972549 |
| fr1_room | 1% | 2027 | +0.0005420 | -0.01023 | 0.0032757 | 3 | 0.988142 |
| fr1_room | 5% | 2026 | +0.0000045 | -0.00008 | 0.0004430 | 1 | 0.999206 |
| fr1_room | 5% | 2027 | +0.0000351 | -0.00467 | 0.0069371 | 4 | 0.996827 |
| fr1_room | 10% | 2026 | -0.0000015 | +0.00009 | 0.0003470 | 1 | 0.999603 |
| fr1_room | 10% | 2027 | -0.0000097 | -0.00015 | 0.0016148 | 3 | 0.998809 |
| fr1_room | 15% | 2026 | -0.0000092 | -0.00016 | 0.0005121 | 2 | 0.999470 |
| fr1_room | 15% | 2027 | -0.0000092 | -0.00016 | 0.0011397 | 2 | 0.999470 |
| fr1_xyz | 1% | 2026 | +0.0070617 | -0.40117 | 0.0608890 | 4 | 0.966667 |
| fr1_xyz | 1% | 2027 | -0.0030544 | +0.20976 | 0.2725840 | 24 | 0.815385 |
| fr1_xyz | 5% | 2026 | +0.0009123 | -0.20528 | 0.0742052 | 27 | 0.955335 |
| fr1_xyz | 5% | 2027 | -0.0009183 | +0.09740 | 0.0532426 | 21 | 0.965087 |
| fr1_xyz | 10% | 2026 | +0.0008384 | -0.02570 | 0.1758653 | 123 | 0.901086 |
| fr1_xyz | 10% | 2027 | +0.0023307 | -0.15427 | 0.3039578 | 177 | 0.860685 |
| fr1_xyz | 15% | 2026 | +0.0003532 | -0.07619 | 0.0958243 | 84 | 0.953719 |
| fr1_xyz | 15% | 2027 | +0.0024195 | -0.32983 | 0.3624255 | 276 | 0.855573 |
| fr2_xyz | 1% | 2026 | +0.0025163 | -0.21924 | 0.2848092 | 19 | 0.915556 |
| fr2_xyz | 1% | 2027 | +0.0070938 | -0.55101 | 0.5436032 | 52 | 0.784679 |
| fr2_xyz | 5% | 2026 | -0.0001027 | +0.02734 | 0.0626939 | 22 | 0.979770 |
| fr2_xyz | 5% | 2027 | +0.0000696 | -0.00898 | 0.0568084 | 11 | 0.989834 |
| fr2_xyz | 10% | 2026 | +0.0002796 | -0.04401 | 0.3247429 | 114 | 0.948416 |
| fr2_xyz | 10% | 2027 | +0.0000000 | +0.00000 | 0.0007728 | 0 | 1.000000 |
| fr2_xyz | 15% | 2026 | +0.0016209 | -0.15598 | 0.4695379 | 199 | 0.940222 |
| fr2_xyz | 15% | 2027 | +0.0001300 | -0.02219 | 0.0624933 | 28 | 0.991367 |

seed_consistency.csv 保存新旧 MLP 与 PG 的跨 seed 四指标差异。若新模型更接近 Raw、集合变化很少，较小 seed 差异不能单独解释为有效 regularization。

**Correctness 与冻结条件**

- 五组历史基线均用原冻结 renderer 重渲染，9940 个 view 的 PSNR/SSIM/LPIPS/photo 与历史记录逐项 exact，aggregate exact。
- 修复版 2840 个 view：Bank 808、non-Bank 2032；全部为唯一且严格 same-K 的 hard 集合。训练 admission 仍为期望 K 的旧 Bernoulli 协议。
- 24 个模型的 cached uniform 与 Bank frame schedule exact 对齐 PG-ST，初始化完整排序 exact Raw；step0/250/1000/2000 的权重经独立 Python forward 核对。
- Gaussian/TF/Master native tensor exact，输入/相机/Bank/AABB/runtime 文件训练前后 SHA 检查通过。
- 全部更新中最大 normalized gradient sum 为 3.41e-16；correction 均值最大绝对值 1.34e-15。
- 最终 selected sets/rankings 在新模型 quality render 前锁定 SHA；没有 GT-driven 选点、eval tuning、early stopping、regularization、grid 或 split。

首次 gate 因新脚本额外关闭 cuDNN TF32 导致 LPIPS 差异而停止，未训练任何模型。单帧对照确认历史 LPIPS 使用 cuDNN TF32=True；恢复原 metric 环境后才重跑并通过全部 exact gate，没有改变 metric 公式或放宽容差。首次失败日志与环境诊断均保留。

**输出**

- [quality_summary.csv](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/spatial_mlp_calibrated_quality_20261020/quality_summary.csv)、[quality_per_view.csv](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/spatial_mlp_calibrated_quality_20261020/quality_per_view.csv)、[paired_statistics.csv](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/spatial_mlp_calibrated_quality_20261020/paired_statistics.csv)。
- [training_curves.csv](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/spatial_mlp_calibrated_quality_20261020/training_curves.csv)、[hard_training_dynamics.csv](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/spatial_mlp_calibrated_quality_20261020/hard_training_dynamics.csv)、[training_dynamics_summary.csv](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/spatial_mlp_calibrated_quality_20261020/training_dynamics_summary.csv)。
- [selected_set_overlap.csv](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/spatial_mlp_calibrated_quality_20261020/selected_set_overlap.csv)、[raw_rank_promotion.csv](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/spatial_mlp_calibrated_quality_20261020/raw_rank_promotion.csv)、[correction_statistics.csv](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/spatial_mlp_calibrated_quality_20261020/correction_statistics.csv)、[seed_consistency.csv](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/spatial_mlp_calibrated_quality_20261020/seed_consistency.csv)。
- [correctness.json](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/spatial_mlp_calibrated_quality_20261020/correctness.json)、[audit_summary.json](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/spatial_mlp_calibrated_quality_20261020/audit_summary.json)、rankings.npz、selected_sets.npz。
- 每个模型均保存 step0/250/1000/2000 的 MLP 权重和 tensor 快照、final_field.npz、normalization.json 与原始 training.log。

本轮完成后停止，未修改 architecture、LR、loss、sampling protocol，也未实现 grid/split；没有 stage、commit、push。
