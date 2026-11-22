# Grid-Level Spatial MLP SI Audit

**结论：D. REGIONAL CORRECTION NOT SUPPORTED。** 在本轮固定grid、calibrated ST、统一LR与2000步协议下，Grid-MLP未获得跨scene/ratio/seed稳定收益。non-Bank相对Raw仅5/24个case的PSNR改善，平均ΔPSNR=-0.441525 dB，photo仅6/24改善；Independent-Cell为9/24个PSNR改善、平均ΔPSNR=-0.001689 dB，photo同样仅6/24改善。两组均24/24发生真实cell间预算转移，因此不能用coefficient非零或budget movement判成功。当前不建议进入gradient-threshold split。

**最终问题**

| 问题 | 回答 |
|---|---|
| Q1. MLP只对occupied cells forward？ | YES |
| Q2. 同cell严格共享一个coefficient？ | YES |
| Q3. 发生真实cell-to-cell预算移动？ | YES |
| Q4. Grid-MLP稳定优于Raw？ | NO |
| Q5. 优于已有calibrated Gaussian-input MLP？ | NO |
| Q6. 接近或超过PG-ST？ | NO |
| Q7. 缩小Fixed/C gap？ | NO |
| Q8. Independent-Cell说明grid参数化本身有效？ | NO |
| Q9. 主要限制？ | INCONCLUSIVE |
| Q10. 进入gradient-EMA threshold split？ | NO-GO |

**判读与因果边界**

- Grid-MLP的负结果主要集中在fr1_xyz与fr2_xyz：两scene各0/8个non-Bank PSNR case优于Raw；fr1_room为5/8。不能由room的少量改善外推跨scene有效。
- Independent-Cell整体明显减轻Grid-MLP的损失，但未稳定优于Raw；因此不满足“Independent-Cell明显成功、MLP失败”的C类前提。fr1_xyz的局部正向信号仍保留在完整表中，没有被全局平均掩盖。
- Q8的NO表示本轮未证明grid-level parameterization能稳定提高quality，不表示数学上不存在更好的cell coefficients。两种参数化共享当前ST目标；Independent-Cell也复用Grid-MLP选出的LR。当前实验不能隔离regional representation、MLP sharing和optimization/objective各自的因果份额，所以Q9为INCONCLUSIVE。
- LR sanity的3e-4在step250改善Bank photo，但该模型训练到step2000后变差；短期gate通过不能替代固定预算的最终hard quality。本轮保持预注册的2000步，没有事后选择step250或扩大LR搜索。
- 加权gauge与calibrated VJP正确并不保证ST training objective会提高最终deterministic hard Top-K。当前结果是对这一固定训练协议的负证据，不能归因于common-score-shift错误仍存在，也不能单独证明grid表达能力不足。
- 下一步决策为NO-GO：本轮不进入adaptive grid/split，也不据gradient统计调整threshold。若继续研究，应先解释固定regional参数化下hard质量为何无法稳定改善；本轮不自动执行后续实验。

**结构与冻结grid**

本轮MLP的对象是occupied cell：`normalized cell center[M,3] → 3→32→32→1 / SiLU → raw r[M]`。最后层bias=false、weight=0初始化，共1216参数。按occupancy计算 `mu=sum(n_C*r_C)/N`，保留autograd；`a_C=r_C-mu`，通过固定cell_id广播到N个Gaussian。score=`log(RawSI+1e-12)+a_C(i)`，严格保留cell内部Raw排序。

grid固定world origin=(0,0,0)，cell key=floor(XYZ/h)，center=(key+0.5)*h，cell ID按key字典序确定。h仅由冻结valid AND observed XYZ的occupied count选取：seed=max XYZ span/cbrt(2048)，doubling bracket，128点log probes，再对最佳邻域固定128点细化；误差相同选较大h。沿用已有sparse root的key/搜索结构，但不调用其radius-based seed；本轮没有读取radius、hierarchy、GT或quality决定grid。

| scene | N | M | M/N | h | occupancy min/median/p90/max |
|---|---:|---:|---:|---:|---|
| fr1_room | 50330 | 2048 | 4.07% | 0.299870915031 | 1 / 5 / 57 / 1189 |
| fr1_xyz | 23643 | 2048 | 8.66% | 0.0724666956951 | 1 / 3 / 30 / 401 |
| fr2_xyz | 43059 | 2047 | 4.75% | 0.124471327367 | 1 / 2 / 43.4 / 1065 |

cell centers按各scene occupied-center min/max box映射到[-1,1]^3；构造后h、mapping、occupancy、center及normalization全程冻结。occupancy只用于gauge和posthoc统计，未作为MLP输入。Gaussian XYZ也未逐行输入MLP。

**LR gate与训练协议**

| fr1_xyz /5%/seed2026，250steps LR | initial Bank photo | final Bank photo | Δphoto | ΔPSNR |
|---|---:|---:|---:|---:|
| 3e-05 | 0.121580737 | 0.121580737 | +0.000000000 | +0.000000 |
| 0.0001 | 0.121580737 | 0.121747195 | +0.000166458 | -0.013725 |
| 0.0003 | 0.121580737 | 0.121463695 | -0.000117042 | +0.011235 |

按预注册Bank hard photo优先准则选定LR=0.0003，在读取任何baseline/eval质量前锁定；未触发三档全部恶化的停止条件。全部Grid-MLP和Independent-Cell均使用该LR、Adam(.9,.999,eps1e-8,wd0)、2000steps，不early stop。

复用已修复的implicit expected-K VJP，未改校准器实现。采样仍为cached-uniform ST Bernoulli，compact selected-row renderer与opacity ST，photo=0.8L1+0.2(1-fast_ssim)，Bank schedule与旧实验exact。正式评估统一采用step2000 deterministic hard same-K。没有额外loss/regularizer/feature，也没有PG/C teacher。

Independent-Cell control每cell直接学习一个scalar，不经过MLP；使用完全相同的gauge/calibration/sampler/Bank/loss/LR/budget。共完成24个Primary和24个control模型，正式96,000次更新，另有750次LR sanity更新。

**质量主表：每项24个scene/ratio/seed case**

| arm | reference | split | PSNR改善/相同/变差 | SSIM改善 | LPIPS改善 | photo改善 | mean ΔPSNR |
|---|---|---|---:|---:|---:|---:|---:|
| Grid-MLP | Raw | Bank | 3/0/21 | 5 | 9 | 3 | -0.356631 |
| Grid-MLP | Raw | non-Bank | 5/0/19 | 8 | 8 | 6 | -0.441525 |
| Grid-MLP | PG-ST | Bank | 1/0/23 | 0 | 3 | 1 | -0.541002 |
| Grid-MLP | PG-ST | non-Bank | 1/0/23 | 2 | 2 | 1 | -0.626832 |
| Grid-MLP | Fixed | Bank | 2/0/22 | 8 | 0 | 4 | -0.739526 |
| Grid-MLP | Fixed | non-Bank | 1/0/23 | 7 | 0 | 6 | -0.829019 |
| Grid-MLP | C | Bank | 2/0/22 | 7 | 0 | 4 | -0.746302 |
| Grid-MLP | C | non-Bank | 2/0/22 | 8 | 0 | 6 | -0.835994 |
| Grid-MLP | Per-Gaussian-MLP | Bank | 4/0/20 | 4 | 10 | 4 | -0.279039 |
| Grid-MLP | Per-Gaussian-MLP | non-Bank | 6/0/18 | 8 | 9 | 7 | -0.339118 |
| Independent-Cell | Raw | Bank | 11/0/13 | 8 | 8 | 9 | +0.011865 |
| Independent-Cell | Raw | non-Bank | 9/0/15 | 4 | 7 | 6 | -0.001689 |
| Independent-Cell | PG-ST | Bank | 2/0/22 | 1 | 1 | 0 | -0.172506 |
| Independent-Cell | PG-ST | non-Bank | 2/0/22 | 2 | 1 | 1 | -0.186995 |
| Independent-Cell | Fixed | Bank | 5/0/19 | 12 | 0 | 9 | -0.371030 |
| Independent-Cell | Fixed | non-Bank | 6/0/18 | 10 | 0 | 8 | -0.389182 |
| Independent-Cell | C | Bank | 4/0/20 | 9 | 0 | 8 | -0.377807 |
| Independent-Cell | C | non-Bank | 4/0/20 | 9 | 0 | 9 | -0.396157 |
| Independent-Cell | Per-Gaussian-MLP | Bank | 15/0/9 | 12 | 13 | 14 | +0.089457 |
| Independent-Cell | Per-Gaussian-MLP | non-Bank | 14/0/10 | 10 | 12 | 11 | +0.100719 |
| Independent-Cell | Grid-MLP | Bank | 18/0/6 | 16 | 15 | 15 | +0.368496 |
| Independent-Cell | Grid-MLP | non-Bank | 14/0/10 | 14 | 12 | 14 | +0.439837 |

四指标全部absolute aggregate/per-view保存在quality_summary.csv和quality_per_view.csv。photo/LPIPS下降为改善；PSNR/SSIM上升为改善。未定义新的dB GO阈值，未挑最好seed或checkpoint。这里Per-Gaussian-MLP指已有calibrated XYZ-per-Gaussian-input MLP，直接复用既有质量结果；本轮不做coefficient MSE或PG coefficient拟合。

**non-Bank各case**

| scene | ratio | seed | Grid PSNR | ΔRaw | ΔPG-ST | ΔFixed | ΔC | ΔGaussian-MLP | Grid Δphoto Raw | Independent ΔPSNR Raw |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fr1_room | 1% | 2026 | 12.98445 | -0.00382 | -0.25565 | -0.89800 | -0.85006 | -0.02845 | +0.0030638 | -0.07433 |
| fr1_room | 1% | 2027 | 13.18309 | +0.19483 | -0.19072 | -0.69936 | -0.65142 | +0.18870 | -0.0030132 | -0.00365 |
| fr1_room | 5% | 2026 | 16.66303 | +0.00989 | +0.00874 | -0.31733 | -0.38391 | +0.01039 | -0.0001049 | -0.03890 |
| fr1_room | 5% | 2027 | 16.64503 | -0.00812 | -0.01565 | -0.33533 | -0.40192 | -0.00442 | +0.0000512 | -0.01675 |
| fr1_room | 10% | 2026 | 17.68256 | +0.00221 | -0.01155 | -0.11965 | -0.07752 | +0.00227 | -0.0000405 | +0.00151 |
| fr1_room | 10% | 2027 | 17.69230 | +0.01195 | -0.01418 | -0.10991 | -0.06779 | +0.01295 | -0.0001031 | +0.00350 |
| fr1_room | 15% | 2026 | 18.05046 | -0.00425 | -0.03170 | -0.17509 | -0.12967 | -0.00445 | -0.0000300 | -0.01623 |
| fr1_room | 15% | 2027 | 18.06043 | +0.00572 | -0.03740 | -0.16512 | -0.11969 | +0.00552 | -0.0000737 | -0.02149 |
| fr1_xyz | 1% | 2026 | 11.03227 | -1.98840 | -2.79175 | -3.55254 | -3.17374 | -1.47063 | +0.0397794 | -0.04386 |
| fr1_xyz | 1% | 2027 | 12.12631 | -0.89436 | -1.61384 | -2.45850 | -2.07970 | -1.17510 | +0.0210664 | +0.42595 |
| fr1_xyz | 5% | 2026 | 16.84592 | -0.96871 | -1.48087 | -1.93146 | -1.97615 | -0.77832 | +0.0088346 | +0.00028 |
| fr1_xyz | 5% | 2027 | 16.55781 | -1.25681 | -1.50159 | -2.21956 | -2.26426 | -1.33086 | +0.0114463 | +0.15674 |
| fr1_xyz | 10% | 2026 | 19.31930 | -0.49280 | -0.73524 | -0.94190 | -0.98267 | -0.33641 | +0.0050935 | +0.27841 |
| fr1_xyz | 10% | 2027 | 18.97396 | -0.83815 | -1.03502 | -1.28724 | -1.32802 | -0.41967 | +0.0092639 | +0.15586 |
| fr1_xyz | 15% | 2026 | 20.10412 | -0.78602 | -0.82684 | -0.71342 | -0.79610 | -0.68908 | +0.0052089 | -0.07125 |
| fr1_xyz | 15% | 2027 | 20.45941 | -0.43073 | -0.52372 | -0.35814 | -0.44081 | +0.05291 | +0.0035801 | +0.01851 |
| fr2_xyz | 1% | 2026 | 15.16434 | -1.06967 | -1.33595 | -1.16536 | -1.56244 | -0.89022 | +0.0145088 | -0.16358 |
| fr2_xyz | 1% | 2027 | 15.04700 | -1.18701 | -1.45238 | -1.28270 | -1.67978 | -0.62380 | +0.0127341 | -0.01487 |
| fr2_xyz | 5% | 2026 | 19.73091 | -0.15254 | -0.21797 | -0.52323 | -0.50193 | -0.16998 | +0.0017710 | -0.35339 |
| fr2_xyz | 5% | 2027 | 19.77975 | -0.10371 | -0.17049 | -0.47439 | -0.45309 | -0.09270 | +0.0005629 | -0.05474 |
| fr2_xyz | 10% | 2026 | 20.91579 | -0.32292 | -0.27614 | -0.11652 | -0.14987 | -0.28254 | +0.0026928 | -0.19061 |
| fr2_xyz | 10% | 2027 | 21.22840 | -0.01031 | -0.06634 | +0.19609 | +0.16275 | -0.01031 | +0.0000103 | -0.12353 |
| fr2_xyz | 15% | 2026 | 21.22787 | -0.25209 | -0.35471 | -0.22455 | -0.17868 | -0.08063 | +0.0024686 | -0.01303 |
| fr2_xyz | 15% | 2027 | 21.42915 | -0.05080 | -0.11300 | -0.02326 | +0.02260 | -0.02397 | +0.0004283 | +0.11891 |

**Paired-view与seed一致性**

| Grid-MLP reference | split | PSNR paired median>0 cases | 多于一半views改善cases | improved-view fraction跨case中位数 |
|---|---|---:|---:|---:|
| Raw | Bank | 7/24 | 6/24 | 0.3214 |
| Raw | non-Bank | 10/24 | 10/24 | 0.1154 |
| PG-ST | Bank | 0/24 | 0/24 | 0.2086 |
| PG-ST | non-Bank | 0/24 | 0/24 | 0.0641 |
| Fixed | Bank | 4/24 | 4/24 | 0.2319 |
| Fixed | non-Bank | 3/24 | 3/24 | 0.1270 |
| C | Bank | 4/24 | 4/24 | 0.2222 |
| C | non-Bank | 4/24 | 4/24 | 0.1208 |
| Per-Gaussian-MLP | Bank | 8/24 | 8/24 | 0.3214 |
| Per-Gaussian-MLP | non-Bank | 10/24 | 10/24 | 0.2692 |

每个scene/ratio/seed、每项指标的paired mean/median/p10/p90/improved-view fraction完整保存在paired_statistics.csv，未把不同view池混合作为独立重复。

| arm | non-Bank PSNR两seed绝对差，中位数/最大值(dB) |
|---|---:|
| Grid-MLP | 0.199962 / 1.094040 |
| Independent-Cell | 0.106154 / 0.469808 |
| Per-Gaussian-MLP | 0.092504 / 0.798510 |
| PG-ST | 0.043002 / 0.267392 |

**区域预算移动必须与质量同时评估**

| scene | ratio | seed | Grid changed cells | increase/decrease | transferred points | max abs delta_k | ΔBank photo | control transferred |
|---|---:|---:|---:|---|---:|---:|---:|---:|
| fr1_room | 1% | 2026 | 161 | 85/76 | 117 | 16 | +0.00378490 | 101 |
| fr1_room | 1% | 2027 | 94 | 48/46 | 61 | 10 | -0.00139109 | 96 |
| fr1_room | 5% | 2026 | 47 | 24/23 | 25 | 2 | -0.00011576 | 182 |
| fr1_room | 5% | 2027 | 205 | 99/106 | 154 | 6 | +0.00045992 | 186 |
| fr1_room | 10% | 2026 | 118 | 59/59 | 69 | 4 | +0.00001671 | 213 |
| fr1_room | 10% | 2027 | 176 | 90/86 | 104 | 4 | +0.00005727 | 227 |
| fr1_room | 15% | 2026 | 213 | 103/110 | 150 | 6 | +0.00004035 | 278 |
| fr1_room | 15% | 2027 | 200 | 97/103 | 137 | 6 | -0.00003840 | 285 |
| fr1_xyz | 1% | 2026 | 33 | 15/18 | 18 | 3 | +0.03265468 | 45 |
| fr1_xyz | 1% | 2027 | 82 | 45/37 | 51 | 6 | +0.01865226 | 41 |
| fr1_xyz | 5% | 2026 | 209 | 104/105 | 134 | 4 | +0.00727074 | 144 |
| fr1_xyz | 5% | 2027 | 247 | 122/125 | 172 | 6 | +0.00916125 | 149 |
| fr1_xyz | 10% | 2026 | 314 | 137/177 | 246 | 8 | +0.00339456 | 216 |
| fr1_xyz | 10% | 2027 | 376 | 159/217 | 325 | 9 | +0.00596510 | 221 |
| fr1_xyz | 15% | 2026 | 430 | 177/253 | 376 | 10 | +0.00357810 | 272 |
| fr1_xyz | 15% | 2027 | 321 | 140/181 | 235 | 8 | +0.00226136 | 271 |
| fr2_xyz | 1% | 2026 | 188 | 97/91 | 119 | 10 | +0.01217858 | 73 |
| fr2_xyz | 1% | 2027 | 139 | 75/64 | 86 | 6 | +0.01097932 | 70 |
| fr2_xyz | 5% | 2026 | 205 | 78/127 | 150 | 7 | +0.00120018 | 205 |
| fr2_xyz | 5% | 2027 | 89 | 52/37 | 53 | 3 | +0.00058321 | 170 |
| fr2_xyz | 10% | 2026 | 241 | 86/155 | 176 | 9 | +0.00304047 | 261 |
| fr2_xyz | 10% | 2027 | 36 | 18/18 | 19 | 2 | +0.00009738 | 275 |
| fr2_xyz | 15% | 2026 | 317 | 100/217 | 274 | 15 | +0.00272183 | 382 |
| fr2_xyz | 15% | 2027 | 162 | 63/99 | 116 | 6 | +0.00038264 | 431 |

每个模型均满足sum(delta_k)=0；sum(positive delta_k)=-sum(negative delta_k)。intra-cell排序保持使得transferred points等于相对Raw新增Gaussian数。完整per-cell值、系数、occupancy、direct选中事件与final/Raw count见cell_budget_redistribution.csv；histogram见cell_budget_histogram.csv。非零coefficient或大量changed cells本身不作为成功判据。

**occupancy与gradient空间分布**

| arm | scene | final corr(occupancy, a_C) min/max | final corr(occupancy, EMA) min/max | final top10% EMA mass share min/max |
|---|---|---:|---:|---:|
| Grid-MLP | fr1_room | -0.0336/+0.0663 | +0.2391/+0.7699 | 0.5480/0.8487 |
| Grid-MLP | fr1_xyz | +0.1272/+0.2394 | +0.1103/+0.4558 | 0.5754/0.8373 |
| Grid-MLP | fr2_xyz | -0.1924/+0.1867 | +0.2214/+0.6598 | 0.6357/0.8814 |
| Independent-Cell | fr1_room | +0.0450/+0.1703 | +0.2114/+0.7605 | 0.5478/0.8669 |
| Independent-Cell | fr1_xyz | -0.0586/+0.1772 | +0.0433/+0.5179 | 0.5412/0.8258 |
| Independent-Cell | fr2_xyz | -0.0625/+0.1218 | +0.1491/+0.6827 | 0.6222/0.8973 |

gradient统计保留raw cell output r_C的|gradient|与EMA(beta=.95)，每100step及step250记录，含mean/std/p50/p90/p99和top10% mass share。这些数据仅用于描述空间集中性，没有据此选择threshold或split。相关性只作posthoc描述，不能把cell coefficient直接称为occupancy predictor或认为集中就值得split。完整系数/exp(a_C)分布见cell_coefficient_statistics.csv。

**Correctness与证据边界**

- 所有Primary训练记录的MLP forward count=M，control为0；M/N为上述4%–9%左右，每个candidate恰有一个cell_id，按cell广播的correction exact相同。
- 全部weighted gauge和calibration检查通过；最大normalized gradient sum=2.99e-16，最大weighted centering误差=1.31e-15。CPU/CUDA加权gauge有限差分和原calibration regression均通过。
- 全部cell的Raw local order与final local order保持一致，violated cell count=0；initial full ranking及四ratio Top-K exact Raw。
- Native每步验证共享系数/gauge/intra-cell顺序，定期验证Gaussian/TF/Master/grid tensor exact冻结；Python从保存MLP权重独立forward复核cell输出，并独立重算最终排序和预算转移。
- 原始五组基线直接复用已有冻结质量文件，文件SHA核对；Raw的1420次Bank/non-Bank view render重新执行并逐view四指标exact，随后新48模型5680次view render。
- 训练不读取C/Fixed/PG coefficient或non-Bank quality；LR选择只使用预指定Bank场景。全部新集合在quality render前锁定SHA。
- 相对旧Gaussian-input MLP，本轮同时改变了forward对象、cell tying和normalization定义；不能把质量差异单独归因其中某一项。Independent-Cell control用于帮助区分fixed regional tying和MLP sharing，但不同参数化的优化条件仍可能不同。

**输出文件**

- [grid_metadata.csv](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/grid_mlp_implicit_si_20261020/grid_metadata.csv)、[cell_mapping.npz](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/grid_mlp_implicit_si_20261020/cell_mapping.npz)、[occupancy_statistics.csv](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/grid_mlp_implicit_si_20261020/occupancy_statistics.csv)。
- [quality_summary.csv](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/grid_mlp_implicit_si_20261020/quality_summary.csv)、[quality_per_view.csv](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/grid_mlp_implicit_si_20261020/quality_per_view.csv)、[paired_statistics.csv](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/grid_mlp_implicit_si_20261020/paired_statistics.csv)、[independent_cell_quality.csv](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/grid_mlp_implicit_si_20261020/independent_cell_quality.csv)。
- [training_curves.csv](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/grid_mlp_implicit_si_20261020/training_curves.csv)、[hard_training_dynamics.csv](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/grid_mlp_implicit_si_20261020/hard_training_dynamics.csv)、[cell_coefficient_statistics.csv](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/grid_mlp_implicit_si_20261020/cell_coefficient_statistics.csv)。
- [cell_budget_redistribution.csv](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/grid_mlp_implicit_si_20261020/cell_budget_redistribution.csv)、[cell_gradient_statistics.csv](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/grid_mlp_implicit_si_20261020/cell_gradient_statistics.csv)、[selected_set_overlap.csv](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/grid_mlp_implicit_si_20261020/selected_set_overlap.csv)、[raw_rank_promotion.csv](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/grid_mlp_implicit_si_20261020/raw_rank_promotion.csv)。
- [correctness.json](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/grid_mlp_implicit_si_20261020/correctness.json)、[audit_summary.json](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/grid_mlp_implicit_si_20261020/audit_summary.json)、[质量对照图](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/grid_mlp_implicit_si_20261020/quality_deltas.png)。
- 每scene保存key/center/occupancy/mapping/normalization/h；每模型保存step0/250/1000/2000权重或独立scalar、raw output、cell coefficient、Gaussian broadcast、gradient EMA、final score/ranking/selected set。

完成后停止：未实现adaptive split、threshold、child机制，未改grid size/normalization/network/feature/loss；未修改production Selector，未stage、commit、push。
