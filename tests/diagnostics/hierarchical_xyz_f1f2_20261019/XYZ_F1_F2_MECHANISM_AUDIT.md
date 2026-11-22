# fr1_xyz / fr2_xyz：F1/F2 Mechanism Audit

## 结论

**两scene均不支持用F1或F2替换当前C。** F1没有稳定降低统一Fixed-h8重复度，F2的depth scheduling在极低budget的收益依赖场景，且带来高档损失。F3的已有跨场景反例继续成立。

| scene | F1 | F2 | F3 | 本轮建议 |
|---|---|---|---|---|
| fr1_xyz | NOT SUPPORTED | MIXED | NOT SUPPORTED | KEEP CURRENT C |
| fr2_xyz | NOT SUPPORTED | NOT SUPPORTED | NOT SUPPORTED | KEEP CURRENT C |

F2的MIXED表示低档收益gate通过而高档保护gate失败；不能解释为支持整体替换。这里KEEP CURRENT C仅指保留对照实现，不表示C全ratio质量GO或production GO。

## 1. 冻结协议与基线复现

复用两个scene上一轮的Master、TF及AABB、实际Bank contribution、Raw SI、valid/observed、geometry、tree与固定non-Bank归档评价视角；未重算Raw贡献，未更新Bank、TF、Master或Selector。所有GT仍来自用户指定的TUM_RGBD目录。

| scene | checkpoint | N total/candidate | Bank/eval views | K 1/5/10/15% |
|---|---|---|---|---|
| fr1_xyz | 9900_shutdown | 23733/23643 | 18/54 | 236/1182/2364/3546 |
| fr2_xyz | 37900_shutdown | 44142/43059 | 14/39 | 431/2153/4306/6459 |

A/B/C/F3先重建完整排序并真实重渲染所有ratio/view：两scene逐视角和aggregate四指标最大差均为0，C的输出binary和tree CSV逐字一致，F3完整ranking和priority与旧实验exact一致。通过后才生成新F1/F2。

本轮baseline重复render 1488次，新F1/F2 render 744次，共 **2232次真实render**。evaluation views是参与过原SLAM的fixed non-Bank archived evaluation views，不称unseen或独立测试集。

## 2. 算法定义与原预注册

- C：final leaf的前3代表真实Delta，其余为independent residual，全局score排序。
- F1：与C完全相同的final leaves，完整local conditional marginal sequence；前三代表与C exact一致，global head-only merge保留predecessor order。
- F2：相同冻结tree，active-frontier progressive unlock，每node最多3个新rep，child继承路径ancestor coverage，final leaf仍用固定independent residual。
- F3：F2的final leaf继续marginal tail，作为已有组合对照重新exact复现。

target=2048、h搜索、origin=(0,0,0)、3rep/node、Q<0.80 split、max L3、scalar geometry κ、maturity=1全部不变。F2/F3的ordinal score只表达ranking，不解释物理mass。

| 机制 | 沿用判据 |
|---|---|
| F1 | 10/15% mean redundancy各下降≥10%，κ>.9 pairs不增加；两档平均ΔPSNR≥−0.03 dB |
| F2 | 1% ΔPSNR≥+0.05 dB；5/10/15%各ΔPSNR≥−0.03 dB |
| F3 | 四档mean ΔPSNR≥0，1%>0，10/15% mean redundancy<C |

本轮补做实验时已知C/F3旧结果，但未知F1/F2质量；判据逐字沿用fr1_room原预注册，没有重设阈值。各scene的preregistration.json在新render前写出，并在结束时验证SHA未变。

## 3. PSNR主表与机制分离

| scene | arm | 1% ΔC | 5% ΔC | 10% ΔC | 15% ΔC |
|---|---|---|---|---|---|
| fr1_xyz | F1 Tail | +0.000000 | -0.006025 | -0.021000 | -0.028425 |
| fr1_xyz | F2 Progressive | +0.250820 | -0.114585 | -0.119675 | -0.040932 |
| fr1_xyz | F3 Combined | +0.250820 | -0.110466 | -0.124511 | -0.059770 |
| fr2_xyz | F1 Tail | -0.074099 | -0.011100 | +0.021465 | +0.012406 |
| fr2_xyz | F2 Progressive | -0.090470 | -0.057788 | -0.041529 | +0.028432 |
| fr2_xyz | F3 Combined | -0.157235 | -0.075359 | -0.038847 | +0.023599 |

F1−C隔离tail替换；F2−C隔离progressive；F3−F2检查在progressive下tail的附加作用；F3−F1检查tail存在时progressive的作用。interaction=F3−F2−F1+C仅描述非加性，不用于选参。

## fr1_xyz

### 六arm绝对PSNR

| ratio | K | A Raw | B FixedMarginal | C Current | F1 Tail | F2 Progressive | F3 Combined |
|---|---|---|---|---|---|---|---|
| 1% | 236 | 13.020670 | 14.584813 | 14.206011 | 14.206011 | 14.456831 | 14.456831 |
| 5% | 1182 | 17.814622 | 18.777373 | 18.822068 | 18.816043 | 18.707482 | 18.711601 |
| 10% | 2364 | 19.812102 | 20.261198 | 20.301973 | 20.280973 | 20.182299 | 20.177463 |
| 15% | 3546 | 20.890139 | 20.817547 | 20.900224 | 20.871800 | 20.859292 | 20.840454 |

### 新机制四指标差值

| comparison | ratio | ΔPSNR ↑ | ΔSSIM ↑ | ΔLPIPS ↓ | Δphoto ↓ |
|---|---|---|---|---|---|
| F1−C | 1% | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| F1−C | 5% | -0.006025 | -0.000145 | -0.000387 | +0.000045 |
| F1−C | 10% | -0.021000 | -0.000107 | -0.000296 | +0.000077 |
| F1−C | 15% | -0.028425 | -0.000466 | +0.000283 | +0.000259 |
| F2−C | 1% | +0.250820 | -0.000114 | -0.002688 | -0.001319 |
| F2−C | 5% | -0.114585 | -0.001568 | -0.001382 | +0.000753 |
| F2−C | 10% | -0.119675 | -0.001095 | -0.000395 | +0.000687 |
| F2−C | 15% | -0.040932 | -0.000399 | -0.002631 | +0.000213 |
| F3−F2 | 1% | +0.000000 | +0.000000 | +0.000000 | +0.000000 |
| F3−F2 | 5% | +0.004119 | +0.000258 | -0.000640 | -0.000108 |
| F3−F2 | 10% | -0.004836 | -0.000060 | -0.000438 | +0.000025 |
| F3−F2 | 15% | -0.018838 | -0.000349 | +0.000009 | +0.000192 |

### Paired per-view

| comparison | ratio | metric | mean | median | p10 | p90 | improved-view fraction |
|---|---|---|---|---|---|---|---|
| F1_MarginalTail_minus_C_Hierarchical | 1% | psnr | +0.000000 | +0.000000 | +0.000000 | +0.000000 | 0.00% |
| F1_MarginalTail_minus_C_Hierarchical | 1% | ssim | +0.000000 | +0.000000 | +0.000000 | +0.000000 | 0.00% |
| F1_MarginalTail_minus_C_Hierarchical | 1% | lpips | +0.000000 | +0.000000 | +0.000000 | +0.000000 | 0.00% |
| F1_MarginalTail_minus_C_Hierarchical | 1% | photo | +0.000000 | +0.000000 | +0.000000 | +0.000000 | 0.00% |
| F1_MarginalTail_minus_C_Hierarchical | 5% | psnr | -0.006025 | -0.005597 | -0.023559 | +0.008939 | 29.63% |
| F1_MarginalTail_minus_C_Hierarchical | 5% | ssim | -0.000145 | -0.000168 | -0.000940 | +0.000534 | 40.74% |
| F1_MarginalTail_minus_C_Hierarchical | 5% | lpips | -0.000387 | -0.000631 | -0.001835 | +0.001248 | 57.41% |
| F1_MarginalTail_minus_C_Hierarchical | 5% | photo | +0.000045 | +0.000047 | -0.000199 | +0.000311 | 40.74% |
| F1_MarginalTail_minus_C_Hierarchical | 10% | psnr | -0.021000 | -0.020542 | -0.055200 | +0.008231 | 24.07% |
| F1_MarginalTail_minus_C_Hierarchical | 10% | ssim | -0.000107 | -0.000054 | -0.000578 | +0.000341 | 40.74% |
| F1_MarginalTail_minus_C_Hierarchical | 10% | lpips | -0.000296 | -0.000232 | -0.001168 | +0.000409 | 66.67% |
| F1_MarginalTail_minus_C_Hierarchical | 10% | photo | +0.000077 | +0.000041 | -0.000105 | +0.000324 | 35.19% |
| F1_MarginalTail_minus_C_Hierarchical | 15% | psnr | -0.028425 | -0.026304 | -0.075137 | +0.014921 | 27.78% |
| F1_MarginalTail_minus_C_Hierarchical | 15% | ssim | -0.000466 | -0.000513 | -0.001296 | +0.000544 | 22.22% |
| F1_MarginalTail_minus_C_Hierarchical | 15% | lpips | +0.000283 | +0.000269 | -0.000483 | +0.001242 | 37.04% |
| F1_MarginalTail_minus_C_Hierarchical | 15% | photo | +0.000259 | +0.000217 | -0.000055 | +0.000598 | 12.96% |
| F2_ProgressiveResidual_minus_C_Hierarchical | 1% | psnr | +0.250820 | +0.240795 | +0.095179 | +0.445554 | 94.44% |
| F2_ProgressiveResidual_minus_C_Hierarchical | 1% | ssim | -0.000114 | +0.000055 | -0.001834 | +0.001664 | 50.00% |
| F2_ProgressiveResidual_minus_C_Hierarchical | 1% | lpips | -0.002688 | -0.003118 | -0.006020 | +0.001188 | 83.33% |
| F2_ProgressiveResidual_minus_C_Hierarchical | 1% | photo | -0.001319 | -0.001514 | -0.002882 | +0.000071 | 88.89% |
| F2_ProgressiveResidual_minus_C_Hierarchical | 5% | psnr | -0.114585 | -0.098683 | -0.210079 | -0.049772 | 1.85% |
| F2_ProgressiveResidual_minus_C_Hierarchical | 5% | ssim | -0.001568 | -0.002057 | -0.003226 | +0.000407 | 14.81% |
| F2_ProgressiveResidual_minus_C_Hierarchical | 5% | lpips | -0.001382 | -0.001331 | -0.004302 | +0.002038 | 62.96% |
| F2_ProgressiveResidual_minus_C_Hierarchical | 5% | photo | +0.000753 | +0.000811 | +0.000198 | +0.001341 | 5.56% |
| F2_ProgressiveResidual_minus_C_Hierarchical | 10% | psnr | -0.119675 | -0.124355 | -0.181353 | -0.042168 | 3.70% |
| F2_ProgressiveResidual_minus_C_Hierarchical | 10% | ssim | -0.001095 | -0.001314 | -0.002628 | +0.000560 | 22.22% |
| F2_ProgressiveResidual_minus_C_Hierarchical | 10% | lpips | -0.000395 | -0.000590 | -0.002609 | +0.001708 | 57.41% |
| F2_ProgressiveResidual_minus_C_Hierarchical | 10% | photo | +0.000687 | +0.000761 | -0.000002 | +0.001212 | 11.11% |
| F2_ProgressiveResidual_minus_C_Hierarchical | 15% | psnr | -0.040932 | -0.035541 | -0.104763 | +0.011943 | 14.81% |
| F2_ProgressiveResidual_minus_C_Hierarchical | 15% | ssim | -0.000399 | -0.000497 | -0.001375 | +0.000712 | 25.93% |
| F2_ProgressiveResidual_minus_C_Hierarchical | 15% | lpips | -0.002631 | -0.002999 | -0.004642 | -0.000087 | 90.74% |
| F2_ProgressiveResidual_minus_C_Hierarchical | 15% | photo | +0.000213 | +0.000207 | -0.000080 | +0.000610 | 24.07% |

### 统一重复度与F1 gate

| arm | ratio | mean | median | p90 | κ>.8 | κ>.9 | κ>.95 |
|---|---|---|---|---|---|---|---|
| C Current | 1% | 0.134665 | 0.000000 | 0.630343 | 10 | 4 | 1 |
| C Current | 5% | 0.226700 | 0.006920 | 0.737619 | 49 | 18 | 10 |
| C Current | 10% | 0.280846 | 0.091284 | 0.820299 | 169 | 64 | 31 |
| C Current | 15% | 0.296314 | 0.140379 | 0.822272 | 267 | 104 | 51 |
| F1 Tail | 1% | 0.134665 | 0.000000 | 0.630343 | 10 | 4 | 1 |
| F1 Tail | 5% | 0.228424 | 0.007120 | 0.741037 | 50 | 19 | 11 |
| F1 Tail | 10% | 0.280551 | 0.092421 | 0.817809 | 161 | 58 | 26 |
| F1 Tail | 15% | 0.297527 | 0.144722 | 0.820018 | 257 | 95 | 40 |
| F2 Progressive | 1% | 0.126496 | 0.000000 | 0.593684 | 9 | 4 | 1 |
| F2 Progressive | 5% | 0.213034 | 0.002376 | 0.720190 | 48 | 18 | 9 |
| F2 Progressive | 10% | 0.266714 | 0.073130 | 0.808474 | 160 | 62 | 28 |
| F2 Progressive | 15% | 0.278308 | 0.109470 | 0.806019 | 242 | 93 | 45 |
| F3 Combined | 1% | 0.126496 | 0.000000 | 0.593684 | 9 | 4 | 1 |
| F3 Combined | 5% | 0.214508 | 0.002408 | 0.716178 | 48 | 19 | 10 |
| F3 Combined | 10% | 0.265370 | 0.073130 | 0.803401 | 151 | 55 | 23 |
| F3 Combined | 15% | 0.279021 | 0.110627 | 0.805505 | 234 | 85 | 36 |

| arm | 10% mean redundancy下降 | 15% mean redundancy下降 | 高档平均ΔPSNR | gate |
|---|---|---|---|---|
| F1 | +0.11% | -0.41% | -0.024712 | NOT SUPPORTED |

重复度完全复用统一Fixed-h8 cell内geometry诊断：逐selected row最大off-diagonal κ，singleton0；pair为unordered。不因F1使用adaptive leaves而切换诊断metric。

### Progressive composition与空间budget

| arm | ratio | internal reps | final reps | tail/residual | emit L1/L2/L3 |
|---|---|---|---|---|---|
| F2 Progressive | 1% | 50 | 186 | 0 | 236/0/0 |
| F2 Progressive | 5% | 426 | 733 | 23 | 1059/121/2 |
| F2 Progressive | 10% | 856 | 1377 | 131 | 1779/522/63 |
| F2 Progressive | 15% | 1231 | 2049 | 266 | 2314/1015/217 |
| F3 Combined | 1% | 50 | 186 | 0 | 236/0/0 |
| F3 Combined | 5% | 426 | 728 | 28 | 1060/120/2 |
| F3 Combined | 10% | 854 | 1372 | 138 | 1781/520/63 |
| F3 Combined | 15% | 1225 | 2033 | 288 | 2316/1010/220 |

| arm | ratio | hit roots | hit leaves | hit-root Raw mass | hit-leaf Raw mass |
|---|---|---|---|---|---|
| C Current | 1% | 206 | 215 | 61.79% | 55.63% |
| C Current | 5% | 665 | 910 | 91.92% | 85.15% |
| C Current | 10% | 978 | 1559 | 97.08% | 92.95% |
| C Current | 15% | 1181 | 2131 | 98.63% | 96.21% |
| F1 Tail | 1% | 206 | 215 | 61.79% | 55.63% |
| F1 Tail | 5% | 665 | 908 | 91.92% | 85.10% |
| F1 Tail | 10% | 972 | 1550 | 97.01% | 92.89% |
| F1 Tail | 15% | 1176 | 2119 | 98.61% | 96.16% |
| F2 Progressive | 1% | 210 | 216 | 62.55% | 55.50% |
| F2 Progressive | 5% | 680 | 894 | 92.45% | 84.23% |
| F2 Progressive | 10% | 1000 | 1543 | 97.31% | 92.44% |
| F2 Progressive | 15% | 1201 | 2120 | 98.74% | 96.00% |

### 2×2机制对照（PSNR）

| ratio | F1−C | F2−C | F3−F2 | F3−F1 | interaction |
|---|---|---|---|---|---|
| 1% | +0.000000 | +0.250820 | +0.000000 | +0.250820 | +0.000000 |
| 5% | -0.006025 | -0.114585 | +0.004119 | -0.104441 | +0.010144 |
| 10% | -0.021000 | -0.119675 | -0.004836 | -0.103511 | +0.016164 |
| 15% | -0.028425 | -0.040932 | -0.018838 | -0.031346 | +0.009587 |

## fr2_xyz

### 六arm绝对PSNR

| ratio | K | A Raw | B FixedMarginal | C Current | F1 Tail | F2 Progressive | F3 Combined |
|---|---|---|---|---|---|---|---|
| 1% | 431 | 16.234008 | 16.329698 | 16.726775 | 16.652676 | 16.636306 | 16.569540 |
| 5% | 2153 | 19.883452 | 20.254137 | 20.232834 | 20.221734 | 20.175046 | 20.157475 |
| 10% | 4306 | 21.238710 | 21.032306 | 21.065654 | 21.087119 | 21.024125 | 21.026807 |
| 15% | 6459 | 21.479951 | 21.452410 | 21.406546 | 21.418952 | 21.434978 | 21.430145 |

### 新机制四指标差值

| comparison | ratio | ΔPSNR ↑ | ΔSSIM ↑ | ΔLPIPS ↓ | Δphoto ↓ |
|---|---|---|---|---|---|
| F1−C | 1% | -0.074099 | -0.001583 | +0.001180 | +0.000979 |
| F1−C | 5% | -0.011100 | +0.000130 | +0.000267 | +0.000102 |
| F1−C | 10% | +0.021465 | +0.000705 | -0.000331 | -0.000198 |
| F1−C | 15% | +0.012406 | +0.000531 | -0.001967 | -0.000028 |
| F2−C | 1% | -0.090470 | -0.001660 | +0.000570 | +0.001365 |
| F2−C | 5% | -0.057788 | -0.001590 | +0.001866 | +0.000591 |
| F2−C | 10% | -0.041529 | +0.000084 | -0.000336 | +0.000112 |
| F2−C | 15% | +0.028432 | +0.000402 | -0.001474 | -0.000129 |
| F3−F2 | 1% | -0.066765 | -0.001520 | +0.001199 | +0.000906 |
| F3−F2 | 5% | -0.017571 | +0.000048 | -0.000086 | +0.000181 |
| F3−F2 | 10% | +0.002682 | -0.000034 | +0.000534 | +0.000026 |
| F3−F2 | 15% | -0.004833 | +0.000055 | -0.000398 | +0.000086 |

### Paired per-view

| comparison | ratio | metric | mean | median | p10 | p90 | improved-view fraction |
|---|---|---|---|---|---|---|---|
| F1_MarginalTail_minus_C_Hierarchical | 1% | psnr | -0.074099 | -0.072531 | -0.156403 | -0.008916 | 0.00% |
| F1_MarginalTail_minus_C_Hierarchical | 1% | ssim | -0.001583 | -0.001554 | -0.003502 | -0.000077 | 7.69% |
| F1_MarginalTail_minus_C_Hierarchical | 1% | lpips | +0.001180 | +0.001247 | -0.000165 | +0.002689 | 23.08% |
| F1_MarginalTail_minus_C_Hierarchical | 1% | photo | +0.000979 | +0.000920 | +0.000170 | +0.001899 | 0.00% |
| F1_MarginalTail_minus_C_Hierarchical | 5% | psnr | -0.011100 | -0.007477 | -0.051182 | +0.024605 | 43.59% |
| F1_MarginalTail_minus_C_Hierarchical | 5% | ssim | +0.000130 | -0.000015 | -0.000436 | +0.000771 | 46.15% |
| F1_MarginalTail_minus_C_Hierarchical | 5% | lpips | +0.000267 | +0.000350 | -0.001161 | +0.001903 | 28.21% |
| F1_MarginalTail_minus_C_Hierarchical | 5% | photo | +0.000102 | +0.000076 | -0.000513 | +0.000669 | 41.03% |
| F1_MarginalTail_minus_C_Hierarchical | 10% | psnr | +0.021465 | +0.021162 | -0.001519 | +0.044497 | 87.18% |
| F1_MarginalTail_minus_C_Hierarchical | 10% | ssim | +0.000705 | +0.000725 | -0.000254 | +0.001500 | 84.62% |
| F1_MarginalTail_minus_C_Hierarchical | 10% | lpips | -0.000331 | -0.000097 | -0.001349 | +0.000687 | 53.85% |
| F1_MarginalTail_minus_C_Hierarchical | 10% | photo | -0.000198 | -0.000226 | -0.000404 | +0.000054 | 82.05% |
| F1_MarginalTail_minus_C_Hierarchical | 15% | psnr | +0.012406 | +0.009506 | -0.023846 | +0.064291 | 69.23% |
| F1_MarginalTail_minus_C_Hierarchical | 15% | ssim | +0.000531 | +0.000673 | -0.000771 | +0.001497 | 71.79% |
| F1_MarginalTail_minus_C_Hierarchical | 15% | lpips | -0.001967 | -0.002100 | -0.003968 | +0.000573 | 79.49% |
| F1_MarginalTail_minus_C_Hierarchical | 15% | photo | -0.000028 | -0.000092 | -0.000396 | +0.000472 | 56.41% |
| F2_ProgressiveResidual_minus_C_Hierarchical | 1% | psnr | -0.090470 | -0.069544 | -0.233854 | -0.002394 | 10.26% |
| F2_ProgressiveResidual_minus_C_Hierarchical | 1% | ssim | -0.001660 | -0.001476 | -0.003167 | -0.000418 | 2.56% |
| F2_ProgressiveResidual_minus_C_Hierarchical | 1% | lpips | +0.000570 | +0.000127 | -0.003660 | +0.006415 | 46.15% |
| F2_ProgressiveResidual_minus_C_Hierarchical | 1% | photo | +0.001365 | +0.001067 | +0.000318 | +0.003017 | 10.26% |
| F2_ProgressiveResidual_minus_C_Hierarchical | 5% | psnr | -0.057788 | -0.073301 | -0.129768 | +0.039856 | 17.95% |
| F2_ProgressiveResidual_minus_C_Hierarchical | 5% | ssim | -0.001590 | -0.001622 | -0.003187 | -0.000134 | 10.26% |
| F2_ProgressiveResidual_minus_C_Hierarchical | 5% | lpips | +0.001866 | +0.001712 | -0.002705 | +0.006827 | 25.64% |
| F2_ProgressiveResidual_minus_C_Hierarchical | 5% | photo | +0.000591 | +0.000787 | -0.000402 | +0.001314 | 20.51% |
| F2_ProgressiveResidual_minus_C_Hierarchical | 10% | psnr | -0.041529 | -0.029615 | -0.104549 | +0.000465 | 10.26% |
| F2_ProgressiveResidual_minus_C_Hierarchical | 10% | ssim | +0.000084 | +0.000130 | -0.000472 | +0.000693 | 64.10% |
| F2_ProgressiveResidual_minus_C_Hierarchical | 10% | lpips | -0.000336 | -0.000111 | -0.001993 | +0.000885 | 64.10% |
| F2_ProgressiveResidual_minus_C_Hierarchical | 10% | photo | +0.000112 | +0.000075 | -0.000157 | +0.000426 | 30.77% |
| F2_ProgressiveResidual_minus_C_Hierarchical | 15% | psnr | +0.028432 | +0.023012 | -0.008239 | +0.072414 | 82.05% |
| F2_ProgressiveResidual_minus_C_Hierarchical | 15% | ssim | +0.000402 | +0.000151 | -0.000366 | +0.001422 | 66.67% |
| F2_ProgressiveResidual_minus_C_Hierarchical | 15% | lpips | -0.001474 | -0.001511 | -0.003241 | +0.000426 | 84.62% |
| F2_ProgressiveResidual_minus_C_Hierarchical | 15% | photo | -0.000129 | -0.000075 | -0.000445 | +0.000156 | 64.10% |

### 统一重复度与F1 gate

| arm | ratio | mean | median | p90 | κ>.8 | κ>.9 | κ>.95 |
|---|---|---|---|---|---|---|---|
| C Current | 1% | 0.010920 | 0.000000 | 0.000000 | 1 | 0 | 0 |
| C Current | 5% | 0.080929 | 0.000000 | 0.361809 | 50 | 26 | 18 |
| C Current | 10% | 0.129833 | 0.000000 | 0.660157 | 164 | 91 | 57 |
| C Current | 15% | 0.159680 | 0.000000 | 0.748998 | 363 | 194 | 111 |
| F1 Tail | 1% | 0.010920 | 0.000000 | 0.000000 | 1 | 0 | 0 |
| F1 Tail | 5% | 0.079867 | 0.000000 | 0.361809 | 43 | 18 | 9 |
| F1 Tail | 10% | 0.125411 | 0.000000 | 0.651005 | 141 | 69 | 30 |
| F1 Tail | 15% | 0.148194 | 0.000000 | 0.710567 | 277 | 121 | 54 |
| F2 Progressive | 1% | 0.002571 | 0.000000 | 0.000000 | 0 | 0 | 0 |
| F2 Progressive | 5% | 0.072383 | 0.000000 | 0.285682 | 44 | 19 | 10 |
| F2 Progressive | 10% | 0.119804 | 0.000000 | 0.618754 | 139 | 69 | 40 |
| F2 Progressive | 15% | 0.145847 | 0.000000 | 0.710175 | 287 | 139 | 77 |
| F3 Combined | 1% | 0.002571 | 0.000000 | 0.000000 | 0 | 0 | 0 |
| F3 Combined | 5% | 0.066366 | 0.000000 | 0.211796 | 33 | 11 | 4 |
| F3 Combined | 10% | 0.115075 | 0.000000 | 0.576013 | 120 | 54 | 22 |
| F3 Combined | 15% | 0.140384 | 0.000000 | 0.687148 | 249 | 103 | 41 |

| arm | 10% mean redundancy下降 | 15% mean redundancy下降 | 高档平均ΔPSNR | gate |
|---|---|---|---|---|
| F1 | +3.41% | +7.19% | +0.016935 | NOT SUPPORTED |

重复度完全复用统一Fixed-h8 cell内geometry诊断：逐selected row最大off-diagonal κ，singleton0；pair为unordered。不因F1使用adaptive leaves而切换诊断metric。

### Progressive composition与空间budget

| arm | ratio | internal reps | final reps | tail/residual | emit L1/L2/L3 |
|---|---|---|---|---|---|
| F2 Progressive | 1% | 176 | 250 | 5 | 398/32/1 |
| F2 Progressive | 5% | 888 | 1123 | 142 | 1311/667/175 |
| F2 Progressive | 10% | 1443 | 2330 | 533 | 1971/1375/960 |
| F2 Progressive | 15% | 1824 | 3404 | 1231 | 2393/1968/2098 |
| F3 Combined | 1% | 175 | 249 | 7 | 398/32/1 |
| F3 Combined | 5% | 894 | 1132 | 127 | 1298/675/180 |
| F3 Combined | 10% | 1442 | 2327 | 537 | 1940/1367/999 |
| F3 Combined | 15% | 1821 | 3394 | 1244 | 2347/1966/2146 |

| arm | ratio | hit roots | hit leaves | hit-root Raw mass | hit-leaf Raw mass |
|---|---|---|---|---|---|
| C Current | 1% | 258 | 349 | 79.32% | 67.57% |
| C Current | 5% | 735 | 1374 | 97.29% | 91.86% |
| C Current | 10% | 1023 | 2169 | 99.24% | 96.97% |
| C Current | 15% | 1171 | 2747 | 99.63% | 98.56% |
| F1 Tail | 1% | 257 | 347 | 79.28% | 67.48% |
| F1 Tail | 5% | 735 | 1373 | 97.29% | 91.85% |
| F1 Tail | 10% | 1022 | 2166 | 99.24% | 96.96% |
| F1 Tail | 15% | 1171 | 2747 | 99.63% | 98.56% |
| F2 Progressive | 1% | 269 | 351 | 81.26% | 67.00% |
| F2 Progressive | 5% | 747 | 1360 | 97.47% | 91.35% |
| F2 Progressive | 10% | 1031 | 2159 | 99.29% | 96.77% |
| F2 Progressive | 15% | 1183 | 2736 | 99.65% | 98.46% |

### 2×2机制对照（PSNR）

| ratio | F1−C | F2−C | F3−F2 | F3−F1 | interaction |
|---|---|---|---|---|---|
| 1% | -0.074099 | -0.090470 | -0.066765 | -0.083136 | +0.007334 |
| 5% | -0.011100 | -0.057788 | -0.017571 | -0.064259 | -0.006470 |
| 10% | +0.021465 | -0.041529 | +0.002682 | -0.060311 | -0.018783 |
| 15% | +0.012406 | +0.028432 | -0.004833 | +0.011193 | -0.017239 |

## 4. 机制解释

**fr1_xyz：Progressive是5/10%回退的主要来源。** F1−C仅为−0.006025/−0.021000 dB，F2−C为−0.114585/−0.119675 dB。1% F1与C selected set完全相同，F2与F3也完全相同且全部emit来自L1；该档+0.250820 dB明确来自coarse scheduling。这个收益无法保住高档。

**fr2_xyz：两种机制都能引起极低budget损失。** 1% F1−C为−0.074099 dB，F2−C为−0.090470 dB；在F2上加入tail还损失−0.066765 dB。F1在10/15%有小幅PSNR增益，但重复度仅下降3.41%/7.19%，不达原门槛，且1%损失不可忽略。

fr2_xyz 1%中，F1相对C仅交换4个rows（Jaccard=0.981609），F3相对F2也仅交换4个rows，却分别导致上述−0.074099/−0.066765 dB。统一mean redundancy在该档不变。说明少量关键row的重建价值不能由这个几何重复度均值充分解释；这不是可以靠“重复度更低”忽略的反例。个别row的独立因果价值尚未做render-oracle验证。

F2在fr2_xyz 1%命中269个roots，C为258，且92.34%的emit在L1，仍然质量下降。因此更多root或更粗depth本身不是质量成功条件。

两scene的F1高档平均PSNR均满足不低于−0.03 dB，但重复度gate均失败；不能把NOT SUPPORTED误写成F1在每档都显著损害质量，也不能因为个别小幅改善就采用它。当前最明确的问题是depth scheduling的跨scene不稳定，tail的质量作用也依赖预算与场景。

## 5. Correctness、SHA与完整证据

两个scene均通过：同population、完整permutation、unique rows、same-K、nested prefixes；重复完整ranking/gain/emit/unlock trace exact；F1 first3 ID exact、gain数值容差一致、局部gain非负/非增、local sequence telescoping mass检查通过；C tree完全不变。

独立重放F1/F2/F3总计200106次row argmax/gain及全局head/frontier unlock，检查ancestor coverage和已emit排除；每scene最大gain误差0。没有silent approximation、priority二次sort、重复row或parent rep计入C。

| scene | hierarchy input SHA | tree SHA | preregistration SHA |
|---|---|---|---|
| fr1_xyz | 9518c1b79e2bd00f499fcbb81244c7a82be9cc10c3f0610ed745695a99a67fac | 7a296f65a278cdd202b67a0bae8a9c679f69989a4cb62f05b92ad2633c2a34eb | b0365b848d358504c92cb30676ba2c056c65ac3e5a5584722c05adaf56aafa02 |
| fr2_xyz | e782809b8e6395ef2357ee58745ffc1763f76f3cf540d875daca5b5bd78bbada | af4cd411a4bc42a38bd59392c8b33743c01632cd641f694e3c0d21582f3107c7 | 72c2abc92fa2f78edff7caa8f887b8ef0c7151888709c2d8296dd41293f30ee1 |

某些低ratio的C selected non-reps为0，相应均值/中位统计输出空值，不能写成NaN或伪造0；这项诊断呈现修正不改变ranking、render或机制结果，原warning日志保留。

每scene保存：完整六arm aggregate/per-view、paired差值、uniform redundancy、高相似pairs、overlap与逐row增删、composition、空间mass coverage、tail完整Delta sequence、tail Top-K rows、unlock trace、原始ranking/geometry/mask数组、SHA及所有验证日志。运行耗时保留在JSON，未参与任何判定。

## 6. 限制与停止

每scene一个已训练的frozen final map；评价视角参与原SLAM；未做online训练、Selector适配或ATE比较。Metric方向混合时逐项报告，不用geometry Q或redundancy替代真实质量。

本轮不调root/Q/rep/depth/h或任何progressive权重，不补做新机制设计，不修改production。**保留当前C作为对照；F1/F2没有获得两个XYZ场景一致的机制支持。** 未commit、未push。
