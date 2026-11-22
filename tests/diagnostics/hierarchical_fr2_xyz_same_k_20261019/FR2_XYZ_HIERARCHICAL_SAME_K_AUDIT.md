# fr2_xyz：A/B/C/F3 Frozen Same-K 审计

## 结论

**F3 本场景判定：NOT SUPPORTED；Strong improvement：NO。** F3−C的1/5/10/15% PSNR差依次为 **-0.157235 / -0.075359 / -0.038847 / +0.023599 dB**，四档平均 **-0.061960 dB**。F3相对Fixed h8平均 **+0.028854 dB（EXCEEDS，仅PSNR水平）**。
C相对Raw的PSNR主gate：**FAIL**；对应四档ΔPSNR：+0.492767 / +0.349382 / -0.173057 / -0.073405 dB。其余指标方向必须结合下表判断，不能把PSNR gate等同所有指标全面占优。

本轮只新增fr2_xyz离线结果，没有改变算法、参数或production开关，也没有训练或online运行。跨场景结论以各scene分别呈现，不用总体平均掩盖某个scene的回退。

## 1. 冻结数据与评价协议

- 数据集：`/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg2_xyz`；GT直接从用户指定目录读取。
- checkpoint：已有 `full/attempt_1/model/37900_shutdown/ply`，按最终shutdown选择，没有读质量挑checkpoint。
- Master **44142** rows；valid **44142**；observed/candidate **43059**。candidate=`valid AND observed`，effective maturity=1。
- 共53个归档KF，其中14个实际Bank views只用于Raw SI，其余39个全部作为 **fixed non-Bank archived evaluation views**。这些视角参与原SLAM，不称unseen/test-only/independent novel views。
- 没有既有fr2_xyz同协议贡献fixture；本轮用已归档Bank和原renderer导出一次完整frame contribution并冻结，四arm共用Raw SI=Bank contribution mean，不更新Bank。
- `view_bank_orb.json`的scene文本仍为过期标签`fr3_long_office_household`；本轮不使用标签选择数据。实际Bank IDs全属于同一checkpoint相机，所有GT图像来自明确的fr2_xyz路径，均保存SHA。
- 相机/FOV/GT转换、renderer、TF consumer和四项metric实现逐字复用上一轮。每个ratio所有arm共用同一TF参数与AABB，除了selected rows没有其它输入差别。
- production photo=`0.8*L1 + 0.2*(1-production fast_ssim)`；单独SSIM沿用TorchMetrics实现，不与photo里的fast_ssim混同。

| 输入 | 路径 | SHA256 |
|---|---|---|
| master | /home/prmi/zwj/mycartgs/resultsall/batch_20261018_tum_aligned/jobs/tum/rgbd/fr2_xyz/full/attempt_1/model/37900_shutdown/ply/point_cloud/iteration_37900/point_cloud.ply | f2a83521139547736804bf5be56e334c11ee51ec0dd5f3741f5c5116a83b9ac8 |
| tf | /home/prmi/zwj/mycartgs/resultsall/batch_20261018_tum_aligned/jobs/tum/rgbd/fr2_xyz/full/attempt_1/model/37900_shutdown/ply/transform_field.pt | 050069eb1a664db8a77ccb0d15e0e0fb86add45162df3e55c14deb6a8eef9eca |
| config | /home/prmi/zwj/mycartgs/resultsall/batch_20261018_tum_aligned/jobs/tum/rgbd/fr2_xyz/full/attempt_1/model/37900_shutdown/ply/selector_config.yaml | 6f047e8a3f101cfe73885e4c1689aaea861f47408191504221ae06527ec72f78 |
| cameras | /home/prmi/zwj/hierarchical-fr2-xyz-same-k-20261019/inputs/fr2_xyz/cameras.pt | e145e03f8c787663df2a17418240c8262696ffab6beb19e50ac0639d0c98057e |
| contributions.pt | /home/prmi/zwj/hierarchical-fr2-xyz-same-k-20261019/contributions.pt | fac1d4b74e4b41dd18fadb50db3facd591e70193e68275261842d384a305484d |
| ranking_inputs.npz | /home/prmi/zwj/hierarchical-fr2-xyz-same-k-20261019/ranking_inputs.npz | c8dc3c1bb52afe73fecebf56e70a8e8e723efddbb21ae12af63f4352d268e96c |
| hierarchy_input.bin | /home/prmi/zwj/hierarchical-fr2-xyz-same-k-20261019/hierarchy_input.bin | e782809b8e6395ef2357ee58745ffc1763f76f3cf540d875daca5b5bd78bbada |
| preregistration.json | /home/prmi/zwj/hierarchical-fr2-xyz-same-k-20261019/preregistration.json | 09d4fc88476394a64e24cd4bd161a311299106bef6ee21bd824e0e1f07ae5a6a |

完整原始输入SHA见 `frozen_sources.json`；相机archive SHA和GT图像SHA见 `camera_checks.json`。全部输入结束时重新核对。

## 2. Arms、参数与correctness

| Arm | 定义 |
|---|---|
| A | Raw SI全局降序，tie source row |
| B | Fixed h8 geometry-only完整local marginal greedy + predecessor-preserving merge |
| C | SparseHash当前3rep Delta + independent residual，全局score排序 |
| F3 | 同一冻结tree，ancestor-conditioned progressive coarse-to-fine unlock + final-leaf conditional marginal tail |

target occupied roots=2048，origin=(0,0,0)，h搜索不变，3 reps/node，Q<0.80 split，max L3，scalar-radius geometry κ，maturity=1。没有参数搜索、view/covariance、dedup或boundary correction。

F3最终ranking等于heap实际emit顺序，不按priority二次排序；tie为priority降序、root key+octree path字典序、source row升序。已emit ancestor row被排除，每row恰好一次。Ordinal score只表达order，不解释物理importance mass。

| ratio | K |
|---|---|
| 1% | 431 |
| 5% | 2153 |
| 10% | 4306 |
| 15% | 6459 |

- 四arm同candidate population、完整permutation、same-K与nested prefixes通过。
- B两次完整重算exact一致，逐cell predecessor order通过。
- C用原C++静态库构建两次，score、membership、h、tree、rep IDs/Delta/Q exact一致；全部节点first3 greedy和Q独立重算通过。
- F3完整order/gain/emit/unlock重复运行exact一致；synthetic测试通过；独立重放全部43059次argmax/gain及frontier/unlock，最大gain差0。
- C mass相对误差：0.000e+00。F3评价ranking，不宣称ordinal score守恒物理mass。
- 这是新scene，没有历史fr2_xyz同协议质量可做exact对照；本轮核对沿用数学/worker/metric的SHA，并在本scene做repeat和独立验证。

## 3. Sparse hierarchy结构

| 统计 | 值 |
|---|---|
| occupied_roots | 2048 |
| h_root | 0.1251487810923318 |
| d50 | 0.002020284574903986 |
| h | 0.01616227659923189 |
| final_leaves | 5090 |
| final_representatives | 11448 |
| weighted_leaf_Q | 0.9008886653003791 |

结构Q不参与质量gate。完整冻结tree/membership和内部provisional reps已保存。

## 4. Same-K真实render质量

### PSNR ↑

| ratio | K | A Raw | B FixedMarginal | C Current | F3 Progressive+Tail |
|---|---|---|---|---|---|
| 1% | 431 | 16.234008 | 16.329698 | 16.726775 | 16.569540 |
| 5% | 2153 | 19.883452 | 20.254137 | 20.232834 | 20.157475 |
| 10% | 4306 | 21.238710 | 21.032306 | 21.065654 | 21.026807 |
| 15% | 6459 | 21.479951 | 21.452410 | 21.406546 | 21.430145 |

### SSIM ↑

| ratio | K | A Raw | B FixedMarginal | C Current | F3 Progressive+Tail |
|---|---|---|---|---|---|
| 1% | 431 | 0.591295 | 0.589100 | 0.593566 | 0.590386 |
| 5% | 2153 | 0.657687 | 0.665472 | 0.665330 | 0.663788 |
| 10% | 4306 | 0.699281 | 0.695580 | 0.695924 | 0.695973 |
| 15% | 6459 | 0.711983 | 0.709923 | 0.709145 | 0.709602 |

### LPIPS ↓

| ratio | K | A Raw | B FixedMarginal | C Current | F3 Progressive+Tail |
|---|---|---|---|---|---|
| 1% | 431 | 0.722452 | 0.670712 | 0.669190 | 0.670960 |
| 5% | 2153 | 0.506864 | 0.469105 | 0.461077 | 0.462857 |
| 10% | 4306 | 0.406361 | 0.385843 | 0.378207 | 0.378405 |
| 15% | 6459 | 0.363814 | 0.343629 | 0.345040 | 0.343167 |

### Production photo ↓

| ratio | K | A Raw | B FixedMarginal | C Current | F3 Progressive+Tail |
|---|---|---|---|---|---|
| 1% | 431 | 0.159823 | 0.160234 | 0.156128 | 0.158399 |
| 5% | 2153 | 0.114430 | 0.111829 | 0.111950 | 0.112722 |
| 10% | 4306 | 0.099144 | 0.100980 | 0.101006 | 0.101144 |
| 15% | 6459 | 0.095278 | 0.096358 | 0.096823 | 0.096779 |

合计4 arms ×4 ratios ×39views = **624次真实render**。

### C − Raw

| ratio | ΔPSNR ↑ | ΔSSIM ↑ | ΔLPIPS ↓ | Δphoto ↓ |
|---|---|---|---|---|
| 1% | +0.492767 | +0.002271 | -0.053262 | -0.003695 |
| 5% | +0.349382 | +0.007643 | -0.045788 | -0.002480 |
| 10% | -0.173057 | -0.003357 | -0.028154 | +0.001862 |
| 15% | -0.073405 | -0.002838 | -0.018774 | +0.001545 |

### F3 − C

| ratio | ΔPSNR ↑ | ΔSSIM ↑ | ΔLPIPS ↓ | Δphoto ↓ |
|---|---|---|---|---|
| 1% | -0.157235 | -0.003180 | +0.001770 | +0.002271 |
| 5% | -0.075359 | -0.001542 | +0.001780 | +0.000772 |
| 10% | -0.038847 | +0.000049 | +0.000198 | +0.000138 |
| 15% | +0.023599 | +0.000457 | -0.001873 | -0.000043 |

### F3 − FixedMarginal

| ratio | ΔPSNR ↑ | ΔSSIM ↑ | ΔLPIPS ↓ | Δphoto ↓ |
|---|---|---|---|---|
| 1% | +0.239842 | +0.001286 | +0.000248 | -0.001835 |
| 5% | -0.096662 | -0.001683 | -0.006248 | +0.000893 |
| 10% | -0.005499 | +0.000394 | -0.007438 | +0.000164 |
| 15% | -0.022266 | -0.000321 | -0.000462 | +0.000422 |

### F3相对C的指标方向

| ratio | PSNR | SSIM | LPIPS | photo |
|---|---|---|---|---|
| 1% | 回退 | 回退 | 回退 | 回退 |
| 5% | 回退 | 回退 | 回退 | 回退 |
| 10% | 回退 | 改善 | 回退 | 回退 |
| 15% | 改善 | 改善 | 改善 | 改善 |

## 5. Paired per-view

F3−C的全部配对统计如下。LPIPS/photo为负差才是改善；positive_delta_fraction与improved_view_fraction在CSV中分别保存。所有其它比较及逐view差值也已保存。

| ratio | metric | mean | median | p10 | p90 | improved views |
|---|---|---|---|---|---|---|
| 1% | psnr | -0.157235 | -0.138897 | -0.364015 | -0.022147 | 7.69% |
| 1% | ssim | -0.003180 | -0.002533 | -0.006535 | -0.001461 | 0.00% |
| 1% | lpips | +0.001770 | +0.000734 | -0.002152 | +0.007427 | 35.90% |
| 1% | photo | +0.002271 | +0.001936 | +0.000833 | +0.004635 | 7.69% |
| 5% | psnr | -0.075359 | -0.080473 | -0.172094 | +0.035909 | 15.38% |
| 5% | ssim | -0.001542 | -0.001674 | -0.003081 | +0.000543 | 17.95% |
| 5% | lpips | +0.001780 | +0.001866 | -0.003286 | +0.006705 | 23.08% |
| 5% | photo | +0.000772 | +0.000858 | -0.000440 | +0.001738 | 17.95% |
| 10% | psnr | -0.038847 | -0.027584 | -0.115255 | +0.000958 | 15.38% |
| 10% | ssim | +0.000049 | -0.000050 | -0.000640 | +0.000848 | 46.15% |
| 10% | lpips | +0.000198 | +0.000119 | -0.001665 | +0.002199 | 48.72% |
| 10% | photo | +0.000138 | +0.000114 | -0.000099 | +0.000398 | 30.77% |
| 15% | psnr | +0.023599 | +0.022402 | -0.016343 | +0.087064 | 69.23% |
| 15% | ssim | +0.000457 | +0.000476 | -0.000580 | +0.001802 | 64.10% |
| 15% | lpips | -0.001873 | -0.002073 | -0.004276 | +0.000091 | 84.62% |
| 15% | photo | -0.000043 | -0.000036 | -0.000451 | +0.000406 | 58.97% |

## 6. 统一Fixed-h8 geometry重复度

完全沿用上一轮：同一固定h8 cell内selected rows，逐row最大off-diagonal κ，singleton记0；另统计unordered高相似pair。它是统一局部diagnostic，不是无边界全局overlap。

| ratio | arm | mean | median | p90 | κ>.8 | κ>.9 | κ>.95 |
|---|---|---|---|---|---|---|---|
| 1% | A Raw | 0.131687 | 0.000000 | 0.890781 | 41 | 28 | 22 |
| 5% | A Raw | 0.274623 | 0.000000 | 0.974423 | 493 | 353 | 239 |
| 10% | A Raw | 0.348540 | 0.000000 | 0.985737 | 1341 | 959 | 646 |
| 15% | A Raw | 0.370401 | 0.000611 | 0.986838 | 2208 | 1549 | 1025 |
| 1% | B FixedMarginal | 0.002422 | 0.000000 | 0.000000 | 0 | 0 | 0 |
| 5% | B FixedMarginal | 0.039549 | 0.000000 | 0.005921 | 16 | 5 | 1 |
| 10% | B FixedMarginal | 0.093896 | 0.000000 | 0.469953 | 80 | 29 | 7 |
| 15% | B FixedMarginal | 0.121832 | 0.000000 | 0.603936 | 176 | 67 | 21 |
| 1% | C Current | 0.010920 | 0.000000 | 0.000000 | 1 | 0 | 0 |
| 5% | C Current | 0.080929 | 0.000000 | 0.361809 | 50 | 26 | 18 |
| 10% | C Current | 0.129833 | 0.000000 | 0.660157 | 164 | 91 | 57 |
| 15% | C Current | 0.159680 | 0.000000 | 0.748998 | 363 | 194 | 111 |
| 1% | F3 Progressive+Tail | 0.002571 | 0.000000 | 0.000000 | 0 | 0 | 0 |
| 5% | F3 Progressive+Tail | 0.066366 | 0.000000 | 0.211796 | 33 | 11 | 4 |
| 10% | F3 Progressive+Tail | 0.115075 | 0.000000 | 0.576013 | 120 | 54 | 22 |
| 15% | F3 Progressive+Tail | 0.140384 | 0.000000 | 0.687148 | 249 | 103 | 41 |

不将重复度下降本身当作quality GO；重建质量由四项真实render指标判断。

## 7. Composition、空间覆盖和ranking overlap

| ratio | arm | C-final-rep fraction | internal reps | final reps | tail | emit L1/L2/L3 |
|---|---|---|---|---|---|---|
| 1% | C Current | 98.61% |  |  |  | // |
| 5% | C Current | 90.71% |  |  |  | // |
| 10% | C Current | 81.47% |  |  |  | // |
| 15% | C Current | 72.91% |  |  |  | // |
| 1% | F3 Progressive+Tail | 94.90% | 175 | 249 | 7 | 398/32/1 |
| 5% | F3 Progressive+Tail | 89.22% | 894 | 1132 | 127 | 1298/675/180 |
| 10% | F3 Progressive+Tail | 79.54% | 1442 | 2327 | 537 | 1940/1367/999 |
| 15% | F3 Progressive+Tail | 71.76% | 1821 | 3394 | 1244 | 2347/1966/2146 |

| ratio | arm | hit roots | hit leaves | hit-root mass | hit-leaf mass | dense-root budget | important-root budget |
|---|---|---|---|---|---|---|---|
| 1% | C Current | 258 | 349 | 79.32% | 67.57% | 46.87% | 79.81% |
| 5% | C Current | 735 | 1374 | 97.29% | 91.86% | 60.89% | 66.65% |
| 10% | C Current | 1023 | 2169 | 99.24% | 96.97% | 66.81% | 65.10% |
| 15% | C Current | 1171 | 2747 | 99.63% | 98.56% | 70.88% | 65.12% |
| 1% | F3 Progressive+Tail | 268 | 350 | 81.14% | 66.95% | 47.10% | 79.35% |
| 5% | F3 Progressive+Tail | 751 | 1370 | 97.51% | 91.43% | 60.33% | 65.77% |
| 10% | F3 Progressive+Tail | 1030 | 2158 | 99.28% | 96.77% | 66.81% | 64.82% |
| 15% | F3 Progressive+Tail | 1182 | 2731 | 99.65% | 98.45% | 70.82% | 64.67% |

| ratio | comparison | Jaccard | intersection | added/removed |
|---|---|---|---|---|
| 1% | B_FixedMarginal_vs_F3_ProgressiveMarginal | 0.553153 | 307 | 124/124 |
| 1% | C_Hierarchical_vs_F3_ProgressiveMarginal | 0.795833 | 382 | 49/49 |
| 5% | B_FixedMarginal_vs_F3_ProgressiveMarginal | 0.654245 | 1703 | 450/450 |
| 5% | C_Hierarchical_vs_F3_ProgressiveMarginal | 0.866493 | 1999 | 154/154 |
| 10% | B_FixedMarginal_vs_F3_ProgressiveMarginal | 0.744379 | 3675 | 631/631 |
| 10% | C_Hierarchical_vs_F3_ProgressiveMarginal | 0.887769 | 4050 | 256/256 |
| 15% | B_FixedMarginal_vs_F3_ProgressiveMarginal | 0.776891 | 5648 | 811/811 |
| 15% | C_Hierarchical_vs_F3_ProgressiveMarginal | 0.881171 | 6051 | 408/408 |

1% F3的internal/final/tail数量为175/249/7，emit L1/L2/L3为398/32/1。高档F3同时改变depth scheduling与tail，本轮未跑F1/F2，不能唯一分离二者因果；不自动追加实验或调参数。

## 8. 预注册判定和三场景对照

| gate | 结果 |
|---|---|
| C相对Raw的PSNR主gate | FAIL |
| F3四档mean ΔPSNR≥0 | FAIL：-0.061960 dB |
| F3 1%优于C | FAIL：-0.157235 dB |
| F3 10/15% mean redundancy<C | PASS |
| Combined | NOT SUPPORTED |
| Strong improvement | NO |
| F3相对B PSNR | EXCEEDS |

以下为各scene内部F3−C的ΔPSNR，不能当作跨scene绝对PSNR直接比较，也不按view数量加权掩盖反例。

| scene | 1% | 5% | 10% | 15% | 四档平均 |
|---|---|---|---|---|---|
| fr1_room | +0.103887 | +0.012905 | +0.059143 | +0.062435 | +0.059593 |
| fr1_xyz | +0.250820 | -0.110466 | -0.124511 | -0.059770 | -0.010982 |
| fr2_xyz | -0.157235 | -0.075359 | -0.038847 | +0.023599 | -0.061960 |

fr1_room支持F3，fr1_xyz未通过Combined；本轮fr2_xyz结果增加一条零调参证据，不改写已有反例。任何跨scene整体替换结论都必须同时保留这些差异。

本场景C的1/5% PSNR收益不能掩盖10/15%回退，因此C的四档Primary PSNR gate也未通过。F3的四档平均相对B为正主要由1%贡献，5/10/15%均低于B；不能把EXCEEDS均值标签解释为逐档或全指标支配。当前不支持用F3统一替代C，也不据此宣称C具有全ratio质量GO。

## 9. Runtime与已知限制

| 阶段 | 单次记录ms |
|---|---|
| B | 978.543 |
| C | 7904.143 |
| F3 | 2578.794 |

CPU NumPy reference与GPU同步C构建口径不同，不作速度比较，runtime不参与gate。

单个frozen final map、原SLAM归档视角、实际Bank较少；未做online训练、ATE比较或Selector适配。本轮只能评价same-K reconstruction ranking。SHA与构建/执行日志均保留，结束时重新核对输入与源码。

## 10. 交付

主结果为 `summary.json`、`quality_summary.csv`、`quality_per_view.csv`；完整四metric差值、paired统计、ranking overlap、逐row变化、selected redundancy、composition、空间覆盖、完整emit/unlock traces、ranking数组、SHA和validation均已保存。

**fr2_xyz：NOT SUPPORTED；Strong improvement：NO。结论限定本场景离线same-K，未commit、未push、未改production。**
