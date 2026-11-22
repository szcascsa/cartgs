# Grid MLP Cached Bernoulli 推理质量审计

**结论：Cached推理没有修复Grid-MLP的PSNR/photo问题，LPIPS存在明显取舍。** 相对同uniform初始化Raw-Cached，Grid-MLP在non-Bank仅10/24个PSNR改善，平均ΔPSNR=-0.135063 dB；相同实际L下比learned-score hard Top-L低0.518126 dB，比Raw hard Top-L低0.995056 dB。Independent-Cell相对Raw-Cached的PSNR/photo均24/24改善，PSNR平均+0.169661 dB，但仍低于same-L Raw hard的整体PSNR/photo。Cached相对hard的LPIPS多数更好，不能把四指标概括为一致变差。

## 推理变化与完成范围

本轮不训练。复用3scene×4ratio×2seed的24个Grid-MLP与24个Independent-Cell的step2000 checkpoint。推理读取保存的score和cached_uniform，直接调用训练相同的C++ calibratedSelectionProbability与sampleSelectorST；保持42次expected-K二分、原nominal ratio、冻结TF与renderer。没有生成新noise、没有补点或截断。

训练ST mask的forward严格为0/1，因此推理只render被选rows，opacity保持原值；没有把opacity乘以概率。重复相同checkpoint推理得到exact相同集合。

**K是nominal expected budget，L是Cached实际选中数。** Cached不保证L=K。本轮同时真实render learned-score hard Top-L、Raw hard Top-L，严格匹配该Cached集合的实际点数；TF仍输入原nominal ratio，不改为L/N。原hard Top-K只作为历史导出方式对照，不能当成same-count比较。

| arm | actual L/K min | median | max | max abs(L−K) |
|---|---:|---:|---:|---:|
| Grid-MLP | 0.965197 | 1.001042 | 1.033898 | 44 |
| Independent-Cell | 0.941995 | 1.011421 | 1.059322 | 77 |

## 核心质量结果：每项24个scene/ratio/seed case

| Cached arm | reference | split | PSNR改善 | photo改善 | SSIM改善 | LPIPS改善 | mean ΔPSNR | mean Δphoto | actual-count相同case |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Grid-MLP | Raw-Cached | Bank | 8/24 | 9/24 | 11/24 | 10/24 | -0.095174 | +0.0011382 | 0/24 |
| Grid-MLP | Raw-Cached | non-Bank | 10/24 | 11/24 | 12/24 | 8/24 | -0.135063 | +0.0013296 | 0/24 |
| Grid-MLP | Grid-MLP-Hard-L | Bank | 5/24 | 1/24 | 0/24 | 23/24 | -0.570831 | +0.0127140 | 24/24 |
| Grid-MLP | Grid-MLP-Hard-L | non-Bank | 5/24 | 1/24 | 0/24 | 23/24 | -0.518126 | +0.0117917 | 24/24 |
| Grid-MLP | Raw-Hard-L-Grid-MLP | Bank | 0/24 | 0/24 | 0/24 | 23/24 | -0.949989 | +0.0179676 | 24/24 |
| Grid-MLP | Raw-Hard-L-Grid-MLP | non-Bank | 0/24 | 0/24 | 0/24 | 24/24 | -0.995056 | +0.0181781 | 24/24 |
| Grid-MLP | Grid-MLP-Hard-K | Bank | 5/24 | 0/24 | 0/24 | 23/24 | -0.592518 | +0.0131049 | 0/24 |
| Grid-MLP | Grid-MLP-Hard-K | non-Bank | 5/24 | 0/24 | 0/24 | 23/24 | -0.549823 | +0.0123633 | 0/24 |
| Grid-MLP | Raw-Hard-K | Bank | 0/24 | 0/24 | 0/24 | 23/24 | -0.949149 | +0.0179772 | 0/24 |
| Grid-MLP | Raw-Hard-K | non-Bank | 0/24 | 0/24 | 0/24 | 24/24 | -0.991348 | +0.0181633 | 0/24 |
| Grid-MLP | Fixed-Hard-K | Bank | 0/24 | 0/24 | 0/24 | 14/24 | -1.332044 | +0.0211301 | 0/24 |
| Grid-MLP | Fixed-Hard-K | non-Bank | 0/24 | 0/24 | 0/24 | 14/24 | -1.378842 | +0.0217001 | 0/24 |
| Grid-MLP | C-Hard-K | Bank | 0/24 | 0/24 | 0/24 | 16/24 | -1.338820 | +0.0213975 | 0/24 |
| Grid-MLP | C-Hard-K | non-Bank | 0/24 | 0/24 | 0/24 | 13/24 | -1.385817 | +0.0220077 | 0/24 |
| Independent-Cell | Raw-Cached | Bank | 24/24 | 24/24 | 24/24 | 18/24 | +0.183612 | -0.0030741 | 1/24 |
| Independent-Cell | Raw-Cached | non-Bank | 24/24 | 24/24 | 24/24 | 17/24 | +0.169661 | -0.0030342 | 1/24 |
| Independent-Cell | Independent-Cell-Hard-L | Bank | 0/24 | 0/24 | 0/24 | 23/24 | -0.702822 | +0.0142047 | 24/24 |
| Independent-Cell | Independent-Cell-Hard-L | non-Bank | 1/24 | 0/24 | 0/24 | 23/24 | -0.707298 | +0.0140792 | 24/24 |
| Independent-Cell | Raw-Hard-L-Independent-Cell | Bank | 0/24 | 0/24 | 0/24 | 23/24 | -0.704069 | +0.0142805 | 24/24 |
| Independent-Cell | Raw-Hard-L-Independent-Cell | non-Bank | 1/24 | 0/24 | 0/24 | 23/24 | -0.731109 | +0.0144740 | 24/24 |
| Independent-Cell | Independent-Cell-Hard-K | Bank | 0/24 | 0/24 | 0/24 | 23/24 | -0.682227 | +0.0138403 | 2/24 |
| Independent-Cell | Independent-Cell-Hard-K | non-Bank | 1/24 | 0/24 | 0/24 | 23/24 | -0.684935 | +0.0136417 | 2/24 |
| Independent-Cell | Raw-Hard-K | Bank | 1/24 | 0/24 | 0/24 | 23/24 | -0.670363 | +0.0137650 | 2/24 |
| Independent-Cell | Raw-Hard-K | non-Bank | 1/24 | 0/24 | 0/24 | 23/24 | -0.686624 | +0.0137996 | 2/24 |
| Independent-Cell | Fixed-Hard-K | Bank | 0/24 | 0/24 | 0/24 | 17/24 | -1.053257 | +0.0169179 | 2/24 |
| Independent-Cell | Fixed-Hard-K | non-Bank | 0/24 | 0/24 | 0/24 | 17/24 | -1.074117 | +0.0173364 | 2/24 |
| Independent-Cell | C-Hard-K | Bank | 0/24 | 0/24 | 0/24 | 18/24 | -1.060034 | +0.0171853 | 2/24 |
| Independent-Cell | C-Hard-K | non-Bank | 0/24 | 0/24 | 0/24 | 19/24 | -1.081092 | +0.0176439 | 2/24 |

Raw-Cached为a=0、同一组saved uniform的初始化基线，可观察Cached策略下训练前后的变化；其实际点数与训练后可能不同。Hard-L两组是严格same-count对照。PSNR/SSIM越高越好，photo/LPIPS越低越好。所有差值均为Cached−reference。

## Grid-MLP non-Bank逐case

| scene | ratio | seed | nominal K | Cached L | Cached PSNR | ΔRaw-Cached | Δ原Grid hard-K | ΔGrid hard-L | ΔRaw hard-L |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fr1_room | 1% | 2026 | 503 | 514 | 10.999463 | -0.082747 | -1.984989 | -2.006206 | -2.012794 |
| fr1_room | 1% | 2027 | 503 | 513 | 12.343623 | +0.212087 | -0.839471 | -0.826336 | -0.668632 |
| fr1_room | 5% | 2026 | 2517 | 2524 | 15.413911 | +0.024706 | -1.249120 | -1.253923 | -1.243223 |
| fr1_room | 5% | 2027 | 2517 | 2533 | 15.571724 | +0.029924 | -1.073302 | -1.089258 | -1.091667 |
| fr1_room | 10% | 2026 | 5033 | 5000 | 17.166488 | +0.014899 | -0.516074 | -0.513454 | -0.513470 |
| fr1_room | 10% | 2027 | 5033 | 5058 | 17.110135 | +0.047967 | -0.582164 | -0.581573 | -0.571659 |
| fr1_room | 15% | 2026 | 7550 | 7512 | 17.909517 | +0.027366 | -0.140939 | -0.134497 | -0.142298 |
| fr1_room | 15% | 2027 | 7550 | 7556 | 17.837815 | +0.019546 | -0.222614 | -0.224546 | -0.216792 |
| fr1_xyz | 1% | 2026 | 236 | 244 | 11.860051 | -0.116298 | +0.827776 | +0.776671 | -1.482756 |
| fr1_xyz | 1% | 2027 | 236 | 232 | 12.507512 | +0.130692 | +0.381197 | +1.082727 | -0.181604 |
| fr1_xyz | 5% | 2026 | 1182 | 1179 | 17.517542 | -0.128145 | +0.671626 | +0.672966 | -0.283425 |
| fr1_xyz | 5% | 2027 | 1182 | 1207 | 16.747633 | -0.440436 | +0.189823 | +0.060545 | -1.151625 |
| fr1_xyz | 10% | 2026 | 2364 | 2366 | 18.835149 | -0.144660 | -0.484151 | -0.484069 | -1.010620 |
| fr1_xyz | 10% | 2027 | 2364 | 2375 | 18.184332 | -0.856459 | -0.789624 | -0.797970 | -1.729012 |
| fr1_xyz | 15% | 2026 | 3546 | 3552 | 19.964514 | -0.203208 | -0.139609 | -0.140967 | -0.927410 |
| fr1_xyz | 15% | 2027 | 3546 | 3590 | 19.820489 | -0.375794 | -0.638922 | -0.658970 | -1.087739 |
| fr2_xyz | 1% | 2026 | 431 | 423 | 14.063826 | -1.191457 | -1.100513 | -1.012674 | -2.118362 |
| fr2_xyz | 1% | 2027 | 431 | 416 | 15.405804 | +0.823019 | +0.358806 | +0.543754 | -0.762019 |
| fr2_xyz | 5% | 2026 | 2153 | 2136 | 18.782192 | +0.108029 | -0.948717 | -0.932977 | -1.015813 |
| fr2_xyz | 5% | 2027 | 2153 | 2151 | 18.851315 | -0.015967 | -0.928430 | -0.928387 | -1.033272 |
| fr2_xyz | 10% | 2026 | 4306 | 4267 | 19.598051 | -0.450534 | -1.317737 | -1.311384 | -1.625021 |
| fr2_xyz | 10% | 2027 | 4306 | 4325 | 20.028296 | -0.009910 | -1.200106 | -1.210126 | -1.220785 |
| fr2_xyz | 15% | 2026 | 6459 | 6446 | 20.286121 | -0.636417 | -0.941745 | -0.940388 | -1.214014 |
| fr2_xyz | 15% | 2027 | 6459 | 6467 | 20.902404 | -0.027713 | -0.526743 | -0.523984 | -0.577323 |

## Paired-view改善范围

| arm | reference | split | paired PSNR median>0 cases | 多数views改善cases |
|---|---|---|---:|---:|
| Grid-MLP | Raw-Cached | Bank | 14/24 | 13/24 |
| Grid-MLP | Raw-Cached | non-Bank | 14/24 | 14/24 |
| Grid-MLP | Grid-MLP-Hard-L | Bank | 4/24 | 4/24 |
| Grid-MLP | Grid-MLP-Hard-L | non-Bank | 5/24 | 5/24 |
| Grid-MLP | Raw-Hard-L-Grid-MLP | Bank | 0/24 | 0/24 |
| Grid-MLP | Raw-Hard-L-Grid-MLP | non-Bank | 0/24 | 0/24 |
| Grid-MLP | Grid-MLP-Hard-K | Bank | 4/24 | 4/24 |
| Grid-MLP | Grid-MLP-Hard-K | non-Bank | 5/24 | 5/24 |
| Independent-Cell | Raw-Cached | Bank | 24/24 | 24/24 |
| Independent-Cell | Raw-Cached | non-Bank | 24/24 | 24/24 |
| Independent-Cell | Independent-Cell-Hard-L | Bank | 1/24 | 1/24 |
| Independent-Cell | Independent-Cell-Hard-L | non-Bank | 1/24 | 1/24 |
| Independent-Cell | Raw-Hard-L-Independent-Cell | Bank | 1/24 | 1/24 |
| Independent-Cell | Raw-Hard-L-Independent-Cell | non-Bank | 1/24 | 1/24 |
| Independent-Cell | Independent-Cell-Hard-K | Bank | 1/24 | 1/24 |
| Independent-Cell | Independent-Cell-Hard-K | non-Bank | 1/24 | 1/24 |

四指标的mean/median/p10/p90/improved-view fraction逐case保存在paired_statistics.csv；没有挑最好seed、view或checkpoint。

## 判读

- Grid-MLP与原hard Top-K相比，non-Bank PSNR仅5/24改善、平均-0.549823 dB，photo为0/24改善；严格same-L比较learned hard Top-L仍平均-0.518126 dB，说明PSNR/photo损失不只是实际点数波动。
- Grid-MLP即使使用与训练相同的Cached推理，Bank相对初始化Raw-Cached也只有8/24个PSNR、9/24个photo改善；non-Bank分别10/24、11/24。这削弱了“只要把hard导出改成forward采样，MLP训练就会有效”的解释。
- Independent-Cell在相同Cached策略内出现一致学习信号：Bank/non-Bank的PSNR、SSIM、photo均24/24优于初始化；non-Bank LPIPS为17/24改善。这支持固定regional系数能够在当前Cached协议下学到信号，但23/24个case与初始化实际点数不同，不能把全部收益归因于更好的等预算选择；也没有测试更换uniform后的迁移。
- 相同L的Raw hard比较中，Grid-MLP PSNR/photo均0/24改善，Independent-Cell PSNR仅1/24、photo0/24改善，平均ΔPSNR分别-0.995056/-0.731109 dB。因此Cached协议内的学习进展尚未转化为超过已有Raw hard的重建质量。
- 四指标出现取舍：Grid-MLP Cached比same-L Raw hard的LPIPS在24/24个case更好，平均ΔLPIPS=-0.037025；同时PSNR/SSIM/photo全部更差。Independent-Cell的对应LPIPS为23/24更好。不能只看PSNR否认LPIPS收益，也不能只看LPIPS宣称整体方案成功。
- 这次结果收窄了此前机制判断：训练sample与hard集合不一致是事实，但它不自动等价于hard导出在损害质量。实际把推理对齐forward后，PSNR/photo总体更差。当前Grid-MLP还存在参数共享/优化本身的问题，单纯换导出不是充分修复；本轮无法进一步区分MLP容量、平滑共享和优化条件。
- 本轮保留Cached实验推理模式与全部对照，不把它部署到production，不自动改LR、network、grid、sampling或loss，也不重新训练。

## Correctness

- 96项step0/2000检查：原生CUDA C++与Python CUDA重建集合exact，重复Cached推理exact，ST forward mask exact为0/1；step0点数与训练首步相同。
- Grid-MLP与Independent-Cell共享同scene/ratio/seed的原始uniform，初始Raw-Cached集合exact相同；uniform没有被重新生成。
- 推理读取保存checkpoint的score；没有更新MLP/coefficients。全部96个state文件本轮推理前后SHA不变，导出集合在quality前锁SHA。
- 所有hard-L对照与对应Cached实际点数exact一致、rows unique；不把L当成K。
- 原Raw hard-K的1420次Bank/non-Bank view render四指标逐view exact复现；新168个case共19880次view render。所有Gaussian/TF/Master与输入runtime冻结核验通过。
- 仍使用原Bank/non-Bank划分、cameras、GT和metrics；没有用quality反向选score、uniform、ratio或checkpoint。没有新训练、grid或split，production未修改。

## 结果文件

- [quality_summary.csv](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/grid_mlp_cached_bernoulli_inference_20261020/quality_summary.csv)、[quality_per_view.csv](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/grid_mlp_cached_bernoulli_inference_20261020/quality_per_view.csv)、[quality_deltas.csv](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/grid_mlp_cached_bernoulli_inference_20261020/quality_deltas.csv)、[paired_statistics.csv](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/grid_mlp_cached_bernoulli_inference_20261020/paired_statistics.csv)。
- [selection_counts.csv](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/grid_mlp_cached_bernoulli_inference_20261020/selection_counts.csv)、[selected_sets.npz](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/grid_mlp_cached_bernoulli_inference_20261020/selected_sets.npz)、[inference_arrays.npz](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/grid_mlp_cached_bernoulli_inference_20261020/inference_arrays.npz)、[selected_set_overlap.csv](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/grid_mlp_cached_bernoulli_inference_20261020/selected_set_overlap.csv)。
- [correctness.json](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/grid_mlp_cached_bernoulli_inference_20261020/correctness.json)、[audit_summary.json](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/grid_mlp_cached_bernoulli_inference_20261020/audit_summary.json)、[原生推理代码](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/grid_mlp_cached_bernoulli_inference_20261020/cached_inference.cpp)、[推理与导出脚本](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/grid_mlp_cached_bernoulli_inference_20261020/prepare.py)。
- [质量差值图](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/grid_mlp_cached_bernoulli_inference_20261020/quality_deltas.png)。

完成后停止。未修改production Selector；未stage、commit、push。
