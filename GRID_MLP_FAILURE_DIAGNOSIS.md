# Grid MLP 失败机制诊断

**结论：两个层面都有问题，但当前证据不支持“只是2000步没训够”。** 当前grid确实限制cell内部身份调整；与此同时，训练连已能exact表示的Raw都经常越训越差，不能用grid过粗解释全部退化。最应优先处理的是训练目标/优化与deterministic hard Top-K的错位。

| 问题 | 本轮判断 | 证据强度 |
|---|---|---|
| grid是否偏粗？ | 对复现已有PG-ST/Fixed/C集合，YES | 约束与组合上界直接证明；不是quality上界 |
| 是否仅因训练步数不足？ | 不支持 | 原24轨迹+预指定3case从2000延长到5000 |
| MLP是否正常更新？ | YES，输出/排名在持续变化 | finite gradient、非零field、持续预算转移 |
| 训练目标/导出错位？ | 有直接证据 | sample/hard集合分离，采样loss下降与hard退化并存 |
| 能否断言只要细化grid就会好？ | NO | 同一1216参数MLP在更多centers采样，容量没有随cell数增长 |

## 1. “没训够”检查：没有持续朝hard目标收敛

对原48个模型，比较前500与后500步sample photo，先按Bank frame分别平均、再frame等权，避免view构成变化；hard指标使用每100步与step250的全Bank deterministic same-K。

| arm | sample photo下降 | final hard photo优于Raw | sample下降但hard差于Raw | 后1000步hard photo趋势变差 |
|---|---:|---:|---:|---:|
| Grid-MLP | 8/24 | 3/24 | 7/24 | 16/24 |
| Independent-Cell | 24/24 | 9/24 | 15/24 | 17/24 |

Grid-MLP的24个final checkpoint均差于各自已记录的最低Bank hard photo；这只用于诊断，没有把最好step替换成正式结果。后半程梯度仍非零、coefficient仍更新，不能把“还在更新”当作“继续训练必然更好”。首尾sample loss统计是移动训练轨迹的描述，不是冻结checkpoint在同一完整sample上的精确因果比较。

## 2. 直接延长到5000步：原LR多训没有解决

预指定三scene的5%、seed2026，复用原grid_worker二进制、lr=3e-4、optimizer/loss/sampler/Bank schedule。由于旧模型未保存Adam状态，从step0重放以保留完整optimizer历史；前三个case的前2000步采样loss、全部已记录hard集合和Bank photo/PSNR均exact复现。随后继续到5000步，不调LR、不选最好step、不读取non-Bank。

| scene | Bank PSNR step0 | step2000 | step5000 | ΔPSNR 5000−2000 | Δphoto 5000−2000 |
|---|---:|---:|---:|---:|---:|
| fr1_room | 16.734824 | 16.749014 | 16.683320 | -0.065695 | +0.00099900 |
| fr1_xyz | 17.984379 | 17.099316 | 16.570045 | -0.529271 | +0.00533695 |
| fr2_xyz | 19.713109 | 19.625655 | 19.633068 | +0.007413 | +0.00008772 |

![5000步Bank hard质量](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/grid_mlp_failure_diagnosis_20261020/continuation_quality.png)

三个case的photo全部比step2000更差，step5000也全部差于Raw。fr2_xyz的PSNR比step2000微升，但photo继续变差，不能称为一致改善。因此“同设置只加步数”没有支持；本轮不排除其他LR/optimizer可改善，也不把三个case外推为所有ratio/seed的证明。

## 3. grid确有表达限制，但这不能解释为何低于Raw

同cell中score_i=base_i+a_C，因此只能从该cell按Raw顺序取一个prefix。给定目标Top-K T，若目标含有cell内第r个Gaussian，就必须同时取前r−1个。即使每cell coefficient完全独立且可任意设定，MLP也不能绕过这个约束。

本轮对每个目标集合建立所有cell的prefix选项，用动态规划求在总预算K内能命中的最多目标rows。DP先通过100个小规模随机fixture穷举校验。它允许每cell自由选择prefix长度、忽略跨cell score ties，因而是MLP可达集合的宽松上界。全过程只做已完成实验的posthoc集合分析，不用目标labels训练，不输出新quality方法。

| 目标集合 | 当前h最大recall中位数 | h/2 | h/4 |
|---|---:|---:|---:|
| PG-ST | 95.72% | 97.17% | 98.31% |
| Fixed | 85.27% | 88.99% | 92.19% |
| C | 84.00% | 88.19% | 91.90% |

当前h下，PG-ST的24个集合、Fixed/C各12个集合均无法exact表示；完全覆盖PG-ST目标rows所需的cell-prefix预算中位数为原K的1.26倍，C为2.57倍。h/2和h/4只建立临时、固定world-origin的离线mapping，不重训MLP、不修改原mapping；occupied cells分别变为4251–5157、8451–10589。

![网格分辨率与集合表示上界](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/grid_mlp_failure_diagnosis_20261020/prefix_resolution_bounds.png)

图中线为跨case中位数，阴影为min/max范围，不是置信区间。

**上界不是质量结论。** 不同集合可能达到相近或更好的render质量；不能由C recall不足就推断所有regional correction都差，也不能由细网格recall上升就宣称PSNR会上升。更关键的是，a_C=0已经exact表示Raw：grid表示限制可以妨碍超过Raw，却不会迫使优化器把已可表示的Raw变差。当前大量Bank退化仍需从优化/目标解释。

| scene | 原h | Gaussian位于occupancy≥100的cell的比例 |
|---|---:|---:|
| fr1_room | 0.29987092 | 51.78% |
| fr1_xyz | 0.07246670 | 21.80% |
| fr2_xyz | 0.12447133 | 63.95% |

这些比例说明“2048 occupied cells”不等于Gaussian均匀分成小组；room/fr2的大量Gaussian处于大cell内。另一方面，细化只减少cell tying；tiny MLP仍只有1216个共享参数，并不会因为M增大而拥有M个独立自由度。

## 4. 代码路径：calibration已修复，目标错位仍保留

- [cell-only forward与weighted gauge](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/grid_mlp_implicit_si_20261020/train_worker.cpp:25)：MLP输入M个center，按occupancy保留autograd减均值，最后层bias=false。没有退回Gaussian逐点MLP或detached mean。
- [calibrated VJP](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/grid_mlp_implicit_si_20261020/dependencies/calibrated_selection.h:24)：保留offset的一阶隐式导数。原96,000步normalized gradient sum最大约2.99e-16；本轮continuation也通过。此次没有发现旧common-score-shift错误残留。
- [训练选择](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/grid_mlp_implicit_si_20261020/train_worker.cpp:65)与[ST helper](H:/AIProjects/cartgs优化更细致版/mycartgs/include/selector_handoff.h:18)：训练用cached uniform与sigmoid admission，并非hard Top-K。expected count校准为K不代表sample集合等于hard集合。
- [compact photo路径](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/grid_mlp_implicit_si_20261020/dependencies/train_worker.cpp:52)：只把当前sample的rows送入renderer，并在opacity上使用ST mask。校准VJP让unselected scores也收到归一化梯度，但这不是它们真实加入/替换后的render marginal。

| Grid-MLP checkpoint | sample vs hard recall中位数 | Jaccard中位数 | hard集合获得的expected selection mass/K |
|---|---:|---:|---:|
| 0 | 51.36% | 0.3460 | 51.03% |
| 2000 | 50.98% | 0.3438 | 51.60% |

final训练sample和最终hard集合只有约一半Gaussian相同。结合Independent-Cell“24/24采样loss下降、15/24 hard仍差于Raw”，这是当前训练目标与导出质量分离的直接证据。它尚不能单独定量证明cached randomness、ST近似与空间共享各占多少因果份额。

## 5. MLP sharing是不是额外限制？

相同grid、gauge、calibration、LR和训练预算下，Independent-Cell相对Grid-MLP在Bank有18/24个PSNR更好、non-Bank有14/24个更好，non-Bank平均高0.439837 dB。说明MLP这一参数化/优化组合增加了负担。但control本身仍未稳定优于Raw；不同参数化的梯度尺度和优化条件也不同，不能直接断言tiny MLP容量就是唯一主因。

本轮没有拟合PG-ST系数，也没有拟合Independent-Cell field；没有通过增加网络深度、feature或新regularizer改变问题。若后续要把MLP smoothness与优化彻底分开，需要独立的cell-field可表示性对照；当前不把它当作已证明结论。

## 6. 研究优先级

1. **优先：固定grid下验证训练能否稳定提高hard same-K目标。** Raw已在可表示集合内，而当前训练常偏离它；先解释sample与hard质量分离，不能用加步数掩盖。
2. **其次：在训练目标可优化之后再验证分辨率。** 本轮已证明粗grid有集合表示损失，值得保留这条线索；但直接细化同一个平滑MLP，不会自动增加regional field表达自由度。
3. **当前不进入split。** 高gradient、更多changed cells都不是收益证明；本轮未改grid、网络、loss、production，也未stage/commit/push。

## 7. 可复核输出

- [轨迹诊断](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/grid_mlp_failure_diagnosis_20261020/trajectory_diagnosis.csv)、[5000步训练曲线](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/grid_mlp_failure_diagnosis_20261020/continuation_training_curves.csv)、[hard动态](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/grid_mlp_failure_diagnosis_20261020/continuation_hard_dynamics.csv)、[Bank per-view](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/grid_mlp_failure_diagnosis_20261020/continuation_hard_per_view.csv)。
- [当前grid prefix上界](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/grid_mlp_failure_diagnosis_20261020/grid_prefix_representability.csv)、[分辨率表示上界](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/grid_mlp_failure_diagnosis_20261020/resolution_prefix_bounds.csv)、[sample/hard错位](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/grid_mlp_failure_diagnosis_20261020/sampling_export_mismatch.csv)。
- [长预算预注册](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/grid_mlp_failure_diagnosis_20261020/continuation_preregistration.json)、[长预算结果](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/grid_mlp_failure_diagnosis_20261020/continuation_results.json)、[correctness](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/grid_mlp_failure_diagnosis_20261020/correctness.json)、[输入SHA](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/grid_mlp_failure_diagnosis_20261020/input_manifest.json)、[结论JSON](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/grid_mlp_failure_diagnosis_20261020/diagnosis_summary.json)。
- 三个5000步模型的最终weights/field、step0/250/1000/5000快照、每100步diagnostic和原始日志保存在continuation/<scene>/；原24+24正式模型与结果保持不变。
