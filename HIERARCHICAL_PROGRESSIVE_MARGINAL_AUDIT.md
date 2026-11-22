# Hierarchical Representative SI：Marginal Tail 与 Progressive Ranking 审计

结论：**PROGRESSIVE SUPPORTED；COMBINED SUPPORTED。独立F1未达到预注册“明显重复度下降”门槛；F3未达到Strong improvement。** 推荐下一阶段保留F3组合机制作为候选（ADOPT BOTH），本轮不修改正式C，不接online或Teacher，不搜索参数。

F3相对C的四档PSNR为 +0.103887 / +0.012905 / +0.059143 / +0.062435 dB，平均+0.059593 dB；相对Fixed h8四档均高，平均+0.042367 dB。但统一geometry重复度仍高于B，因此不能称全面超过FixedMarginal，更不能据此称production/online GO。

## 1. 研究问题与冻结范围

只检验两种ranking机制：Independent Residual→Marginal Tail，以及ancestor-conditioned coarse-to-fine unlock。SparseHash target2048、h搜索、origin=(0,0,0)、3rep/node、Q<0.80、L3、radius/κ、candidate valid AND observed、Raw SI/Bank、Master/TF、renderer、maturity=1、Selector均不变。

HEAD=`3cfccc4a92c8e1ec6c3d491dac1c9efbf151daeb`，branch=`redundancy-aware-si`。现有tracking文件保持原样；本轮只新增离线诊断脚本/报告/证据，没有production源码改动、commit或push。

## 2. 运行前预注册

`preregistration.json` 与 `mechanism_preregistration.json`字节相同，SHA256：

```text
a094f25781f8f5834beab3b21880fa16a01b4facababb12c3d526218505d4021
```

门槛在新arm渲染前写入，结束时重新核验SHA，没有调门槛：


|假设|固定判定|
|---|---|
|F1|10%/15% mean redundancy均下降≥10%，κ>0.9 pair不增加；两档平均ΔPSNR≥−0.03 dB|
|F2|1% ΔPSNR≥+0.05；5/10/15%每档ΔPSNR≥−0.03|
|F3|相对C四档平均ΔPSNR≥0、1% ΔPSNR>0、10/15% mean redundancy均低于C|
|Strong|F3通过；四档平均PSNR≥B−0.02；四档mean redundancy平均及κ>0.9 total pairs均≤B的1.10倍|
|Q5 PSNR水平|四档平均F3−B≥+0.02为EXCEEDS，≥−0.02为MATCHES，否则BELOW；四项指标仍逐项报告|

10%与1.10倍是对“明显下降/接近”的预先操作定义，不是算法参数，不参与生成ranking。按本轮新指令，高budget SSIM/photo记录但不专门优化，也不引入上一轮的全指标统一降级gate；上一轮C整体QUALITY INCONCLUSIVE的历史结论不重写。

## 3. 冻结输入与Baseline reproduction gate

复用fr1_room final `16900_shutdown`：N_total=52635，candidate=50330；69 Bank views的原始contribution/Raw与161个 **fixed non-Bank archived evaluation views**。这些视角参与过原SLAM，不作独立新场景证据。复用同一Master/geometry/opacity/SH/rotation、TF/AABB、camera、GT、mask及requested-ratio TF consumer。没有重新训练或重渲染GI。



|输入|SHA256|
|---|---|
|Master|73f23088aab5c39c9864cf3a0b0fb1fab6384ae162bc848dadeefec5f0b1c4bb|
|TF含AABB buffers|a01c7b908dee8a9fd29603babe49fb5a2c99e8e0010072d8ef13b85429cfa4bd|
|camera/GT archive|90e74ad824dddcdd2a740d004082d747f170ce912aaaff07e99f6c1c9e5caabf|
|原69-view contribution|146ddbfa407e5c933d95c9b30de8b9886e929c6dae240dca63f521ced26169d5|
|原candidate Raw/geometry数组|c89a9e157e3dde1eaca80a17153f68821e5e7ae7a71d7383304ef608dafaece4|
|hierarchy冻结输入|e54e055fd9be3a211ba87cf6997fd088960659ff3f415d2affeebc7636d09a55|

实际数据与worker共240个SHA条目全部通过；另核对上一轮complete ranking、selected rows、质量CSV、完整manifest和tree的artifact SHA。当前production库已不同于旧实验，但从历史runtime_before复制的旧mapper/rasterizer库与冻结SHA精确一致；ldd与实际worker maps均指向本轮隔离runtime，未覆盖production。TF参数及全部buffer在worker退出前逐tensor检查未变。

A/B/C重新生成完整50330 permutation、score与四档selected sets，与上一轮逐值exact相等；重新真实渲染12个arm-ratio×161视角，PSNR/SSIM/LPIPS/photo最大逐view和aggregate误差均为0。新arm只在该gate通过后生成。


|ratio|K|A Raw PSNR|B FixedMarginal PSNR|C Current PSNR|
|---|---|---|---|---|
|1.00%|503|12.988268|13.882451|13.834515|
|5.00%|2517|16.653142|16.980360|17.046943|
|10.00%|5033|17.680354|17.802212|17.760086|
|15.00%|7550|18.054710|18.225545|18.180123|

固定h_root=0.29986755746366806 m、2048 occupied roots、5601 final leaves、13715 final reps、weighted leaf Q=0.8837622487608384。C重建与上一轮tree/score/assignment完全一致；F1/F2/F3均只读取这棵冻结tree，不重新决定split。

## 4. 离线实现与arm定义


|Arm|排序语义|
|---|---|
|A|Raw SI降序|
|B|原Fixed h8 geometry-only完整local marginal + predecessor-preserving merge|
|C|现有final-leaf first3 Delta + independent residual，全局score降序|
|F1|同一final leaves完整local marginal sequence，先验证first3 IDs exact，head-only全局merge|
|F2|root active frontier、ancestor-conditioned最多3个新rep、解锁children；final leaf固定independent residual list|
|F3|同F2的frontier/coarse phase；final leaf继续conditional marginal tail|

`progressive_ranker.py`为独立NumPy FP64 reference，按cell/node矩阵执行精确公式，无approximate neighbors、quantization或深度权重。`experiment_data.py`从原tree按root key与midpoint bits恢复membership；`new_rankings.py`生成三条完整ranking和emit/unlock记录。没有修改C++正式实现。

Shared frozen formula：

```text
r_i = cbrt(s_ix*s_iy*s_iz)
κ_ij = exp(-||x_i-x_j||² / (2*((r_i+r_j)/2)²+1e-12)); κ_ii=1
Delta_i = Σ_j I_j*max(κ_ij-cover_j,0)
```

F1只对final leaf求完整sequence；每leaf前三ID与C exact一致，Delta以rtol5e-12/atol1e-8核验。所有local gain非负、diminishing returns通过，没有通过sort或clamp gain修复顺序；完整local gains之和核验为leaf Raw mass。

Progressive的child cover继承路径上所有已emit ancestor reps的max κ，包括坐标不在该child内的ancestor。已经emit的row不再作为候选，但原node所有rows仍参与加权覆盖目标；被选ancestor自覆盖为1。每node最多emit3个新的rep；不足3个时完成remaining后解锁，不把ancestor数量充当该node的新容量。被ancestor完全耗尽的节点不emit，仍按冻结children关系完成遍历。

F2完成leaf coarse后只排序一次I*(1-cover)，residual之间不更新cover；F3继续greedy，每次tail emit都更新cover。所有arm只暴露当前head，local tie按source row，global tie按priority降序、(gx,gy,gz,path)字典序、source row。允许unlock后priority增加，代码不事后全局resort；本fixture四档prefix未观察到超过1e-10的priority上升，不把这当作普遍性质。

F2/F3导出的score为 `N-rank0`，验证按score排序与实际emit sequence exact相等。它们是ranking mechanism，ordinal magnitude没有物理mass含义，不能替代mass-conserving SI定义。F1另保存真实gain；正式C的mass redistribution没有修改。

## 5. Correctness、唯一row与独立重放


|测试/检查|结果|
|---|---|
|单root无split|F1 first3与原C一致，随后完整tail|
|两root简单/复杂|child必须等parent coarse capacity完成才可emit|
|ancestor row在child中再次最优|排除已emit row，无重复输出|
|高κ ancestor|child conditional gain下降|
|独立ancestor|child gain不被错误压低|
|两个重复residual|第一个tail gain=6，第二个降为0|
|3层重复ancestor path|每row恰好一次，parent/child身份不会重复emit|
|repeat exact|三arm完整order/gain/event/unlock记录重复一致|
|same-K / nested|503/2517/5033/7550，均为同一complete ranking前缀|
|tree immutable|6619 nodes原children/split/leaf membership及tree SHA不变|

独立 `replay_checks.py`不调用ranker的Cell或progressive函数，重新构建κ与inherited cover，逐节点核对全部150990次emit的候选argmax和priority，最大gain差为0；再单独从local heads重放unlock/frontier，完整ranking exact相同。所有arm长度=unique rows=50330，无漏row、重复row或NaN/Inf。

初次baseline工程检查读取worker maps过早，尚未完成动态库加载；在发出verify并等待worker就绪后读取，旧SHA正常通过。保留失败日志，不涉及冻结输入或算法变化。

## 6. 主质量表


|ratio|K|Raw|Fixed h8|Current C|F1|F2|F3|
|---|---|---|---|---|---|---|---|
|1.00%|503|12.988268|13.882451|13.834515|13.834515|13.938402|13.938402|
|5.00%|2517|16.653142|16.980360|17.046943|17.041150|17.048579|17.059848|
|10.00%|5033|17.680354|17.802212|17.760086|17.818372|17.780563|17.819229|
|15.00%|7550|18.054710|18.225545|18.180123|18.224950|18.219173|18.242558|

下表为相对C的四项差值；PSNR/SSIM越高越好，LPIPS/photo越低越好。


|新arm|ratio|ΔPSNR|ΔSSIM|ΔLPIPS|Δphoto|
|---|---|---|---|---|---|
|F1_MarginalTail|1.00%|+0.000000|+0.000000|+0.000000|+0.000000|
|F1_MarginalTail|5.00%|-0.005792|-0.000167|+0.000867|-0.000018|
|F1_MarginalTail|10.00%|+0.058285|+0.000730|-0.002422|-0.000564|
|F1_MarginalTail|15.00%|+0.044827|+0.000749|-0.002302|-0.000408|
|F2_ProgressiveResidual|1.00%|+0.103887|+0.000553|+0.000165|-0.000862|
|F2_ProgressiveResidual|5.00%|+0.001637|-0.000268|+0.000793|+0.000002|
|F2_ProgressiveResidual|10.00%|+0.020477|+0.000143|-0.002179|-0.000171|
|F2_ProgressiveResidual|15.00%|+0.039050|+0.000530|-0.001425|-0.000378|
|F3_ProgressiveMarginal|1.00%|+0.103887|+0.000553|+0.000165|-0.000862|
|F3_ProgressiveMarginal|5.00%|+0.012905|-0.000054|+0.000576|-0.000186|
|F3_ProgressiveMarginal|10.00%|+0.059143|+0.000407|-0.002710|-0.000495|
|F3_ProgressiveMarginal|15.00%|+0.062435|+0.001042|-0.003183|-0.000611|

F3与F1/F2/B/Raw的全部差值保存在quality_deltas.csv。以下是与B的直接对照：


|ratio|F3−B ΔPSNR|ΔSSIM|ΔLPIPS|Δphoto|
|---|---|---|---|---|
|1.00%|+0.055951|-0.000414|+0.002765|+0.000003|
|5.00%|+0.079488|+0.000424|+0.001302|-0.000644|
|10.00%|+0.017017|+0.000904|-0.002971|-0.000433|
|15.00%|+0.017013|+0.000979|-0.001051|-0.000380|

F3四档PSNR均高于B，但1%/5% LPIPS略差，1% SSIM略差，故EXCEEDS只表示预注册PSNR水平，不是全部指标严格支配。

![质量与重复度](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/hierarchical_progressive_marginal_20261019/progressive_quality.png)

## 7. Paired per-view统计

固定161个frame_index严格配对。改善fraction按指标方向定义；CSV另存原始positive-delta fraction，避免LPIPS/photo正负歧义。


|比较|ratio|metric|mean Δ|median Δ|p10 Δ|p90 Δ|改善views|
|---|---|---|---|---|---|---|---|
|F1_MarginalTail_minus_C_Hierarchical|1.00%|psnr|+0.000000|+0.000000|+0.000000|+0.000000|0.00%|
|F1_MarginalTail_minus_C_Hierarchical|1.00%|ssim|+0.000000|+0.000000|+0.000000|+0.000000|0.00%|
|F1_MarginalTail_minus_C_Hierarchical|1.00%|lpips|+0.000000|+0.000000|+0.000000|+0.000000|0.00%|
|F1_MarginalTail_minus_C_Hierarchical|1.00%|photo|+0.000000|+0.000000|+0.000000|+0.000000|0.00%|
|F1_MarginalTail_minus_C_Hierarchical|5.00%|psnr|-0.005792|-0.002443|-0.079048|+0.108943|47.20%|
|F1_MarginalTail_minus_C_Hierarchical|5.00%|ssim|-0.000167|+0.000067|-0.002153|+0.001174|50.93%|
|F1_MarginalTail_minus_C_Hierarchical|5.00%|lpips|+0.000867|+0.000676|-0.003136|+0.004896|39.75%|
|F1_MarginalTail_minus_C_Hierarchical|5.00%|photo|-0.000018|-0.000026|-0.001296|+0.001026|52.17%|
|F1_MarginalTail_minus_C_Hierarchical|10.00%|psnr|+0.058285|+0.025200|-0.057978|+0.230942|60.25%|
|F1_MarginalTail_minus_C_Hierarchical|10.00%|ssim|+0.000730|+0.000480|-0.001021|+0.003239|63.35%|
|F1_MarginalTail_minus_C_Hierarchical|10.00%|lpips|-0.002422|-0.001477|-0.007895|+0.001857|68.32%|
|F1_MarginalTail_minus_C_Hierarchical|10.00%|photo|-0.000564|-0.000293|-0.002275|+0.000707|62.73%|
|F1_MarginalTail_minus_C_Hierarchical|15.00%|psnr|+0.044827|+0.012491|-0.067587|+0.201738|57.14%|
|F1_MarginalTail_minus_C_Hierarchical|15.00%|ssim|+0.000749|+0.000373|-0.001327|+0.002892|65.22%|
|F1_MarginalTail_minus_C_Hierarchical|15.00%|lpips|-0.002302|-0.001221|-0.007026|+0.001390|68.32%|
|F1_MarginalTail_minus_C_Hierarchical|15.00%|photo|-0.000408|-0.000196|-0.001824|+0.000724|59.01%|



|比较|ratio|metric|mean Δ|median Δ|p10 Δ|p90 Δ|改善views|
|---|---|---|---|---|---|---|---|
|F2_ProgressiveResidual_minus_C_Hierarchical|1.00%|psnr|+0.103887|+0.092944|-0.233522|+0.400724|69.57%|
|F2_ProgressiveResidual_minus_C_Hierarchical|1.00%|ssim|+0.000553|+0.000583|-0.004741|+0.005740|59.01%|
|F2_ProgressiveResidual_minus_C_Hierarchical|1.00%|lpips|+0.000165|+0.000000|-0.006795|+0.007350|49.07%|
|F2_ProgressiveResidual_minus_C_Hierarchical|1.00%|photo|-0.000862|-0.001244|-0.005132|+0.004510|63.35%|
|F2_ProgressiveResidual_minus_C_Hierarchical|5.00%|psnr|+0.001637|+0.008396|-0.120485|+0.122471|52.80%|
|F2_ProgressiveResidual_minus_C_Hierarchical|5.00%|ssim|-0.000268|-0.000363|-0.002293|+0.001677|35.40%|
|F2_ProgressiveResidual_minus_C_Hierarchical|5.00%|lpips|+0.000793|+0.000704|-0.003411|+0.004883|42.24%|
|F2_ProgressiveResidual_minus_C_Hierarchical|5.00%|photo|+0.000002|-0.000019|-0.001354|+0.001509|50.93%|
|F2_ProgressiveResidual_minus_C_Hierarchical|10.00%|psnr|+0.020477|+0.025522|-0.074116|+0.106245|62.73%|
|F2_ProgressiveResidual_minus_C_Hierarchical|10.00%|ssim|+0.000143|+0.000143|-0.001181|+0.001791|52.80%|
|F2_ProgressiveResidual_minus_C_Hierarchical|10.00%|lpips|-0.002179|-0.002040|-0.005549|+0.000571|78.88%|
|F2_ProgressiveResidual_minus_C_Hierarchical|10.00%|photo|-0.000171|-0.000186|-0.001099|+0.000775|60.25%|
|F2_ProgressiveResidual_minus_C_Hierarchical|15.00%|psnr|+0.039050|+0.041767|-0.056618|+0.134357|73.91%|
|F2_ProgressiveResidual_minus_C_Hierarchical|15.00%|ssim|+0.000530|+0.000446|-0.000963|+0.002031|71.43%|
|F2_ProgressiveResidual_minus_C_Hierarchical|15.00%|lpips|-0.001425|-0.001068|-0.004437|+0.001840|70.81%|
|F2_ProgressiveResidual_minus_C_Hierarchical|15.00%|photo|-0.000378|-0.000321|-0.001271|+0.000513|75.78%|



|比较|ratio|metric|mean Δ|median Δ|p10 Δ|p90 Δ|改善views|
|---|---|---|---|---|---|---|---|
|F3_ProgressiveMarginal_minus_C_Hierarchical|1.00%|psnr|+0.103887|+0.092944|-0.233522|+0.400724|69.57%|
|F3_ProgressiveMarginal_minus_C_Hierarchical|1.00%|ssim|+0.000553|+0.000583|-0.004741|+0.005740|59.01%|
|F3_ProgressiveMarginal_minus_C_Hierarchical|1.00%|lpips|+0.000165|+0.000000|-0.006795|+0.007350|49.07%|
|F3_ProgressiveMarginal_minus_C_Hierarchical|1.00%|photo|-0.000862|-0.001244|-0.005132|+0.004510|63.35%|
|F3_ProgressiveMarginal_minus_C_Hierarchical|5.00%|psnr|+0.012905|+0.010073|-0.120764|+0.167313|53.42%|
|F3_ProgressiveMarginal_minus_C_Hierarchical|5.00%|ssim|-0.000054|-0.000057|-0.002247|+0.002079|47.83%|
|F3_ProgressiveMarginal_minus_C_Hierarchical|5.00%|lpips|+0.000576|+0.000876|-0.004238|+0.004786|41.61%|
|F3_ProgressiveMarginal_minus_C_Hierarchical|5.00%|photo|-0.000186|-0.000134|-0.001824|+0.001500|55.90%|
|F3_ProgressiveMarginal_minus_C_Hierarchical|10.00%|psnr|+0.059143|+0.031591|-0.076847|+0.224457|62.73%|
|F3_ProgressiveMarginal_minus_C_Hierarchical|10.00%|ssim|+0.000407|+0.000119|-0.001690|+0.002823|55.28%|
|F3_ProgressiveMarginal_minus_C_Hierarchical|10.00%|lpips|-0.002710|-0.001653|-0.007683|+0.001741|72.05%|
|F3_ProgressiveMarginal_minus_C_Hierarchical|10.00%|photo|-0.000495|-0.000326|-0.002042|+0.000927|65.84%|
|F3_ProgressiveMarginal_minus_C_Hierarchical|15.00%|psnr|+0.062435|+0.041945|-0.057964|+0.218922|75.16%|
|F3_ProgressiveMarginal_minus_C_Hierarchical|15.00%|ssim|+0.001042|+0.000876|-0.001153|+0.002969|75.16%|
|F3_ProgressiveMarginal_minus_C_Hierarchical|15.00%|lpips|-0.003183|-0.002410|-0.007983|+0.001489|75.78%|
|F3_ProgressiveMarginal_minus_C_Hierarchical|15.00%|photo|-0.000611|-0.000584|-0.001925|+0.000570|72.67%|

F3−B、F3−F1/F2/Raw的同样统计及每view原始差值完整保存在paired CSV，不以全局mean替代paired分布。归档KF彼此相关，不作跨scene独立统计推断。

## 8. Same fixed-h8 redundancy诊断

与上一轮完全相同的geometry-only κ和h8 grouping：每selected row取同h8 cell其他selected κ最大值，singleton为0；unordered pairs只计一次。所有arm使用同一partition/metric，不用各自leaf/root尺度。A/B/C统计与上一轮逐值exact一致。


|arm|ratio|mean|median|p90|κ>0.8|κ>0.9|κ>0.95|
|---|---|---|---|---|---|---|---|
|A_Raw|1.00%|0.246853|0.000000|0.948935|69|46|34|
|A_Raw|5.00%|0.445519|0.374900|0.987712|911|608|400|
|A_Raw|10.00%|0.514889|0.647991|0.990276|2489|1657|1039|
|A_Raw|15.00%|0.540227|0.705815|0.991629|4263|2826|1821|
|B_FixedMarginal|1.00%|0.023485|0.000000|0.017722|0|0|0|
|B_FixedMarginal|5.00%|0.099069|0.000000|0.421159|24|1|0|
|B_FixedMarginal|10.00%|0.167655|0.000003|0.651343|127|27|3|
|B_FixedMarginal|15.00%|0.195875|0.002214|0.724681|320|99|19|
|C_Hierarchical|1.00%|0.056587|0.000000|0.176776|5|2|2|
|C_Hierarchical|5.00%|0.145934|0.000000|0.607335|67|29|16|
|C_Hierarchical|10.00%|0.209354|0.000592|0.758143|243|97|48|
|C_Hierarchical|15.00%|0.246278|0.016273|0.829476|601|278|133|
|F1_MarginalTail|1.00%|0.056587|0.000000|0.176776|5|2|2|
|F1_MarginalTail|5.00%|0.150317|0.000000|0.614983|71|30|15|
|F1_MarginalTail|10.00%|0.208984|0.000647|0.752612|227|75|28|
|F1_MarginalTail|15.00%|0.233034|0.014340|0.797904|461|174|58|
|F2_ProgressiveResidual|1.00%|0.040547|0.000000|0.040076|3|1|1|
|F2_ProgressiveResidual|5.00%|0.131082|0.000000|0.550567|46|12|8|
|F2_ProgressiveResidual|10.00%|0.199823|0.000416|0.740301|214|69|33|
|F2_ProgressiveResidual|15.00%|0.230000|0.010968|0.795583|492|194|89|
|F3_ProgressiveMarginal|1.00%|0.040547|0.000000|0.040076|3|1|1|
|F3_ProgressiveMarginal|5.00%|0.132781|0.000000|0.555994|48|14|8|
|F3_ProgressiveMarginal|10.00%|0.197303|0.000487|0.726980|196|55|21|
|F3_ProgressiveMarginal|15.00%|0.225803|0.012374|0.787495|435|161|53|

F1在10%/15%的mean redundancy仅降0.1765%/5.3778%，未达两档各≥10%门槛；但κ>0.9 pairs从97/278降至75/174，强重复pair确有减少。不能因为高ratio PSNR有收益就事后降低“明显下降”标准。

F3相比C的mean redundancy在四档下降28.35%/9.01%/5.76%/8.31%；仍高于B。四档mean平均F3=0.149109，B=0.121521；κ>0.9 total pairs为231 vs127。Strong的重复度接近条件失败。

## 9. Progressive预算身份与depth

internal rep指实际emit节点为internal；final rep指实际leaf coarse phase；tail/residual按实际emit phase分类，不以它是否恰好也是C的final rep替代。


|arm|ratio|internal rep|final-leaf rep|marginal tail|independent residual|L1|L2|L3|
|---|---|---|---|---|---|---|---|---|
|F1_MarginalTail|1.00%|0|502|1|0|217|144|142|
|F1_MarginalTail|5.00%|0|2359|158|0|804|762|951|
|F1_MarginalTail|10.00%|0|4279|754|0|1397|1486|2150|
|F1_MarginalTail|15.00%|0|5916|1634|0|1903|2143|3504|
|F2_ProgressiveResidual|1.00%|282|221|0|0|462|39|2|
|F2_ProgressiveResidual|5.00%|1167|1285|0|65|1562|694|261|
|F2_ProgressiveResidual|10.00%|1886|2781|0|366|2504|1615|914|
|F2_ProgressiveResidual|15.00%|2350|4233|0|967|3149|2524|1877|
|F3_ProgressiveMarginal|1.00%|282|221|0|0|462|39|2|
|F3_ProgressiveMarginal|5.00%|1163|1279|75|0|1563|694|260|
|F3_ProgressiveMarginal|10.00%|1870|2742|421|0|2498|1604|931|
|F3_ProgressiveMarginal|15.00%|2335|4148|1067|0|3138|2490|1922|

1%时F2/F3完全相同：282 internal rep、221 final-leaf rep，无tail/residual；462/503来自L1（91.85%），L2=39、L3=2。C同档的emitted final-leaf深度为L1=217、L2=144、L3=142。这证实coarse-first约束确实改变了depth scheduling。

![实际emit depth组成](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/hierarchical_progressive_marginal_20261019/progressive_depth.png)

## 10. Spatial coverage与每root预算


|arm|ratio|selected roots|selected final leaves|hit-root SI fraction|hit-leaf SI fraction|mean rows/all roots|mean rows/hit roots|
|---|---|---|---|---|---|---|---|
|C_Hierarchical|1.00%|331|459|73.70%|54.59%|0.246|1.520|
|C_Hierarchical|5.00%|882|1669|95.58%|88.15%|1.229|2.854|
|C_Hierarchical|10.00%|1201|2668|98.93%|95.91%|2.458|4.191|
|C_Hierarchical|15.00%|1367|3355|99.54%|98.24%|3.687|5.523|
|F2_ProgressiveResidual|1.00%|333|461|74.44%|53.61%|0.246|1.511|
|F2_ProgressiveResidual|5.00%|896|1638|95.92%|87.37%|1.229|2.809|
|F2_ProgressiveResidual|10.00%|1208|2631|98.98%|95.52%|2.458|4.166|
|F2_ProgressiveResidual|15.00%|1373|3330|99.55%|98.08%|3.687|5.499|
|F3_ProgressiveMarginal|1.00%|333|461|74.44%|53.61%|0.246|1.511|
|F3_ProgressiveMarginal|5.00%|896|1636|95.92%|87.35%|1.229|2.809|
|F3_ProgressiveMarginal|10.00%|1203|2611|98.96%|95.42%|2.458|4.184|
|F3_ProgressiveMarginal|15.00%|1367|3300|99.54%|97.98%|3.687|5.523|

1% root覆盖仅331→333、leaf覆盖459→461；hit-root Raw mass从73.70%升至74.44%，但hit-leaf Raw mass从54.59%降至53.61%。因此不能把+0.104 dB的主要原因说成“大幅覆盖更多root/leaf”。证据更支持在基本相同的空间区域内，先用ancestor coarse表示、延后细节。hit-cell mass只是包含至少一个selected点的cell全部Raw mass，不是render utility。

mean rows/all roots=K/2048由same-K决定，各arm天然相同；mean rows/hit roots才反映命中region内的分配差别。

## 11. Ranking overlap与逐row变化


|比较|ratio|Jaccard|intersection|added|removed|
|---|---|---|---|---|---|
|C_Hierarchical_vs_F1_MarginalTail|1.00%|1.000000|503|0|0|
|C_Hierarchical_vs_F2_ProgressiveResidual|1.00%|0.796429|446|57|57|
|C_Hierarchical_vs_F3_ProgressiveMarginal|1.00%|0.796429|446|57|57|
|B_FixedMarginal_vs_F3_ProgressiveMarginal|1.00%|0.594295|375|128|128|
|C_Hierarchical_vs_F1_MarginalTail|5.00%|0.965638|2473|44|44|
|C_Hierarchical_vs_F2_ProgressiveResidual|5.00%|0.872768|2346|171|171|
|C_Hierarchical_vs_F3_ProgressiveMarginal|5.00%|0.870680|2343|174|174|
|B_FixedMarginal_vs_F3_ProgressiveMarginal|5.00%|0.721614|2110|407|407|
|C_Hierarchical_vs_F1_MarginalTail|10.00%|0.923194|4832|201|201|
|C_Hierarchical_vs_F2_ProgressiveResidual|10.00%|0.902116|4774|259|259|
|C_Hierarchical_vs_F3_ProgressiveMarginal|10.00%|0.882903|4720|313|313|
|B_FixedMarginal_vs_F3_ProgressiveMarginal|10.00%|0.779388|4409|624|624|
|C_Hierarchical_vs_F1_MarginalTail|15.00%|0.906806|7181|369|369|
|C_Hierarchical_vs_F2_ProgressiveResidual|15.00%|0.905363|7175|375|375|
|C_Hierarchical_vs_F3_ProgressiveMarginal|15.00%|0.873914|7042|508|508|
|B_FixedMarginal_vs_F3_ProgressiveMarginal|15.00%|0.822351|6814|736|736|

`ranking_selected_row_changes.csv`保存每个added/removed source row、xyz、Raw SI、C score/rank、新rank、C representative level、新emit身份和depth。`ranking_emit_trace.csv`保存三arm全50330row的实际priority与phase。F1与C在1% selected set和质量均exact相同；没有为每ratio单独排序。

## 12. Marginal Tail完整诊断

`marginal_tail_statistics.csv`逐final leaf保存occupancy、原C first3 Delta、当前conditional sequence的Delta1…n、Delta4/Delta5、正gain行数、Delta4/Delta3、tail总gain与Raw mass比例，并保存全部source rows。F3排除已emit ancestors后可能少于3个新的leaf reps；此时字段按实际新emit sequence定义，original_Delta仍独立保留，不能混淆为C的原始Delta。


|arm|量|有效leaf数|mean|median|p90|
|---|---|---|---|---|---|
|F1_MarginalTail|positive_gain_rows|5601|8.985895|4.000000|22.000000|
|F1_MarginalTail|Delta4|3173|31.154430|7.821946|77.477199|
|F1_MarginalTail|Delta4_over_Delta3|3173|0.498873|0.504730|0.890175|
|F1_MarginalTail|tail_total_gain|5601|60.292347|0.170225|140.038955|
|F1_MarginalTail|tail_gain_over_Raw_mass|5601|0.061864|0.002395|0.188323|
|F3_ProgressiveMarginal|positive_gain_rows|5601|8.440814|4.000000|20.000000|
|F3_ProgressiveMarginal|Delta4|2966|20.565299|4.734876|50.128633|
|F3_ProgressiveMarginal|Delta4_over_Delta3|2966|0.521007|0.535194|0.914090|
|F3_ProgressiveMarginal|tail_total_gain|5601|41.779224|0.012355|91.639525|
|F3_ProgressiveMarginal|tail_gain_over_Raw_mass|5601|0.043628|0.000286|0.143425|

当前C的Top-K non-reps在新ranking中的位置：


|arm|C budget|原C non-reps|rank上升|下降|不变|仍在新Top-K|median新priority|
|---|---|---|---|---|---|---|---|
|F1_MarginalTail|1.00%|1|0|0|1|1|1128.770788|
|F3_ProgressiveMarginal|1.00%|1|0|1|0|0|1116.013313|
|F1_MarginalTail|5.00%|126|31|94|1|114|308.201363|
|F3_ProgressiveMarginal|5.00%|126|99|27|0|113|308.201363|
|F1_MarginalTail|10.00%|662|169|492|1|553|132.677443|
|F3_ProgressiveMarginal|10.00%|662|410|252|0|562|131.642522|
|F1_MarginalTail|15.00%|1554|411|1142|1|1265|67.924379|
|F3_ProgressiveMarginal|15.00%|1554|720|832|2|1270|66.964988|

F1高budget多数原non-reps向后移动；F3部分原non-reps会变成internal或conditioned coarse rep，因此其rank上升不能一概解读为tail机制提升。逐row的marginal priority、phase/depth和原Raw/C score均已保留。

## 13. Progressive unlock trace

`progressive_unlock_trace.csv`记录每个internal node的level、Raw mass、冻结Q、实际coarse rows/Delta/global ranks、child unlock rank、child数及first direct-child item rank。节点候选和原provisional reps/child relation另存于 `frozen_tree_membership.npz` / `frozen_tree_nodes.json`。


|arm|ratio|已unlock internal|L1|L2|变为可见的children|prefix priority增加次数|
|---|---|---|---|---|---|---|
|F2_ProgressiveResidual|1.00%|41|38|3|233|0|
|F2_ProgressiveResidual|5.00%|277|184|93|1473|0|
|F2_ProgressiveResidual|10.00%|530|316|214|2664|0|
|F2_ProgressiveResidual|15.00%|699|378|321|3395|0|
|F3_ProgressiveMarginal|1.00%|41|38|3|233|0|
|F3_ProgressiveMarginal|5.00%|277|184|93|1473|0|
|F3_ProgressiveMarginal|10.00%|524|312|212|2641|0|
|F3_ProgressiveMarginal|15.00%|688|374|314|3348|0|

1%时38个L1 internal与3个L2 internal完成coarse capacity，233个child已可见；“可见”不等于都已获预算。完整trace包含高rank才unlock的复杂region，而非只保存Top-K。

## 14. 三个机制假设的判定


|假设|数值证据|预注册结论|
|---|---|---|
|F1 marginal tail|10/15%平均ΔPSNR=+0.051556 dB；mean redundancy降幅未达10%/档|NOT SUPPORTED（严格gate）；方向性质量信号存在|
|F2 progressive|1%+0.103887；5/10/15%+0.001637/+0.020477/+0.039050|PROGRESSIVE SUPPORTED|
|F3 combined|四档PSNR均正，平均+0.059593；高ratio重复度均下降|COMBINED SUPPORTED|
|Strong|平均F3−B=+0.042367，但统一重复度未接近B|NOT REACHED|

F2相对历史L1-only 1% PSNR高0.004685 dB，覆盖了此前Full−L1-only约−0.099203 dB的差距；5/10/15% PSNR均未牺牲。F3比F2在1%相同，5/10/15%继续有PSNR收益，因此组合推荐有直接质量证据。F1单独未达严格gate，不能改写成“单独Tail已明确解决重复”。

## 15. 高budget SSIM/photo及指标取舍

本轮按任务只记录，不以其反向调算法或metric：


|ratio|F3−Raw ΔPSNR|ΔSSIM|ΔLPIPS|Δphoto|
|---|---|---|---|---|
|1.00%|+0.950134|+0.009244|-0.013353|-0.016125|
|5.00%|+0.406705|+0.002508|-0.016401|-0.003057|
|10.00%|+0.138875|-0.002269|-0.026058|+0.000501|
|15.00%|+0.187848|-0.000240|-0.030125|-0.000305|

F3相对C的10/15%四项指标均改善；相对Raw，10%仍有SSIM/photo回退，15% SSIM仍略低而photo已改善。5%相对C的SSIM/LPIPS略差，1% LPIPS略差。PSNR机制gate通过不等于所有指标一致提高；本轮没有宣布无条件整体QUALITY GO。

## 16. Runtime与证据边界

只记录：本轮完整F1/F2/F3 reference构建约2.450/1.824/2.644秒（CPU NumPy FP64，单次）；原C包含GPU同步构建，路径不同，不作速度对比或优化结论。每个新arm完整ranking重复一次。质量总计6 arms×4ratio×161view=3864次真实render，其中A/B/C baseline1932次、新arm1932次。

没有写CUDA、近似KNN、batch优化或新cache策略；没有因耗时减少arm、views、ratio或改变公式。

## 17. 输出与复现

必需输出全部具备：progressive_marginal_quality_summary.csv、progressive_marginal_per_view.csv、ranking_overlap.csv、selected_redundancy.csv、progressive_composition.csv、progressive_unlock_trace.csv、marginal_tail_statistics.csv、marginal_tail_topk_rows.csv、spatial_budget_coverage.csv、mechanism_preregistration.json、mechanism_result.json。

另有paired逐view和分布、全emit trace、完整permutation/ordinal/gain、frozen node membership、逐row增删、synthetic与independent replay、SHA清单及原C结构快照。完整证据位于本地 `hierarchical-progressive-marginal-20261019` 与服务器 `/home/prmi/zwj/hierarchical-progressive-marginal-20261019`，项目内保留同名diagnostic目录；不提交Git。

脚本执行顺序：prepare.py → generate_rankings.py → baseline.py → test_progressive.py → new_rankings.py → replay_checks.py → quality.py → diagnostics.py → analyze.py。baseline gate不通过时禁止进入新机制阶段。Renderer依赖旧冻结runtime；单独ranking/replay只需NumPy和冻结fixture，环境路径见脚本。

## 18. 已知限制与下一步机制结论

单个冻结fr1_room归档协议，未复验第二scene，未训练Selector/Teacher，未测online、ATE、refresh或production稳定性。本轮推荐ADOPT BOTH仅表示后续机制候选，正式C保持原样。

主证据更偏向 **depth scheduling**：F2明确修复1%损失且未损害较高ratio。Residual ordering有高ratio与组合依赖的次要收益，但独立F1的mean redundancy改善不足预定门槛。F3的剩余统一重复度高于B，不能把剩余误差唯一归因于Tail；不同competition boundaries与active constraint也可能贡献，当前实验未进一步拆解。

Progressive质量收益不等价于大幅增加root coverage，当前主要变化是在相近的root集合内使用coarse ancestor。绝不因该结果调整Q、2048、3rep、depth、h或新增progressive权重。本轮结束，不自动开展下一算法、性能、online或参数搜索。

## 19. 最终明确回答


|问题|回答|依据|
|---|---|---|
|Q1 Marginal Tail降低重复/提高质量？|MIXED|高ratio质量提高、strong pairs减少；mean redundancy下降幅度未达预注册门槛|
|Q2 Progressive修复1%相对L1-only损失？|YES|F2−C +0.103887 dB；比历史L1-only +0.004685 dB|
|Q3 Progressive伤害5/10/15%？|NO|按PSNR gate均无回退；其它指标逐项记录|
|Q4 F3整体优于C？|YES|四档PSNR均正、平均+0.059593 dB，高ratio重复度下降|
|Q5 F3相对Fixed h8？|EXCEEDS|PSNR平均+0.042367 dB；重复度及部分LPIPS未超过B|
|Q6 主要机制瓶颈？|depth scheduling|depth scheduling主证据强；tail有次要、组合依赖收益|
|Q7 下一步？|ADOPT BOTH|推荐组合为后续研究候选；本轮正式C/production保持原样|
