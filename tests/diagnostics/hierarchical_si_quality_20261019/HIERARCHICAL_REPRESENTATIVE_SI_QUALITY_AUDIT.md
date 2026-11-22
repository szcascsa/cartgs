# Hierarchical Representative SI：Frozen Same-K Reconstruction Quality Audit

结论：**整体QUALITY INCONCLUSIVE；低budget收益明确，不认定Strong GO。** Hierarchical相对Raw四档PSNR和LPIPS改善，1%/5%四项指标均改善、平均PSNR提升0.620024 dB；但10%/15%出现SSIM和photo回退，按任务原文“指标方向混合必须降级”作保守判定。PSNR表现接近Case A，不能将其写成无条件的整体Case A成功。

与Fixed h8 Marginal的四档平均PSNR差为−0.017225 dB，满足预注册“接近”标准。该MATCHES仅表示预定义质量距离，不表示ranking、selected set或每项指标相同。

本轮仅评价冻结final map；没有训练Gaussian/Selector/TF，没有重算GI，没有修改算法、参数、Bank、Master或production代码。未commit、未stage、未push。运行时间未参与任何GO判定。

## 1. 研究问题与预注册

Q1比较C相对Raw的same-K质量；Q2比较C与已验证的Fixed h8 Marginal；Q3利用D/E、residual组成、代表层级、统一重复度和空间budget解释机制。结构Q不作为质量gate。

`preregistration.json`在质量渲染前写入，SHA256为 `ebb36b8fdb16b0583e59ffd27c92b78127c7565127e24d3d23c3a2e374bfc6cb`。Primary要求至少3/4档ΔPSNR≥−0.02 dB、低ratio平均ΔPSNR>+0.10 dB或满足+0.20/−0.02替代条件，且无系统性SSIM/LPIPS交换。Strong另要求四档非负/等价、LPIPS多数改善、低ratio paired明显偏正、重复度下降、接近FixedMarginal。

将“明显”固定为低ratio paired PSNR改善视角≥60%且median>0，mean redundancy至少下降10%且κ>0.9 pairs减少。指标交换的诊断容差固定为SSIM 0.002、LPIPS 0.005；未调整这些阈值。

**判读修正记录：** 初始量化实现只在低ratio或至少两个ratio超幅度容差时降级，范围窄于任务原文“PSNR/SSIM/LPIPS方向混合必须降级”。复核发现高ratio确有混合方向，因此保留原始preregistration、全部数值gate和结果，按优先级更高的用户原文进行保守降级。`mechanism_cases.json`同时记录量化gate通过和最终INCONCLUSIVE；不重设阈值、不修改算法、不重跑/筛选质量结果，也不把修正后的判读伪称为最初代码已正确实现。

## 2. 冻结输入、SHA与runtime恢复

HEAD：`40751d07e194aaaa997d06bdea14d90f813b1222`，分支 `redundancy-aware-si`。工作树保留上一轮未提交改动，详细开始状态见 `audit_before.json`。

完全复用fr1_room final `16900_shutdown`：Master topology、xyz、activated scale、opacity/SH/rotation、TF参数及AABB buffers、69-view贡献矩阵、Raw SI、mask、evaluation cameras与GT。实际N_total=52635，valid AND observed候选N=50330，effective maturity=1。



|冻结输入|SHA256|
|---|---|
|Master|73f23088aab5c39c9864cf3a0b0fb1fab6384ae162bc848dadeefec5f0b1c4bb|
|TF（含AABB buffers）|a01c7b908dee8a9fd29603babe49fb5a2c99e8e0010072d8ef13b85429cfa4bd|
|camera/GT archive|90e74ad824dddcdd2a740d004082d747f170ce912aaaff07e99f6c1c9e5caabf|
|checkpoint config|f6453a80643323f7e579b78787da5b19208db4a7f0ba2f2a9aafe6fa29381953|
|69-view contribution matrix|146ddbfa407e5c933d95c9b30de8b9886e929c6dae240dca63f521ced26169d5|
|Raw/geometry ranking input|c89a9e157e3dde1eaca80a17153f68821e5e7ae7a71d7383304ef608dafaece4|
|hierarchy binary input|e54e055fd9be3a211ba87cf6997fd088960659ff3f415d2affeebc7636d09a55|

发现当前production mapper库SHA已从旧审计的 `523bba…` 变为 `2f381f…`，因此暂停启动质量渲染。随后从历史 `geometry-marginal-online-20261019/runtime_before/lib` 找到字节完全一致的旧mapper与rasterizer库，复制到本轮隔离runtime，通过LD_LIBRARY_PATH选择，并核对ldd及实际worker `/proc/PID/maps`。未覆盖当前production库。使用的旧mapper SHA为 `523bba9fea150989e07f93eb4977e95bbd91cafb4c58cc0ed3fcdb2964345b1c`；rasterizer为 `019cbd9fb052c8fa01dd03a977ec0f0c262cb01c963c03e2bb87c4e383f5e6a7`。

全部实际冻结输入SHA匹配后才继续。配置/camera清单中69 Bank与161 evaluation frames严格不交叉并覆盖归档230个KF；评价集合完整复用旧协议。正确称为 **fixed non-Bank archived evaluation views**：这些视角参与过原SLAM，不将其视为新的独立采集数据。

## 3. Arm定义与TF/renderer语义


|Arm|冻结定义|
|---|---|
|A Raw SI|I=mean_v(c_iv)，直接降序，source row打破tie|
|B Fixed h8 Marginal|旧GeometryOnlyReference：固定h8=8*d50、geometry-only κ、完整local marginal greedy及保留predecessor的global merge|
|C Hierarchical|当前SparseHash2048/3rep/Q<0.80/L3，代表score=真实Delta，non-rep=I*(1-cover)，直接全局score降序|
|D Representative-only|仅把C的non-rep score置0；相同tree/reps/Delta，不替代primary|
|E L1-only|直接复用C每个root已记录的临时3-rep与Delta，无split；按同一公式重分配，仅diagnostic|

所有score tie按source row升序；B沿用其冻结global merge tie规则并验证完整local predecessor顺序。C没有第二次marginal ranking。E没有改production depth/threshold，只用已有root计算记录构造诊断score。

每arm每ratio使用相同Master、requested ratio、同一冻结TF consumer与全1 maturity；保留原TF candidate/AABB筛选，selected rows先按source row排序再compact render。worker在退出前逐tensor检查Master及TF参数/buffers未改变。

指标沿用原worker/common：PSNR为逐view MSE的dB后取mean；SSIM与LPIPS使用旧PyTorch指标；production photo为0.8*L1+0.2*(1-fast_ssim)。photo在原render tensor计算，质量指标按原协议clamp到[0,1]。

## 4. Same-K、nested与结构复现


|ratio|N_candidate|K=floor(r*N+0.5)|全部arm exact K / nested|
|---|---|---|---|
|1.00%|50330|503|PASS|
|5.00%|50330|2517|PASS|
|10.00%|50330|5033|PASS|
|15.00%|50330|7550|PASS|

每个arm均生成完整50330行确定性permutation，四档是同一ranking前缀。C在当前binary重新构建两轮，每轮内部另有重复检查：完整score、assignment、tree及complete ranking exact一致，并与上一轮SparseHash fixture exact一致。


|结构量|本轮复现|
|---|---|
|occupied roots|2048|
|h_root (m)|0.299867557464|
|final leaves|5601|
|final representatives|13715|
|Raw-SI weighted leaf Q|88.38%|

Input Raw/xyz/activated scale及candidate rows与上轮完全一致。Q和这些结构量只用于确认复现，不参与质量GO。

## 5. FixedMarginal与Raw基线复现

重新执行旧GeometryOnlyReference的完整local greedy，完整order/gain/local rank/cell/key与旧GeometryOnly8及已冻结reference.npz逐值一致。随后独立重渲染A/B，四档各161视角；photo/PSNR/SSIM/LPIPS每项最大逐view差均为 **0**。只有该gate通过后才渲染C/D/E。详见 `baseline_reproduction.json`。

## 6. Primary质量表


|ratio|K|Raw PSNR|FixedMarginal PSNR|Hierarchical PSNR|C−Raw dB|C−Fixed dB|
|---|---|---|---|---|---|---|
|1.00%|503|12.988268|13.882451|13.834515|+0.846247|-0.047936|
|5.00%|2517|16.653142|16.980360|17.046943|+0.393800|+0.066582|
|10.00%|5033|17.680354|17.802212|17.760086|+0.079732|-0.042126|
|15.00%|7550|18.054710|18.225545|18.180123|+0.125413|-0.045422|



|ratio|Raw ssim|FixedMarginal ssim|Hierarchical ssim|
|---|---|---|---|
|1.00%|0.552427|0.562086|0.561118|
|5.00%|0.616889|0.618974|0.619452|
|10.00%|0.640308|0.637135|0.637632|
|15.00%|0.649803|0.648584|0.648521|



|ratio|Raw lpips|FixedMarginal lpips|Hierarchical lpips|
|---|---|---|---|
|1.00%|0.752445|0.736327|0.738927|
|5.00%|0.648476|0.630773|0.631498|
|10.00%|0.599721|0.576634|0.576373|
|15.00%|0.571317|0.542243|0.544375|



|ratio|Raw photo|FixedMarginal photo|Hierarchical photo|
|---|---|---|---|
|1.00%|0.219589|0.203461|0.204326|
|5.00%|0.153112|0.150699|0.150241|
|10.00%|0.137312|0.138246|0.138309|
|15.00%|0.131872|0.131947|0.132179|

## 7. 四项指标差值

差值均为前者减后者。PSNR/SSIM正值改善，LPIPS/photo负值改善。


|比较|ratio|ΔPSNR|ΔSSIM|ΔLPIPS|Δphoto|
|---|---|---|---|---|---|
|C_vs_Raw|1.00%|+0.846247|+0.008691|-0.013518|-0.015263|
|C_vs_Raw|5.00%|+0.393800|+0.002562|-0.016978|-0.002871|
|C_vs_Raw|10.00%|+0.079732|-0.002677|-0.023348|+0.000997|
|C_vs_Raw|15.00%|+0.125413|-0.001282|-0.026942|+0.000307|
|C_vs_FixedMarginal|1.00%|-0.047936|-0.000967|+0.002600|+0.000865|
|C_vs_FixedMarginal|5.00%|+0.066582|+0.000478|+0.000726|-0.000458|
|C_vs_FixedMarginal|10.00%|-0.042126|+0.000496|-0.000262|+0.000062|
|C_vs_FixedMarginal|15.00%|-0.045422|-0.000063|+0.002133|+0.000232|

C相对Raw：1%/5%四项指标同向改善；10%/15%的PSNR及LPIPS改善，而SSIM分别下降0.002677/0.001282、photo分别增加0.000997/0.000307。因此不能用PSNR收益覆盖指标交换。相对FixedMarginal，C在5% PSNR/SSIM/photo略好；其他PSNR略低，LPIPS在1/5/15%略差、10%略好。MATCHES不等于全面超过B。

FixedMarginal自身在10%/15%相对Raw也出现SSIM和photo回退：SSIM约−0.003173/−0.001219，photo约+0.000935/+0.000075。因此这个高budget权衡不是C独有的新现象；本轮仍按当前任务的严格mixed-direction条款降级，不能拿历史B的GO标签替代当前评价规则。

![四项指标的方向与高budget交换](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/hierarchical_si_quality_20261019/metric_tradeoffs.png)

## 8. Paired per-view统计

全部比较严格配对同161个frame_index。下表“改善视角”按指标方向：PSNR/SSIM差值>0，LPIPS/photo差值<0；CSV另存原始positive-delta fraction，避免对lower-is-better指标产生歧义。


|比较|ratio|指标|mean Δ|median Δ|p10 Δ|p90 Δ|改善视角|
|---|---|---|---|---|---|---|---|
|C_vs_Raw|1.00%|psnr|+0.846247|+0.638310|-0.316733|+2.254586|78.88%|
|C_vs_Raw|1.00%|ssim|+0.008691|+0.005118|-0.011717|+0.033824|68.94%|
|C_vs_Raw|1.00%|lpips|-0.013518|-0.009543|-0.046836|+0.010290|73.29%|
|C_vs_Raw|1.00%|photo|-0.015263|-0.011153|-0.040378|+0.003910|78.26%|
|C_vs_Raw|5.00%|psnr|+0.393800|+0.329634|-0.033308|+0.953476|86.96%|
|C_vs_Raw|5.00%|ssim|+0.002562|+0.002803|-0.004835|+0.010445|67.70%|
|C_vs_Raw|5.00%|lpips|-0.016978|-0.015785|-0.034953|+0.001505|88.20%|
|C_vs_Raw|5.00%|photo|-0.002871|-0.002670|-0.008331|+0.003842|75.78%|
|C_vs_Raw|10.00%|psnr|+0.079732|+0.092148|-0.369482|+0.463543|58.39%|
|C_vs_Raw|10.00%|ssim|-0.002677|-0.003104|-0.011412|+0.007756|31.68%|
|C_vs_Raw|10.00%|lpips|-0.023348|-0.024835|-0.043611|-0.003025|95.03%|
|C_vs_Raw|10.00%|photo|+0.000997|+0.001454|-0.004791|+0.005822|36.02%|
|C_vs_Raw|15.00%|psnr|+0.125413|+0.008883|-0.275738|+0.645319|53.42%|
|C_vs_Raw|15.00%|ssim|-0.001282|-0.001615|-0.009139|+0.006215|36.02%|
|C_vs_Raw|15.00%|lpips|-0.026942|-0.027560|-0.048664|-0.004915|96.89%|
|C_vs_Raw|15.00%|photo|+0.000307|+0.001226|-0.005094|+0.004693|40.37%|



|比较|ratio|指标|mean Δ|median Δ|p10 Δ|p90 Δ|改善视角|
|---|---|---|---|---|---|---|---|
|C_vs_FixedMarginal|1.00%|psnr|-0.047936|-0.033821|-0.526704|+0.720327|45.96%|
|C_vs_FixedMarginal|1.00%|ssim|-0.000967|-0.000316|-0.011267|+0.009699|45.96%|
|C_vs_FixedMarginal|1.00%|lpips|+0.002600|+0.002155|-0.008027|+0.013758|40.37%|
|C_vs_FixedMarginal|1.00%|photo|+0.000865|+0.001240|-0.011591|+0.010255|40.37%|
|C_vs_FixedMarginal|5.00%|psnr|+0.066582|+0.031456|-0.147879|+0.345619|57.76%|
|C_vs_FixedMarginal|5.00%|ssim|+0.000478|+0.000491|-0.002873|+0.003877|57.14%|
|C_vs_FixedMarginal|5.00%|lpips|+0.000726|+0.000300|-0.009318|+0.010598|48.45%|
|C_vs_FixedMarginal|5.00%|photo|-0.000458|-0.000184|-0.003587|+0.001675|54.66%|
|C_vs_FixedMarginal|10.00%|psnr|-0.042126|-0.050711|-0.222204|+0.158966|37.27%|
|C_vs_FixedMarginal|10.00%|ssim|+0.000496|+0.000191|-0.002365|+0.003532|52.17%|
|C_vs_FixedMarginal|10.00%|lpips|-0.000262|+0.000416|-0.008230|+0.007543|46.58%|
|C_vs_FixedMarginal|10.00%|photo|+0.000062|+0.000094|-0.001763|+0.002001|45.34%|
|C_vs_FixedMarginal|15.00%|psnr|-0.045422|-0.035023|-0.237507|+0.124138|36.65%|
|C_vs_FixedMarginal|15.00%|ssim|-0.000063|+0.000099|-0.002807|+0.002460|52.17%|
|C_vs_FixedMarginal|15.00%|lpips|+0.002133|+0.001097|-0.004873|+0.010863|40.99%|
|C_vs_FixedMarginal|15.00%|photo|+0.000232|+0.000153|-0.001286|+0.002004|44.72%|

每view四指标差值保存在 `paired_per_view_deltas.csv`，原始值在 `quality_per_view.csv`。归档KF在空间/时间上相关，不能将161个paired样本当作161个独立场景。

## 9. Ranking overlap与增删rows


|比较|ratio|Jaccard|intersection|added rows|removed rows|
|---|---|---|---|---|---|
|C_vs_A_Raw|1.00%|0.368707|271|232|232|
|C_vs_B_FixedMarginal|1.00%|0.579278|369|134|134|
|C_vs_A_Raw|5.00%|0.484518|1643|874|874|
|C_vs_B_FixedMarginal|5.00%|0.694379|2063|454|454|
|C_vs_A_Raw|10.00%|0.503510|3371|1662|1662|
|C_vs_B_FixedMarginal|10.00%|0.736116|4268|765|765|
|C_vs_A_Raw|15.00%|0.514088|5127|2423|2423|
|C_vs_B_FixedMarginal|15.00%|0.759702|6519|1031|1031|

`ranking_selected_set_changes.csv`逐row保存增删方向、source row ID、Raw I、hierarchical score、代表标签/层级、leaf/root及xyz；不是只给Jaccard或只给数量。

## 10. 统一geometry重复度

对所有arm使用同一个冻结Fixed-h8分组和scalar-radius κ：selected Gaussian的redundancy是同h8 cell中其余selected κ的最大值，singleton为0；unordered pair只计一次。该诊断不使用view、C的leaf尺寸或自适应阈值，不统计跨h8 boundary的pairs。


|arm|ratio|mean|median|p90|κ>0.8 pairs|κ>0.9 pairs|κ>0.95 pairs|
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

C相对Raw明显减少重复选择，但各档重复度仍高于B；更低重复度不保证每个ratio都更高PSNR，例如5%的C比B重复度高而PSNR更好。这里不存在把“dense region选得更少”当作成功条件。

## 11. Representative与residual组成


|ratio|K|final-leaf reps|rep fraction|non-rep residual|residual fraction|
|---|---|---|---|---|---|
|1.00%|503|502|99.80%|1|0.20%|
|5.00%|2517|2391|94.99%|126|5.01%|
|10.00%|5033|4371|86.85%|662|13.15%|
|15.00%|7550|5996|79.42%|1554|20.58%|

各档全部selected score为正；D也没有为了凑K选入零score non-rep。composition CSV中的所有arm统一按C的final-leaf代表身份分类；E行不是E自身L1代表占比。

## 12. Final representative层级

下表count/fraction的fraction以该档C选中的representatives为分母；split parent的临时代表不计入C。CSV另存占K的比例。


|ratio|level|count|占selected representatives|
|---|---|---|---|
|1.00%|1|217|43.23%|
|1.00%|2|144|28.69%|
|1.00%|3|141|28.09%|
|5.00%|1|784|32.79%|
|5.00%|2|737|30.82%|
|5.00%|3|870|36.39%|
|10.00%|1|1258|28.78%|
|10.00%|2|1360|31.11%|
|10.00%|3|1753|40.11%|
|15.00%|1|1577|26.30%|
|15.00%|2|1823|30.40%|
|15.00%|3|2596|43.30%|

1%同时使用L1/L2/L3代表，未出现预算全部被粗层或细层单独占据；层级计数用于解释，不以层级分布本身判断质量。

## 13. Residual score诊断

non-rep residual_ratio=I′/(I+eps)，近似1-cover；仅对non-rep统计。


|ratio|population|count|mean|median|p10|p90|cover≥0.9|cover≥0.95|cover≥0.99|
|---|---|---|---|---|---|---|---|---|---|
|all|all_non_representatives|36615|0.561619|0.622781|0.019927|1.000000|8595|6163|2385|
|1.00%|selected_non_representatives|1|0.999714|0.999714|0.999714|0.999714|0|0|0|
|1.00%|unselected_non_representatives|36614|0.561607|0.622754|0.019926|1.000000|8595|6163|2385|
|5.00%|selected_non_representatives|126|0.733655|0.775531|0.378803|1.000000|0|0|0|
|5.00%|unselected_non_representatives|36489|0.561025|0.621233|0.019831|1.000000|8595|6163|2385|
|10.00%|selected_non_representatives|662|0.637843|0.670430|0.210033|0.999997|16|4|0|
|10.00%|unselected_non_representatives|35953|0.560215|0.620284|0.019357|1.000000|8579|6159|2385|
|15.00%|selected_non_representatives|1554|0.649906|0.712875|0.166162|1.000000|62|12|0|
|15.00%|unselected_non_representatives|35061|0.557706|0.613593|0.018547|1.000000|8533|6151|2385|

1%仅1个non-rep，其residual_ratio约0.999714，基本没有被代表覆盖；5%的126个selected non-reps没有cover≥0.9者。10%/15%分别出现16/62个cover≥0.9者，但所有ratio均无selected non-rep的cover≥0.99。当前低budget不符合“大Raw SI的高cover duplicate经residual大规模回流”的模式。

## 14. Representative Delta诊断

全部13715个final reps逐row保存于 `representative_delta_statistics.csv`：Raw I_r、Delta、Delta/(I_r+eps)、level、leaf occupancy/Q/mass、Raw与C全局rank。

Delta与Raw I的Spearman=0.891098070。这说明二者相关但发生了实质性重排；并非重新命名Raw score。


|ratio|K（取全candidate档位）|仅在代表集合内：Delta Top-K vs Raw Top-K Jaccard|
|---|---|---|
|1.00%|503|0.428977|
|5.00%|2517|0.692670|
|10.00%|5033|0.751218|
|15.00%|7550|0.822131|

这里两条ranking的population均限定在13715个representatives，K使用全candidate实验档位；CSV还保存以representative population自身ratio计算K的版本，避免混淆。

## 15. Selected leaf与Raw-SI mass coverage


|arm|ratio|selected leaves / 5601|leaf fraction|hit-leaf Raw SI / global SI|
|---|---|---|---|---|
|A_Raw|1.00%|352|6.28%|40.61%|
|B_FixedMarginal|1.00%|416|7.43%|50.11%|
|C_Hierarchical|1.00%|459|8.19%|54.59%|
|A_Raw|5.00%|1139|20.34%|76.50%|
|B_FixedMarginal|5.00%|1522|27.17%|85.49%|
|C_Hierarchical|5.00%|1669|29.80%|88.15%|
|A_Raw|10.00%|1847|32.98%|88.71%|
|B_FixedMarginal|10.00%|2476|44.21%|94.54%|
|C_Hierarchical|10.00%|2668|47.63%|95.91%|
|A_Raw|15.00%|2431|43.40%|93.83%|
|B_FixedMarginal|15.00%|3214|57.38%|97.64%|
|C_Hierarchical|15.00%|3355|59.90%|98.24%|

hit-leaf mass把至少有一个selected点的leaf全部Raw mass计入，是空间分布诊断，不等于实际render贡献覆盖或设施目标F。

## 16. Dense-region budget share

运行前固定：Fixed-h8 occupied cells中occupancy最高的前ceil(10%)为dense；按Raw-SI mass降序取最短前缀覆盖≥80%总mass为important。所有tie按key升序。另对Sparse roots重复相同规则，并保存top10 dense明细；不看quality定义region。


|region|arm|ratio|selected count|budget share|区域内统一κ mean redundancy|
|---|---|---|---|---|---|
|dense_top10percent|A_Raw|1.00%|134|26.64%|0.511576|
|dense_top10percent|A_Raw|5.00%|1041|41.36%|0.609344|
|dense_top10percent|A_Raw|10.00%|2290|45.50%|0.662137|
|dense_top10percent|A_Raw|15.00%|3573|47.32%|0.680114|
|dense_top10percent|B_FixedMarginal|1.00%|200|39.76%|0.050468|
|dense_top10percent|B_FixedMarginal|5.00%|985|39.13%|0.180206|
|dense_top10percent|B_FixedMarginal|10.00%|2078|41.29%|0.273407|
|dense_top10percent|B_FixedMarginal|15.00%|3263|43.22%|0.295244|
|dense_top10percent|C_Hierarchical|1.00%|199|39.56%|0.117168|
|dense_top10percent|C_Hierarchical|5.00%|1031|40.96%|0.239574|
|dense_top10percent|C_Hierarchical|10.00%|2081|41.35%|0.299707|
|dense_top10percent|C_Hierarchical|15.00%|3291|43.59%|0.341008|
|important_cover80percent|A_Raw|1.00%|503|100.00%|0.246853|
|important_cover80percent|A_Raw|5.00%|2235|88.80%|0.501732|
|important_cover80percent|A_Raw|10.00%|4019|79.85%|0.617482|
|important_cover80percent|A_Raw|15.00%|5545|73.44%|0.658348|
|important_cover80percent|B_FixedMarginal|1.00%|503|100.00%|0.023485|
|important_cover80percent|B_FixedMarginal|5.00%|1967|78.15%|0.126770|
|important_cover80percent|B_FixedMarginal|10.00%|3202|63.62%|0.258342|
|important_cover80percent|B_FixedMarginal|15.00%|4388|58.12%|0.314358|
|important_cover80percent|C_Hierarchical|1.00%|502|99.80%|0.056700|
|important_cover80percent|C_Hierarchical|5.00%|1981|78.70%|0.185170|
|important_cover80percent|C_Hierarchical|10.00%|3235|64.28%|0.302457|
|important_cover80percent|C_Hierarchical|15.00%|4484|59.39%|0.362681|

空间名额分配和内部去重复分别报告；区域candidate密度不参与split或score。`region_definitions.json`保存明确cell/root列表，`dense_region_budget_share.csv`包含全部arm与两种空间分组。

具体地，1%时dense前10% cells中的budget从Raw的134/503（26.64%）增至C的199/503（39.56%），区域内κ mean redundancy却从0.511576降至0.117168；5%则在近似预算1041→1031下从0.609344降至0.239574。这支持“重要dense区域可以保留甚至增加名额，同时内部减少重复”的机制，而非简单density penalty。

## 17. Arm D：是否应去掉residual


|ratio|C PSNR|D PSNR|D−C PSNR|C/D Jaccard|D rep fraction|C redundancy|D redundancy|
|---|---|---|---|---|---|---|---|
|1.00%|13.834515|13.841405|+0.006891|0.996032|100.00%|0.056587|0.059447|
|5.00%|17.046943|17.022405|-0.024537|0.904654|100.00%|0.145934|0.141805|
|10.00%|17.760086|17.653783|-0.106304|0.767515|100.00%|0.209354|0.167812|
|15.00%|18.180123|17.938483|-0.241640|0.658612|100.00%|0.246278|0.175255|



|ratio|D−C ΔSSIM|D−C ΔLPIPS|D−C Δphoto|
|---|---|---|---|
|1.00%|+0.000024|+0.000056|-0.000055|
|5.00%|-0.000507|-0.000321|+0.000583|
|10.00%|-0.003353|-0.001030|+0.002231|
|15.00%|-0.006297|-0.000646|+0.004103|

D在1%只增加0.006891 dB，5/10/15%分别下降0.024537/0.106304/0.241640 dB，不满足“D明显优于C”。高ratio虽然D重复度更低，质量反而下降，支持residual保留有用未覆盖结构；不能据此把正式non-rep score置0。Case E不成立。

## 18. Arm E：Octree的可观察价值


|ratio|L1-only PSNR|full hierarchy PSNR|C−E ΔPSNR|ΔSSIM|ΔLPIPS|Δphoto|
|---|---|---|---|---|---|---|
|1.00%|13.933717|13.834515|-0.099203|-0.000077|-0.001146|+0.000236|
|5.00%|16.920159|17.046943|+0.126783|+0.000772|-0.004770|-0.000881|
|10.00%|17.723632|17.760086|+0.036454|-0.001146|-0.011946|+0.000370|
|15.00%|18.100419|18.180123|+0.079704|-0.000498|-0.011231|+0.000051|

Full hierarchy在5/10/15%提高PSNR，LPIPS全部四档优于L1-only；但1% PSNR下降0.099203 dB，且SSIM并非每档改善。可回答有可观察的质量价值，但不支持“split对所有预算都更好”。Full与L1-only未达到预注册近似等价，Case F不成立。

## 19. Case分类与质量gate


|Case|判定|依据|
|---|---|---|
|A|有条件接近|低ratio及PSNR/LPIPS支持，PSNR距离MATCHES FixedMarginal；高ratio指标混合，整体结论降级|
|B|NO|不是明显低于FixedMarginal的PSNR档位|
|C|NO|不是所有质量都近似Raw，低ratio有清晰增益|
|D|NO|未出现PSNR整体回退，但不能忽视高ratio SSIM/photo损失|
|E|NO|Representative-only未明显优于C，较高ratio反而更差|
|F|NO|Full与L1-only在多档质量不同|



|Primary/Strong gate|结果|
|---|---|
|A|True|
|B|True|
|C_no_joint_systematic_tradeoff|True|
|quantitative_mixed_downgrade|False|
|user_required_direction_mixed_downgrade|True|
|all_PSNR_nonnegative_or_equivalent|True|
|low_ratio_gain|True|
|LPIPS_majority_improved|True|
|paired_low_bias_positive|True|
|redundancy_reduced|True|
|near_or_above_FixedMarginal|True|

低ratio平均ΔPSNR=+0.620024 dB；四ratio平均C−B=-0.017225 dB。数值Strong条目通过，但原文mixed-direction降级条款触发，不能宣布Strong GO。最终 **QUALITY INCONCLUSIVE；Strong GO=False**。

## 20. Runtime与完整性

本轮不以runtime判GO。


|记录量|时间|
|---|---|
|B完整reference重算|1258.630 ms|
|C一次完整build|8951.455 ms|
|20个arm-ratio、3220次view render及指标|27.560 s|

hierarchy程序使用上轮已验证binary，源码SHA与当前本地core一致；排序及tree两轮重建通过。worker完整检查frozen Master/TF未变，源码及输入在结束时再次SHA核对。图表只从最终CSV读取，不改变实验。

最终 `final_validation.json` 独立核对240个冻结输入/worker条目、原ranking archive SHA、完整input manifest、same-K/nested以及3220条逐view指标与20个aggregate的一致性。`initial_quantitative_cases.json`仅保留判读修正前的审计记录，最终结论以 `mechanism_cases.json` 为准。

## 21. 输出文件

必需文件全部保存：`quality_summary.csv`、`quality_per_view.csv`、`ranking_overlap.csv`、`selected_redundancy.csv`、`hierarchical_topk_composition.csv`、`representative_level_composition.csv`、`residual_statistics.csv`、`representative_delta_statistics.csv`、`leaf_budget_coverage.csv`、`mechanism_cases.json`。

额外保存完整ranking/selected rows、逐row增删、paired delta/statistics、dense-region budget、代表Delta统计/Jaccard、pre-registration、SHA/runtime/TF验证以及结构重建输出。

![冻结same-K质量与机制](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/hierarchical_si_quality_20261019/quality_and_mechanism.png)

![Raw与Hierarchical完整rank重排](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/diagnostics/hierarchical_si_quality_20261019/raw_vs_hierarchical_rank.png)

## 22. 已知限制与机制结论

本结论仅适用于一个冻结fr1_room final map和固定归档评价协议。未检验第二scene、online teacher训练、refresh staleness、ATE或production运行；不能从本轮quality结论推导在线/production结论。

Hierarchical与Fixed h8使用不同competition partition，且C只记录每leaf前三步Delta，再保留residual；B生成完整conditional marginal序列。两者并非等价算法。C的重复度仍高于B，少量质量差距可能同时来自partition、代表容量与score近似，当前消融不能唯一归因。

低budget由final representatives主导，residual主要来自未充分覆盖的信息；D不支持删除residual。Octree具有ratio相关的可观察收益，不能宣称对1%也必然有利。高ratio的SSIM/photo交换同时见于B，尚无证据将其唯一归因于hierarchical representative/residual/split某一层。

下一步首先是明确是否接受这种高budget指标取舍，再判断近似/容量分配差异能否跨场景稳定；不是继续调整Grid、0.80、2048、rep数或residual公式。本轮到此停止，不自动开展下一轮实验。

## 23. 最终明确回答


|问题|回答|限定|
|---|---|---|
|Q1 Hierarchical优于Raw？|INCONCLUSIVE|低ratio YES；整体PSNR/LPIPS改善，但10/15% SSIM/photo回退|
|Q2 1%/5%稳定收益？|YES|四项mean和paired均支持；单冻结scene|
|Q3 接近/超过Fixed h8？|MATCHES|四档平均PSNR差−0.017225 dB；不是所有指标完全一致|
|Q4 代表真正进入Top-K？|YES|低budget代表99.80%/94.99%，不是大量高cover residual回流|
|Q5 Adaptive Octree有质量价值？|YES|5/10/15%有PSNR收益；1%负收益，不能泛化为全预算胜出|
|Q6 Case分类|Case A (conditional: mixed metrics)|接近A的条件性表现，原文指标混合条款触发；不属于E或F|
|Q7 最终|QUALITY INCONCLUSIVE|不认定整体QUALITY GO或Strong GO，不外推至online/production|

