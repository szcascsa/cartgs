# 四Head复现、统一80宽头与基础Rotation输入实验

本轮完成七次新增训练，复用一次匹配的统一32头参考。全部fr1_xyz冻结地图、seed3100/3101、12000真实LoD步，每ratio3000曝光。两个seed同时控制网络初始化与Bank帧序，不是只改变顺序，也不是两张独立SLAM地图。

## 主要结果

- 四Head相对统一32头：两个seed分别 **+0.820/+0.422 dB**，平均 **+0.621 dB**。
- 统一80宽头相对统一32头：分别 **+0.393/+0.227 dB**，平均 **+0.310 dB**。
- 等参数量附近，四Head相对统一80宽头：分别 **+0.426/+0.195 dB**，平均 **+0.311 dB**。
- 基础Rotation输入相对不加Rotation的四Head：分别 **-0.295/-0.099 dB**，平均 **-0.197 dB**。

以上都是七档等权、non-Bank、deterministic same-K。各ratio、其他指标、Bank留出、原Cached实际预算与逐视图配对分布完整保存在CSV；不能只凭平均值或两个seed认定统计显著。

## 训练ratio与未见ratio

| 比较 | 训练四档ΔPSNR | 未见三档ΔPSNR | 内部Bank留出ΔPSNR |
| --- | --- | --- | --- |
| split4 − unified32 | +0.543 | +0.724 | +0.585 |
| unified80 − unified32 | +0.247 | +0.395 | +0.359 |
| split4 − unified80 | +0.297 | +0.329 | +0.226 |
| split4_rotation − split4 | -0.301 | -0.057 | -0.084 |

## 每个seed的四指标

| 方案 | seed | PSNR↑ | SSIM↑ | LPIPS↓ | photo↓ |
| --- | --- | --- | --- | --- | --- |
| unified32 | 3100 | 16.44933 | 0.57862 | 0.62675 | 0.16531 |
| unified32 | 3101 | 16.74537 | 0.59115 | 0.61691 | 0.15833 |
| unified80 | 3100 | 16.84266 | 0.59256 | 0.61117 | 0.15728 |
| unified80 | 3101 | 16.97208 | 0.59061 | 0.61625 | 0.15705 |
| split4 | 3100 | 17.26895 | 0.59778 | 0.60214 | 0.15163 |
| split4 | 3101 | 17.16740 | 0.59986 | 0.59949 | 0.15240 |
| split4_rotation | 3100 | 16.97426 | 0.59032 | 0.60084 | 0.15642 |
| split4_rotation | 3101 | 17.06853 | 0.59805 | 0.60604 | 0.15343 |

## 两seed等权平均四指标

| 方案 | PSNR↑ | SSIM↑ | LPIPS↓ | photo↓ |
| --- | --- | --- | --- | --- |
| unified32 | 16.59735 | 0.58489 | 0.62183 | 0.16182 |
| unified80 | 16.90737 | 0.59158 | 0.61371 | 0.15717 |
| split4 | 17.21817 | 0.59882 | 0.60082 | 0.15201 |
| split4_rotation | 17.02139 | 0.59419 | 0.60344 | 0.15493 |

## 内部Bank留出收敛

| 方案 | seed | 4000步 | 8000步 | 12000步 |
| --- | --- | --- | --- | --- |
| unified32 | 3100 | 16.194 | 16.451 | 16.646 |
| unified32 | 3101 | 15.729 | 16.388 | 16.837 |
| unified80 | 3100 | 16.516 | 16.798 | 17.064 |
| unified80 | 3101 | 15.798 | 16.704 | 17.137 |
| split4 | 3100 | 16.650 | 17.149 | 17.408 |
| split4 | 3101 | 16.761 | 16.926 | 17.244 |
| split4_rotation | 3100 | 16.795 | 17.111 | 17.320 |
| split4_rotation | 3101 | 16.751 | 16.901 | 17.165 |

统一用最终12000步，未回选中间checkpoint；观察到的差距包括有限训练预算和初始化/优化几何影响，不能当作充分收敛后的表达上限。

## 历史四Head重复性

{"state_dict_exact": false, "maximum_parameter_error": 2.331587076187134, "maximum_metric_error": {"psnr": 2.4670143127441406, "ssim": 0.08700105547904968, "lpips": 0.038129329681396484, "photo": 0.030991852283477783}, "same_k_mean_metric_change": {"psnr": 0.2679990223475865, "ssim": 0.004189104314834353, "lpips": 0.0007313754823472765, "photo": -0.00241027684911849}, "same_seed": 3100, "note": "repeatability is not an independent optimization-order seed"}

重复seed3100仅检查重现性，不算第三个独立seed。原始历史结果保留，没有用重复结果覆盖历史文件。

独立进程核对初始全部权重exact相同，完整3000帧循环与日志ratio/frame/step序列exact相同。第1/2步photo相同，第3步差约1.49e-8，随后选择和训练轨迹分化；这与GPU数值非确定性被长程优化放大一致，但未单独隔离具体CUDA算子。不能把固定seed当成完全确定性，也不能忽略约0.268 dB的同seed平均PSNR波动。

## 对同seed重复波动的敏感性

| 比较 | 本轮重跑作seed3100 | 历史结果作seed3100 | seed3100两次平均后再等权两个seed |
| --- | --- | --- | --- |
| split4 − unified32 | +0.621 | +0.487 | +0.554 |
| split4 − unified80 | +0.311 | +0.177 | +0.244 |
| split4_rotation − split4 | -0.197 | -0.063 | -0.130 |

三种算法均保留seed3101不变，仅展示同seed重复的影响，不是挑选最好seed。四Head相对统一80宽头平均优势在约0.177～0.311 dB之间，加入Rotation的平均变化在约−0.197～−0.063 dB之间，方向均未翻转。该敏感性分析不是置信区间，仍需独立场景与更多重复才能作稳定因果结论。

## Rotation输入是否真的被使用

三个几何Head的新增Rotation输入列权重从零变为非零，最终范数范围 2.462～3.205。q/-q输入等价性通过，四元数norm最大误差约1.19e−7；数值输出依赖诊断保存在analysis_B/rotation_input_dependence.csv。因而不能把本次退化解释为新增输入完全没有训练或输入无效，但输出依赖不等于图像收益。

## 本轮决策

**保留16维ratio latent＋原始11维＋共享h32＋四个两层私有Head作为优先候选，暂不加入基础Quaternion输入。** 增大统一Head容量也有效，但本轮在近似等参数量下四Head的PSNR、SSIM、LPIPS、photo均更好，两个seed方向一致。这支持属性专用隐藏层有价值，尚不能排除优化几何与可变选择集合的共同影响。

Rotation加入后两个seed七档平均PSNR、SSIM与photo均较差，未见ratio方向不一致；本次没有证据支持该4维输入方案。不能推广为所有Rotation表征无价值，也不据此自动改用其他Rotation编码或调参。

生产默认继续保持E50。下一步先在另一独立地图/场景按相同步数与same-K验证四Head，再决定是否进入真实在线Joint；本轮没有自动运行该扩展。

## 决策边界

- split4与wide80参数量近似，但参数化、初始化嵌入与梯度路由不同；对照支持或削弱“属性拆分优于统一容量”的假设，不能当作完全隔离所有优化条件。
- Rotation对照只追加基础四元数输入，几何Head多384参数；不会改Selector输入或输出补偿语义。不能由该对照推断所有Rotation表示都有效或无效。
- 所有方案Selector/Encoder仍共同训练，集合可变；同K固定预算不等于固定点身份。不能将全部PSNR变化归因于Compensation表达。
- 历史non-Bank集合已反复使用，只有fr1_xyz一个冻结地图；本轮不声称独立场景泛化，进入生产前仍需要另一地图/场景和在线Joint确认。
- 生产源码、Loss、默认E50、在线二进制均未修改；无新在线SLAM或CUDA Mapping构建；未stage/commit/push。

## 完整协议与对照表


本报告阶段 B；相同fr1_xyz冻结地图、两个网络初始化与Bank帧序seed3100/3101、统一12000步和每ratio3000曝光。以下PSNR为non-Bank exact same-K。

| 对照 | ratio组 | seed3100 ΔPSNR | seed3101 ΔPSNR | 平均ΔPSNR |
| --- | --- | --- | --- | --- |
| split4 − unified32 | all | +0.820 | +0.422 | +0.621 |
| split4 − unified32 | train | +0.666 | +0.421 | +0.543 |
| split4 − unified32 | unseen | +1.024 | +0.424 | +0.724 |
| unified80 − unified32 | all | +0.393 | +0.227 | +0.310 |
| unified80 − unified32 | train | +0.253 | +0.241 | +0.247 |
| unified80 − unified32 | unseen | +0.581 | +0.208 | +0.395 |
| split4 − unified80 | all | +0.426 | +0.195 | +0.311 |
| split4 − unified80 | train | +0.414 | +0.180 | +0.297 |
| split4 − unified80 | unseen | +0.443 | +0.215 | +0.329 |
| split4_rotation − split4 | all | -0.295 | -0.099 | -0.197 |
| split4_rotation − split4 | train | -0.404 | -0.199 | -0.301 |
| split4_rotation − split4 | unseen | -0.149 | +0.034 | -0.057 |

## 各方案绝对质量

| 方案 | seed | 七档PSNR | SSIM | LPIPS | photo |
| --- | --- | --- | --- | --- | --- |
| unified32 | 3100 | 16.4493 | 0.57862 | 0.62675 | 0.16531 |
| unified32 | 3101 | 16.7454 | 0.59115 | 0.61691 | 0.15833 |
| unified80 | 3100 | 16.8427 | 0.59256 | 0.61117 | 0.15728 |
| unified80 | 3101 | 16.9721 | 0.59061 | 0.61625 | 0.15705 |
| split4 | 3100 | 17.2689 | 0.59778 | 0.60214 | 0.15163 |
| split4 | 3101 | 17.1674 | 0.59986 | 0.59949 | 0.15240 |
| split4_rotation | 3100 | 16.9743 | 0.59032 | 0.60084 | 0.15642 |
| split4_rotation | 3101 | 17.0685 | 0.59805 | 0.60604 | 0.15343 |

![各seed曲线](tests/diagnostics/s00_split_wide_rotation_20261026/evidence/analysis_B/quality_curves.png)

## 结构、初始化与公平性

- 输入为Anchor/ratio latent16、原始属性11、共享Encoder输出h32，共59维。原Selector保持原11→32→32→2，h不detach。
- unified32：59→32→32→14，3438个Head参数；unified80：59→80→80→14，12414参数；split4：四个59→32→32→3/3/4/4，12366参数。宽头与四Head差48参数(约0.39%)。
- 宽头保留原前32隐藏通道函数，其余通道固定seed初始化；输出新增列零初始化，第二层前32行对新增通道权重零。零输出起点一致，但参数化和优化几何仍不同，不声称完全隔离全部容量因素。
- 四Head复制原两层隐藏权重四份、按XYZ/Scale/Rotation/Other切分输出行。其同seed重复训练独立保存，不把重复运行当第三个seed。
- split4_rotation：仅XYZ/Scale/Rotation头追加基础归一化四元数4维，输入63；Other仍59。按最大绝对分量为正统一q/-q符号；新增列初始为零，保持原函数，训练中可收到梯度。Head参数12750，较split4多384。Selector不读取Rotation。
- Rotation只改变补偿输入，不改变输出14D语义：XYZ加法、Scale乘exp、增量四元数归一化与基础旋转复合、Opacity在logit域相加、SH DC加法。
- 相同Master/IDs/Anchor关联中心尺度/相机/eligible/cached uniform冻结，N6838/M2066；54 Bank训练、13内部留出、45历史non-Bank。原photo+.01预算项、Adam fused lr=.001 eps1e-15、帧序与ratio轮转不变。
- Selector与Encoder共同训练，选中集合可变化。same-K严格同预算，原Cached实际点数另存；不是冻结Selection的纯Compensation因果实验。
- 训练ratio1/5/10/15%，未见2.5/7.5/12.5%；先raw-ratio线性插值latent再解码，不为未见ratio拟合参数；最终12000步评价，不回选4k/8k。
- 两个seed同时控制网络初始化与Bank顺序，不是两个独立SLAM地图，不以逐视图数量代替独立重复，不报告不可靠的显著性；历史non-Bank轨迹不是新盲测。
- 生产算法、Loss、默认E50与在线二进制未修改，无新在线SLAM或CUDA Mapping构建；未stage/commit/push。

## 重复性

{"state_dict_exact": false, "maximum_parameter_error": 2.331587076187134, "maximum_metric_error": {"psnr": 2.4670143127441406, "ssim": 0.08700105547904968, "lpips": 0.038129329681396484, "photo": 0.030991852283477783}, "same_k_mean_metric_change": {"psnr": 0.2679990223475865, "ssim": 0.004189104314834353, "lpips": 0.0007313754823472765, "photo": -0.00241027684911849}, "same_seed": 3100, "note": "repeatability is not an independent optimization-order seed"}


## 固定渲染示例

固定seed3100与中间non-Bank视图，未按效果选图。

![1%](tests/diagnostics/s00_split_wide_rotation_20261026/evidence/analysis_B/1pct_examples.png)

![5%](tests/diagnostics/s00_split_wide_rotation_20261026/evidence/analysis_B/5pct_examples.png)
