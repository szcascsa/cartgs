# Split尺度采样修复与独立质量对照

实验启动：2026-10-21。项目：mycartgs。修复已写入本地生产源码；未stage、commit或push。

## 1. 结论

**尺度采样错误已修复，并通过CPU／CUDA原生验证。** legacy与可变成员两个split入口均改为`Δx=Lz`，`L=R diag(scale)`；原`Δx=Σz`会让扰动协方差从应有的Σ变为Σ²。协方差API值保持不变。

完成fr1_xyz两seed、原版／修复版共4次独立训练。默认停止规则未改，4组实际均为7400 Mapping；统一评价64个RGB视图，其中46个为四组共同non-Bank视图。

共同non-Bank的same-K平均变化：**PSNR +0.1646 dB，photo -0.000107**。8个seed×K比较中，5/8同时改善PSNR与photo。Full平均ΔPSNR为+0.3937 dB。Δ均为修复版−原版。

**质量结论：Full的PSNR两个seed均改善；LoD为混合结果。** 四个same-K预算的平均LPIPS均变差，SSIM在其中三个预算变差；seed2026的中高预算PSNR下降。保留全部seed和预算，未选择最好checkpoint或最好seed。

## 2. 实际改动

| 文件 | 改动 |
|---|---|
| [gaussian_model.cpp](H:/AIProjects/cartgs优化更细致版/mycartgs/src/gaussian_model.cpp) | 提取原先已有的尺度旋转因子L；getCovarianceActivation仍返回L Lᵀ；两个split入口使用L乘原随机噪声 |
| [gaussian_model.h](H:/AIProjects/cartgs优化更细致版/mycartgs/include/gaussian_model.h) | 声明getScalingRotationActivation |
| CMakeLists.txt／gaussian_split_sampling_test.cpp | 添加原生数学、实际split位置和生命周期回归测试 |

生产算法只改变split扰动矩阵。随机噪声的调用顺序、形状和`|z|>3→0`规则、clone、split后scale除以1.6、父点替换、Anchor归属、Feature、Adam、pruning／growing、TF／ChangeDistill、Selector网络、Loss、ratio轮转、Cached Bernoulli均不变。没有引入Offset clamp或重关联规则。

这会影响之后创建的split子点。已有checkpoint中历史长尾Gaussian不会被自动移动；本轮质量结果来自重新训练的模型。

## 3. 正确性验证

- CPU／CUDA均通过：新采样分布、实际legacy split位置、实际variable split位置、Anchor容量／归属、继承属性与Feature／Gaussian Adam状态检查。
- 另外通过既有variable_anchor_test和anchor_topology_policy_test的CPU／CUDA回归；共6项修复版测试执行。
- 同一测试对原版的两个实际split入口，在CPU和CUDA上均按预期失败（共4次），错误明确指向使用协方差而非尺度因子。原版新增因子API自身的数学测试仍通过。
- 每个尺度／旋转场景使用200000个样本；实测协方差相对误差最大0.4091%，门限2.5%。检验覆盖scale<1、scale>1及非均匀尺度／旋转，旧Σ采样与正确目标的偏差由负向断言识别。
- getCovarianceActivation与原矩阵运算输出逐元素exact相同；空输入和scaling_modifier的长度线性关系通过。

保留3σ置零后，噪声每轴二阶矩κ≈0.970709；修复后的扰动协方差为κΣ，原版为κΣ²。本轮没有同时改变噪声截断，因此不宣称采样为无截断的标准正态。

首次测试夹具把当前多层初始化的两个Anchor当作一个，造成期望噪声批量大小不匹配。已修正夹具，生产实现未因该失败增加改动；初始失败和复测日志均保留。

## 4. 单变量协议与训练计数

场景fr1_xyz；seed2026／2027；默认16维Feature、27维输入、14维选后补偿。两组使用同一配置，`max_mapping_iterations=0`，不强制10000步。训练顺序baseline2026→fixed2026→fixed2027→baseline2027。TF／SI／GI／ChangeDistill关闭，joint、LR和所有阈值不变。最终shutdown checkpoint统一评价。

两套独立库保留于服务器`/home/prmi/zwj/split-scale-fix-20261021/variants/{baseline,fixed}`；通过LD_LIBRARY_PATH和ldd核对实际加载版本。两臂源码的采样差异严格为两个调用点，原协方差接口的兼容性测试通过。服务器原项目、旧实验和模型未覆盖。

本轮重新训练的基线为默认停止规则下的7400步模型。此前Offset=210.62、1%位移/h p90=3.37来自另一次10000步checkpoint，不用于计算本轮修复增益；本轮没有将两种训练预算混作配对。

| 方法 | seed | Mapping | Gaussian | Anchor | L0/L1/L2 | Full render | LoD render | Gaussian更新 | 秒 | GPU峰值MiB |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|
| baseline | 2026 | 7400 | 14550 | 5298 | 1476/3690/132 | 1996 | 6050 | 8046 | 45.72 | 941 |
| fixed | 2026 | 7400 | 15331 | 5550 | 1537/3848/165 | 1915 | 6029 | 7944 | 44.77 | 961 |
| baseline | 2027 | 7400 | 14661 | 5326 | 1480/3764/82 | 1948 | 6037 | 7985 | 44.82 | 945 |
| fixed | 2027 | 7400 | 14796 | 5480 | 1468/3793/219 | 2040 | 6061 | 8101 | 45.67 | 947 |

Gaussian更新次数包括LoD和Full两条路径，所以不应直接等同于Mapping计数。尽管Mapping终点一致，在线SLAM关键帧／位姿、地图数量以及各路径实际更新次数仍可能不同；按原始计数比较。额外出生快照在两臂同时开启，会增加同步；训练耗时与显存是完整实验实测，不是等N性能微测。

## 5. 新生点与最终长尾

以下出生统计是每次split即时保存的子点；新跨越次数不代表最终仍存活的独立Gaussian数量。矩阵、相同噪声、父子XYZ和归属h完整保存。

| 方法 | seed | split事件 | 出生子点 | 最大新生Offset分量 | 从≤10跨到>10 | 从≤100跨到>100 | 最大单坐标扰动 |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 2026 | 7 | 7754 | 51.868 | 17 | 0 | 4.178 |
| fixed | 2026 | 7 | 8628 | 40.195 | 21 | 0 | 4.529 |
| baseline | 2027 | 7 | 7828 | 368.064 | 21 | 8 | 19.812 |
| fixed | 2027 | 7 | 8514 | 68.369 | 31 | 0 | 4.988 |

| 方法 | seed | 最终Offset最大分量 | Offset>10数量 | Offset>100数量 | 1%选中尾点>10 | 1%选中Offset p90 | 1%位移/h p90 |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline | 2026 | 51.881 | 22 | 0 | 6 | 6.112 | 1.798 |
| fixed | 2026 | 40.197 | 22 | 0 | 0 | 4.032 | 1.481 |
| baseline | 2027 | 368.064 | 28 | 8 | 16 | 11.209 | 1.760 |
| fixed | 2027 | 68.326 | 35 | 0 | 18 | 10.504 | 1.193 |

总体Offset分量的p99从原版两seed的1.522／1.301升至3.960／4.466；Offset>10的数量分别22→22、28→35。超大极值下降，但一般尾部没有整体收窄，这与小于1的scale恢复为线性扰动后覆盖范围变大相符。seed2027的1%最大scale multiplier由4508.94降至3.91，显示极端输入与输出放大得到缓解。

| ratio | seed2026 位移/h p90 原版→修复 | seed2027 位移/h p90 原版→修复 |
|---|---:|---:|
| 1% | 1.798 → 1.481 | 1.760 → 1.193 |
| 5% | 0.454 → 0.857 | 0.609 → 0.451 |
| 10% | 0.203 → 0.753 | 0.347 → 0.212 |
| 15% | 0.137 → 0.733 | 0.253 → 0.113 |

修复不保证Offset被限制在cell内。Gaussian的真实尺度可能大于Anchor边长，普通XYZ优化及子点继承归属也会继续造成跨cell位置；对scale<1的父点，正确扰动反而比错误的scale²扰动更大。尾点数量、极值、补偿幅度与质量应分别看，不能将任何单项作为成功判据。

## 6. 正式Cached Bernoulli与Full质量

以下为共同non-Bank视图，按原ratio输入、原Cached Bernoulli推理。LoD点数允许原机制产生偏差，逐组列出；Full不使用Selector补偿。

| seed | 档位 | 原版/修复版选中数 | ΔPSNR | ΔSSIM | ΔLPIPS | Δphoto |
|---|---|---|---:|---:|---:|---:|
| 2026 | Full | 14550/15331 | +0.4202 | +0.00521 | -0.01547 | -0.004546 |
| 2026 | 1 | 146/152 | +0.6995 | +0.00017 | -0.00704 | -0.005889 |
| 2026 | 5 | 729/773 | -0.0052 | -0.01764 | +0.03132 | +0.003623 |
| 2026 | 10 | 1424/1476 | -0.0836 | -0.01183 | +0.03243 | +0.001949 |
| 2026 | 15 | 2120/2230 | -0.1596 | -0.01233 | +0.02985 | +0.002326 |
| 2027 | Full | 14661/14796 | +0.3672 | +0.00563 | -0.01686 | -0.001993 |
| 2027 | 1 | 145/143 | +0.3031 | -0.00011 | +0.02187 | -0.000018 |
| 2027 | 5 | 729/723 | -0.0177 | -0.00116 | +0.01660 | +0.001230 |
| 2027 | 10 | 1456/1423 | +0.2866 | +0.00442 | +0.01008 | -0.001598 |
| 2027 | 15 | 2181/2143 | +0.4061 | +0.00737 | +0.00297 | -0.003009 |

## 7. 相同绝对K质量

共同K=[154, 830, 1506, 2182]。输入ratio为各自K/N，均在[0.01,0.15]内；按Cached Bernoulli的logit(p)−logit(cached_uniform)排序，源行号打破ties，取严格K个。这是预算诊断，不替换生产Cached Bernoulli。

| seed | K | ΔPSNR | ΔSSIM | ΔLPIPS | Δphoto | PSNR改善视图比例 |
|---|---:|---:|---:|---:|---:|---:|
| 2026 | 154 | +0.6964 | -0.00099 | -0.00713 | -0.005472 | 87.0% |
| 2026 | 830 | -0.0720 | -0.01992 | +0.03901 | +0.004955 | 56.5% |
| 2026 | 1506 | -0.1829 | -0.01678 | +0.03940 | +0.003729 | 45.7% |
| 2026 | 2182 | -0.1771 | -0.01224 | +0.03280 | +0.002290 | 52.2% |
| 2027 | 154 | +0.2553 | +0.00112 | +0.01822 | -0.001030 | 50.0% |
| 2027 | 830 | +0.1036 | +0.00178 | +0.01320 | -0.000247 | 54.3% |
| 2027 | 1506 | +0.2905 | +0.00554 | +0.00680 | -0.001972 | 65.2% |
| 2027 | 2182 | +0.4031 | +0.00762 | +0.00285 | -0.003105 | 69.6% |

逐视图mean、median、p10、p90、improved fraction全部保存在paired_statistics.csv；全64视图的结果也完整保存。没有事后挑选最好seed。same-K消除选中点数差异，未冻结两组各自的Gaussian属性、训练轨迹和位姿。

| 共同K | 两seed平均ΔPSNR | 平均ΔSSIM | 平均ΔLPIPS | 平均Δphoto |
|---|---:|---:|---:|---:|
| K154 | +0.4758 | +0.00006 | +0.00554 | -0.003251 |
| K830 | +0.0158 | -0.00907 | +0.02610 | +0.002354 |
| K1506 | +0.0538 | -0.00562 | +0.02310 | +0.000879 |
| K2182 | +0.1130 | -0.00231 | +0.01783 | -0.000408 |

统计汇总脚本曾将已有字段gaussian_updates误写成gaussian_steps；已修正后只重跑汇总，四次训练和36项评价没有重复，初始日志完整保留。

## 8. 验证状态与范围

| 项目 | 状态 |
|---|---|
| 两个生产split入口的尺度采样修复 | IMPLEMENTED／TESTED，服务器CPU与CUDA |
| 协方差API、原噪声截断、拓扑与归属／Adam回归 | TESTED |
| 四组重新训练、Full／1／5／10／15%与same-K四指标 | TESTED |
| 64共享视图、共同non-Bank、同K计数、模型SHA | TESTED |
| 跨场景稳定性、等N端到端性能 | NOT TESTED |
| 自动清理历史Gaussian、Offset clamp、重新关联、MLP或loss改造 | NOT IMPLEMENTED，未扩大本轮范围 |

本轮只修复采样尺度并完成独立对照。默认Feature16、max_mapping_iterations=0保持；未stage／commit／push，没有自动调整增长阈值或补偿网络。

## 9. 证据入口

- [原生正确性](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/split_scale_validation/evidence/validation/correctness.json)；[预注册协议](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/split_scale_validation/evidence/validation/preregistration.json)；[训练计数](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/split_scale_validation/evidence/validation/training_runtime.csv)
- [出生统计](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/split_scale_validation/evidence/validation/split_birth_statistics.csv)；[最终模型／补偿统计](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/split_scale_validation/evidence/validation/final_model_statistics.csv)
- [质量汇总](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/split_scale_validation/evidence/validation/quality_summary.csv)；[逐视图](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/split_scale_validation/evidence/validation/quality_per_view.csv)；[配对统计](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/split_scale_validation/evidence/validation/paired_statistics.csv)
- [模型及共同视图SHA](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/split_scale_validation/evidence/validation/evaluation_plan.json)；[本地交付检查](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/split_scale_validation/local_delivery_checks.json)

![出生点与补偿](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/split_scale_validation/evidence/validation/split_and_compensation.png)

![same-K质量](H:/AIProjects/cartgs优化更细致版/mycartgs/tests/anchor_selector/split_scale_validation/evidence/validation/same_k_quality.png)
