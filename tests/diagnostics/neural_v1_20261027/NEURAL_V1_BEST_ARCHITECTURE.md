# 神经V1推荐研究架构：确定性物理编码＋共享连续神经场

目前没有证据支持一个网络在两张地图、所有ratio和训练预算下唯一最优。建议V2优先深入`SpatialFiLM`连续共享神经场，并保留`SpatialLocalXYZ`几何单位对照；`SpatialOutputKnots`是本轮较早锁定的短程主候选，完整报告保留其长程与另一集合的反例。两个函数家族在读取本轮non-Bank前已锁定，后续没有根据test新增推荐架构。以下给出共同输入与两种ratio组织的完整定义。

输出插值的一个具体风险已经定位：Gaussian未在低端点参与训练时，其低端点网络输出仍会混入未见ratio。只读Bank条件替换显示这种外推可造成明显损害。因而输出插值不作为唯一推荐；支持感知的神经条件化尚未实现，属于V2假设。

## 1. 输入与合法性

只读取显式Gaussian原始属性及Anchor几何参考：

\[
u_i=[\hat x_i,\operatorname{asinh}(o_i),\log\sigma_i,SH_{DC,i},\operatorname{logit}\alpha_i,
\operatorname{vec}_6(\Sigma_i/\operatorname{tr}\Sigma_i)]\in\mathbb R^{19}.
\]

这里世界XYZ为3维，Offset3，log-scale3，DC3，opacity-logit1，协方差对称6维。协方差来自真实Scale与Rotation，无Quaternion符号歧义；没有额外维护可漂移几何。世界坐标用冻结地图的median中心与逐轴RMS尺度归一化，19维再按冻结人口均值/标准差标准化并截断±15。此归一化属于实验预处理，本轮没有改生产。

对归一化世界XYZ做固定六频带编码：

\[
\gamma(x)=[\sin(2^k\pi\hat x),\cos(2^k\pi\hat x)]_{k=0}^{5}\in\mathbb R^{36},
\quad X_i=[u_i,\gamma(x_i)]\in\mathbb R^{55}.
\]

所有频率确定、不可训练。场景统计只读；不存在Gaussian/Anchor ID embedding、私有latent、ratio补偿表、空间网格或个体残差。原始属性和这些确定性变换可缓存，缓存不参与optimizer。

## 2. 网络与数学形式

共享主干55→64→ReLU→64→ReLU，四个属性分支各为64→64→ReLU→输出。

连续`SpatialFiLM`以归一化log-ratio为输入，经共享1→32→ReLU→256调制网络产生两层gain/bias：

\[
h_1=\operatorname{ReLU}((1+g_1(r))\odot(W_1X+b_1)+\beta_1(r)),
\quad h_2=\operatorname{ReLU}((1+g_2(r))\odot(W_2h_1+b_2)+\beta_2(r)).
\]

四个Head分别输出XYZ3、log-scale3、Quaternion残差4、Opacity/DC4。总计33806个共享参数，不需要节点插值。`SpatialLocalXYZ`只将位移转换为`Anchor_reference_scale_i × Head_xyz(h2)`，其余相同、参数量相同。该参考尺度是几何数据，不能被更新为Gaussian私有参数。

输出插值`SpatialOutputKnots`的每个属性分支一次预测四个训练ratio节点的函数值：XYZ12、log-scale12、Quaternion残差16、Opacity/DC16，共56个输出，总36536个共享参数。它们是共享网络对当前Gaussian输入的预测值，**没有保存为按Gaussian索引的可学习参数**。

\[
D_{i,k}=C_{\theta,k}(X_i),\qquad
\Delta_i(r)=(1-t)D_{i,k}+tD_{i,k+1},
\quad t=\frac{r-r_k}{r_{k+1}-r_k}.
\]

训练节点为1%、5%、10%、15%。所有节点的函数权重跨Gaussian共享，任意合法输入都用同一组网络计算。未见ratio直接插值网络输出，不能重新拟合特征或创建新参数。

当前实现保留了从FiLM原型继承的两组通道仿射调制，但对这个输出插值分支，调制网络输入固定为0；**它不是ratio FiLM**。其常数gain/bias可以折叠进主干Linear，`SpatialKnotsFolded`为同初始函数的优化参数化控制。ratio的作用来自上述连续输出插值。不能把冗余常数调制的优化效果宣传成新的ratio机制。

连续FiLM对照使用相同55维输入，但共享网络以归一化log-ratio生成两层gain/bias，直接输出14维；它不依赖输出节点插值。

## 3. 输出与Renderer

最终14维顺序沿用：XYZ3 / log-scale3 / Quaternion残差4 / Opacity-logit1 / SH DC3。

- XYZ加到显式位置。
- Scale乘exp增量。
- Rotation使用normalize(identity+delta_q)，左乘原Quaternion再normalize。
- Opacity在logit空间相加后sigmoid。
- SH DC相加，高阶SH不变。

初始输出层全零，step0与无补偿Gaussian相同。所有补偿都由共享神经函数生成。本轮没有使用R3或Factored作为候选Forward的基值，也没有给新网络叠加冻结的个体补偿。

## 4. 训练

主要证据使用固定R3或Baseline59终点集合：Master、相机、Gaussian身份、Anchor归属、Selector和K全部冻结；新网络从零训练。LoD photo仍为`.8L1+.2(1-fusedSSIM)`，Adam .001 / eps1e-15，每ratio3000次真实曝光，总12000步。

由于固定集合，本轮没有优化Selector，也没有预算Loss梯度。等曝光Reference59及可训练补偿Encoder的Reference59控制单独保留；它们含历史Anchor/ratio表，只是对照，不能成为本轮推荐。

长程对照从12000恢复完整Adam并接续原Bank帧序，再24000步，总36000；全部臂匹配预算。36k与12k R3比较只表示更多训练后的潜力，不能声称相同步数超越。

输出节点在同一个Linear参数矩阵中，未激活节点的梯度为零但Adam动量仍会演化。`SpatialKnotsSparseClock`将末层拆成四组共享函数权重，仅激活对应节点，验证这一优化差异；它依旧不是任何个体参数表。

## 5. 可训练参数清单

准确参数名、形状和数量保存在每个正式候选`formal/trainable_parameters.json`，并汇总到`NEURAL_V1_TRAINABLE_PARAMETERS.csv`。全部参数只属于`nn.Linear.weight/bias`；没有`nn.Embedding`或按N/M分配的Parameter。参数总数与Gaussian/Anchor数量无关，fr1/fr2使用相同结构参数数目。

原始场景缓存为只读数据，不纳入trainable参数。不能将只读缓存当私有可学习特征，也不能忽略其真实显存成本。参数审查与真实photo非零梯度gate见`neural_correctness.json`、`parameter_audit.json`；固定输入身份的顺序置换gate见机制脚本。

## 6. 复杂度与部署边界

两个神经场仅对选中K行执行共享补偿网络，网络成本约O(K·64²)，与N无关；确定性属性若缓存需O(19N)数据。没有全人口可学习补偿表或对应Adam状态。

局部Attention/全地图消息分支是本轮其他探索臂，**不是主候选必需计算**。实验Fixture为方便比较预建kNN和多个集合缓存，不能把其启动成本算成主候选必需Forward，也不能在完整显存统计中隐去这些缓存。

在线部署需要在Master或拓扑变化后更新输入属性与归一化统计策略；这会改变神经场输入分布。保持固定坐标参考、避免随着AABB移动归一化是V2需要验证的工程问题。当前无C++/CUDA、在线Joint、clone/split/prune或闭环接入，本轮不自动集成。

本方法的潜力来自更可区分的确定性输入和ratio函数组织，不是已证明的新理论或普适最优架构。剩余问题应由配对质量与收敛证据决定，不能仅凭网络参数少就宣称接近R3。
