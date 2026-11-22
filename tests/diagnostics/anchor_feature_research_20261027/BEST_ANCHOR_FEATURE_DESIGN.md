# Anchor Feature 设计与在线接入边界

本文件说明已经实现和实际训练的设计。最终研究候选由 `BANK_LOCKED_CANDIDATE.json` 在读取本轮 non-Bank 指标前锁定；质量和是否值得进入下一阶段，以主报告为准。当前生产默认仍为原 SpatialLocalXYZ。测试前锁定的PSNR候选为FrozenBaseHead8；它的LPIPS没有同步超过等步数基线，不能称为多指标最优或直接部署方案。

## 1. 基础函数

保留当前生产的55D输入、两层64通道、log-ratio FiLM、四Head和14D属性解码。记其第二层共享编码为h_i(r)。不修改Selector11、Gaussian/Anchor结构、Loss或固定选点集合。

本轮Feature是补偿专用的抽象表示，与历史Selector Anchor Feature、历史Anchor/ratio补偿表、R3 Gaussian/ratio直接残差均不同。

当前AnchorBank通用`features_`只接受0/16/32维，旧`compensation_features_`带多个ratio节点。生产化时应显式新增单张8D补偿表并纳入featureTables/序列化/UID重排，而不是把Selector表改成8D，或伪造两个相同ratio表。此结构扩展尚未实施。

## 2. Learned Head8

每个稳定UID对应Anchor保存一行 e_a∈R⁸；全部ratio共用同一行。每个属性分支t∈{XYZ,Scale,Rotation,Other}使用共享投影 U_t∈R^(64×8)：

\[
a_{it}(r)=W_{t1}h_i(r)+b_{t1}+U_te_{a(i)},\qquad
y_{it}(r)=W_{t2}\operatorname{ReLU}(a_{it}(r))+b_{t2}.
\]

XYZ输出为s_a·y_XYZ，其余沿用原网络的log-scale、Quaternion residual、opacity logit和SH DC。e的坐标没有被解释成任何物理属性；8维同一表示通过四个共享非线性Head生成14D输出。同Anchor不同Gaussian有不同55D输入和激活，输出不相同。

未见ratio直接代入原log-ratio FiLM；没有ratio索引、插值表或新建Feature。Feature投影本身不直接读取r，但它与随r变化的h在ReLU前相加，输出影响仍随ratio变化。

### 初始化

e=0；U_t使用固定seed、小幅Normal(0,0.02²)；原网络权重直接复制，因此continuation step0与原网络输出完全相同。U非零且原非零Head权重保留，使e第一次真实photo backward即可得到有效梯度。第一步U的梯度为0属于e=0的预期结果，之后e非零才更新U，不存在零门控与零权重双重阻断。

scratch的seed同时决定base初始化和帧序；同一seed两臂的base初始化相同，分支在fork_rng内初始化。scratch原网络末层为零，第一次上游梯度为零是原始零输出初始化的正常现象，不能将continuation的首步非零要求套用到所有scratch上游层。

### Backward

令g_it为Head第一层激活前的Loss梯度，则：

\[
\frac{\partial L}{\partial e_a}=\sum_{i:a(i)=a}\sum_t U_t^Tg_{it},\qquad
\frac{\partial L}{\partial U_t}=\sum_i g_{it}e_{a(i)}^T.
\]

g包含原14D解码、Renderer、photo loss及XYZ参考尺度的链式导数。同Anchor成员梯度Scatter/Add到同一行；这里只读取选中的K行，未选中的Gaussian不计算补偿。不存在Gaussian私有可训练补偿。

## 3. 其他已测融合

| 结构 | 融合 | 额外共享投影参数 |
| --- | --- | ---: |
| Early8 | 第一层Linear后、原FiLM前加Ue | 512 |
| Mid8 | 第一层FiLM后加Ue | 512 |
| AnchorFiLM8 | 第二层以U[e,phi(log r)e]产生gain/bias | 2048 |
| Head8 | 四个Head的第一层激活前分别加U_t e | 2048 |

因此Head8与AnchorFiLM8的融合位置比较具有相同投影参数量。Early/Mid不同参数量，仅作机制筛选，不把数值差异纯归因于位置。

## 4. Context与Hybrid

每Anchor对完整成员的标准化physical19计算mean、max、std，再加log(1+occupancy)，构成58D确定性上下文c_a。共享58→32→ReLU→8网络生成：

\[
e_a^{context}=G_\phi(c_a)-G_{\phi_0}(c_a).
\]

φ₀是初始化网络的冻结副本。两分支初始化相同，step0差为0；φ仍能从第一次非零photo获得梯度。G共有2152个可训练参数，另保留2152个冻结参数。它在固定地图中可以预存c；在线属性变化时必须重算对应Anchor，冻结G₀的输出也不能沿用过期上下文。

Context包含邻居成员的局部统计，这些信息不在单个Gaussian的55D输入中；它没有显式编码某ratio究竟删除哪些成员。本轮未将其表现推广成所有Context编码器的结论。Hybrid连接learned8/context8，只在短程筛选中使用，完整报告保留其负结果。

## 5. 训练路线和Adam

Continuation：从原SpatialLocalXYZ 12k模型及完整base Adam恢复，再做12k真实LoD步；新U/e各自Adam状态从0开始。lr=.001，eps=1e-15，fused Adam，与基线相同。Feature为独立参数组，但本轮未做学习率搜索。

FrozenBase路线：base所有权重固定，仅训练新U/e或Context/U；每ratio仍3000次真实photo曝光。它研究优化耦合，不能声称与全网络联合训练的参数更新量相同。Global/Fake/Context控制用于识别收益来源。

**FrozenBase依赖已经训练为非零的base输出层。** 生产从零初始化的四Head末层为0；若此时冻结全部base，U/e的photo梯度会被这个0末层永久阻断。不能把本轮冻结历史12k网络的结果当作“在线从第一步就冻结”的可用方案。在线若采用该机制，需要另行验证非零warmup或阶段切换，当前没有完成。

本轮Dense Adam对整张M×8表更新，tensor级step每次增加；未访问行梯度为0，曾访问行可能因动量继续变化。没有将“实际非零梯度次数”“表的Adam时钟”“行值改变次数”混为一谈。Sparse Adam未实测，不能作为不改变优化语义的等价替换。

## 6. 参数与空间

| 项目 | fr1 M=2066 | fr2 M=2724 |
| --- | ---: | ---: |
| Anchor Feature参数8M | 16528 | 21792 |
| 四Head投影 | 2048 | 2048 |
| 原base共享参数 | 33806 | 33806 |
| 总参数 | 52382 | 57646 |
| 表权重+梯度+Adam两moment，128M bytes | 264448 bytes | 348672 bytes |

参数随M增长，不随ratio档数增长。一个单成员Anchor的latent客观上可产生Gaussian身份记忆作用；本轮允许Anchor表，但必须报告occupancy和对应控制，不能把这部分称为多成员共享证据。

## 7. 生命周期设计与已验证边界

- 创建：用稳定block UID注册新行，默认e=0、moment=0；保留共享U。此处保证原网络函数起点，不保证新观测立即有训练收益。
- clone/split：归属已有Anchor则读取同一e；新Anchor使用零e。当前固定Master实验没有实际执行densification。
- Growing：新增Anchor拥有独立新行，不迁移既有Gaussian所属Feature。
- prune/reorder：用存活UID建立行映射，替换Parameter leaf与全部optimizer alias；同样迁移exp_avg/exp_avg_sq，新行补零。不要只resize `.data`。
- 时钟：沿用当前AnchorBank的tensor级Adam step；新行零moment不意味着独立年龄。不能用伪造零梯度step调整时钟。
- 几何修正：整Anchor移动保留UID/e，physical55按当前属性/中心/尺度重算；Context重新聚合。
- 局部闭环迁移：可复用当前成员数量加权latent合并与接收行moment清零规则，但非线性Head使其不保证输出连续。此为在线待验证项，未在本轮实现生产迁移。
- 空间碰撞：空间(level,key)可以有多个block；不能把key当作UID而合并错Feature。
- Checkpoint：保存schema、UID顺序、Feature表、U、原base、归一化buffer、完整Adam和step；读取时校验UID与行映射。本轮原型checkpoint已保存UID/owner/Adam，未改生产schema。

`af_gates.py::remap_embedding`实际验证了稳定UID的重排/删除/新增、旧值与moment保持、新行零状态及optimizer参数alias。真实SLAM事件、Feature混合后的质量和CUDA并发尚未完成。

## 8. CUDA集成方案

保留当前生产live55D和FiLM Kernel。补偿只读K个Gaussian的owner；新增Gather输出K×8；四个U_t e GEMM合并为8→256共享Linear后切片，数学上相同，需数值/梯度gate后才能替换原型的四次独立GEMM。将结果加到四Head第一层ReLU前，Backward对U求和、Feature按owner Scatter/Add。

在冻结地图或安全事件边界可缓存owner→Feature Gather索引；不能缓存会过期的Feature值或Master属性。主路径不需要N人口补偿Forward；Dense Adam仍有O(8M)开销。

纯推理且e/U都固定时，可以预计算每Anchor的256D `concat(U_t e)`，用额外约1024M bytes缓存换去四次私有投影GEMM；这是网络计算的只读缓存，不是独立训练的物理补偿表。e/U任一更新、Anchor增删/重排或checkpoint切换都必须使缓存失效。在线每步更新时不能照搬该推理缓存，也不能用过期值换速度。本轮只提供设计，未测这项性能。

若未来采用冻结base权重的在线策略，只能 `requires_grad(false)` 冻结权重；不能用整个base的`no_grad`切断Master输入梯度，否则会改变当前Joint语义。本轮冻结输入性能不包含这部分在线梯度成本。

Context在线需要按Anchor属性/成员版本维护58D聚合及G₀输出；安全实现应先正确重算，再研究增量统计。max和variance的删除更新不能用简单累加替代。所有新增CUDA只是设计，本轮没有修改生产Kernel或默认配置。
