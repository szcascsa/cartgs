# 集合 Selector：名义格子位置与 slot eligibility 修正

日期：2026-11-02。

## 实际改动

仅修改集合 Selector 的位置表示与成员聚合资格。网络仍为成员`10→16→16`、上下文33D、评分`50→32→32→2`，共3,202个共享参数。Compensator、Anchor补偿Feature、Loss、训练调度、ratio和全图N预算保持原实现。

对于通过Anchor名义格子视锥筛选的成员：

\[
q_i=(x_i-c_{a(i)})/h_{a(i)},\quad
v_i=\neg[\operatorname{slot\_inactive}_i\land(\alpha_i<1/255)].
\]

`c`为现有Anchor名义格子中心，`h=offsetScale`；没有使用成员质心、Gaussian动态包围盒或新的场景归一化。没有开启slot deactivation时所有成员eligible。

仅对eligible成员使用物理输入：

\[
u_i=[q_i^3,s_i^3,SHDC_i^3,\alpha_i],\quad
h_i=\begin{cases}\phi(u_i)&v_i=1\\0&v_i=0\end{cases}.
\]

实现中在MLP输入处先将不可选成员置零，在MLP输出处再置零，阻断不可选成员异常属性影响共享参数梯度的通路。没有向Master新增参数，没有detach有效成员的物理输入。

\[
m_a=\sum_{i:a(i)=a}v_i,\quad
g_a=[\operatorname{Mean}_{v_i=1}h_i,\operatorname{Max}_{v_i=1}h_i,m_a/10].
\]

Mean按有效成员数归一化；Max使用排除值，不让不可选成员成为最大值来源；Count也只统计eligible成员。`m_a=0`时33D上下文为零，Encoder输出仍包含原`log(r)`。评分MLP可能产生有限logits，最终原slot mask保证这些成员不能选中。

每次forward使用当前拓扑、中心、offsetScale、slot状态与当前Opacity重算。compact输入通过全局`source_rows`查询slot资格，不能用局部数组位置索引全局slot表。资格查询不改变`slot_inactive`或`slot_recovered`；停用与恢复仍由既有生命周期流程处理。

## 梯度与兼容性

- 单成员Anchor仍有名义格子相对位置，XYZ输入梯度不再因`x−Mean(x)=0`而恒为零。
- 有效成员保留Photo/Budget到Master的Joint梯度及同Anchor上下文耦合；不可选成员的Encoder属性输入梯度为零。
- Maturity仍只对Selector私有参数求VJP，不额外更新Master、Compensator或Anchor补偿Feature。
- 新创建集合模型记录`selector_set_encoder_schema=2`。保存/加载和Frozen克隆保留schema。
- 旧schema 1模型仍使用质心相对位置和全部成员池化，不静默转换。原11D模型保持原路径。不同schema不能在一个已配置模块中混用。
- 当前集合配置开关仍是`Selector.anchor_set_encoder: 1`；本轮没有修改生产默认开关。

## 代码位置

| 文件 | 修改 |
|---|---|
| `improvements/selector/anchor_set_encoder.h` | schema 2、名义格子位置、eligible Mean/Max/Count、全无效处理 |
| `improvements/selector/anchor_bank.h` | 原纯查询slot mask支持compact全局行索引 |
| `improvements/selector/gumbel_network.h` | 训练与其他forward入口传入中心/资格，版本加载与Frozen克隆 |
| `python/anchor_set_encoder.py` | 对应C++数学定义 |
| `python/selector.py` | 纯查询slot资格、compact行映射、schema 1/2加载 |
| `src/gaussian_mapper.cpp` | 导出实际schema及上下文语义 |
| `tests/anchor_selector/anchor_set_encoder_test.cpp` | 原生前后向、失效成员隔离、空集合、旧版本回归 |
| `research/anchor_set_selector_20261102/test_python.py` | Python与C++对照、生命周期、gradcheck、旧模型兼容 |

## 验证状态

独立目录：`/home/prmi/zwj/experiments/anchor_set_selector_nominal_eligible_20261102/`。旧schema 1证据目录保持原样。训练队列不恢复。

首批6项Python测试通过：CPU/CUDA单成员位置、slot资格纯查询与compact映射、不可选成员污染隔离、全无效集合及梯度、成员变化、FP64 gradcheck、新旧checkpoint和历史schema 1前向。

完整验证已完成：

| 检查 | 实测结果 |
|---|---|
| 独立构建 | CUDA 11.8 / RTX3090，核心库、测试、`tum_rgbd`和`tum_mono`编译通过 |
| 原生CPU/CUDA集合测试 | 名义格子位置、单成员非零XYZ VJP、inactive污染隔离、全无效集合有限零VJP、变长成员、排列/重排、空集合通过 |
| C++/Python前向和VJP | 86项数值对照通过，最大绝对差`1.1920928955078125e-7` |
| Python测试 | 8个测试通过，覆盖FP64 gradcheck、slot资格纯查询、compact行映射、成员变化、新旧schema加载 |
| 实际LoD Mapping函数 | 合成地图`N=16`、候选`C=4`；六档全图预算、Maturity私有VJP、Photo/Budget、Joint Adam、空候选及全inactive通过 |
| 原路径回归 | 原11D采样CPU/CUDA、原视锥Mapping及Python推理7项测试通过 |
| Residual48原路径 | 97项输入打包Forward/VJP逐位检查通过 |
| 保存加载与Frozen克隆 | schema 2及历史schema 1均通过；历史Encoder前向对照通过 |
| 生产保护 | 334个受保护源码、配置、二进制和动态库SHA256一致；两批队列仍暂停 |
| 下载核验 | 50份证据文件及9份patch源文件哈希校验通过，本地代码与已验证服务器源码一致 |

CUDA Pooling相同输入重复、独立Anchor对照、Anchor重排最大差均为`1.19209e-7`，CPU对应检查为0。集合聚合保留FP32原子求和，因此不宣称CUDA数学排列不变性等于逐位一致性；这是集合模式已有的性质。原Residual48逐位回归单独检查，不将集合误差放宽传播给原路径。

验证中最后一项推理回归初次未执行，原因是独立包漏带未修改的`test_anchor_frustum.py`。保留失败日志到`failures/missing_inference_test/`，补齐原文件并只重跑该阶段后通过，没有为此修改模型或推理算法。正式构建和此前成功测试没有重跑，最终11个阶段均成功。

证据目录：`research/anchor_set_selector_nominal_eligible_20261102/evidence/`；本地完整性核验：`LOCAL_VERIFICATION.json`；原始证据包：`VALIDATION_EVIDENCE.zip`，SHA256为`ecbaaf07069c5f637bfbfe1e7dd806b24d9bc793e9acac2d11b32d64300435d2`。

本次没有完整场景训练或PSNR实验，不能据此声称渲染质量提升。未替换服务器主项目或生产运行库，未恢复暂停队列，未commit/push。
