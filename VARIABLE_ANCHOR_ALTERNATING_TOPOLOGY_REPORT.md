# 可变成员 Anchor 与交替增删实现

## 交付范围

细化前的相关代码已提交到本地 `redundancy-aware-si`：

`aad134d035fc9d0f0b8d16dbaa0936bfa3c52654` — `feat(anchor): preserve opacity pruning and validated LoD training variants`。

本轮新增实现没有 stage、commit 或 push。

## 实际规则

- 一个存活 Anchor 持有 **1～10 个显式 Gaussian**。成员数为零时删除空 Anchor。
- 观测初始化按原有 L0/L1 多尺度 voxel 组织；世界原点、`h0=0.1`、最大 L2 保持原设置。每个新 Cell 对真实观测 XYZ 去重，最多取 10 个；超过 10 个继续使用确定性最远点采样。**不再通过人工偏移补足十点**。
- 已占用的 `(level,key)` 继续跳过重复初始化；已有图像与深度仍参与正常优化。不会为了填满容量而重复导入观测。
- topology 事件从 Gaussian 轮开始，与 Anchor 轮交替。不是每次 Mapping 改变训练分支；原有 LoD/Full、ratio 和 optimizer 调度保持原配置。
- **Gaussian 轮**：先按原逐点 opacity/尺寸规则 pruning，满十点 Anchor 同样允许逐点删除。再在每个 Anchor 的空余容量内，按屏幕梯度从高到低选择原有 clone/split 候选；相同梯度按 source row 排序。每个 clone 净增 1 点，每个二分 split 删除父点后净增 1 点。
- **Anchor 轮**：按原 Full 屏幕梯度进行同层或下一层 growing；按原 Anchor opacity 窗口整体 pruning。`T_A > 0.8 × check_interval` 且 `O_A < 0.005 × T_A` 才整组删除，不加入失败成员比例规则。
- clone/split 保留父 Anchor 归属，不额外迁移已有成员。split 的追加与父点删除在无计算图、持有 Mapping 渲染锁的 topology 边界完成；渲染不可见中间态。结束时强制校验每 Anchor 不超过十点。

Gaussian 与 Anchor 增长分别累计 Full-render 屏幕梯度，避免一轮把另一轮尚未使用的证据清空。每类统计在自身回合完成后重置，新行从零累计；被删除行同步裁剪。

## 参数、窗口与几何状态

`rows[N]` 保存 Gaussian→Anchor；`member_counts[M]` 保存实际成员数。`topology_round` 记录交替相位，并进入 checkpoint。Gaussian 不再要求按十点连续排布。

- 部分成员增删不重建该 Anchor 的 Feature，保持 UID、Feature 及其 Adam 状态。
- 新成员的 Gaussian Adam 一、二阶状态为零，存活 Gaussian 的状态、step、Cached Uniform 与保护状态按原行映射保留。
- 空 Anchor 或整组删除时，同时裁剪两套 optimizer 持有的 Feature 状态；新增 Anchor 继续采用同层逐通道 max 聚合、细层零初始化，新增 Feature 的 Adam 状态为零。
- 仅成员集合发生变化的 Anchor 重启 opacity/slot 窗口；其他 Anchor 保留已有统计。这样不会把旧成员数的累计 opacity 与新成员数混在一个窗口。
- slot 窗口统计改为按 Gaussian 行保存，访问分母仍是所属 Anchor 的实际可见访问次数。未访问不作为 opacity=0。原停用开关、恢复规则保持可用。
- 几何修正继续保持稳定 UID 和碰撞桶；部分修正使用“受影响实际成员数 / 当前实际成员数”迁移 Anchor 参考坐标，不再除以固定十点。Teacher snapshot、Change Distill 和 Python 加载器携带可变成员布局。

保留原有 27 维网络、Loss、Cached Bernoulli、TF/CD、maturity=1、SI/GI disabled。没有新增损失或增长阈值。初始化的单点 opacity、局部尺度公式也保持此前显式 Anchor 实现；不另外按成员数重新校准 opacity。

## 使用方式

`cfg/selector/octree_explicit.yaml` 已启用：

```yaml
Selector.explicit_anchor_blocks: 1
Selector.variable_anchor_members: 1
```

设置 `variable_anchor_members: 0` 可复现固定十点路径；历史配置缺失该键时仍使用旧路径。新 checkpoint 通过成员计数 buffer 自动识别，旧固定十点 checkpoint 继续可加载。此处新增的是布局信息，Selector 输入与网络参数形状不变。

## 验证状态

**已通过功能验证**：服务器独立构建下的 11 项门控，覆盖 CPU/CUDA 初始化、容量、满 Anchor 逐点删除、clone/split、空 Anchor 回收、两类交替窗口、Feature 继承与 Adam 迁移、Teacher/checkpoint、C++/Python Cached Bernoulli 一致性、实际 Mapping 的梯度路由、全局/局部几何修正与 Selector/TF Change Distill。固定十点和原 Anchor pruning、串行 joint Mapping 回归均通过。本地另通过 Python 语法检查和 `git diff --check`。

证据位于 `tests/anchor_selector/variable_anchor_validation/`；真实训练与最终审计数据见该目录下的证据文件。服务器使用独立目录 `/home/prmi/zwj/variable-anchor-20261021`，不覆盖既有模型与生产源码。

**真实训练验证完成**：`fr1_xyz / seed2026 / 10,000 Mapping`。以此前串行 joint 的完整输入配置为基准，只增加 `Selector.variable_anchor_members: 1`；Full100 关闭，原 slot 停用配置保留。

| 项目 | 实测 |
|---|---:|
| Full / LoD Render | 各 10,000 次 |
| Master Gaussian Adam 更新 | 20,000 次 |
| 每个 ratio 的 LoD 更新 | 2,500 次 |
| 非 finite LoD | 0 |
| topology 事件 | 19：Gaussian 10，Anchor 9 |
| 最终 Anchor / Gaussian | 1,919 / 8,457 |
| L0 / L1 / L2 Anchor | 649 / 1,266 / 4 |
| 平均成员数 | 4.407 |
| 实际 split 父点 | 6,922 |
| 实际 clone 父点 | 0（单元测试已独立触发 clone） |
| 累计创建 / 删除 Anchor | 8,956 / 7,037（含逐点删除后变空） |
| 端到端训练 wall time | 91.17 秒；非严格性能对照 |
| 最终 Selector forward | 五次均值测量 0.160～0.162 ms，仅此地图规模 |

最终成员数直方图：

| 成员数 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Anchor 数 | 681 | 243 | 131 | 121 | 76 | 74 | 59 | 61 | 42 | 431 |

每个 topology 事件结束以及观测插入后都执行成员数与归属校验；最终 checkpoint 再独立重算 `bincount(rows)==member_counts`。19 个相位从零连续递增，严格交替，无超过十点的存活 Anchor。

最终 C++ 原生推理与 Python 四个 ratio 的 Cached Bernoulli mask exact 一致。1%/5%/10%/15% 实际选中 **87/419/836/1,249** 点，低于 renderer `1/255` 的选中点均为 **0**。这是最终 checkpoint 的选择统计，不代表重建质量或未来所有时刻都满足该性质。

最终本地独立复核通过：57 个同步源文件与 66 个交付证据文件的 SHA256、10,000 行训练 trace、19 个交替事件、成员数直方图和 PLY 实际点数全部匹配。此前串行 joint 实验源码的原 SHA 清单也继续全部匹配。训练、模型和证据使用独立目录，旧实验未被覆盖。

## 解释边界

本轮验证的是新生命周期能够训练、增长、裁剪并正确导出；**尚未证明 LoD 重建质量提高**。真实训练人口、初始覆盖和增长轨迹都发生变化，不应仅凭名义 ratio 或 Gaussian 数减少判定有效。

交替使每类 topology 操作每两个事件才执行一次；Anchor 的首次梯度标定窗口也随此变化。Gaussian pruning 会释放容量，但不保证最终不存在低 opacity 成员：最后一个逐点 pruning 之后仍可新增或降低 opacity。

原 Anchor pruning 比较的是成员 opacity 总和，成员数减少会自然改变保留概率；本轮没有改变其阈值。clone 仍按原 Gaussian densification 允许初始位置重合，这与观测初始化阶段的人工补点是不同操作。
