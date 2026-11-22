# R3 / R2在线Gaussian-private 14D补偿

本轮将R3设为实验默认，保留R2和旧共享MLP为显式配置。目标是实际SLAM训练与ratio泛化对照，不将离线Oracle收益当作在线保证。

## 定义

`Selector.compensation_method`：

- `r3_independent14`（默认）：`δ_i(r)=stopgrad(C(h_i(r)))+b_i,r`，四张Gaussian身份绑定14D表，对应1/5/10/15%。
- `r2_shared14`：`δ_i(r)=stopgrad(C(h_i(r)))+b_i`，一张共享14D表。
- `shared_mlp`：原共享补偿头训练路径，兼容既有checkpoint。

C参数在私有模式冻结。全新SLAM初始化C最终层为零，因此初始C输出与私有残差均为exact零；没有预先加载离线Oracle。Selector、Encoder、Anchor和Master按原joint训练，Full仅Master，补偿仍从iteration3000开始。

R3未见ratio对相邻训练节点的**14D补偿系数**按raw ratio线性插值，非新增潜变量/MLP。区间外使用最近端点；ratio1%与5%之间的3%为两表均值。四元数仍按原代码在属性应用时归一化，插值原始系数不保证所有ratio渲染质量或SO(3)最短路径。

## 在线身份与状态

私有状态独立于Anchor Feature。新增真实观测Gaussian残差为零；clone/split子点继承父残差、分配新UID、Adam新行moment为零。prune/reorder按同一source row压缩表与Adam状态；Anchor重关联不更换Gaussian私有参数。

几何修正时XYZ残差按Sim(3)的线性部分迁移，Rotation补偿做四元数共轭，使基础几何与补偿方向一致。对受迁移身份重置Adam行moment（对角二阶moment无法精确旋转），保留其他行及表step。平移不改变位移残差；Scale/opacity/SH DC残差保留。

表属于Selector模块，复用已有Adam组及拓扑安全边界，不混入前六组Master属性。R3非当前ratio表grad=None；R2一张表接收全部ratio梯度，更新频次差异作为实验事实记录。Forward只gather选中K行；现有dense Adam仍会处理整表状态，不宣称优化器仅K行成本。

ChangeDistill本轮保持关闭；私有模式显式拒绝开启ChangeDistill，避免默默改变其teacher/student目标。teacher clone和C++/Python checkpoint均包含私有表、ratio节点和Gaussian UID。

## 验证与训练协议

本地native及Python测试覆盖step0、R3档位梯度隔离、冻结C/Encoder补偿梯度、插值/端点、append/prune/reorder/Adam、几何迁移、teacher、checkpoint一致性。服务器CPU/CUDA及真实Gaussian clone/split/prune生命周期回归已通过；原共享MLP、joint Mapping与Anchor迁移回归也通过。此处不把功能通过表述为在线质量收益。

在线使用fr1_xyz，seed2026/2027，先R3再R2；原joint、自然停止、`max_mapping_iterations=0`，不强制10000。训练档位仍1/5/10/15%，其余Loss/LR/TF/SI/GI/Anchor策略不变。

用户确认评价1%、5%、10%…100%，补充2.5/7.5/12.5%区间内泛化；Full单独渲染Master。>15%同时存在Selector输入外推与R3补偿端点使用，不能归为纯补偿插值能力。比较原Cached实际预算与统一绝对K两种协议，报告Full、PSNR/SSIM/LPIPS/photo、点数、轨迹/更新次数及时间显存差异。

本文件只描述实现与锁定协议。功能测试不代表在线质量更好；正式结果在训练完成后独立报告。
