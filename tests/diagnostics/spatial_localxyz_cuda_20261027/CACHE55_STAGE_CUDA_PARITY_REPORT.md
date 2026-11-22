# CUDA数值与训练状态等价报告

## 1. 阈值和输入

同一非零SpatialLocalXYZ Checkpoint、FP32、TF32关闭、相同K和选择身份。Forward atol=1e-6/rtol=1e-5；Backward atol=1e-5/rtol=1e-4，按每元素`abs(a-b) <= atol+rtol*abs(a)`，未放宽阈值。relative分母截断1e-12，同时报告绝对误差和RMS。

## 2. 最终推荐后端逐组误差

| 张量组 | Backward | 比较数 | 最大绝对误差 | 最大相对误差 | 最大张量RMS误差 |
| --- | --- | --- | --- | --- | --- |
| attribute | False | 84 | 0 | 0 | 0 |
| body_film | False | 14 | 0 | 0 | 0 |
| body_linear | False | 14 | 0 | 0 | 0 |
| delta | False | 14 | 0 | 0 | 0 |
| encoder_film | False | 14 | 0 | 0 | 0 |
| encoder_linear | False | 14 | 0 | 0 | 0 |
| extreme_delta_0.01 | False | 4 | 0 | 0 | 0 |
| extreme_delta_0.025 | False | 4 | 0 | 0 | 0 |
| extreme_delta_0.075 | False | 4 | 0 | 0 | 0 |
| extreme_delta_0.125 | False | 4 | 0 | 0 | 0 |
| extreme_delta_0.15 | False | 4 | 0 | 0 | 0 |
| extreme_gradient_anchor_scale | True | 3 | 0 | 0 | 0 |
| extreme_gradient_coords | True | 3 | 7.45058e-09 | 6.45222e-07 | 2.59372e-09 |
| extreme_gradient_physical | True | 3 | 0 | 0 | 0 |
| input55 | False | 14 | 0 | 0 | 0 |
| input_gradient | True | 24 | 2.98023e-08 | 6.96136e-05 | 1.36448e-09 |
| other_head | False | 14 | 0 | 0 | 0 |
| parameter_gradient | True | 336 | 0 | 0 | 0 |
| photo | False | 14 | 0 | 0 | 0 |
| ratio_modulation | False | 14 | 0 | 0 | 0 |
| renderer | False | 14 | 0 | 0 | 0 |
| rotation_head | False | 14 | 0 | 0 | 0 |
| scale_head | False | 14 | 0 | 0 | 0 |
| xyz_head | False | 14 | 0 | 0 | 0 |

输入55、Encoder Linear/FiLM、ratio调制向量、body、四Head、14D、XYZ/Scale/Rotation/DC/高阶SH/Opacity、Renderer和photo均逐张量比较。完整数值在`parity_graph_cached55_fused.csv`与`parity_graph_cached55_fused_fr2_xyz.csv`。极端合法输入Scale1e-6到100、规范Quaternion、世界坐标量级1000、Opacity logit±20、K0/1/17及重复索引的误差在`graph_cached55_fused_boundary_gate.csv`。没有渲染巨大合成Gaussian来伪造质量证据。

## 3. Adam单步和跨ratio多步

同一原始Optimizer状态，12个真实步、1/5/10/15%轮转；每步参数、step、exp_avg、exp_avg_sq及下一次7.5%输出共1176项，最大绝对误差 **0**。上游采用同一真实Renderer photo VJP隔离网络计算，不能称两次独立Renderer反传。

## 4. 独立Renderer反传的严格失败记录

| 方法 | 最大绝对误差 | 超初始阈值张量数 |
| --- | --- | --- |
| fused | 4.72069e-05 | 32 |
| graph_fused | 4.95911e-05 | 16 |
| graph_reference | 0.000175476 | 33 |
| reference_repeat | 5.67436e-05 | 40 |

原实现重复本身存在差异；没有因此把失败改为PASS或直接放宽阈值。真实256步的损失与Bank留出结果另存，作为轨迹波动观测，不取代上述共享VJP下的精确参数/Moments验证。

## 5. 梯度与生命周期边界

检查了全部网络参数、19维输入表、世界坐标表和Anchor参考尺度的VJP。原Features在no_grad下预处理，不能声称已测试完整Master原始属性经动态人口归一化的Joint梯度。ratio网络梯度包含全部K行贡献，没有近似梯度。安全边界的输入表替换、原地更新、append/prune/reorder、同K不同ratio/IDs、输出保留和invalidate通过；不是生产在线几何系统测试。

结论：推荐神经实现通过直接数值与隔离Adam状态验收；完整独立Renderer训练不是bitwise轨迹等价，严格失败数如实保留。未进行生产替换。
