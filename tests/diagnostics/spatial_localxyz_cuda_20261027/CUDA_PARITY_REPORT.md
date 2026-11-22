# CUDA数值与训练状态等价报告

## 1. 验收约束

FP32、TF32=False，无fast math、无近似梯度、无删减曝光。Forward atol=1e-6/rtol=1e-5，Backward atol=1e-5/rtol=1e-4；逐元素abs差≤atol+rtol×abs(reference)，未放宽。relative分母clamp1e-12，同时报告max abs与RMS。

## 2. 解码Graph逐组实际误差

| 组 | Backward | 张量数 | max abs | max relative | 最大RMS |
| --- | --- | --- | --- | --- | --- |
| boundary | False | 36 | 0 | 0 | 0 |
| gradient | True | 336 | 0 | 0 | 0 |
| input_vjp_graph | True | 28 | 0 | 0 | 0 |
| input_vjp_live | True | 54 | 1.49012e-08 | 1.87815e-05 | 1.67123e-10 |
| lifecycle | False | 72 | 0 | 0 | 0 |
| next_step_output | False | 144 | 0 | 0 | 0 |
| output | False | 84 | 0 | 0 | 0 |
| photo | False | 14 | 0 | 0 | 0 |
| preserved_output | False | 60 | 0 | 0 | 0 |
| renderer | False | 14 | 0 | 0 | 0 |

共842项，详细数据`decoded_direct_fr1_xyz.csv`、`decoded_direct_fr2_xyz.csv`。包括七档训练/未见ratio、输出14D和所有解码属性、原Renderer、photo、全部共享参数梯度，输入leaf实时及缓存Graph路径，K0/1/17、重复IDs、Scale1e-6~100、世界坐标1000量级、opacity logit±20。无把巨大合成点云渲染冒充质量测试。

19D输入→55D、Encoder、FiLM、四Head内部还沿用原已验收Core；两个地图与边界655项在`parity_graph_cached55_fused*.csv`和`graph_cached55_fused_boundary_gate.csv`，逐组误差另见CACHE55_STAGE数值报告。默认配置入口14项在configured_default_parity.csv。

## 3. 单步/跨ratio原Adam

每地图12步，原12000状态及原Adam：1%/5%/10%/15%轮转。解码Graph共2304项参数、step、exp_avg、exp_avg_sq，max abs=0；下一步7.5%输出6张量在direct文件记录。使用同一真实photo对解码属性的VJP隔离Renderer非确定性；无改Loss或Optimizer。

## 4. 独立Renderer反传的失败如实保留

| 方法 | max abs | 超过初始严格轨迹阈值项数 |
| --- | --- | --- |
| fused | 4.72069e-05 | 32 |
| graph_fused | 4.95911e-05 | 16 |
| graph_reference | 0.000175476 | 33 |
| reference_repeat | 5.67436e-05 | 40 |

原实现重复也失败，未将其改PASS。原失败step_parity_tensors.csv与初版输入别名bug失败common_vjp_tensors.csv均保留。后者已定位修复，不混入验收；独立Renderer长程不保证bitwise轨迹。真实256步见decoded_real_training_*，不以最终PSNR接近代替梯度/Adam证明。

## 5. 失效与使用边界

同K不同ratio/IDs、返回Tensor保留、物理表及bases原地修改、append、prune/reorder、invalidate全部逐张量检查。原Features no_grad行为未变：没有声称raw Master Joint或生产几何生命周期已集成。首捕获应在安全初始化边界，不能保留同参数未消费的Autograd图。普通调用按Forward→Backward顺序消费；不支持同图积压多次Forward后乱序反传。
