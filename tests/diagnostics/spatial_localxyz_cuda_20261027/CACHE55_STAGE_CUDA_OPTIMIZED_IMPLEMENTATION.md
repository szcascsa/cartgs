# 优化实现、运行与在线接口

## 1. 三条执行路径

`execution_config.json`：reference=原PyTorch；optimized=graph_cached55_fused；experimental=单独Triton融合。还保留graph_reference、graph_fused、graph_all_fused、graph_shared_fused作为逐阶段控制。模型始终是SpatialLocalXYZ。

```python
torch.backends.cuda.matmul.allow_tf32 = False
from implementation import make_execution
execute = make_execution(network, backend="optimized")
optimizer.zero_grad(set_to_none=True)
delta = execute(ratio, selected_indices)
loss = original_photo(original_renderer(delta), ground_truth)
loss.backward()
optimizer.step()
```

参数仍属于原network；不要再次注册或重复优化Core的别名参数。Checkpoint直接保存原network与原optimizer；Graph和只读缓存不进入模型参数。`reference`可随时回退。

## 2. Triton内核

- Pack：Own[N,19]、Coords[N,3]、IDs[K]、Freq[6] → Out[K,55]。顺序为19物理、18sin、18cos；grid=ceil(K*55/256)，每program256逻辑元素、默认4warps；FP32 libdevice __nv_sinf/__nv_cosf，enable_fp_fusion=False。
- Pack VJP：19维scatter-add和6频带坐标导数，重复ID累加；完整N行梯度初始化。Frozen输入无需梯度时Autograd自然不执行该VJP，并未删除活跃梯度。
- FiLM：H[K,64]、g/b[1,64] → ReLU((1+g)*H+b)。grid=ceil(K*64/256)，不将乘加改成FMA。Backward生成dH/dg-row/db-row，PyTorch sum聚合所有K行，保留ratio网络完整梯度。
- Output：XYZ3/Scale3/Quaternion4/Other4及AnchorScale[K,1] → K×14，XYZ乘参考尺度，grid=ceil(K*14/256)。Backward保留Anchor尺度导数，属性解码仍是原PyTorch Renderer路径。

不使用CUDA Extension或nvcc，不改库安装。所有GEMM、Head独立权重、Linear参数梯度和Adam均仍由PyTorch/cuBLAS负责。未将四Head隐藏层共享化。

## 3. Graph、内存和版本

Graph捕获完整补偿前后向；Adam图外。只读表按storage/N/K/梯度模式建key，IDs/ratio是私有可写静态缓冲，返回clone保持旧输出不被后续replay覆写。不能在同一Graph上积压多个Forward后再任意乱序Backward。

Shared warmup只把已安装Torch make_graphed_callables的一行`torch.cuda.Stream()`替换为调用方共享预热Stream，其余源代码不变；不共享不同Graph的激活池、不改site-packages。实际副本、原/改SHA、源码快照均保存。该做法依赖当前Torch版本，升级后必须重做gates；不隐式沿用未知版本。

全打包Graph的CUPTI trace后处理出现停滞；最终延迟来自Events，神经图内部Kernel数来自CUDA原生DOT。Graph节点数不是整步骤Kernel数。首次捕获、Graph常驻显存和每步临时峰值分开报告。

空间换时间路径缓存N×55的确定性输入；它没有可训练行。own/coords需要梯度时自动使用实时Pack VJP，不截断输入梯度。缓存用父表storage、shape、PyTorch版本追踪；外部CUDA原地写入不会自动增加版本，调用方必须显式invalidate。N或几何变化还必须正确更新原有19维生产者，不能用新缓存遮掩旧数据。

## 4. 冻结与在线边界

当前原Features仅在冻结地图初始化时计算19维、坐标及统计；没有完整raw Master Joint路径。在线若原地更新Master属性，必须先按同一规则更新输入表；改变归一化参考需另立语义决定，不能静默采用过期值。不能把新的图捕获或detach当作实现这一更新。

输入表新storage/N/K会创建新Graph，旧图仍占内存；在无未完成Forward/Backward的安全边界调用execute.invalidate()释放旧图。参数storage替换、to(device/dtype)、模型结构或requires_grad模式变化也必须重建。普通原地Adam更新使用当前权重，无需重建。当前没有自动LRU或后台异步拓扑维护。

原始num_rendered/bucket_sum存在CPU读取和动态buffer尺寸。允许更多显存不代表可以复用旧计数或改变有效tile数；本轮没有改Renderer调度算法。

初始化约秒级，需要长期复用shape才能摊销；动态图不能直接套用稳态倍数。优化接口验证顺序训练及no_grad评价；inference_mode/完全冻结网络的特殊捕获组合未作为推荐范围。生产C++入口完全未修改。

## 5. 构建和复现

服务器目录/home/prmi/zwj/spatial-localxyz-cuda-20261027，使用/home/prmi/anaconda3/envs/zwj/bin/python。无需隔离构建，Triton首次调用JIT编译。原地图、相机和历史参考脚本路径见spatial_localxyz_default.json及baseline_profile.py；包不包含数据集或完整Master点云。

```bash
python baseline_profile.py
python parity.py --methods graph_shared_fused
python parity.py --scene fr2_xyz --methods graph_shared_fused
python step_parity.py --common-vjp --methods graph_shared_fused --tag shared_step
python lifecycle_gate.py --method graph_shared_fused
python boundary_gate.py --method graph_shared_fused
python benchmark.py --methods reference graph_all_fused graph_shared_fused --tag shared
python benchmark.py --scene fr2_xyz --blocks 2 --methods reference graph_all_fused graph_shared_fused --tag shared
python paired_backward.py
python extra_benchmarks.py --methods graph_shared_fused --tag iso_shared_
python extra_benchmarks.py --methods graph_all_fused --tag iso_all_
python extra_benchmarks.py --methods reference --tag iso_reference_
python real_training.py --optimized-method graph_shared_fused --tag shared_
python compiler_probe.py
python full_graph_probe.py
python graph_nodes.py
python report.py
```

最终空间换时间后端的完整复现还包括：

```bash
python parity.py --methods graph_cached55_fused
python parity.py --scene fr2_xyz --methods graph_cached55_fused
python step_parity.py --common-vjp --methods graph_cached55_fused --tag cached55_step
python lifecycle_gate.py --method graph_cached55_fused
python boundary_gate.py --method graph_cached55_fused
python benchmark.py --methods reference graph_all_fused graph_shared_fused graph_cached55_fused --tag matched
python benchmark.py --scene fr2_xyz --blocks 2 --methods reference graph_all_fused graph_shared_fused graph_cached55_fused --tag matched
python paired_backward.py --methods reference graph_cached55_fused --tag cached55_
python extra_benchmarks.py --methods graph_cached55_fused --tag iso_cached55_
python real_training.py --optimized-method graph_cached55_fused --tag cached55_
python graph_nodes.py --method graph_cached55_fused
```

以上命令会重写同名实验输出；新重复应复制目录或调整tag。source code_versions与run_versions记录最终进程加载前源码SHA；首批探索没有完整import快照，保留阶段日志及已知失败版本，不能伪称每次探索均有精确loaded SHA。
