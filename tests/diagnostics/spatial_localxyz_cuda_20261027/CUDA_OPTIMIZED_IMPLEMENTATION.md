# 优化实现与复现

## 1. 两个明确的计算范围

Compensation-only入口`implementation.make_execution`：reference原PyTorch，optimized为graph_cached55_fused，experimental为eager Triton fused。返回K×14。

NN+原生Gaussian解码入口`decoded_graph.make_render_execution`：reference原网络+原解码，optimized为decoded_graph，experimental为已验收的cached55网络+图外解码。返回(delta, xyz, scale, rotation, dc, opacity)，或调用render原Renderer。默认仅性能研究；生产不接入。

```python
torch.backends.cuda.matmul.allow_tf32 = False
from decoded_graph import make_render_execution
execute = make_render_execution(network, scene.render_bases, backend="optimized")
# 首次建图在无未完成Autograd图的安全边界。
with torch.no_grad():
    execute(ratio, ids)
optimizer.zero_grad(set_to_none=True)
image = execute.render(scene, ratio, ids, view)
loss = original_photo(image, view["gt"])
loss.backward()
optimizer.step()
```

Parameter仍是原network对象；不得把Core别名再注册进Optimizer。Checkpoint仅存原network和原optimizer，缓存/Graph不作为可训练行保存。参数清单见parameter_inventory.csv：两地图均33806，shape完全相同，不依赖N/M。

## 2. 计算与内存

Pack：Own[N,19]/Coords[N,3]/IDs[K]/Freq[6]→K×55；顺序19物理、18sin、18cos，grid=ceil(K*55/256)、每program256元素、默认4warps。libdevice sin/cos，enable_fp_fusion=False。VJP scatter原19D和六频带coords导数，重复ID累加，输出完整N行梯度；Frozen不需要输入梯度时不执行该VJP。

FiLM：H[K,64]及g/b[1,64]→ReLU((1+g)*H+b)，grid=ceil(K*64/256)。Backward dH/dg-row/db-row融合，PyTorch sum聚合K行，保持ratio全部参数梯度。

Output：四Head3/3/4/4+AnchorScale[K,1]→K×14，XYZ乘尺度；grid=ceil(K*14/256)。VJP保留尺度导数。

原GEMM及参数梯度仍cuBLAS；四Head权重独立，不合并隐藏层。原生exp/sigmoid/Quaternion归一化、乘法、DC/XYZ加法被capture，未改数学函数。高阶SH Gather、means2D、Renderer、Loss及原Adam留图外。无CUDA Extension/nvcc构建，不改安装库。

Frozen缓存N×55确定性输入，不含可学习行。own/coords需梯度时自动实时Pack，解码入口回图外原生decode，保持梯度。bases/Anchor尺度单独需梯度可走解码Graph，已经VJP验证。

Graph固定K与输入storage/shape/requires_grad，IDs和ratio独立clone为可写缓冲，输出clone防replay覆盖旧返回值。仅共享预热Stream，不共享激活内存池。独立Torch helper只修改一行Stream构造，原三次预热与capture/autograd保留，provenance含原改SHA；升级Torch必须重验。

## 3. 生命周期与在线边界

输入表storage/版本变化重建缓存；bases原地更新读当前值，storage变更重建图。外部CUDA写入必须显式invalidate。拓扑/参数storage/梯度模式变化需在安全训练边界invalidate；替换bases时同步execute.bases和scene.render_bases。

原19D特征生产者仍Frozen/no_grad，网络优化不等于实现raw Master Joint。在线必须先正确更新19D、coords、Anchor尺度及归一化参考，再维护缓存。频繁K/N改变会增加捕获和显存；旧图需显式释放，本轮未做自动LRU。不得缓存旧Renderer计数改变渲染工作量。

不支持同图多个未消费Forward、乱序Backward；首捕获时不能有同参数未完成Autograd图。已验证顺序训练和no_grad评价，不宣称inference_mode和并发全组合。

## 4. 复现命令

目录/home/prmi/zwj/spatial-localxyz-cuda-20261027；Python /home/prmi/anaconda3/envs/zwj/bin/python。Triton首次调用JIT，无隔离构建。源数据/参考脚本在spatial_localxyz_default.json及baseline_profile.py；本归档不复制数据集和完整Master地图。

```bash
python baseline_profile.py
python parity.py --methods graph_cached55_fused
python parity.py --methods graph_cached55_fused --scene fr2_xyz
python step_parity.py --common-vjp --methods graph_cached55_fused --tag cached55_step
python lifecycle_gate.py --method graph_cached55_fused
python boundary_gate.py --method graph_cached55_fused
python benchmark.py --methods reference graph_all_fused graph_shared_fused graph_cached55_fused --tag matched
python benchmark.py --scene fr2_xyz --blocks 2 --methods reference graph_all_fused graph_shared_fused graph_cached55_fused --tag matched
python paired_backward.py --methods reference graph_cached55_fused --tag cached55_
python decoded_validation.py
python decoded_validation.py --scene fr2_xyz
python decoded_benchmark.py
python decoded_benchmark.py --scene fr2_xyz --blocks 2
python real_training.py --optimized-method decoded_graph --tag decoded_
python decoded_benchmark.py --memory-only --method decoded_graph
python decoded_benchmark.py --memory-only --method decoded_graph --scene fr2_xyz
python decoded_benchmark.py --memory-only --method reference
python decoded_benchmark.py --memory-only --method reference --scene fr2_xyz
python decoded_benchmark.py --memory-only --method graph_cached55_fused
python decoded_benchmark.py --memory-only --method graph_cached55_fused --scene fr2_xyz
python graph_nodes.py --method decoded_graph
python final_interface_gate.py
python final_report.py
```

相同命令会重写同名输出，新重复应使用复制目录/tag。code_versions和run_versions保存进程导入时源码SHA与命令；初期探索并非每次都有完整import快照，失败日志/已知错误版本保留，不伪造完整溯源。源码、指标、日志、trace/DOT及续训Checkpoint按manifest归档；历史Neural V1结果、生产文件未覆盖。
