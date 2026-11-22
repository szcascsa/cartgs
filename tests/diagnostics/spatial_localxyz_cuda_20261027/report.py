"""从原始测量生成中文交付；不同范围/阶段不混合计算加速比。"""
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parent
BEST='graph_cached55_fused'


def table(frame,columns):
    rows=['| '+' | '.join(v for k,v in columns)+' |','| '+' | '.join('---' for _ in columns)+' |']
    for _,record in pd.DataFrame(frame).iterrows():
        values=[]
        for key,label in columns:
            value=record[key];values.append(f'{value:.6g}' if isinstance(value,(float,np.floating)) else str(value))
        rows.append('| '+' | '.join(values)+' |')
    return '\n'.join(rows)


def main():
    parity=pd.concat([pd.read_csv(ROOT/name) for name in ['parity_graph_cached55_fused.csv','parity_graph_cached55_fused_fr2_xyz.csv',
        'graph_cached55_fused_boundary_gate.csv']],ignore_index=True)
    steps=pd.read_csv(ROOT/'cached55_step_tensors.csv');assert parity.pass_.all() and steps.pass_.all()
    assert json.loads((ROOT/'graph_cached55_fused_lifecycle_gate.json').read_text())['passed']
    backward=pd.read_csv(ROOT/'cached55_paired_backward.csv');memory=pd.concat([pd.read_csv(ROOT/(name+'_persistent_memory.csv')) for name in ['iso_reference','iso_all','iso_shared','iso_cached55']])
    all_direct=pd.concat([pd.read_csv(ROOT/name) for name in ['parity_tensors.csv','parity_graph_all_fused.csv',
        'parity_graph_all_fused_fr2_xyz.csv','parity_graph_shared_fused.csv','parity_graph_shared_fused_fr2_xyz.csv',
        'parity_graph_cached55_fused.csv','parity_graph_cached55_fused_fr2_xyz.csv']],ignore_index=True)
    rows=[]
    for phase,prefix in [('initial','benchmark'),('packed_graph','release'),('shared_warmup','shared'),('matched_cache55','matched')]:
        for scene in ['fr1_xyz','fr2_xyz']:
            data=pd.read_csv(ROOT/f'{prefix}_{scene}.csv')
            grouped=data.groupby(['method','ratio','K','stage']).agg(ms=('cuda_median_ms','mean'),p10=('cuda_p10_ms','mean'),
                p90=('cuda_p90_ms','mean'),cpu_ms=('cpu_submit_median_ms','mean'),peak=('peak_increment_bytes','max'),setup_ms=('setup_ms','max'))
            for (method,ratio,K),g in grouped.groupby(level=[0,1,2]):
                def value(stage,field='ms'):return float(g.xs(stage,level='stage')[field].iloc[0])
                scope=backward[(backward.scene==scene)&(backward.method==method)&(backward.K==K)]
                observed=all_direct[all_direct.method==method]
                record=dict(phase=phase,scene=scene,method=method,ratio=ratio,K=K,scope='full selected-K compensation; Frozen Master',
                    forward_ms=value('forward'),backward_ms=float(scope.backward_median_ms.mean()) if len(scope) and phase=='matched_cache55' else np.nan,
                    backward_measurement_scope='separate cached55_paired_backward.csv run; not subtraction',
                    forward_backward_ms=value('forward_backward'),optimizer_step_ms=value('adam_only'),network_step_ms=value('optimizer_step'),
                    end_to_end_ms=value('renderer_loss_step'),forward_p10_ms=value('forward','p10'),forward_p90_ms=value('forward','p90'),
                    fb_p10_ms=value('forward_backward','p10'),fb_p90_ms=value('forward_backward','p90'),
                    end_to_end_p10_ms=value('renderer_loss_step','p10'),end_to_end_p90_ms=value('renderer_loss_step','p90'),
                    forward_cpu_submit_ms=value('forward','cpu_ms'),fb_cpu_submit_ms=value('forward_backward','cpu_ms'),
                    peak_increment_bytes=int(g['peak'].max()),first_shape_setup_ms=float(g.setup_ms.max()),
                    numerical_forward_max_abs=0. if method=='reference' else float(observed[~observed.backward].max_abs.max()),
                    numerical_backward_max_abs=0. if method=='reference' else float(observed[observed.backward].max_abs.max()),
                    numerical_error_scope='all recorded direct gates for this method; per-tensor files retained',
                    validation_status='PASS_DIRECT_AND_COMMON_VJP_STEP' if method==BEST else ('REFERENCE' if method=='reference' else 'CONTROL_NOT_RECOMMENDED'),
                    kernel_launch_count=np.nan,kernel_launch_scope='not available for packed graph with CUPTI; captured nodes reported separately')
                if phase=='initial':
                    path=ROOT/f'profile_summary_{scene}.csv'
                    pp=pd.read_csv(path);pp=pp[(pp.method==method)&(pp.K==K)]
                    if len(pp):record.update(kernel_launch_count=float(pp.kernels_per_step.iloc[0]),kernel_launch_scope='actual Renderer+Loss+Adam step under profiler')
                rows.append(record)
    results=pd.DataFrame(rows)
    for i,row in results.iterrows():
        base=results[(results.phase==row.phase)&(results.scene==row.scene)&(results.ratio==row.ratio)&(results.method=='reference')].iloc[0]
        for label,column in [('forward_speedup','forward_ms'),('fb_speedup','forward_backward_ms'),('e2e_speedup','end_to_end_ms')]:results.loc[i,label]=base[column]/row[column]
    results.to_csv(ROOT/'CUDA_PERFORMANCE_RESULTS.csv',index=False)
    main=results[(results.phase=='matched_cache55')&results.method.isin(['reference',BEST])].copy()
    main.to_csv(ROOT/'accepted_performance.csv',index=False)
    macro=main.groupby(['scene','method'])[['forward_ms','backward_ms','forward_backward_ms','network_step_ms','end_to_end_ms']].mean().reset_index()
    pairs=[]
    for scene in ['fr1_xyz','fr2_xyz']:
        a=macro[(macro.scene==scene)&(macro.method=='reference')].iloc[0];b=macro[(macro.scene==scene)&(macro.method==BEST)].iloc[0]
        pairs.append(dict(scene=scene,forward_reference=a.forward_ms,forward_optimized=b.forward_ms,forward_speedup=a.forward_ms/b.forward_ms,
            backward_reference=a.backward_ms,backward_optimized=b.backward_ms,fb_reference=a.forward_backward_ms,fb_optimized=b.forward_backward_ms,
            fb_speedup=a.forward_backward_ms/b.forward_backward_ms,e2e_reference=a.end_to_end_ms,e2e_optimized=b.end_to_end_ms,e2e_speedup=a.end_to_end_ms/b.end_to_end_ms))
    pd.DataFrame(pairs).to_csv(ROOT/'performance_summary.csv',index=False)
    m=memory.copy();m['persistent_MiB']=m.persistent_increment_bytes/2**20;m.to_csv(ROOT/'memory_summary.csv',index=False)
    direct=parity.copy();direct['group']=direct.tensor.str.split('/').str[0]
    errors=direct.groupby(['group','backward']).agg(tensors=('pass_','size'),max_abs=('max_abs','max'),max_relative=('max_relative','max'),max_rms=('rms','max')).reset_index()
    errors.to_csv(ROOT/'parity_summary_by_group.csv',index=False)
    training=json.loads((ROOT/'cached55_real_training_summary.json').read_text());quality=pd.read_csv(ROOT/'cached55_real_training_quality.csv')
    q=quality.groupby('method')[['psnr','photo']].mean().reset_index();tr=pd.DataFrame(training)
    tr['total_with_setup_ms']=tr.seconds*1000+tr.setup_ms
    original=tr[tr.method=='reference'].iloc[0];optimized=tr[tr.method=='optimized'].iloc[0]
    breakeven=(optimized.setup_ms-original.setup_ms)/(original.ms_per_step-optimized.ms_per_step)
    integration=pd.read_csv(ROOT/'step_parity_tensors.csv')
    integration_summary=integration.groupby('method').agg(max_abs=('max_abs','max'),failed_tensors=('pass_',lambda v:int((~v).sum()))).reset_index()
    node_rows=[]
    for path in sorted(ROOT.glob('*captured_K*.dot')):
        node_rows.append(dict(file=path.name,K=int(re.search(r'captured_K(\d+)',path.stem).group(1)),kernel_nodes=path.read_text().count('shape="octagon"')))
    pd.DataFrame(node_rows).to_csv(ROOT/'captured_kernel_nodes.csv',index=False)
    compiler=json.loads((ROOT/'compiler_probe.json').read_text());capture=json.loads((ROOT/'full_graph_probe.json').read_text())
    baseline=json.loads((ROOT/'environment.json').read_text());p1=pairs[0];p2=pairs[1]
    fwdcols=[('scene','地图'),('method','方法'),('forward_ms','Forward ms'),('backward_ms','Backward-only ms'),('forward_backward_ms','F+B ms'),('end_to_end_ms','Renderer+Loss+Adam ms')]
    perfcols=[('scene','地图'),('forward_speedup','Forward倍数'),('fb_speedup','F+B倍数'),('e2e_speedup','完整冻结步骤倍数')]
    perK=main[main.scene=='fr1_xyz'].pivot(index=['ratio','K'],columns='method',values=['forward_ms','forward_backward_ms','end_to_end_ms'])
    perK.columns=['_'.join(v) for v in perK.columns];perK=perK.reset_index()
    report=f'''# SpatialLocalXYZ CUDA性能优化报告

## 1. 结论与提交边界

当前代码基线已提交为 `74fc1524d8d6345eda8e732a1da2efa9a4934951`，包含Neural V1源码、复现配置和SpatialLocalXYZ研究默认。用户随后确认：**仅性能研究默认，生产配置保持不变**。本轮优化代码保留在新目录，未另行stage/commit/push，未修改生产Selector或Compensation。

推荐冻结地图后端为 **{BEST}：缓存确定性的完整55维输入＋严格FP32 FiLM/输出融合＋完整神经补偿前后向Graph＋复用预热Stream**。输入需要梯度时自动回到实时Triton打包。未改变33,806个共享参数、初始化、55维字段、FiLM、14D变换、原Loss、原Adam或ratio轮转。

同阶段、同K、交错配对的七档等权平均：

{table(macro,fwdcols)}

{table(pairs,perfcols)}

这些是**冻结Master的选后补偿与真实Renderer步骤**，不含SLAM跟踪、Mapping调度、拓扑维护或Full Gaussian优化，不能称完整SLAM提速。

## 2. 真实路径审计

原型为`neural.py:SharedNeural`，55→64→64，log-ratio小网络产生两层FiLM，四个独立64→64→输出Head，XYZ乘Anchor参考尺度。四Head权重仍完全独立；没有私有可训练表。所有Linear及其Backward保留PyTorch/cuBLAS，没有重写GEMM、TF32或低精度。

重要边界：原`Features.__init__`在`no_grad`下，预计算并detach了19维物理特征、归一化世界坐标和Anchor尺度。原每次Forward实际执行的是**19维Gather＋36维Fourier＋MLP**，并非从Master重新构建协方差/人口统计。首个融合核保持这一真实范围，没有把冻结优化冒充Online Joint。没有新加detach去截断既有梯度。

本地生产默认仍是原E50路径；服务器源码/Schema较旧，与本地不同，两端均未改动。`gumbel_network.h`/`gaussian_mapper.cpp`的输入、参数与Checkpoint布局不同。本轮没有将研究网络塞进不兼容生产布局，也未声称已在服务器部署E50或SpatialLocalXYZ生产版本。

## 3. 环境与原始Profile

RTX3090，Torch {baseline['torch']}、Triton {baseline['triton']}；TF32关闭。没有可用Nsight/nvcc，使用现有环境Triton实际JIT编译、PyTorch Profiler、CUDA Events及CUDA Graph原生DOT。没有安装或修改系统包。正式性能队列顺序执行，未终止其他用户作业；未锁GPU时钟。

原始K68：Forward约0.713ms、F+B约2.004ms、真实步骤约4.306ms。CPU提交与CUDA区间接近，大量极小Kernel之间存在提交空隙；不能把区间时长全当GPU算力时间。原始独立Profile在K68记录4次迭代共432个Kernel，包含额外独立input_pack探针及Adam，不能当纯Forward数量。

更严格的首轮端到端Profiler在K68记录reference约250个GPU Kernel/step、graph_fused约236个；CUDA Graph是减少**主机调度**，不是把所有设备工作变成一个Kernel。`profile_summary_fr1_xyz.csv`保存GPU active时间、allocation调用与Memcpy事件。Profiler本身扰动明显，GPU active和硬件占用率不是一回事；没有伪造occupancy或DRAM带宽。

全打包Graph参与同进程trace时，本环境Profiler在Collection后停滞，原因未最终归结为上游缺陷。仅终止本任务挂起进程，原日志保留。最终延迟用Events；captured Kernel节点用原生DOT补充：

{table(node_rows,[('file','CUDA原生转储'),('K','K'),('kernel_nodes','实际捕获Kernel节点')])}

DOT是神经图内部节点，不包括外部Renderer、Adam、输入copy和返回clone，不能与整步骤Kernel数量混为一谈。

## 4. 已实现优化及对照

1. Triton将19维Gather与36维Fourier直接写入连续K×55；使用`__nv_sinf/__nv_cosf`，禁用浮点乘加融合，不用`tl.sin`近似。
2. 两层FiLM的multiply/add/ReLU融合；Backward保留全部Gaussian对共享ratio参数的归约。使用PyTorch sum维持原归约形式。
3. 四Head输出拼接及XYZ乘Anchor尺度融合；其余属性解码和Quaternion处理仍走原Runtime。
4. 用Graph捕获整个补偿前后向，cuBLAS保持原样，原Adam在图外。Graph输入IDs/ratio有私有写入缓冲，输出clone避免后续replay污染已返回Tensor。每个Forward/Backward必须顺序消费。
5. 复用一个预热Stream，减少cuBLAS工作区；**不共享不同Graph的激活池**。Torch helper独立副本仅替换一行Stream构造，原三次warmup、capture/autograd逻辑不变，未改site-packages；来源和SHA在`shared_helper_provenance.json`。
6. 用户允许空间换时间后，增加Frozen-only的N×55静态输入缓存，fr1额外约1.435MiB、fr2约1.946MiB。缓存来自Gaussian显式属性和固定Fourier，**不是可学习个体特征**。物理输入/坐标需要梯度时禁用该缓存；PyTorch版本或storage变化触发重建，外部CUDA写入必须显式invalidate。

完整55维缓存相对已优化shared_warmup的额外收益主要在Forward，完整Renderer步骤收益较小，不能把原始→优化的总倍数全部归于静态缓存。最终matched阶段每个计时范围都从同一Checkpoint和Adam恢复，Adam-only探针不污染后续Renderer权重状态；此前阶段同样A/B匹配，但阶段内有连续更新，单独保留。

首轮单独Triton融合未提速：主机Autograd/launch增加抵消了Kernel减少。Graph解决主要提交瓶颈后才保留融合。Inductor同Core K68控制通过初步阈值，但冷编译约{compiler['cold_compile_ms']/1000:.3f}秒，Core Forward {compiler['reference_core']['cuda_median_ms']:.4f}→{compiler['compiled_core']['cuda_median_ms']:.4f}ms，无优势；仅K68，不进入推荐路径。

## 5. 逐K性能与公平性

fr1为68/170/341/512/683/854/1025；fr2为实际92/231/463/695/927/1159/1391。所有方法相同权重、输入、选择身份和真实photo VJP，预热后交错测量。fr1三块、fr2两块，每阶段35次样本。CSV每行是块内median再跨块平均，保留P10/P90及CPU提交时间。

{table(perK,[('ratio','ratio'),('K','K'),('forward_ms_reference','原Forward'),('forward_ms_'+BEST,'优化Forward'),('forward_backward_ms_reference','原F+B'),('forward_backward_ms_'+BEST,'优化F+B'),('end_to_end_ms_reference','原完整步骤'),('end_to_end_ms_'+BEST,'优化完整步骤')])}

Backward-only另用同一真实photo VJP交错测量，将Forward置于事件区间外，数据见`cached55_paired_backward.csv`。没有用F+B减Forward冒充Backward；独立计时不可要求严格相加。Adam-only是隔离计时，不能把该微基准当实际训练曝光。

## 6. 正确性与真实训练

最终推荐后端两地图及边界共{len(parity)}项直接比较、跨ratio12步共{len(steps)}项状态比较均通过原阈值。正常输入下网络输出/参数梯度在已测样本精确相同；完整误差含近零relative指标见`CUDA_PARITY_REPORT.md`。

三组从同一12k模型及完整Adam状态续训256个真实Bank photo步：原实现、原实现重复、优化。每ratio64次，使用原随机帧序的后续64轮；没有修改Loss、学习率、时钟或曝光。

{table(tr,[('method','方法'),('ms_per_step','稳态每步ms'),('setup_ms','首次建图ms'),('total_with_setup_ms','含初始化总ms'),('adam_clock','最终Adam时钟')])}

{table(q,[('method','方法'),('psnr','Bank留出PSNR'),('photo','Bank留出photo')])}

这是真实Frozen Renderer训练，不是在线SLAM。独立Renderer反传存在非确定性：原实现与自身重复也越过最初严格轨迹阈值。单独状态比较没有掩盖失败；用同一真实photo VJP隔离神经实现后，原Adam单步及跨ratio多步参数/Moments精确一致。不能用上述PSNR接近替代数值证明，也不能把一次轻微差值归因为系统性质量变化。

## 7. 初始化与显存成本

256步优化稳态约{original.ms_per_step/optimized.ms_per_step:.3f}倍，但首次4个训练shape建图约{optimized.setup_ms/1000:.3f}秒；包含初始化后，这段短任务反而更慢。按当前平均步耗时估算约{breakeven:.0f}步后才抵消额外初始化。这只是当前固定shape的估算，不是在线保证。

以下每种后端由独立进程测量，含网络、原Adam、七shape Graph及最后Backward缓冲；不含共同Fixture/地图。fr1为该进程冷测，fr2继承运行库工作区，是warm测，不能把差异当场景复杂度效应。

{table(m,[('scene','地图/冷暖'),('method','后端'),('persistent_MiB','常驻增量MiB'),('graphs','Graph数'),('parameter_bytes','权重bytes'),('optimizer_moment_bytes','Adam双Moments bytes')])}

共享预热Stream显著降低工作区开销，但Graph仍比eager占更多显存。参数保持135224 bytes、Adam双Moments270448 bytes；没有新增可学习参数。每次步骤峰值增量另在CSV，不是整张地图显存。

## 8. 失败尝试和回退

- 最初全打包Graph把调用方ratio缓存当可写静态缓冲，单次Forward通过、第五次跨ratio更新失败；已定位、隔离缓冲并通过完整重验，旧336项失败记录保留在`common_vjp_tensors.csv`，不用于推荐。
- Torch2.3.1的`num_warmup_iters=0`触发未初始化局部变量错误；没有修补安装包，改用可审计的一行Stream替换独立副本。
- 原始完整步骤Graph捕获在Adam处失败：`{capture.get('error','')}`。未改capturable或任何Optimizer配置，Adam保留图外。该错误不能直接证明Renderer本身一定不可capture。
- 源码另有num_rendered和bucket_sum的Device-to-Host计数及动态buffer布局；没有用过期CPU计数、固定有效点数或截断工作区来换取整Renderer捕获。若继续GPU化这些计数，需要单独的Renderer等价工程。
- Inductor未形成收益，单独eager融合略慢，未推荐。没有继续手写SGEMM或合并成共享Head：剩余GEMM设备工作很小，而Renderer及主机路径已占主要时间。

## 9. Online Joint与交付状态

**IMPLEMENTED / TESTED：** Frozen Master七档、两地图、FP32输入/中间层/输出/参数及输入VJP、真实短续训、生命周期缓冲与Graph重建。

**PARTIALLY VERIFIED：** 输入leaf的梯度正确，但原原型没有raw Master→19维统计的活跃梯度路径；这不是生产Joint梯度接入证明。合成feature表append/prune/reorder验证不等于Gaussian optimizer与闭环集成。

**NOT IMPLEMENTED：** 生产C++/CUDA接入、真实SLAM、Full优化竞争、动态地图归一化策略、缓存自动逐步维护、ATE或15%以上ratio质量。动态K/N需要新图或安全边界invalidate，频繁重建会损害收益。参数storage替换必须invalidate；普通原地Adam更新不用。当前优化接口针对顺序训练及`no_grad`评价，不宣称支持任意并发Forward或`inference_mode`/整体requires_grad关闭的全部组合。

性能研究默认在`execution_config.json`，reference可随时切换；原`neural.py`与生产默认未覆盖。四份报告、逐张量/逐K原始数据、代码版本、构建运行命令及Checkpoint随包交付。
'''
    (ROOT/'SPATIAL_LOCALXYZ_CUDA_OPTIMIZATION_REPORT.md').write_text(report,encoding='utf-8')
    parity_report=f'''# CUDA数值与训练状态等价报告

## 1. 阈值和输入

同一非零SpatialLocalXYZ Checkpoint、FP32、TF32关闭、相同K和选择身份。Forward atol=1e-6/rtol=1e-5；Backward atol=1e-5/rtol=1e-4，按每元素`abs(a-b) <= atol+rtol*abs(a)`，未放宽阈值。relative分母截断1e-12，同时报告绝对误差和RMS。

## 2. 最终推荐后端逐组误差

{table(errors,[('group','张量组'),('backward','Backward'),('tensors','比较数'),('max_abs','最大绝对误差'),('max_relative','最大相对误差'),('max_rms','最大张量RMS误差')])}

输入55、Encoder Linear/FiLM、ratio调制向量、body、四Head、14D、XYZ/Scale/Rotation/DC/高阶SH/Opacity、Renderer和photo均逐张量比较。完整数值在`parity_graph_cached55_fused.csv`与`parity_graph_cached55_fused_fr2_xyz.csv`。极端合法输入Scale1e-6到100、规范Quaternion、世界坐标量级1000、Opacity logit±20、K0/1/17及重复索引的误差在`graph_cached55_fused_boundary_gate.csv`。没有渲染巨大合成Gaussian来伪造质量证据。

## 3. Adam单步和跨ratio多步

同一原始Optimizer状态，12个真实步、1/5/10/15%轮转；每步参数、step、exp_avg、exp_avg_sq及下一次7.5%输出共{len(steps)}项，最大绝对误差 **{steps.max_abs.max():.9g}**。上游采用同一真实Renderer photo VJP隔离网络计算，不能称两次独立Renderer反传。

## 4. 独立Renderer反传的严格失败记录

{table(integration_summary,[('method','方法'),('max_abs','最大绝对误差'),('failed_tensors','超初始阈值张量数')])}

原实现重复本身存在差异；没有因此把失败改为PASS或直接放宽阈值。真实256步的损失与Bank留出结果另存，作为轨迹波动观测，不取代上述共享VJP下的精确参数/Moments验证。

## 5. 梯度与生命周期边界

检查了全部网络参数、19维输入表、世界坐标表和Anchor参考尺度的VJP。原Features在no_grad下预处理，不能声称已测试完整Master原始属性经动态人口归一化的Joint梯度。ratio网络梯度包含全部K行贡献，没有近似梯度。安全边界的输入表替换、原地更新、append/prune/reorder、同K不同ratio/IDs、输出保留和invalidate通过；不是生产在线几何系统测试。

结论：推荐神经实现通过直接数值与隔离Adam状态验收；完整独立Renderer训练不是bitwise轨迹等价，严格失败数如实保留。未进行生产替换。
'''
    (ROOT/'CUDA_PARITY_REPORT.md').write_text(parity_report,encoding='utf-8')
    implementation='''# 优化实现、运行与在线接口

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
'''
    (ROOT/'CUDA_OPTIMIZED_IMPLEMENTATION.md').write_text(implementation,encoding='utf-8')
    summary=dict(best=BEST,baseline_commit='74fc1524d8d6345eda8e732a1da2efa9a4934951',performance=pairs,
        direct_tensors=len(parity),direct_pass=True,step_tensors=len(steps),step_max_abs=float(steps.max_abs.max()),
        estimated_amortization_steps=float(breakeven),production_changed=False,optimization_committed=False,
        limitations=['Frozen Master only','independent Renderer backward nondeterministic','CUPTI replay trace hangs','graph initialization and memory overhead'])
    (ROOT/'optimization_summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
