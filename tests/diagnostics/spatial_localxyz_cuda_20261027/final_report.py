"""汇总经过验收的两类计算范围，保留此前阶段报告和失败证据。"""
import json
import importlib.util
import re
import numpy as np
import pandas as pd
from baseline_profile import ROOT
spec=importlib.util.spec_from_file_location('cuda_stage_report',ROOT/'report.py')
previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)


def md(data,columns):return previous.table(data,columns)


def main():
    previous.main()
    for name in ['SPATIAL_LOCALXYZ_CUDA_OPTIMIZATION_REPORT.md','CUDA_PARITY_REPORT.md','CUDA_OPTIMIZED_IMPLEMENTATION.md']:
        (ROOT/('CACHE55_STAGE_'+name)).write_bytes((ROOT/name).read_bytes())
    early=pd.read_csv(ROOT/'CUDA_PERFORMANCE_RESULTS.csv');rows=[]
    direct=pd.concat([pd.read_csv(ROOT/f'decoded_direct_{s}.csv') for s in ['fr1_xyz','fr2_xyz']])
    steps=pd.concat([pd.read_csv(ROOT/f'decoded_steps_{s}.csv') for s in ['fr1_xyz','fr2_xyz']])
    assert direct.pass_.all() and steps.pass_.all()
    assert json.loads((ROOT/'final_interface_gate.json').read_text())['pass_']
    memory=pd.concat([pd.read_csv(p) for p in sorted(ROOT.glob('decoded_memory_*_xyz.csv'))],ignore_index=True)
    memory['persistent_MiB']=memory.persistent_increment_bytes/2**20;memory.to_csv(ROOT/'decoded_memory_summary.csv',index=False)
    for scene in ['fr1_xyz','fr2_xyz']:
        raw=pd.read_csv(ROOT/f'decoded_benchmark_{scene}.csv')
        back=pd.read_csv(ROOT/f'decoded_backward_{scene}.csv')
        grouped=raw.groupby(['method','ratio','K','stage']).agg(ms=('cuda_median_ms','mean'),p10=('cuda_p10_ms','mean'),p90=('cuda_p90_ms','mean'),cpu=('cpu_submit_median_ms','mean'),peak=('peak_increment_bytes','max'),setup=('setup_ms','max'))
        for (method,ratio,K),g in grouped.groupby(level=[0,1,2]):
            def val(stage,key='ms'):return float(g.xs(stage,level='stage')[key].iloc[0])
            bw=back[(back.method==method)&(back.K==K)]
            mem=memory[(memory.scene==scene)&(memory.method==method)].iloc[0]
            rows.append(dict(phase='matched_decode_graph',scope='selected-K NN + original Gaussian decode; Frozen Master',scene=scene,method=method,ratio=ratio,K=K,
                forward_ms=val('forward_decode'),backward_ms=float(bw.backward_median_ms.mean()),forward_backward_ms=val('forward_backward_decode'),
                optimizer_step_ms=val('adam_only'),network_step_ms=val('network_step_decode'),end_to_end_ms=val('renderer_loss_step'),
                forward_p10_ms=val('forward_decode','p10'),forward_p90_ms=val('forward_decode','p90'),fb_p10_ms=val('forward_backward_decode','p10'),fb_p90_ms=val('forward_backward_decode','p90'),
                end_to_end_p10_ms=val('renderer_loss_step','p10'),end_to_end_p90_ms=val('renderer_loss_step','p90'),forward_cpu_submit_ms=val('forward_decode','cpu'),fb_cpu_submit_ms=val('forward_backward_decode','cpu'),
                first_shape_setup_ms=float(g.setup.max()),peak_increment_bytes=int(g.peak.max()),persistent_increment_bytes=int(mem.persistent_increment_bytes),
                numerical_forward_max_abs=float(direct[~direct.backward].max_abs.max()) if method=='decoded_graph' else (float(early[early.method==method].numerical_forward_max_abs.max()) if method!='reference' else 0.),
                numerical_backward_max_abs=float(direct[direct.backward].max_abs.max()) if method=='decoded_graph' else (float(early[early.method==method].numerical_backward_max_abs.max()) if method!='reference' else 0.),
                numerical_error_scope='all two-scene direct decoded gates, including extreme inputs; component CSV retains per-tensor errors',
                validation_status='PASS_DIRECT_INPUT_VJP_AND_COMMON_PHOTO_VJP_ADAM' if method=='decoded_graph' else 'REFERENCE_OR_PREVIOUS_VALIDATED_CONTROL',
                kernel_launch_count=np.nan,kernel_launch_scope='CUPTI packed-graph trace unavailable; captured kernel nodes separately',backward_measurement_scope='direct CUDA Events; forward outside interval'))
    accepted=pd.DataFrame(rows)
    for i,row in accepted.iterrows():
        b=accepted[(accepted.scene==row.scene)&(accepted.ratio==row.ratio)&(accepted.method=='reference')].iloc[0]
        for label,column in [('forward_speedup','forward_ms'),('fb_speedup','forward_backward_ms'),('e2e_speedup','end_to_end_ms')]:accepted.loc[i,label]=b[column]/row[column]
    pd.concat([early,accepted],ignore_index=True).to_csv(ROOT/'CUDA_PERFORMANCE_RESULTS.csv',index=False)
    accepted.to_csv(ROOT/'accepted_decoded_performance.csv',index=False)
    macro=accepted.groupby(['scene','method'])[['forward_ms','backward_ms','forward_backward_ms','network_step_ms','end_to_end_ms']].mean().reset_index()
    macro.to_csv(ROOT/'decoded_performance_summary.csv',index=False)
    pairs=[]
    for scene in ['fr1_xyz','fr2_xyz']:
        a=macro[(macro.scene==scene)&(macro.method=='reference')].iloc[0];b=macro[(macro.scene==scene)&(macro.method=='decoded_graph')].iloc[0]
        pairs.append(dict(scene=scene,forward_reference=a.forward_ms,forward_optimized=b.forward_ms,forward_speedup=a.forward_ms/b.forward_ms,backward_reference=a.backward_ms,backward_optimized=b.backward_ms,
            fb_reference=a.forward_backward_ms,fb_optimized=b.forward_backward_ms,fb_speedup=a.forward_backward_ms/b.forward_backward_ms,e2e_reference=a.end_to_end_ms,e2e_optimized=b.end_to_end_ms,e2e_speedup=a.end_to_end_ms/b.end_to_end_ms))
    training=pd.DataFrame(json.loads((ROOT/'decoded_real_training_summary.json').read_text()));training['total_ms']=training.seconds*1000+training.setup_ms
    quality=pd.read_csv(ROOT/'decoded_real_training_quality.csv').groupby(['method','ratio'])[['psnr','photo']].mean().reset_index()
    original=training[training.method=='reference'].iloc[0];opt=training[training.method=='optimized'].iloc[0]
    breakeven=(opt.setup_ms-original.setup_ms)/(original.ms_per_step-opt.ms_per_step)
    errors=direct.assign(group=direct.tensor.str.split('/').str[0]).groupby(['group','backward']).agg(tensors=('pass_','size'),max_abs=('max_abs','max'),max_relative=('max_relative','max'),max_rms=('rms','max')).reset_index()
    errors.to_csv(ROOT/'decoded_parity_summary.csv',index=False)
    nodes=[]
    for path in sorted(ROOT.glob('*captured_K*.dot')):
        nodes.append(dict(file=path.name,K=int(re.search(r'captured_K(\d+)',path.stem).group(1)),kernel_nodes=path.read_text().count('shape="octagon"')))
    pd.DataFrame(nodes).to_csv(ROOT/'captured_kernel_nodes.csv',index=False)
    original_steps=pd.read_csv(ROOT/'step_parity_tensors.csv')
    non_det=original_steps.groupby('method').agg(max_abs=('max_abs','max'),failed=('pass_',lambda v:int((~v).sum()))).reset_index()
    cfg=json.loads((ROOT/'execution_config.json').read_text());manifests=json.loads((ROOT/'model_manifests.json').read_text())
    assert cfg['render_backends']['optimized']=='decoded_graph'
    report=f'''# SpatialLocalXYZ CUDA性能优化报告

## 1. 结论与版本边界

基线已提交：`74fc1524d8d6345eda8e732a1da2efa9a4934951`（anchor-si）。SpatialLocalXYZ只作为**性能研究默认**；生产配置保持原样。本轮新增优化不stage、不另行commit、不push。

最终推荐冻结地图后端 **decoded_graph**：只读55D输入缓存＋严格FP32 Triton FiLM/输出融合＋原cuBLAS网络与原生Gaussian解码共同CUDA Graph。33,806个共享参数、55D字段、Fourier、Anchor单位XYZ、Quaternion、Loss、Adam、训练顺序和ratio均不变，没有新增Gaussian/Anchor私有训练表。

同K、同Checkpoint、同Adam、交错测量的七档等权结果：

{md(pairs,[('scene','地图'),('forward_reference','原NN+解码 Forward ms'),('forward_optimized','优化Forward ms'),('forward_speedup','Forward倍数'),('fb_reference','原F+B ms'),('fb_optimized','优化F+B ms'),('e2e_reference','原完整步ms'),('e2e_optimized','优化完整步ms'),('e2e_speedup','完整步倍数')])}

以上是Frozen Master真实Renderer+Loss+原Adam步骤，**不是完整在线SLAM加速**，也不是重建质量提升实验。独立Backward计时见下表；没有用F+B减Forward代替Backward。

{md(macro,[('scene','地图'),('method','后端'),('forward_ms','NN+解码Forward'),('backward_ms','Backward-only'),('forward_backward_ms','NN+解码F+B'),('network_step_ms','加Adam'),('end_to_end_ms','加Renderer+Loss')])}

## 2. 审计与原始Profile

原Features在`no_grad`下计算19维物理输入及世界坐标，训练时Gather和Fourier才实时执行。19维为世界XYZ3、asinh局部Offset3、logScale3、SH DC3、Opacity logit1、归一化协方差形状6，经原均值/标准差和clamp。另有六频带36维sin/cos，合计55维。主干55→64→64；ratio log归一化后1→32→256，两层FiLM；四Head64→64→3/3/4/4。XYZ输出乘原Anchor参考尺度。

fr1 N={manifests[0]['N']}、Anchor={manifests[0]['M']}；fr2 N={manifests[1]['N']}、Anchor={manifests[1]['M']}。原选择身份、Cached uniform和same-K排序未修改。源Checkpoint完整路径及SHA见`model_manifests.json`；网络、原Renderer与生产源码SHA分别在environment、production_source_verification等JSON。

RTX3090、Torch2.3.1+cu118、Triton2.3.1、FP32、TF32关闭。无nvcc/nsys/ncu，使用实际Triton JIT、PyTorch Profiler、CUDA Events及CUDA Graph DOT；没有修改系统安装。测试队列顺序运行，正式测量时未发现其他GPU计算作业，未锁时钟。使用用户现有conda环境，无隔离构建。

Profile发现小K的大量极小Kernel、Tensor构造与CPU提交间隙。K68原始整步骤Profiler约250个GPU Kernel，graph_fused约236个；不是GEMM计算量主导。基线独立探针含额外input_pack，432 kernels/4次不能当纯网络Kernel数。`profile_summary_*`和`operators_*`保留CPU/GPU active、分配事件；active时间不等于硬件occupancy。

最终打包Graph的CUPTI Collection后处理在本环境停滞；仅停止本任务挂起进程，保留日志。最终延迟来自CUDA Events。DOT报告如下：

{md([n for n in nodes if n['file'].startswith('decoded_graph')],[('file','原生DOT'),('K','K'),('kernel_nodes','捕获Kernel节点')])}

这些是**图内**前向/反向节点，不包括Renderer、Adam、copy和clone。Graph主要减少主机调度，不能声称将所有GPU工作融合成一个Kernel。

## 3. 有效优化及空间换时间

1. Pack Triton直接写K×55，合并Gather、六频带Fourier与cat；使用libdevice `__nv_sinf/__nv_cosf`，不用近似`tl.sin`，所有内核关闭FP融合。
2. 两层FiLM multiply/add/ReLU融合；Backward保留dH和全部K行对共享ratio参数的sum。四个独立Head权重与cuBLAS矩阵运算保持原样。
3. 四Head拼接及XYZ×AnchorScale融合；尺度梯度保留。
4. 缓存只读确定性N×55输入。fr1约1.435MiB、fr2约1.946MiB；不是可训练身份表。输入需梯度时使用实时Pack，不能detach活跃路径。
5. 每种K/输入storage/梯度模式保留独立Graph及激活池。只共享预热Stream以减少cuBLAS工作区，**不共享不同Graph的激活池**。Torch helper独立副本仅改一行Stream创建，原三次warmup和Autograd不变，安装包未改。
6. 将原生Gaussian解码一起capture：XYZ加法、Scale×exp、Quaternion两次normalize与multiply、Opacity logit+sigmoid、DC加法，逐值与原Runtime一致。减少这些逐元素算子的Python/Autograd调度，Renderer和原Adam仍在图外。

仅缓存55D的Compensation-only后端仍推荐`graph_cached55_fused`。该接口的Forward约0.118/0.122ms、F+B约0.498/0.510ms；它不含Gaussian属性解码，不能直接与上表更大范围的0.176/0.173ms比较。旧阶段报告保存在`CACHE55_STAGE_*`。

## 4. 逐K与测量公平性

fr1三个交错块、fr2两个，每阶段35次＋预热。每一阶段恢复同一Checkpoint和完整Adam状态，Adam-only探针不污染后续Renderer状态；每块改变方法顺序。每行保留块内median/P10/P90，汇总为块间mean。Frozen NN+解码、Compensation-only分不同phase/scope，CSV不混算。

'''
    for scene in ['fr1_xyz','fr2_xyz']:
        data=accepted[(accepted.scene==scene)&accepted.method.isin(['reference','decoded_graph'])]
        wide=data.pivot(index=['ratio','K'],columns='method',values=['forward_ms','backward_ms','forward_backward_ms','end_to_end_ms']);wide.columns=['_'.join(v) for v in wide.columns];wide=wide.reset_index()
        report+=f'### {scene}\n\n'+md(wide,[('ratio','ratio'),('K','K'),('forward_ms_reference','原Forward'),('forward_ms_decoded_graph','优化Forward'),('backward_ms_reference','原Backward'),('backward_ms_decoded_graph','优化Backward'),('end_to_end_ms_reference','原完整步'),('end_to_end_ms_decoded_graph','优化完整步')])+'\n\n'
    report+=f'''## 5. 数值、梯度与真实续训

解码Graph两地图共{len(direct)}项直接/输入VJP/边界/生命周期比较，全部通过原阈值；两地图12步跨ratio共{len(steps)}项参数、step、exp_avg、exp_avg_sq比较，最大绝对误差 **{steps.max_abs.max():.9g}**。另有Compensation-only 655项及1176项状态检查、配置默认入口14项。所有逐张量误差保留，无放宽阈值。

12步状态比较使用**同一真实Renderer photo的解码属性VJP**隔离网络实现与Renderer非确定性，既检查NN又检查解码Backward。输入VJP覆盖19D/coords/Anchor尺度以及XYZ、Scale、Rotation、DC、Opacity原属性leaf；既有缓存Graph梯度路径，也有输入需梯度时的实时fallback。无梯度的高阶SH不伪造活跃依赖。

原独立Renderer反传与自身重复也会越过初始严格轨迹阈值，失败没有被改为PASS；详见数值报告。因此完整真实256步用于确认可训练、时钟与轨迹波动，不能要求长程bitwise相同。

三组均从原12000步模型及完整Adam恢复，原Bank后续64轮×4ratio=256真实photo步，lr=.001、eps=1e-15、fused=True、capturable不变。每个参数Adam时钟都到12256，Master未更新。

{md(training,[('method','方法'),('ms_per_step','稳态ms/step'),('setup_ms','首次4shape初始化ms'),('total_ms','含初始化总ms'),('adam_clock','Adam时钟')])}

{md(quality,[('method','方法'),('ratio','ratio'),('psnr','Bank留出PSNR'),('photo','photo')])}

这些评价来自原Bank留出，不是全新独立场景质量认证；未重新做PSNR模型选择。数值验收才是等价依据，不以轻微PSNR差值代替。

## 6. 初始化与显存

真实续训稳态{original.ms_per_step/opt.ms_per_step:.3f}倍；4训练shape初始化约{opt.setup_ms/1000:.3f}s。当前短256步含初始化后优化仍较慢，约{breakeven:.0f}步可摊销额外初始化。此估算不包括反复拓扑/K变化的重捕获。

以下是每种后端、每张地图**独立新进程冷测**，相同NN+解码范围，含网络、原Adam、七shape及最后反传buffer，不含共同Fixture/完整地图：

{md(memory,[('scene','地图'),('method','后端'),('persistent_MiB','常驻增量MiB'),('graphs','图数'),('parameter_bytes','参数bytes'),('optimizer_moment_bytes','Adam双Moment bytes')])}

CSV另有每步临时峰值，不是GPU整进程/整地图峰值。参数135224 bytes、Adam双Moment270448 bytes不变；空间增长来自缓存、工作区、Graph静态激活和缓冲。

## 7. 失败、限制与剩余热点

- 单独eager融合不快，CPU launch抵消局部Kernel收益；Graph后才获明显收益。相同Core的Inductor K68初步通过，但冷编译约5.9秒且稳态略慢，未推荐；没有无目标重写cuBLAS GEMM。
- 初版Graph让静态IDs/ratio别名调用方缓存，跨ratio第五步失败336项；修复为私有clone后完整重验通过，原失败数据保留。
- 首次capture不能在同一组参数有未完成Autograd图时执行。接口探针曾违反此约束，日志保留；改在安全初始化边界/no_grad预热后通过。不是通过detach实际训练梯度规避。
- Torch helper三次预热保持；不采用上游版本中失败的zero-warmup。独立helper依赖Torch版本，升级需重验。
- 完整Renderer+Adam capture在原Adam capturable=False处失败，未修改Optimizer。不能据此断言Renderer本身不可捕获。源码另有num_rendered、bucket_sum的Device-to-Host计数和动态buffer；本轮未复用旧计数、截断tile或改变Renderer工作量。
- 剩余约1.66ms含Renderer、Loss、原Adam和图外调度。下一步若继续降延迟，应独立研究Renderer动态计数与buffer；需要新的严格等价工程，不应继续为很小的神经GEMM付出过高维护成本。

## 8. 在线适用性与交付状态

**IMPLEMENTED / TESTED：** 两地图七档K、Frozen Master、共享神经网络训练、原生解码Graph、全部网络与输入leaf VJP、真实256步、缓存/Graph失效与边界。

**PARTIALLY VERIFIED：** 输入leaf与bases梯度正确，但原Features生产者本来在no_grad下，raw Master→19D全链路及动态人口归一化不是本轮实现。合成append/prune/reorder不等于生产SLAM生命周期集成。

**NOT IMPLEMENTED：** 生产C++/在线SLAM接入、实时Master更新、闭环及Gaussian优化器迁移、ATE、动态shape重捕获策略、并发Forward/乱序Backward、inference_mode全部组合。不能将固定地图倍数推广为Mapping总收益。

缓存记录父表data_ptr、shape和PyTorch版本，原地PyTorch写入重建55D；外部CUDA写入必须invalidate。bases原地更新由图重放读取当前storage；更换storage/N/K重建。调用方必须同步更新19D生产者、Anchor尺度和render_bases，图不会自动替用户重新推导原19D统计。参数storage/requires_grad变化、append/prune/reorder后应在无未完成反传边界invalidate并释放旧图；普通原地Adam更新无需重建。

保持reference / optimized / experimental三路径；仅性能研究默认启用。生产源码SHA与研究开始前历史快照匹配，本地tracked生产无改动。全部源码、配置、指标、失败日志、代码版本、trace/DOT及续训模型随归档交付。
'''
    (ROOT/'SPATIAL_LOCALXYZ_CUDA_OPTIMIZATION_REPORT.md').write_text(report,encoding='utf-8')
    parity=f'''# CUDA数值与训练状态等价报告

## 1. 验收约束

FP32、TF32=False，无fast math、无近似梯度、无删减曝光。Forward atol=1e-6/rtol=1e-5，Backward atol=1e-5/rtol=1e-4；逐元素abs差≤atol+rtol×abs(reference)，未放宽。relative分母clamp1e-12，同时报告max abs与RMS。

## 2. 解码Graph逐组实际误差

{md(errors,[('group','组'),('backward','Backward'),('tensors','张量数'),('max_abs','max abs'),('max_relative','max relative'),('max_rms','最大RMS')])}

共{len(direct)}项，详细数据`decoded_direct_fr1_xyz.csv`、`decoded_direct_fr2_xyz.csv`。包括七档训练/未见ratio、输出14D和所有解码属性、原Renderer、photo、全部共享参数梯度，输入leaf实时及缓存Graph路径，K0/1/17、重复IDs、Scale1e-6~100、世界坐标1000量级、opacity logit±20。无把巨大合成点云渲染冒充质量测试。

19D输入→55D、Encoder、FiLM、四Head内部还沿用原已验收Core；两个地图与边界655项在`parity_graph_cached55_fused*.csv`和`graph_cached55_fused_boundary_gate.csv`，逐组误差另见CACHE55_STAGE数值报告。默认配置入口14项在configured_default_parity.csv。

## 3. 单步/跨ratio原Adam

每地图12步，原12000状态及原Adam：1%/5%/10%/15%轮转。解码Graph共{len(steps)}项参数、step、exp_avg、exp_avg_sq，max abs={steps.max_abs.max():.9g}；下一步7.5%输出6张量在direct文件记录。使用同一真实photo对解码属性的VJP隔离Renderer非确定性；无改Loss或Optimizer。

## 4. 独立Renderer反传的失败如实保留

{md(non_det,[('method','方法'),('max_abs','max abs'),('failed','超过初始严格轨迹阈值项数')])}

原实现重复也失败，未将其改PASS。原失败step_parity_tensors.csv与初版输入别名bug失败common_vjp_tensors.csv均保留。后者已定位修复，不混入验收；独立Renderer长程不保证bitwise轨迹。真实256步见decoded_real_training_*，不以最终PSNR接近代替梯度/Adam证明。

## 5. 失效与使用边界

同K不同ratio/IDs、返回Tensor保留、物理表及bases原地修改、append、prune/reorder、invalidate全部逐张量检查。原Features no_grad行为未变：没有声称raw Master Joint或生产几何生命周期已集成。首捕获应在安全初始化边界，不能保留同参数未消费的Autograd图。普通调用按Forward→Backward顺序消费；不支持同图积压多次Forward后乱序反传。
'''
    (ROOT/'CUDA_PARITY_REPORT.md').write_text(parity,encoding='utf-8')
    implementation='''# 优化实现与复现

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
'''
    (ROOT/'CUDA_OPTIMIZED_IMPLEMENTATION.md').write_text(implementation,encoding='utf-8')
    summary=dict(best_render_backend='decoded_graph',best_compensation_only_backend='graph_cached55_fused',baseline_commit='74fc1524d8d6345eda8e732a1da2efa9a4934951',performance=pairs,direct_tensors=len(direct),direct_pass=True,step_tensors=len(steps),step_max_abs=float(steps.max_abs.max()),estimated_amortization_steps=breakeven,production_changed=False,optimization_committed=False)
    (ROOT/'optimization_summary.json').write_text(json.dumps(summary,indent=2));print('FINAL_SUMMARY',json.dumps(summary),flush=True)


if __name__=='__main__':main()
