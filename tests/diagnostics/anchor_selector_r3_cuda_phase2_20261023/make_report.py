"""从最终证据生成中文报告；不把理论潜力或共享GPU单次结果写成已实现收益。"""
from pathlib import Path
import json,shutil
import numpy as np
import pandas as pd
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];E=HERE/'evidence_final'
def read(name):return pd.read_csv(E/name,low_memory=False)
def md(rows,columns):
    return '| '+' | '.join(columns)+' |\n| '+' | '.join(['---']*len(columns))+' |\n'+''.join('| '+' | '.join(str(row.get(k,'')) for k in columns)+' |\n' for row in rows)
def source(file,needle):
    lines=(ROOT/file).read_text(encoding='utf-8').splitlines();n=next(i+1 for i,s in enumerate(lines) if needle in s)
    return f'[{file}:{n}](<{(ROOT/file).as_posix()}:{n}>)'
perf=read('phase2_performance_summary.csv');lat=read('phase2_online_tail_performance.csv');launch=read('phase2_kernel_launch_comparison.csv');parity=read('phase2_correctness_parity.csv');quality=read('fixed_quality_metrics.csv');modules=read('module_paired_summary.csv');memory=read('memory_measurements.csv');online=read('actual_online_mapping.csv')
for name in ['phase2_performance_summary.csv','phase2_kernel_launch_comparison.csv','phase2_correctness_parity.csv','phase2_cpu_gpu_sync.csv','phase2_topology_event_benchmark.csv','phase2_online_tail_performance.csv']:shutil.copy2(E/name,ROOT/name)
# Normalize mixed bool/0/1 CSV rows so machine readers do not treat "False" as true.
for col in ['exact','passed','shape_match','grad_none_match']:
    if col in parity:parity[col]=parity[col].map(lambda x:np.nan if pd.isna(x) else int(str(x).lower() in ['true','1','1.0']))
threshold=read('threshold_after.csv');initial_threshold=read('threshold_before.csv')
for i,r in threshold.iterrows():
    parity.loc[len(parity),['comparison','tensor','exact','bad_elements','max_abs','baseline_self_differences']]=['artificial_ulp_boundary',f'trial_{i}',1,r.fused_mask_differences,r.max_opacity_abs,r.legacy_repeat_mask_differences]
parity.to_csv(ROOT/'phase2_correctness_parity.csv',index=False)
fixed=lat[lat.scope.str.startswith('fixed')];L=fixed.pivot_table(index=['resolution','case'],columns='variant',values='p50')
latrows=[]
for (resolution,case),r in L.iterrows():latrows.append(dict(范围=('Full 原分辨率' if case=='Full' else ('Online半分辨率' if resolution==.5 else 'Tail原分辨率')),档位=case,**{v:f'{r[v]:.3f}' for v in ['A','B','C','D','E_workspace','E']},相对B=f'{100*(1-r.E/r.B):+.1f}%'))
alllaunch=launch[launch.stage=='all'].pivot(index='case',columns='variant',values='launches');launchrows=[]
for case,r in alllaunch.iterrows():launchrows.append(dict(档位=case,**{v:int(r[v]) for v in ['A','B','C','D','E']},本轮减少=int(r.B-r.E)))
def mod(module,stage=None,**filters):
    a=modules[modules.module==module]
    if stage:a=a[a.stage==stage]
    for k,v in filters.items():a=a[a[k]==v]
    return a
def stage(variant,case,stage_,resolution=1.):
    a=perf[(perf.variant==variant)&(perf.case==case)&(perf.stage==stage_)&(perf.resolution==resolution)&(perf.metric=='cuda_elapsed_ms')];return float(a.p50.mean())
def change(a,b):return f'{a:.3f} → {b:.3f} ms（减少 {100*(1-b/a):.1f}%）'
sync=read('phase2_cpu_gpu_sync.csv');syncrows=[]
for v in ['B','C','D','E']:
 for case,st in [('Full','full/anchor_statistics'),('lod_0.050000','sample_budget_compact'),('lod_0.050000','renderer_forward')]:
    a=sync[(sync.variant==v)&(sync.case==case)&(sync.stage==st)&sync.api.str.contains('Memcpy|Synchronize')]
    for _,r in a.iterrows():syncrows.append(dict(版本=v,阶段=st,API=r.api,每步次数=r.calls_per_step))
quality_cols=['psnr','ssim','lpips','photo','l1','loss_ssim','budget'];qbase=quality[quality.variant=='B'].set_index(['resolution','case']);qdiff=[]
krows=[dict(档位=r['case'],N=int(r['N']),K=int(r['K']),radii正值=int(r['visible'])) for r in quality[(quality.variant=='B')&(quality.resolution==1)].to_dict('records')]
for v in ['A','C','D','E_workspace','E']:
    q=quality[quality.variant==v].set_index(['resolution','case']);qdiff.append(dict(版本=v,**{k:f'{(q[k]-qbase[k]).abs().max():.3g}' for k in quality_cols}))
normal=parity[parity.comparison.str.match(r'^(0.5|1.0)/')];fixed_bad=normal[normal.bad_elements.fillna(0)>0]
exact_bad=normal[(normal.exact==1)&(normal.bad_elements.fillna(0)>0)]
topo_parity=parity[parity.comparison.str.startswith('topology/')]
curve=read('continuous_training_curves.csv');last=curve.groupby(['run','kind']).tail(1);curverows=[]
for run,g in last.groupby('run'):
    curverows.append(dict(运行=run,Full_photo=f'{g[g.kind=="Full"].photo.iloc[0]:.8f}',LoD_photo=f'{g[g.kind=="LoD"].photo.iloc[0]:.8f}'))
state=json.loads((E/'online_final_states.json').read_text());onlinerows=[]
for s in state:
    a=s['anchor'];m=s['mapping'];onlinerows.append(dict(版本=s['variant'],Gaussian=a['gaussian_count'],Anchor=a['anchor_count'],Online_steps=m['online']['mappings'],Tail_steps=m['tail']['mappings'],Online_topology=m['online']['topology'],Tail_topology=m['tail']['topology'],R3_Adam_steps='/'.join(str(x['adam_step']) for x in a['private_compensation_state'])))
online_latency=online[(online.stage=='total')&online.phase.isin(['online','tail'])];onlat=[]
for _,r in online_latency.iterrows():onlat.append(dict(版本=r.variant,阶段=r.phase,样本=int(r.samples),mean=f'{r["mean"]:.3f}',p50=f'{r.p50:.3f}',p95=f'{r.p95:.3f}',p99=f'{r.p99:.3f}'))
memrows=[]
for case in ['Full','lod_0.050000']:
    g=memory[(memory.resolution==1)&memory.case.eq(case)]
    for v in ['B','C','D','E_workspace','E']:
        r=g[g.variant==v].median(numeric_only=True);memrows.append(dict(阶段=case,版本=v,allocated_MiB=f'{r.allocated_bytes/2**20:.2f}',peak_MiB=f'{r.peak_allocated_bytes/2**20:.2f}',Tensor分配请求=int(r.allocation_requests),设备分配=int(r.device_allocations)))
topo=read('phase2_topology_event_benchmark.csv');extra=[]
for path,label in [(E/'topology_repeat/topology_events.csv','repeat'),(E/'production_phase4/anchor_timing.csv','anchor_primitive')]:
    if not path.exists():continue
    a=pd.read_csv(path)
    if label=='repeat':
        for (scenario,fused),g in a.groupby(['scenario','fused']):extra.append(dict(variant='fused' if fused else 'legacy',round=1,operation=scenario,scope='real_checkpoint_synthetic_event_gradient',N_before=g.N_before.iloc[0],A_before=g.A_before.iloc[0],N_after=g.N_after.iloc[0],A_after=g.A_after.iloc[0],samples=len(g),mean=g.cuda_ms.mean(),p50=g.cuda_ms.median(),p95=g.cuda_ms.quantile(.95),p99=g.cuda_ms.quantile(.99)))
    else:
        for (implementation,stage_),g in a[a.stage.str.startswith('topology_')].groupby(['implementation','stage']):extra.append(dict(variant=implementation,round=0,operation=stage_,scope='real_anchor_primitive',samples=len(g),mean=g.cuda_ms.mean(),p50=g.cuda_ms.median(),p95=g.cuda_ms.quantile(.95),p99=g.cuda_ms.quantile(.99)))
topo=pd.concat([topo,pd.DataFrame(extra)],ignore_index=True);topo.to_csv(ROOT/'phase2_topology_event_benchmark.csv',index=False)
toporows=[]
for _,r in topo[topo.scope.eq('real_checkpoint_synthetic_event_gradient')].iterrows():toporows.append(dict(运行=int(r['round']),事件=r.operation,版本=r.variant,N=f'{int(r.N_before)}→{int(r.N_after)}',A=f'{int(r.A_before)}→{int(r.A_after)}',p50=f'{r.p50:.3f}',p95=f'{r.p95:.3f}'))
scalerows=[]
for file,groups,tag in [('anchor_scaling.csv',['N','fused','stage'],'Anchor人工满10成员'),('production_phase5/sampling_scaling.csv',['N','fused','stage'],'Selector人工概率')]:
    a=read(file)
    for keys,g in a.groupby(groups):scalerows.append(dict(范围=tag,N=keys[0],CUDA=keys[1],阶段=keys[2],p50=f'{g.cuda_ms.median():.4f}',p95=f'{g.cuda_ms.quantile(.95):.4f}'))
scale=pd.DataFrame(scalerows);scale.to_csv(HERE/'scaling_summary.csv',index=False)
ssimrows=[]
for (h,w),g in modules[modules.module=='production_ssim'].groupby(['H','W']):
    a=g[g.optimized==0].p50.iloc[0];b=g[g.optimized==1].p50.iloc[0];ssimrows.append(dict(尺寸=f'{int(w)}×{int(h)}',Legacy=f'{a:.4f}',CUDA=f'{b:.4f}',减少=f'{100*(1-b/a):.1f}%'))
statslaunch=launch[(launch.case=='Full')&launch.stage.eq('full/anchor_statistics')].set_index('variant').launches
e_gpu=launch[(launch.variant=='E')&launch.case.eq('lod_0.050000')].set_index('stage')
gpu_total=e_gpu.loc['all','kernel_ms'];back=e_gpu.loc['backward','kernel_ms'];render=e_gpu.loc['renderer_forward','kernel_ms']
remaining=f'''# 第二阶段之后的剩余热点

以下是实测归属开销，不是承诺可全部回收的加速空间。RTX3090 由其他任务共享；当前全部新开关默认关闭。

1. **P0：SSIM / Renderer Backward。** 5% LoD Nsight 中，Backward 为 {back:.3f} ms，占所计 kernel 时间 {back/gpu_total:.1%}；Renderer Forward 为 {render:.3f} ms。先研究 SSIM 前向统计复用与 Renderer Backward 的真实热点，保留 11×11、原公式和VJP。NCU仍无计数器权限，不能断言属于DRAM瓶颈。
2. **P1：Renderer动态计数与CPU边界。** `num_rendered` 控制CUB sort长度与binning容量，`bucket_sum`控制sample容量；另有radii的nonzero用于监督索引。这些读取有真实消费者。下一步需设备计数接口和容量策略的独立设计；不能删除同步或以padding Gaussian代替真实K。
3. **P1：Topology批量迁移。** 可研究同一survivor permutation同时迁移Master/R3/Adam，必须保留叶子替换、grad=None、别名与clock。emptyCache与驱动释放开销在共享GPU上波动很大，不能以单轮事件结果宣称稳定加速。
4. **P1：MLP输入与Encoder前向融合微基准。** 先比较真实N/K下cuBLAS和自定义融合；不自动重写GEMM、不启用不同精度。Budget继续原ATen归约，不为多省几个launch改变零点附近的符号。
5. **P2：额外Anchor哈希/R3 lookup/workspace复杂化。** Full统计已减少到 {int(statslaunch['E'])} 次launch/1次错误回读，R3仍只gather K行；GPU hash低频且涉及FP64负坐标量化。Workspace只少4个分配请求，steady-state物理设备分配原本就是0，常驻显存反而增加。

边界：人工ULP prune probe连Legacy自身也发生翻转。最终总opacity仍调用原FP64 index_add_；不得以调阈值、扩大容差或取消检查宣称全域exact。长期连续训练存在Renderer atomic和tiny-gradient Adam敏感性，严格差异文件保留，不能改epsilon以通过Gate。
'''
(ROOT/'phase2_remaining_hotspots.md').write_text(remaining,encoding='utf-8')
remaining_report=remaining.replace('# 第二阶段之后的剩余热点','### 剩余热点优先级',1)
gates={
 'anchor_validation':dict(operator_parity='pass',normal_state_parity='pass',default_enabled=False),
 'anchor_statistics':dict(normal_windows='pass',ulp_prune_exact='not_certified; legacy is also nondeterministic',original_fp64_reduction_preserved=True,default_enabled=False),
 'sampling_compaction':dict(forward_ids_K_ST_budget='exact',default_enabled=False),
 'renderer_workspace':dict(normal_vjp_saved_context='pass',arbitrary_vjp_stress='not_certified',performance='no_stable_gain',default_enabled=False),
 'ssim_backward':dict(operator_vjp='bitwise_equal_on_five_sizes',performance='module_gain',default_enabled=False),
 'continuous_training':dict(long_term_elementwise_parity='not_passed; baseline also drifts',default_deployment=False)}
(HERE/'phase2_gate_status.json').write_text(json.dumps(gates,ensure_ascii=False,indent=2),encoding='utf-8')
text=f'''# mycartgs 第二阶段 CUDA 性能优化报告

日期：2026-10-23。源 HEAD：`4269d7e3bad6cf5af7e1df1763634b61e98e0b51`；实际运行包含上轮及本轮未提交修改，以[源码manifest](<{(HERE/'final_source_manifest.json').as_posix()}>)和[服务器SHA核对](<{(E/'source_verification.json').as_posix()}>)为准。没有 stage、commit、push。

## 1. 结论与验收边界

本轮实现了 Anchor 校验/窗口统计融合、Cached Bernoulli ST 与稳定压缩、可安全保留Backward上下文的workspace池、SSIM通道分块Backward，以及两个范围严格的Topology快路。生产C++/CUDA编译成功，保留所有Legacy分支和上一轮三个开关。

- Full阶段Anchor统计：**{int(statslaunch['B'])} → {int(statslaunch['E'])} 次CUDA launch**；错误检查回读从8次变为1次。
- Selector Sampling/Budget/Compaction：18 → 7次launch。完整LoD从上轮177 → 166；完全Legacy为418。
- 固定地图、相机、Cached uniform与K下，六个实现的PSNR/SSIM/LPIPS/photo一致；原生身份、选中集合、窗口访问计数和正常prune mask exact。
- **不宣称所有正确性Gate通过。** FP64人工ULP阈值存在Legacy自身不确定性；连续训练中的少量初步梯度差异会被eps=1e-15的Adam放大。差异未抹去、未扩大统一容差。Workspace没有稳定端到端收益。
- **全部七个开关仍默认0。** 可选实现保留用于复现，不宣称已完成生产默认部署或独占GPU下稳定加速认证。

## 2. 真实架构与运行配置

同一fr1_xyz源checkpoint：15,891 Gaussian / 6,121 Anchor，Variable Anchor 1～10成员，L0/L1/L2，Feature16、Input27、Encoder27→32、Selector32→32→2、原补偿头32→32→14。

实际K与可见性如下；radii正值表示preprocess可见/有效半径，不冒充逐像素贡献数量：

{md(krows,['档位','N','K','radii正值'])}

R3是生产C++/LibTorch路径，四张独立[N,14]表：`delta=stopgrad(C(h))+residual_ratio[ids]`。C冻结/detach；R3维持Dense Adam，其他ratio的grad=None与独立step不变，未选行的历史moment仍更新。训练ratio仍1%/5%/10%/15%；未见ratio维持原始raw-ratio线性插值与端点。TF、ChangeDistill、SI/GI状态未改。本轮不增加Loss或网络。

Online LoD仍半分辨率，Tail仍原分辨率；自然停止、Full/LoD实际顺序和Mapping预算配置保持原样。没有把Mapping强改为10000。

## 3. 实施位置与接口

| 模块 | 实际源码 | 计算与反向 |
| --- | --- | --- |
| Anchor校验 | {source('src/anchor_stats_cuda.cu','torch::Tensor validateAnchors')} | 计数/非法owner标记/成员排序/容量、层级、finite、scale检查；6个错误类别统一回读48字节，报告最小出错row或Anchor |
| Full统计 | {source('src/anchor_stats_cuda.cu','void anchorFullStats')} | visibility按owner置位，每Anchor访问只加1；成员slot统计FP64；恢复逻辑保持原float cutoff；总opacity保留原FP64 index_add_ |
| 生产统计入口 | {source('src/gaussian_model.cpp','void GaussianModel::addAnchorPruningStats')} | no-grad路径，不新增反向或Loss；仍调用assertBlocks |
| Owner检查入口 | {source('improvements/selector/explicit_anchor_blocks.h','inline void AnchorBankImpl::assertBlocks')} | 不支持的dtype/layout维持Legacy；没有降低频率 |
| Sampling/压缩 | {source('src/selector_sampling_cuda.cu','selectorSamplingForward')} | 保留Softmax，支持stride=2；CUB Flagged稳定压缩；逻辑shape为真实K，容量N，不向Renderer padding |
| ST VJP | {source('src/selector_sampling_cuda.cpp','class SampleCompact')} | `dp=(grad_mask+grad_probability)*eligible`；uniform、bool gate不产生梯度；Budget仍原sum/abs/div |
| 生产LoD入口 | {source('src/gaussian_mapper.cpp','torch::Tensor selector_mask, fused_selected_ids')} | 与原Gaussian gather、R3、14D decode衔接；不改参数注册与梯度路由 |
| Workspace池 | {source('cuda_rasterizer/rasterize_points.cu','std::array<torch::Tensor,4> acquire')} | 每线程最多4组；只在Tensor/Storage均无外部持有且default stream时复用；不足容量扩容，不覆盖live context |
| Saved State | {source('src/gaussian_rasterizer.cpp','ctx->save_for_backward')} | 各次Forward的geom/binning/image/sample仍由对应Autograd context持有 |
| SSIM通道并行 | {source('cuda_rasterizer/conv.cu','template <int CH, bool SplitChannels')} | 原32×32 tile、halo、11×11权重与计算顺序；grid.z按通道独立写回，无新atomic |
| SSIM线程路由 | {source('include/loss_utils.h','ctx->saved_data["optimized_backward"]')} | Forward保存后端选择，Backward在LibTorch工作线程中按context恢复；不能只依赖Mapping线程TLS |
| Topology无候选快路 | {source('src/gaussian_model.cpp','The legacy transaction returns')} | 仅在无有限梯度达阈值时跳过4个人口拷贝与CPU排序；不跳最终检查/窗口清理/拓扑时钟 |
| 部分prune快路 | {source('improvements/selector/variable_anchor_blocks.h','previous_anchor_count')} | Anchor行/key/level均没变才跳空间桶重建；所有Gaussian/R3/moment迁移仍执行 |

### Tensor契约

- owner `[N] int64 CUDA`；member_counts/level `[A] int64`；center `[A,3] FP32`；参考尺度`[A,1] FP32`；成员工作表`[A,10] int64`。
- Anchor O为`[A] FP64`、访问次数`[A] int64`；slot O为`[N] FP64`、两个低opacity计数为`[N] int64`。统计分母仍是Anchor可见访问次数，不把未观察当作opacity=0。
- probability/uniform为`[N] FP32`，probability常见stride=2；eligible与flags为`[N] bool`；输出IDs为`[K] int64`连续view，底层容量N。u==p不选择。
- 不对Offset做clamp，不跨训练step缓存可变XYZ、Feature或属性。R3/UID、Feature维度与checkpoint schema未改。

## 4. CPU/GPU执行依赖与同步

```mermaid
flowchart TD
  CPU["CPU Mapping / mutex_render"] --> ACT["GPU Master activation"]
  ACT --> FULL["Full Renderer"]
  FULL --> VAL["GPU批量Anchor检查"]
  VAL --> ERR["CPU一次错误flag回读"]
  ERR --> STAT["GPU窗口统计 + 原FP64 index_add"]
  ACT --> PACK["Input27 → Encoder → 原Softmax"]
  PACK --> SAMPLE["Cached Bernoulli ST + CUB stable select"]
  SAMPLE --> K["CPU读取一次真实K"]
  K --> GATHER["K行gather / R3 / Decode"]
  GATHER --> RF["Renderer preprocess + scan"]
  RF --> RCOUNT["CPU num_rendered → sort长度/容量"]
  RCOUNT --> SORT["GPU sort / bucket scan"]
  SORT --> BCOUNT["CPU bucket_sum → sample容量"]
  BCOUNT --> RGB["GPU RGB / radii /监督可见索引"]
  RGB --> LOSS["原L1 + SSIM + Budget"]
  STAT --> LOSS
  LOSS --> BW["Renderer/MLP/R3 backward；SSIM按通道调度"]
  BW --> ADAM["原独立时钟Dense Adam"]
  ADAM --> TOPO["必要时Topology / UID与moments迁移"]
```

Sampling仍有一次动态K同步。Renderer仍有真实依赖的instance/bucket计数，以及radii nonzero产生的监督索引；后者被updateElasticSupervision消费，不能直接删掉。生产还保留finite-loss等检查。Native重放不包含全部Mapper控制/日志，因此不能把其core latency直接等同完整在线Mapping。

未引入多ratio并发Backward、Full/LoD并行step、异步拓扑写或跨streamRenderer；mutex_render和既有同步没有删除。安全并行的实际实现是SSIM通道的独立block调度。

## 5. 性能协议

RTX3090 24GB、CUDA11.8、LibTorch2.3.1 ABI1、g++9、sm86。Python指标环境独立ABI0。新Anchor/Selector CUDA沿用--fmad=false；SSIM/Rasterizer沿用原编译选项，不静默改变精度。

| 版本 | 开启内容 |
| --- | --- |
| A | 完全Legacy |
| B | 上轮Adam、Decode、Input27 |
| C | B + Anchor统计/校验/安全Topology快路 |
| D | C + Sampling/ST/Stable Compaction |
| E_workspace | D + workspace池 |
| E | E_workspace + SSIM通道Backward |

每种分辨率两轮，第二轮顺序反转；每case 10次warmup、60次replay，rep0用于完整fixture，剩余59次/轮进入计时。下面表格是**两轮p50的均值**，单位ms；逐轮mean/p50/p95/p99与CPU提交/CUDA Event分开保留在CSV。

GPU存在其他任务，采用成对交替而非杀进程；负载、功率、显存记录见gpu_load.csv。Nsight只用于launch、kernel和API分析，独立于正式计时。共享GPU与非独占驱动分配使某些档位/Topology结果波动，不能承诺统一倍率。

预处理边界：瞬时半分辨率微基准使用uint8 OpenCV resize，再转FP32；连续半分辨率重放与真实在线使用FP32 bilinear缓存。候选之间输入一致，但这两套协议的绝对质量不能混用；原生640×480 fixture不受该量化差异影响。没有将微基准当作完整Mapper或改变真实在线GT路径。

## 6. 固定checkpoint的核心Mapping时延

![固定checkpoint交替计时](<{(HERE/'phase2_latency_summary.png').as_posix()}>)

图中误差线仅表示两次运行p50的范围，不是置信区间；原始逐轮p95/p99另见CSV。

{md(latrows,['范围','档位','A','B','C','D','E_workspace','E','相对B'])}

Full半分辨率只是额外诊断，不当作真实Full训练路径列入上表。Online的Full实际仍原分辨率，LoD半分辨率。

### Kernel launch

{md(launchrows,['档位','A','B','C','D','E','本轮减少'])}

原生Full统计包含真正的GaussianModel调用，最终是{int(statslaunch['B'])}→{int(statslaunch['E'])}，不是将独立微基准估算成在线次数。R3 lookup、MLP结构和上一轮kernel未变；LoD减少的11个launch来自sampling/compaction。Workspace和SSIM调度不减少launch数量。

### 必要同步

{md(syncrows,['版本','阶段','API','每步次数'])}

API耗时另见CSV：Memcpy/同步等待可能包含之前排队的GPU工作，不能把API等待、queue与kernel时间简单相加。没有声称消除了所有GPU→CPU同步。

## 7. Anchor统计、pruning与Topology验证

独立1000步窗口覆盖重复visible IDs、空可见集合、1～10成员、opacity变化、slot恢复/失活、所有原始检查类别；正常窗口prune mask exact。真实checkpoint受控事件的owner、UID、counts、level、key、R3表及两套Adam状态逐项一致，topology fixture超阈值元素总计{int(topo_parity.bad_elements.fillna(0).sum())}。

**严格阈值补测的结论不能省略：** 最初确定性逐Anchor求和，在80轮ULP probe中有{int(initial_threshold.fused_mask_differences.sum())}次prune差异，同批Legacy自身有{int(initial_threshold.legacy_repeat_mask_differences.sum())}次。最终保留原FP64 index_add_之后，probe仍有{int(threshold.fused_mask_differences.sum())}次跨运行差异，Legacy自身有{int(threshold.legacy_repeat_mask_differences.sum())}次，均属于人工设置的严格阈值邻接。CSV逐例保留。**原子加法本身不确定，不能承诺任意ULP边界的run-to-run prune exact；没有调阈值、改删除规则或放宽FP64容差。** 因此这项全域exact Gate不标绿，默认仍关闭。

最终没有融合决定pruning的总opacity归约，避免主动引入另一种求和顺序；只融合观测/slot处理与校验。初版结果、失败probe和最终结果分别保存，不择优丢弃。

### Topology时延

{md(toporows,['运行','事件','版本','N','A','p50','p95'])}

这是实际checkpoint上的**人工梯度/窗口/birth clock事件**，不是在线自然事件分布。若存在重复运行，其差异全部保留。无候选快路稳定减少拷贝/排序；其它事件受emptyCache/驱动释放与外部负载影响，不能只取有利的一轮。子阶段NVTX与cudaFree计数见topology_substage_profile.csv；嵌套inclusive范围不能相加当成总耗时。

没有给整个prunePoints增加mask全false就return：该函数仍替换叶子、迁移优化器别名并清理旧grad；直接返回会改变现有训练行为。没有重写空间Hash或改变CPU stable_sort/负坐标量化。

## 8. Selector与R3正确性

Cached Bernoulli仍严格u<p。Inactive gate、全人口Budget分母、原Softmax、零budget邻接与stable row顺序不变。独立318项Forward/ST/Budget/IDs测试误差为0，覆盖stride2、K0/1/N、u==p、cutoff相等、100K规模。当前ratio R3梯度、其他ratio grad=None、UID与leaf clock在原生fixture中保持相同。

固定梯度Dense Adam重复验证13640项记录max error=0，包括tiny gradient、旧moment、独立LR/step和fallback；本轮未修改Adam。

## 9. Workspace与SSIM

Workspace按线程/device管理，default stream才允许命中。引用计数同时检查TensorImpl与Storage，四组上限，所有live Forward/retained graph继续持有自己的byte Tensor。第二次Forward不会覆盖第一次Backward上下文；覆盖K0/1、Full、四档LoD、importance输出和反向。

正常photo量级VJP通过；大幅随机VJP压力测试有少量超过三次Legacy逐元素包络的结果，原始失败文件保留，不宣称任意反向探针均严格通过。Workspace端到端收益弱，内存成本明显，不建议默认使用。

{md(memrows,['阶段','版本','allocated_MiB','peak_MiB','Tensor分配请求','设备分配'])}

steady-state物理设备分配原本就是0，workspace主要少4个Tensor/Storage分配请求，不能说避免了4次cudaMalloc。容量只增长，活跃context多时缓存显存上升；clearIdle只清理无持有项。

SSIM原loss保持`0.8*L1+0.2*(1-fast_ssim)`。5种尺寸的独立VJP逐元素误差0，Autograd context保存后端选择测试通过；Nsight确认优化版本实际执行bool=1实例。寄存器46→40，shared memory14112B不变，每SM resident block仍1；不能据此声称occupancy提高。

{md(ssimrows,['尺寸','Legacy','CUDA','减少'])}

Nsight Compute实际返回ERR_NVGPUCTRPERM，保留日志。没有硬件计数器依据将SSIM定性为显存带宽瓶颈，也没有改窗口、近似Loss或盲目搜tile配置。

## 10. 固定渲染与连续训练Gate

下表是相同K/图像的各指标相对B的**最大绝对差**；PSNR按clamp[0,1]标准图像，另存unclamped PSNR；photo使用原始渲染值。标准SSIM与Loss中的fast_ssim分开存储。

{md(qdiff,['版本']+quality_cols)}

固定fixture精确项异常{len(exact_bad)}项，shape/grad=None一致。前向和梯度按预注册FP32 atol2e-5/rtol2e-4检验；固定梯度Adam单独采用2e-7/2e-6。FP64采用atol1e-12/rtol2e-14，未变更。

固定Backward之后仍有{len(fixed_bad)}组parameter-after超普通浮点门槛；包括B自身重复，不把这些覆盖为通过。最坏元素、Gaussian UID/owner与输入在worst_error_locations.csv及完整fixture中。初始render、selector、loss一致不等于所有Adam后参数一致。

连续控制：原分辨率50轮Full→轮转LoD，半分辨率LoD另30轮；两次B、两次E，固定地图/相机/缓存uniform/拓扑，使用相同fresh Adam初始化，**没有冒充PLY恢复了完整在线优化器状态**。快照0/9/24/49或29保存全部梯度、参数、moments、step；严格差异仍记录。

{md(curverows,['运行','Full_photo','LoD_photo'])}

这些是固定单视图短程验证，不是独立场景质量实验。后期RGB和参数逐元素差异会累积，B自身也发生漂移；候选个别重复超过基线包络。因此不标记长期逐元素Parity为通过，也不以loss接近替代梯度正确性。

## 11. 完整在线fr1_xyz运行

两次完整轨迹均正常退出，非串行joint，Loss/调度/分辨率/自然停止配置未改。以下是**真实异步系统**的观测，不是同地图/同训练预算的因果加速实验。

{md(onlat,['版本','阶段','样本','mean','p50','p95','p99'])}

{md(onlinerows,['版本','Gaussian','Anchor','Online_steps','Tail_steps','Online_topology','Tail_topology','R3_Adam_steps'])}

最终checkpoint检查owner范围、1～10成员、member_counts、L0～L2、UID唯一性、四张R3表[N,14]全部通过；LoD nonfinite计数为0。未保存在线完整Master Adam的既有导出限制没有被假装消除，moment迁移依靠受控fixture验证。

在线步数/Topology/最终N不同，不能由这里的最终地图质量差异判定CUDA错误，也不能将该时延差全部归因kernel优化。本轮按要求不继续研究历史在线质量差异。完整进程总时长见suite_commands.json；输入时间推进及自然停止使更快Mapping通常体现为更多迭代，不能等同轨迹吞吐提升。

## 12. 规模扩展与剩余瓶颈

人工N=50K/100K分别保存Anchor、Sampling Forward/Backward计时；原始分布和真实地图明确区分。Anchor人工压力为满10成员布局，Selector为人工同ratio概率/固定uniform，不能当成100K真实SLAM端到端测量。

{md(scalerows,['范围','N','CUDA','阶段','p50','p95'])}

{remaining_report}

## 13. 开关与复现

```yaml
CUDA.fused_dense_adam: 0
CUDA.fused_compensation_decode: 0
CUDA.fused_anchor_input_pack: 0
CUDA.fused_anchor_full_stats: 0
CUDA.fused_selector_compaction: 0
CUDA.reuse_renderer_workspace: 0
CUDA.optimized_ssim_backward: 0
```

独立变更位置、接口和源码SHA均在本报告及manifest；脚本与完整命令见本目录README、run_suite.py、suite_commands.json。最终结果目录为服务器`anchor-selector-r3-cuda-phase2-20261023/final_legacy_reduction`；初版确定性归约保留在独立证据中，不与最终速度混算。

本地完整Tensor与失败证据已提取到[证据目录](<{(HERE/'evidence_archive').as_posix()}>)/results；包括B/E四ratio+Full的Forward/VJP/Adam fixture、连续训练快照与Topology迁移fixture。328个文件逐项SHA已验证，45个源码文件与服务器一致；检查结果见[delivery_checks.json](<{(HERE/'delivery_checks.json').as_posix()}>)。原始未选入压缩包的重放和完整在线模型仍保留在服务器实验目录。

交付CSV均为实测数据。未完成：独占GPU认证、任意ULP prune exact、长期逐元素一致性认证、Renderer设备端动态长度接口、跨stream训练、SSIM硬件计数器。没有为通过测试修改epsilon、阈值、Loss、网络或采样。

**下一步优先SSIM/Renderer Backward；Anchor高频校验和Selector压缩已明显缩小。R3 gather与更复杂workspace池不是当前优先项。生产默认保持Legacy，后续按本轮失败/未确定Gate逐项放行。**
'''
(ROOT/'ANCHOR_SELECTOR_R3_CUDA_OPTIMIZATION_PHASE2_REPORT.md').write_text(text,encoding='utf-8')
print('REPORT_PASS',flush=True)
