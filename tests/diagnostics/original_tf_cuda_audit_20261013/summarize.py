"""用保存的实测材料生成中文结果、图和完成检查；不连接服务器。"""
import csv,hashlib,json,sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
dest=Path(__file__).resolve().parent;repo=dest.parents[2]
native=list(csv.DictReader((dest/'native_benchmark.csv').open()))
stages=list(csv.DictReader((dest/'stage_benchmark.csv').open()))
profile=json.loads((dest/'profiler_summary.json').read_text())
nsight=json.loads((dest/'native_nsys_summary.json').read_text())
parity=json.loads((dest/'mirror_parity.json').read_text())
validity=list(csv.DictReader((dest/'native_validity.csv').open()))
inventory=json.loads((dest/'inventory.json').read_text())
manifest=dict(local_branch='cuda-tf',local_head='99a9bb1f05e458617f50fabb30a910f14aaf7c9c',
             compared={f:dict(local=hashlib.sha256((repo/f).read_bytes()).hexdigest(),remote=h) for f,h in inventory['source_hashes'].items()})
manifest['all_equal']=all(v['local']==v['remote'] for v in manifest['compared'].values())
(dest/'local_manifest.json').write_text(json.dumps(manifest,indent=2))
def get(mode,n,stage):return next(v for v in native if v['mode']==mode and int(v['N'])==n and v['stage']==stage)
lines=['## 实测结果与复现材料','',f"16个源码/配置SHA-256全部一致：{manifest['all_equal']}。native↔Python镜像的{len(parity)}项forward/input/parameter gradient全部通过，最大绝对差{max(v['max_abs'] for v in parity.values()):.3g}。{len(validity)}组连续训练后的参数、输出、输入/参数梯度、Adam m/v有限值检查全部通过：{all(v['parameters_outputs_gradients_states_finite']=='1' for v in validity)}（见native_validity.csv）。真实TF共有26个参数tensor、1,186,762个FP32参数。`checkpoint_optimizer.txt`确认该checkpoint shadow optimizer为`groups=2 states=0`。",'',
       '### Native生产源码CUDA events','',
       '以下均为600次连续训练中的后500次，warmup100。每个单元为mean / p50 / p95（ms）。','']
for mode in ('core_adam','regularized_adam','regularized_preview'):
    lines += [f'**{mode}**','','| N | forward | loss/regularization | backward | zero_grad | optimizer | total | wall |','|---:|---:|---:|---:|---:|---:|---:|---:|']
    for n in (1000,5000,10000,20000,50000):
        parts=[str(n)]
        for stage in ('forward','loss_regularization','backward','zero_grad','optimizer','total','wall'):
            r=get(mode,n,stage);parts.append(' / '.join(f'{float(r[k]):.3f}' for k in ('mean_ms','p50_ms','p95_ms')))
        lines.append('| '+' | '.join(parts)+' |')
    a=float(get(mode,10000,'total')['mean_ms']);b=float(get(mode,50000,'total')['mean_ms'])
    lines += ['',f'10k→50k总时间：{a:.3f}→{b:.3f} ms，倍率{b/a:.3f}，增加{(b/a-1)*100:.1f}%。','']
isolated=list(csv.DictReader((dest/'native_optimizer_isolated.csv').open()))
(dest/'native_benchmark.json').write_text(json.dumps(dict(
    protocol=dict(warmup=100,measurements=500,precision='FP32',ratio=0.05,
                  input_sampling='cycle representative checkpoint rows',
                  optimizer=dict(mlp_lr=0.00016,grid_lr=0.0016,betas=[0.9,0.999],eps=1e-15,
                                 schedule='fixed diagnostic init LR; fresh state for each mode/N'),
                  checkpoint=inventory['checkpoint'],local_head=manifest['local_head']),
    native=native,optimizer_isolated=isolated,finite_checks=validity),indent=2))
lines += ['### 隔离optimizer / zero_grad','','上一段阶段event允许CPU继续enqueue：50k backward更长会隐藏后一阶段CPU launch gap，导致阶段optimizer计时低于1k，不能解释为Adam工作量减少。以下在阶段开始前同步，独立warmup100/measurement500，N=0表示固定TF参数池、没有Gaussian输入。zero_grad保留参考gradbuffer以隔离set_to_none控制成本，内存回收另见native流水阶段。','','| 操作 | CUDA mean/p50/p95 ms | wall mean/p50/p95 ms |','|---|---:|---:|']
for mode in ('adam','preview','zero_grad'):
    vals=[]
    for stage in ('cuda_event','wall'):
        r=next(v for v in isolated if v['mode']==mode and v['stage']==stage)
        vals.append(' / '.join(f'{float(r[k]):.4f}' for k in ('mean_ms','p50_ms','p95_ms')))
    lines.append('| '+mode+' | '+' | '.join(vals)+' |')
lines += ['','### Nsight Systems：生产C++直接trace，N=8491','','Nsight Systems 2022.4.2.1，cuda/nvtx tracing，关闭CPU sampling/context-switch tracing；预热100次后由cudaProfilerStart/Stop采集10步regularized_adam。使用runtime correlation归属NVTX阶段，不依赖Python镜像。GPU active包含该阶段kernel、D2D copy和memset的区间并集。','','| 阶段 | CPU NVTX mean ms | kernel数/step | kernel active ms/step | D2D copy数/step | memset数/step | GPU span空隙范围 |','|---|---:|---:|---:|---:|---:|---:|']
for stage in ('zero_grad','forward','loss','regularization','backward','optimizer','synchronize'):
    r=nsight['stages'][stage];g=[v['gap_fraction'] for v in r['per_iteration_gpu']]
    gap=f'{100*min(g):.1f}–{100*max(g):.1f}%' if g else '无device工作'
    lines.append(f"| {stage} | {r['cpu_nvtx_mean_ms']:.3f} | {r['kernels_per_iteration']:.0f} | {r['kernel_active_ms_per_iteration']:.3f} | {r['copies_per_iteration']:.0f} | {r['memsets_per_iteration']:.0f} | {gap} |")
iters=nsight['per_iteration'];wall=sum(v['wall_ms'] for v in iters)/len(iters);active=sum(v['device_work_active_ms'] for v in iters)/len(iters);gap=sum(v['gap_fraction'] for v in iters)/len(iters)
lines += ['',f"每步{nsight['kernel_count']/len(iters):.0f}个计算kernel，其中{nsight['short_kernel_fraction']*100:.1f}%短于10us。instrumented wall mean={wall:.3f} ms，GPU device work区间并集mean={active:.3f} ms，平均span空隙{gap*100:.1f}%；**这不是无插桩的framework百分比**。Nsight明显放大短kernel的API/dispatch成本，不能拿18.28 ms替代前面的native CUDA event结果，也不能把CPU与GPU时间相加。",'',
          '10步中cudaLaunchKernel 9,910次、cudaMemcpyAsync 1,560次、cudaMemsetAsync 100次、显式末尾cudaStreamSynchronize 10次。copy均为D2D，计时内没有H2D/D2H或cudaMalloc/cudaFree；allocator的Tensor/缓存复用成本仍可能存在，不能说没有allocation overhead。原始TF内部未新增sync。完整API CPU时间和每步span在native_nsys_summary.json，原始trace为native_nsys.nsys-rep/SQLite。','',
          f"native GEMM/GEMV占kernel active time的{nsight['gemm_kernel_fraction']*100:.1f}%，支持优先处理正则、plane scatter、内存操作与launch链条，而非只重写MLP。C++与Python/不同LibTorch构建的regularization/backward计数略有差异（native 93/602，镜像87/612）；镜像的32项parity只验证forward及其梯度，不能把镜像trace当作native精确计数。",'',
          '| 序号 | native kernel（短名，完整名见JSON） | CUDA ms（10步） |','|---:|---|---:|']
for i,r in enumerate(nsight['top20_kernels'],1):
    name=r['name'].removeprefix('void at::native::').replace('<unnamed>::','')
    if 'grid_sampler_' in name:name=name.split('(',1)[0]
    lines.append(f"| {i} | `{name[:140]}` | {r['total_ms']:.3f} |")
lines += ['','### Python镜像分段归因','','全部stage的mean/p50/p95在stage_benchmark.csv。这里列mean ms；每个planequery/product的event分段会放大插桩成本，不能与native total求和。','','| N | input prep | 12 plane queries合计 | fusion合计 | MLP | output transform |','|---:|---:|---:|---:|---:|---:|']
for n in (1000,5000,10000,20000,50000):
    vals=[next(v for v in stages if int(v['N'])==n and v['stage']==k) for k in ('input_preparation','plane_query','feature_fusion','mlp_forward','output_transform')]
    lines.append('| '+str(n)+' | '+' | '.join(f'{float(v["mean_ms"]):.3f}' for v in vals)+' |')
lines += ['','### Python Profiler：真实N=8491，预热100次、3次采集，以下计数均除以3','','使用CUDA runtime correlation将kernel归到CPU record_function范围，避免Kineto external-ID歧义。CPU profiler/shape/memory插桩影响绝对gap；launch数和短kernel分布用于结构归因。500次性能测量见native/分段CSV，这3次trace用于补充operator/shape/memory，C++真实launch计数见上面Nsight表。','','| 模式 | forward kernels | backward kernels（含probe loss） | optimizer kernels | regularization forward kernels | aten calls | empty allocation calls | <10us kernel占比 |','|---|---:|---:|---:|---:|---:|---:|---:|']
for mode,p in profile.items():
    k=p['kernels'];lines.append(f"| {mode} | {k.get('forward',0)/3:.0f} | {k.get('backward',0)/3:.0f} | {k.get('optimizer',0)/3:.0f} | {k.get('regularization',0)/3:.0f} | {p['aten_calls']/3:.0f} | {p['allocation_calls']/3:.0f} | {100*p['small_kernel_count']/p['total_kernel_count']:.1f}% |")
lines += ['','计时内没有H2D/D2H；copy为D2D（mean每步core10次、regularized166次、preview270次）。2次cudaDeviceSynchronize来自diagnostic显式末尾同步/profiler收尾，不是TF内部插入的同步。CPU launch gaps明显，详见各stage GPU span与kernel active time；allocator存在大量empty调用，但本次不能量化cudaMalloc占总时间的独立比例。','',
          '| 模式 | forward GPU kernel active ms/iter | backward active ms/iter | optimizer active ms/iter | forward span空隙范围 | backward span空隙范围 | resident baseline MiB | peak allocated MiB |','|---|---:|---:|---:|---:|---:|---:|---:|']
for mode,p in profile.items():
    t=p['kernel_us'];f=[v['gap_fraction'] for v in p['per_iteration_gpu_spans']['forward']];b=[v['gap_fraction'] for v in p['per_iteration_gpu_spans']['backward']]
    lines.append(f"| {mode} | {t['forward']/3000:.3f} | {t['backward']/3000:.3f} | {t['optimizer']/3000:.3f} | {min(f)*100:.1f}–{max(f)*100:.1f}% | {min(b)*100:.1f}–{max(b)*100:.1f}% | {p['resident_at_start']/2**20:.2f} | {p['peak_allocated']/2**20:.2f} |")
lines += ['','上述空隙包括launch/CPU执行与copy等待，不能标成纯framework百分比；baseline包含模型参数、Adam、grads、真实input和诊断上下文；peak-resident更适合比较额外autograd工作区，仍不是只算TF常驻池的值。','',
          '### 最贵20个operator / kernel','','regularized模式。operator按aten名称跨shape汇总self CUDA时间；kernel按实际device duration汇总。两种表不能相加，否则双重计数。完整kernel模板名、shape、次数、CPU/CUDA/memory在profiler_summary.json。','','| 序号 | aten operator | self CUDA us（3步） | count（3步） |','|---:|---|---:|---:|']
aggregate={}
for o in profile['regularized']['operators']:
    if not o['name'].startswith('aten::'):continue
    a=aggregate.setdefault(o['name'],[0,0]);a[0]+=o['self_cuda_us'];a[1]+=o['count']
for i,(name,(t,c)) in enumerate(sorted(aggregate.items(),key=lambda v:v[1][0],reverse=True)[:20],1):lines.append(f'| {i} | `{name}` | {t:.1f} | {c} |')
lines += ['','| 序号 | kernel（短名，完整名见JSON） | CUDA us（3步） |','|---:|---|---:|']
for i,(name,t) in enumerate(profile['regularized']['top20_kernels'],1):
    short=name
    if name.startswith('void at::native::'):
        short=name.removeprefix('void at::native::').replace('(anonymous namespace)::','').split('(',1)[0][:140]
    lines.append(f'| {i} | `{short}` | {t:.0f} |')
gemm=sum(t for name,t in profile['regularized']['top20_kernels'] if any(v in name.lower() for v in ('gemm','gemv')))
# Exact GEMM share comes from the whole saved trace, not truncated top20.
events=json.loads((dest/'profile_regularized.json').read_text())['traceEvents']
kernels=[e for e in events if e.get('cat')=='kernel']
gemm=sum(e['dur'] for e in kernels if any(v in e['name'].lower() for v in ('gemm','gemv')))
total=sum(e['dur'] for e in kernels)
lines += ['',f'GEMM/GEMV kernel合计{gemm/3:.1f} us/step，占regularized已观测kernel active time {gemm/total*100:.1f}%。这不支持“MLP GEMM占主导”；内存/填零/copy、plane backward及regularization和大量launch更值得优先处理。未用Nsight Compute测DRAM/L2/occupancy，因此不能给出精确显存带宽占比。','',
          '### 真实历史输入N分布','','只统计历史原始TF日志中LoD步骤的正tf_N（初始化前0另列）；来自多场景UltraFast轮转与稀疏normal日志，带调度/场景偏差。','','| ratio | 正N样本数 | zero N样本数 | min | median | p90 | max |','|---:|---:|---:|---:|---:|---:|---:|']
for r in json.loads((dest/'logged_tf_shapes.json').read_text())['summaries']:
    lines.append('| '+' | '.join(f'{r[k]:.1f}' if k in ('median','p90') else str(r[k]) for k in ('ratio','active_tf_samples','zero_tf_rows','min','median','p90','max'))+' |')
reg=float(get('regularized_adam',10000,'total')['mean_ms']);core=float(get('core_adam',10000,'total')['mean_ms'])
lines += ['','### 最终性能判断','',f'在该fast模型/realistic重复采样负载下，原始regularization使10k训练步从{core:.3f}增加到{reg:.3f} ms，额外{reg-core:.3f} ms，占regularized total的{(reg-core)/reg*100:.1f}%。这是两组诊断的差值归因，含CPU/operator/backward调度差异，不等于正则纯GPU算力成本。regularized和preview对N不敏感，纯core也远小于5倍增长；固定全plane工作和framework/launch/optimizer overhead是主要可优化方向，不能把主要收益只押在MLP FLOPs。','',
          '推荐优先顺序：保语义融合正则forward/backward与optimizer事务buffer操作；融合plane query+product/coordinate；再比较GEMM周边融合与fast H32 warp MLP。自定义backward需要plane atomic、MLP dW reduction和normal输入梯度，custom optimizer需要prepare/commit、缺失grad和per-tensorstep。','',
          '原始实现确有10k→50k时间接近不变现象，尤其包含正则/preview时；数据已分模式给出。该结果不能推导完整Mapping加速倍数，不能归因于所有GPU配置，也不支持删除rotation/ChangeDistill/regularization。','',
          '### 复现命令与交付文件','',
          '```bash','cd /home/prmi/zwj/mycartgs/tests/diagnostics/original_tf_cuda_audit_20261013','bash run.sh','bash run_nsight.sh','# 有源码/配置修改时先重新检查inventory.json与local_manifest.json','# 单独重建trace归因：','/home/prmi/anaconda3/envs/zwj/bin/python reparse_profile.py','/home/prmi/anaconda3/envs/zwj/bin/python parse_nsight.py','```','',
          'run.sh固定CUDA11.8、Torch_DIR和TORCH_CUDA_ARCH_LIST=8.6；诊断CMake适配服务器3.16.3。checkpoint/input/state/config/sourcehash、native CSV/逐样本CSV/finite检查、分段CSV、profiler trace+summary、native parity archive+JSON、autograd graph、构建log、复现脚本、曲线SVG/PNG均在同目录。没有rendered RGB。原始checkpoint不修改，native使用新模型副本。','',
          '曲线点为mean，阴影为p50–p95；完整统计见上表。',
          f'![N增长曲线]({(dest/"native_scaling.png").as_posix()})','']
section='\n'.join(lines)
report=repo/'ORIGINAL_TF_CUDA_AUDIT.md'
old=report.read_text(encoding='utf-8').split('## 实测结果与复现材料')[0]
old=old.split('## 20. 重写范围与明确结论')[0]
conclusions=[
    '## 20. 重写范围与明确结论','',
    '1. 精确architecture：6L个FP32 NCHW plane→每级6feature product→L级concat→共享F→H Linear→三个H→H→3/3/4 head；fast12plane/F16/H32，总1,186,762参数。',
    '2. 单Gaussian forward：world AABB normalize、raw ratio坐标、bilinear border/align_corners query、product/cat、7Linear/6ReLU、XYZ加法、scale exp乘法、identity quaternion residual两次normalize与左乘；没有tanh/bound。',
    f'3. 本负载主要瓶颈是全plane正则、Adam/preview和大量短kernel调度。10k原始正则使总时间增加{reg-core:.3f} ms（regularized total的{(reg-core)/reg*100:.1f}%）；GEMM/GEMV仅占实测kernel active time的{gemm/total*100:.1f}%，不支持MLP算力主导。',
    f'4. native Nsight每步991个kernel、97.0%短于10us；forward span空隙69–74%、backward79–80%，整步平均{gap*100:.1f}%。这些已经包含device copy/memset，却仍受Nsight/driver插桩影响，不能标为无插桩纯framework百分比。Python profiler会进一步放大gap。缺少Nsight Compute和端到端Mapping测量，不给DRAM效率或完整工程加速的虚假估计。',
    '5. Six-plane query+fusion适合fused CUDA：消除12/18次grid_sample中间feature、多次product/cat；必须保留border derivative、非连续布局和plane atomic gradient。',
    '6. fast H32适合warp级MLP候选，default H64需多warp或GEMM。建议先保留GEMM建立数值基线，再测fully fused；当前证据不足以断言后者更快。',
    '7. backward内部可全部自定义CUDA，但normal input gradient、Guard VJP和renderer loss链要求thin autograd/显式VJP桥。单独设requires_grad=false并取消桥接不兼容现有调用链。',
    '8. optimizer可CUDA化：保留两个LR逻辑组、per-tensor step/has_grad、normal/CD/DSD状态隔离、prepare/commit事务；不能用单个无条件原地update替代。',
    '9. 重写复杂点：ChangeDistill teacher/domain transport/resample，Guard projection/fallback/candidate舍入，Replay共享state与独立clock，checkpoint shadow optimizer没有真实shared state，以及Gaussian属性梯度。保留Parameter identity和snapshot生命周期。',
    '10. kernel族：可选mask/gather；query+product+normalize；MLP/GEMM+ReLU；output compose/scatter；output VJP；MLP activation/input/dW/db reductions；query coordinate/plane atomic backward；全plane正则/reduction；Adam prepare；candidate/commit/reset。',
    '11. 推荐按以上kernel族逐步融合。首版GEMM+周边融合验证FP32 parity，backward拆dW reduction与plane scatter，optimizer拆prepare/commit。理想的单forward、单backward、单optimizer三kernel忽略了全局reduction和事务。',
    '12. 可消除：[N,4]坐标、每plane grid、12/18份sample、5L份product、3次重复共享ReLU物理tensor、多份quaternion norm/cross小tensor。输出attribute、注册参数、必要grad和candidate保留。',
    '13. 实测优先顺序：保语义融合正则forward/backward及Adam/preview事务buffer操作；融合plane query+product/coordinate；优化MLP/输出周边并比较warp MLP。不得删正则、rotation、ChangeDistill换取速度。',
    '14. 存在10k→50k耗时接近不变现象：core +13.8%，regularized +6.0%，preview +0.1%。全plane工作量固定，CPU enqueue可与GPU执行重叠；阶段Adam时间下降不表示Adam工作量减少。不能由此推导完整Mapping加速倍数。',
    '15. 第4–19节及parameter_shapes.json、inventory/config、CSV、profiler、parity、autograd graph给出shape/layout、梯度、state及接口规格。后续仍须实现并验证边界、Adam多步、Guard/CD/Replay与checkpoint parity；本轮没有CUDA TF实现、生产改动或commit。','']
report.write_text(old+section+'\n'+'\n'.join(conclusions),encoding='utf-8')
(dest/'RESULTS.md').write_text(section,encoding='utf-8')
plt.rcParams.update({'font.size':10,'svg.fonttype':'none','font.sans-serif':['Microsoft YaHei','Noto Sans CJK SC','DejaVu Sans'],'axes.unicode_minus':False})
fig,ax=plt.subplots(figsize=(7.5,4.2))
for mode,label in [('core_adam','TF核心 + Adam'),('regularized_adam','核心 + 原始正则 + Adam'),('regularized_preview','核心 + 原始正则 + Adam preview')]:
    ns=[1000,5000,10000,20000,50000];ys=[float(get(mode,n,'total')['mean_ms']) for n in ns]
    ax.plot(ns,ys,marker='o',label=label)
    low=[float(get(mode,n,'total')['p50_ms']) for n in ns];high=[float(get(mode,n,'total')['p95_ms']) for n in ns]
    ax.fill_between(ns,low,high,alpha=.12)
ax.set(xlabel='TF输入Gaussian数量 N',ylabel='CUDA event时间（ms）',title='原始TF / RTX 3090 / FP32 / 预热100次 + 测量500次')
ax.grid(alpha=.2);ax.legend(loc='best',fontsize=9);fig.tight_layout()
fig.savefig(dest/'native_scaling.svg');fig.savefig(dest/'native_scaling.png',dpi=180);plt.close(fig)
print('results written, parity',all(v['passed'] for v in parity.values()),'sourcehash',manifest['all_equal'])
