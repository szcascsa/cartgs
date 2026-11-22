"""中文最终报告，所有速度/质量从真实CSV提取，不使用开发集代替验收集。"""
from pathlib import Path
import json
import numpy as np,pandas as pd

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent;R=HERE/'reports'
read=lambda n:json.loads((R/n).read_text())
geo=lambda x:float(np.exp(np.log(x).mean()))
table=lambda header,rows:'| '+' | '.join(header)+' |\n| '+' | '.join(['---']*len(header))+' |\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in rows)
result=read('RESULTS_SUMMARY.json');integrity=read('INTEGRITY_FINAL.json');frozen=read('FROZEN_JOINT_CANDIDATE.json')
assert result['passed'] and integrity['passed']
for name in ['boundaries_final','memcheck_final','synccheck_final','racecheck_new_final']:
    assert read('jobs/'+name+'.json')['returncode']==0,name
t=pd.read_csv(R/'TIMINGS_ALL_VARIANTS.csv');paired=pd.read_csv(R/'PAIRED_ALL_VARIANTS.csv');agg=pd.read_csv(R/'AGGREGATE_ALL_VARIANTS.csv')
s=pd.read_csv(R/'PRIMARY_SUMMARY.csv');co=pd.read_csv(R/'COMPONENT_RESULTS.csv');q=pd.read_csv(R/'QUALITY_PER_CONDITION.csv');qraw=pd.read_csv(R/'QUALITY_ALL_FRAMES.csv')
primary=['local0.1','local0.2','local0.3','local0.4','local0.5']
def item(ref,variant,scope):return s[(s.reference==ref)&(s.variant==variant)&(s.scope==scope)].iloc[0]
def times(v,scope):return t[(t.variant==v)&(t.scope==scope)&t.condition.isin(primary)]
quality=result['quality']['D'];dA=item('A','D','pipeline');dC=item('C','D','pipeline')
text=f'''# Tensor Core与准备/调度联合推理优化：9小时预算报告

## 结论和范围

最终保留普通CUDA准备、C++相机、GPU动态调度与Tensor Core网络的混合方案D。主要Ratio 0.1–0.5，Replica全部16组、两轮、连续运动和跳转视角配对：准备＋渲染相对原融合FP32 CUDA A加速{dA.speedup:.4f}×（延迟降低{100*(1-1/dA.speedup):.2f}%），相对已验收最佳网络C再加速{dC.speedup:.4f}×（降低{100*(1-1/dC.speedup):.2f}%）。这些是当前帧热态单帧延迟，不是跨帧流水线吞吐。

D相对A平均额外PSNR损失{quality['mean_psnr_loss']:+.8f}dB，最差单帧{quality['worst_frame_loss']:.6f}dB，最差场景条件均值{quality['worst_condition_mean_loss']:.6f}dB；平均≤0.05dB、场景条件均值≤0.2dB门限通过。新增准备相对C的平均额外损失{quality['extra_C_mean_loss']:+.8f}dB。FP16网络与预投影运算重排仍可能改变选点和局部像素，不能称逐位渲染等价。

原六小时网络专项源码、报告和所有证据保留。9小时是上限，沿用原研究目录2026-11-10 22:00（香港时间）为保守起点，原截止为11-11 04:00，新截止为07:00；没有从新请求重计。最早保护清单mtime为22:02:52，首次编译为22:08:07。新增准备专项从23:50:57开始。最终完整性检查时，距保守原起点{integrity['elapsed_from_original_start_hours']:.3f}小时，新增专项{integrity['elapsed_preparation_hours']:.3f}小时；未声称实际运行满9小时，也未以空等或重复测试填满预算。

只修改独立研究目录`{HERE}`；不改训练、网络权重、生产默认或原checkpoint，不commit/push/部署。相同蒸馏学生用于所有版本，不把Teacher→Student旧收益混入本轮；Renderer和原R相机对照复用已冻结版本。

## A/B/C/D和独立开关

| 版本 | 定义 | 当前帧准备 |
| --- | --- | --- |
| A | 原v31融合FP32 CUDA，原学生 | 原R相机与原准备/Host阈值 |
| B | 原网络专项最初V1 Tensor Core，TF32 WMMA、4warp | 原准备；大图旧host路径仍为原网络 |
| C | 网络专项最终冻结：Selector寄存器FP16 MMA 4warp、学生Compensator FP16 WMMA 2warp | 原R相机和原准备/调度 |
| D | C＋本阶段接受的准备与调度 | CUDA视锥融合、32D线性预投影、GPU动态N/K、C++相机 |

新增3小时开始时的已存在版本实际是C。本阶段没有保留额外网络专属改动，因此按“新增阶段起始网络”定义的B与C是同一份后端；为保留整个研究链的四组归因，另测并明确命名历史V1为表中B。最关键的新增收益始终以D/C计算，绝不以较慢的历史B冒充新增收益。

D_noF关闭视锥融合，D_noP关闭投影，D_noS恢复旧大图Host调度，D_noH恢复R相机。CUDA_preparation采用原FP32两网络＋CUDA准备/调度＋C++相机，关闭预投影，用于识别纯准备收益与混合协同。所有开关与状态均在独立后端，生产入口不导入。

'''
rows=[]
for v in ['A','B','C','D']:
    values=[]
    for scope in ['camera','engine_prepare','prepare','pipeline']:
        h=times(v,scope);values.append(f'{h.p50.median():.4f} / {h.p95.median():.4f}')
    rows.append([v]+values)
text+=table(['版本','纯相机 p50/p95 ms','地图/网络准备 p50/p95 ms','相机＋完整准备 p50/p95 ms','准备＋渲染 p50/p95 ms'],rows)
text+='\n耗时列是场景/条件/轮次中位耗时及p95的中位数；加速为相同场景条件轮次的等权几何平均，不应直接用总表两列中位数相除。每次重做当前相机；engine_prepare只排除相机，prepare包含相机，pipeline再加原G Renderer。模型读取、冷启动编译、Graph首次构建、显示/HTTP和在线SLAM不在热态耗时内。分阶段分别计时，不把独立中位数强行相加。\n\n'
rows=[]
for c in primary:
    x=agg[(agg.variant=='D')&(agg.condition==c)]
    a=x[(x.reference=='A')&(x.scope=='pipeline')].iloc[0];z=x[(x.reference=='C')&(x.scope=='pipeline')].iloc[0];p=x[(x.reference=='C')&(x.scope=='prepare')].iloc[0]
    rows.append([c[5:],f'{a.reference_p50:.4f}→{a.p50:.4f}',f'{a.speedup:.4f}×',f'{z.speedup:.4f}×',f'{p.speedup:.4f}×',f'{int(z.slower_rounds)}/{int(z.scene_rounds)}'])
text+=table(['Ratio','完整路径A→D ms','D/A加速','D/C新增加速','准备D/C新增加速','完整D/C慢例'],rows)

text+='''
## 准备热点和实际分工

上一阶段两个网络变快，但完整路径只快约1%，原因是相机准备、视锥、候选compact、预算控制、选择、状态回读和发布仍在关键路径。最终分工如下：

| 运算 | 采用方式 | 实测与理由 |
| --- | --- | --- |
| 48×32批量静态Feature投影 | 加载时FP32矩阵乘法 | 只依赖冻结权重和Feature，移出每帧关键路径；未用TF32近似预计算 |
| Selector隐藏层和Head | FP16 Tensor Core、FP32累加 | 延续网络阶段已验收实现 |
| Compensator剩余32→32、32→14 | FP16 WMMA、FP32累加 | 原输入层通过线性分解移出，保留ReLU边界 |
| 6个平面的3维点积、半径及Mask | 普通CUDA融合 | 维度小、补零与布局成本高，CUDA更快且保持候选Mask |
| 当前4×4相机变换与逆矩阵 | C++ CPU计算＋一次异步HtoD | 避免小矩阵GPU调度和相机错误码DtoH同步；保留当前请求校验 |
| Base/Residual按Ratio组合、View/Mode投影 | 普通CUDA逐Anchor | 不规则读取，逐点MMA不合算；当前帧结果仅供本次调用 |
| compact、严格K、阈值选择、finite状态 | CUDA/CUB、C++、Device-side控制 | 整数计数和索引不是密集矩阵问题，不强行交给Tensor Core |
| XYZ/Scale/Rotation解码与发布 | 原FP32 CUDA融合 | 保留物理公式、SH、UID顺序和有限值检查 |

### 线性分解、显存和失效

原Compensator第一层在非线性之前为

`z_i = W_o o_i + W_f(F_a^base + Σ_j λ_j(r)F_a^j) + W_r e(r) + W_v v_a(view,mode)`。

冻结加载时计算`P_i = W_o o_i + W_f F_a^base`和每级`Q_a^j = W_f F_a^j`，每帧普通CUDA按原Residual层顺序形成`D_a = Σ_j λ_j Q_a^j + W_r e(r) + W_v v_a`。选中点直接读取`P_i + D_a + bias`，随后原ReLU、FP16转换和剩余网络。没有跨越ReLU，也没有把不同ratio的完整网络输出相加。实数线性关系等价，但FP32加法分组、输入half舍入、原输入层MMA顺序变化导致数值微差，故重新做图像质量验收。

静态Cache是`N×32`和`(L−1)×A×32`的FP32，临时当前Anchor输入为`A×32`，额外容量`128×(N+L×A)`字节。它依赖冻结Offset、Anchor映射、Base/Residual表、输入层权重和拓扑；私有冻结快照不支持原地训练。更换地图/权重/Feature/Clone/Split/Prune/重排/Anchor迁移须新建实例；切换准备选项清除Graph并重建投影。没有依赖视角或Ratio的跨帧预测缓存。

'''
text+='新增源码改动集中在9个文件：preparation_kernels.cu完成视锥/计数融合和被拒绝的TF32诊断；preparation_camera.cpp/.py完成当前相机C++准备及公开接口；tensor_networks.cuh和tensor_components.inc加入预投影后的输入路径；deployment_runtime_kernels.cu连接当前Anchor投影；deployment_prepare.cpp管理静态投影、Graph失效和GPU调度；deployment_runtime_bindings.cpp及deployment_runtime.py暴露独立开关。相对C的源码补丁为reports/COMBINED_PREPARATION.patch，未触碰训练或Renderer源码。\n\n'
pc=pd.read_csv(R/'precompute_final/samples.csv')
text+=f"最大新增投影和动态工作区{result['peak_projection_bytes']/(1<<20):.3f}MiB；预计算单独Host中位{pc.host_ms.median():.4f}ms、GPU Event中位{pc.gpu_event_ms.median():.4f}ms，包含新增finite检查，不含读取模型。观察到一次{pc.host_ms.max():.4f}ms冷启动长尾，未把首次库/执行计划初始化与投影单独剥离，不隐藏为稳态成本；它不属于后续每帧收益。已保存各场景初始化开销、5次重复和内存CSV；process peak是多版本共同驻留的测量进程，不能归为D单实例新增显存。\n\n"

text+='''### 输入、发布和调度

Offset本来已在冻结实例加载时完成asinh/clamp；原Tensor版也已经把Feature Gather、half转换、Head Bias、物理解码、SH和ID发布放在网络Kernel内。本阶段没有把这些旧优化重复计功。新收益是删除48D逐点查询与51→32输入MMA，将输入边界移动到32D、按Anchor共享当前动态投影。

旧地图N>65,536时需要Host候选计数，再以CPU计算K和Ratio控制、发射网络并读finite状态。D为全部已验收工作量保留GPU动态计数和预算，通过Device Frame传递N/K，Graph按有限容量桶组织；高Ratio超过原Graph容量仍走device普通发射，未缓存旧计数。输出每帧独立分配并由请求携带指针，一次最终96字节控制回读同时取得K和finite状态。必要检查在返回前完成，不延后、不取消。没有逐场景阈值。

C++相机只缓存固定内参；每帧从当前CPU pose计算FP64逆矩阵，转换原FP32 View，FP32 FMA计算Projection，再从FP32 View求当前Camera Center。一次140字节异步上传。保留奇异与非有限矩阵检查；Pinned staging由ATen记录Stream事件，上传完成前不会回收，同帧输出独立。Camera Center可有微小舍入差，已纳入质量；没有以刚体转置代替一般逆矩阵。

多Stream未作为最终方案：视锥→compact→K控制→Selector→选择→Compensator→检查形成主要依赖链，View与部分投影只有很小重叠机会；本轮优先删除同步和算子，未以额外Stream等待增加当前帧延迟。没有声称未实际实施的并行收益。

## 逐项归因和协同

'''
rows=[]
for v,label in [('D_noF','关闭CUDA视锥融合'),('D_noP','关闭线性预投影'),('D_noS','恢复旧Host调度'),('D_noH','恢复R相机'),('CUDA_preparation','FP32网络＋新增准备，无预投影')]:
    a=item('D',v,'prepare');b=item('D',v,'pipeline')
    rows.append([label,f'{100*(1/a.speedup-1):+.2f}%',f'{100*(1/b.speedup-1):+.2f}%',f'{int((paired[(paired.reference=="D")&(paired.variant==v)&(paired.scope=="pipeline")&paired.condition.isin(primary)].speedup>1).sum())}/160'])
text+=table(['独立条件','相对D准备延迟变化','相对D完整延迟变化','该条件比D快的轮次'],rows)
cuda=item('A','CUDA_preparation','pipeline');net=item('A','C','pipeline')
text+=f"\n原网络单独C/A为{net.speedup:.4f}×；普通CUDA准备＋原FP32网络为{cuda.speedup:.4f}×；联合D/A为{dA.speedup:.4f}×。各项存在共享固定开销、GPU/CPU重叠与算子边界变化，不能把百分比相加。D的预投影必须结合冻结学生输入层，因此CUDA_preparation刻意关闭该项，作为可解释控制。\n\n"
text+='GPU动态调度开关只改变原先超过65,536点的地图；其余小图执行路径相同，表中小幅正负差异属于配对测量波动，不能称所有小图都得到调度加速。固定身份的D补偿微测采用所有Anchor动态投影，可能比C更慢；完整路径只处理可见Anchor且与其他准备融合，因此以完整墙钟和实际Kernel Profile共同归因，不掩盖保守组件边界的不利结果。\n\n'
rows=[]
for stage in ['selector','compensator']:
    for v in ['A','C','D']:
        x=co[(co.stage==stage)&(co.variant==v)];rows.append([stage,v,f'{x.p50.median()*1000:.3f}',f'{x.p95.median()*1000:.3f}'])
text+=table(['组件','版本','p50 μs','p95 μs'],rows)
text+='\n组件为全16组同候选/选中身份、CUDA Event、16次固定输入Graph的隔离成本。D补偿组件包含所有Anchor的动态投影，未使用可见Mask，是保守边界；完整路径按可见Anchor做当前投影。历史B没有直接组件导出接口，未改写原二进制，其Selector/Compensator p50/p95另以profile_B实际Kernel证据报告，不将新版TF32算子冒充旧B。组件结果不能替代移动视角完整延迟。\n\n'
text+='## 测量与保留验证协议\n\n'
text+=f"RTX3090、i5-9600K、CUDA11.8、PyTorch2.3.1。只用office2 RGB-D、room0 RGB-D、office1 mono开发；其余Replica13组与视角留作冻结后验证，没有逐场景调参。9个条件版本、14类Ratio/固定预算/Full、每组3连续＋3跳转视角、每条件8次重复、两轮反序并轮换版本顺序，共{result['samples']:,}条计时，{result['selection_checks']:,}条候选/K/身份检查。所有场景、低Ratio、固定0/1/1000/5000和Full均保留，未隐藏慢例。\n\n"
m=result['monitor']
text+=f"采用的正式短批监测到{m['observations']}个低频负载样本，外部GPU活动{m['foreign_gpu_samples']}项、外部CPU活动{m['foreign_cpu_samples']}项；SM时钟{m['clock_mhz_min']:.0f}–{m['clock_mhz_max']:.0f}MHz、温度{m['temperature_c_min']:.0f}–{m['temperature_c_max']:.0f}℃。{len(result['rejected'])}个受干扰正式批次整批排除并重测，名单与原始CSV/监测记录保留；组件长批有干扰后拆为同工作量短批。低频采样不能完全排除瞬时OS扰动，因此同时给两轮、p95和全部退化项，不将Profiler时间当正式加速。\n\n"

text+='## 每场景每Ratio、Full和大图\n\n'
rows=[]
for scene in sorted(t.scene.unique()):
    row=[scene,int(t[t.scene==scene].N.iloc[0])]
    for c in primary:
        x=paired[(paired.scene==scene)&(paired.reference=='C')&(paired.variant=='D')&(paired.scope=='pipeline')&(paired.condition==c)]
        row.append(f'{x.reference_p50.median():.4f}→{x.p50.median():.4f} ({geo(x.speedup):.3f}×)')
    rows.append(row)
text+=table(['场景/模式','N','10% C→D ms','20%','30%','40%','50%'],rows)
full=paired[(paired.reference=='A')&(paired.variant=='D')&(paired.condition=='full')&(paired.scope=='pipeline')]
text+=f"\nFull全部逐场景轮次记录于CSV，共{len(full)}项，D/A几何平均{geo(full.speedup):.4f}×，其中{int((full.speedup<1).sum())}项慢于A。Full不调用Selector/Compensator，其差异只能归因相机准备和系统波动，不能记为Tensor Core网络收益。没有通过特殊Ratio/K阈值让Full绕开新代码来制造结果。所有慢例包括ALL_SLOWER_CASES.csv，不删除反例。\n\n"
largest=t.groupby('scene').N.first().idxmax();x=paired[(paired.scene==largest)&(paired.reference=='C')&(paired.variant=='D')&(paired.scope=='pipeline')&paired.condition.isin(primary)]
text+=f"最大Replica地图{largest}，N={int(t[t.scene==largest].N.iloc[0]):,}，完整路径D/C几何平均{geo(x.speedup):.4f}×，两轮每Ratio结果见上表和逐轮CSV。新调度是否改善由这组及迁移组实测判断，不以原65,536阈值为不可改条件。\n\n"
text+='## TUM/VECtor准备迁移验证\n\n'
transfer=read('TRANSFER_SUMMARY.json');assert transfer['passed']
tr=pd.read_csv(R/'TRANSFER_SUMMARY.csv')
text+=table(['冻结地图','N','阶段','参考p50/p95 ms','新增准备p50/p95 ms','配对加速','慢例'],[[x.scene,x.N,x.scope,
    f'{x.reference_p50:.4f}/{x.reference_p95:.4f}',f'{x.p50:.4f}/{x.p95:.4f}',f'{x.speedup:.4f}×',f'{x.slower_rounds}/{x.round_conditions}'] for x in tr.itertuples()])
text+=f"\n4组TUM/VECtor、Ratio 0.1–0.5、两轮6视角，共{transfer['samples']:,}条计时、{transfer['quality_checks']}次图像/身份检查。Selector均使用相同FP16执行，Compensator保持原64/64 Teacher CUDA，关闭学生预投影；只验证准备、相机和调度的迁移，不声称这些场景已具备同场景学生或同等Tensor补偿收益。平均额外PSNR损失{transfer['mean_psnr_loss']:+.8f}dB，最差单帧{transfer['worst_frame_loss']:.6f}dB。未重新训练、未复制checkpoint。逐Ratio和慢例均另存CSV。\n\n"

text+='## 质量和速度—质量Pareto\n\n'
rows=[]
old=json.loads((ROOT/'reports/RESULTS_SUMMARY.json').read_text())
for v in ['A','B','C','D']:
    speed=1. if v=='A' else item('A',v,'pipeline').speedup
    loss=0. if v=='A' else old['all_mean_psnr_loss'] if v=='C' else result['quality'][v]['mean_psnr_loss']
    worst=0. if v=='A' else old['worst_frame_loss'] if v=='C' else result['quality'][v]['worst_frame_loss']
    rows.append([v,f'{speed:.4f}×',f'{loss:+.8f}',f'{worst:.6f}'])
text+=table(['同一学生执行路径','完整加速相对A','平均额外PSNR损失dB','最差单帧损失dB'],rows)
text+=f"\nD/B重新评估16组×3轨迹视角×14条件，共{result['quality_rows']}图像对照；C复用相同输入与权重已验证672组，不重复渲染。PSNR/SSIM/LPIPS逐帧CSV均保留，原分辨率不变，不保存图片。不是全轨迹每帧质量保证。D相对A选点集合完全相同占{quality['same_ids_A_fraction']*100:.2f}%，相对C占{quality['same_ids_C_fraction']*100:.2f}%；严格K和候选计数在正式测试逐样本核验一致。\n\n"

text+='## Nsight：Kernel、同步和发布\n\n'
profile=pd.read_csv(R/'PROFILE_KERNELS.csv');runtime=pd.read_csv(R/'PROFILE_RUNTIME.csv');copies=pd.read_csv(R/'PROFILE_MEMCPY.csv');rows=[]
for label,g in profile.groupby('profile',sort=False):
    a=runtime[runtime.profile==label];sync=a[a.api.str.contains('Synchronize')]['count'].sum()
    c=copies[(copies.profile==label)&(copies.copy_kind==2)]
    rows.append([label,int(g['count'].sum()),f'{g.total_us.sum()/36:.3f}',int(sync),int(c['count'].sum()),int(c['bytes'].sum())])
text+=table(['36帧Profile','GPU Kernel数','累计GPU/帧 μs','Host同步API数','DtoH次数','DtoH总字节'],rows)
bp=profile[(profile.profile=='profile_B')&profile.stage.isin(['selector','compensator'])]
text+='\n历史B在office2 RGB-D、Ratio=0.3移动视角的实际Kernel分解：\n\n'
text+=table(['阶段','p50 μs','p95 μs'],[[x.stage,f'{x.p50_us:.3f}',f'{x.p95_us:.3f}'] for x in bp.itertuples()])
text+='\nProfile区间包含当前相机和完整engine准备，排除加载、冷启动；与正式完整延迟使用不同工具，Nsight扰动不作加速依据。GPU Kernel总时间、同步API耗时和墙钟不能直接相加。PROFILE_KERNELS.csv逐项给出View、候选、选择、动态投影、两网络与其余阶段，PROFILE_RUNTIME/PROFILE_MEMCPY记录Host路径；SASS确认最终两个网络的真实HMMA。\n\n'
text+='原SGEMM Kernel同时用于相机和Anchor投影，仅凭名称不能精确拆分，CSV将其标为camera_or_frustum_matrix，不伪造独立归因。其余未拆分的相机逆矩阵Kernel归入other，名称和原始Trace可复查。office2中C为每帧27个Kernel、D为10个；C每帧2次Stream同步、D为1次。room0的C每帧3次、D为1次。最终一次检查与读回保留，没有取消当前帧错误处理。\n\n'

text+='''## 失败候选与边界

- TF32 WMMA视锥3×6：确实实现并测量布局、补零、转换与输出；虽然比原小GEMM算子边界快，但普通CUDA融合更快，且9个开发视角累计改变8个Anchor Mask。禁止更改候选池，因此撤回生产候选；仅保留debug实现和证据。
- 逐点投影组合：相对逐Anchor方案84组逐位一致，微测略快，双轮完整延迟反而约慢0.8%–0.9%；恢复逐Anchor实现，失败源和二进制留档。
- 投影后寄存器MMA 1/2/4warp：内部开销与共享准备抵消，完整路径无稳定收益；撤回，没有因为HMMA占用率更高就保留。
- 回滚时Ninja沿用较新时间戳的旧Object，首次最终加载报未定义符号。已使受影响编译单元失效后重新编译，冻结的是重新构建并通过加载/边界的二进制；失败日志保留。
- 首个正式批次因显存CSV字段集合不一致退出，修正统计脚本后整批重跑。不是CUDA数值失败，未把不完整样本计入结论。外部CPU/GPU负载导致的批次整批排除并保留；不删除单个慢样本。

## 安全、完整性和剩余限制

'''
text+=f"边界检查{read('boundaries_final/SUMMARY.json')['checks']}项通过，memcheck和synccheck为0错误；新增Kernel定向racecheck为0竞争。前阶段整程序racecheck中两项ATen long-sum告警已在不加载新增扩展的原v31独立复现，本阶段不修改共享PyTorch，不能宣称整程序racecheck全部通过。定向检查涵盖新增CUDA准备、投影、Tensor网络和控制，过滤范围及日志保留。\n\n"
api=read('api_final/SUMMARY.json');assert api['passed'] and not api['tf32_matmul_enabled']
text+=f"公开from_checkpoint入口另通过{api['checks']}个Ratio的输出逐值对照，所有新开关可从公开参数设置。预计算时Matmul TF32开关为false。INPUT_CHECKPOINTS.json记录76份实际模型/config/PLY文件的路径、字节和SHA，模型只读且未复制。\n\n"
text+=f"{integrity['protected_files']}个受保护文件哈希变化0、缺失0；A和C二进制、C源码与原checkpoint完整。冻结D二进制SHA256 `{frozen['binary_sha256']}`。渲染图片0，训练/生产/共享conda未改，未commit/push/部署。资源、源码补丁、输入checkpoint路径与指纹、逐样本CSV、Sanitizer、代表Profile及绘图清单随交付保留。\n\n"
text+='''剩余成本主要是严格选择、View投影、当前帧属性发布及本来就很短的CUDA网络，另有Python/C++入口和最终必要检查。多Stream依赖流水线、极小MLP更多量化或与Renderer跨边界融合需另行实验；本轮没有降低分辨率、减少训练、改高斯预算、修改Renderer或声称在线SLAM加速。最快可接受方案以全部正式样本结果为准，所有不利场景和局部数值差异可从交付CSV复查。
'''
(ROOT/'TENSOR_CORE_AND_PREPARATION_9H_OPTIMIZATION_REPORT.md').write_text(text,encoding='utf-8')
(HERE/'TENSOR_CORE_AND_PREPARATION_9H_OPTIMIZATION_REPORT.md').write_text(text,encoding='utf-8')
print('JOINT_CHINESE_REPORT',len(text),flush=True)
