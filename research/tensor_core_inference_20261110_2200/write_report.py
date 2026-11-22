"""中文实测报告：先结论，再实现、失败候选、完整验证和所有慢例。"""
from pathlib import Path
import json
import numpy as np,pandas as pd
HERE=Path(__file__).resolve().parent;R=HERE/'reports'
def read(n):return json.loads((R/n).read_text())
def geo(x):return float(np.exp(np.log(x).mean()))
def table(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,r))+' |\n' for r in rows)
r=read('RESULTS_SUMMARY.json');f=read('FROZEN_CANDIDATE.json');integrity=read('INTEGRITY_FINAL.json');assert r['passed'] and integrity['passed']
for name in ['boundaries02','memcheck01','synccheck01','racecheck_tensor02']:assert read('jobs/'+name+'.json')['returncode']==0,name
assert read('SANITIZER_ATTRIBUTION.json')['attribution_passed']
d=pd.read_csv(R/'PAIRED_RESULTS.csv');agg=pd.read_csv(R/'AGGREGATE.csv');q=pd.read_csv(R/'QUALITY_ALL_FRAMES.csv');co=pd.read_csv(R/'COMPONENT_RESULTS.csv');primary=['local0.1','local0.2','local0.3','local0.4','local0.5']
speed=r['primary']['both'];text=f'''# Selector与蒸馏Compensator的Tensor Core推理优化

## 实测结论

已完成独立CUDA实现、多版编译迭代、真实checkpoint图像检查及冻结后的Replica16组双轮配对。对照是原v31融合FP32 CUDA版本，使用同一个32/32学生，所有收益均为本轮新增收益，不混入Teacher→Student、CPU或Renderer优化。

Ratio 0.1–0.5：Selector组件加速{r['components']['selector']:.4f}×，Compensator组件加速{r['components']['compensator']:.4f}×；完整准备加速{speed['prepare']['speedup']:.4f}×（延迟降低{100*(1-1/speed['prepare']['speedup']):.2f}%），包含当前相机准备和原G渲染的完整路径加速{speed['pipeline']['speedup']:.4f}×（延迟降低{100*(1-1/speed['pipeline']['speedup']):.2f}%）。组件是3组开发场景的固定输入CUDA Event，完整路径是16组真实移动视角，二者不能等同。

最终组合：Selector用寄存器PTX MMA，{f['warps']} warp计算64行；学生Compensator用共享内存WMMA，{f['comp_warps']}个矩阵计算warp处理{f['comp_warps']*16}行，每16点以128线程协作输入与发布。均采用FP16乘法、FP32累加；物理解码、Master和概率阶段继续FP32。两网络都实际包含HMMA指令，证据见TENSOR_SASS_EVIDENCE.json，不是仅设置allow_tf32。

全部条件双模块平均额外PSNR损失{r['all_mean_psnr_loss']:+.8f}dB；主要区间{r['primary_mean_psnr_loss']:+.8f}dB。最差单帧额外损失{r['worst_frame_loss']:.6f}dB，最差场景条件均值{r['worst_condition_mean_loss']:.6f}dB。相对同一学生满足预设平均≤0.05dB、场景条件均值≤0.2dB门限。它不是FP32逐位等价版本，也没有消除学生原有的蒸馏质量损失。

'''
rows=[]
for c in primary:
    a=agg[(agg.variant=='both')&(agg.condition==c)];p=a[a.scope=='prepare'].iloc[0];h=a[a.scope=='pipeline'].iloc[0]
    rows.append([c[5:],f'{p.baseline_ms:.4f}→{p.optimized_ms:.4f}',f'{p.speedup:.4f}×',f'{h.baseline_ms:.4f}→{h.optimized_ms:.4f}',f'{h.speedup:.4f}×',f'{int(p.slower_rounds)}/{int(p.scene_rounds)}',f'{int(h.slower_rounds)}/{int(h.scene_rounds)}'])
text+=table(['Ratio','完整准备ms CUDA→TC','准备加速','准备＋渲染ms CUDA→TC','完整加速','准备慢例','完整慢例'],rows)
text+='\n延迟列为场景轮次配对中位数的中位；加速为配对等权几何平均，两列不能直接相除。保留逐样本、p95、最大值与所有退化项，不以平均收益宣称逐场景均加速。\n\n'
text+='''## 保留的实现

1. Selector保留冻结静态物理前缀，不恢复已删除的10D重复计算。动态Ratio/View/Mode小投影与32→32隐藏层走Tensor Core；最终32→2也在寄存器MMA中完成，Gumbel固定噪声和softmax概率仍FP32。C accumulator的两行两列分布直接打包为下一层A operand，避免隐藏层落共享内存；ReLU与原Bias在FP32上处理后转换FP16。权重加载时预转置为列主序，使每线程两个half连续读取。
2. Compensator将Offset3、Residual48、Ratio3和View8布局成64列，剩余列补零。Base/Residual查询仍按原顺序执行FP32乘法和加法，然后转FP16。原51→32输入层、Ratio/View线性投影合并到同一次MMA；实数表达式等价，但加法分组和舍入改变，因此按实测质量验收，不声称逐位相同。32→32和四个直接Head32→14分别MMA，Head权重按原输出顺序补零到16列，未合并参数、未改变网络容量。
3. Compensator根据实测采用WMMA而非寄存器链。输入和隐藏激活复用一个有界共享Buffer，浮点输出行跨度36，减少后处理的bank冲突。控制码只在真实有效块加载；矩阵计算warp与加载/发布线程数量分开，每16点对应128线程，避免外围工作只有一个warp。
4. 继续调用原物理解码函数。XYZ参考尺度、物理Scale的Softplus及exp(log)、四元数乘法和三次归一化、Opacity、DC、SH与means2D语义沿用原实现。SH改为同Kernel内的连续线程复制，复制原FP32值，没有改变SH阶数或精度。
5. 准备、视锥、候选、eligibility、严格K、UID、Radix稳定选择、有限值检查与Stream顺序均复用原流程。动态N/K仍留在GPU，容量不足和原host大图路径沿用原调度，未按测试场景硬编码Ratio/K阈值。

网络权重、架构、训练数据与优化器均未修改。每帧重新计算相机、可见候选、Feature查询和网络；没有复用同帧输出、视锥Mask或旧Ratio预测。只在冻结实例加载时预打包有限权重，切换Master/Feature/网络/拓扑须重建实例。

## 精度、选择与质量

预设允许少量推理误差。TF32、FP16两种乘法均实际编译和验证，最终以开发集真实延迟选择FP16。全部累加为FP32；FP16并不意味着整个地图、解码或Renderer转half。Selector的logits及排序可能微变，最后概率仍FP32不能恢复此前隐藏层舍入；因此严格K保持一致，选点身份变化单独量化。

'''
text+=f'''双模块{int((q.variant=='both').sum())}组渲染中，选点集合完全相同占{r['same_ids_fraction']*100:.2f}%，最低Jaccard为{r['minimum_jaccard']:.8f}；仅Compensator TC的身份始终要求与原CUDA相同。两图最低相互PSNR为{r['minimum_image_psnr']:.5f}dB，最大RGB差{r['max_rgb_error']:.7g}，radii不同{r['radii_mismatches']}项。最大像素差与平均PSNR是不同量，局部差异没有隐藏。不存渲染图片。

质量为每组3个完整轨迹视角、14条件、3种模块组合，共{r['quality_cases']}组；不是全轨迹逐帧质量保证。GT指标包含PSNR/SSIM/LPIPS，原图分辨率不变。质量数据、逐节点属性差异和模型SHA/指纹另存CSV/JSON。

'''
text+=table(['最差额外质量项','条件','帧','损失dB','Jaccard'],[[x.scene,x.condition,x.frame,f'{x.psnr_loss:.7f}',f'{x.jaccard:.8f}'] for x in q[q.variant=='both'].nlargest(8,'psnr_loss').itertuples()])
text+='\n## 独立单模块归因\n\n'
text+=table(['启用模块','准备加速','准备＋渲染加速','准备慢例','完整慢例'],[[v,f'{r["primary"][v]["prepare"]["speedup"]:.4f}×',f'{r["primary"][v]["pipeline"]["speedup"]:.4f}×',f'{r["primary"][v]["prepare"]["slower_rounds"]}/{r["primary"][v]["prepare"]["scene_rounds"]}',f'{r["primary"][v]["pipeline"]["slower_rounds"]}/{r["primary"][v]["pipeline"]["scene_rounds"]}'] for v in ['selector','compensator','both']])
text+='\n## 每场景每Ratio完整路径\n\n以下每格为两轮原CUDA→TC中位毫秒及配对几何平均加速。\n\n'
rows=[]
for scene in sorted(d.scene.unique()):
    row=[scene,int(d[d.scene==scene].N.iloc[0])]
    for c in primary:
        g=d[(d.scene==scene)&(d.variant=='both')&(d.condition==c)&(d.scope=='pipeline')];row.append(f'{g.baseline_ms.median():.4f}→{g.ms.median():.4f} ({geo(g.speedup):.3f}×)')
    rows.append(row)
text+=table(['场景/模式','地图N','10%','20%','30%','40%','50%'],rows)
large=d[(d.scene=='Replica/room0/rgbd')&(d.variant=='both')&d.condition.isin(primary)]
text+=f'''\n最大地图room0 RGB-D有{int(large.N.iloc[0]):,}点，Ratio 0.1–0.5完整准备加速{geo(large[large.scope=='prepare'].speedup):.4f}×、完整路径{geo(large[large.scope=='pipeline'].speedup):.4f}×，没有整体大图退化。其0.4准备在两轮分别为{', '.join(f'{x:.4f}×' for x in large[(large.condition=='local0.4')&(large.scope=='prepare')].sort_values('round').speedup)}，存在局部慢例且两轮方向不一致；完整路径两轮几何平均{geo(large[(large.condition=='local0.4')&(large.scope=='pipeline')].speedup):.4f}×。这些结果不足以证明所有大图/Ratio都加速，也不支持为了平均收益进一步叠加未经实测的调度。当前保持最简单的既有调度；未来可以独立改变，不受相同调度限制。\n\n'''
profile=pd.read_csv(R/'FINAL_PROFILE_KERNELS.csv')
text+='## Nsight与实际Tensor指令\n\n'
rows=[]
for label,g in profile.groupby('profile',sort=False):
    rows.append([label,int(g['count'].sum()),f'{g.total_us.sum()/36:.3f}',f'{g[g.kernel.str.contains("selector")].total_us.sum()/36:.3f}',f'{g[g.kernel.str.contains("compensation")].total_us.sum()/36:.3f}'])
text+=table(['Profile（每个36次prepare）','GPU Kernel数','累计GPU时间/次 μs','Selector Kernel/次 μs','Compensator Kernel/次 μs'],rows)
text+='\nProfile只统计TC_MEASURE区间，含连续和跳转视角；正式加速仍使用独立非Profiler双轮。profile_base_full和profile_final对应office2 RGB-D，另有room0大图用于核实host路径实际调用Tensor核。Profile不包含R相机准备和Renderer，不能与完整pipeline墙钟直接相减。新路径没有减少Kernel发射数，收益来自两网络内部执行；余下候选、选择、打包、同步和调用成本仍然存在。SASS为最终二进制的真实HMMA，不以开启后端标志代替指令证据。\n\n'
text+=f'''
## 正式测速协议和Full

RTX3090、CUDA11.8、Torch2.3.1，本机硬件范围。3组开发、13组保留；候选冻结后不逐场景调参。正式采用的轮次见MEASUREMENT_RUNS.json，干扰轮次整轮排除并保留原记录，不删除单个慢样本。全部Replica16场景/模式、每组3连续＋3跳转轨迹视角、14条件、4版本、6次重复、2轮，第二轮场景和视角反序，版本顺序轮换，共{r['samples']:,}条计时与{r['checks']:,}条K/身份检查。使用同一原G Renderer和R相机准备，不把历史Renderer/相机优化记为本轮收益。

prepare墙钟包含当前相机与网络准备及最终同步，pipeline再包含原G渲染；模型加载、Graph首次编译、GT读取、编码/显示、在线Tracking/Mapping不在热态计时内。每帧重新准备，固定输入组件Graph仅用于隔离网络成本，不当成移动视角完整速度。

Full直接使用原冻结Master属性，不调用两个网络。Full配对准备加速{r['full_prepare']:.4f}×，完整路径{r['full_pipeline']:.4f}×；差异是相机/调用/系统测量波动，不能记为Tensor Core收益，也没有为Full切换新的算法路径。

formal02/formal03的长轮受到Xorg活动，整轮排除；首个C1候选复用通过的formal04第一轮，第二轮采用短批；最终C2候选配置变化后独立重跑双轮短批，原C1结果保留。为复用formal04，曾停止本轮短批调度器并让已启动子批正常结束，身份核验和理由在BATCH_RESTART_AUDIT.json；额外短批保留但不计入结论。所有采用的正式性能轮次监测通过，无外部CPU/GPU活动采样；采样不能完全排除瞬时OS和GPU时钟波动。所有慢例保留ALL_SLOWER_CASES.csv。components03和tune04有Xorg/CPU活动，留作诊断，不作为正式性能证据。低Ratio/固定预算/Full全部记录，不挑有利项。

## 迭代和失败候选

'''
rows=[]
for run in ['tune01','tune02','tune03','tune04','tune05','tune06','tune07']:
    t=pd.read_csv(R/run/'PAIRED_SUMMARY.csv');best=t.iloc[0];job=read('jobs/'+run+'.json')
    rows.append([run,best.variant,f'{best.speedup:.4f}×',bool(job['measurement_valid']),int(best.slower_conditions)])
text+=table(['开发轮次','当轮最快候选','准备加速','无外部负载采样','慢例/15'],rows)
text+='''
- V1直接WMMA：实际SASS含HMMA，但代表输入Selector约15→33μs、Compensator约46→86μs，完整准备变慢。算术吞吐增加不等于整网变快；共享内存、外围串行与不规则访存主导。
- V2并入条件投影、共享scratch和SH连续复制：Selector接近持平，Compensator仍慢。V3增加外围线程后FP16准备约1.017×。V4隐藏层寄存器链提升Selector，但Compensator寄存器输入查询/解码负担使其退化。V5合并读取和列主序权重有所恢复，仍不如较简单的WMMA补偿。
- V6将两网络的最佳执行形式分开，Selector寄存器MMA、Compensator WMMA，独立warp调优。首次冻结开发集准备约1.037×，但formal01在room0 RGB-D检查时发现>65,536点的host计数路径仍调用旧网络，主动停止；已完成的记录不作为正式双轮。修复host两网络路由并共享原Selector finite status后，仅在原3组开发集复测tune07，参数不根据其余13组调整；原冻结文件存FROZEN_CANDIDATE_V6_REJECTED.json，重新冻结后完整重跑quality03、MEASUREMENT_RUNS.json列明的双轮。tune07中补偿warp=1和2仅相差约0.1%整准备速度，首次选1；16组有效双轮完整准备约1.014×。随后在原3组开发组件测试中发现补偿warp=2约1.066×而warp=1约0.966×，因此只依据开发集改为2，重新做全部双轮速度；改块大小对quality03全部672种图像输入逐位相同，BLOCK_PARITY核验通过后复用既有2016组质量，不重复渲染。原1-warp完整证据保存C1_EVALUATION。
- TF32路径保留独立开关，实际多次慢于FP16；没有因为存储是FP32就称数值不变。8-warp补偿超过48KiB静态共享内存上限，候选撤回，保留编译错误证据。首次编译的TF32转换namespace错误已修复；未改共享工具链。
- quality01因本轮profile.py遮蔽标准库profile而在导入指标阶段退出；改名profile_workload.py后quality02完成；补齐大地图host路径后以quality03作为最终质量验收，未修改冻结候选或门限。
- 早期cudaProfilerStart/Stop捕获触发当前Nsight/PyTorch运行时异常，另一次NVTX capture未产生活动；保留日志。最终完整capture+TC_MEASURE NVTX区间及graph node trace可用，仅截取目标区间统计，不使用Profiler墙钟作正式加速。

## 内存、安全、范围和复现

'''
text+=f'''冻结派生权重额外{r['tensor_weight_bytes']:,}字节（{r['tensor_weight_bytes']/1024:.1f}KiB）。新Kernel只用有界共享内存/寄存器，不增加每Gaussian历史缓存；Graph和输出容量沿用原有限方案。内存数据是派生权重和静态资源，非声称整个进程RSS或所有GPU临时内存均为此值。

常规边界{read('boundaries02/SUMMARY.json')['checks']}项通过，覆盖空集合、K尾部、未见Ratio、Buffer独立性、Stream及开关恢复；关闭开关与v31的选点和属性逐位相同。Compute Sanitizer整程序memcheck、synccheck均为0错误。整程序racecheck报告2项ATen long-sum归约竞争（Read PC 0x9280、Write PC 0x9370，各6144 hazards），**整程序racecheck未通过**。独立原v31、不加载Tensor扩展的对照复现完全相同的2项报告，日志逐段匹配；不能据此断言它们是误报，也未修改共享PyTorch。随后仅过滤检查全部新增Tensor Kernel（packTensorWeights、selectorTensor、compensationTensor）为0竞争。SANITIZER_ATTRIBUTION.json及失败原日志均保留，没有把原版告警改写为整体安全通过。Sanitizer只覆盖所测入口和输入，不代替所有GPU环境/线程安全证明。

部署接口为独立source/deployment_runtime.py中的DeploymentRenderer，研究默认selector=2、compensator=2、selector warp=4、compensator warp=2。tensor_selector=0或tensor_compensator=0可单独关闭；两者0回退原融合FP32实现。旧生产入口仍原样保留。TF32=1仅供已记录候选比较。

旧checkpoint仍绑定原Teacher/Master/Anchor Feature/行序指纹。网络更新或拓扑改变必须重建实例；本研究不修改训练。超过65,536点时仍沿用原host计数和同步流程，两网络已接入相同Tensor Kernel，保留shared finite status；其余地图仍GPU动态N/K。首轮为控制变量沿用原地图调度阈值；用户已授权后续可调整调度，是否修改取决于大点数实际退化证据，不把原阈值作为不可变约束。

最终二进制SHA256：`{f['binary_sha256']}`。
{integrity['protected_files']}个受保护文件哈希变化0、缺失0，冻结源码与二进制一致。未修改生产、原checkpoint、训练或共享conda，未commit/push/部署，渲染图片0。

## 下一步

优先依据正式慢例的候选数量与网络组件占比研究共同数据布局及候选准备。若只剩几微秒的网络节省，继续增加MMA复杂度未必能显著改善整帧。更激进的权重/输入量化或跨帧流水线属于独立方案，需重新验收质量/单帧延迟；不能用缓存同帧或理论Tensor吞吐替代真实移动视角测速。

参考实现规范：[CUDA11.8 WMMA/PTX](https://docs.nvidia.com/cuda/archive/11.8.0/parallel-thread-execution/index.html)、[CUDA11.8 Compute Sanitizer](https://docs.nvidia.com/cuda/archive/11.8.0/compute-sanitizer/index.html)。实际速度结论均来自本轮CSV和原始日志。
'''
(HERE/'TENSOR_CORE_INFERENCE_OPTIMIZATION_REPORT.md').write_text(text,encoding='utf-8');print('TENSOR_REPORT',len(text))
