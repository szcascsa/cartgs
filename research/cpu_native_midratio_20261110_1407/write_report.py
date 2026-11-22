"""0.1–0.5实际实测报告；逐位语义与小量旋转误差分别披露。"""
from pathlib import Path
import json
import pandas as pd
import numpy as np
HERE=Path(__file__).resolve().parent
def read(name):return json.loads((HERE/'reports'/name).read_text())
def table(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in rows)
def geo(x):return float(np.exp(np.log(x).mean()))
r=read('RESULTS_SUMMARY.json');f=read('FROZEN_CANDIDATE.json');i=read('INTEGRITY_FINAL.json');assert r['passed'] and i['passed']
for name in ['selection_edges01','boundaries01','handoff_edges01','checkpoint01','asan01','ubsan01']:assert read('jobs/'+name+'.json')['returncode']==0,name
a=pd.read_csv(HERE/'reports/CPU_AGGREGATE.csv');s=pd.read_csv(HERE/'reports/CPU_PAIRED_RESULTS.csv');h=pd.read_csv(HERE/'reports/HANDOFF_PAIRED_RESULTS.csv');m=pd.read_csv(HERE/'reports/CPU_MEMORY.csv');q=pd.read_csv(HERE/'reports/QUALITY_ALL_FRAMES.csv');st=pd.read_csv(HERE/'reports/CPU_STAGES.csv')
primary=['local0.1','local0.2','local0.3','local0.4','local0.5']
text=f'''# CPU中Ratio深度优化：精确直方图选点与6点微核

## 实测结论

本轮以用户主要使用的Ratio 0.1–0.5为主验收区间。实际完成C++实现、编译、逐节点比较、双轮配对测速、跨场景渲染与Sanitizer。最终候选为精确直方图选点与6点Compensator微核，独立开关保留。仅用于训练完成后的冻结CPU推理，训练、生产默认、原checkpoint、原GPU Rasterizer不改。

对照是上一轮Small冻结二进制（小输出微核＋自适应顺序压缩），不重复计算历史CPU原生化或三槽交接收益。24组×2轮×5个主要Ratio等权几何平均CPU加速{r['primary_CPU_speedup']:.4f}×，相应延迟降低{100*(1-1/r['primary_CPU_speedup']):.2f}%；3个代表场景完整CPU准备＋H2D＋原GPU渲染加速{r['primary_handoff_speedup']:.4f}×，延迟降低{100*(1-1/r['primary_handoff_speedup']):.2f}%。这是样本聚合结果，不能解释为每个场景都更快。

'''
text+=table(['条件','旧CPU ms','新CPU ms','几何平均加速','延迟降低','慢例/48轮次'],[[x.condition,f'{x.median_baseline_ms:.4f}',f'{x.median_new_ms:.4f}',f'{x.geometric_speedup:.4f}×',f'{100*(1-1/x.geometric_speedup):+.2f}%',x.slower_rounds] for x in a[a.variant=='combined'].itertuples()])
text+='''
延迟列为48个场景轮次中位的中位，与逐配对加速的几何平均口径不同，不能直接相除。CSV保留每个场景、Ratio、N/M/K、中位、p95及最大值；所有慢例保留。

## 设计、代码与内存

### 精确直方图选点

原版本在Selector之后对全部M个Key执行串行nth_element。新版本在原Selector产生FP32概率时，同时统计各worker的256桶直方图，只统计资格有效点。桶号为min(255,int(score×256))；此分类不改变Score，不用桶号代替最终排名。

主线程累加桶计数，定位第K个点所在桶；并行收集该桶内的原始Score/UID Key，只对边界桶执行原Better比较器的nth_element。最差入选Key确定后，仍按原候选顺序压缩，保留严格K、Score降序及同分原行号优先的集合与最终原行号顺序。

边界Key的ID加1作为严格Better的排除cutoff；ID受原N<INT_MAX约束，不存在uint32溢出。所有同Score且ID更优的点保留。全局分位数路径不使用此优化，保留原阈值及ties行为。

M<2048时使用原路径；边界桶包含超过M/4点时，使用原nth_element，避免概率饱和导致无效额外扫描。判定只看工作量及分布，不按场景设参数；256桶、2048及1/4均在3个开发场景确定后冻结。

额外空间为N个Key（8N字节）与每线程256个int（6线程为6144字节）。边界列表复用固定容量，收集后按worker区间顺序memmove，不做无界缓存。每帧概率与直方图重新计算。

### 6点×16输出微核

Compensator主干和四Head隐藏层继续使用独立原权重；一次处理6个Gaussian，增加权重向量复用，保留每个输出j=0…D−1的FMA累加顺序。SSE小输出层同步支持5/6行尾部。网络深度、宽度、激活、四Head输出和物理解码公式均不变。Selector仍使用原4点分块。

6点块使用12条AVX累加链，比4点块的8条更多；是否获益受寄存器压力、输入规模和调度影响，由完整测试决定，未把理论吞吐当作实测收益。未使用FP16、INT8、TF32或fast-math，也没有量化Ratio、缓存旧帧输出或在GPU执行网络。

Pool与上一轮源码逐字一致。NativeHandoff仅改C++类型名以便同进程加载，仍是原有三份有界pinned Host/Device槽、ready/done Event及代际租约。

## 数值差异定位与验收

tune01首个全属性逐位门槛未通过：最大差2.98e−8，未直接当作通过。随后PROBE_NODES.csv逐节点比较发现，Selector概率、身份、MLP原始残差以及XYZ/Scale/Opacity/DC/SH均逐位一致；差异仅落在最终Rotation解码范围，开发探针最大5.96e−8。即使关闭新功能的control也有该差异，说明它与编译布局改变有关，不能归因于直方图选择或网络累加顺序改变；尚未用机器指令定位到单个旋转子表达式。

根据本轮既有授权“少量误差、速度优先”，明确采用以下合同：Score、UID、所有非Rotation属性逐位一致；Rotation最大绝对差≤1e−6；真实GPU渲染独立检查GT PSNR平均损失≤0.05dB、每条件平均损失≤0.2dB，披露最差帧。该误差来自CPU旋转解码，不是修改了训练或GPU Rasterizer。不能将此版本称为相对上一版全属性逐位一致。

'''
text+=f'''正式CPU {r['CPU_checks']:,}组检查，非Rotation与Score逐位通过，Rotation实测最大绝对差{r['CPU_rotation_max_abs_error']:.9g}；全Rotation逐位相同={r['CPU_rotation_bitwise']}。GPU质量共{r['quality_rows']}组，RGB全部逐位相同={r['quality_RGB_bitwise']}，最大RGB绝对差{r['quality_max_rgb_error']:.9g}，平均GT PSNR损失{r['mean_psnr_loss']:+.8f}dB，最差单帧{r['worst_psnr_loss']:.8f}dB。radii及选点身份检查通过。逐条件SSIM/LPIPS见质量CSV，不只报告PSNR。

'''
text+=table(['最差质量项','帧','条件','PSNR损失dB','最大RGB差'],[[x.scene,x.frame,x.condition,f'{x.psnr_loss:.8f}',f'{x.max_rgb_error:.8g}'] for x in q.nlargest(8,'psnr_loss').itertuples()])
text+='''
质量对照为本轮开始的Small CPU后端，不是最初GPU PyTorch；原有CPU近似差异另见历史报告。仅验证保存相机样本，不声称全轨迹或所有未测姿态均满足同样误差。

## 配对协议和独立归因

CPU：i5-9600K，6物理核、AVX2/FMA，原生池6线程，Torch辅助1线程；双方每帧重新准备，同一checkpoint/固定键/相机。只使用office2 RGB-D、room0 RGB-D、TUM fr1_xyz RGB-D开发，两轮正反版本顺序调优后冻结，其余21组保留验证。正式共Replica16、TUM4、VECtor4，每组3连续＋3跳转相机；两轮场景反序、版本轮换，每条件每版本每轮36计时样本。未在同一CPU正式计时中共驻留多个线程池。

测试15条件：local .1/.2/.3/.4/.5、插值 .15/.25/.45、额外 .01/.7、fixed0/1000/5000、global.1及Full。分段统计在正式计时外进行。CPU时间包括完整准备/网络/解码/输出，不含相机矩阵构造、I/O、显示、上传或光栅化。

外部CPU/GPU活动由每秒进程监测记录；正式有效轮次不含检测到的活跃外部进程。该监测不能排除短时OS抢占、频率变化与中断。长尾和慢例均保留，不以未检测到外部负载证明每个样本无扰动。

'''
text+=table(['条件','仅直方图','仅6点微核','合用'],[[c]+[f'{a[(a.condition==c)&(a.variant==v)].iloc[0].geometric_speedup:.4f}×' for v in ['histogram','six','combined']] for c in primary])
text+=f'\nCPU正式样本{r["CPU_samples"]:,}个。主要Ratio共{r["primary_CPU_slower_rounds"]}/240个场景轮次条件更慢；全条件慢例{r["CPU_slower_rounds"]}项，详见CPU_ALL_SLOWER_CASES.csv，未删除Full或低Ratio反例。\n\n'
distribution=pd.read_csv(HERE/'reports/CPU_SELECTION_DISTRIBUTIONS.csv');active=distribution[distribution.histogram_active];assert read('DISTRIBUTION_SUMMARY.json')['passed']
text+=f'\n计时外另检查{len(distribution)}个场景/相机/Ratio分布：满足直方图工作量条件{len(active)}项，饱和回退{int(active.saturation_fallback.sum())}项；边界桶规模占M中位{active.boundary_fraction.median()*100:.3f}%，最大{active.boundary_fraction.max()*100:.3f}%。分布CSV保留Score=0/1的数量，用于解释退化和回退，没有据此再调参数。\n\n'
text+='## 每场景主要Ratio\n\n每格为两轮中位旧→新CPU ms及几何平均加速。\n\n'
rows=[]
for scene in s.scene.unique():
    row=[scene]
    for c in primary:
        g=s[(s.scene==scene)&(s.condition==c)&(s.variant=='combined')];row.append(f'{g.baseline_ms.median():.3f}→{g.ms.median():.3f} ({geo(g.speedup):.3f}×)')
    rows.append(row)
text+=table(['场景/模式','10%','20%','30%','40%','50%'],rows)
text+='\n## 完整准备＋上传＋原渲染器\n\n双方均为原生交接，本帧51个相机FP32及属性同包上传。每样本同步完成，包含CPU准备、H2D、依赖、原GPU光栅化及最终同步，不含相机矩阵构造/I/O/显示。两轮3场景×3跳转相机×13条件，版本轮换；不把流水线吞吐当单帧延迟。\n\n'
text+=table(['场景','Ratio','旧完整ms','新完整ms','加速','旧较大p95','新较大p95'],[[scene,c[5:],f'{g.baseline_ms.median():.4f}',f'{g.ms.median():.4f}',f'{geo(g.speedup):.4f}×',f'{g.baseline_p95.max():.4f}',f'{g.p95.max():.4f}'] for (scene,c),g in h[h.condition.isin(primary)].groupby(['scene','condition'])])
text+=f'\n完整路径样本{r["handoff_samples"]:,}个，主要Ratio {r["primary_handoff_slower_rounds"]}/30项中位更慢；所有13条件与慢例均见HANDOFF_PAIRED_RESULTS.csv。仅三个代表场景验证了完整路径，不将CPU的24组结论扩展为24组GPU端到端证据。\n\n'
text+='## 分段热点\n\n单位ms，各阶段独立采样中位不能直接相加。\n\n'
text+=table(['条件','版本','候选/视角','额外视角','Selector','选点','补偿及发布'],[[c,v]+[f'{st[(st.condition==c)&(st.variant==v)&(st.stage==n)].ms.median():.5f}' for n in range(5)] for c in ['local0.1','local0.3','local0.5'] for v in ['baseline','combined']])
text+=f'''
## 边界、安全与空间成本

- 同分、离散分数、资格all/striped/one/none、空候选、K边界、严格局部K及全局ties：{read('selection_edges01/SUMMARY.json')['checks']}组通过。
- 组合候选的常规CPU边界{read('boundaries01/SUMMARY.json')['checks']}项通过，含caller Buffer哨兵、独立输出、源模型隔离、非法输入拒绝及错误恢复、同实例并发串行化。
- 三类checkpoint全CPU恢复{read('checkpoint01/SUMMARY.json')['checks']}组通过，保留原GPU固定随机键，CUDA未初始化。
- 原生交接动态K/错误恢复/不同消费者Stream/相机属性传输{read('handoff_edges01/SUMMARY.json')['checks']}项通过；原Rasterizer继续默认Stream渲染。
- ASan+UBSan正常28项、独立UBSan完整33项通过。ASan与当前Torch主动异常的兼容限制已在前轮独立定位，本轮明确排除5项主动异常；泄漏检测关闭，不声称覆盖GPU内存或证明无数据竞争。
- Workspace主要逻辑估算{m.workspace_bytes.min()/2**20:.2f}–{m.workspace_bytes.max()/2**20:.2f}MiB；不含所有权重、分配器和线程栈，不是进程RSS。新增8N+6144字节直方图Workspace，6点块只增加有界线程栈激活。三槽交接容量不变。
- 模型、Feature、Anchor、拓扑变化必须重建冻结快照；相机变化不复用旧结果。仅验证当前Teacher与x86 AVX2，Student/ARM未独立验收。

## 失败项与剩余空间

tune01的全属性逐位门槛失败、tune02/03的开发慢例和control结果均保留。控制版本也出现旋转微差，不能把编译布局变化隐藏为“全属性零变化”。概率饱和时直方图需要回退，其成本仍需计入；没有用Ratio阈值或逐场景专用配置隐藏反例。

本轮按实际主区间优先完成选点与补偿两项。Anchor成员索引/候选融合没有进入最终候选：现有候选阶段同时包含视角几何，简单改成按Anchor枚举还需恢复原行序，未有证据表明能覆盖排序或合并成本。后续应按剩余分段占比决定是否研究稀疏行块索引、跨Anchor视角SIMD，或继续优化Compensator权重访问；不再仅凭低Ratio的重要性决定优先级。

Full为整图属性打包诊断，原GPU Full路径没有改动。所有Full实测慢例在CSV保留，未宣称逐场景Full性能不退化。

## 完整性与交付

基线：`{f['baseline']}`；其二进制SHA256为`42d4cc3fc34d656297955e420db0b62be8db9a7a6f62cbb1cc4acc91fbb05863`。

最终二进制：`{f['binary']}`；SHA256：`{f['binary_sha256']}`。

原生产、训练、checkpoint及历史关键证据{i['protected_files']}个保护文件哈希变化0、缺失0；冻结源码/二进制未变，无活跃本轮任务，无渲染图片。未commit、push、部署，未修改共享conda。代码、独立开关、完整结果/原始样本、失败现场、曲线与数据哈希均在本研究目录；仅清理本轮可重建Sanitizer对象并保留审计。
'''
(HERE/'CPU_MIDRATIO_DEEP_OPTIMIZATION_REPORT.md').write_text(text,encoding='utf-8');print('MIDRATIO_REPORT',len(text))
