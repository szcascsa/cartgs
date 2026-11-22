"""中文报告：保留所有场景与退化项，不把历史收益重复计入。"""
from pathlib import Path
import json
import pandas as pd
import numpy as np
HERE=Path(__file__).resolve().parent

def read(name):return json.loads((HERE/'reports'/name).read_text())
def table(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in rows)
def geo(x):return float(np.exp(np.log(x).mean()))

r=read('RESULTS_SUMMARY.json');f=read('FROZEN_CANDIDATE.json');integrity=read('INTEGRITY_FINAL.json')
assert r['passed'] and integrity['passed']
for run in ['boundaries01','selection_edges01','checkpoint01','handoff_edges01','asan01','ubsan01']:
    assert read('jobs/'+run+'.json')['returncode']==0,run
a=pd.read_csv(HERE/'reports/CPU_AGGREGATE.csv');s=pd.read_csv(HERE/'reports/CPU_PAIRED_RESULTS.csv');h=pd.read_csv(HERE/'reports/HANDOFF_PAIRED_RESULTS.csv')
q=pd.read_csv(HERE/'reports/QUALITY_ALL_FRAMES.csv');st=pd.read_csv(HERE/'reports/CPU_STAGES.csv');mem=pd.read_csv(HERE/'reports/CPU_MEMORY.csv');hm=pd.read_csv(HERE/'reports/HANDOFF_MEMORY.csv')
text='''# CPU 深度优化：小输出微核、顺序压缩与调度长尾

## 结论与适用范围

本轮实际实现、编译和验证了两个精确优化：Selector 32→2及Compensator 64→3/4的小输出微核；保留选点身份和行序的自适应顺序压缩。最终候选没有新增数值误差。线程等待策略和分段累加候选未获得稳定收益，已撤回。

收益集中在较大输出K或全局选点；低Ratio 1%–10%没有稳定的普遍加速。不能将本轮结论写成“所有预算更快”。当前只是独立研究后端，没有替换生产默认，也没有再次计入此前原生化、AVX2或CPU→GPU交接的收益。

本轮真实对照是上一轮`cpu_native_deep_20261110_1222`的冻结二进制，双方均用原生准备与三槽交接。表中为24组×2轮等权几何平均加速，延迟列为48个场景轮次中位的中位；这两种聚合不能直接互相相除。

'''
text+=table(['条件','上一版ms','本轮ms','几何平均加速','延迟降低','范围','慢例/48轮次'],[
    [x.condition,f'{x.median_baseline_ms:.4f}',f'{x.median_new_ms:.4f}',f'{x.geometric_speedup:.3f}×',f'{100*(1-1/x.geometric_speedup):+.2f}%',f'{x.min_speedup:.3f}–{x.max_speedup:.3f}',x.slower_rounds]
    for x in a[a.variant=='combined'].itertuples()])
text+=f'''
两轮正式CPU计时样本{r['CPU_samples']:,}个，身份/属性检查{r['CPU_checks']:,}组，全部逐位相同；CPU测量结束时CUDA未初始化。24组含Replica16、TUM4、VECtor4；开发仅用Replica office2 RGB-D、room0 RGB-D、TUM fr1_xyz RGB-D，其余21组保留验证。候选在正式测试前冻结，未根据保留集调参数。

## 实现及等价性

### 小输出微核

上一轮面向8/16输出的AVX2微核，在2/3/4输出时仍计算无消费者的高通道。本轮以SSE128/FMA仅计算低4通道，Inputs=32/64、Outputs=2/3/4、Rows=1…4编译期展开。每个有效输出仍严格按j=0…D−1执行FMA，再加相同Bias及ReLU；没有改变求和分组、FMA链或激活顺序。只写有效2/3/4个FP32，避免越界。

只涉及Selector最终2输出层与Compensator四个Head的最终3/3/4/4输出层。主干及四Head隐藏层、独立权重、Feature共享、模型结构与物理解码均不变。没有FP16、INT8或fast-math。

### 顺序压缩

- 局部严格K保留原`nth_element`选择。Key仍为Score降序、原Gaussian行号升序，第K+1个Key作为cutoff；从原候选行序扫描，只保留`Better(key,cutoff)`，因此集合、K和最终原行号顺序不变。
- 全局分位数仍使用原阈值及`score >= threshold`，同分保留数量不变，没有把它偷换为严格K。
- 各worker只写自己区间，主线程按区间顺序`memmove`向前压缩；复用既有selected容量，无新增N规模Buffer，不依赖旧帧缓存。
- 低K全候选扫描可能比K项排序更慢。最终`ordered_mode=3`在局部仅当`K*log2(max(2,K)) > M+4096`时使用压缩，否则使用原排序。该常量只在三个开发场景确定，不按场景或最终测试集设参数。
- Full没有调用Selector或Compensator；本轮Full诊断是原整图属性打包，不是Rasterizer速度测试。

### 独立归因

'''
text+=table(['条件','仅小输出','仅自适应压缩','合用'],[
    [condition]+[f'{a[(a.condition==condition)&(a.variant==v)].iloc[0].geometric_speedup:.4f}×' for v in ['small','adaptive','combined']]
    for condition in ['local0.01','local0.05','local0.1','local0.3','local0.7','fixed1000','fixed5000','global0.1','full']])
text+='''
小输出层在整个网络中占比有限，独立端到端收益较小。顺序压缩主要节约选择后的排序成本，因此高K更有效。低Ratio受候选/视角/Selector等固定成本和OS调度影响，减少几个输出通道不会自动带来毫秒级收益。

## 测量协议

硬件为Intel i5-9600K，6核6线程、AVX2/FMA；每次只驻留一个原生6线程池，Torch辅助1线程。相同checkpoint/权重/固定随机键/背景/相机，双方每帧重新准备。正式CPU测试包含每场景3个连续相机和3个跳转相机，两轮反转场景并轮换版本顺序。每条件每轮每版本3块×12样本，块前6次切换视角预热。包含11种已见/未见local Ratio、fixed0/1000/5000、global1%/10%和Full，共17条件。

CPU时间包含候选、视角、Selector、选择、Compensator、物理解码和输出分配打包；不含磁盘读取、相机矩阵构造、显示或GPU渲染。分段计时另行采集，不放在正式计时区间。监测保留每秒外部CPU/GPU活动，两轮均未检测到外部活跃负载。该监测不能排除OS调度、中断或时钟波动，不等同于每个样本都无扰动。

## 每场景CPU准备

每格为上一版→本轮的两轮中位ms及两轮几何平均加速。全部Ratio、p95、最大延迟、N/M/K见CSV。

'''
rows=[]
for scene in s.scene.unique():
    row=[scene]
    for condition in ['local0.01','local0.1','local0.7']:
        g=s[(s.scene==scene)&(s.condition==condition)&(s.variant=='combined')];row.append(f'{g.baseline_ms.median():.4f}→{g.ms.median():.4f} ({geo(g.speedup):.3f}×)')
    rows.append(row)
text+=table(['场景/模式','1%','10%','70%'],rows)
slow=s[(s.variant=='combined')&(s.speedup<1)&(s.condition!='full')]
text+=f'\n### 所有非Full慢例\n\n共{len(slow)}个场景轮次条件的中位更慢。保留每项，不以“原版也有波动”为理由删除。\n\n'
text+=table(['场景','轮','条件','旧ms','新ms','加速倍数'],[[x.scene,x.round,x.condition,f'{x.baseline_ms:.5f}',f'{x.ms:.5f}',f'{x.speedup:.4f}'] for x in slow.itertuples()])
full=s[(s.variant=='combined')&(s.condition=='full')]
text+=f'\nFull诊断48个场景轮次中{int((full.speedup<1).sum())}项更慢，几何平均{geo(full.speedup):.4f}×。Full代码路径没有新增网络计算；实测变化仍需如实记录，不能声称逐场景Full不退化。本轮不改GPU光栅化，不能将其与此前低点数Rasterizer研究的Full验收混同。\n\n'
text+='## CPU分段诊断\n\n分段是计时外独立采样，各项中位不能机械相加。\n\n'
text+=table(['条件','版本','候选/视锥','视角','Selector','选择','Comp及发布'],[[condition,variant]+[f'{st[(st.condition==condition)&(st.variant==variant)&(st.stage==i)].ms.median():.5f}' for i in range(5)] for condition in ['local0.01','local0.7','global0.1'] for variant in ['baseline','combined']])
text+='''
## 完整CPU准备＋H2D＋原GPU渲染

双方均使用上一轮已有的原生三槽交接，不拿旧Python交接再次算收益。本帧属性和51个相机FP32同包上传；GPU只渲染。三开发场景、3个跳转相机、11条件、两轮反序，每条件54同步完成样本。包含CPU准备、排队、上传、依赖、原Rasterizer及最终同步；不含相机矩阵构造/I/O/显示，未把流水线吞吐当单帧延迟。两边线程池在该测试中共驻留但依次使用，不能机械地把CPU单池微测相加得到完整路径。

'''
text+=table(['场景','条件','旧完整ms','新完整ms','加速','旧较大p95','新较大p95'],[[scene,condition,f'{g.baseline_ms.median():.4f}',f'{g.ms.median():.4f}',f'{geo(g.speedup):.4f}×',f'{g.baseline_p95.max():.4f}',f'{g.p95.max():.4f}'] for (scene,condition),g in h.groupby(['scene','condition'])])
text+=f'\n共{r["handoff_samples"]:,}个完整路径样本、{r["handoff_checks"]}组RGB/radii/身份核对；{r["handoff_slower_rounds"]}/66个场景轮次条件更慢。当前证据不支持把CPU组件收益直接宣称为所有实际帧的渲染加速。原始CSV保留host准备排队时间和总延迟。\n\n'
text+='''## 线程长尾及撤回候选

诊断构建记录每次线程分发的worker启动延迟、线程墙钟时间、线程CPU时间及起止CPU编号。trace01覆盖16,200请求，含统计开销，只用于定位，不作正式速度结论。默认策略在一些请求中出现毫秒级分发等待及wall−threadCPU差，说明长尾不全是矩阵运算；起止CPU编号相同不能排除期间迁移，wall−CPU差也不能唯一归因为某个外部进程。

'''
t=pd.read_csv(HERE/'reports/trace01/samples.csv')
text+=table(['策略','条件','中位ms','p95 ms','最大ms','最大启动延迟ms','最大wall−CPU ms'],[[variant,condition,f'{g.ms.median():.4f}',f'{g.ms.quantile(.95):.4f}',f'{g.ms.max():.4f}',f'{g.max_start_delay_ms.max():.4f}',f'{g.max_cpu_gap_ms.max():.4f}'] for (variant,condition),g in t.groupby(['variant','condition'])])
text+='''
- 缩短自旋、worker绑核、作用域主线程绑核均实际编译测试，没有稳定跨预算收益；绑核后仍有较大长尾。最终Pool与上一轮源码逐字相同，未保留这些策略。
- 将小输出FP32拆为偶/奇两条累加链会改变求和顺序，开发集最大属性差约0.000488，速度收益不稳定，已撤回；没有因为用户允许小误差就默认接受无收益的误差。
- 始终扫描全部候选的压缩在低K可能比原排序更慢，保留为独立实验开关，最终使用预先冻结的工作量判断。
- 开发版本源码与二进制在`source/development`、`development_binary`保留，tune01/02/03与trace01保留实际结果。没有重复使用已失败的候选充当最终版本。

## 正确性与边界

'''
text+=f'''- 正式CPU {r['CPU_checks']:,}组：身份、K、属性顺序、全部七类属性SHA、finite全部通过，相对上一版最大差0。
- 同分/离散分数、all/striped/one/none资格、空视锥、K跨SIMD/线程边界、局部严格K及全局ties，共{read('selection_edges01/SUMMARY.json')['checks']}组通过。
- 常规CPU {read('boundaries01/SUMMARY.json')['checks']}项：输出canary、独立性、源Master修改隔离、非法Ratio/NaN相机拒绝、恢复、同实例并发串行化通过。
- 三种地图CPU checkpoint恢复{read('checkpoint01/SUMMARY.json')['checks']}组属性/身份逐位通过，CUDA未初始化；原GPU固定噪声快照被保留，未在CPU重新采样替代。
- 原生交接{read('handoff_edges01/SUMMARY.json')['checks']}项：错误恢复、动态K、不同消费者Stream、上传属性和相机、回默认Stream渲染逐位通过。原Rasterizer默认Stream假设未改。
- 24组×3个相机×17条件={r['quality_rows']}组配对GPU渲染，RGB及radii逐位相同，GT PSNR/SSIM/LPIPS差均为0。只证明本轮相对上一版无新增差异；上一版相对最初GPU PyTorch的近似另见其报告，不能称整体从最初版本始终逐位相同。该测试不是全轨迹质量证明。
- 独立ASan+UBSan正常路径28项通过，独立UBSan完整33项通过。系统ASan与Torch主动异常不兼容已在前轮独立复现，本轮明确排除5个主动异常路径；不称ASan完整异常覆盖。泄漏检测关闭；CPU Sanitizer不覆盖GPU内存或证明无数据竞争。

## 内存、缓存与部署边界

主要Workspace逻辑估算{mem.workspace_bytes.min()/2**20:.2f}–{mem.workspace_bytes.max()/2**20:.2f}MiB。新小输出微核只使用线程栈寄存器/少量局部数组，压缩复用原已reserve的selected容量，不新增每Gaussian缓存。估算不含全部权重、分配器、线程栈和进程共享页，不能当RSS；`CPU_MEMORY.csv`保留逐场景逐版本数据。

三槽位pinned Host/Device容量取决于地图N，与上一版同一实现。每槽属性/相机上传由ready/done Event管理，未完成的ticket不能覆盖；prepare/wait/consume/finish的Stream约定同上一版。缓存只属于加载时冻结快照；参数、Feature、Anchor或拓扑变更必须重建。每帧相机/可见性/选择与补偿仍重新计算，没有用同帧缓存命中或GPU网络运算冒充CPU速度。

本轮正式验证Teacher；Student包装兼容接口未独立验收。结果限定当前x86 AVX2 CPU，不代表ARM/手机或其他核心数。还未测长时间能耗、温控下的持续吞吐，不能声称达到CPU硬件极限。

## 后续优化

1. 低Ratio主成本仍是全候选处理与Selector。继续研究固定快照上的空间数据组织、减少不需要的行访问、让连续候选占用更少cacheline；必须保持名义Anchor候选及严格K，不按历史视角近似筛选。
2. 调度长尾需进一步结合OS调度跟踪、频率和硬件计数器，区分worker唤醒/抢占/功耗状态。当前只得到线程时间证据，未定位每个正式慢样本，不能承诺绑核即可修复。
3. 若尝试更激进INT8/近似激活，应单独验证Selector边界身份及补偿投影长尾；本轮无需引入这些误差。空间换时间仍只允许有界且明确失效的冻结缓存。

## 完整性与复现

- 研究目录：`/home/prmi/zwj/experiments/cpu_native_small_20261110_1318`。
- 基线二进制SHA256：`cc51643a8cd9ab07a340aeb8e3a93f56c8499b0c87568139822678ee310263db`。
- 最终二进制：`{f['binary']}`。
- 最终SHA256：`{f['binary_sha256']}`。
- 保护文件{integrity['protected_files']}个：原生产/训练/模型/前轮关键证据变化0、缺失0；本轮冻结源码与二进制哈希一致，无活跃本轮任务，无渲染图片。
- CSV包含逐场景逐Ratio、N/M/K、中位/p95/最大延迟、全部慢例、质量和分段；原始样本与监测日志均保留。曲线与数据哈希见figures。使用方式见`README_CPU_SMALL.md`。
- 未commit、push、部署，未改共享环境。清理仅限本轮可重建Sanitizer对象，清单另存；最终二进制、失败候选及回退版本均保留。
'''
(HERE/'CPU_SMALL_OUTPUT_SELECTION_OPTIMIZATION_REPORT.md').write_text(text,encoding='utf-8')
print('CHINESE_SMALL_REPORT',len(text))
