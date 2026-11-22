"""中文报告只引用本轮实际数据，不将上一轮收益重复计入。"""
from pathlib import Path
import json
import pandas as pd
import numpy as np
HERE=Path(__file__).resolve().parent
def read(name):return json.loads((HERE/'reports'/name).read_text())
def table(headers,rows):return '| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in rows)
r=read('RESULTS_SUMMARY.json');f=read('FROZEN_CANDIDATE.json');integrity=read('INTEGRITY_FINAL.json')
assert r['passed'] and integrity['passed'] and read('boundaries01/SUMMARY.json')['passed'] and read('handoff_edges02/SUMMARY.json')['passed']
a=pd.read_csv(HERE/'reports/CPU_AGGREGATE.csv');s=pd.read_csv(HERE/'reports/CPU_PAIRED_RESULTS.csv');h=pd.read_csv(HERE/'reports/HANDOFF_PAIRED_RESULTS.csv')
q=pd.read_csv(HERE/'reports/QUALITY_ALL_FRAMES.csv');st=pd.read_csv(HERE/'reports/CPU_STAGES.csv');mem=pd.read_csv(HERE/'reports/CPU_MEMORY.csv')
o=pd.read_csv(HERE/'reports/onednn02/samples.csv');om=o.groupby(['matrix','n','variant']).ms.median().unstack()
asan=read('jobs/asan01.json');ubsan=read('jobs/ubsan01.json')
text=f'''# CPU 深度推理优化：原生调度与 AVX2 微核

## 结论

本轮只优化训练完成后的冻结推理，保留原 Teacher 架构和权重。采用两条独立路径：①减少准备阶段线程池分发，并将相机打包、属性视图、槽位及上传事件管理放到C++；②4点×16输出的FP32 AVX2微核，固定32/64输入维度，按原FMA顺序展开循环。训练、生产默认、原checkpoint和Rasterizer不变。

以下对照是本轮重新测得的上一版V4，**不是最初CPU PyTorch**，不重复计算历史约11–13倍的收益。场景轮次等权几何平均加速，延迟为48份场景轮次中位的中位。

'''
text+=table(['条件','V4 ms','本轮 ms','加速倍数','范围','慢例/48轮次'],[
    [x.condition,f'{x.median_baseline_ms:.4f}',f'{x.median_new_ms:.4f}',f'{x.geometric_speedup:.3f}',f'{x.min_speedup:.3f}–{x.max_speedup:.3f}',x.slower_rounds]
    for x in a[a.variant=='combined'].itertuples()])
text+=f'''
正式单池CPU样本{r['CPU_samples']:,}个、单池选点身份哈希核对{r['CPU_isolated_identity_checks']:,}组；此前四池测试另有选点/预算/属性核对{r['CPU_checks']:,}组，相对V4全部属性逐位相同={r['CPU_all_bitwise']}，最大绝对差={r['CPU_max_abs_error']}。CPU测量结束时CUDA未初始化。准备时间包含候选、视角、Selector、选择、Compensator、解码、分配和打包；不含相机矩阵构造、磁盘、上传、渲染。

另以原GPU PyTorch实现和原固定随机键做{r['quality_rows']}组图像核验，平均GT PSNR损失{r['mean_psnr_loss']:+.7f}dB，最差单帧{r['worst_psnr_loss']:.7f}dB，CPU图对GPU图最低PSNR {r['minimum_image_equivalence_psnr']:.4f}dB。这是包括上一轮近似在内的总差异；本轮相对V4的CPU属性没有新增误差。不保证所有未测相机状态，也未声称CPU/GPU逐位相同。

## 实现和正确性依据

### 1. 原生调度与准备融合

- 视锥mask计算完成后，在同一Anchor任务内计算当前视角与Selector动态项。每帧重算，未缓存旧视角、候选或网络输出。
- 候选生成从两次全图扫描/两次线程池分发改为一次扫描：各worker在自己的固定行号区间写入候选和计数，主线程按区间顺序向前memmove压缩。目标偏移始终不大于源区间起点，不覆盖后续区间，原Gaussian顺序保持不变，无额外N规模临时Buffer。
- Selector输出概率时同时写选择Key，省去一次线程池分发和独立扫描。稳定Softmax中最大项的exp(0)直接使用1，只省去已知恒等计算，不使用近似exp；非有限输入走原检查逻辑。
- CPU返回数据的Tensor视图及stats字典由C++构造；应用仍通过薄Python绑定调用扩展，没有声称整个Viewer程序已改写为C++。
- NativeHandoff在C++管理3份有界pinned Host/Device槽、代际租约、copy Stream和ready/done Event。属性及本帧51个相机FP32一次上传；没有GPU网络计算。槽复用前等上一消费者完成；未finish的租约拒绝覆盖。消费/finish需在同一CUDA Stream。
- 原Rasterizer保留默认Stream假设。本轮正式渲染始终在默认Stream；自定义Stream测试只验证数据交接，再回默认Stream渲染，不宣称修改了原Rasterizer的Stream行为。

### 2. AVX2 专用矩阵微核

原V4每次4点×8输出，只有4条独立向量累加链。本轮4点×16输出使用8条链，两个相邻输出块共享输入广播，权重仍保持各Head独立。输入32/64维用编译期分支和4次循环展开，减少动态循环控制与重复广播。每个输出仍按j=0…D−1执行FMA，再加Bias和ReLU，不引入FP16、INT8或fast-math。

小输出Head继续采用原微核，未强行把不同权重相加或修改网络结构。固定维度路径仅用于匹配维度，其余维度仍按合法动态维度处理。原始参数、Feature和Adam语义未涉及，因为本后端完全不执行训练。

### 失败候选

8点×8输出候选实际编译和测量，见tune02及source/eight_candidate.*。它在低K时任务数更少，未给出稳定收益；部分属性差最多5.96e−8，未定位到具体机器指令。本轮未将该候选纳入最终版本，恢复4点路径后重新完成调优和验收。数据保留，不因差异小就当作通过。

## 实验协议

硬件Intel i5-9600K，6核6线程、AVX2/FMA，GPU RTX3090仅用于图像验证/渲染。C++编译-O3/-mavx2/-mfma。双方原生池均6线程，Torch辅助操作1线程，进程均设置OMP_WAIT_POLICY=PASSIVE、KMP_BLOCKTIME=0、GOMP_SPINCOUNT=0，避免上一轮已定位的OpenMP空闲自旋竞争。

开发仅使用Replica office2 RGB-D、room0 RGB-D和TUM fr1_xyz RGB-D。候选冻结后验证其余21组，共Replica16组、TUM4组、VECtor4组。两轮反转场景次序，各场景轮换版本顺序；每个版本单独创建线程池并完成13条件后销毁。每条件三块，每块先6次切换视角预热，再12样本，合计36样本/条件/版本/轮。每场景3连续相机＋3跳转相机，严格K/同分/全局分位数语义保持。

formal01/02最初采用四池共驻留的短块版本轮换，外部负载为零却仍出现空预算约0.9ms、部分请求2ms以上的异常长尾。不能只看外部CPU监控就声称独占CPU；保留其{r['CPU_multipool_samples']:,}个样本及CPU_MULTIPOOL_RESULTS.csv。候选二进制完全不变，新增isolated01/02每次只驻留一个池，版本计时相隔约秒级，用两轮顺序轮换控制漂移。主表报告单实例部署条件；多池条件作为并发资源争抢限制单列。没有按场景改变算法或删除原始慢例。

不复用结果，每帧仍执行全部准备。分段诊断在正式计时外采集。外部CPU/GPU负载监测与进程身份保留logs/*MONITOR.jsonl及reports/jobs；只接受无外部活动的正式轮次。

## 两项优化独立归因

'''
text+=table(['条件','仅调度加速','仅微核加速','合用加速'],[
    [condition]+[f'{a[(a.condition==condition)&(a.variant==v)].iloc[0].geometric_speedup:.3f}×' for v in ['schedule','kernel','combined']]
    for condition in ['local0.01','local0.05','local0.1','local0.7','fixed1000','global0.1']])
text+='\n## 每场景 CPU 完整准备\n\n每格为本轮配对V4→本轮两轮中位ms，括号为两轮几何平均加速。完整条件、p95、N/M/K及慢例见CSV。\n\n'
scene_rows=[]
for scene in s.scene.unique():
    row=[scene]
    for condition in ['local0.01','local0.1','local0.7']:
        g=s[(s.scene==scene)&(s.condition==condition)&(s.variant=='combined')]
        row.append(f'{g.baseline_ms.median():.3f}→{g.ms.median():.3f} ({np.exp(np.log(g.speedup).mean()):.2f}×)')
    scene_rows.append(row)
text+=table(['场景/模式','1%','10%','70%'],scene_rows)
text+='\n### 单池仍存在的慢例\n\n低Ratio主验收的1%/5%/10%均在48份场景轮次中获益，但不能称所有请求都更快。下表列出combined所有非Full中位退化项，不删除空预算异常。它们未在相同条件两轮同时退化，当前未用调度跟踪定位每次长尾的具体等待来源，不能直接断言全部来自外部负载或视角。Full诊断的所有退化另在完整CSV保留。\n\n'
slow=s[(s.variant=='combined')&(s.speedup<1)&(s.condition!='full')]
text+=table(['场景','轮次','条件','V4 ms','本轮ms','加速倍数'],[[x.scene,x.round,x.condition,f'{x.baseline_ms:.5f}',f'{x.ms:.5f}',f'{x.speedup:.3f}'] for x in slow.itertuples()])
text+='\n## CPU分段诊断\n\n分段为独立采样中位，不能机械相加；融合后阶段归属改变，例如Anchor视角成本计入候选阶段。\n\n'
stage_names=['候选/视锥','视角动态项','Selector','选择','Compensator及发布']
text+=table(['条件','版本']+stage_names,[[condition,variant]+[f'{st[(st.condition==condition)&(st.variant==variant)&(st.stage==i)].ms.median():.4f}' for i in range(5)]
    for condition in ['local0.01','local0.7'] for variant in ['baseline','combined']])
text+='''
## 准备＋上传＋原渲染器

两轮配对分别包含baseline=V4＋Python交接、core_only=本轮CPU核心＋旧交接、native_all=本轮CPU核心＋原生交接。包含本帧相机矩阵及属性打包、H2D、跨Stream依赖及原FP32 Rasterizer；不含CPU相机矩阵构造、I/O、显示。表中为两轮分组中位的中位，p95列保留两轮较大的p95。每样本同步完成，未把流水线吞吐当作单帧延迟。

'''
handoff_rows=[]
for (scene,condition),g in h[h.variant=='native_all'].groupby(['scene','condition']):
    core=h[(h.scene==scene)&(h.condition==condition)&(h.variant=='core_only')]
    handoff_rows.append([scene,condition,f'{g.baseline_ms.median():.4f}',f'{core.ms.median():.4f}',f'{g.ms.median():.4f}',f'{g.p95.max():.4f}',f'{np.exp(np.log(g.speedup).mean()):.3f}×'])
text+=table(['场景','条件','V4完整ms','仅核心优化ms','原生交接完整ms','较大p95','总加速'],handoff_rows)
text+='\n原生交接相对core_only的差异单独展示，不把GPU渲染本身计为网络提速。所有条件均保留，包括空预算、未见ratio、固定点数和全局导出。\n\n## oneDNN对照\n\n'
text+='真实Selector body、Compensator body与四Head隐藏层权重，一线程FP32配对微测。AVX输出复用；oneDNN权重预打包。dense_io包含动态输入/输出布局转换，opaque_io只测已转换布局的算子上限，不能解释为完整部署收益。下表单位微秒。\n\n'
text+=table(['矩阵','行数','原AVX4×8','新AVX4×16','ATen','oneDNN dense_io','oneDNN opaque_io'],[
    [index[0],index[1]]+[f'{row[name]*1000:.2f}' for name in ['avx4x8','avx4x16','aten','onednn_dense_io','onednn_opaque_io']]
    for index,row in om.iterrows()])
text+='''
本轮测试的Torch预打包oneDNN入口在4行64→256矩阵上约75微秒（dense_io）/46微秒（opaque_io），新AVX约1.85微秒；4096行时约1612/1107微秒，新AVX约1231微秒。大批次理想布局有局部优势，但动态dense输入/输出时未胜出，且实际融合路径每worker按4点处理。微测新AVX矩阵吞吐接近原版2倍，不等于完整准备或渲染2倍。

这只是当前Torch/oneDNN入口，不是直接oneDNN C API的理论极限。最终保留自有小批次融合微核，没有将opaque理想布局结果直接当作收益。下一步若研究整批GEMM，必须计入布局转换和网络中间激活成本，重新验证整条网络，不依据微测替换部署路径。onednn01曾因进程PATH缺少ninja在编译前失败，修正本进程PATH后onednn02完成；未安装或修改共享环境。

## 正确性、内存安全与限制

'''
text+=f'''CPU常规33项边界通过；原生交接错误恢复、动态K和不同消费者Stream共{read('handoff_edges02/SUMMARY.json')['checks']}项通过；三帧在途租约、未释放/二次释放拒绝、相机/属性和交接后RGB/radii逐位一致检查通过。checkpoint原固定键CPU恢复{read('checkpoint01/SUMMARY.json')['checks']}组逐位一致，CUDA未初始化。所有图像评价不保存渲染图片。

两个测试脚本问题已单独修正并保留失败日志：quality01沿用旧类型判定，误将Deep实例当成PyTorch oracle并传入measure_time参数，修正分派后quality02完成；handoff_edges01对非连续矩阵的flatten副本注入NaN，没有实际修改相机，改用连续副本及明确的注入断言后handoff_edges02完成。均未改变冻结算子或重跑已通过的正式性能样本。

ASan+UBSan正常路径任务退出码{asan['returncode']}；因上一轮已独立复现系统ASan在Torch主动异常上的兼容问题，明确排除5项故意异常，保留其余28项。独立UBSan完整33项退出码{ubsan['returncode']}。ASan关闭泄漏检测，不能称LeakSanitizer验收；GPU交接为行为与Stream测试，不等于ASan覆盖CUDA内存。失败探针见上一版目录，本轮未重复无意义复现。

主要Workspace估算{mem.workspace_bytes.min()/2**20:.2f}–{mem.workspace_bytes.max()/2**20:.2f}MiB。新微核不新增每Gaussian缓存；三槽上传容量与上一版相同，取决于地图N。该估算未含所有权重/分配器/线程栈，不是进程RSS。无需随视角失效的静态缓存只属于冻结快照，地图或参数变化须重建实例。

Full为直接Master渲染，不需要Selector/Compensator。本轮Full诊断仍是整图属性重打包；逐轮结果及波动照实保留，不能把该诊断接口称作Full渲染优化。原生产Full渲染器源码和入口没有修改。

未修改网络、训练、选点、checkpoint或生产默认；不支持同一实例边训练边推理。实测仅适用于当前x86 AVX2服务器，未在ARM或移动CPU测试，不能声称已达到硬件极限。

## 可复现性与交付

- 研究目录：`/home/prmi/zwj/experiments/cpu_native_deep_20261110_1222`。
- 最终二进制：`{f['binary']}`。
- SHA256：`{f['binary_sha256']}`。
- 原生产、原模型、上一版证据等{integrity['protected_files']}文件哈希变化0、缺失0；最终源码/二进制哈希一致，无活跃本轮任务，无渲染图片。
- 正式CSV：CPU_PAIRED_RESULTS、CPU_AGGREGATE、CPU_CORRECTNESS、CPU_STAGES、HANDOFF_PAIRED_RESULTS、QUALITY_ALL_FRAMES。
- 曲线及原始数据哈希见figures；调用示例见README_CPU_DEEP.md。没有commit、push或部署。
'''
(HERE/'CPU_DEEP_INFERENCE_OPTIMIZATION_REPORT.md').write_text(text,encoding='utf-8');print('CHINESE_DEEP_REPORT',len(text))
