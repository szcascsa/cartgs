from pathlib import Path
import json,csv,statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
ROOT=Path(__file__).resolve().parents[3];R=ROOT/'reports'
def read(n):return json.loads((R/(n+'.json')).read_text())
def table(headers,rows):return '\n'.join(['|'+'|'.join(headers)+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(str(x) for x in row)+'|' for row in rows])
rounds=[read('DYNAMIC_BENCHMARK_'+str(j)) for j in [1,2,3]]
labels={'ratio1':'局部1%','ratio70':'局部70%','points2000':'固定2000点','changing_ratio':'逐帧变化ratio','changing_points':'逐帧变化点预算'}
rows=[];csvrows=[]
for i,c in enumerate(rounds[0]['results']):
 vals=[v['results'][i] for v in rounds]
 a=statistics.median(v['summary']['baseline']['median_ms'] for v in vals);b=statistics.median(v['summary']['fused']['median_ms'] for v in vals)
 med=[100*(1-v['summary']['fused']['median_ms']/v['summary']['baseline']['median_ms']) for v in vals]
 avg=[100*(1-v['summary']['fused']['mean_ms']/v['summary']['baseline']['mean_ms']) for v in vals]
 rows.append([labels[c['case']],f'{a:.3f}',f'{b:.3f}',f'{min(med):.1f}%–{max(med):.1f}%',f'{min(avg):.1f}%–{max(avg):.1f}%',f"{c['K_min']}–{c['K_max']}"])
 for j,v in enumerate(vals,1):
  for tag,s in v['summary'].items():csvrows.append(dict(round=j,case=c['case'],backend=tag,**s,K_min=v['K_min'],K_max=v['K_max'],unique_K=v['unique_K']))
with (R/'DYNAMIC_LATENCY.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(csvrows[0]));w.writeheader();w.writerows(csvrows)
profile={v['label']:v for v in read('PROFILE_ANALYSIS')['summary']};pt=[]
for ratio in ['0.01','0.1','0.7']:
 a,b=[profile[f'prepare/{tag}/{ratio}'] for tag in ['baseline','fused']]
 pt.append([ratio,int(a['kernels']),int(b['kernels']),f"{100*(1-b['kernels']/a['kernels']):.1f}%",f"{a['kernel_us']:.1f} → {b['kernel_us']:.1f}",f"{a['syncs']:.0f} → {b['syncs']:.0f}"])
vt=[]
for v in read('VIEWER_BENCHMARK')['results']:
 a,b=[v['summary'][tag] for tag in ['baseline','fused']]
 vt.append([labels[v['case']],f"{a['median_ms']:.3f}",f"{b['median_ms']:.3f}",f"{100*(1-b['median_ms']/a['median_ms']):.1f}%",f"{a['summary_median_ms']:.3f} → {b['summary_median_ms']:.3f}"])
checks=[]
for n,title in [('QUATERNION_PROBE','Quaternion逐节点'),('PROBE','概率与视锥'),('EDGE_VALIDATION','极值／空集／流／稳定排序'),('LAYER_VALIDATION','五种网络分层'),('PIPELINE_VALIDATION','五模型完整路径'),('STATE_VALIDATION','Graph／状态失效／所有权'),('TOPOLOGY_VALIDATION','拓扑／保存重载'),('PURITY_VALIDATION','权重／Master／RNG／梯度只读'),('DYNAMIC_VALIDATION','792帧动态渲染'),('SUMMARY_VALIDATION','Anchor整数统计'),('UNIT_VALIDATION','预算／Viewer／CLI单测'),('VIEWER_BENCHMARK','Viewer像素封包与指标')]:
 v=read(n);assert v['passed'];checks.append([title,v.get('comparisons',v.get('tests')),'通过'])
mb,mf=read('MEMORY_baseline'),read('MEMORY_fused')
alloc=lambda v,name:next((o['count']/v['iterations'] for o in v['operations'] if o['name']==name),0)
mt=[[name,f'{alloc(mb,name):g}',f'{alloc(mf,name):g}'] for name in ['aten::empty','aten::empty_strided','aten::resize_','aten::cat','aten::index','aten::clone','aten::as_strided']]
mic=[[v['ratio'],v['module'],f"{v['baseline']['median_us']:.1f}",f"{v['fused']['median_us']:.1f}",f"{v['speedup']:.2f}×"] for v in read('MICROBENCHMARK')['results']]
ident=read('BINARY_IDENTITY')
parts=['''# 动态输入推理CUDA融合优化报告

## 结论与测量范围

已实现五项优化：名义Anchor视锥外围融合、Selector两类概率融合、索引有效位排序、Quaternion解码融合、Viewer整数Anchor统计融合。仅修改冻结推理CUDA路径，网络结构、预算、同分规则、Residual48累加和四Head保持原语义。

性能对照为上一轮已优化的CUDA v2；数值检查同时使用真实PyTorch分层及v2完整路径。实际模型为fr1_xyz RGB-D：13298个Gaussian、2790个Anchor、792轨迹帧，双真实视角＋LI=0.1。兼容旧11D及joint schema1/2/3/4。未训练、不保存渲染图片。

没有新增同帧视锥Mask、ratio→Anchor48查询结果或相机输出缓存。每帧重新计算视锥和视角编码、查询Residual表。主测关闭Graph，不使用全局导出缓存；相机逐帧变化，预算和ratio变化。已有静态Master缓存和Graph兼容能力不计作本轮收益。

**负载限制：GPU上有其他用户的训练和推理；未终止或调整这些进程。以下是实际并发环境的配对观察，不能当作独占GPU的绝对FPS、p99承诺或在线SLAM训练加速。** 后端顺序逐帧轮换，分配、检查、同步和Rasterizer工作一致。三轮原始样本完整保留。

## 动态准备＋渲染

每轮每条件792帧；耗时取三轮中位数的中位数，下降区间按每轮分别计算，未挑最好轮次。
''',table(['条件','v2中位ms','新中位ms','中位下降范围','平均下降范围','实际K范围'],rows),'''
变化ratio：.01/.03/.05/.08/.1/.2/.3/.4/.5/.6/.7；变化点预算：1000/2000/3000/4000/5000/10000。变化ratio有668种实际K，变化点预算有141种；各帧按自身候选N计算ratio/K，不使用跨场景same-K。shortfall如实记录。

原始每帧值和mean、median、p90、p99见DYNAMIC_BENCHMARK_1/2/3.json与DYNAMIC_LATENCY.csv。boundary只开视锥和概率融合；fused再开Quaternion和有效位排序。计时从prepare调用前到Rasterizer完成，包含必要同步与CPU发射；相机对象在三个后端计时外共同构建。HTTP和显示不在此范围。

## Viewer完整处理链

含相机、Anchor统计、PSNR和FP32封包。每条件114帧，共228个请求，像素封包和几何／质量统计一致；不含HTTP与客户端显示。
''',table(['条件','v2中位ms','新中位ms','下降','Anchor统计ms'],vt),'''
## Nsight Systems、NVTX与CUDA Event

每ratio各10个相机，两个后端逐帧交错。以Host NVTX范围和CUPTI correlation归属Kernel；嵌套node范围不重复相加。原始trace保存在服务器reports/PROFILE.nsys-rep与PROFILE.sqlite。
''',table(['ratio','原准备Kernel','新准备Kernel','减少','累计Kernel μs 原→新','同步 原→新'],pt),'''
累计GPU Kernel时间受到并发SM争用污染，不将其比值称为独占GPU算子加速；Kernel数量和数据流依赖的变化更可靠。准备阶段仍有3次DtoH、5次同步API，保留候选计数和finite依赖。

Viewer Anchor统计：64个Kernel、12次同步、12次DtoH，降到6个Kernel、1次同步、1次DtoH；仍为每次统计重新计算视锥。后续projected_gaussians由相同整数结果相加，避免再发GPU sum/readback。

## 实现与严格等价

1. 视锥：一个Kernel构造六平面，保留原centers@planes[:,:3].T的GEMM和strided布局，一个Kernel完成偏移、AABB半径、六平面判断和occupied筛选。三项求和复现ATen的(a+c)+b。
2. 概率：按全局行ID读取固定noise，融合加法、两类Softmax和正类score写入。保留exp和直接除法，温度仍为已校验的1；不改成sigmoid(logit差)，不重采样noise。
3. 索引排序：score仍完整32位稳定排序。索引非负且小于N，第二次排序只需max(1,ceil(log2(N)))位，与64位整数比较完全等价；返回仍为原行递增顺序，严格K不变。
4. Quaternion：融合identity、旋转Gather、四元数乘积和clamp/div，保留原三次范数归约及三次归一化。直接用原Python实际调用的linalg_vector_norm。叉乘显式使用原CUDA FMA，其余运算显式FP32舍入；编译保持--fmad=false，无fast-math、TF32或低精度。
5. Anchor统计：逐点累计每Anchor提交／投影计数，再按Anchor做整数汇总，替换unique/bincount及多个归约。整数求和精确；保留unique Anchor和重复点ID的原统计含义。每次清零scratch，无上帧统计缓存。

仅修改六个文件：python/anchor_budget.py、anchor_visibility.py、budget_viewer.py、frozen_inference_cuda.py、frozen_inference_bindings.cpp、frozen_inference_kernels.cu。INCREMENTAL_DIFF.patch以本轮开始时的v2工作区源码为基线；无关训练C++等既有修改保留。

## 正确性验证
''',table(['验证','比较／测试数','状态'],checks),'''
792帧×19条件=15048组，覆盖11个ratio和8个固定预算（含0、1、10000）。逐bit比较ID、属性、radii与像素；不以PSNR近似替代等价。同一像素输入沿用相同评价代码时质量不变。本轮不重跑训练或全套SSIM/LPIPS，不据此推断新场景速度。

拓扑测试含Clone/Split、新Anchor、Gaussian/Anchor重排、Prune、106416点规模、空拓扑、几何修正和checkpoint重载。这是受控内存迁移，不冒充在线训练。模型、Master、RNG和grad只读性通过；训练Autograd和Adam路径未进入优化。

## 显存与分配

独立进程、相同预热，CPU-only Profiler归因分配请求，计时不使用Profiler结果。
''',table(['项目','v2','新版本'],[['额外保留活跃字节',mb['retained_bytes'],mf['retained_bytes']],['准备＋渲染峰值增量字节',mb['peak_additional_bytes'],mf['peak_additional_bytes']]]),table(['每请求ATen操作','v2','新版本'],mt),'''
empty/resize是ATen请求次数，不能等同于cudaMalloc次数；仍使用LibTorch缓存分配器。本轮empty与empty_strided请求没有减少（分别36→38、13→16），显存峰值也未下降；收益主要来自Kernel和同步减少。范数API改为直接linalg调用后，未观察到分配请求进一步减少，不将该API清理单独列为性能收益。Viewer scratch为8(2A+30)字节，本模型44880字节，随A线性、调用后释放，无无界缓存。

## 组件微测

20次预热、50组×10次相同输入操作。CUDA Event区间包含CPU发射空隙和其他进程GPU争用；与Nsight纯Kernel时间分开报告，不声称全部是GPU计算收益。
''',table(['ratio','组件','原μs','新μs','观察加速'],mic),'''
## 失败候选与修正

- Quaternion初稿普通乘减和(a+b)+c出现约1e-8差异；按原归约和叉乘FMA修正，再完整验证，未接受该误差。
- Softmax初稿倒数乘法出现2.98e-8差异；改回原直接除法，不以最终选点相同为理由接受score差异。
- 微测脚本最初把stride=2的score直接传入要求contiguous的扩展；按接口对相同输入预打包后重新测量，未放宽产品检查。
- 未尝试合并四Head大GEMM、重排Gaussian、删归一化或改稳定排序。未新增动态ratio形状自动捕获Graph策略。
- norm_wrapper_reference保存最终范数API清理前的证据；正式表格为最终源码重新验证和测量的结果。

## 接入、回退与剩余瓶颈
''',f"独立扩展：{ident['path']}。SHA256：{ident['sha256']}。Torch {ident['torch']}，CUDA {ident['cuda']}，RTX3090。编译参数、源码SHA及加载路径见BINARY_IDENTITY.json和RUNTIME_MAPS.txt。",'''
现有cuda后端启用本轮冻结推理优化；pytorch保留原算法回退。实例dynamic_fusion=False可回退视锥／score外围，quaternion_fusion=False、narrow_sort=False、summary_fusion=False分别回退其他项。cuda_graph兼容，但主收益不依赖Graph。参数、Feature、Master或拓扑变更后仍要求rebuild，保留旧状态失效检查。

剩余热点主要是原形状SGEMM、score稳定排序、视角编码GEMM、必要计数／finite同步，以及Rasterizer tile展开排序。下一步先独占GPU复测p90/p99，再依据NVTX确认四Head实际占比，受控尝试同形状四Head并行；每Head仍需逐bit验证。小ratio和大ratio分别验收，不能直接换成改变累加顺序的大GEMM。

未更新生产mycartgs/realLOD、未切生产默认、未commit/push。604个生产关键文件与42个直接模型文件前后SHA，以及28个历史模型文件匹配结果，见FINAL_INTEGRITY.json和逐文件manifest。服务器实验目录：/home/prmi/zwj/experiments/dynamic_inference_fusion_20261106。
''']
(R/'DYNAMIC_INFERENCE_FUSION_OPTIMIZATION_REPORT.md').write_text('\n\n'.join(parts))
font=FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc');plt.rcParams['font.family']=font.get_name();plt.rcParams['axes.unicode_minus']=False
figdir=R/'figures';figdir.mkdir(exist_ok=True)
fig,ax=plt.subplots(figsize=(10.8,4.8));x=list(range(len(rows)));width=.36
for j,tag in enumerate(['baseline','fused']):
 y=[statistics.median(v['results'][i]['summary'][tag]['median_ms'] for v in rounds) for i in x]
 ax.bar([i+(j-.5)*width for i in x],y,width,label='上一轮CUDA' if j==0 else '本轮融合')
ax.set_xticks(x,[labels[v['case']] for v in rounds[0]['results']]);ax.set_ylabel('准备＋渲染中位耗时（ms）');ax.set_title('连续相机与变化输入：三轮中位数（存在其他GPU任务）');ax.legend();ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(figdir/'dynamic_latency.png',dpi=180);fig.savefig(figdir/'dynamic_latency.pdf');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(10.8,4.4))
for ax,name in zip(axes,['changing_ratio','changing_points']):
 for tag,col,label in [('baseline','#64748b','上一轮CUDA'),('fused','#0284c7','本轮融合')]:
  vals=sorted(v for rd in rounds for case in rd['results'] if case['case']==name for v in case['raw_ms'][tag]);ax.plot(vals,[(i+1)/len(vals) for i in range(len(vals))],color=col,label=label)
 ax.set_title(labels[name]);ax.set_xlabel('完整准备＋渲染墙钟（ms）');ax.set_xscale('log');ax.grid(alpha=.2)
axes[0].set_ylabel('累计帧占比');axes[1].legend();fig.suptitle('全部三轮原始时延分布，包含并发负载长尾');fig.tight_layout();fig.savefig(figdir/'latency_distribution.png',dpi=180);fig.savefig(figdir/'latency_distribution.pdf');plt.close(fig)
print('REPORT_CREATED',flush=True)
