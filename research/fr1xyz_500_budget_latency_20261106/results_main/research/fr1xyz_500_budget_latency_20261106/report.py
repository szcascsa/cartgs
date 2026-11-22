"""读取全部实测结果，生成中文速度报告和静态科学图。"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT=Path(__file__).resolve().parent/'results'
OUT=ROOT/'reports';FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
S={(v['method'],v['repeat']):v for v in json.loads((OUT/'SUMMARY.json').read_text())}
Q={v['method']:v for v in json.loads((OUT/'QUALITY_SUMMARY.json').read_text())}
P=json.loads((OUT/'PROFILE_SUMMARY.json').read_text())
EVENT=json.loads((OUT/'CUDA_EVENT_SUMMARY.json').read_text())
ENV=json.loads((OUT/'ENVIRONMENT.json').read_text())
INTEGRITY=json.loads((OUT/'FINAL_INTEGRITY.json').read_text())
PAIRS=json.loads((OUT/'PAIRED_COMPARISONS.json').read_text())
names={'cartgs_full':'原版CaRtGS Full','cartgs_full_cached':'原版Full（缓存静态属性）','old_joint_500':'原无视角联合500点',
       'default_500':'当前默认双视角＋LI，500点','default_full':'当前默认Master Full'}
methods=list(names)
a=S['default_500','all'];b=S['cartgs_full','all'];c=S['default_500','0'];d=S['cartgs_full','0']
lines=['# fr1_xyz局部500点预算：推理与Full渲染速度对比','',
    f"**结论：当前默认模型的500点局部预算在计入推理后更慢。三轮全部样本均值为推理{a['inference_ms']['mean']:.3f}ms＋渲染{a['render_ms']['mean']:.3f}ms＝{a['total_ms']['mean']:.3f}ms；原版CaRtGS Full为{b['total_ms']['mean']:.3f}ms，耗时比{a['total_ms']['mean']/b['total_ms']['mean']:.2f}倍。**",'',
    f"首轮前后均无新增活动GPU进程（仅原有三个Viewer驻留）时：默认500点推理{c['inference_ms']['mean']:.3f}ms＋渲染{c['render_ms']['mean']:.3f}ms＝{c['total_ms']['mean']:.3f}ms；原版Full为{d['total_ms']['mean']:.3f}ms，耗时比{c['total_ms']['mean']/d['total_ms']['mean']:.2f}倍。其余两轮存在外部GPU进程，全部保留，不按耗时丢弃样本。首轮仅作负载较轻的分块参考，不冒称独占GPU或三轮清洁复现。",'',
    '## 测量对象与边界','',
    'Replica等其他场景未测。本轮为TUM fr1_xyz RGB-D，RTX3090，640×480、FP32、TF32关闭、batch1，覆盖相同792个RGB时间戳。默认模型是两网络真实视角＋LI=0.1的既有终态，未重新训练；代码冻结于本地main提交26c355b。', '',
    '原版地图35301点，当前默认及原无视角联合地图均13298点。500点预算指保留500个Gaussian，所有点均计入K；当前名义Anchor视锥中的候选人口平均8211、范围4467–10822，不是仅对500点执行Selector。实际network_ratio=500/Nv，范围0.04620–0.11193。', '',
    '每模型使用各自保存SLAM位姿，分辨率／RGB时间戳一致；本方法新旧模型相机逐值一致。原版与本方法的地图和位姿不同，这是实际模型延迟比较，不能将全部速度差当成同一输入下的算法算子加速。补充当前默认Master Full用于同地图同位姿对照。', '',
    '当前双视角条件模型的已支持入口为PyTorch，实际设置view_conditioned=True；没有借用旧无视角融合CUDA／Graph速度。LI仅训练期使用，推理不计算Full贡献或LI Loss。原版对照调用原始Python Renderer和原始CUDA扩展，不以本方法的Renderer冒充原版。', '',
    '## 同步墙钟耗时','',
    '每方法预热64次，三轮分别顺序／逆序／固定seed乱序遍历792帧，逐帧轮转方法顺序。合计11880次主计时调用，每方法2376次；三轮计时参数相同。模型文件加载、相机构建、RGB读取、指标、Viewer传输显示均在计时之外。动态局部每次重新选点与补偿，没有缓存500点输出。','',
    '|方法|Gaussian输入K|推理／准备均值ms|渲染均值ms|总均值ms|总median ms|总p90 ms|','|---|---:|---:|---:|---:|---:|---:|']
for m in methods:
    v=S[m,'all'];lines.append('|'+names[m]+'|'+f"{v['K']['mean']:.0f}|{v['inference_ms']['mean']:.3f}|{v['render_ms']['mean']:.3f}|{v['total_ms']['mean']:.3f}|{v['total_ms']['median']:.3f}|{v['total_ms']['p90']:.3f}|")
lines+=['','推理／准备包括状态与finite检查、名义视锥、候选Gather、view编码、Selector、Gumbel概率／稳定排序／严格K、Feature查询、Compensator与属性打包。渲染含实际Renderer调用及等待GPU完成。原版常规Full还包含其每次属性激活和screen-space分配；缓存对照仅预计算静态属性，仍调用同一原版CUDA Rasterizer，首／中／末颜色和radii逐bit一致。', '',
    '|方法|首轮总ms|第二轮总ms|第三轮总ms|','|---|---:|---:|---:|']
for m in methods:lines.append('|'+names[m]+'|'+'|'.join(f"{S[m,str(k)]['total_ms']['mean']:.3f}" for k in range(3))+'|')
lines+=['','首轮GPU前后为原三个Viewer和本轮计时进程；第二轮结束出现PID4134006，第三轮也存在该进程，GPU利用率由约29%升至80%–94%，SM时钟也变化。外部进程未终止或干预；本轮不把共享负载带来的差值当成模型开销。进程快照只在时间块边界记录，不能证明块内部始终没有短任务。','',
    '首轮默认500点相对原无视角联合500点多约1.02ms（总耗时约增加23.1%），主要位于真实view编码和条件网络的选点阶段。两个模型权重／点属性也不同，因此不是只改8D条件的严格同权重算子消融。','',
    '## 当前默认模型的耗时分解','',
    '|计时块|Selector＋视角／选择ms|Compensator ms|其他准备ms|渲染ms|总ms|','|---|---:|---:|---:|---:|---:|']
for rep,label in [('0','首轮，块边界无新增活动任务'),('all','全部三轮')]:
    v=S['default_500',rep];lines.append('|'+label+'|'+'|'.join(f"{v[k]['mean']:.3f}" for k in ['selector_ms','compensation_ms','other_prepare_ms','render_ms','total_ms'])+'|')
lines+=['','其他准备包含候选池／Gather、activated Master属性、资格与预算标量、最终打包及finite检查等。Selector栏也包含view7生成、条件拼接、softmax／Gumbel及排序，并不是纯MLP GEMM时间。各项内部计时有原有同步；主总耗时只做一次原样prepare＋一次渲染，没有将分项独立重跑后相加。','',
    f"首轮默认500点约{100*c['inference_ms']['mean']/c['total_ms']['mean']:.1f}%时间花在推理／准备，真正渲染约{100*c['render_ms']['mean']/c['total_ms']['mean']:.1f}%。只看500点Rasterizer会得到较快的结论，但不足以抵消推理开销；同地图Master Full首轮{S['default_full','0']['total_ms']['mean']:.3f}ms，500点纯渲染{c['render_ms']['mean']:.3f}ms，减少点数在此小场景已接近固定调用开销。",'',
    '## CUDA Event与Profiler归因','',
    'CUDA Event另取均匀96帧、共480次调用，事件标记夹住实际stream执行和Host发射间隙。该值不是纯Kernel累计时间，测量时外部任务状态可能不同，不与首轮数值直接拼接。','',
    '|方法|推理stream ms|渲染stream ms|总stream ms|','|---|---:|---:|---:|']
for v in EVENT:lines.append('|'+names[v['method']]+'|'+'|'.join(f"{v[k]['mean']:.3f}" for k in ['inference_stream_ms','render_stream_ms','total_stream_ms'])+'|')
lines+=['','独立Torch Profiler记录四帧×五方法。只累加实际kernel事件，2580个Kernel全部归属到互不重叠的方法范围；不重复累加CPU范围或gpu_user_annotation。原始trace保留在服务器reports/PROFILE.json，Profiler耗时不作为主测速。','',
    '|方法|平均Kernel Launch|Kernel累计ms|Device同步次数|Stream同步次数|','|---|---:|---:|---:|---:|']
for v in P['summary']:lines.append('|'+names[v['method']]+'|'+'|'.join(f"{v[k]['mean']:.3f}" if k=='kernel_ms' else f"{v[k]['mean']:.1f}" for k in ['kernel_launches','kernel_ms','device_syncs','stream_syncs'])+'|')
lines+=['','同步次数包含测试外层同步（Full两次，局部加上原prepare内部同步）；不是全部可删除的等待。默认路径约320次Kernel，与原版Full29次和缓存Full21次相比，小算子与调度显著更多。该证据支持优先检查候选压缩与重复属性激活、view7编码／拼接、Selector／Feature／Compensator打包以及多处标量/finite同步；仅继续优化500点Rasterizer难以解决主耗时。', '',
    '本轮没有修改或加速实现，也没有把尚未通过数值验收的Graph或融合版本设为默认。后续如优化，应保留视角路由、500严格预算、stable排序、finite语义与FP32结果，逐项验收。', '',
    '## 点数与图像质量','',
    '|方法|提交Gaussian数|radii>0均值|全轨迹PSNR dB|','|---|---:|---:|---:|']
for m in ['cartgs_full','old_joint_500','default_500','default_full']:
    v=Q[m];lines.append('|'+names[m]+f"|{v['K']['mean']:.0f}|{v['projected_K']['mean']:.2f}|{v['psnr']['mean']:.4f}|")
lines+=['','当前默认500点PSNR约19.48dB，原版Full约22.64dB，所以也不是等质量速度对比。radii>0不是实际Alpha贡献点数；本轮没有用计数型Rasterizer污染主计时。所有500点样本都严格K=500，候选不足0；首／中／末重复推理的选点和颜色逐bit一致。', '',
    '## 完整性与复现','',
    f"只读保护{INTEGRITY['protected_files']}个文件，包括604个生产关键文件及模型、源码、扩展与轨迹；前后SHA一致。训练步数0，不保存渲染帧图片，研究目录独立，生产代码与默认配置未修改。计时进程自行正常退出，未终止其他进程。",'',
    '服务器目录：`/home/prmi/zwj/experiments/fr1xyz_500_budget_latency_20261106`。逐调用TIMINGS.csv、CUDA_EVENTS.csv、逐帧QUALITY_COUNTS.csv、GPU状态、输入SHA、Profiler汇总和运行脚本均归档。', '',
    '局限：单场景、单设备、固定分辨率；共享GPU有任务变动，三轮为测速复验而非训练seed复验；不含Viewer显示、网络传输或完整SLAM。对更大地图／更高分辨率的速度关系需另测，不能从此结果外推。', '',
    '![耗时分解](figures/LOCAL500_LATENCY.png)','',
    '![各轮延迟](figures/LOCAL500_ROUNDS.png)']
(OUT/'LOCAL500_RENDER_LATENCY_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
plt.rcParams.update({'font.family':font_manager.FontProperties(fname='C:/Windows/Fonts/msyh.ttc').get_name(),
    'font.size':10,'axes.unicode_minus':False,'axes.spines.top':False,'axes.spines.right':False})
short=['CaRtGS Full','CaRtGS Full\n静态缓存','原无视角联合\n局部500点','当前默认\n局部500点','当前默认\nMaster Full']
fig,axes=plt.subplots(1,2,figsize=(14,5.8),sharey=True)
for ax,rep,title in zip(axes,['0','all'],['首轮：块边界无新增GPU任务','全部三轮：含外部GPU任务进入']):
    x=np.arange(5);infer=np.array([S[m,rep]['inference_ms']['mean'] for m in methods]);render=np.array([S[m,rep]['render_ms']['mean'] for m in methods])
    ax.bar(x,infer,label='推理与准备',color='#0072B2');ax.bar(x,render,bottom=infer,label='渲染',color='#E69F00')
    for i,v in enumerate(infer+render):ax.text(i,v+.13,f'{v:.3f}',ha='center',fontsize=10)
    ax.set_xticks(x,short,fontsize=9);ax.set(title=title,ylabel='每帧同步墙钟耗时（ms）',ylim=(0,8));ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True);ax.legend(frameon=False)
fig.suptitle('fr1_xyz RGB-D｜500点渲染减少，推理开销占主要时间',fontsize=14)
fig.text(.5,.015,'RTX3090，640×480，792帧×3轮；CaRtGS 35,301点，当前Master 13,298点；不含Viewer显示。',ha='center',fontsize=9)
fig.tight_layout(rect=(0,.06,1,.94))
for ext in ['png','pdf']:fig.savefig(FIG/('LOCAL500_LATENCY.'+ext),dpi=190,facecolor='white')
plt.close(fig)
fig,ax=plt.subplots(figsize=(11,5.8));x=np.arange(3);width=.14;colors=['#64748B','#B6C2CE','#009E73','#0072B2','#E69F00']
for i,(m,color) in enumerate(zip(methods,colors)):
    vals=[S[m,str(k)]['total_ms']['mean'] for k in range(3)];bars=ax.bar(x+(i-2)*width,vals,width,color=color,label=names[m])
    ax.bar_label(bars,fmt='%.2f',fontsize=8,padding=3)
ax.set_xticks(x,['首轮\n边界快照：无新增活动进程','第二轮\n结束时外部任务进入','第三轮\n外部任务存在'])
ax.set(ylabel='总耗时均值（ms）',title='测速三轮全部保留，负载变化单列',ylim=(0,10.7));ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
ax.legend(frameon=False,ncol=2,loc='upper left',fontsize=9);fig.tight_layout()
for ext in ['png','pdf']:fig.savefig(FIG/('LOCAL500_ROUNDS.'+ext),dpi=190,facecolor='white')
plt.close(fig)
print('已生成中文报告与两张PNG/PDF曲线')
