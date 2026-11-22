"""2000方形Tensor方案：网络算子、纯渲染和完整墙钟分列。"""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager,FontProperties
HERE=Path(__file__).resolve().parent;R=HERE/'reports';OUT=HERE/'figures';OUT.mkdir(exist_ok=True)
s=pd.read_csv(R/'samples.csv');c=pd.read_csv(R/'checks.csv');q=pd.read_csv(R/'quality_native.csv')
u=pd.read_csv(R/'components/samples.csv');uc=pd.read_csv(R/'components/checks.csv')
v=json.loads((R/'VALIDATION.json').read_text());cj=json.loads((HERE/'logs/components_job.json').read_text());fj=json.loads((HERE/'logs/formal_job.json').read_text())
pj=json.loads((HERE/'logs/profiles_job.json').read_text())
assert v['passed'] and fj['measurement_valid'] and cj['measurement_valid']
assert len(s)==67392 and len(c)==1296 and len(q)==216 and len(u)==11520 and len(uc)==720
assert c.rgb_radii_bitwise.all() and np.isfinite(s.ms).all() and np.isfinite(u.ms).all()
assert s.groupby(['scene','condition','variant','scope']).size().isin([72,216]).all()
assert c.groupby(['scene','condition','frame']).K.nunique().eq(1).all()
assert c.groupby(['scene','condition','frame']).M.nunique().eq(1).all()
assert c[c.variant.isin(['C','D'])].groupby(['scene','condition','frame']).ids_sha256.nunique().eq(1).all()
g=s.groupby(['scene','condition','variant','scope']).agg(ms=('ms','median'),p95=('ms',lambda x:x.quantile(.95)),
    maximum=('ms','max'),N=('N','median'),M=('M','median'),K=('K','median'),samples=('ms','size')).reset_index()
g.to_csv(R/'TIMING_SUMMARY.csv',index=False)
sr=s.groupby(['scene','condition','variant','scope','round']).ms.median().reset_index();sr.to_csv(R/'TIMING_ROUNDS.csv',index=False)
ug=u.groupby(['scene','ratio','variant','stage']).agg(ms=('ms','median'),p95=('ms',lambda x:x.quantile(.95)),
    N=('N','median'),M=('M','median'),K=('K','median'),samples=('ms','size')).reset_index()
ug.to_csv(R/'NETWORK_COMPONENT_SUMMARY.csv',index=False)
q.groupby(['scene','condition','variant']).agg(psnr=('psnr','mean'),mean_loss=('psnr_loss_vs_A','mean'),
    worst_frame_loss=('psnr_loss_vs_A','max')).reset_index().to_csv(R/'QUALITY_SUMMARY.csv',index=False)


def val(scene,condition,variant,scope):
    row=g[(g.scene==scene)&(g.condition==condition)&(g.variant==variant)&(g.scope==scope)]
    assert len(row)==1
    return float(row.iloc[0].ms)


rows=[]
for (scene,condition),group in g.groupby(['scene','condition'],sort=False):
    row=dict(scene=scene,condition=condition,N=int(group.iloc[0].N),M=group.iloc[0].M,K=group.iloc[0].K)
    for variant in ['A','C','D']:
        for scope in ['camera_wall','engine_prepare_wall','prepare_wall','raster_event','pipeline_wall']:
            row[variant+'_'+scope]=val(scene,condition,variant,scope)
    row['pipeline_speedup_D_vs_A']=row['A_pipeline_wall']/row['D_pipeline_wall']
    row['pipeline_speedup_C_vs_A']=row['A_pipeline_wall']/row['C_pipeline_wall']
    row['pipeline_speedup_D_vs_C']=row['C_pipeline_wall']/row['D_pipeline_wall']
    row['D_pipeline_vs_D_full']=val(scene,'full','D','pipeline_wall')/row['D_pipeline_wall']
    if condition!='full':
        ratio=float(condition[5:]);row['ratio']=ratio
        for variant in ['A','C','D']:
            for stage in ['selector','compensator']:
                values=ug[(ug.scene==scene)&(ug.ratio==ratio)&(ug.variant==variant)&(ug.stage==stage)]
                assert len(values)==1
                row[variant+'_'+stage+'_isolated_us']=float(values.iloc[0].ms)*1000
    rows.append(row)
t=pd.DataFrame(rows);t.to_csv(R/'TENSOR_CORE_SEPARATED_RESULTS.csv',index=False)
paired=s[s.scope=='pipeline_wall'].pivot(index=['scene','condition','round','repeat','frame'],columns='variant',values='ms')
paired['D_vs_A']=paired.A/paired.D;paired['C_vs_A']=paired.A/paired.C;paired['D_vs_C']=paired.C/paired.D
paired.to_csv(R/'PAIRED_PIPELINE.csv')
def geo(x):return float(np.exp(np.log(x).mean()))
lod=t[t.condition!='full'];pair_lod=paired.reset_index();pair_lod=pair_lod[pair_lod.condition!='full']
profile=pd.read_csv(R/'PROFILE_NETWORKS.csv')
summary=dict(passed=True,samples=len(s),correctness_cases=len(c),native_quality_rows=len(q),network_event_samples=len(u),
    network_checks=len(uc),profiles=profile.profile.nunique(),
    D_pipeline_min_ms=lod.D_pipeline_wall.min(),D_pipeline_max_ms=lod.D_pipeline_wall.max(),
    D_prepare_min_ms=lod.D_prepare_wall.min(),D_prepare_max_ms=lod.D_prepare_wall.max(),
    D_vs_A_pipeline_geomean=geo(pair_lod.D_vs_A),C_vs_A_pipeline_geomean=geo(pair_lod.C_vs_A),D_vs_C_pipeline_geomean=geo(pair_lod.D_vs_C),
    D_native_mean_psnr_loss=v['quality']['D']['mean_loss'],D_native_worst_frame_loss=v['quality']['D']['worst_frame'],
    protected_files=v['protected_files'],foreign_gpu=fj['foreign_active_samples']+cj['foreign_active_samples'],
    foreign_cpu=fj['foreign_cpu_active_samples']+cj['foreign_cpu_active_samples'])
(R/'ANALYSIS_VALIDATION.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
lines=['# 2000×2000 Tensor Core方案速度验证','','## 结论','',
    f"使用报告中最终D方案，RTX3090、2000×2000、GPU Selector＋Student、新G Renderer，4场景×5 Ratio完整延迟为{lod.D_pipeline_wall.min():.3f}–{lod.D_pipeline_wall.max():.3f}ms。相机＋完整推理准备约{lod.D_prepare_wall.min():.3f}–{lod.D_prepare_wall.max():.3f}ms。当前视角每次重新计算，非跨帧流水线吞吐。",'',
    f"同批次逐视角配对，D完整流程相对A的等权样本几何平均加速{summary['D_vs_A_pipeline_geomean']:.3f}×；只换Tensor网络的C相对A为{summary['C_vs_A_pipeline_geomean']:.3f}×；D相对C为{summary['D_vs_C_pipeline_geomean']:.3f}×。全部D/A收益不能归为Tensor Core，D还含相机、CUDA准备、预投影与调度优化。",'',
    '本报告把Tensor Core网络单独标注。纯渲染列来自独立CUDA Event：只有属性/相机已就绪后的G光栅过程，不含Selector、Compensator、动态投影、网络准备、上传或CPU提交开销。完整延迟来自另一次独立墙钟采样，不能用各组件中位数相加或相减代替。','',
    '## A/C/D定义','',
    '| 版本 | 两网络 | 准备与相机 |','|---|---|---|',
    '| A | 原v31 FP32融合CUDA、同一3534参数学生 | 原准备＋R相机 |',
    '| C | FP16 Tensor Core、FP32累加 | 原准备＋R相机 |',
    '| D | C网络，补偿输入采用冻结FP32预投影 | CUDA视锥融合、GPU动态N/K、C++当前相机 |','',
    '## 速度分项','',
    '单位ms；Selector/Compensator算子另见下表。Full没有网络，D的Full收益只属于相机准备与调度。','',
    '| 场景 | Ratio | 实际K中位数 | A完整 | C完整 | D完整准备 | D纯GPU渲染 | D完整 | D/A倍速 |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for _,row in t.iterrows():
    lines.append(f"| {row.scene.replace('Replica/','')} | {row.condition} | {row.K:g} | {row.A_pipeline_wall:.4f} | {row.C_pipeline_wall:.4f} | {row.D_prepare_wall:.4f} | {row.D_raster_event:.4f} | {row.D_pipeline_wall:.4f} | {row.pipeline_speedup_D_vs_A:.3f}× |")
lines+=['','## Tensor Core网络单独计时','',
    '以下单位为μs，使用同一候选与选中Gaussian身份、同一View/Ratio和相同学生。CUDA Event测16次固定输入Graph后除16，两轮交替顺序；不含候选筛选、排序和Rasterizer。Selector算子包含输入读取、ReLU、Score与状态，Compensator算子还包含物理解码和打包；它们不是单独HMMA指令时间。',
    'D补偿组件另包含全部Anchor动态投影，是保守组件边界；正式D仅处理当前可见Anchor。该组件微测不应当作真实移动帧的网络分项，也不能与完整准备中位数相加。真实移动调用的网络Kernel归因另由Nsight给出。','',
    '| 场景 | Ratio | A Selector | C Tensor Selector | D Tensor Selector | A Comp | C Tensor Comp | D Tensor Comp＋全Anchor投影 |',
    '|---|---:|---:|---:|---:|---:|---:|---:|']
for _,row in lod.iterrows():
    lines.append(f"| {row.scene.replace('Replica/','')} | {row.ratio:g} | {row.A_selector_isolated_us:.3f} | {row.C_selector_isolated_us:.3f} | {row.D_selector_isolated_us:.3f} | {row.A_compensator_isolated_us:.3f} | {row.C_compensator_isolated_us:.3f} | {row.D_compensator_isolated_us:.3f} |")
lines+=['','## 实际移动帧的Nsight归因','',
    f"5个短Trace、各36帧。这里用Profiler确认实际执行的Tensor网络Kernel，与render_only范围分开。节点级Profile记录到{pj['foreign_active_samples']}个外部GPU活跃样本和{pj['foreign_cpu_active_samples']}个外部CPU活跃样本；Profiler本身也扰动时序，因此以下只解释归因，不作为正式速度或加速证据。正式耗时来自此前无外部活跃负载的墙钟/组件批次。网络融合Kernel含索引/激活和补偿物理解码发布，不能冒称纯MMA。",'',
    '| 场景 | Ratio | Selector Tensor Kernel μs/帧 | Compensator Tensor Kernel μs/帧 |','|---|---:|---:|---:|']
for (scene,ratio),group in profile.groupby(['scene','ratio'],sort=False):
    sv=group[group.stage=='tensor_selector_kernel'].us_per_frame.sum();cv=group[group.stage=='tensor_compensator_kernel_including_decode_publish'].us_per_frame.sum()
    lines.append(f"| {scene.replace('Replica__','').replace('__','/')} | {ratio:g} | {sv:.3f} | {cv:.3f} |")
lines+=['','## 正确性、质量与限制','',
    '- 分辨率2000×2000，保留原Pose与水平/垂直FoV；实际尺寸传入视角编码。4场景18个轨迹视角，双轮×6重复，A/C/D交替顺序。完整/准备/相机各216样本/场景/条件/版本，纯渲染Event72样本。冻结模型/拓扑、相同预算和Strict K保持。',
    f"- {len(c)}个版本/相机/条件通过自己的属性包在A原光栅器与G新光栅器间的RGB/radii逐位及finite检查。候选数及K跨版本相同，C/D选点身份相同；C/D相对A的FP16选点变化单独保留。不能把原光栅器A和网络版本A混为同一个优化模块。",
    f"- 原生1200×680、每场景3个GT轨迹视角共{len(q)}条质量检查。D相对原学生执行A平均额外PSNR损失{v['quality']['D']['mean_loss']:+.8f}dB，最差单帧{v['quality']['D']['worst_frame']:.8f}dB，最差条件均值{v['quality']['D']['worst_condition_mean']:.8f}dB，通过原有平均≤.05dB、场景条件均值≤.2dB门限。没有2000方形原生GT，方形仅记录版本间图像误差，不能宣称方形全轨迹质量已验收。",
    '- 同一蒸馏学生的既有Teacher损失仍存在，本轮只比较执行后端；FP16网络与预投影会引入微差，不能称FP32逐位等价。未新增训练或扩大模型。',
    '- 相机加载、Graph预编译、模型导入、GT读取及质量计算在计时之外；完整路径包括当前相机求逆/上传与必要finite检查，但不含Viewer显示、HTTP、在线SLAM Tracking/Mapping。GPU模型/静态投影有界常驻，地图或权重/Feature/拓扑更新必须重建快照。',
    f"- 正式墙钟与网络组件合计{fj['observations']+cj['observations']}次低频监测，外部活跃GPU样本{summary['foreign_gpu']}，外部单进程>10% CPU样本{summary['foreign_cpu']}；无法排除所有瞬时OS扰动。Profile与正式计时串行，不混用Profiler数据。全部双轮、p95、最大值和慢例保留CSV。",
    '- 驱动首次因保护清单层级路径错误在测速前退出，failed_setup01保留；一次完成后因NumPy bool写JSON失败，failed_finalize01保留，修正驱动并完整重跑；组件首次误用仅D才有的_tensor_state接口，failed_components01保留。均未改冻结后端或数值算法，失败数据不混入正式结果。',
    '- Nsight 2022.4采用cudaProfilerApi延迟捕获时退出139，未得到有效事件，现场保留于failed_profiler_api01；改为全进程采集后用NVTX截取热态。默认Graph级追踪无法显示内部网络Kernel，该诊断保留于graph_level_profile_diagnostic；最终显式启用cuda-graph-trace=node解析网络节点，没有把空事件表当作网络零耗时。',
    f"- {v['protected_files']}份受保护文件SHA前后一致；冻结A/C/D和G二进制按报告核对，只读加载、无JIT。生产、训练、共享conda和checkpoint未改，无commit/push。未保存渲染图片。",'',
    '## 交付','',
    '服务器目录：`/home/prmi/zwj/experiments/tensor_core_square2000_speed_20261111`。分项总表TENSOR_CORE_SEPARATED_RESULTS.csv、微测NETWORK_COMPONENT_SUMMARY.csv、Nsight PROFILE_NETWORKS.csv、原始样本和中文图均保留。']
(HERE/'TENSOR_CORE_SQUARE2000_SPEED_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

font=Path('C:/Windows/Fonts/msyh.ttc');assert font.is_file();fontManager.addfont(str(font));plt.rcParams['font.family']=FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,
    'legend.frameon':False,'axes.unicode_minus':False,'svg.fonttype':'none','savefig.dpi':450})
outputs=[];scenes=list(s.scene.unique());titles=['office1 · 单目','office2 · RGB-D','room0 · RGB-D','room2 · 单目']
def savefig(fig,name):
    for ext in ['png','svg']:
        path=OUT/(name+'.'+ext);fig.savefig(path,dpi=450,facecolor='white');outputs.append(path.name)
    plt.close(fig)

fig,axes=plt.subplots(2,2,figsize=(12,8.6))
for ax,scene,title in zip(axes.flat,scenes,titles):
    x=lod[lod.scene==scene].sort_values('ratio');full=t[(t.scene==scene)&(t.condition=='full')].iloc[0]
    for field,label,color in [('A_pipeline_wall','A：原FP32完整流程','#4E79A7'),('C_pipeline_wall','C：仅Tensor网络完整流程','#F28E2B'),
        ('D_pipeline_wall','D：Tensor＋新准备完整流程','#A23B72'),('D_raster_event','D：纯GPU渲染（不含两网络）','#59A14F')]:
        ax.plot(x.ratio,x[field],'o-',ms=4,lw=1.8,color=color,label=label)
    ax.axhline(full.D_pipeline_wall,color='#6B7280',ls='--',lw=1.5,label='D：Full完整流程（无网络）')
    ax.axhline(full.D_raster_event,color='#59A14F',ls='--',lw=1.5,label='D：Full纯GPU渲染')
    ax.set_title(f'{title}    N={int(full.N):,}',loc='left',fontsize=12,fontweight='bold',pad=12)
    ax.set_ylim(0,max(full.D_pipeline_wall,x.A_pipeline_wall.max())*1.3);ax.set_xlim(0,.735)
    ax.set_xticks([.01,.05,.1,.3,.7],['1%','5%','10%','30%','70%'],rotation=35,ha='right')
    ax.set_xlabel('Ratio（线性轴）');ax.set_ylabel('耗时（ms，越低越好）')
    row=x[np.isclose(x.ratio,.1)].iloc[0]
    ax.text(.03,.96,f'10%：D完整 {row.D_pipeline_wall:.3f} ms\n纯渲染 {row.D_raster_event:.3f} ms',transform=ax.transAxes,va='top',fontsize=9,
        bbox=dict(boxstyle='round,pad=.3',fc='white',ec='#dddddd',alpha=.95))
fig.suptitle('2000×2000：Tensor Core推理与纯渲染分别计时',fontsize=17,fontweight='bold',y=.985)
handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.945),ncol=2)
fig.text(.5,.018,'绿色仅含G光栅过程，属性与相机已就绪；Tensor网络和新准备的收益均未计入渲染时间。\n完整流程独立墙钟；RTX3090；18视角 × 双轮；无平滑；D还含相机/预投影/调度，不能全部归于Tensor Core。',ha='center',fontsize=9,color='#555555')
fig.subplots_adjust(left=.08,right=.975,bottom=.14,top=.78,hspace=.55,wspace=.25);savefig(fig,'TENSOR_CORE_RENDERING_SEPARATED')

fig,axes=plt.subplots(2,2,figsize=(12,8.4))
for ax,scene,title in zip(axes.flat,scenes,titles):
    x=lod[lod.scene==scene].sort_values('ratio')
    for stage,marker,linestyle in [('selector','o','--'),('compensator','s','-')]:
        for variant,color,label in [('A','#4E79A7','A FP32'),('C','#F28E2B','C Tensor'),('D','#A23B72','D Tensor')]:
            suffix='Selector' if stage=='selector' else ('补偿＋全Anchor投影' if variant=='D' else '补偿＋解码')
            ax.plot(x.ratio,x[f'{variant}_{stage}_isolated_us'],color=color,marker=marker,ls=linestyle,lw=1.6,ms=3,label=label+' '+suffix)
    ax.set_title(title,loc='left',fontsize=12,fontweight='bold',pad=10);ax.set_ylim(bottom=0);ax.set_xlim(0,.735)
    ax.set_xticks([.01,.05,.1,.3,.7],['1%','5%','10%','30%','70%'],rotation=35,ha='right')
    ax.set_xlabel('Ratio（线性轴）');ax.set_ylabel('独立网络算子耗时（μs）')
fig.suptitle('Tensor Core两网络单独计时，不计入渲染耗时',fontsize=17,fontweight='bold',y=.985)
handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.945),ncol=2,fontsize=9)
fig.text(.5,.018,'同一候选/选点身份；CUDA Event固定输入Graph，16次调用均摊；包含融合激活/状态，补偿另含解码发布。\nD补偿微测包含全部Anchor动态投影，正式路径仅处理可见Anchor；这些数值不是纯HMMA时间，也不是整帧速度。',ha='center',fontsize=9,color='#555555')
fig.subplots_adjust(left=.08,right=.975,bottom=.14,top=.78,hspace=.55,wspace=.25);savefig(fig,'TENSOR_NETWORKS_ONLY')
sources={path.relative_to(HERE).as_posix():hashlib.sha256(path.read_bytes()).hexdigest() for path in [R/'samples.csv',R/'components/samples.csv',R/'PROFILE_NETWORKS.csv']}
(OUT/'DATA_MANIFEST.json').write_text(json.dumps(dict(real_data=True,dpi=450,smoothing='none',sources=sources,outputs=outputs,
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'data-manifest.md').write_text('# 数据清单\n\n| Figure | Data file | Real/mock | Source | Script | Outputs |\n|---|---|---|---|---|---|\n| 网络/渲染分离 | reports/samples.csv | Real | 本轮A/C/D方形18视角双轮 | analyze.py | TENSOR_CORE_RENDERING_SEPARATED.png / .svg |\n| 网络组件 | reports/components/samples.csv | Real | 固定身份Graph Event | analyze.py | TENSOR_NETWORKS_ONLY.png / .svg |\n\n真实数据，未平滑；纯渲染不含两网络。完整流程/组件独立计时，不机械相加。原数据及脚本SHA见DATA_MANIFEST.json。\n',encoding='utf-8')
for name,expected in sources.items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==expected
print(json.dumps(summary,indent=2))
print(t[t.condition=='local0.1'][['scene','A_pipeline_wall','C_pipeline_wall','D_prepare_wall','D_raster_event','D_pipeline_wall','D_selector_isolated_us','D_compensator_isolated_us']].to_string(index=False))
