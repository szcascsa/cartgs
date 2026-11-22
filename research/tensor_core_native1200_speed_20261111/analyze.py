"""正常1200×680原生分辨率，实际A/C/D和两网络微测。"""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager,FontProperties
HERE=Path(__file__).resolve().parent;R=HERE/'reports';OUT=HERE/'figures';OUT.mkdir(exist_ok=True)
s=pd.read_csv(R/'samples.csv');checks=pd.read_csv(R/'checks.csv');quality=pd.read_csv(R/'quality_native.csv')
u=pd.read_csv(R/'components/samples.csv');uc=pd.read_csv(R/'components/checks.csv')
v=json.loads((R/'VALIDATION.json').read_text());fj=json.loads((HERE/'logs/formal_job.json').read_text());cj=json.loads((HERE/'logs/components_job.json').read_text())
assert v['passed'] and fj['measurement_valid'] and cj['measurement_valid']
assert len(s)==67392 and len(checks)==1296 and len(quality)==216 and len(u)==11520 and len(uc)==720
assert checks.rgb_radii_bitwise.all() and np.isfinite(s.ms).all() and np.isfinite(u.ms).all()
assert checks.groupby(['scene','frame','condition']).K.nunique().eq(1).all()
assert checks.groupby(['scene','frame','condition']).M.nunique().eq(1).all()
assert checks[checks.variant.isin(['C','D'])].groupby(['scene','frame','condition']).ids_sha256.nunique().eq(1).all()
g=s.groupby(['scene','condition','variant','scope']).agg(ms=('ms','median'),p95=('ms',lambda x:x.quantile(.95)),
    maximum=('ms','max'),N=('N','median'),M=('M','median'),K=('K','median'),samples=('ms','size')).reset_index()
g.to_csv(R/'TIMING_SUMMARY.csv',index=False)
s.groupby(['scene','condition','variant','scope','round']).ms.median().reset_index().to_csv(R/'TIMING_ROUNDS.csv',index=False)
ug=u.groupby(['scene','ratio','variant','stage']).agg(ms=('ms','median'),p95=('ms',lambda x:x.quantile(.95)),
    N=('N','median'),M=('M','median'),K=('K','median'),samples=('ms','size')).reset_index()
ug.to_csv(R/'NETWORK_COMPONENT_SUMMARY.csv',index=False)
quality.groupby(['scene','condition','variant']).agg(psnr=('psnr','mean'),mean_loss=('psnr_loss_vs_A','mean'),
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
    row['D_pipeline_vs_D_full']=val(scene,'full','D','pipeline_wall')/row['D_pipeline_wall']
    if condition!='full':
        ratio=float(condition[5:]);row['ratio']=ratio
        for variant in ['A','C','D']:
            for stage in ['selector','compensator']:
                x=ug[(ug.scene==scene)&(ug.ratio==ratio)&(ug.variant==variant)&(ug.stage==stage)];assert len(x)==1
                row[variant+'_'+stage+'_isolated_us']=float(x.iloc[0].ms)*1000
    rows.append(row)
t=pd.DataFrame(rows);t.to_csv(R/'NATIVE1200_SEPARATED_RESULTS.csv',index=False)
paired=s[s.scope=='pipeline_wall'].pivot(index=['scene','condition','round','repeat','frame'],columns='variant',values='ms')
paired['D_vs_A']=paired.A/paired.D;paired['C_vs_A']=paired.A/paired.C;paired['D_vs_C']=paired.C/paired.D
paired.to_csv(R/'PAIRED_PIPELINE.csv',index=True)
def geo(x):return float(np.exp(np.log(x).mean()))
lod=t[t.condition!='full'];pl=paired.reset_index();pl=pl[pl.condition!='full']
summary=dict(passed=True,resolution=[1200,680],samples=len(s),checks=len(checks),quality_rows=len(quality),
    network_samples=len(u),network_checks=len(uc),protected_files=v['protected_files'],
    D_pipeline_min_ms=lod.D_pipeline_wall.min(),D_pipeline_max_ms=lod.D_pipeline_wall.max(),
    D_prepare_min_ms=lod.D_prepare_wall.min(),D_prepare_max_ms=lod.D_prepare_wall.max(),
    D_raster_min_ms=lod.D_raster_event.min(),D_raster_max_ms=lod.D_raster_event.max(),
    D_vs_A_pipeline=geo(pl.D_vs_A),C_vs_A_pipeline=geo(pl.C_vs_A),D_vs_C_pipeline=geo(pl.D_vs_C),
    quality=v['quality'],foreign_gpu=fj['foreign_active_samples']+cj['foreign_active_samples'],
    foreign_cpu=fj['foreign_cpu_active_samples']+cj['foreign_cpu_active_samples'])
(R/'ANALYSIS_VALIDATION.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
lines=['# 正常分辨率1200×680：Tensor Core方案速度复测','','## 结论','',
    f"RTX3090、1200×680、同一GPU Student和新G Renderer，最终D方案在4场景×5 Ratio的完整推理中位耗时为{lod.D_pipeline_wall.min():.3f}–{lod.D_pipeline_wall.max():.3f}ms，完整准备约{lod.D_prepare_wall.min():.3f}–{lod.D_prepare_wall.max():.3f}ms，纯GPU渲染约{lod.D_raster_event.min():.3f}–{lod.D_raster_event.max():.3f}ms。",'',
    f"同批次逐视角配对D/A完整加速为{summary['D_vs_A_pipeline']:.3f}×；仅Tensor网络C/A为{summary['C_vs_A_pipeline']:.3f}×；D/C为{summary['D_vs_C_pipeline']:.3f}×。D还含相机、准备、预投影和调度，不能把全部收益记为Tensor Core。",'',
    '## 计时范围','',
    '- A为原v31融合FP32学生＋R相机；C只换FP16 Tensor网络、FP32累加；D使用C网络，加CUDA视锥、FP32输入预投影、GPU动态N/K与C++当前相机。都是相同3534参数学生、Master、Selector、Anchor Feature及Strict K。',
    '- 正常分辨率为Replica原生1200×680，FoV/Pose保持实际原设置，网络接收当前宽高比1200/680；没有从2000×2000按像素比例估计。每帧重新计算相机、候选、两网络、选点和补偿，模型与静态投影只在加载时建立冻结快照。',
    '- 新准备总时间是当前相机＋候选＋两网络＋选择＋打包＋必要同步，已经含网络；地图/网络准备engine_prepare排除相机但仍含网络。网络组件另用CUDA Event微测，不能重复累加。',
    '- 纯渲染仅属性和相机已就绪后的G光栅CUDA Event；不含推理、H2D或CPU提交。完整流程是单独墙钟，包括相机、整个准备、G渲染及同步。各列中位数不能相加代替完整延迟。',
    '- Full固定Master属性包GPU常驻，仍执行当前相机准备，不调用两网络、不每帧复制整图。Full不是在线SLAM，所有计时也不含训练、Tracking/Mapping、模型读取、GT指标、显示或HTTP。','',
    '## D逐场景逐Ratio','',
    '| 场景 | Ratio | K中位数 | TC Selector μs | TC补偿算子 μs | 新准备总ms | 纯渲染ms | 完整ms |',
    '|---|---:|---:|---:|---:|---:|---:|---:|']
for _,r in t.iterrows():
    if r.condition=='full':sel=comp='—'
    else:sel=f'{r.D_selector_isolated_us:.2f}';comp=f'{r.D_compensator_isolated_us:.2f}'
    lines.append(f"| {r.scene.replace('Replica/','')} | {r.condition} | {r.K:g} | {sel} | {comp} | {r.D_prepare_wall:.4f} | {r.D_raster_event:.4f} | {r.D_pipeline_wall:.4f} |")
lines+=['','## A/C/D完整流程对照','',
    '| 场景 | Ratio | A完整ms | C完整ms | D完整ms | D/A倍速 |','|---|---:|---:|---:|---:|---:|']
for _,r in t.iterrows():lines.append(f"| {r.scene.replace('Replica/','')} | {r.condition} | {r.A_pipeline_wall:.4f} | {r.C_pipeline_wall:.4f} | {r.D_pipeline_wall:.4f} | {r.pipeline_speedup_D_vs_A:.3f}× |")
lines+=['','## 验证与局限','',
    '- 同此前18个轨迹视角（连续＋跳转）、同Ratios=1%/5%/10%/30%/70%，双轮×6重复，A/C/D交替顺序。每版本每场景每条件墙钟216样本、纯渲染Event72样本，原始数据、双轮、p95及最大值全部保留。',
    f"- {len(checks)}组版本自身属性包通过原A光栅与G的RGB/radii逐位及finite比较；跨版本候选数和K相同，C/D选点身份一致。不同版本的网络精度并不逐位一致，差异记录在checks.csv。",
    f"- 原生GT的{len(quality)}项配对质量中，D相对A平均额外PSNR损失{v['quality']['D']['mean_loss']:+.8f}dB，最差单帧{v['quality']['D']['worst_frame']:.8f}dB，最差条件均值{v['quality']['D']['worst_condition_mean']:.8f}dB，沿用平均≤.05dB、条件均值≤.2dB门限。不是全轨迹质量验收，也不能抵消学生本身相对Teacher的已有损失。",
    '- 两网络单独计时在六个固定视角上使用同候选/同选点身份，Graph重复16次后CUDA Event均摊。Selector含读取、激活、输出/状态；补偿含物理解码发布，D还含全Anchor动态投影。正式D只处理可见Anchor，此微测是保守边界，不是单独HMMA指令时间或真实整帧分解。',
    f"- 正式主测试和组件测试合计{fj['observations']+cj['observations']}次低频监测，外部活跃GPU样本{summary['foreign_gpu']}，外部单进程>10% CPU样本{summary['foreign_cpu']}。低频监测仍不能排除所有瞬时OS调度；每次复测结论只使用本轮配对样本。",
    '- 本轮未修改Kernel，因此不重复已完成的Nsight/HMMA实现归因，网络微测与完整路径均为本轮重新运行，不套用方形或旧报告数值。静态投影随地图、权重、Feature或拓扑改变须重建冻结实例。',
    f"- {v['protected_files']}个受保护文件SHA前后无变化，A/C/D和G均核对冻结版本后只读加载，无JIT/编译。未修改生产默认、训练、模型、共享conda或历史结果，未commit/push，没有渲染图片保存。",'',
    '服务器目录：`/home/prmi/zwj/experiments/tensor_core_native1200_speed_20261111`。完整分项CSV为reports/NATIVE1200_SEPARATED_RESULTS.csv。']
(HERE/'TENSOR_CORE_NATIVE1200_SPEED_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

font=Path('C:/Windows/Fonts/msyh.ttc');assert font.is_file();fontManager.addfont(str(font));plt.rcParams['font.family']=FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,
    'legend.frameon':False,'axes.unicode_minus':False,'svg.fonttype':'none','savefig.dpi':450})
outputs=[]
fig,axes=plt.subplots(2,2,figsize=(12,8.6))
for ax,scene,title in zip(axes.flat,list(s.scene.unique()),['office1 · 单目','office2 · RGB-D','room0 · RGB-D','room2 · 单目']):
    x=lod[lod.scene==scene].sort_values('ratio');f=t[(t.scene==scene)&(t.condition=='full')].iloc[0]
    for field,label,color in [('A_pipeline_wall','A：原FP32完整流程','#4E79A7'),('C_pipeline_wall','C：仅Tensor网络完整流程','#F28E2B'),
        ('D_pipeline_wall','D：Tensor＋新准备完整流程','#A23B72'),('D_raster_event','D：纯GPU渲染（不含两网络）','#59A14F')]:
        ax.plot(x.ratio,x[field],'o-',color=color,ms=4,lw=1.8,label=label)
    ax.axhline(f.D_pipeline_wall,color='#6B7280',ls='--',lw=1.5,label='D：Full完整流程（无网络）')
    ax.axhline(f.D_raster_event,color='#59A14F',ls='--',lw=1.5,label='D：Full纯GPU渲染')
    ax.set_title(f'{title}    N={int(f.N):,}',loc='left',fontsize=12,fontweight='bold',pad=12)
    ax.set_ylim(0,max(f.D_pipeline_wall,x.A_pipeline_wall.max())*1.3);ax.set_xlim(0,.735)
    ax.set_xticks([.01,.05,.1,.3,.7],['1%','5%','10%','30%','70%'],rotation=35,ha='right')
    ax.set_xlabel('Ratio（线性轴）');ax.set_ylabel('耗时（ms，越低越好）')
    r=x[np.isclose(x.ratio,.1)].iloc[0]
    ax.text(.03,.96,f'10%：D完整 {r.D_pipeline_wall:.3f} ms\n新准备 {r.D_prepare_wall:.3f} / 纯渲染 {r.D_raster_event:.3f} ms',transform=ax.transAxes,va='top',fontsize=9,
        bbox=dict(boxstyle='round,pad=.3',fc='white',ec='#dddddd',alpha=.95))
fig.suptitle('正常分辨率1200×680：Tensor Core推理与纯渲染',fontsize=17,fontweight='bold',y=.985)
h,l=axes.flat[0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',bbox_to_anchor=(.5,.945),ncol=2)
fig.text(.5,.018,'绿色仅GPU光栅CUDA Event；完整流程包含当前相机、候选、两网络、打包与同步，均为独立实测。\nRTX3090；18视角 × 双轮 × 6重复；无平滑；D还含相机/预投影/调度，不将全部收益归于Tensor Core。',ha='center',fontsize=9,color='#555555')
fig.subplots_adjust(left=.08,right=.975,bottom=.14,top=.78,hspace=.55,wspace=.25)
for ext in ['png','svg']:
    path=OUT/('TENSOR_CORE_NATIVE1200_SEPARATED.'+ext);fig.savefig(path,dpi=450,facecolor='white');outputs.append(path.name)
plt.close(fig)
sources={f.relative_to(HERE).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in [R/'samples.csv',R/'components/samples.csv']}
(OUT/'DATA_MANIFEST.json').write_text(json.dumps(dict(real_data=True,dpi=450,smoothing='none',sources=sources,outputs=outputs,
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'data-manifest.md').write_text('# 数据清单\n\n| Figure | Data file | Real/mock | Source | Script | Outputs |\n|---|---|---|---|---|---|\n| 正常分辨率推理/渲染分离 | reports/samples.csv | Real | 本轮1200×680 A/C/D双轮 | analyze.py | TENSOR_CORE_NATIVE1200_SEPARATED.png / .svg |\n\n原始数据，未平滑。网络微测另列CSV，不计入纯渲染。完整流程独立墙钟；源SHA见JSON。\n',encoding='utf-8')
for name,expected in sources.items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==expected
print(json.dumps(summary,indent=2))
print(t[t.condition=='local0.1'][['scene','A_pipeline_wall','D_prepare_wall','D_raster_event','D_pipeline_wall','D_selector_isolated_us','D_compensator_isolated_us']].to_string(index=False))
