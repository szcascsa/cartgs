"""真实GPU Student三分辨率对照：方形输出的完整推理及纯渲染。"""
from pathlib import Path
import json,hashlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager,FontProperties
HERE=Path(__file__).resolve().parent;R=HERE/'reports';OUT=HERE/'figures';OUT.mkdir(exist_ok=True)
s=pd.read_csv(R/'samples.csv');c=pd.read_csv(R/'checks.csv');ids=pd.read_csv(R/'cross_resolution_ids.csv')
v=json.loads((R/'VALIDATION.json').read_text());job=json.loads((HERE/'logs/formal_job.json').read_text())
assert v['passed'] and job['measurement_valid']
assert len(s)==51840 and len(c)==1296 and len(ids)==864 and c.rgb_radii_bitwise.all()
assert np.isfinite(s.ms).all() and (s.ms>0).all()
assert s.groupby(['scene','width','height','condition','scope']).size().isin([72,216]).all()
s['resolution']=s.width.astype(str)+'x'+s.height.astype(str)
a=s.groupby(['scene','resolution','condition','scope']).agg(ms=('ms','median'),p95=('ms',lambda x:x.quantile(.95)),
    maximum=('ms','max'),N=('N','median'),M=('M','median'),K=('K','median'),samples=('ms','size')).reset_index()
a.to_csv(R/'TIMING_SUMMARY.csv',index=False)
s.groupby(['scene','resolution','condition','scope','round']).ms.median().reset_index().to_csv(R/'TIMING_ROUNDS.csv',index=False)


def val(scene,resolution,condition,scope):
    row=a[(a.scene==scene)&(a.resolution==resolution)&(a.condition==condition)&(a.scope==scope)]
    assert len(row)==1
    return float(row.iloc[0].ms)


rows=[]
for (scene,condition),g in a.groupby(['scene','condition'],sort=False):
    row=dict(scene=scene,condition=condition,N=int(g.iloc[0].N))
    for label,resolution in [('original','1200x680'),('rectangular','2000x1133'),('square','2000x2000')]:
        row[label+'_K']=float(g[g.resolution==resolution].iloc[0].K)
        for scope in ['prepare_wall','raster_wall','raster_event','pipeline_wall']:row[label+'_'+scope]=val(scene,resolution,condition,scope)
    row['square_added_vs_rectangular_ms']=row['square_pipeline_wall']-row['rectangular_pipeline_wall']
    row['square_factor_vs_rectangular']=row['square_pipeline_wall']/row['rectangular_pipeline_wall']
    row['square_factor_vs_original']=row['square_pipeline_wall']/row['original_pipeline_wall']
    row['square_speedup_vs_full']=val(scene,'2000x2000','full','pipeline_wall')/row['square_pipeline_wall']
    rows.append(row)
t=pd.DataFrame(rows);t.to_csv(R/'SQUARE2000_RESULTS.csv',index=False)
sq_ids=ids[ids.height==2000]
assert len(sq_ids)==432
id_summary=sq_ids.groupby(['scene','condition']).agg(frames=('frame','size'),same_id_frames=('ids_same','sum'),
    low_K=('K1200','median'),square_K=('Kcurrent','median'),intersection=('intersection','sum'),union=('union','sum')).reset_index()
id_summary['weighted_jaccard']=id_summary.intersection/id_summary.union.replace(0,1)
id_summary.to_csv(R/'SQUARE_ID_CHANGES.csv',index=False)
lod=t[t.condition!='full'];full=t[t.condition=='full']
summary=dict(passed=True,samples=len(s),checks=len(c),conditions=len(t),width=2000,height=2000,
    square_pipeline_min_ms=lod.square_pipeline_wall.min(),square_pipeline_max_ms=lod.square_pipeline_wall.max(),
    square_prepare_min_ms=lod.square_prepare_wall.min(),square_prepare_max_ms=lod.square_prepare_wall.max(),
    square_factor_vs_rectangular_min=lod.square_factor_vs_rectangular.min(),square_factor_vs_rectangular_max=lod.square_factor_vs_rectangular.max(),
    square_factor_vs_original_min=lod.square_factor_vs_original.min(),square_factor_vs_original_max=lod.square_factor_vs_original.max(),
    square_id_comparisons=len(sq_ids),square_different_id_cases=int((~sq_ids.ids_same).sum()),
    square_different_K_cases=int((sq_ids.K1200!=sq_ids.Kcurrent).sum()),
    protected_files=v['protected_files'],foreign_gpu=job['foreign_active_samples'],foreign_cpu=job['foreign_cpu_active_samples'])
(R/'ANALYSIS_VALIDATION.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
lines=['# GPU Student：2000×2000完整推理实测','','## 结论','',
    f"RTX3090、GPU Selector＋32/32 Student Compensator、新G Renderer，宽和高都为2000时，4场景×5 Ratio的完整推理中位延迟为{lod.square_pipeline_wall.min():.3f}–{lod.square_pipeline_wall.max():.3f}ms。相机＋候选＋两网络＋属性打包的独立准备耗时为{lod.square_prepare_wall.min():.3f}–{lod.square_prepare_wall.max():.3f}ms。完整流程另含当前分辨率渲染和同步。",'',
    f"相对于同批次2000×1133，LoD完整耗时变为{lod.square_factor_vs_rectangular.min():.2f}–{lod.square_factor_vs_rectangular.max():.2f}倍；相对于1200×680变为{lod.square_factor_vs_original.min():.2f}–{lod.square_factor_vs_original.max():.2f}倍。两个倍数均由真实完整样本计算，不通过像素数量或分项相加外推。",'',
    '## 2000×2000完整流程','',
    '单位ms；包含相机、候选、GPU Selector、GPU Student、打包、G渲染及最终同步。Full不运行两网络，固定Master每帧使用当前相机直接渲染。','',
    '| 场景 | 1% | 5% | 10% | 30% | 70% | Full |','|---|---:|---:|---:|---:|---:|---:|']
for scene in s.scene.unique():
    values=[val(scene,'2000x2000',condition,'pipeline_wall') for condition in ['local0.01','local0.05','local0.1','local0.3','local0.7','full']]
    lines.append('| '+scene.replace('Replica/','')+' | '+' | '.join(f'{x:.4f}' for x in values)+' |')
lines+=['','## 各分辨率逐项对照','',
    '| 场景 | 条件 | 方形K中位数 | 原图完整 | 2000×1133完整 | 2000×2000完整 | 方形准备 | 方形纯GPU渲染 | 相比2000×1133新增 |',
    '|---|---|---:|---:|---:|---:|---:|---:|---:|']
for _,r in t.iterrows():
    lines.append(f"| {r.scene.replace('Replica/','')} | {r.condition} | {r.square_K:g} | {r.original_pipeline_wall:.4f} | {r.rectangular_pipeline_wall:.4f} | {r.square_pipeline_wall:.4f} | {r.square_prepare_wall:.4f} | {r.square_raster_event:.4f} | +{r.square_added_vs_rectangular_ms:.4f} |")
lines+=['','## 相机、宽高比与解释边界','',
    '- 输出确实为2000×2000。保持原Pose、水平/垂直FoV和投影矩阵，即保持相同世界视野、不额外扩展垂直可见区域。等效于分别缩放fx和fy；方形像素画布的显示比例变化，不是裁剪成方形或增加真实观测。',
    '- 网络本来接收实际image_width/image_height；2000×2000时该输入为1，2000×1133约为1.765，因此此次每个视角都真实重新运行候选、Selector、视角编码和Compensator，没有冻结旧预测。',
    f"- 方形与原尺寸共{len(sq_ids)}个同视角/条件对照，{int((~sq_ids.ids_same).sum())}个选点集合发生变化，{int((sq_ids.K1200!=sq_ids.Kcurrent).sum())}个K发生变化；逐场景交并比见SQUARE_ID_CHANGES.csv。即使K不变，所选点或补偿几何变化仍可能影响渲染工作量。",
    '- 因此这是用户指定尺寸下的实际网络＋渲染速度，不是完全固定Gaussian属性的单一像素数因果实验，也不是高分辨率GT质量验证。若另需严格隔离像素成本，须额外锁定属性包；本轮未混称两种口径。',
    '- 4场景、18个轨迹视角、双轮×6重复，三尺寸轮换提交顺序；各条件各尺寸准备/渲染墙钟/完整墙钟216样本，纯GPU渲染Event72样本。所有计划及尺寸预热，逐帧重算，没有复用旧帧结果。',
    '- 准备、纯渲染及完整流程均独立采样，不把分项中位数相加替代完整延迟；纯渲染Event不含相机/候选/网络/上传。完整流程含CPU相机逆矩阵和调度，但两网络都在GPU。',
    '- 不含模型读取、磁盘IO、相机外部获取、GT指标、显示或在线SLAM Tracking/Mapping；倒数不是完整SLAM FPS。',
    '- 使用相同3534参数学生、Teacher绑定、Master、Selector、Anchor Feature、Ratio和冻结.so；没有训练、重编译、修改生产默认或改变网络。','',
    '## 验证与完整性','',
    f"- {len(c)}个场景/相机/Ratio/尺寸组合均通过同尺寸A与G的RGB/radii逐位及finite检查。原尺寸R相机继续与旧API逐位一致；不声称不同尺寸或Teacher/Student之间逐位一致。",
    f"- {job['observations']}次负载采样中，外部活跃GPU为{job['foreign_active_samples']}，外部单进程>10% CPU为{job['foreign_cpu_active_samples']}；低频采样仍有局限。全部正式样本保留，双轮、p95和最大值见CSV。",
    '- 首次尝试通过数值验收，但监控一次捕捉到外部PID2416105约13.9% CPU活动，第二次出现外部GPU活动；两次性能数据整轮未采用，现场保留于rejected_attempt01/02，没有只删慢样本或停止外部进程。',
    f"- {v['protected_files']}个受保护源码/配置/模型/二进制前后SHA无变化；未修改生产、共享conda或历史结果，未commit/push，未保存渲染图片。",
    '- 图像质量未新增GT评价，既有蒸馏损失仍适用；速度提升与渲染质量是不同问题。','',
    '目录：`/home/prmi/zwj/experiments/gpu_student_square2000_20261110`。原始数据、尺寸、模型身份、逐位检查和完整性清单保存在reports；进程身份及负载在logs。']
(HERE/'GPU_STUDENT_SQUARE2000_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

font=Path('C:/Windows/Fonts/msyh.ttc');assert font.is_file()
fontManager.addfont(str(font));plt.rcParams['font.family']=FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,
    'legend.frameon':False,'axes.unicode_minus':False,'svg.fonttype':'none','savefig.dpi':450})
fig,axes=plt.subplots(2,2,figsize=(12,8.3))
for ax,scene,title in zip(axes.flat,list(s.scene.unique()),['office1 · 单目','office2 · RGB-D','room0 · RGB-D','room2 · 单目']):
    g=t[(t.scene==scene)&(t.condition!='full')].copy();g['ratio']=g.condition.str[5:].astype(float);g=g.sort_values('ratio')
    f=t[(t.scene==scene)&(t.condition=='full')].iloc[0]
    for field,label,color in [('original_pipeline_wall','1200×680 GPU Student','#4E79A7'),('rectangular_pipeline_wall','2000×1133 GPU Student','#F28E2B'),('square_pipeline_wall','2000×2000 GPU Student','#A23B72')]:
        ax.plot(g.ratio,g[field],'o-',color=color,ms=4,lw=1.8,label=label)
    ax.axhline(f.square_pipeline_wall,color='#6B7280',ls='--',lw=1.5,label='2000×2000 Full')
    ax.set_title(f'{title}    N={int(f.N):,}',loc='left',fontsize=12,fontweight='bold',pad=12)
    ax.set_ylim(0,max(f.square_pipeline_wall,g.square_pipeline_wall.max())*1.28);ax.set_xlim(0,.735)
    ticks=[.01,.05,.1,.3,.7];ax.set_xticks(ticks,[f'{r:.0%}' for r in ticks],rotation=35,ha='right')
    ax.set_xlabel('Ratio（线性轴）');ax.set_ylabel('完整推理耗时（ms）')
    row=g[np.isclose(g.ratio,.1)].iloc[0]
    ax.text(.03,.96,f'方形10%：{row.square_pipeline_wall:.3f} ms\n方形Full：{f.square_pipeline_wall:.3f} ms',transform=ax.transAxes,va='top',fontsize=10,
        bbox=dict(boxstyle='round,pad=.3',fc='white',ec='#dddddd',alpha=.95))
fig.suptitle('宽高均为2000：GPU蒸馏版完整推理实测',fontsize=17,fontweight='bold',y=.985)
handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.947),ncol=2)
fig.text(.5,.018,'相机FoV固定；网络按实际尺寸重新运行，方形宽高比输入为1；含候选、两网络、打包、渲染和同步。\nRTX 3090；18视角 × 双轮 × 6重复；不含加载、显示或SLAM；原始实测，无平滑。',ha='center',fontsize=9,color='#555555')
fig.subplots_adjust(left=.08,right=.975,bottom=.13,top=.81,hspace=.55,wspace=.25)
outputs=[]
for ext in ['png','svg']:
    path=OUT/('GPU_STUDENT_SQUARE2000.'+ext);fig.savefig(path,dpi=450,facecolor='white');outputs.append(path.name)
plt.close(fig)
sources={p.relative_to(HERE).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in [R/'samples.csv',R/'checks.csv',R/'cross_resolution_ids.csv']}
(OUT/'DATA_MANIFEST.json').write_text(json.dumps(dict(real_data=True,sources=sources,dpi=450,smoothing='none',outputs=outputs,
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'data-manifest.md').write_text('# 数据清单\n\n| Figure | Data file | Real/mock | Source | Script | Outputs |\n|---|---|---|---|---|---|\n| 三种尺寸GPU Student完整推理 | reports/samples.csv | Real | 本轮三尺寸18视角双轮 | analyze.py | GPU_STUDENT_SQUARE2000.png / .svg |\n\n线性Ratio轴，未平滑；完整流程独立实测，不是分项相加。输入、脚本SHA见DATA_MANIFEST.json。\n',encoding='utf-8')
for name,sha in sources.items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==sha
print(json.dumps(summary,indent=2))
