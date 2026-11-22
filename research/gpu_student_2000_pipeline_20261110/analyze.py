"""GPU Student双分辨率配对推理的中文报告与曲线。"""
from pathlib import Path
import json,hashlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager,FontProperties
HERE=Path(__file__).resolve().parent;R=HERE/'reports';OUT=HERE/'figures';OUT.mkdir(exist_ok=True)
s=pd.read_csv(R/'samples.csv');checks=pd.read_csv(R/'checks.csv');ids=pd.read_csv(R/'cross_resolution_ids.csv')
v=json.loads((R/'VALIDATION.json').read_text());j=json.loads((HERE/'logs/formal_job.json').read_text())
assert v['passed'] and j['measurement_valid']
assert len(s)==34560 and len(checks)==864 and len(ids)==432
assert checks.rgb_radii_bitwise.all() and ids.ids_same.all() and (ids.K1200==ids.K2000).all()
assert s.groupby(['scene','condition','width','scope']).size().isin([72,216]).all()
assert np.isfinite(s.ms).all() and (s.ms>0).all()
agg=s.groupby(['scene','width','condition','scope']).agg(ms=('ms','median'),p95=('ms',lambda x:x.quantile(.95)),
    maximum=('ms','max'),samples=('ms','size'),N=('N','median'),M=('M','median'),K=('K','median')).reset_index()
agg.to_csv(R/'TIMING_SUMMARY.csv',index=False)
rounds=s.groupby(['scene','width','condition','scope','round']).ms.median().reset_index()
rounds.to_csv(R/'TIMING_ROUNDS.csv',index=False)


def val(scene,width,condition,scope):
    row=agg[(agg.scene==scene)&(agg.width==width)&(agg.condition==condition)&(agg.scope==scope)]
    assert len(row)==1
    return float(row.iloc[0].ms)


rows=[]
for (scene,condition),g in agg.groupby(['scene','condition'],sort=False):
    row=dict(scene=scene,condition=condition,N=int(g.iloc[0].N),M=g.iloc[0].M,K=g.iloc[0].K)
    for width in [1200,2000]:
        for scope in ['prepare_wall','raster_wall','raster_event','pipeline_wall']:row[f'{scope}_{width}']=val(scene,width,condition,scope)
    row['pipeline_added_ms']=row['pipeline_wall_2000']-row['pipeline_wall_1200']
    row['pipeline_time_factor']=row['pipeline_wall_2000']/row['pipeline_wall_1200']
    row['speedup_vs_full_2000']=val(scene,2000,'full','pipeline_wall')/row['pipeline_wall_2000']
    rows.append(row)
t=pd.DataFrame(rows);t.to_csv(R/'GPU_STUDENT_RESOLUTION_RESULTS.csv',index=False)
lod=t[t.condition!='full'];full=t[t.condition=='full']
pair=s[s.scope=='pipeline_wall'].pivot(index=['scene','condition','frame','round','repeat'],columns='width',values='ms')
pair['added_ms']=pair[2000]-pair[1200];pair['factor']=pair[2000]/pair[1200];pair.to_csv(R/'PAIRED_PIPELINE.csv')
summary=dict(passed=True,samples=len(s),checks=len(checks),conditions=len(t),cross_resolution_ids_same=True,
    high_pipeline_ms_min=lod.pipeline_wall_2000.min(),high_pipeline_ms_max=lod.pipeline_wall_2000.max(),
    lod_pipeline_factor_min=lod.pipeline_time_factor.min(),lod_pipeline_factor_max=lod.pipeline_time_factor.max(),
    high_prepare_ms_min=lod.prepare_wall_2000.min(),high_prepare_ms_max=lod.prepare_wall_2000.max(),
    protected_files=v['protected_files'],foreign_gpu=j['foreign_active_samples'],foreign_cpu=j['foreign_cpu_active_samples'])
(R/'ANALYSIS_VALIDATION.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
lines=['# GPU Student在2000像素宽度下的完整推理实测','','## 结论','',
    f"RTX3090、宽度2000像素（2000×1133）、新G渲染器、GPU Selector＋32/32蒸馏Compensator：4场景×5 Ratio的完整推理中位延迟为{lod.pipeline_wall_2000.min():.3f}–{lod.pipeline_wall_2000.max():.3f}ms。包含当前相机准备、候选、Selector、Student Compensator、属性打包、G渲染及同步。",'',
    f"同一批1200×680配对基线下，LoD完整流程变为原来的{lod.pipeline_time_factor.min():.2f}–{lod.pipeline_time_factor.max():.2f}倍，约增加{100*(lod.pipeline_time_factor.min()-1):.1f}%–{100*(lod.pipeline_time_factor.max()-1):.1f}%。GPU前向准备（含CPU相机逆矩阵和提交）在2000宽度约{lod.prepare_wall_2000.min():.3f}–{lod.prepare_wall_2000.max():.3f}ms。这里不能将准备阶段称为Compensator单网耗时。",'',
    '## 高分辨率完整流程','',
    '单位ms；Full直接使用冻结Master属性包，仍每视角执行同样R相机准备；LoD使用当前视角重新运行两个GPU网络。','',
    '| 场景 | 1% | 5% | 10% | 30% | 70% | Full |','|---|---:|---:|---:|---:|---:|---:|']
for scene in s.scene.unique():
    values=[val(scene,2000,c,'pipeline_wall') for c in ['local0.01','local0.05','local0.1','local0.3','local0.7','full']]
    lines.append('| '+scene.replace('Replica/','')+' | '+' | '.join(f'{x:.4f}' for x in values)+' |')
lines+=['','## 同批次1200→2000逐项比较','',
    '各项独立采样，不能将准备中位数与渲染中位数机械相加代替完整流程。纯GPU渲染是CUDA Event；渲染墙钟额外包含提交和同步。','',
    '| 场景 | 条件 | K中位数 | 1200准备 | 2000准备 | 1200 GPU渲染 | 2000 GPU渲染 | 1200完整 | 2000完整 | 完整新增ms | 完整倍数 |',
    '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
for _,r in t.iterrows():
    lines.append(f"| {r.scene.replace('Replica/','')} | {r.condition} | {r.K:g} | {r.prepare_wall_1200:.4f} | {r.prepare_wall_2000:.4f} | {r.raster_event_1200:.4f} | {r.raster_event_2000:.4f} | {r.pipeline_wall_1200:.4f} | {r.pipeline_wall_2000:.4f} | +{r.pipeline_added_ms:.4f} | {r.pipeline_time_factor:.3f}× |")
lines+=['','## 协议与范围','',
    '- 同一学生continuous_calibration/best.pt、Teacher指纹、Master、Selector、Anchor Feature和Ratio；学生3534个MLP参数。两网络计算均在GPU，CPU仍承担位姿求逆、Python/C++调度和必要检查，因此不是无CPU参与。',
    '- 宽度1200→2000，高度680→1133，保持同一FoV/投影/位姿。真实高分辨率camera尺寸传入视角条件，宽高比取整差异约0.029%；每帧重新进行候选、视角编码、选点和补偿，没有沿用1200分辨率的属性或网络结果。',
    '- 使用此前GPU研究的18个轨迹视角（连续＋跳转），双轮×6重复，两个尺寸成对切换顺序。完整/准备/渲染墙钟每条件每尺寸216样本；渲染Event72样本。各场景总地图与逐视角K绑定对应模型。',
    '- 与上一次CPU学生分辨率实验的六个保存相机不同；本轮不通过跨实验相减归因，不把CPU专项版的属性包当作本轮GPU学生输出。',
    '- 相机R及渲染器G在计时前完成加载/预编译/预热。每视角相机求逆、上传、GPU inverse/BMM、候选、网络、选点、打包、必要同步均保留。Full直接复用固定Master，避免每帧整图复制。',
    '- 不含模型/数据加载、GT/质量指标、图像编码、Viewer显示、在线Tracking/Mapping。不能把倒数当作完整SLAM FPS。没有为测速重训或改网络、训练及生产默认。',
    '- Full和LoD均使用相同G/R流程；prepare、raster、pipeline分别独立计量，保留GPU Event与墙钟两种渲染口径。',
    '- 部署v31和冻结输入v3均核对既有.so路径及SHA后加载，G/R复用已冻结验收版本，无JIT或重编译。所有参数、拓扑和模型只读。','',
    '## 正确性、波动与完整性','',
    f"- {len(checks)}个同尺寸输入组合通过A/G RGB/radii逐位及finite检查；两种分辨率的432组相同视角/条件，选点ID和K全部一致。原分辨率R相机张量同时与此前相机API逐位相同。跨分辨率图像不相同，也没有宣称学生与Teacher输出等价。",
    f"- {len(s)}条正式计时全部保留。{j['observations']}次低频负载监测未见外部活跃GPU或单进程>10% CPU负载；仍不能排除所有OS中断/瞬时频率变化。双轮、p95和最大值均保留。",
    '- room0 70%在2000宽度与Full几乎打平，两轮胜负互换，不能将总体中位数约0.00003ms的差值当作稳定收益。office2 30%也接近Full，不能夸大为显著加速。',
    '- 本轮是速度与实现一致性验证，未重新评估GT PSNR；学生既有蒸馏质量损失仍存在，不能把速度收益解释为质量不变。',
    f"- {v['protected_files']}个受保护文件前后SHA一致，未修改生产/共享conda，未commit/push，未保存渲染图片。所有运行位于独立目录`/home/prmi/zwj/experiments/gpu_student_2000_pipeline_20261110`。"]
(HERE/'GPU_STUDENT_2000_PIPELINE_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

font=Path('C:/Windows/Fonts/msyh.ttc');assert font.is_file()
fontManager.addfont(str(font));plt.rcParams['font.family']=FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,
    'legend.frameon':False,'axes.unicode_minus':False,'svg.fonttype':'none','savefig.dpi':450})
fig,axes=plt.subplots(2,2,figsize=(12,8.3))
for ax,scene,title in zip(axes.flat,list(s.scene.unique()),['office1 · 单目','office2 · RGB-D','room0 · RGB-D','room2 · 单目']):
    g=t[(t.scene==scene)&(t.condition!='full')].copy();g['ratio']=g.condition.str[5:].astype(float);g=g.sort_values('ratio')
    f=t[(t.scene==scene)&(t.condition=='full')].iloc[0]
    ax.plot(g.ratio,g.pipeline_wall_1200,'o-',color='#4E79A7',ms=4,lw=1.8,label='1200×680 GPU Student完整流程')
    ax.plot(g.ratio,g.pipeline_wall_2000,'o-',color='#F28E2B',ms=4,lw=1.8,label='2000×1133 GPU Student完整流程')
    ax.axhline(f.pipeline_wall_1200,color='#4E79A7',ls='--',lw=1.5,label='1200×680 Full完整流程')
    ax.axhline(f.pipeline_wall_2000,color='#F28E2B',ls='--',lw=1.5,label='2000×1133 Full完整流程')
    ax.set_title(f'{title}    N={int(f.N):,}',loc='left',fontsize=12,fontweight='bold',pad=12)
    ax.set_ylim(0,max(f.pipeline_wall_2000,g.pipeline_wall_2000.max())*1.28);ax.set_xlim(0,.735)
    ticks=[.01,.05,.1,.3,.7];ax.set_xticks(ticks,[f'{r:.0%}' for r in ticks],rotation=35,ha='right')
    ax.set_xlabel('Ratio（线性轴）');ax.set_ylabel('完整推理耗时（ms）')
    r=g[np.isclose(g.ratio,.1)].iloc[0]
    ax.text(.03,.96,f'10%档：{r.pipeline_wall_1200:.3f} → {r.pipeline_wall_2000:.3f} ms\n新增 {r.pipeline_added_ms:.3f} ms',transform=ax.transAxes,va='top',fontsize=10,
        bbox=dict(boxstyle='round,pad=.3',fc='white',ec='#dddddd',alpha=.95))
fig.suptitle('GPU蒸馏Compensator：宽度2000像素的完整推理',fontsize=17,fontweight='bold',y=.985)
handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.947),ncol=2,fontsize=10)
fig.text(.5,.018,'包含当前相机、候选准备、GPU Selector＋Student、属性打包、G渲染及同步；不含加载、显示或SLAM。\nRTX 3090；18视角 × 双轮 × 6重复；真实配对样本，无平滑；子图纵轴独立。',ha='center',fontsize=9,color='#555555')
fig.subplots_adjust(left=.08,right=.975,bottom=.13,top=.81,hspace=.55,wspace=.25)
outputs=[]
for ext in ['png','svg']:
    path=OUT/('GPU_STUDENT_2000_PIPELINE.'+ext);fig.savefig(path,dpi=450,facecolor='white');outputs.append(path.name)
plt.close(fig)
sources={p.relative_to(HERE).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in [R/'samples.csv',R/'checks.csv',R/'cross_resolution_ids.csv']}
(OUT/'DATA_MANIFEST.json').write_text(json.dumps(dict(real_data=True,sources=sources,dpi=450,smoothing='none',outputs=outputs,
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'data-manifest.md').write_text('# 数据清单\n\n| Figure | Data file | Real/mock | Source | Script | Outputs |\n|---|---|---|---|---|---|\n| GPU Student 2000宽度完整流程 | reports/samples.csv | Real | 本轮18视角双轮配对 | analyze.py | GPU_STUDENT_2000_PIPELINE.png / .svg |\n\n线性Ratio轴，未平滑。完整流程独立实测，不是组件相加；原始样本、两轮及逐条件结果位于reports/。SHA见DATA_MANIFEST.json。\n',encoding='utf-8')
for name,sha in sources.items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==sha
print(json.dumps(summary,indent=2))
