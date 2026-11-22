"""1200→2000宽度的真实配对渲染结果、中文报告与450dpi图。"""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager,FontProperties
HERE=Path(__file__).resolve().parent;R=HERE/'reports';OUT=HERE/'figures';OUT.mkdir(exist_ok=True)
source=R/'samples.csv';s=pd.read_csv(source);checks=pd.read_csv(R/'checks.csv')
v=json.loads((R/'VALIDATION.json').read_text());j=json.loads((HERE/'logs/formal_job.json').read_text())
assert v['passed'] and j['measurement_valid']
assert len(s)==10368 and len(checks)==432 and checks.rgb_radii_bitwise.all()
assert checks.groupby(['scene','condition','frame']).ids_sha256.nunique().eq(1).all()
assert s.groupby(['scene','condition','frame']).K.nunique().eq(1).all()
assert s.groupby(['scene','condition','width']).size().eq(144).all()
assert s.groupby(['scene','condition','frame','round','repeat']).size().eq(2).all()
agg=s.groupby(['scene','condition','width']).agg(gpu_ms=('gpu_ms','median'),wall_ms=('wall_ms','median'),
    gpu_p95=('gpu_ms',lambda v:v.quantile(.95)),wall_p95=('wall_ms',lambda v:v.quantile(.95)),
    gpu_max=('gpu_ms','max'),wall_max=('wall_ms','max'),N=('N','median'),K=('K','median')).reset_index()
agg.to_csv(R/'RESOLUTION_TIMING_SUMMARY.csv',index=False)
rounds=s.groupby(['scene','condition','width','round'])[['gpu_ms','wall_ms']].median().reset_index()
rounds.to_csv(R/'RESOLUTION_TIMING_ROUNDS.csv',index=False)
paired=s.pivot(index=['scene','condition','frame','round','repeat'],columns='width',values=['gpu_ms','wall_ms'])
rows=[]
for (scene,condition),g in agg.groupby(['scene','condition'],sort=False):
    low=g[g.width==1200].iloc[0];high=g[g.width==2000].iloc[0];pair=paired.loc[scene,condition]
    rows.append(dict(scene=scene,condition=condition,N=int(low.N),K=low.K,
        low_gpu_ms=low.gpu_ms,high_gpu_ms=high.gpu_ms,added_gpu_ms=high.gpu_ms-low.gpu_ms,
        gpu_time_factor=high.gpu_ms/low.gpu_ms,gpu_paired_added_ms=(pair.gpu_ms[2000]-pair.gpu_ms[1200]).median(),
        gpu_paired_factor=(pair.gpu_ms[2000]/pair.gpu_ms[1200]).median(),
        low_wall_ms=low.wall_ms,high_wall_ms=high.wall_ms,added_wall_ms=high.wall_ms-low.wall_ms,
        wall_time_factor=high.wall_ms/low.wall_ms))
t=pd.DataFrame(rows);t.to_csv(R/'RESOLUTION_PAIRED_RESULTS.csv',index=False)
full=t[t.condition=='full'];lod=t[t.condition!='full']
summary=dict(passed=True,samples=len(s),checks=len(checks),conditions=len(t),pixel_factor=2000*1133/(1200*680),
    gpu_time_factor_min=t.gpu_time_factor.min(),gpu_time_factor_max=t.gpu_time_factor.max(),
    lod_gpu_time_factor_min=lod.gpu_time_factor.min(),lod_gpu_time_factor_max=lod.gpu_time_factor.max(),
    full_gpu_time_factor_min=full.gpu_time_factor.min(),full_gpu_time_factor_max=full.gpu_time_factor.max(),
    added_gpu_ms_min=t.added_gpu_ms.min(),added_gpu_ms_max=t.added_gpu_ms.max(),
    foreign_gpu=j['foreign_active_samples'],foreign_cpu=j['foreign_cpu_active_samples'],
    protected_files=v['protected_files'],cpu_inference_included=False)
(R/'ANALYSIS_VALIDATION.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
lines=['# 宽度2000像素的渲染分辨率对照实测','','## 结论','',
    f"将1200×680改为2000×1133，像素数为原来的{summary['pixel_factor']:.4f}倍。4场景、8档LoD Ratio＋Full共36个条件，纯GPU渲染中位耗时变为原来的{t.gpu_time_factor.min():.2f}–{t.gpu_time_factor.max():.2f}倍，新增{t.added_gpu_ms.min():.3f}–{t.added_gpu_ms.max():.3f}ms。该范围是本轮固定输入实际测速，不是按像素数外推。",'',
    f"Full耗时为原来的{full.gpu_time_factor.min():.2f}–{full.gpu_time_factor.max():.2f}倍；LoD为{lod.gpu_time_factor.min():.2f}–{lod.gpu_time_factor.max():.2f}倍。CPU推理与H2D不在渲染计时内。保持相同Gaussian属性、选点及相机条件时，CPU网络计算量与属性上传字节数不随图像像素数增加；本轮没有把前次CPU样本和新GPU样本相加宣称完整流程收益。",'',
    '## 控制条件与计时','',
    '- “2000”按宽度2000像素理解，保持原宽高比并将高度四舍五入为1133；不是2000×2000。保持同一Pose、FoV、投影矩阵，像素分辨率变化由Rasterizer尺寸参数表达。',
    '- 沿用CPU蒸馏专项context版本（2/1/1）、同一Master、Selector及学生。先在原分辨率准备并冻结属性包，两个尺寸使用完全相同的Tensor和Gaussian顺序，仅改变图像宽高，隔离纯渲染分辨率成本。没有重新选点、重训或调整补偿。',
    '- 图像高度取整使宽高比有约0.029%的差别；这里保持FoV及网络输入固定。真实应用若因新宽高比重新构造视角编码，可能额外产生极小输入差异，不能将本测试称所有部署输出不变。',
    '- RTX3090、已冻结新G Renderer及完整容量Graph计划；两种尺寸分别在计时前预编译和预热。每帧依然重新处理当前高斯和相机，未缓存渲染图像。',
    '- 4场景各3连续＋3跳转保存相机，双轮×12重复；每个条件每种尺寸144样本。每对尺寸轮换顺序，第二轮反转相机与条件顺序，保留全部样本。',
    '- CUDA Event计量完整GPU渲染，独立墙钟计量渲染调用和最终同步；两者均不含CPU网络推理、属性H2D、模型读取、相机矩阵构造、Graph预编译、GT质量指标及显示。本次没有测完整在线SLAM。','',
    '## 逐场景逐Ratio结果','',
    '单位ms；两尺寸使用相同实际K。表中增加和倍数由对应原始样本中位数计算，另存逐对差值和倍数中位数。','',
    '| 场景 | 条件 | 实际K中位数 | 1200×680 GPU | 2000×1133 GPU | 新增ms | 耗时倍数 | 1200墙钟 | 2000墙钟 |',
    '|---|---|---:|---:|---:|---:|---:|---:|---:|']
for _,r in t.iterrows():
    lines.append(f"| {r.scene.replace('Replica__','').replace('__','/')} | {r.condition} | {r.K:g} | {r.low_gpu_ms:.4f} | {r.high_gpu_ms:.4f} | +{r.added_gpu_ms:.4f} | {r.gpu_time_factor:.3f}× | {r.low_wall_ms:.4f} | {r.high_wall_ms:.4f} |")
lines+=['','## 正确性、完整性与局限','',
    f"- {len(checks)}份属性/相机/尺寸组合通过新G与原A RGB/radii逐位比较，并核对finite；相同视角同Ratio在两尺寸的Gaussian ID和K相同。跨分辨率图像本来不同，没有做或声称跨尺寸逐位一致。",
    f"- 正式监控{j['observations']}次，外部活跃GPU样本{j['foreign_active_samples']}，外部单进程>10% CPU样本{j['foreign_cpu_active_samples']}。外部评价进程当时已经处于T暂停状态，本轮没有操作其状态。低频监控不能排除所有瞬时调度和硬件频率变化。",
    f"- {v['protected_files']}个受保护源码/配置/模型/二进制前后SHA不变；生产默认、共享环境和已有结果未修改；没有commit/push，没有渲染图片保存。使用已有二进制，无编译。",
    '- GPU耗时不等于像素数量成比例增加：高斯投影预处理、排序与Host调度不完全随像素数变化，像素合成与输出写入则随分辨率增加。本轮只提供总渲染实测，没有新增Nsight分项归因。',
    '- 分辨率改变会影响高斯屏幕footprint、Tile覆盖和Alpha截断；固定3D属性并不代表渲染像素计算量仅按一个比例缩放。',
    '- 本轮不是高分辨率图像质量验证，也不能由六个保存相机推断全轨迹或其他GPU性能。当前仅在2000宽度验证，不外推4K、方形或移动端。','',
    '## 交付','',
    '原始samples/checks、双轮TIMING_ROUNDS、逐条件汇总和配对结果位于reports；图为figures/RESOLUTION_1200_VS_2000.png及.svg。服务器目录：`/home/prmi/zwj/experiments/render_resolution_2000_20261110`。']
(HERE/'RENDER_RESOLUTION_2000_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

font=Path('C:/Windows/Fonts/msyh.ttc')
assert font.is_file();fontManager.addfont(str(font));plt.rcParams['font.family']=FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,
    'legend.frameon':False,'axes.unicode_minus':False,'svg.fonttype':'none','savefig.dpi':450})
fig,axes=plt.subplots(2,2,figsize=(12,8.3))
for ax,scene,title in zip(axes.flat,list(s.scene.unique()),['office1 · 单目','office2 · RGB-D','room0 · RGB-D','room2 · 单目']):
    g=t[(t.scene==scene)&(t.condition!='full')].copy();g['ratio']=g.condition.str[5:].astype(float);g=g.sort_values('ratio')
    f=t[(t.scene==scene)&(t.condition=='full')].iloc[0]
    for field,label,color in [('low_gpu_ms','1200×680 LoD','#4E79A7'),('high_gpu_ms','2000×1133 LoD','#F28E2B')]:
        ax.plot(g.ratio,g[field],'o-',color=color,lw=1.8,ms=4,label=label)
    ax.axhline(f.low_gpu_ms,color='#4E79A7',ls='--',lw=1.5,label='1200×680 Full')
    ax.axhline(f.high_gpu_ms,color='#F28E2B',ls='--',lw=1.5,label='2000×1133 Full')
    ax.set_title(f'{title}    N={int(f.N):,}',loc='left',pad=10,fontweight='bold',fontsize=12)
    ax.set_xlabel('Ratio（线性轴）');ax.set_ylabel('纯GPU渲染耗时（ms）');ax.set_ylim(0,f.high_gpu_ms*1.5);ax.set_xlim(0,.735)
    ticks=[.01,.05,.1,.2,.3,.4,.5,.7];ax.set_xticks(ticks,[f'{x:.0%}' for x in ticks],rotation=45,ha='right')
    r=g[np.isclose(g.ratio,.1)].iloc[0]
    ax.text(.03,.94,f'Full：+{f.added_gpu_ms:.3f} ms（{f.gpu_time_factor:.2f}×）\nLoD 10%：+{r.added_gpu_ms:.3f} ms（{r.gpu_time_factor:.2f}×）',
        transform=ax.transAxes,va='top',fontsize=10,bbox=dict(boxstyle='round,pad=.3',fc='white',ec='#dddddd',alpha=.95))
fig.suptitle('宽度1200 → 2000像素：渲染时间实际增加多少',fontsize=17,fontweight='bold',y=.985)
handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.943),ncol=4)
fig.text(.5,.018,'2000像素指2000×1133，像素数为原来的2.78倍；相机FoV、Gaussian选点及补偿属性固定。\n仅GPU渲染CUDA Event，不含CPU推理或上传；双轮真实样本，无平滑；各子图纵轴独立。',ha='center',fontsize=9,color='#555555')
fig.subplots_adjust(left=.08,right=.975,bottom=.13,top=.84,hspace=.54,wspace=.25)
outputs=[]
for ext in ['png','svg']:
    path=OUT/('RESOLUTION_1200_VS_2000.'+ext);fig.savefig(path,dpi=450,facecolor='white');outputs.append(path.name)
plt.close(fig)
manifest=dict(real_data=True,dpi=450,smoothing='none',outputs=outputs,
    sources={p.relative_to(HERE).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,R/'checks.csv']},
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(OUT/'DATA_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'data-manifest.md').write_text('# 数据清单\n\n真实数据；原始samples.csv，无平滑。\n\n| Figure | Data file | Real/mock | Source | Script | Outputs |\n|---|---|---|---|---|---|\n| 分辨率1200/2000对照 | reports/samples.csv | Real | 同属性包双分辨率G渲染双轮 | analyze.py | RESOLUTION_1200_VS_2000.png / .svg |\n\n中位数、差值与倍数可查reports/RESOLUTION_PAIRED_RESULTS.csv；输入和脚本SHA见DATA_MANIFEST.json。CPU推理和上传均不在计时内。\n',encoding='utf-8')
for p,expected in manifest['sources'].items():assert hashlib.sha256((HERE/p).read_bytes()).hexdigest()==expected
print(json.dumps(summary,ensure_ascii=False,indent=2))
print(t[t.condition.isin(['full','local0.1'])].to_string(index=False))
