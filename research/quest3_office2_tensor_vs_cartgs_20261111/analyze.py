"""Quest3像素负载模拟：双次独立准备/渲染，中文报告与图。"""
from pathlib import Path
import json,hashlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import fontManager,FontProperties
HERE=Path(__file__).resolve().parent;R=HERE/'reports';OUT=HERE/'figures';OUT.mkdir(exist_ok=True)
s=pd.read_csv(R/'samples.csv');checks=pd.read_csv(R/'checks.csv')
v=json.loads((R/'VALIDATION.json').read_text());job=json.loads((HERE/'logs/formal_job.json').read_text())
ident=json.loads((R/'MODEL_IDENTITIES.json').read_text());protocol=json.loads((R/'PROTOCOL.json').read_text())
assert v['passed'] and job['measurement_valid']
assert len(s)==18144 and len(checks)==252 and checks.original_rasterizer_bitwise.all() and checks.finite.all()
assert s.groupby(['condition','scope']).size().eq(432).all() and checks['shape'].eq('3x2208x2064').all()
assert checks.groupby(['condition','frame']).ids_sha256.nunique().eq(1).all()
assert checks.groupby(['condition','frame']).K.nunique().eq(1).all()
assert np.isfinite(s.ms).all() and (s.ms>0).all()
agg=s.groupby(['condition','scope']).agg(ms=('ms','median'),p95=('ms',lambda x:x.quantile(.95)),
    maximum=('ms','max'),N=('N','median'),M=('M','median'),K=('K_per_eye','median'),K_min=('K_per_eye','min'),
    K_max=('K_per_eye','max'),samples=('ms','size')).reset_index()
agg.to_csv(R/'TIMING_SUMMARY.csv',index=False)
s.groupby(['condition','scope','round']).ms.median().reset_index().to_csv(R/'TIMING_ROUNDS.csv',index=False)
byframe=s.groupby(['condition','frame','scope']).ms.median().reset_index();byframe.to_csv(R/'PER_VIEW_TIMING.csv',index=False)


def val(condition,scope):
    row=agg[(agg.condition==condition)&(agg.scope==scope)];assert len(row)==1
    return float(row.iloc[0].ms)


rows=[]
for condition in protocol['conditions']:
    g=agg[agg.condition==condition];row=dict(condition=condition,N=int(g.iloc[0].N),M=g.iloc[0].M,
        K_per_eye_median=g.iloc[0].K,K_per_eye_min=int(g.iloc[0].K_min),K_per_eye_max=int(g.iloc[0].K_max))
    for scope in ['prepare_pair_wall','raster_eye0_event','raster_eye1_event','raster_pair_event','raster_pair_wall','pipeline_pair_wall']:
        row[scope+'_ms']=val(condition,scope)
    row['pure_render_speedup_vs_cartgs']=val('cartgs_full','raster_pair_event')/row['raster_pair_event_ms']
    row['pipeline_speedup_vs_cartgs']=val('cartgs_full','pipeline_pair_wall')/row['pipeline_pair_wall_ms']
    rows.append(row)
t=pd.DataFrame(rows);t.to_csv(R/'QUEST3_OFFICE2_SEPARATED_RESULTS.csv',index=False)
pair=s.pivot(index=['frame','round','repeat','scope'],columns='condition',values='ms')
for condition in protocol['conditions'][1:]:pair[condition+'_speedup']=pair.cartgs_full/pair[condition]
pair.to_csv(R/'PAIRED_SPEEDUP.csv')
lo=t[t.condition.str.startswith('ours_')&~t.condition.eq('ours_full')]
summary=dict(passed=True,samples=len(s),checks=len(checks),display_frames=432,views=18,rounds=3,repeat_per_round=8,
    width=2064,height=2208,independent_renders_per_frame=2,device='RTX3090',
    cartgs_pure_pair_ms=val('cartgs_full','raster_pair_event'),cartgs_prepare_pair_ms=val('cartgs_full','prepare_pair_wall'),cartgs_pipeline_pair_ms=val('cartgs_full','pipeline_pair_wall'),
    ours_pure_pair_min_ms=lo.raster_pair_event_ms.min(),ours_pure_pair_max_ms=lo.raster_pair_event_ms.max(),
    ours_prepare_pair_min_ms=lo.prepare_pair_wall_ms.min(),ours_prepare_pair_max_ms=lo.prepare_pair_wall_ms.max(),
    ours_pipeline_pair_min_ms=lo.pipeline_pair_wall_ms.min(),ours_pipeline_pair_max_ms=lo.pipeline_pair_wall_ms.max(),
    foreign_gpu=job['foreign_active_samples'],foreign_cpu=job['foreign_cpu_active_samples'],protected_files=v['protected_files'])
(R/'ANALYSIS_VALIDATION.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
def label(condition):
    if condition=='cartgs_full':return 'CaRtGS原版Full'
    if condition=='ours_full':return '我们D Full'
    return f"我们D {float(condition[5:]):.0%}"
lines=['# Quest 3像素负载模拟：office2 CaRtGS与Tensor＋新准备D','','## 结论','',
    f"RTX3090上，每个显示帧独立渲染两张2064×2208图像（共{2*2064*2208:,}像素），原版CaRtGS双眼纯渲染中位{summary['cartgs_pure_pair_ms']:.3f}ms、双眼准备{summary['cartgs_prepare_pair_ms']:.3f}ms、双眼完整{summary['cartgs_pipeline_pair_ms']:.3f}ms。我们D LoD各Ratio的双眼纯渲染{summary['ours_pure_pair_min_ms']:.3f}–{summary['ours_pure_pair_max_ms']:.3f}ms，双眼准备{summary['ours_prepare_pair_min_ms']:.3f}–{summary['ours_prepare_pair_max_ms']:.3f}ms，双眼完整{summary['ours_pipeline_pair_min_ms']:.3f}–{summary['ours_pipeline_pair_max_ms']:.3f}ms。",'',
    '这里是Quest 3每眼像素尺寸及两次调用的负载模拟，不是Quest 3真机性能。每显示帧两次使用同一当前位姿，分别完整准备和渲染，不设眼间距，不使用OpenXR multiview/单次双眼渲染、注视点降采样或时间扭曲。保留Replica原FoV，不将其称为Quest真实光学视野或立体视觉质量验证。','',
    '## 分离计时结果','',
    '单位ms；左/右是同位姿两次独立调用编号，不代表真实左右眼视差。所有列分别直接实测，不通过单眼×2外推。准备及完整墙钟和纯渲染Event是独立采样，中位数不能机械相加。','',
    '| 方法 | 地图N | 每眼实际K中位数 | 双眼准备 | 第1眼纯渲染 | 第2眼纯渲染 | 双眼纯渲染 | 双眼完整 | 纯渲染倍速 |',
    '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
for _,r in t.iterrows():
    lines.append(f"| {label(r.condition)} | {r.N:g} | {r.K_per_eye_median:g} | {r.prepare_pair_wall_ms:.4f} | {r.raster_eye0_event_ms:.4f} | {r.raster_eye1_event_ms:.4f} | {r.raster_pair_event_ms:.4f} | {r.pipeline_pair_wall_ms:.4f} | {r.pure_render_speedup_vs_cartgs:.3f}× |")
lines+=['','## 对照身份与口径','',
    f"- CaRtGS为已有归档Vanilla算法（项目关闭Selector/Compensator等扩展），Replica office2 RGB-D三遍中Full全轨迹PSNR最高的repeat3，原生历史PSNR34.0468dB，地图84,203点。该记录用于确定既有对照，不代表本次高分辨率质量实测。checkpoint为`{ident['vanilla_checkpoint']}`。不将其声称为新克隆、未修改上游仓库训练结果。",
    f"- 我们仅用最终D：同一3534参数Student＋GPU Selector、FP16 Tensor网络/FP32累加、FP32静态预投影、CUDA视锥、GPU动态N/K、C++相机、新G Renderer。Master地图43,197点，checkpoint为`{ident['ours']['checkpoint']}`，学生为`{ident['student']}`。不加入A/C或其他网络对照。",
    f"- 原版渲染器是此前原始inference路径，扩展SHA256为`{ident['original_rasterizer_sha256']}`；不是新G或balanced渲染器。两种地图来自同场景的不同训练结果，因此收益同时包含地图规模、选择补偿、渲染器和准备优化，不能全部归于Tensor Core或减少点数。",
    '- 同一组18个RGB时间戳（12连续＋6跳转）；两张地图分别使用自己SLAM保存的对应位姿，避免错误跨地图坐标系套用相机。相同2064×2208、同背景、同SH阶数/视野语义。两图像仍可能因地图、位姿及补偿不同而有不同footprint和渲染复杂度，不是固定同属性的算子对照。',
    '- 双眼准备墙钟：分别构造两次当前相机和属性；原版每眼计算物理属性激活/打包，我们每眼独立重新运行候选、选择、两网络、解码/打包。Full Master本身允许GPU常驻；我们Full包加载时构造一次，LoD不复用前一眼的Mask/网络结果。',
    '- 双眼纯渲染：先在计时外完成两眼各自准备，然后CUDA Event在两个Rasterizer调用前/间/后记录，返回每眼与合计。完全不含Tensor网络、相机准备或属性激活。Event区间可能包含光栅器内部发射与依赖造成的GPU时间线间隙，不是Nsight Kernel时间求和；另存raster_pair_wall。',
    '- 双眼完整墙钟：眼0准备→眼0渲染→眼1准备→眼1渲染→同步，使用同一Stream顺序执行，两个输出同时持有。没有跨眼并发，没有单次multiview优化，也没有用单眼中位数翻倍代替双眼实测。',
    '- 原版CaRtGS的准备时间也包含原MiniCam的当前相机求逆/矩阵与物理属性激活；我们的D相机采用已验收新路径。原版两眼相机未被替换为D来掩盖完整路径差异。纯渲染比较则两侧均在相机/属性准备后开始。',
    '- 不含模型读取、Graph首次构建、图片读取/编码/保存、HTTP、Viewer显示、OpenXR合成、镜片畸变、Tracking/Mapping；因此也不将倒数称为Quest实际显示FPS。',
    '- FoV沿用Replica而非Quest真实镜片，不指定IPD。若需要真实左右眼不同位姿、非对称投影及头显GPU，需要独立协议；本轮只按用户指定的每帧两次像素负载测试。','',
    '## 验证与限制','',
    f"- 三轮×8重复×18视角×7配置，每个配置每种计时432样本，总{len(s)}条计时。每眼实际K、候选M、原始p95/最大值、逐视角及逐轮保留，严格预算不改变。",
    f"- {len(checks)}组眼/视角/配置检查输出尺寸3×2208×2064、finite，D各自属性包的新G与原Rasterizer RGB/radii逐位一致。同位姿独立两眼输出和所选ID相同，证明没有漏掉第二次渲染。原版与我们地图不同，未声称两方法图像逐位或PSNR等价。",
    f"- {job['observations']}次低频监控，外部活跃GPU样本{job['foreign_active_samples']}，外部单进程>10% CPU样本{job['foreign_cpu_active_samples']}；不能排除所有瞬时调度。所有有效计时保留，无速度阈值筛样本。",
    f"- {v['protected_files']}个受保护源码/配置/模型/二进制前后SHA无变化；checkpoint只读，无训练、无JIT/编译、无生产配置修改、无共享conda修改、无commit/push，渲染图片保存0。",'',
    '## 交付','',
    '服务器目录：`/home/prmi/zwj/experiments/quest3_office2_tensor_vs_cartgs_20261111`；分项表reports/QUEST3_OFFICE2_SEPARATED_RESULTS.csv，逐轮/逐视角和全部原始samples.csv，模型与渲染器身份见MODEL_IDENTITIES.json。']
(HERE/'QUEST3_OFFICE2_TENSOR_VS_CARTGS_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

font=Path('C:/Windows/Fonts/msyh.ttc');assert font.is_file();fontManager.addfont(str(font));plt.rcParams['font.family']=FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,
    'legend.frameon':False,'axes.unicode_minus':False,'svg.fonttype':'none','savefig.dpi':450})
fig,axes=plt.subplots(1,3,figsize=(13,4.9))
styles=[('prepare_pair_wall_ms','双眼准备墙钟（D含两网络）','#4E79A7'),('raster_pair_event_ms','双眼纯渲染（CUDA Event）','#59A14F'),
    ('pipeline_pair_wall_ms','双眼完整流程墙钟','#A23B72')]
base=t[t.condition=='cartgs_full'].iloc[0];ourfull=t[t.condition=='ours_full'].iloc[0]
lo=lo.copy();lo['ratio']=lo.condition.str[5:].astype(float);lo=lo.sort_values('ratio')
for ax,(field,title,color) in zip(axes,styles):
    ax.axhline(base[field],color='#6B7280',ls='--',lw=1.8,label='CaRtGS原版Full（N=84,203）')
    ax.axhline(ourfull[field],color=color,ls=':',lw=1.8,label='我们D Full（N=43,197）')
    ax.plot(lo.ratio,lo[field],color=color,marker='o',lw=2,ms=4,label='我们D LoD')
    ax.set_title(title,loc='left',fontsize=11,fontweight='bold',pad=12)
    ax.set_xlabel('Ratio（线性轴）');ax.set_ylabel('双眼合计耗时（ms）');ax.set_ylim(0,max(base[field],lo[field].max())*1.22)
    ax.set_xlim(0,.735);ax.set_xticks([.01,.05,.1,.3,.7],['1%','5%','10%','30%','70%'],rotation=40,ha='right')
    ax.text(.04,.96,f'原版：{base[field]:.3f} ms',transform=ax.transAxes,va='top',fontsize=9)
    ax.legend(loc='upper left',bbox_to_anchor=(0,-.26),fontsize=8)
    if field!='prepare_pair_wall_ms':
        r=lo[np.isclose(lo.ratio,.1)].iloc[0]
        ax.annotate(f'10%：{r[field]:.3f} ms',xy=(.1,r[field]),xytext=(12,15),textcoords='offset points',fontsize=9,color=color)
fig.suptitle('office2 RGB-D：2064×2208 × 2次渲染的Quest 3像素负载模拟',fontsize=15,fontweight='bold',y=.98)
fig.text(.5,.025,'RTX3090实测；同一当前视角独立渲染两遍，无眼间距/多视图/畸变合成；不是Quest真机速度。\n准备、纯渲染、完整流程独立计时；三轮 × 18视角 × 8重复；不同方法各用自身地图和对应轨迹。',ha='center',fontsize=9,color='#555555')
fig.subplots_adjust(left=.065,right=.98,bottom=.35,top=.80,wspace=.29)
outputs=[]
for ext in ['png','svg']:
    path=OUT/('QUEST3_OFFICE2_TIMING.'+ext);fig.savefig(path,dpi=450,facecolor='white');outputs.append(path.name)
plt.close(fig)
sources={f.relative_to(HERE).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in [R/'samples.csv',R/'checks.csv',R/'MODEL_IDENTITIES.json']}
(OUT/'DATA_MANIFEST.json').write_text(json.dumps(dict(real_data=True,dpi=450,smoothing='none',sources=sources,outputs=outputs,
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'data-manifest.md').write_text('# 数据清单\n\n| Figure | Data file | Real/mock | Source | Script | Outputs |\n|---|---|---|---|---|---|\n| office2双眼负载 | reports/samples.csv | Real | 本轮2064×2208每帧两次，三轮 | analyze.py | QUEST3_OFFICE2_TIMING.png / .svg |\n\n原始中位数，未平滑。相机/网络准备不计入纯渲染；不同地图、原版Rasterizer与新G。输入及脚本SHA见JSON。\n',encoding='utf-8')
for name,expected in sources.items():assert hashlib.sha256((HERE/name).read_bytes()).hexdigest()==expected
print(json.dumps(summary,indent=2));print(t.to_string(index=False))
