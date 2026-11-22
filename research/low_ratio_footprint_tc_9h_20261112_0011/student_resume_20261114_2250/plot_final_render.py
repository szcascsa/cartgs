"""真实逐样本测量：实际K曲线、归一化缩放及速度—质量；不修改或平滑原始样本。"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

HERE=Path(__file__).resolve().parent
OUT=HERE/'figures'
COLORS={'F2':'#4E79A7','tile_safe_bitexp_stop':'#F28E2B','encoded_signed_square':'#59A14F'}
LABELS={'F2':'冻结F2','tile_safe_bitexp_stop':'历史mode38','encoded_signed_square':'本轮182'}


def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=450,bbox_inches='tight')
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight');plt.close(fig)


def main():
    OUT.mkdir(exist_ok=True)
    for path in [Path('C:/Windows/Fonts/msyh.ttc'),Path('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')]:
        if path.exists():
            font_manager.fontManager.addfont(str(path));plt.rcParams['font.family']=font_manager.FontProperties(fname=str(path)).get_name();break
    plt.rcParams.update({'font.size':8,'axes.spines.top':False,'axes.spines.right':False,'axes.unicode_minus':False,
        'legend.frameon':False,'axes.grid':True,'grid.alpha':.22,'svg.fonttype':'none'})
    data=pd.read_csv(HERE/'reports/RENDER_PER_FRAME.csv.gz')
    ratios=data[(data.condition.str.startswith('local'))|(data.condition=='full')]
    entries=[]
    for scene in sorted(data.scene.unique()):
        fig,axes=plt.subplots(1,2,figsize=(7.2,2.8))
        for ax,(kind,title) in zip(axes,[('nested','A：固定属性嵌套删点'),('student_lod','B：真实Student LoD')]):
            frame=ratios[(ratios.scene==scene)&(ratios.workload==kind)]
            if frame.empty:ax.text(.5,.5,'缺少匹配Student\n仅报告固定属性实验',ha='center',va='center',transform=ax.transAxes)
            for variant,color in COLORS.items():
                points=frame[frame.variant==variant].groupby('condition').agg(K=('K','median'),T=('p50_ms','median')).sort_values('K')
                if len(points):ax.plot(points.K,points['T']*1000,'.-',lw=1.2,color=color,label=LABELS[variant],markersize=4)
            ax.set_title(title);ax.set_xlabel('实际输入K');ax.set_ylabel('纯Rasterizer墙钟中位延迟（μs）');ax.set_xscale('symlog',linthresh=100)
        axes[0].legend(fontsize=7);fig.suptitle(scene);fig.tight_layout()
        name='scene_'+scene.replace('/','__');save(fig,name);entries.append(name)
    fig,axes=plt.subplots(1,2,figsize=(7.2,3.0))
    for ax,(kind,title) in zip(axes,[('nested','固定属性'),('student_lod','真实Student LoD')]):
        frame=ratios[ratios.workload==kind]
        for variant,color in COLORS.items():
            points=frame[frame.variant==variant].groupby('condition').agg(K=('normalized_K','median'),T=('normalized_T','median'),
                low=('normalized_T',lambda x:x.quantile(.1)),high=('normalized_T',lambda x:x.quantile(.9))).sort_values('K')
            ax.plot(points.K,points['T'],'.-',label=LABELS[variant],color=color,lw=1.2)
            ax.fill_between(points.K,points.low,points.high,color=color,alpha=.07)
        ax.plot([0,1],[0,1],'--',color='#777777',lw=.9,label='理想T比=K比');ax.set_title(title)
        ax.set_xlabel('K / K_Full');ax.set_ylabel('T / T_Full');ax.set_xlim(0,1.02);ax.set_ylim(bottom=0)
    axes[0].legend(fontsize=7);fig.tight_layout();save(fig,'normalized_scaling');entries.append('normalized_scaling')
    stats=pd.read_csv(HERE/'reports/RENDER_SCENE_RATIO.csv')
    main=stats[(stats.workload=='student_lod')&(stats.variant=='encoded_signed_square')&stats.condition.isin([f'local{x:g}' for x in [.1,.2,.3,.4,.5]])]
    rows=main.groupby(['scene','condition']).agg(f2=('F2_speedup','median'),old=('old38_speedup','median')).reset_index()
    pivot=rows.pivot(index='scene',columns='condition',values='old');fig,ax=plt.subplots(figsize=(7.2,5.0))
    im=ax.imshow(pivot,cmap='YlGnBu',aspect='auto',vmin=1);ax.set_xticks(range(len(pivot.columns)),[x[5:] for x in pivot.columns]);ax.set_yticks(range(len(pivot)),pivot.index,fontsize=6)
    ax.grid(False)
    for y in range(len(pivot)):
        for x in range(len(pivot.columns)):ax.text(x,y,f'{pivot.iloc[y,x]:.2f}×',ha='center',va='center',fontsize=6,color='white' if pivot.iloc[y,x]>1.17 else '#202020')
    ax.set_xlabel('Ratio');ax.set_title('本轮182相对历史mode38：逐场景加速比（两轮中位）');fig.colorbar(im,ax=ax,label='加速比');fig.tight_layout();save(fig,'scene_ratio_speedup');entries.append('scene_ratio_speedup')
    overview=pd.read_csv(HERE/'reports/RENDER_FINAL_OVERVIEW.csv')
    points=overview[overview.workload=='student_lod'];fig,ax=plt.subplots(figsize=(6.0,3.3))
    for _,row in points.iterrows():
        color=COLORS.get(row.variant,'#76B7B2');ax.scatter(row.mean_psnr_loss,row.F2_speedup,color=color,s=35)
        ax.annotate(LABELS.get(row.variant,row.variant),(row.mean_psnr_loss,row.F2_speedup),xytext=(5,4),textcoords='offset points',fontsize=6)
    ax.axvline(.2,color='#E15759',ls='--',lw=.8,label='平均PSNR损失门限0.20 dB');ax.axhline(1,color='#777777',lw=.7)
    ax.set_xlabel('等权平均PSNR损失（dB，相对F2）');ax.set_ylabel('主要Ratio相对F2加速比');ax.set_title('速度—质量对照（同一FP32 Student输入）');ax.legend(fontsize=7);fig.tight_layout();save(fig,'speed_quality_pareto');entries.append('speed_quality_pareto')
    files=['RENDER_PER_FRAME.csv.gz','RENDER_SCENE_RATIO.csv','RENDER_FINAL_OVERVIEW.csv','RENDER_SCALING_FIT.csv']
    manifest={name:hashlib.sha256((HERE/'reports'/name).read_bytes()).hexdigest() for name in files}
    (OUT/'DATA_SHA256.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    lines=['# 性能图数据清单','','所有图均来自实际冻结输入测量，无模拟数据；原始样本保留，未做曲线平滑。','',
        '|图|数据|类型|脚本|输出|','|---|---|---|---|---|']
    for name in entries:lines.append(f'|{name}|reports/RENDER_PER_FRAME.csv.gz、RENDER_SCENE_RATIO.csv、RENDER_FINAL_OVERVIEW.csv|真实|plot_final_render.py|{name}.png / .svg|')
    lines+=['','曲线在每个条件内取中位数；归一化阴影为各帧10%–90%范围。嵌套删点使用固定随机排列前缀；真实LoD保留Student补偿，二者不混同。']
    (OUT/'data-manifest.md').write_text('\n'.join(lines),encoding='utf-8');print('RENDER_FIGURES_COMPLETE',len(entries))


if __name__=='__main__':main()
