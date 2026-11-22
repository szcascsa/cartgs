"""画同Renderer的纯渲染及完整推理曲线，不平滑或合成数据。"""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent


def main():
    rows=list(csv.DictReader((HERE/'reports/TEACHER_LOD_SPEED_SUMMARY.csv').open(encoding='utf-8-sig')))
    scenes=sorted({r['scene'] for r in rows})
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],
        'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    figure,axes=plt.subplots(4,2,figsize=(13,12),layout='constrained')
    ratios=[.01,.05,.1,.3,.7]
    for i,scene in enumerate(scenes):
        for col,scope in enumerate(['raster','pipeline']):
            ax=axes[i,col]
            group=[r for r in rows if r['scene']==scene and r['scope']==scope]
            full=next(r for r in group if r['condition']=='full_direct')
            clone=next(r for r in group if r['condition']=='full_copy')
            lod=[next(r for r in group if r['condition']=='local'+str(ratio)) for ratio in ratios]
            times=[float(r['median_ms']) for r in lod]
            ax.plot([r*100 for r in ratios],times,'o-',color='#2468a0',label='未蒸馏网络LoD＋新Renderer')
            ax.axhline(float(full['median_ms']),color='#202020',linewidth=1.2,label='直接Master Full＋新Renderer')
            if scope=='pipeline':
                ax.axhline(float(clone['median_ms']),color='#bd7536',linestyle='--',label='每帧复制Full（旧报告口径）')
            for ratio,value,row in zip(ratios,times,lod):
                ax.annotate(f"K={float(row['K_median']):g}",(ratio*100,value),
                    xytext=(0,7),textcoords='offset points',ha='center',fontsize=8)
            ax.set_title(f"{scene} · N={int(full['N']):,} · {'纯渲染' if scope=='raster' else '相机＋网络准备＋渲染'}")
            ax.set_xlabel('局部Ratio / %');ax.set_ylabel('耗时中位数 / ms')
            ax.set_xticks([1,5,10,30,70]);ax.grid(alpha=.2)
            ax.margins(y=.18)
            if i==0:ax.legend(fontsize=8,loc='upper left')
    figure.suptitle('相同新渲染器：减少点数的收益与网络开销｜两轮配对、Teacher64/64')
    out=HERE/'figures';out.mkdir(exist_ok=True)
    figure.savefig(out/'TEACHER_LOD_VS_FULL.png',dpi=180)
    figure.savefig(out/'TEACHER_LOD_VS_FULL.svg')
    plt.close(figure)


if __name__=='__main__':main()
