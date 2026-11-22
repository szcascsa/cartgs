"""新G/R下Teacher/Student/Full的真实延迟，不平滑。"""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent


def main():
    rows=list(csv.DictReader((HERE/'reports/STUDENT_VS_TEACHER_SPEED.csv').open(encoding='utf-8-sig')))
    scenes=sorted({r['scene'] for r in rows})
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],
        'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    figure,axes=plt.subplots(2,2,figsize=(13,8.5),layout='constrained')
    for scene,ax in zip(scenes,axes.flat):
        group=sorted([r for r in rows if r['scene']==scene and r['scope']=='pipeline'],key=lambda x:float(x['ratio']))
        xs=[100*float(r['ratio']) for r in group]
        for name,label,color in [('teacher','Teacher64/64','#3878ab'),('student','Student32/32','#d65a43')]:
            ax.plot(xs,[float(r[name+'_ms']) for r in group],'o-',color=color,label=label)
        ax.axhline(float(group[0]['full_ms']),color='#222222',label='直接Master Full',linewidth=1.2)
        for r in group:
            ax.annotate(f"K={float(r['K_median']):g}",(float(r['ratio'])*100,float(r['student_ms'])),
                xytext=(0,-17),textcoords='offset points',ha='center',fontsize=8)
        ax.set_title(f"{scene} · N={int(group[0]['N']):,}")
        ax.set_xlabel('局部Ratio / %（对数坐标）');ax.set_ylabel('相机＋准备＋渲染 / ms')
        ax.set_xscale('log');ax.set_xticks(xs,[f'{v:g}' for v in xs]);ax.grid(alpha=.22)
        ax.margins(y=.22);ax.legend(fontsize=8)
    figure.suptitle('同一新G Renderer＋R相机：蒸馏的额外收益与Full对照｜两轮配对')
    out=HERE/'figures';out.mkdir(exist_ok=True)
    figure.savefig(out/'STUDENT_VS_TEACHER_NEW_RENDERER.png',dpi=180)
    figure.savefig(out/'STUDENT_VS_TEACHER_NEW_RENDERER.svg')
    plt.close(figure)


if __name__=='__main__':main()
