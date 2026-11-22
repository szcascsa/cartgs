"""Replica全部场景，单目与RGB-D分别导出PSNR与误差曲线。"""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE/'results' if (HERE/'results').exists() else HERE.parents[2]
REPORTS = ROOT/'reports'
COLORS = dict(teacher='#111111',image_calibration='#9c755f',geometry_calibration='#4e79a7',continuous_calibration='#e15759')
LABELS = dict(teacher='Teacher',image_calibration='原图像校准',geometry_calibration='几何校准',continuous_calibration='连续Ratio校准')


def main():
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],'axes.unicode_minus':False,
        'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    scenes = json.loads((REPORTS/'SCENES.json').read_text())
    for mode in ['mono','rgbd']:
        completed = [s for s in scenes if s['mode']==mode and (REPORTS/s['slug']/'QUALITY_SUMMARY.json').exists()]
        if not completed:
            continue
        for gap in [False,True]:
            figure,axes = plt.subplots(len(completed),2,figsize=(13,3.1*len(completed)),squeeze=False,layout='constrained')
            for row,scene in enumerate(completed):
                results = json.loads((REPORTS/scene['slug']/'QUALITY_SUMMARY.json').read_text())
                for column,local in enumerate([False,True]):
                    ax = axes[row,column]
                    for name,color in COLORS.items():
                        if gap and name=='teacher':
                            continue
                        group = sorted([r for r in results['summaries'] if r['student']==name and r['local']==local],key=lambda r:r['ratio'])
                        ratios = [r['ratio'] for r in group]
                        ax.plot(ratios,[r['mean_loss'] if gap else r['psnr'] for r in group],
                            'o-',label=LABELS[name],color=color,markersize=3.5)
                    ax.set_title(f"{scene['scene']} · {'单目' if mode=='mono' else 'RGB-D'} · N={scene['gaussians']:,} · {'局部' if local else '全局'}")
                    ax.set_xscale('log')
                    ax.set_xticks(ratios,[f'{r*100:g}%' for r in ratios],rotation=35)
                    ax.grid(axis='y',alpha=.25)
                    ax.set_xlabel('Ratio')
                    ax.set_ylabel('Teacher PSNR − Student PSNR / dB' if gap else 'PSNR / dB')
                    if gap:
                        ax.axhline(0,color='#333333',linewidth=.8)
                        ax.axhline(.01,color='#59a14f',linestyle='--',linewidth=.8,label='平均损失门限')
                    if row==0 and column==1:
                        ax.legend(fontsize=8)
            figure.suptitle(f"Replica · {'单目' if mode=='mono' else 'RGB-D'} · 每场景32个学生留出视角 · {'蒸馏质量差距' if gap else '渲染质量'}")
            figure.savefig(REPORTS/f'REPLICA_{"GAP" if gap else "PSNR"}_{mode.upper()}.png',dpi=180)
            plt.close(figure)


if __name__=='__main__':
    main()
