"""跨场景逐Ratio渲染质量与蒸馏差距。"""
from pathlib import Path
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE/'results' if (HERE/'results').exists() else HERE.parents[2]
REPORTS = ROOT/'reports'
COLORS = dict(teacher='#111111',image_calibration='#9c755f',
    geometry_calibration='#4e79a7',continuous_calibration='#e15759')
LABELS = dict(teacher='Teacher',image_calibration='原图像校准',geometry_calibration='几何校准',continuous_calibration='连续Ratio校准')


def main():
    plt.rcParams.update({'font.sans-serif':['Microsoft YaHei','DejaVu Sans'],
        'axes.unicode_minus':False,'font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    scenes = json.loads((REPORTS/'SCENES.json').read_text())
    completed = [s for s in scenes if (REPORTS/s['slug']/'QUALITY_SUMMARY.json').exists()]
    if not completed:
        return
    figure,axes = plt.subplots(len(completed),2,figsize=(13,3.1*len(completed)),squeeze=False,layout='constrained')
    gap_figure,gap_axes = plt.subplots(len(completed),2,figsize=(13,3.1*len(completed)),squeeze=False,layout='constrained')
    for row,scene in enumerate(completed):
        results = json.loads((REPORTS/scene['slug']/'QUALITY_SUMMARY.json').read_text())
        for column,local in enumerate([False,True]):
            ax,gap_ax = axes[row,column],gap_axes[row,column]
            for name,color in COLORS.items():
                group = sorted([r for r in results['summaries'] if r['student']==name and r['local']==local],key=lambda r:r['ratio'])
                ratios = [r['ratio'] for r in group]
                ax.plot(ratios,[r['psnr'] for r in group],'o-',label=LABELS[name],color=color,markersize=3.5)
                if name!='teacher':
                    gap_ax.plot(ratios,[r['mean_loss'] for r in group],'o-',label=LABELS[name],color=color,markersize=3.5)
            title = f"{scene['dataset_kind']} {scene['scene']} {scene['mode']} · N={scene['gaussians']:,} · {'局部' if local else '全局'}"
            for axis in [ax,gap_ax]:
                axis.set_title(title)
                axis.set_xscale('log')
                axis.set_xticks(ratios,[f'{r*100:g}%' for r in ratios],rotation=35)
                axis.grid(axis='y',alpha=.25)
                axis.set_xlabel('Ratio')
            ax.set_ylabel('PSNR / dB')
            gap_ax.set_ylabel('Teacher PSNR − Student PSNR / dB')
            gap_ax.axhline(0,color='#333333',linewidth=.8)
            gap_ax.axhline(.01,color='#59a14f',linestyle='--',linewidth=.8,label='平均损失门限')
            if row==0 and column==1:
                ax.legend(fontsize=8)
                gap_ax.legend(fontsize=8)
    figure.suptitle('固定蒸馏流程 · 每场景32个学生留出视角 · 同选点身份')
    gap_figure.suptitle('跨场景蒸馏质量差距 · 平均值')
    figure.savefig(REPORTS/'CROSS_SCENE_QUALITY.png',dpi=180)
    gap_figure.savefig(REPORTS/'CROSS_SCENE_MACRO.png',dpi=180)
    plt.close(figure)
    plt.close(gap_figure)


if __name__ == '__main__':
    main()
