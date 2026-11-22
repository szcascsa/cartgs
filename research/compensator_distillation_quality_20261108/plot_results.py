"""把逐 Ratio 的配对蒸馏误差绘成独立科研图。"""
from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
RESULTS = HERE/'results' if (HERE/'results').is_dir() else HERE.parents[2]
REPORTS = RESULTS/'reports'
plt.rcParams.update({'font.sans-serif':['Microsoft YaHei'], 'axes.unicode_minus':False,
    'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
COLORS = dict(previous='#8c6d5c', residual001='#4e79a7', anchor_refine='#7f7f7f', continuous_refine='#e15759')
LABELS = dict(previous='上一版学生', residual001='弱化残差监督', anchor_refine='固定 Ratio 校准', continuous_refine='连续 Ratio 校准')


def main():
    report = json.loads((REPORTS/'FINAL_QUALITY_SUMMARY.json').read_text())
    groups = [r for r in report['summaries'] if r['split'] == 'fresh']
    ratios = sorted(set(r['ratio'] for r in groups))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.1), sharey=True, layout='constrained')
    for ax, local in zip(axes, [False, True]):
        for name in COLORS:
            rows = sorted([r for r in groups if r['student'] == name and r['local'] == local], key=lambda r:r['ratio'])
            ax.plot(ratios, [r['mean_loss'] for r in rows], 'o-', color=COLORS[name], label=LABELS[name], markersize=4)
        ax.axhline(0, color='#333333', linewidth=1, label='教师')
        ax.axhline(.01, color='#59a14f', linestyle='--', linewidth=1, label='平均损失门限')
        ax.set_xscale('log')
        ax.set_xticks(ratios, [f'{r*100:g}%' for r in ratios], rotation=35)
        ax.set_xlabel('Ratio')
        ax.set_title('局部视角模式' if local else '全局模式')
        ax.grid(axis='y', alpha=.25)
    axes[0].set_ylabel('教师 PSNR − 学生 PSNR / dB')
    axes[1].legend(fontsize=9)
    fig.suptitle('fr1_xyz RGB-D · 32 个新留出视角的平均 PSNR 差距')
    fig.savefig(REPORTS/'COMPENSATOR_DISTILLATION_QUALITY.png', dpi=200)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8, 3.8), layout='constrained')
    for name in ['anchor_refine', 'continuous_refine']:
        path = RESULTS/'models'/name/'training.csv'
        if not path.exists():
            continue
        rows = [r for r in csv.DictReader(path.open()) if r['validation']]
        start = report['identities'][name]['initial_validation']
        ax.plot([0]+[int(r['step']) for r in rows], [start]+[float(r['validation']) for r in rows],
            '.-', color=COLORS[name], label=LABELS[name])
    ax.set_xlabel('本轮学生 Adam 更新次数')
    ax.set_ylabel('验证教师图像 MSE')
    ax.set_title('匹配蒸馏的收敛过程')
    ax.grid(alpha=.25)
    ax.legend()
    fig.savefig(REPORTS/'TRAINING_IMAGE_MSE.png', dpi=200)
    plt.close(fig)


if __name__ == '__main__':
    main()
