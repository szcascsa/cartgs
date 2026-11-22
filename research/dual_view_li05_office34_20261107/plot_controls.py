"""由已验收CSV绘制LI权重对照曲线；不生成渲染帧。"""
from pathlib import Path
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

R = Path(__file__).resolve().parent


def main():
    with (R/'results/LI05_COMPARISON.csv').open(encoding='utf-8') as stream:
        rows = list(csv.DictReader(stream))
    font = FontProperties(fname='C:/Windows/Fonts/msyh.ttc')
    output = R/'results/figures'
    output.mkdir(exist_ok=True)
    for scene in ['office3', 'office4']:
        figure, axes = plt.subplots(2, 2, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]}, sharex='col')
        for column, (scope, title) in enumerate([('global', '全局LoD'), ('view_ratio', '局部LoD')]):
            data = sorted([r for r in rows if r['scene'] == scene and r['scope'] == scope], key=lambda r: float(r['ratio']))
            if not data:
                continue
            x = [100*float(r['ratio']) for r in data]
            for key, color, label in [('li01', '#2474B5', 'LI=0.1'), ('li05', '#D46A25', 'LI=0.5')]:
                y = [float(r['psnr_'+key]) for r in data]
                axes[0, column].plot(x, y, color=color, linewidth=2, marker='o', markersize=4, label=label)
            delta = [float(r['psnr_delta']) for r in data]
            axes[1, column].axhline(0, color='#555555', linewidth=1, linestyle='--')
            axes[1, column].plot(x, delta, color='#7C4C9B', marker='o', markersize=4, linewidth=1.8)
            axes[0, column].set_title(title, fontproperties=font, fontsize=12)
            axes[0, column].set_ylabel('PSNR (dB)')
            axes[1, column].set_ylabel('ΔPSNR (dB)\nLI=0.5 − LI=0.1')
            axes[1, column].set_xlabel('Ratio (%)')
            axes[1, column].set_xticks([1, 5, 10, 20, 30, 40, 50, 60, 70])
            axes[0, column].legend(frameon=False)
            for ax in axes[:, column]:
                ax.grid(alpha=.22)
                ax.spines[['right', 'top']].set_visible(False)
        full = next((r for r in rows if r['scene'] == scene and r['scope'] == 'full'), None)
        subtitle = '' if full is None else f"；Full：{float(full['psnr_li01']):.3f} → {float(full['psnr_li05']):.3f} dB"
        figure.suptitle(f'{scene} RGB-D：LI权重对照'+subtitle, fontproperties=font, fontsize=14)
        figure.text(.5, .015, '全局与局部按各自候选池计算预算；所有点均为完整轨迹评价，未见ratio只评价。',
                    ha='center', fontproperties=font, fontsize=10)
        figure.tight_layout(rect=[0, .045, 1, .95])
        figure.savefig(output/(scene+'_LI05_COMPARISON.png'), dpi=180)
        figure.savefig(output/(scene+'_LI05_COMPARISON.pdf'))
        plt.close(figure)


if __name__ == '__main__':
    main()
