"""由已完成的原始结果生成性能曲线，不重跑设备测试。"""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

root = Path(__file__).resolve().parents[1]
font = Path('C:/Windows/Fonts/msyh.ttc')
if font.exists():
    font_manager.fontManager.addfont(str(font))
    plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 10})
data = json.loads((root / 'evidence/RESULTS.json').read_text())['results']
k = [v['k'] for v in data]
fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.8))
for ax in axes:
    for i, color in enumerate(['#4C78A8', '#E69F00']):
        ax.plot(k, [v['rounds'][i]['gpu_median_ms'] for v in data], 'o--',
                color=color, linewidth=1.4, markersize=4, label=f'第 {i + 1} 轮中位数')
    ax.plot(k, [v['app_gpu_median_ms'] for v in data], 'o-', color='#009E73',
            linewidth=2, markersize=4, label='两轮合并读取中位数')
    ax.set_xlabel('全局静态子集输入 K（非当前视角可见点数）')
    ax.set_ylabel('应用双目 GPU 时间（ms，系统聚合指标）')
    ax.grid(alpha=.22)
    ax.spines[['top', 'right']].set_visible(False)
axes[0].set_title('完整范围：0—43,197 点')
axes[0].set_xlim(-500, 44000)
axes[0].legend(loc='upper left', frameon=False)
axes[1].set_title('低点数放大：0—5,000 点')
axes[1].set_xlim(-100, 5200)
axes[1].set_ylim(1.5, 2.8)
fig.suptitle('Quest 3 · Meta 官方 SplatFeature 0.15.0 · Replica office2 RGB-D', fontsize=13)
fig.text(.5, .018, '仅静态 Master 子集；无 Selector／Compensator。3,902 次读取仅 72 种 GPU 数值，不代表独立逐帧延迟。',
         ha='center', fontsize=9, color='#555555')
fig.tight_layout(rect=(0, .075, 1, .94))
for ext in ['png', 'svg']:
    fig.savefig(root / f'report/official_gpu_vs_global_k.{ext}', dpi=180)
plt.close(fig)
print('性能图已生成，原始测量数据未修改。')
