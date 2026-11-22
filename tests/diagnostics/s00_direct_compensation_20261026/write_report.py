"""将锁定指标转为中文结论，保留单seed与归因边界。"""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EVIDENCE = ROOT / 'evidence'
with (EVIDENCE / 'analysis/group_summary.csv').open() as stream:
    values = {(r['config'], r['split'], r['policy'], r['group']):r for r in csv.DictReader(stream)}


def score(config, group='all', metric='psnr'):
    return float(values[config, 'test', 'same_k', group][metric])


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
        ['| ' + ' | '.join(str(value) for value in row) + ' |' for row in rows])


base = 'S00R00E08D12'
configs = [base, 'S00R00E50D12', 'DirectOffsetE08', 'DirectRawE08', 'DirectOffsetE16', 'DirectRawE16', 'R3']
original = (EVIDENCE / 'S00_DIRECT_COMPENSATION_AUDIT.md').read_text(encoding='utf-8')
lead = f'''## 结论

**没有证明共享32维混合表示是主要瓶颈。** 同样8维latent下，绕过h32的Offset组下降{score(base)-score('DirectOffsetE08'):.3f} dB，原始属性组下降{score(base)-score('DirectRawE08'):.3f} dB。

16维直接输入产生正向信号：Offset组比E8旧路径高{score('DirectOffsetE16')-score(base):.3f} dB，原始属性组高{score('DirectRawE16')-score(base):.3f} dB。但本轮没有16维latent+h32的匹配对照，因此不能把16维组的提升单独归因于去掉混合表示。

本轮最高PSNR的新增方案为DirectRawE16，但仍比同seed的E50共享编码方案低{score('S00R00E50D12')-score('DirectRawE16'):.3f} dB、比R3低{score('R3')-score('DirectRawE16'):.3f} dB。所有四组均未接近R3，保留生产E50默认。

单次单场景结果仅用于快速筛查，不表述为稳定提高或统计显著，不从中自动选模型部署。

'''
body = original.split('\n', 1)[1]
report = original.split('\n', 1)[0] + '\n\n' + lead + body
report += '\n## 8→16维与原始信息的配对比较\n\n'
comparisons = [('DirectOffsetE16', 'DirectOffsetE08'), ('DirectRawE16', 'DirectRawE08'),
               ('DirectRawE08', 'DirectOffsetE08'), ('DirectRawE16', 'DirectOffsetE16')]
report += table(['比较（后者为参考）', '七档ΔPSNR', '训练四档ΔPSNR', '未见三档ΔPSNR'],
    [[a+' − '+b, *[f'{score(a,g)-score(b,g):+.3f}' for g in ['all','train','unseen']]] for a,b in comparisons])
report += '''

8→16维两种输入路径均提高本次平均PSNR；Offset组的变化主要在未见ratio，不能解释成训练ratio普遍提高。16维Raw比Offset平均仅高约0.062 dB，训练ratio更好但未见ratio更低，说明附加属性整组也存在权衡。

输入建议：优先保留局部Offset及显式log(ratio)作为空间身份与连续预算条件；Scale、Opacity和SH DC可直接供补偿使用。本轮只比较这几项整体加入，尚未单独验证每项的贡献，不能宣称它们逐项必需或已确定最优输入组合。

若后续继续归因，价值最高的是匹配16维latent+h32控制，而不是继续盲目增加latent或层数；本轮没有追加该实验。

## 七档等权：四指标

'''
report += table(['方案', 'PSNR↑', 'SSIM↑', 'LPIPS↓', 'photo↓'],
    [[c, *[f'{score(c,metric=m):.5f}' for m in ['psnr','ssim','lpips','photo']]] for c in configs])
with (EVIDENCE / 'analysis/quality_summary.csv').open() as stream:
    rows = [r for r in csv.DictReader(stream) if r['split']=='test' and r['policy']=='same_k']
lookup = {(r['config'], float(r['ratio'])):r for r in rows}
ratios = sorted({float(r['ratio']) for r in rows})
report += '\n\n## 逐ratio PSNR与严格相同预算\n\n'
report += table(['ratio', 'K', *configs], [[f'{r*100:g}%', lookup[base,r]['selected'],
    *[f'{float(lookup[c,r]["psnr"]):.3f}' for c in configs]] for r in ratios])
report += '''

DirectOffsetE08在2.5%出现明显退化；DirectOffsetE16在5%仍低于E8，DirectRawE16在7.5%和12.5%也没有提升。平均值不能解释成所有档位均变好。所有5684条Bank留出/non-Bank、same-K/Cached逐视图记录的四项质量指标均finite。

## 训练与验证边界

- 四个新增模型均完成12000步，无空选择训练步；每ratio曝光3000次。初始选择相同、补偿exact零，所有源地图张量保持冻结。
- 私有latent梯度gate是在临时设置非零输出投影后验证链路；实际零输出初始化的第一步与旧方案一样，latent初始梯度可以为零。没有把gate解释为实际首步latent即更新。
- 本地6296个已有tracked文件SHA核对无变化，暂存区为空；本轮仅增加隔离实验与报告。服务器没有替换生产实现或在线二进制。
- 同GPU另有工作负载；耗时记录仅用于实验完整性，不作网络速度比较。
- 评价沿用历史non-Bank轨迹，不是全新盲测。优化顺序seed只有一个，不估计跨seed稳定性或以逐视图数量代替独立重复。
'''
(EVIDENCE / 'S00_DIRECT_COMPENSATION_AUDIT.md').write_text(report, encoding='utf-8')
report = report.replace('(analysis/', '(tests/diagnostics/s00_direct_compensation_20261026/evidence/analysis/')
target = ROOT.parents[2] / 'S00_DIRECT_COMPENSATION_AUDIT.md'
target.write_text(report, encoding='utf-8')
print(target)
