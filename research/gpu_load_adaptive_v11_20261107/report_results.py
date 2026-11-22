"""从完整原始时序生成中文结论和共享GPU条件的汇总表。"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


METHODS = ('cartgs_full', 'ours_fixed4000', 'ours_adaptive')
NAMES = dict(zip(METHODS, ('CaRtGS Full', 'v11 固定4000点', 'v11 自适应预算')))


def read(path):
    with path.open(encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    run = args.run
    out = run / 'reports'
    env = json.loads((run / 'ENVIRONMENT.json').read_text())
    assert json.loads((run / 'COMPLETION.json').read_text())['state'] == 'complete'
    assert json.loads((out / 'ANALYSIS.json').read_text())['passed']
    rows = read(run / 'TIMINGS.csv')
    quality = read(run / 'QUALITY.csv')
    activity = json.loads((run / 'pmon_samples.json').read_text())
    background = env.get('authorized_background') or {'processes': []}
    load_pids = {json.loads(p.read_text())['pid'] for p in run.glob('*/LOAD_IDENTITY.json')}
    benchmark_pid = json.loads((run / 'COMPLETION.json').read_text())['process']['pid']
    background_pids = {v['pid'] for sample in activity for v in sample['records']
        if v['kind'] == 'C' and v['pid'] not in load_pids | {benchmark_pid}}
    assert len(rows) == sum(json.loads(p.read_text())['rows'] for p in run.glob('*/ACCEPTANCE.json'))
    assert len(list(run.glob('*/ACCEPTANCE.json'))) == 9
    assert all(np.isfinite(float(v['total_ms'])) and float(v['total_ms']) > 0 for v in rows)
    assert all(int(v['K']) == int(v['budget']) for v in rows)
    assert all(v['repeated_image_bitwise'] == 'True' for v in quality)
    summary = []
    for stage in range(6):
        for method in METHODS:
            selected = [v for v in rows if v['method'] == method and int(v['stage']) == stage]
            times = np.array([float(v['total_ms']) for v in selected])
            ks = np.array([int(v['K']) for v in selected])
            known_sm = []
            load_sm = []
            for repeat in range(3):
                part = run / f'{repeat}_{method}'
                acceptance = json.loads((part / 'ACCEPTANCE.json').read_text())
                start = acceptance['stages'][stage]['monotonic_s']
                worker = json.loads((part / 'LOAD_IDENTITY.json').read_text())['pid']
                for sample in activity:
                    # pmon是约一秒的活动窗口，避开切档后第一秒。
                    if start + 1 <= sample['monotonic_s'] < start + env['stage_seconds']:
                        known_sm.append(sum(v['sm'] for v in sample['records'] if v['pid'] in background_pids))
                        load_sm.append(sum(v['sm'] for v in sample['records'] if v['pid'] == worker))
            summary.append(dict(method=method, stage=stage, duty_percent=100*env['duties'][stage],
                samples=len(times), mean_ms=times.mean(), std_ms=times.std(), cv_percent=100*times.std()/times.mean(),
                p95_ms=np.percentile(times, 95), p99_ms=np.percentile(times, 99),
                over_target_percent=100*np.mean(times > env['target_ms']), mean_K=ks.mean(),
                min_K=int(ks.min()), max_K=int(ks.max()),
                ambient_sm_mean=np.mean(known_sm) if known_sm else None,
                controlled_load_sm_mean=np.mean(load_sm) if load_sm else None))
    with (out / 'POOLED_STAGE_SUMMARY.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    table = ['|负载阶段|方法|K范围|均值 ms|P95 ms|标准差 ms|CV %|超目标 %|既有服务 SM %|人工负载 SM %|',
             '|---|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for v in summary:
        table.append(f"|{v['stage']}: {v['duty_percent']:.0f}%|{NAMES[v['method']]}|{v['min_K']}–{v['max_K']}|"
            f"{v['mean_ms']:.4f}|{v['p95_ms']:.4f}|{v['std_ms']:.4f}|{v['cv_percent']:.2f}|{v['over_target_percent']:.2f}|"
            f"{v['ambient_sm_mean']:.1f}|{v['controlled_load_sm_mean']:.1f}|")
    qtable = ['|方法/预算|输入K|固定视角PSNR dB|', '|---|---:|---:|']
    for v in quality:
        qtable.append(f"|{v['method']} / {v['budget']}|{v['K']}|{float(v['psnr']):.4f}|")
    recovery = [v for v in read(out / 'RECOVERY.csv') if v['method'] == 'ours_adaptive']
    heavy = {v['method']: v for v in summary if v['stage'] == 4}
    a, f, c = (heavy[m] for m in ('ours_adaptive', 'ours_fixed4000', 'cartgs_full'))
    reached = sum(bool(v['first_recovered_4k_seconds']) for v in recovery)
    ended = sum(int(v['final_K']) == 4000 for v in recovery)
    quality_k = {int(v['K']): float(v['psnr']) for v in quality if v['method'] == 'ours'}
    verdict = '未达到既定延迟目标' if a['over_target_percent'] > 0 else '该阶段所有样本均达到既定延迟目标'
    recovery_text = '\n'.join(f"- 第{int(v['repeat'])+1}轮：结束K={v['final_K']}；"
        + (f"解除人工负载后{float(v['first_recovered_4k_seconds']):.3f}秒首次恢复4000点。"
           if v['first_recovered_4k_seconds'] else '最后5秒内未恢复4000点。') for v in recovery)
    comparisons = []
    for stage in range(6):
        fixed = next(v for v in summary if v['stage'] == stage and v['method'] == 'ours_fixed4000')
        adaptive = next(v for v in summary if v['stage'] == stage and v['method'] == 'ours_adaptive')
        comparisons.append(f"- 阶段{stage}（人工目标{fixed['duty_percent']:.0f}%）：自适应平均K={adaptive['mean_K']:.0f}，"
            f"相对固定4000点的均值变化{100*(adaptive['mean_ms']/fixed['mean_ms']-1):+.1f}%，"
            f"P95变化{100*(adaptive['p95_ms']/fixed['p95_ms']-1):+.1f}%；"
            f"超目标比例{adaptive['over_target_percent']:.1f}% / {fixed['over_target_percent']:.1f}%（自适应/固定）。")
    text = f'''# v11 GPU负载适应实验报告

## 主要结论

预算反馈能触发降低点数，但本轮{verdict}。人工80%负载阶段，三轮自适应均固定在1000点：合并平均总耗时{a['mean_ms']:.4f} ms，P95={a['p95_ms']:.4f} ms，{a['over_target_percent']:.1f}%请求超过预定{env['target_ms']:.4f} ms目标。同期CaRtGS Full观测均值/P95为{c['mean_ms']:.4f}/{c['p95_ms']:.4f} ms，v11固定4000点为{f['mean_ms']:.4f}/{f['p95_ms']:.4f} ms。

不能将观测加速全部归因于降低K：上述阶段固定4000点与自适应测试期间，既有服务平均SM活动分别为{f['ambient_sm_mean']:.1f}%和{a['ambient_sm_mean']:.1f}%，背景暴露不匹配。不同负载阶段也没有一致改善，20%阶段自适应观测甚至更慢。现有证据支持预算控制有响应，尚不能支持共享GPU下稳定总延迟的强结论。

解除人工负载后，{reached}/3轮曾恢复4000点，只有{ended}/3轮以4000点结束；其余结束预算和首次恢复时间见下文。该固定视角从4000降到1000点的PSNR代价为{quality_k[4000]-quality_k[1000]:.4f} dB。后续应在独占GPU环境复用同一协议隔离背景干扰；不调整本轮阈值来包装稳定性结果。

## 实验范围与判断口径

使用最新验收v11二进制，固定TUM fr1_xyz RGB-D第396帧；比较原版CaRtGS Full、v11固定4000点、v11自适应预算。三种方法分别执行同一人工负载日程，重复三遍，方法顺序循环轮换。结果是否支持稳定延迟，必须看总耗时、P95、超目标比例及恢复，而非只看K下降。

本轮按用户后续授权在既有GPU服务运行时测试。`shared_gpu={env.get('shared_gpu', False)}`；既有服务PID={sorted(background_pids)}，其活动保存在pmon原始记录。三种方法并非同时测量，既有服务的非平稳负载不能被本实验严格消除。因此表中的方法差异是当前共享GPU环境下的观测，不是独占GPU因果结论。

本次完整记录{len(rows)}次计时请求，九个条件均完成。候选池沿用名义Anchor视锥；本方法候选N={sorted({int(v['candidate_N']) for v in rows if v['method'] != 'cartgs_full'})}，资格点数E={sorted({int(v['eligible_N']) for v in rows if v['method'] != 'cartgs_full'})}。固定K和自适应K均严格满足预算。

## 固定协议

人工负载目标为0→20→40→60→80→0%，每档5秒；每毫秒调度一个周期，采用不超过200微秒的CUDA计算脉冲，避免长Kernel一直阻塞。目标占空比不等于实际GPU利用率。表中分别记录人工进程和既有服务的实测SM活动，pmon约1秒采样，不是精确的GPU资源份额。

预算档位4000→3000→2000→1000。每10帧计算窗口平均总耗时；高于目标降一档，连续两个窗口低于目标的85%升一档。目标由当时共享GPU环境下、未开启人工负载的4000点200次测量P95×1.10确定：`tau={env['target_ms']:.6f} ms`。开始正式测量前声明规则，未根据正式结果更改阈值。

固定5ms请求周期。总耗时采用CPU墙钟，从选择前到最终CUDA同步，包含Selector、Compensator、打包和渲染。相机、加载、GT和PSNR在计时外。`prepare_ms`记录提交阶段的墙钟分段，不代表纯GPU算子时间；本报告使用`total_ms`判断。所有样本保留，不平滑耗时、不剔除离群值；跨阶段帧有单独标记。标准差为总体标准差（ddof=0），CV=标准差/均值。

## 各负载阶段结果

以下合并三轮原始样本，逐轮明细见`STAGE_SUMMARY.csv`。汇总按请求数量加权，较慢轮次在固定5秒内完成的请求更少，因此不等于三轮均值的简单平均。阶段5是解除人工负载后的恢复阶段；仍可能存在既有服务负载。

{chr(10).join(table)}

## 自适应相对固定预算的观测

{chr(10).join(comparisons)}

## 恢复情况

{recovery_text}

## 质量代价

下表是同一固定视角、同一checkpoint的各预算PSNR；前后复测图像逐位一致。时序中按实际预算绑定对应质量，不在耗时测量内额外渲染。输入K是送入Rasterizer的Gaussian数量，不等于最终产生Alpha贡献的数量。CaRtGS Full输入35301点，本方法Full地图13298点；两模型使用相同RGB帧、各自SLAM估计位姿，不能将它们的质量差异全部归因于预算控制。

{chr(10).join(qtable)}

## 正确性与保护

v11 SHA256：`{env['binary_sha256']}`。启用`shrinking_radix=True, compensation_tile=4`，FP32、TF32关闭。短验收、严格K检查、重复图像一致性和前后文件完整性检查均通过，详见`INTEGRITY.json`及保护清单；短验收结果位于相邻smoke运行目录的`SMOKE_VALIDATION.json`。未训练、未保存渲染帧图片、未修改生产默认或checkpoint、未commit/push。原等待超时、初始化失败和旧固定PID监测中止的formal_01均保留现场，本报告只汇总完整重测的{run.name}；初始化问题只修正测量脚本的对象构建环境，不修改v11二进制。

## 文件

- `TIMINGS.csv`：所有请求的原始时间、实际K、预算、候选数及跨阶段标记。
- `QUALITY.csv`：各K质量、图像SHA和重复一致性。
- `STAGE_SUMMARY.csv`、`POOLED_STAGE_SUMMARY.csv`：逐轮与合并统计。
- `GPU_ACTIVITY.csv`、`pmon_samples.json`及每条件`load.csv`：实测负载。
- `LOAD_RESPONSE_REPEAT0/1/2.png/.pdf`：三轮时序图，耗时不平滑。
- `LOAD_RESPONSE_REPEAT0/1/2_SHARED.png/.pdf`：明确标注共享GPU，点数轴使用对数刻度，保留全部原始延迟点。

## 局限与下一步

只有一个固定视角和一个GPU；本轮验证预算控制响应，不代表全轨迹最优策略。受控计算负载不覆盖所有显存带宽型或多进程调度情形。既有服务的波动会影响标定、不同方法之间的配对与尾延迟。若下降到1000点仍超目标或卸载后未恢复4000点，应作为控制器局限报告；不能事后改变阈值获得更好的曲线。下一次最有信息价值的验证是使用完全相同代码、协议和checkpoint，在GPU空闲时重复三轮，隔离既有服务波动。
'''
    (out / 'GPU_LOAD_ADAPTIVE_V11_REPORT.md').write_text(text, encoding='utf-8')
    plot_shared(rows, read(out / 'GPU_ACTIVITY.csv'), env, out)
    print(out / 'GPU_LOAD_ADAPTIVE_V11_REPORT.md')


def plot_shared(rows, telemetry, env, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FixedFormatter
    colors = ('#36393F', '#D55E00', '#0072B2')
    labels = ('CaRtGS Full', 'Ours, fixed 4k', 'Ours, adaptive K')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
        'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    for repeat in range(3):
        fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.4))
        edges = np.arange(7)*env['stage_seconds']
        axes[0].step(edges, [100*v for v in env['duties']]+[0], where='post',
            color='#6B7280', lw=1.8, label='Target duty')
        for method, color, label in zip(METHODS, colors, labels):
            selected = [v for v in rows if v['method'] == method and int(v['repeat']) == repeat]
            x = [float(v['elapsed_s']) for v in selected]
            axes[1].plot(x, [int(v['K']) for v in selected], color=color, lw=1, label=label)
            axes[2].scatter(x, [float(v['total_ms']) for v in selected], color=color, s=.7, alpha=.45, linewidths=0)
            observed = [v for v in telemetry if v['method'] == method and int(v['repeat']) == repeat and v['load_sm']]
            axes[0].plot([float(v['elapsed_s']) for v in observed], [float(v['load_sm']) for v in observed],
                color=color, lw=.9, label=label+' pass')
        axes[0].set(title='Injected GPU load', ylabel='Duty / load-process SM (%)', ylim=(0, 105))
        axes[1].set(title='Actual input Gaussians', ylabel='K (log scale)', yscale='log', ylim=(850, 42000))
        axes[1].yaxis.set_major_locator(FixedLocator([1000, 2000, 4000, 10000, 35301]))
        axes[1].yaxis.set_major_formatter(FixedFormatter(['1k', '2k', '4k', '10k', '35.3k']))
        axes[1].minorticks_off()
        axes[2].set(title='Raw request latency', ylabel='Preparation + rendering (ms)', ylim=(0, None))
        axes[2].axhline(env['target_ms'], color='#B91C1C', ls='--', lw=1,
            label=f"Target {env['target_ms']:.3f} ms")
        for axis in axes:
            axis.set(xlabel='Elapsed time (s)', xlim=(0, edges[-1]))
            axis.grid(alpha=.16)
            for boundary in edges[1:-1]:
                axis.axvline(boundary, color='#AAAAAA', lw=.5, ls=':')
            axis.legend(fontsize=6.7, frameon=False, loc='upper right' if axis is axes[1] else 'upper left')
        fig.suptitle(f'v11 | Shared GPU | Fixed viewpoint | Run {repeat+1}/3', fontsize=10)
        fig.tight_layout(rect=(0, .04, 1, .94))
        fig.text(.5, .015, 'Methods tested separately; background services remained active. All latency samples retained, without smoothing.',
            ha='center', fontsize=7, color='#555555')
        for extension in ('png', 'pdf'):
            fig.savefig(out / f'LOAD_RESPONSE_REPEAT{repeat}_SHARED.{extension}', dpi=220)
        plt.close(fig)


if __name__ == '__main__':
    main()
