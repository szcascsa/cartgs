"""四方法引言图，使用更新后的4000点时间并标注标准差和CV。"""
import hashlib
import json
from pathlib import Path

import numpy as np

import analyze_plot as plot


def main():
    reports = plot.OUT
    source = reports / 'outlier_sensitivity'
    output = reports / 'four_methods_cv'
    output.mkdir(exist_ok=True)
    inputs = [
        source / 'UPDATED_ALL_METHODS_PER_VIEW.csv',
        source / 'UPDATED_ALL_METHODS_SUMMARY.json',
        reports / 'PAPER_INTRO_PER_VIEW.csv',
    ]
    hashes = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in inputs}
    plot.METHODS = ['cartgs_full', 'ratio_50', 'fixed_2000', 'fixed_4000']
    rows = [row for row in plot.read(inputs[0]) if row['method'] in plot.METHODS]
    summaries = json.loads(inputs[1].read_text(encoding='utf-8'))
    summaries = [row for row in summaries if row['method'] in plot.METHODS]
    assert len(rows) == 3168 and len(summaries) == 4

    for row in rows:
        row['frame'] = int(row['frame'])
        for key in ('K', 'contributing_K', 'psnr', 'psnr_smooth', 'total_ms'):
            row[key] = float(row[key])
        if row['method'] != 'cartgs_full':
            row['network_ratio'] = float(row['network_ratio'])

    for summary in summaries:
        values = sorted(
            [row for row in rows if row['method'] == summary['method']],
            key=lambda row: row['frame'],
        )
        assert [row['frame'] for row in values] == list(range(792))
        times = np.asarray([row['total_ms'] for row in values])
        assert abs(times.std(ddof=0) - summary['total_ms_std']) < 1e-12
        cv = 100 * times.std(ddof=0) / times.mean()
        assert abs(cv - summary['per_view_latency_cv_percent']) < 1e-12

    original = {
        (row['method'], int(row['frame'])): row
        for row in plot.read(inputs[2])
    }
    changed = []
    for row in rows:
        previous = original[row['method'], row['frame']]
        for key in ('K', 'contributing_K', 'psnr', 'psnr_smooth'):
            assert row[key] == float(previous[key])
        if row['total_ms'] != float(previous['total_ms']):
            changed.append((row['method'], row['frame']))
    assert changed == [('fixed_4000', 0)]

    plot.OUT = output
    plot.write(output / 'PLOT_DATA.csv', rows)
    plot.write(output / 'SUMMARY.csv', summaries)
    plot.save(output / 'SUMMARY.json', summaries)
    plot.draw(rows, 'EN', latency_summary=summaries)

    protocol = json.loads((reports / 'PLOT_PROTOCOL.json').read_text(encoding='utf-8'))
    protocol.update(
        methods=plot.METHODS,
        figure_language='English',
        timing=dict(
            smoothing=False,
            reduction='equal-weight per-view means; two repeats at fixed_4000 frame 0 and three elsewhere',
            scope='prepare + render + final synchronization',
        ),
        latency_statistics=dict(
            std_ddof=0,
            cv_formula='100 * population_std(per_view_total_ms) / mean(per_view_total_ms)',
            view_weights='equal',
            location='inside total latency panel',
        ),
        excluded_measurement=dict(
            method='fixed_4000', frame=0, repeat=0, attempt=0,
            original_total_ms=1.9174553453922272,
            remaining_repeats_at_frame=2,
            views_retained=792,
            selection='user requested post hoc outlier exclusion',
        ),
    )
    plot.save(output / 'PLOT_PROTOCOL.json', protocol)

    lines = [
        '# 四方法引言图及总耗时波动指标\n',
        '保留CaRtGS原版、局部ratio 0.5、固定2000点和固定4000点，使用2×3版式。中央下格为统一图例，右列为总耗时曲线及标准差、CV表。\n',
        '| 方法 | 平均总耗时ms | 标准差ms | CV |',
        '|---|---:|---:|---:|',
    ]
    for summary in summaries:
        lines.append(
            f'| {summary["name"]} | {summary["total_ms_mean"]:.4f} | '
            f'{summary["total_ms_std"]:.5f} | {summary["per_view_latency_cv_percent"]:.4f}% |'
        )
    lines.extend([
        '\n总耗时包含准备、选择、补偿、打包、光栅化与最终同步。时间曲线直接使用逐视角重复测量均值，不做跨帧平滑。σ采用792个逐视角值的总体标准差（ddof=0），CV=100σ/平均总耗时。',
        '固定4000点第0帧删除repeat0的一次1.9174553ms测量，用剩余两次测量重算该帧。全部792个视角均保留，图中以星号说明这一用户指定的后验离群点处理。其原始全量标准差为0.0217479ms，CV为2.70297%。',
        'PSNR保持sigma=35帧、truncate=3、reflect的强平滑。点数为渲染后实际Alpha贡献点。固定预算是目标K，原slot资格不足时采用实际K。',
        'CaRtGS所有逐视角数据、各方法的质量与点数，以及其他三种方法的时间曲线均与此前数据逐值一致。',
        '\n![四方法引言图](figures/PAPER_INTRO_EN.png)\n',
    ])
    (output / 'FIGURE_NOTES.md').write_text('\n'.join(lines), encoding='utf-8')
    assert all(hashlib.sha256(Path(name).read_bytes()).hexdigest() == value for name, value in hashes.items())
    plot.save(output / 'VALIDATION.json', dict(
        passed=True, methods=4, frames_each=792, plot_rows=3168,
        changed_latency_points=changed, cartgs_data_unchanged=True,
        psnr_and_counts_unchanged=True, timing_smoothing=False,
        statistics_recomputed_from_plotted_values=True,
        input_hashes=hashes,
    ))
    print('FOUR_METHODS_CV_FIGURE_PASS', output)


if __name__ == '__main__':
    main()
