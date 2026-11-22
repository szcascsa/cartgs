"""按相同时间阶段汇总三轮负载实验，保留每轮原始时序。"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


METHODS = ('cartgs_full', 'ours_fixed4000', 'ours_adaptive')
COLORS = ('#202124', '#D55E00', '#0072B2')
LABELS = ('CaRtGS Full', 'Ours, fixed 4k', 'Ours, adaptive K')


def read(path):
    with Path(path).open(encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def write(path, rows):
    with Path(path).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    run = args.run
    output = run / 'reports'
    output.mkdir(exist_ok=True)
    environment = json.loads((run / 'ENVIRONMENT.json').read_text())
    completion = json.loads((run / 'COMPLETION.json').read_text())
    assert completion['state'] == 'complete'
    rows = read(run / 'TIMINGS.csv')
    qualities = read(run / 'QUALITY.csv')
    quality = {(v['method'], int(v['budget'])): float(v['psnr']) for v in qualities}
    target = environment['target_ms']
    summary = []
    telemetry = []
    recovery = []
    activity = json.loads((run / 'pmon_samples.json').read_text())
    for repeat in sorted({int(v['repeat']) for v in rows}):
        for method in METHODS:
            part = run / f'{repeat}_{method}'
            if not part.exists():
                continue
            accepted = json.loads((part / 'ACCEPTANCE.json').read_text())
            assert accepted['passed']
            worker = json.loads((part / 'LOAD_IDENTITY.json').read_text())['pid']
            samples = [v for v in rows if v['method'] == method and int(v['repeat']) == repeat]
            stages = accepted['stages']
            for stage, duty in enumerate(environment['duties']):
                selected = [v for v in samples if int(v['stage']) == stage]
                assert selected
                times = np.asarray([float(v['total_ms']) for v in selected])
                ks = np.asarray([int(v['K']) for v in selected])
                psnr = [quality['cartgs_full', 35301] if method == 'cartgs_full' else quality['ours', int(v['budget'])] for v in selected]
                stage_start = stages[stage]['monotonic_s']
                stage_end = stage_start + environment['stage_seconds']
                observed = [r['sm'] for item in activity if stage_start+1 <= item['monotonic_s'] < stage_end
                    for r in item['records'] if r['pid'] == worker]
                summary.append(dict(method=method, repeat=repeat, stage=stage, target_duty=duty,
                    samples=len(times), mean_K=float(ks.mean()), min_K=int(ks.min()), max_K=int(ks.max()),
                    mean_ms=float(times.mean()), std_ms=float(times.std()), cv_percent=float(100*times.std()/times.mean()),
                    p50_ms=float(np.percentile(times, 50)), p95_ms=float(np.percentile(times, 95)), p99_ms=float(np.percentile(times, 99)),
                    over_target_percent=float(100*np.mean(times > target)), mean_psnr=float(np.mean(psnr)),
                    background_sm_mean=float(np.mean(observed)) if observed else None))
            for item in activity:
                if accepted['start_monotonic_s'] <= item['monotonic_s'] <= accepted['end_monotonic_s']:
                    record = next((r for r in item['records'] if r['pid'] == worker), None)
                    telemetry.append(dict(method=method, repeat=repeat,
                        elapsed_s=item['monotonic_s']-accepted['start_monotonic_s'], load_pid=worker,
                        load_sm=None if record is None else record['sm'], load_mem=None if record is None else record['mem']))
            final = [v for v in samples if int(v['stage']) == 5]
            restored = next((v for v in final if method == 'ours_adaptive' and int(v['budget']) == 4000), None)
            recovery.append(dict(method=method, repeat=repeat, first_recovered_4k_seconds=None if restored is None else float(restored['elapsed_s'])-5*environment['stage_seconds'],
                final_K=int(final[-1]['K']), final_budget=int(final[-1]['budget'])))
    write(output / 'STAGE_SUMMARY.csv', summary)
    write(output / 'GPU_ACTIVITY.csv', telemetry)
    write(output / 'RECOVERY.csv', recovery)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
    for repeat in sorted({int(v['repeat']) for v in rows}):
        fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
        end = len(environment['duties']) * environment['stage_seconds']
        edges = np.arange(0, end + .001, environment['stage_seconds'])
        axes[0].step(edges, [100*v for v in environment['duties']] + [0], where='post', color='#6B7280', lw=2, label='Target duty')
        for method, color, label in zip(METHODS, COLORS, LABELS):
            selected = [v for v in rows if v['method'] == method and int(v['repeat']) == repeat]
            if not selected:
                continue
            x = [float(v['elapsed_s']) for v in selected]
            axes[1].plot(x, [int(v['K']) for v in selected], color=color, lw=1, label=label)
            axes[2].plot(x, [float(v['total_ms']) for v in selected], color=color, lw=.5, alpha=.75, label=label)
            observed = [v for v in telemetry if v['method'] == method and v['repeat'] == repeat and v['load_sm'] is not None]
            axes[0].plot([v['elapsed_s'] for v in observed], [v['load_sm'] for v in observed], color=color, lw=.8, alpha=.8)
        axes[0].set(title='Controlled background load', ylabel='Duty / per-process SM (%)', ylim=(0, 105))
        axes[1].set(title='Actual input Gaussians', ylabel='K')
        axes[2].set(title='Preparation + rendering', ylabel='Total latency (ms)')
        axes[2].axhline(target, color='#B91C1C', ls='--', label=f'Target {target:.3f} ms')
        for axis in axes:
            axis.set_xlabel('Elapsed time (s)')
            axis.set_xlim(0, end)
            axis.grid(alpha=.18)
            for boundary in edges[1:-1]:
                axis.axvline(boundary, color='#AAAAAA', lw=.5, ls=':')
        axes[0].legend(fontsize=8, frameon=False)
        axes[1].legend(fontsize=8, frameon=False)
        axes[2].legend(fontsize=8, frameon=False)
        fig.tight_layout()
        for ext in ('png', 'pdf'):
            fig.savefig(output / f'LOAD_RESPONSE_REPEAT{repeat}.{ext}', dpi=220)
        plt.close(fig)
    integrity = json.loads((run / 'INTEGRITY.json').read_text())
    assert integrity['passed']
    (output / 'ANALYSIS.json').write_text(json.dumps(dict(passed=True, target_ms=target,
        rows=len(rows), stages=len(summary), telemetry_rows=len(telemetry),
        timing_smoothing=False, dropped_samples=0, quality_timing_separate=True), indent=2))
    print('LOAD_ANALYSIS_PASS', output)


if __name__ == '__main__':
    main()
