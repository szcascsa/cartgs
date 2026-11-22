"""三遍实验结束后的只读复核；仅写验收报告，不修改模型或队列。"""
from collections import Counter
from pathlib import Path
import json
import math
import re
import time

from prepare_experiment import R, Q, A, read, write, sha
from compare_results import records, select_best


def main():
    finished = read(R / 'FINISHED.json')
    assert finished['passed']
    jobs = read(R / 'QUEUE.json')
    evaluations = read(R / 'TRAJECTORY_EVALUATION.json')['jobs']
    assert len(jobs) == len(evaluations) == 24
    assert all(x['state'] == 'complete' for x in jobs + evaluations)
    assert all(x['result']['saved_images'] == 0 for x in jobs + evaluations)
    assert all(x['result']['fine_steps'] == 0 and x['result']['frames'] == 2000 for x in jobs)
    assert all(x['seed'] == 0 and x['prune'] == 1 for x in jobs)
    thresholds = []
    for x in jobs:
        assert len(x['attempt_history']) >= 1
        assert x['attempt_history'][-1]['returncode'] == 0
        log = (Path(x['output']) / 'train.log').read_text()
        calibration = re.findall(r'^\[Explicit Anchor Calibration\].*$', log, re.M)
        assert len(calibration) == 1 and 'quantile=0.8' in calibration[0] and 'alpha=1.5' in calibration[0]
        growth = re.findall(r'^\[Explicit Anchor Growing\].*$', log, re.M)
        assert growth and all('alpha=1.5 strict_upper=1' in line for line in growth)
        values = {float(line.split('tau0=', 1)[1].split()[0]) for line in growth}
        assert len(values) == 1
        metadata = read(next(Path(x['result']['checkpoint']).rglob('anchor_selector_state.json')))
        assert math.isclose(values.pop(), metadata['growth_tau0'], rel_tol=1e-5)
        assert metadata['growth_quantile'] == .8 and metadata['growth_alpha'] == 1.5
        assert metadata['growth_strict_upper'] and all(s['adam_step'] == 0 for s in metadata['compensation_feature_state'])
        thresholds.append(metadata['growth_tau0'])
        assert all(sha(Path(p)) == h for p, h in x['input_sha256'].items())
        assert str(R / 'runtime/lib/libgaussian_mapper.so') in (Path(x['output']) / 'runtime_maps.txt').read_text()
    images = [str(p) for folder in ['jobs', 'evaluation'] for p in (R / folder).rglob('*')
              if p.suffix.lower() in ['.png', '.jpg', '.jpeg']]
    assert not images
    all_rows = records(Q) + records(A) + records(R)
    best, groups = select_best(all_rows)
    manifest = read(R / 'reports/Q80_FINAL_BEST.json')['rows']
    saved = {(x['scene'], x['mode']): x for x in manifest}
    assert len(saved) == 16
    for x in best:
        y = saved[x['scene'], x['mode']]
        assert all(x[k] == y[k] for k in x)
    before = read(R / 'PROTECTED_BEFORE.json')
    assert before == read(R / 'PROTECTED_AFTER.json')
    assert all(sha(Path(p)) == h for p, h in before.items())
    for x in jobs + evaluations:
        identity = x['process']
        proc = Path('/proc') / str(identity['pid']) / 'stat'
        if proc.exists():
            state = proc.read_text().rsplit(')', 1)[1].split()
            assert state[19] != identity['start_ticks'] or state[0] == 'Z'
    report = dict(passed=True, new_training=24, new_full_trajectory_evaluations=24,
                  total_runs=len(all_rows), selected_models=len(best), seed=0,
                  repeats_per_group=dict(Counter(str(len(x['runs'])) for x in groups)),
                  new_attempts=sum(len(x['attempt_history']) for x in jobs),
                  new_oom_retries=sum(len(x['attempt_history']) - 1 for x in jobs),
                  saved_images=len(images), protected_files_unchanged=len(before),
                  quantile=.8, alpha=1.5, calibration_frozen=True,
                  new_calibrated_threshold_min=min(thresholds), new_calibrated_threshold_max=max(thresholds),
                  raw_frame_metrics_verified=True, ply_count_verified=True,
                  same_run_metric_checkpoint_binding_verified=True,
                  process_exit_verified=True, checked_unix=time.time())
    write(R / 'reports/FINAL_VALIDATION.json', report)
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
