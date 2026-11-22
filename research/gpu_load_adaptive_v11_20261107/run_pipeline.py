"""等待GPU自然空闲，串行执行smoke、正式测量与分析。"""
from pathlib import Path
import argparse
import csv
import fcntl
import json
import os
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke-label', required=True)
    parser.add_argument('--formal-label', required=True)
    parser.add_argument('--allow-recorded-background', action='store_true')
    args = parser.parse_args()
    lock = (ROOT / 'reports/pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    for label in (args.smoke_label, args.formal_label):
        if (ROOT / 'runs' / label).exists():
            raise FileExistsError('Preserve existing experiment ' + label)
    identity = dict(pid=os.getpid(), start_ticks=Path('/proc/self/stat').read_text().split()[21],
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
        smoke_label=args.smoke_label, formal_label=args.formal_label, started_unix=time.time())
    (ROOT / 'reports/PIPELINE_CONTROLLER.json').write_text(json.dumps(identity, indent=2))
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
    for name, command in [
        (args.smoke_label, [sys.executable, '-u', str(HERE / 'benchmark.py'), '--smoke', '--label', args.smoke_label]),
        (args.formal_label, [sys.executable, '-u', str(HERE / 'benchmark.py'), '--label', args.formal_label]),
        (args.formal_label + '_analysis', [sys.executable, '-u', str(HERE / 'analyze.py'), str(ROOT / 'runs' / args.formal_label)]),
    ]:
        if args.allow_recorded_background and name in (args.smoke_label, args.formal_label):
            command.append('--allow-recorded-background')
        with (ROOT / 'logs' / f'{name}.log').open('w') as log:
            subprocess.run(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, check=True)
        if name == args.smoke_label:
            smoke = ROOT / 'runs' / name
            assert json.loads((smoke / 'INTEGRITY.json').read_text())['passed']
            assert json.loads((smoke / '0_ours_adaptive/ACCEPTANCE.json').read_text())['passed']
            with (smoke / '0_ours_adaptive/load.csv').open() as stream:
                load_rows = list(csv.DictReader(stream))
            for stage, duty in enumerate((0., .2, .4, .6, .8, 0.)):
                selected = [r for r in load_rows if int(r['stage']) == stage]
                assert len(selected) >= 3, ('Missing smoke load stage', stage)
                assert all(float(r['target_duty']) == duty for r in selected)
                assert all((float(r['event_ms']) > 0) == (duty > 0) for r in selected)
            (smoke / 'SMOKE_VALIDATION.json').write_text(json.dumps(dict(passed=True,
                stages=6, image_bitwise_and_integrity=True, background_activity_recorded=True,
                recorded_background_allowed=args.allow_recorded_background,
                load_pulses_present=True, completed_unix=time.time()), indent=2))
    (ROOT / 'reports/PIPELINE_COMPLETION.json').write_text(json.dumps(dict(passed=True,
        smoke_label=args.smoke_label, formal_label=args.formal_label, completed_unix=time.time()), indent=2))
    print('GPU_LOAD_PIPELINE_COMPLETE', flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        (ROOT / 'reports/PIPELINE_FAILURE.json').write_text(json.dumps(dict(error=repr(error), unix=time.time()), indent=2))
        raise
