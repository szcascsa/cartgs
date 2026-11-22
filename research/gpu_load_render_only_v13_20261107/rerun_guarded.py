"""空闲GPU准入与持续外部进程检查，复用既有v13测量协议。"""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[3]


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def wait_for_idle():
    consecutive = 0
    while consecutive < 3:
        output = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_gpu_memory',
            '--format=csv,noheader'], text=True)
        processes = [line for line in output.splitlines() if line.strip()]
        consecutive = 0 if processes else consecutive + 1
        save(ROOT/'STATUS.json', dict(state='waiting_for_idle_gpu', sampled_unix=time.time(),
            external_compute=processes, consecutive_idle_samples=consecutive,
            pid=os.getpid()))
        if consecutive < 3:
            time.sleep(5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label', required=True)
    args = parser.parse_args()
    lock = (ROOT/'reports/guard.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    wait_for_idle()
    import benchmark
    from activity import ActivityMonitor

    active = []

    class StrictMonitor(ActivityMonitor):
        def __init__(self, folder):
            super().__init__(folder)
            self.cursor = 0
            active.append(self)

        def check_new_samples(self):
            if self.error or self.process.poll() is not None:
                raise RuntimeError(self.error or 'GPU telemetry process exited')
            current = list(self.samples[self.cursor:])
            self.cursor += len(current)
            unexpected = [dict(monotonic_s=sample['monotonic_s'], **record)
                for sample in current for record in sample['records']
                if record['pid'] != os.getpid() and
                    (record['kind'] == 'C' or any(record[k] > 0 for k in ('sm','mem','enc','dec')))]
            if unexpected:
                save(self.folder/'EXTERNAL_INTERFERENCE.json', dict(passed=False, records=unexpected))
                raise RuntimeError('External GPU process appeared; preserve and reject this attempt')

        def unexpected(self, start, end, allowed_pids):
            self.check_new_samples()
            samples, unexpected = super().unexpected(start, end, allowed_pids)
            if unexpected:
                raise RuntimeError('Unexpected external GPU activity')
            return samples, unexpected

    OriginalRenderer = benchmark.Renderer

    class GuardedRenderer(OriginalRenderer):
        def call(self, *args, **kwargs):
            active[0].check_new_samples()
            return super().call(*args, **kwargs)

    benchmark.ActivityMonitor = StrictMonitor
    benchmark.Renderer = GuardedRenderer
    sys.argv = [str(Path(benchmark.__file__)), '--label', args.label]
    try:
        benchmark.main()
        # 最后的采样由原测量流程的四秒尾部覆盖，复核整个运行。
        monitor = active[0]
        records = [record for sample in monitor.samples for record in sample['records']
            if record['pid'] != os.getpid() and
                (record['kind']=='C' or any(record[key]>0 for key in ('sm','mem','enc','dec')))]
        assert not records, records
        save(ROOT/'runs'/args.label/'QUIET_GPU_ACCEPTANCE.json', dict(passed=True,
            external_compute_records=0, external_activity_records=0, sample_count=len(monitor.samples),
            idle_admission_samples=3, idle_admission_interval_seconds=5,
            telemetry_interval_seconds=1, completed_unix=time.time()))
    except Exception as error:
        save(ROOT/'reports/GUARDED_FAILURE.json', dict(label=args.label, error=repr(error), unix=time.time()))
        raise


if __name__ == '__main__':
    main()
