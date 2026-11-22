"""记录pmon活动，外部任务与本轮有意施加的GPU负载分别识别。"""
from pathlib import Path
import json
import os
import subprocess
import threading
import time


class ActivityMonitor:
    def __init__(self, folder, permitted_background=()):
        self.folder = Path(folder)
        self.permitted_background = set(permitted_background)
        self.samples = []
        self.identities = {}
        self.error = None
        self.process = subprocess.Popen(
            ['nvidia-smi', 'pmon', '-s', 'um', '-o', 'T', '-d', '1'],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
        )
        self.thread = threading.Thread(target=self._collect, daemon=True)
        self.thread.start()

    def _collect(self):
        current = None
        records = []
        received = None
        try:
            with (self.folder / 'pmon_raw.log').open('w', buffering=1) as log:
                for line in self.process.stdout:
                    log.write(line)
                    fields = line.split()
                    if not fields or fields[0].startswith('#'):
                        continue
                    if len(fields) < 9:
                        raise RuntimeError('Unexpected pmon record ' + line)
                    if fields[0] != current:
                        if current is not None:
                            self.samples.append(dict(clock=current, monotonic_s=received, records=records))
                        current = fields[0]
                        records = []
                        received = time.monotonic()
                    pid = 0 if fields[2] == '-' else int(fields[2])
                    if pid and pid not in self.identities:
                        proc = Path('/proc') / str(pid)
                        try:
                            self.identities[pid] = dict(start_ticks=proc.joinpath('stat').read_text().split()[21],
                                cmdline=proc.joinpath('cmdline').read_text().replace(chr(0), ' '))
                        except FileNotFoundError:
                            self.identities[pid] = dict(exited_before_identity_read=True)
                    records.append(dict(
                        pid=pid,
                        kind=fields[3],
                        sm=0 if fields[4] == '-' else int(fields[4]),
                        mem=0 if fields[5] == '-' else int(fields[5]),
                        enc=0 if fields[6] == '-' else int(fields[6]),
                        dec=0 if fields[7] == '-' else int(fields[7]),
                    ))
        except Exception as error:
            self.error = repr(error)

    def unexpected(self, start, end, allowed_pids):
        if self.error or self.process.poll() is not None:
            raise RuntimeError(self.error or 'pmon exited')
        samples = [s for s in self.samples if start <= s['monotonic_s'] <= end]
        allowed_pids = set(allowed_pids) | getattr(self, 'permitted_background', set())
        active = [r for s in samples for r in s['records'] if r['pid'] not in allowed_pids and any(r[k] > 0 for k in ('sm', 'mem', 'enc', 'dec'))]
        return samples, active

    def quiet(self):
        if len(self.samples) < 3:
            return False
        last = self.samples[-3:]
        if time.monotonic() - last[-1]['monotonic_s'] > 3:
            return False
        samples, active = self.unexpected(last[0]['monotonic_s'], last[-1]['monotonic_s'], {os.getpid()})
        allowed = {os.getpid()} | getattr(self, 'permitted_background', set())
        external_compute = any(record['kind'] == 'C' and record['pid'] not in allowed
            for sample in samples for record in sample['records'])
        return len(samples) == 3 and not active and not external_compute

    def close(self):
        self.process.terminate()
        self.thread.join(timeout=3)
        (self.folder / 'pmon_samples.json').write_text(json.dumps(self.samples, indent=2))
        (self.folder / 'pmon_process_identities.json').write_text(json.dumps(self.identities, indent=2))
