"""为本次corridors双目隔离系统内存，进程身份在exec前后保持一致。"""
from pathlib import Path
import json
import os
import resource
import signal
import sys
import time


def snapshot(config):
    group = Path(config['cgroup'])
    assert group.parent == Path('/sys/fs/cgroup/memory')
    assert group.name.startswith('cartgs-corridors-stereo-')
    limit = int((group/'memory.limit_in_bytes').read_text())
    assert limit == config['limit_bytes']
    assert int((group/'memory.memsw.limit_in_bytes').read_text()) == limit
    oom = dict(line.split() for line in (group/'memory.oom_control').read_text().splitlines())
    assert oom['oom_kill_disable'] == '0'
    return dict(limit_bytes=limit, oom_kill=int(oom['oom_kill']),
                failcnt=int((group/'memory.failcnt').read_text()),
                peak_bytes=int((group/'memory.max_usage_in_bytes').read_text()))


def watch(process, identity, config, output, stop):
    while not stop.wait(0.1):
        if process.poll() is not None:
            return
        values = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
        available = int(values['MemAvailable'].split()[0])*1024
        if available >= config['system_reserve_bytes']:
            continue
        try:
            fields = Path(f'/proc/{process.pid}/stat').read_text().rsplit(')', 1)[1].split()
            process_group = os.getpgid(process.pid)
        except (FileNotFoundError, ProcessLookupError):
            return
        if fields[19] != identity['start_ticks'] or process_group != process.pid:
            return
        failure = dict(reason='system_memory_reserve', available_bytes=available,
                       reserve_bytes=config['system_reserve_bytes'], process=identity,
                       checked_unix=time.time(), target_process_group=process.pid)
        Path(output, 'memory_reserve_stop.json').write_text(json.dumps(failure, indent=2))
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        return


def main():
    config_file, output, *command = sys.argv[1:]
    config = json.loads(Path(config_file).read_text())
    initial = snapshot(config)
    group = Path(config['cgroup'])
    assert not (group/'cgroup.procs').read_text().strip(), '资源组存在其他进程，拒绝加入'
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    Path('/proc/self/oom_score_adj').write_text('1000')
    (group/'cgroup.procs').write_text(str(os.getpid()))
    membership = [line for line in Path('/proc/self/cgroup').read_text().splitlines()
                  if 'memory' in line.split(':')[1].split(',')]
    assert len(membership) == 1 and membership[0].split(':')[2] == '/'+group.name
    Path(output, 'memory_guard.json').write_text(json.dumps(dict(
        pid=os.getpid(), cgroup=str(group), initial=initial, membership=membership,
        oom_score_adj=1000, checked_unix=time.time()), indent=2))
    os.execvpe(command[0], command, os.environ)


if __name__ == '__main__':
    main()
