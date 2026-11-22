"""本轮实验的退出码、低频GPU监控与进程身份，不干预外部进程。"""
from pathlib import Path
import json,os,subprocess,sys,time

HERE=Path(__file__).resolve().parent


def ticks(pid):return Path(f'/proc/{pid}/stat').read_text().rpartition(')')[2].split()[19]


def descendant(pid,ancestor):
    for _ in range(16):
        if pid==ancestor:return True
        try:pid=int(Path(f'/proc/{pid}/stat').read_text().rpartition(')')[2].split()[1])
        except (FileNotFoundError,ProcessLookupError):return False
        if pid<=1:return False
    return False


def cpu_snapshot():
    result={}
    for path in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields=path.read_text().rpartition(')')[2].split()
            result[int(path.parent.name)]=(fields[19],int(fields[11])+int(fields[12]))
        except (FileNotFoundError,ProcessLookupError,PermissionError):pass
    return result


def main():
    path=Path(sys.argv[1]);identity=json.loads(path.read_text());name=identity['name']
    identity.update(runner_pid=os.getpid(),runner_start_ticks=ticks(os.getpid()),
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),started_at=time.time(),state='running')
    stdout=(HERE/'logs'/(name+'.log')).open('x')
    process=subprocess.Popen(identity['command'],cwd=HERE,stdin=subprocess.DEVNULL,stdout=stdout,stderr=subprocess.STDOUT)
    identity.update(child_pid=process.pid,child_start_ticks=ticks(process.pid));path.write_text(json.dumps(identity,indent=2))
    foreign=0;observations=0;cpu_foreign=0
    cpu_previous=cpu_snapshot();cpu_time=time.monotonic();hz=os.sysconf('SC_CLK_TCK')
    with (HERE/'logs'/(name+'_MONITOR.jsonl')).open('x') as monitor:
        while process.poll() is None:
            tick=time.monotonic()
            raw=subprocess.check_output(['nvidia-smi','pmon','-c','1','-s','um'],text=True)
            gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,clocks.sm,temperature.gpu,power.draw','--format=csv,noheader'],text=True).strip()
            active=[]
            for line in raw.splitlines():
                f=line.split()
                if len(f)>=7 and f[0].isdigit() and f[1].isdigit() and int(f[1]) not in {os.getpid(),process.pid}:
                    if any(v.isdigit() and int(v)>0 for v in f[3:7]) and not descendant(int(f[1]),process.pid):active.append(line)
            foreign+=bool(active);observations+=1
            cpu_now=cpu_snapshot();now=time.monotonic();cpu_active=[]
            for pid,(start_ticks,used) in cpu_now.items():
                old=cpu_previous.get(pid)
                if old is not None and old[0]==start_ticks:
                    percent=100*(used-old[1])/hz/max(.001,now-cpu_time)
                    if percent>10 and pid not in [os.getpid(),process.pid] and not descendant(pid,process.pid):cpu_active.append(dict(pid=pid,cpu_percent=percent,start_ticks=start_ticks))
            cpu_foreign+=bool(cpu_active);cpu_previous=cpu_now;cpu_time=now
            monitor.write(json.dumps(dict(time=time.time(),gpu=gpu,pmon=raw,foreign_active=active,foreign_cpu_active=cpu_active))+'\n');monitor.flush()
            try:process.wait(timeout=max(.01,1.-(time.monotonic()-tick)))
            except subprocess.TimeoutExpired:pass
    validation_path=HERE/identity['output_dir']/'VALIDATION.json'
    validation=json.loads(validation_path.read_text()) if validation_path.is_file() else {}
    identity.update(returncode=process.wait(),ended_at=time.time(),elapsed_s=time.time()-identity['started_at'],
        foreign_active_samples=foreign,observations=observations,state='complete' if process.returncode==0 else 'failed',
        foreign_cpu_active_samples=cpu_foreign,
        measurement_valid=process.returncode==0 and validation.get('passed',False),
        acceptance_scope='benchmark accepts only clean +/-2s windows; initial idle waiting is excluded',
        accepted_blocks=validation.get('accepted_blocks',0),rejected_blocks=validation.get('rejected_blocks',0))
    path.write_text(json.dumps(identity,indent=2));stdout.close();print(json.dumps(identity,indent=2),flush=True)


if __name__=='__main__':main()
