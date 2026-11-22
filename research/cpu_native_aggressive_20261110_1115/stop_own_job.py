"""仅终止本轮登记且PID/start_ticks/boot_id仍一致的指定子进程，并保留审计。"""
from pathlib import Path
import json,os,signal,sys,time
HERE=Path(__file__).resolve().parent
name=sys.argv[1];reason=sys.argv[2];identity=json.loads((HERE/'reports/jobs'/(name+'.json')).read_text())
assert identity['state']=='running'
pid=identity['child_pid'];fields=Path(f'/proc/{pid}/stat').read_text().rpartition(')')[2].split()
assert fields[19]==identity['child_start_ticks'] and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==identity['boot_id']
audit=dict(job=name,pid=pid,start_ticks=fields[19],boot_id=identity['boot_id'],time=time.time(),signal='SIGTERM',reason=reason)
path=HERE/'reports'/('STOP_AUDIT_'+name+'.json');assert not path.exists();path.write_text(json.dumps(audit,indent=2))
os.kill(pid,signal.SIGTERM);print('STOPPED_OWN_VERIFIED_JOB',name,pid,flush=True)
