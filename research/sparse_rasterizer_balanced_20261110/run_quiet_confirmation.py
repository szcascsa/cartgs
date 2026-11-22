"""降低监控自身的CPU/驱动查询开销，独立复核两轮性能。"""
from run_stages import *

OUT=HERE/'reports/quiet'


def run_quiet(name,command):
    observations=[];started=time.time()
    with (OUT/(name+'.log')).open('x') as log:
        child=subprocess.Popen(command,cwd=HERE,stdout=log,stderr=subprocess.STDOUT)
        identity=dict(pid=child.pid,started=started,command=command,
            start_ticks=Path(f'/proc/{child.pid}/stat').read_text().rpartition(')')[2].split()[19],
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
        (OUT/(name+'_PROCESS.json')).write_text(json.dumps(identity,indent=2))
        while child.poll() is None:
            tick=time.monotonic();observations.append(telemetry({os.getpid(),child.pid}))
            try:child.wait(timeout=max(.01,1.-(time.monotonic()-tick)))
            except subprocess.TimeoutExpired:pass
        code=child.wait()
    result=dict(returncode=code,elapsed_s=time.time()-started,telemetry_interval_s=1,
        foreign_active_samples=sum(bool(o['foreign_active']) for o in observations))
    (OUT/(name+'_MONITOR.json')).write_text(json.dumps(observations,indent=2))
    (OUT/(name+'_RESULT.json')).write_text(json.dumps(result,indent=2))
    print('QUIET_STAGE',name,json.dumps(result),flush=True)
    assert code==0 and result['foreign_active_samples']==0,(name,result)


def main():
    formal=HERE/'reports/formal'
    assert json.loads((formal/'COMPLETE.json').read_text())['passed']
    OUT.mkdir(parents=True,exist_ok=False)
    source=json.loads((formal/'SOURCE_MANIFEST.json').read_text())
    identity=json.loads((formal/'BINARY.json').read_text())
    assert digest(identity['path'])==identity['sha256']
    assert all(digest(HERE/p)==v for p,v in source.items())
    python=[sys.executable,'-u']
    for j in [1,2]:
        run_quiet('round'+str(j),python+[str(HERE/'benchmark_balanced.py'),'--all-scenes','--mode','4','--output',str(OUT/f'round{j}')])
    for j in [1,2]:
        run_quiet('pipeline'+str(j),python+[str(HERE/'benchmark_pipeline.py'),'--output',str(OUT/f'pipeline{j}')])
    before=json.loads((formal/'PROTECTED_BEFORE.json').read_text())
    changed=[p for p,v in before.items() if not Path(p).is_file() or digest(p)!=v]
    after={p:digest(p) if Path(p).is_file() else None for p in before}
    (OUT/'PROTECTED_AFTER.json').write_text(json.dumps(after,indent=2))
    result=dict(passed=not changed,files=len(before),changed=changed,
        source_unchanged=all(digest(HERE/p)==v for p,v in source.items()),
        binary_unchanged=digest(identity['path'])==identity['sha256'])
    assert all(result[k] for k in ['passed','source_unchanged','binary_unchanged']),result
    (OUT/'COMPLETE.json').write_text(json.dumps(result,indent=2))
    print('QUIET_CONFIRMATION_COMPLETE',flush=True)


if __name__=='__main__':main()
