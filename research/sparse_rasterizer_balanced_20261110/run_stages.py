"""独立目录内串行验收、两轮测速和输入完整性检查，不修改其他任务。"""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time

HERE=Path(__file__).resolve().parent
OUT=HERE/'reports/formal'


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):value.update(chunk)
    return value.hexdigest()


def telemetry(allowed):
    raw=subprocess.check_output(['nvidia-smi','pmon','-c','1','-s','um'],text=True)
    foreign=[]
    for line in raw.splitlines():
        f=line.split()
        if len(f)>=7 and f[0].isdigit() and f[1].isdigit() and int(f[1]) not in allowed:
            if any(v.isdigit() and int(v)>0 for v in f[3:7]):foreign.append(line)
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,clocks.sm,temperature.gpu,power.draw',
        '--format=csv,noheader'],text=True).strip()
    return dict(time=time.time(),gpu=gpu,pmon=raw,foreign_active=foreign)


def run(name,command,measured=False):
    observations=[];started=time.time()
    with (OUT/(name+'.log')).open('x') as log:
        process=subprocess.Popen(command,cwd=HERE,stdout=log,stderr=subprocess.STDOUT)
        identity=dict(pid=process.pid,command=command,started=started,
            start_ticks=Path(f'/proc/{process.pid}/stat').read_text().rpartition(')')[2].split()[19],
            boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
        (OUT/(name+'_PROCESS.json')).write_text(json.dumps(identity,indent=2))
        while process.poll() is None:observations.append(telemetry({os.getpid(),process.pid}))
        code=process.wait()
    (OUT/(name+'_MONITOR.json')).write_text(json.dumps(observations,indent=2))
    result=dict(returncode=code,elapsed_s=time.time()-started,
        foreign_active_samples=sum(bool(o['foreign_active']) for o in observations))
    (OUT/(name+'_RESULT.json')).write_text(json.dumps(result,indent=2))
    print('STAGE',name,json.dumps(result),flush=True)
    assert code==0,(name,result)
    assert not measured or not result['foreign_active_samples'],('Measured run interfered; retained for inspection',name)


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    before=json.loads((HERE/'reports/PROTECTED_BEFORE.json').read_text())
    assert all(Path(p).is_file() and digest(p)==v for p,v in before.items()),'Protected source changed before benchmark'
    scenes=json.loads(Path('/home/prmi/zwj/experiments/compensator_distillation_replica_all_20261108/reports/SCENES.json').read_text())
    for scene in scenes:
        for p in Path(scene['checkpoint']).rglob('*'):
            if p.is_file():before[str(p)]=digest(p)
    (OUT/'PROTECTED_BEFORE.json').write_text(json.dumps(before,indent=2))
    python=[sys.executable,'-u']
    for script in ['validate.py','test_lifecycle.py','test_boundaries.py']:
        run(Path(script).stem,python+[str(HERE/script)])
    sanitizer='/usr/local/cuda-11.8/compute-sanitizer/compute-sanitizer'
    for tool in ['memcheck','initcheck']:
        run(tool,[sanitizer,'--tool',tool,'--error-exitcode','91']+python+[str(HERE/'sanitizer_smoke.py')])
    source={str(p.relative_to(HERE)):digest(p) for p in (HERE/'source').rglob('*') if p.is_file()}
    source.update({p.name:digest(p) for p in HERE.glob('*.py')})
    binary=HERE/'build/sparse_rasterizer_balanced_v1.so';binary_sha=digest(binary)
    (OUT/'SOURCE_MANIFEST.json').write_text(json.dumps(source,indent=2))
    (OUT/'BINARY.json').write_text(json.dumps(dict(path=str(binary),sha256=binary_sha),indent=2))
    for round_id in [1,2]:
        run('round'+str(round_id),python+[str(HERE/'benchmark_balanced.py'),'--all-scenes','--mode','4',
            '--output',str(OUT/f'round{round_id}')],True)
    for round_id in [1,2]:
        run('pipeline'+str(round_id),python+[str(HERE/'benchmark_pipeline.py'),
            '--output',str(OUT/f'pipeline{round_id}')],True)
    run('profile',python+[str(HERE/'profile_balanced.py'),'--output',str(OUT/'profile')],True)
    changed=[p for p,v in before.items() if not Path(p).is_file() or digest(p)!=v]
    source_changed=[p for p,v in source.items() if digest(HERE/p)!=v]
    result=dict(passed=not changed and not source_changed and digest(binary)==binary_sha,protected_files=len(before),
        changed=changed,source_changed=source_changed,binary_unchanged=digest(binary)==binary_sha,completed_at=time.time())
    (OUT/'INTEGRITY.json').write_text(json.dumps(result,indent=2));assert result['passed'],result
    (OUT/'COMPLETE.json').write_text(json.dumps(result,indent=2))
    print('ALL_STAGES_COMPLETE',flush=True)


if __name__=='__main__':main()
