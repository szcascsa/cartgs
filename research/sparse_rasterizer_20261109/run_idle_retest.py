"""不修改CUDA实现，运行两轮空闲GPU配对复测并留存环境/完整性。"""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import threading
import time

HERE=Path(__file__).resolve().parent
OUT=HERE/'reports/idle_retest_20261110'


def read(path):return json.loads(path.read_text())


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def telemetry(allowed):
    raw=subprocess.check_output(['nvidia-smi','pmon','-c','1','-s','um'],text=True)
    foreign=[]
    for line in raw.splitlines():
        fields=line.split()
        if len(fields)>=7 and fields[0].isdigit() and fields[1].isdigit() and int(fields[1]) not in allowed:
            if fields[2]=='C' and any(v.isdigit() and int(v)>0 for v in fields[3:7]):foreign.append(line)
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,name,utilization.gpu,memory.used,clocks.sm,temperature.gpu,power.draw',
        '--format=csv,noheader'],text=True).strip()
    processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_gpu_memory',
        '--format=csv,noheader'],text=True).strip()
    return dict(time=time.time(),gpu=gpu,processes=processes,pmon=raw,foreign_active=foreign)


def main():
    OUT.mkdir(parents=True,exist_ok=False)
    source=read(HERE/'reports/SOURCE_MANIFEST.json')
    assert all(digest(HERE/name)==value for name,value in source.items()),'Original candidate source changed'
    before_paths=list(read(HERE/'reports/PROTECTED_BEFORE.json'))
    scenes=read(Path('/home/prmi/zwj/experiments/compensator_distillation_replica_all_20261108/reports/SCENES.json'))
    slugs=['Replica__office1__mono','Replica__office2__rgbd','Replica__room0__rgbd','Replica__room2__mono']
    for scene in scenes:
        if scene['slug'] in slugs:before_paths.extend(str(p) for p in Path(scene['checkpoint']).rglob('*') if p.is_file())
    before={name:digest(Path(name)) for name in before_paths if Path(name).is_file()}
    (OUT/'PROTECTED_BEFORE.json').write_text(json.dumps(before,indent=2))
    binary=HERE/'build/sparse_rasterizer_v1.so'
    identity=dict(candidate_sha256=digest(binary),reference_sha256=digest(Path(
        '/home/prmi/zwj/experiments/global_view_joint_lod_20261105/evaluation/python/diff-gaussian-rasterization/diff_gaussian_rasterization/_C.cpython-311-x86_64-linux-gnu.so')),
        source_manifest=source,slugs=slugs,rounds=2,repeats_per_round=3,profile_round=1,
        timestamp=time.time(),algorithm_changed=False,automatic_dispatch_by_K=False)
    (OUT/'PROTOCOL.json').write_text(json.dumps(identity,indent=2))
    initial=telemetry({os.getpid()})
    assert not initial['foreign_active'],initial
    (OUT/'ENVIRONMENT_BEFORE.json').write_text(json.dumps(initial,indent=2))
    for round_id in [1,2]:
        command=[sys.executable,'-u',str(HERE/'benchmark.py'),'--output',str(OUT/f'round{round_id}')]
        if round_id==1:command.append('--profile')
        started=time.time()
        with (OUT/f'round{round_id}.log').open('x') as log:
            process=subprocess.Popen(command,cwd=HERE,stdout=log,stderr=subprocess.STDOUT)
            info=dict(pid=process.pid,started=started,command=command,
                start_ticks=Path(f'/proc/{process.pid}/stat').read_text().rpartition(')')[2].split()[19],
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
            (OUT/f'round{round_id}_PROCESS.json').write_text(json.dumps(info,indent=2))
            observations=[]
            while process.poll() is None:
                observations.append(telemetry({os.getpid(),process.pid}))
            code=process.wait()
        (OUT/f'round{round_id}_MONITOR.json').write_text(json.dumps(observations,indent=2))
        foreign=[v for v in observations if v['foreign_active']]
        result=dict(round=round_id,returncode=code,elapsed_s=time.time()-started,foreign_active_samples=len(foreign))
        (OUT/f'round{round_id}_RESULT.json').write_text(json.dumps(result,indent=2))
        print('IDLE_ROUND',json.dumps(result),flush=True)
        assert code==0,(round_id,code)
        assert not foreign,('External GPU work detected; preserve this run as interfered',round_id)
    changed=[name for name,value in before.items() if not Path(name).is_file() or digest(Path(name))!=value]
    assert digest(binary)==identity['candidate_sha256'],'Candidate binary changed during measurement'
    integrity=dict(passed=not changed,files=len(before),changed=changed,candidate_binary_unchanged=True,
        source_unchanged=all(digest(HERE/name)==value for name,value in source.items()))
    (OUT/'INTEGRITY.json').write_text(json.dumps(integrity,indent=2))
    assert integrity['passed'] and integrity['source_unchanged']
    (OUT/'COMPLETE.json').write_text(json.dumps(dict(passed=True,rounds=2,foreign_gpu_work_observed=False,
        protected_files=len(before),completed_at=time.time()),indent=2))
    print('IDLE_RETEST_COMPLETE',flush=True)


if __name__=='__main__':main()
