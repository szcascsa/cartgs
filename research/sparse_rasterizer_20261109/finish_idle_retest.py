"""首轮已完成，第二轮启动被Xorg瞬时负载拦截后的受控复测。"""
from run_idle_retest import *


def quiet_observation(allowed):
    value=telemetry(allowed)
    active=[]
    for line in value['pmon'].splitlines():
        fields=line.split()
        if len(fields)>=7 and fields[0].isdigit() and fields[1].isdigit() and int(fields[1]) not in allowed:
            if any(v.isdigit() and int(v)>0 for v in fields[3:7]):active.append(line)
    value['foreign_active']=active
    return value


def main():
    assert read(OUT/'round1_RESULT.json')['returncode']==0
    assert read(OUT/'round2_RESULT.json')['returncode']==1
    assert not list((OUT/'round2').glob('samples.csv')),'Failed start unexpectedly produced samples'
    initial=[]
    # 只等待外部负载自然结束，不修改或终止其他进程。
    for _ in range(15):
        initial.append(quiet_observation({os.getpid()}))
        if len(initial)>=3 and all(not v['foreign_active'] for v in initial[-3:]):break
    assert len(initial)>=3 and all(not v['foreign_active'] for v in initial[-3:]),'GPU remains active'
    target=OUT/'round2_retry01'
    command=[sys.executable,'-u',str(HERE/'benchmark.py'),'--output',str(target)]
    started=time.time()
    with (OUT/'round2_retry01.log').open('x') as log:
        process=subprocess.Popen(command,cwd=HERE,stdout=log,stderr=subprocess.STDOUT)
        observations=[]
        while process.poll() is None:observations.append(quiet_observation({os.getpid(),process.pid}))
        code=process.wait()
    (OUT/'round2_retry01_MONITOR.json').write_text(json.dumps(observations,indent=2))
    result=dict(returncode=code,elapsed_s=time.time()-started,foreign_active_samples=sum(bool(v['foreign_active']) for v in observations),
        discarded_start='round2: Xorg activity at initial idle guard; no timed samples',initial_observations=initial)
    (OUT/'round2_retry01_RESULT.json').write_text(json.dumps(result,indent=2))
    assert code==0 and result['foreign_active_samples']==0,result
    before=read(OUT/'PROTECTED_BEFORE.json');identity=read(OUT/'PROTOCOL.json')
    changed=[n for n,v in before.items() if not Path(n).is_file() or digest(Path(n))!=v]
    integrity=dict(passed=not changed,files=len(before),changed=changed,
        candidate_binary_unchanged=digest(HERE/'build/sparse_rasterizer_v1.so')==identity['candidate_sha256'],
        source_unchanged=all(digest(HERE/n)==v for n,v in identity['source_manifest'].items()))
    (OUT/'INTEGRITY.json').write_text(json.dumps(integrity,indent=2))
    assert integrity['passed'] and integrity['candidate_binary_unchanged'] and integrity['source_unchanged']
    (OUT/'COMPLETE.json').write_text(json.dumps(dict(passed=True,rounds=2,accepted_rounds=['round1','round2_retry01'],
        discarded_starts=['round2'],foreign_gpu_work_observed=False,protected_files=len(before),completed_at=time.time()),indent=2))
    print('IDLE_RETEST_COMPLETE',json.dumps(result),flush=True)


if __name__=='__main__':main()
