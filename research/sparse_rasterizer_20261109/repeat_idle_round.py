"""替换原始监控复核发现Xorg活动的轮次，保留全部原始现场。"""
from finish_idle_retest import *
NAME='round1_retry03'


def main():
    target=OUT/NAME
    assert not target.exists()
    initial=[]
    for _ in range(20):
        initial.append(quiet_observation({os.getpid()}))
        if len(initial)>=3 and all(not v['foreign_active'] for v in initial[-3:]):break
    assert len(initial)>=3 and all(not v['foreign_active'] for v in initial[-3:])
    command=[sys.executable,'-u',str(HERE/'benchmark.py'),'--output',str(target)]
    started=time.time()
    with (OUT/(NAME+'.log')).open('x') as log:
        process=subprocess.Popen(command,cwd=HERE,stdout=log,stderr=subprocess.STDOUT)
        observations=[]
        while process.poll() is None:observations.append(quiet_observation({os.getpid(),process.pid}))
        code=process.wait()
    result=dict(returncode=code,elapsed_s=time.time()-started,
        foreign_active_samples=sum(bool(v['foreign_active']) for v in observations),initial_observations=initial)
    (OUT/(NAME+'_MONITOR.json')).write_text(json.dumps(observations,indent=2))
    (OUT/(NAME+'_RESULT.json')).write_text(json.dumps(result,indent=2))
    assert code==0 and result['foreign_active_samples']==0,result
    old=read(OUT/'COMPLETE.json')
    (OUT/'COMPLETE_BEFORE_GRAPHICS_REVIEW.json').write_text(json.dumps(old,indent=2))
    old.update(accepted_rounds=[NAME,'round2_retry01'],discarded_starts=['round2'],
        discarded_measured_rounds=['round1','round1_retry01','round1_retry02'],graphics_review_corrected=True,
        profiling_separate_from_accepted_rounds=True,completed_at=time.time())
    (OUT/'COMPLETE.json').write_text(json.dumps(old,indent=2))
    print('REPLACEMENT_COMPLETE',json.dumps(result),flush=True)


if __name__=='__main__':main()
