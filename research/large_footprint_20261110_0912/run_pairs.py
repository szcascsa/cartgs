"""同一控制器顺序执行两轮，每场景一次独立GPU环境验收；仅外部干扰重测。"""
from pathlib import Path
import json,subprocess,sys,time

HERE=Path(__file__).resolve().parent
scenes=json.loads((HERE/'reports/SCENES.json').read_text())
path=HERE/'reports/PAIRED_RUNS.json';assert not path.exists()
state=dict(started_at=time.time(),unit='scene_round',status='running',runs=[],attempts=[],
    rule='先锁定每场景/每轮为原子单位。只有退出成功且整段监测无外部活动才接受；不参考性能值筛选。')
def save():path.write_text(json.dumps(state,ensure_ascii=False,indent=2))
save()
for round_index in [1,2]:
    for scene in scenes if round_index==1 else list(reversed(scenes)):
        accepted=False
        for attempt in range(1,5):
            name=f'pair_r{round_index}_{scene["slug"]}_a{attempt}'
            identity=HERE/'reports/jobs'/(name+'.json');assert not identity.exists()
            command=[sys.executable,'-u',str(HERE/'scene_benchmark.py'),'--scene',scene['slug'],
                '--output',name,'--round',str(round_index),'--repeats','6']
            identity.write_text(json.dumps(dict(name=name,command=command,measured=True,state='launching'),indent=2))
            print('SCENE_PAIR_START',round_index,scene['id'],attempt,flush=True)
            with (HERE/'logs'/('runner_'+name+'.log')).open('x') as log:
                result=subprocess.run([sys.executable,'-u',str(HERE/'job_runner.py'),str(identity)],cwd=HERE,stdout=log,stderr=subprocess.STDOUT)
            job=json.loads(identity.read_text());state['attempts'].append(name);save()
            if result.returncode==0 and job.get('measurement_valid') and job['state']=='complete':
                state['runs'].append(name);save();accepted=True
                print('SCENE_PAIR_ACCEPTED',round_index,scene['id'],len(state['runs']),flush=True);break
            if not job.get('foreign_active_samples'):
                state.update(status='failed',failed_job=name);save();raise RuntimeError('Non-interference failure: '+name)
            print('SCENE_PAIR_INTERFERENCE_RETRY',name,flush=True)
        if not accepted:
            state.update(status='blocked_by_external_load',failed_scene=scene['id']);save();raise RuntimeError('Repeated external GPU activity')
state.update(status='complete',ended_at=time.time());save();assert len(state['runs'])==48
print('ALL_SCENE_PAIRS_COMPLETE',48,flush=True)
