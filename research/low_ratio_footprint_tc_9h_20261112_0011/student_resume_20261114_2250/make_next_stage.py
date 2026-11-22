"""仅在前序控制器确认结束后生成串行研究与验收计划，不启动并行GPU任务。"""
from pathlib import Path
import json
import subprocess

HERE=Path(__file__).resolve().parent


def main():
    previous=json.loads((HERE/'reports/JOINT_FINAL02_CONTROLLER.json').read_text())
    assert previous['state']=='complete' or (previous['state']=='failed' and '持续资源干扰' in previous.get('error',''))
    py='/home/prmi/anaconda3/envs/zwj/bin/python'
    for script in ['implement_overflow_prefix.py','implement_chunk_loop.py','implement_fast_publish.py']:
        subprocess.run([py,'-B',script],cwd=HERE,check=True)
    jobs=[]
    for stage,runtime,script in [('overflow_prefix','overflow_prefix_runtime.py','overflow_prefix_development.py'),
            ('chunk_loop','chunk_loop_runtime.py','chunk_loop_development.py'),('fast_publish','fast_publish_runtime.py','fast_publish_development.py')]:
        jobs.append(dict(name='student_'+stage+'_build',measure=False,command=['{python}','-u','-B',runtime]))
        for r in [1,2]:
            command=['{python}','-u','-B',script,'--output','{job}','--round',str(r)]
            if stage!='fast_publish':command+=['--group','0']
            jobs.append(dict(name=f'student_{stage}_r{r}',measure=True,command=command))
    # 不重跑已认可样本；新名字保留原四次受干扰尝试，避免覆盖。
    original=json.loads((HERE/'reports/JOINT_FINAL02_PLAN.json').read_text())
    for task in original['jobs']:
        if task['name'].startswith('student_joint_final_g') and not any(name.startswith(task['name']+'_a') for name in previous['accepted']):
            task=dict(task);task['name']+='_resumed';jobs.append(task)
    jobs.append(dict(name='student_thin_edges',measure=False,command=['{python}','-u','-B','thin_dispatch_edges.py','--output','{job}','--modes','182']))
    for script,label,groups in [('thin_dispatch_formal.py','thin_formal',9),('thin_dispatch_nested.py','thin_nested',16)]:
        for r in [1,2]:
            for g in (range(groups) if r==1 else reversed(range(groups))):
                jobs.append(dict(name=f'student_{label}_g{g}_r{r}',measure=True,command=['{python}','-u','-B',script,
                    '--output','{job}','--group',str(g),'--round',str(r),'--reuse-quality']))
    jobs.append(dict(name='student_joint_edges',measure=False,command=['{python}','-u','-B','joint_final_edges.py','--output','{job}']))
    jobs.append(dict(name='student_execution_identities',measure=False,command=['{python}','-u','-B','verify_execution_identities.py','--output','{job}']))
    for script,label in [('render_edges.py','render'),('joint_final_edges.py','joint')]:
        for tool in ['memcheck','initcheck','synccheck']:
            command=['/usr/local/cuda-11.8/bin/compute-sanitizer','--tool',tool,'--error-exitcode','99','--print-limit','10',
                '{python}','-u','-B',script,'--output','{job}','--sanitizer']
            if label=='render':command+=['--modes','182']
            jobs.append(dict(name=f'student_final_{label}_{tool}',measure=False,command=command))
    for script,label,args in [('render_final_profile.py','final_raster',[]),('render_full_profile.py','migration_raster',[]),
            ('joint_final_profile.py','final_joint_dev',['--group','0']),('joint_final_profile.py','final_joint_migration',['--group','1'])]:
        jobs.append(dict(name='student_'+label+'_profile',measure=False,command=['/usr/local/cuda-11.8/bin/nsys','profile',
            '--trace=cuda,nvtx','--cuda-graph-trace=node','--capture-range=cudaProfilerApi','--capture-range-end=stop',
            '--sample=none','--force-overwrite=false','--output','{root}/reports/{job}','{python}','-u','-B',script,*args]))
    p=HERE/'reports/RESEARCH_ACCEPTANCE_PLAN.json';assert not p.exists();p.write_text(json.dumps(dict(name='RESEARCH_ACCEPTANCE',jobs=jobs),indent=2))
    log=(HERE/'logs/research_acceptance_controller.log').open('x')
    child=subprocess.Popen([py,'-u','-B','run_stage.py',str(p)],cwd=HERE,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print('RESEARCH_ACCEPTANCE_LAUNCHED',child.pid,len(jobs),flush=True)


if __name__=='__main__':main()
