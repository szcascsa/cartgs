from pathlib import Path
import json

HERE=Path(__file__).resolve().parent
jobs=[]
for iteration in [1,2]:
    for group in ([0,1] if iteration==1 else [1,0]):
        jobs.append(dict(name=f'native_dev_r{iteration}_g{group}',measure=True,
            command=['{python}','-u','-B','joint_benchmark.py','--output','{job}','--round',str(iteration),
                '--group',str(group),'--pending','--native','--tune','--quick']))
jobs.append(dict(name='joint_profile01',command=['/usr/local/cuda-11.8/bin/nsys','profile','--trace=cuda,nvtx',
    '--cuda-graph-trace=node','--capture-range=cudaProfilerApi','--capture-range-end=stop','--sample=none',
    '--force-overwrite=false','--output','{root}/reports/{job}','{python}','-u','-B','joint_profile.py']))
path=HERE/'reports/NATIVE_DEVELOPMENT_PLAN.json';assert not path.exists()
path.write_text(json.dumps(dict(name='NATIVE_DEVELOPMENT',jobs=jobs),indent=2))
print('NATIVE_DEVELOPMENT_PLAN_READY')
