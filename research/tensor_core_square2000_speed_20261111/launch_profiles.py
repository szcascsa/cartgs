"""正式计时后顺序采集移动视角Nsight；不把Profiler时间用作速度收益。"""
from pathlib import Path
import subprocess,os,json
HERE=Path(__file__).resolve().parent
OUT=HERE/'reports/profiles';OUT.mkdir(exist_ok=False)
cases=[('office1_10','Replica__office1__mono',.1),('office2_10','Replica__office2__rgbd',.1),
    ('room0_10','Replica__room0__rgbd',.1),('room2_10','Replica__room2__mono',.1),('room0_70','Replica__room0__rgbd',.7)]
env=os.environ.copy();env.update(PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
rows=[]
for name,slug,ratio in cases:
    stem=OUT/name
    cmd=['/usr/local/cuda/bin/nsys','profile','--force-overwrite=true','--trace=cuda,nvtx',
        '--sample=none','--cpuctxsw=none','--cuda-graph-trace=node',
        '--output='+str(stem),'/home/prmi/anaconda3/envs/zwj/bin/python','-B','-u',str(HERE/'trace_networks.py'),
        '--slug',slug,'--ratio',str(ratio)]
    with (HERE/'logs'/('profile_'+name+'.log')).open('x') as f:result=subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT)
    assert result.returncode==0,(name,result.returncode)
    with (HERE/'logs'/('export_'+name+'.log')).open('x') as f:
        result=subprocess.run(['/usr/local/cuda/bin/nsys','export','--type=sqlite','--output='+str(stem)+'.sqlite',str(stem)+'.nsys-rep'],stdout=f,stderr=subprocess.STDOUT)
    assert result.returncode==0,(name,'export',result.returncode)
    rows.append(dict(name=name,scene=slug,ratio=ratio,frames=36,sqlite=str(stem)+'.sqlite'))
    print('PROFILE_COMPLETE',name,flush=True)
(OUT/'MANIFEST.json').write_text(json.dumps(rows,indent=2))
