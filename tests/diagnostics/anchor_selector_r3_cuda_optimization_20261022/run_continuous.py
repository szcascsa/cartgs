"""同一真实地图、单一固定相机、100轮Full→轮转LoD，拓扑冻结；真实在线另报。"""
import json
import os
from pathlib import Path
import subprocess
import time
R=Path('/home/prmi/zwj/anchor-selector-r3-cuda-optimization-20261022');S=R/'source'
env=dict(os.environ,OMP_NUM_THREADS='2',LD_LIBRARY_PATH=str(S/'lib')+':'+os.environ.get('LD_LIBRARY_PATH',''))
command=[S/'bin/controlled_mapping_replay','/home/prmi/zwj/private-compensation-online-20261022/r3_independent14_seed2026/7465_shutdown/ply',
    '/home/prmi/datasets/TUM_RGBD/rgbd_dataset_freiburg1_xyz']
results=[]
for name,bits in [('continuous_legacy_a',0),('continuous_fused_a',7),('continuous_legacy_b',0),('continuous_fused_b',7)]:
    start=time.monotonic()
    with (R/(name+'.log')).open('w') as f:
        subprocess.run(list(map(str,command+[R/name,100,bits])),cwd=S,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    results.append(dict(run=name,bits=bits,wall_s=time.monotonic()-start))
(R/'continuous_runs.json').write_text(json.dumps(results,indent=2));print('PASS continuous replays',flush=True)
