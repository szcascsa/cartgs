"""UID canonical v2 完成后顺序复核；评价不与在线训练并行。"""
from pathlib import Path
import json,os,subprocess,time

root=Path('/home/prmi/zwj/anchor-si-20261024')
python='/home/prmi/anaconda3/envs/zwj/bin/python'
checkpoint='/home/prmi/zwj/private-compensation-online-20261022/r3_independent14_seed2026/7465_shutdown/ply'
fixture='/home/prmi/zwj/anchor-selector-r3-performance-20261022/native1/Full_parity.pt'
commands=[('micro',[root/'source/bin/anchor_si_microbenchmark',checkpoint,fixture,root/'micro']),
          ('frozen',[python,root/'evaluate.py','frozen']),
          ('online',[python,root/'evaluate.py','online']),
          ('probe',[python,root/'selection_probe.py'])]
results=[]
try:
    for name,args in commands:
        print('RUN_FINAL',name,flush=True);start=time.time()
        env=dict(os.environ,OMP_NUM_THREADS='2')
        if name!='micro':env.pop('LD_LIBRARY_PATH',None)
        with (root/('final_'+name+'.log')).open('w') as log:
            result=subprocess.run(list(map(str,args)),cwd=root/'source',env=env,stdout=log,stderr=subprocess.STDOUT)
        results.append(dict(case=name,returncode=result.returncode,wall_s=time.time()-start))
        result.check_returncode()
    print('FINAL_EVALUATION_PASS',flush=True)
finally:
    (root/'final_evaluation_status.json').write_text(json.dumps(dict(revision='uid_canonical_v2',commands=results),indent=2))
