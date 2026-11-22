"""固定阶段与输入；任何gate失败停止，不用质量选择参数。"""
from pathlib import Path
import subprocess,sys,json,traceback,time,fcntl
ROOT=Path(__file__).resolve().parent
def save(v):(ROOT/'status.json').write_text(json.dumps(v,indent=2)+'\n')
lock=open(ROOT/'controller.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
try:
    stages=['prepare','build_and_export','rankings','replay_f3','quality','analyze']
    for stage in stages[stages.index(sys.argv[1])if len(sys.argv)>1 else 0:]:
        save(dict(stage=stage,state='running'))
        with open(ROOT/(stage+'.log'),'w')as log:subprocess.run([sys.executable,str(ROOT/(stage+'.py'))],stdout=log,stderr=subprocess.STDOUT,check=True)
        print('PASS',stage,flush=True)
    save(dict(state='complete',all_passed=True))
except BaseException:
    save(dict(state='failed',failure=traceback.format_exc()));raise
