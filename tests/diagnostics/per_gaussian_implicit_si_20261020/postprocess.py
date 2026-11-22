"""训练退出后顺序执行固定ranking导出、四指标评价与最终检查。"""
from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parent
assert (ROOT/'training_complete.json').exists()
for cmd in [['extract.py']]+[[a,s]for s in ['fr1_room','fr1_xyz','fr2_xyz']for a in ['evaluate_pg.py','evaluate_sampling.py']]+[['summarize.py']]:
    name=cmd[0].removesuffix('.py')+('_'+cmd[1]if len(cmd)>1 else'')
    with open(ROOT/(name+'.log'),'w')as f:subprocess.run([sys.executable,*cmd],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
