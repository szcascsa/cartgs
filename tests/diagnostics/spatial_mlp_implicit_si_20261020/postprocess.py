from pathlib import Path
import subprocess,sys
ROOT=Path(__file__).resolve().parent
for command in [['extract.py']]+[['evaluate_mlp.py',n]for n in ['fr1_room','fr1_xyz','fr2_xyz']]+[['summarize.py'],['independent_check.py'],['diagnostics.py']]:
    with open(ROOT/(command[0].removesuffix('.py')+('_'+command[1]if len(command)>1 else'')+'.log'),'w')as f:subprocess.run([sys.executable,*command],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
