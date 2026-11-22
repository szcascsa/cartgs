import subprocess,sys,json
from pathlib import Path
r=Path('/home/prmi/zwj/experiments/joint_view_inference_20261106')
s=Path('/home/prmi/zwj/experiments/joint_view_inference_20261106/source/research/joint_view_inference_20261106')
commands=[['validate_purity.py'], ['validate_state.py'], ['validate_layers.py'], ['benchmark_viewer.py'], ['benchmark.py', '--round', '1'], ['benchmark.py', '--round', '2']]
for command in commands:
 with (r/"logs"/("review_"+"_".join(command)+".log")).open("w") as f:
  result=subprocess.run([sys.executable,"-u",str(s/command[0])]+command[1:],cwd=r,stdout=f,stderr=subprocess.STDOUT)
 assert result.returncode==0,command
(r/"reports/REVIEW_COMPLETE.json").write_text(json.dumps(dict(passed=True,commands=commands)))
