"""记录完整结果的研究判断，生成报告并打包；不执行任何后续训练。"""
from pathlib import Path
import json,subprocess,sys
root=Path(__file__).resolve().parent
path=root/'audit_summary.json';summary=json.loads(path.read_text())
assert summary['correctness']['all_passed']
summary['decision']='SPATIAL SHARING NOT SUPPORTED UNDER THE FROZEN PROTOCOL'
summary['next_step']='HOLD: keep Current C; do not enter grid/split or change training automatically'
summary['interpretation']='Gradient fix recovers most legacy MLP degradation, but does not establish improvement over Raw/PG-ST. Bank results also mostly fail to improve; optimization/objective/spatial parameterization are not isolated.'
path.write_text(json.dumps(summary,indent=2)+'\n')
for script in ['report.py','pack.py']:
    subprocess.run([sys.executable,str(root/script)],check=True)
