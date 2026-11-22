"""一个登记控制器顺序运行CPU边界、原生GPU交接与checkpoint恢复。"""
from pathlib import Path
import subprocess,sys
HERE=Path(__file__).resolve().parent
for script,name in [('handoff_edges.py','handoff_edges02'),('checkpoint_test.py','checkpoint01')]:
    print('VALIDATING',script,name,flush=True)
    subprocess.run([sys.executable,'-B','-u',str(HERE/script),'--output',name],cwd=HERE,check=True)
print('VALIDATION_SUITE_COMPLETE',flush=True)
