"""锁定训练完成后汇总，不根据non-Bank调整实验。"""
from pathlib import Path
import subprocess
import sys
import time
root=Path(__file__).resolve().parent
try:
    while not (root/'run.exit').exists():time.sleep(5)
    assert (root/'run.exit').read_text().strip()=='0'
    with (root/'analyze.log').open('w') as stream:
        result=subprocess.run([sys.executable,'-u',str(root/'analyze.py')],cwd=root,stdin=subprocess.DEVNULL,
            stdout=stream,stderr=subprocess.STDOUT)
    assert result.returncode==0
except Exception:
    (root/'controller.exit').write_text('1');raise
(root/'controller.exit').write_text('0')
