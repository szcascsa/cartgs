"""训练门控成功后执行一次既定三臂评价与证据收集。"""
import json
from pathlib import Path
import subprocess
import sys
import time
root=Path('/home/prmi/zwj/serial-joint-lod-20261021')
status=root/'validation/status.json'
while json.loads(status.read_text())['stage']!='TRAINED':
    process=Path('/proc/833926/stat')
    if not process.exists() or process.read_text().split()[2]=='Z':
        raise RuntimeError('Training pipeline ended before TRAINED; inspect pipeline.log')
    time.sleep(5)
for script,log in [('evaluate.py','evaluation.log'),('collect.py','collection.log')]:
    with (root/log).open('w') as output:
        subprocess.run([sys.executable,str(root/script)],stdout=output,stderr=subprocess.STDOUT,check=True)
print('EVALUATED_AND_COLLECTED',flush=True)
