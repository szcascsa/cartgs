"""等待当前隔离构建结束，随后按最终源码 SHA 执行功能门控与一次训练。"""
from pathlib import Path
import runpy
import time
root=Path('/home/prmi/zwj/serial-full-lod-20261021')
old=Path('/proc/804954/stat')
while old.exists():
    if old.read_text().split()[2]=='Z':break
    time.sleep(2)
runpy.run_path(str(root/'run_experiment.py'),run_name='__main__')
