"""单次启动，后台执行，避免会话中断丢失正式测试。"""
from pathlib import Path
import subprocess,sys,json
root=Path(__file__).resolve().parent
with open(root/'controller.log','x')as log:
    process=subprocess.Popen([sys.executable,str(root/'run_quality.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=root)
(root/'process.json').write_text(json.dumps(dict(pid=process.pid))+'\n')
print(json.dumps(dict(pid=process.pid)))
