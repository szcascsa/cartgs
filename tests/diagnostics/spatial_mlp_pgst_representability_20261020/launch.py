from pathlib import Path
import subprocess,sys,json
root=Path(__file__).resolve().parent
with open(root/'controller.log','x')as log:
    p=subprocess.Popen([sys.executable,str(root/'fit.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True,cwd=root)
(root/'process.json').write_text(json.dumps(dict(pid=p.pid))+'\n');print(p.pid)
