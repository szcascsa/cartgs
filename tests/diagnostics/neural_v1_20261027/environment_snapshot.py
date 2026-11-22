"""只读版本与GPU快照，不把本地Git冒充服务器构建。"""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import torch

root=Path(__file__).resolve().parent;repo=Path('/home/prmi/zwj/mycartgs')
files=['cfg/selector/selector_config.yaml','python/selector.py','improvements/selector/gumbel_network.h','improvements/selector/gaussian_compensation_bank.h','src/gaussian_model.cpp']
renderer=repo/'python/diff-gaussian-rasterization/diff_gaussian_rasterization/_C.cpython-311-x86_64-linux-gnu.so'
value=dict(utc=datetime.now(timezone.utc).isoformat(),python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,
    device=torch.cuda.get_device_name(),source_sha256={f:hashlib.sha256((repo/f).read_bytes()).hexdigest() for f in files},
    renderer_sha256=hashlib.sha256(renderer.read_bytes()).hexdigest(),
    gpu_processes=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],capture_output=True,text=True).stdout,
    repo_git='server has no usable .git; actual source hashes recorded',production_changed=False)
path=root/'environment.json'
if path.exists():
    before=json.loads(path.read_text());assert before['source_sha256']==value['source_sha256'] and before['renderer_sha256']==value['renderer_sha256']
    path=root/'environment_final.json'
path.write_text(json.dumps(value,indent=2),encoding='utf-8');print('ENVIRONMENT_SHA_RECORDED',str(path),flush=True)
