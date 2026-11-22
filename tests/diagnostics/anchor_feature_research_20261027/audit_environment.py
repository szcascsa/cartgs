"""记录当前真实生产版本与研究开始时间；认证信息不进入文件。"""
from datetime import datetime,timezone
import hashlib,json,shutil,subprocess
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parent;REPO=Path('/home/prmi/zwj/mycartgs');REF=Path('/home/prmi/zwj/neural-v1-20261027')
files=['cfg/selector/selector_config.yaml','improvements/selector/gumbel_network.h','improvements/selector/spatial_compensation.h',
    'improvements/selector/anchor_bank.h','improvements/selector/variable_anchor_blocks.h','src/gaussian_mapper.cpp',
    'src/gaussian_model.cpp','src/spatial_compensation_live.cu','python/selector.py','python/spatial_compensation.py']
records={}
for name in files:
    p=REPO/name;records[name]=hashlib.sha256(p.read_bytes()).hexdigest();target=ROOT/'production_snapshot'/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,target)
shutil.copy2(REPO/'python/spatial_compensation.py',ROOT/'production_spatial.py');shutil.copy2(REF/'scenes.json',ROOT/'scenes.json')
env=dict(started_utc='2026-10-27T08:00:00+00:00',budget_minutes=180,audit_utc=datetime.now(timezone.utc).isoformat(),torch=torch.__version__,cuda=torch.version.cuda,
    device=torch.cuda.get_device_name(),production_sha256=records,production_modified=False,
    gpu_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_memory','--format=csv,noheader'],text=True),
    constraint='Anchor abstract embeddings allowed; no Gaussian-private physical compensation',architecture='current production SpatialLocalXYZ and independent Selector11')
(ROOT/'environment.json').write_text(json.dumps(env,indent=2),encoding='utf-8');print(json.dumps(env),flush=True)
