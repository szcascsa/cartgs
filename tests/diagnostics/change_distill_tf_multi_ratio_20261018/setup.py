"""冻结 Transport V2 正式基线，复用真实事件 harness，独立保存本轮结果。"""
from pathlib import Path
import hashlib,json,shutil,subprocess
D=Path(__file__).resolve().parent
ROOT=D.parents[2]
OLD=D.parent/'change_distill_tf_cuda_optimization_20261018'
out=D/'baseline';assert not out.exists()
manifest=json.loads((OLD/'baseline/raw/source_manifest.json').read_text(encoding='utf-8'))
for p in ['include/change_distill_transport_cuda.h','src/change_distill_transport.cu']:
    manifest[p]={}
hashes={}
for name in manifest:
    data=(ROOT/name).read_bytes();p=out/'source'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    hashes[name]={'sha256':hashlib.sha256(data).hexdigest(),'normalized_sha256':hashlib.sha256(data.decode('utf-8').replace('\r\n','\n').encode()).hexdigest()}
(out/'raw').mkdir()
(out/'raw/source_manifest.json').write_text(json.dumps(hashes,indent=2))
for name in ['benchmark.cpp','perf_scope.h']:shutil.copy2(OLD/'baseline'/name,out/name)
for command,name in [(['diff','--binary'],'initial.diff'),(['status','--porcelain=v1'],'initial_status.txt'),(['rev-parse','HEAD'],'head.txt')]:
    (D/name).write_bytes(subprocess.check_output(['git',*command],cwd=ROOT))
for name in ['run.py','paired.py','compare.py','lifecycle.py']:
    shutil.copy2(OLD/name,D/name)
print(out)
