from pathlib import Path
import difflib
import hashlib
import json
import subprocess
import re
D=Path(__file__).resolve().parent
ROOT=D.parents[2]
names=['CMakeLists.txt','src/change_distill.cpp','include/change_distill_transport_cuda.h','src/change_distill_transport.cu']
manifest=json.loads((D/'baseline/raw/source_manifest.json').read_text(encoding='utf-8'))
unchanged=[p for p,h in manifest.items() if p not in names and hashlib.sha256((ROOT/p).read_text(encoding='utf-8').encode()).hexdigest()!=h['normalized_sha256']]
assert not unchanged,unchanged
for cmd,name in [(['diff','--binary'],'final_working_tree.diff'),(['status','--porcelain=v1'],'final_status.txt')]:
    (D/name).write_bytes(subprocess.check_output(['git',*cmd],cwd=ROOT))
def blocks(data):
    return {part.split(b'\n',1)[0]:part for part in re.split(b'(?=diff --git )',data) if part}
before=blocks((D/'baseline/working_tree.diff').read_bytes())
after=blocks((D/'final_working_tree.diff').read_bytes())
unrelated=[k.decode('utf-8',errors='replace') for k in before.keys()|after.keys()
           if not any(b'a/'+p.encode()+b' ' in k for p in names) and before.get(k)!=after.get(k)]
assert not unrelated,unrelated
patch='';hashes={}
for name in names:
    old=D/'baseline/source'/name;new=ROOT/name
    a=old.read_text(encoding='utf-8').splitlines(True) if old.exists() else []
    b=new.read_text(encoding='utf-8').splitlines(True)
    patch+='diff --git a/'+name+' b/'+name+'\n'+('new file mode 100644\n' if not old.exists() else '')
    patch+=''.join(difflib.unified_diff(a,b,fromfile='a/'+name if old.exists() else '/dev/null',tofile='b/'+name))
    hashes[name]={'sha256':hashlib.sha256(new.read_bytes()).hexdigest(),
                  'normalized_sha256':hashlib.sha256(new.read_text(encoding='utf-8').encode()).hexdigest()}
(D/'production_transport_v2.patch').write_text(patch,encoding='utf-8')
(D/'final_source_manifest.json').write_text(json.dumps(hashes,indent=2),encoding='utf-8')
(D/'integrity.json').write_text(json.dumps({'unrelated_source_drift':unchanged,'unrelated_tracked_diff_drift':unrelated,'production_files_changed':names,'committed':False},indent=2))
print('Unrelated source drift:',unchanged)
