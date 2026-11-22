from pathlib import Path
import difflib
import hashlib
import json
import re
D=Path(__file__).resolve().parent
ROOT=D.parents[2]
names=['CMakeLists.txt','src/change_distill.cpp','include/change_distill_transport_cuda.h','src/change_distill_transport.cu']
manifest=json.loads((D/'baseline/raw/source_manifest.json').read_text())
patch='*** Begin Patch\n'
for name in names:
    p=ROOT/name
    new=(D/'final/source'/name).read_text()
    if p.exists():
        assert name in manifest
        assert hashlib.sha256(p.read_text().encode()).hexdigest()==manifest[name]['normalized_sha256'],name
        diff=list(difflib.unified_diff(p.read_text().splitlines(True),new.splitlines(True),n=3))
        body=re.sub(r'@@ -[0-9,]+ \+[0-9,]+ @@', '@@', ''.join(diff[2:]))
        patch+='*** Update File: '+str(p).replace('\\','/')+'\n'+body
    else:
        patch+='*** Add File: '+str(p).replace('\\','/')+'\n'+''.join('+'+line+'\n' for line in new.splitlines())
patch+='*** End Patch\n'
print(patch,end='')
