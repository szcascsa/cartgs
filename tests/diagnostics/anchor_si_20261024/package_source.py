"""按源码清单打包最终实验版本；不打包模型、运行证据或凭据。"""
from pathlib import Path
import hashlib,json,zipfile

directory=Path(__file__).resolve().parent
repo=directory.parents[2]
previous=json.loads((directory/'source_manifest.json').read_text(encoding='utf-8'))
names=set(previous['files'])
names.update(['include/anchor_si.h','src/anchor_si.cpp','src/anchor_si.cu',
              'cuda_rasterizer/anchor_si_capture.h','tests/anchor_selector/anchor_si_test.cpp',
              'cfg/selector/anchor_si_l2.yaml'])
names.update(p.relative_to(repo).as_posix() for p in directory.iterdir() if p.suffix in {'.py','.cpp'} or p.name=='protocol.json')
files={}
with zipfile.ZipFile(directory/'source.zip','w',zipfile.ZIP_DEFLATED) as archive:
    for name in sorted(names):
        data=(repo/name).read_bytes();files[name]=hashlib.sha256(data).hexdigest();archive.writestr(name,data)
manifest=dict(base_commit='27b9439',branch='anchor-si',implementation_revision='uid_canonical_v2',files=files)
(directory/'source_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('SOURCE_PACKAGE',len(files))
