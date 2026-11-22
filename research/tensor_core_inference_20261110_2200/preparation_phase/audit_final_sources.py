from pathlib import Path
import json,difflib,hashlib

HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
patch=[];metadata=[]
for path in sorted((HERE/'source').iterdir()):
    if not path.is_file():continue
    original=ROOT/'source'/path.name
    old=original.read_text().splitlines(keepends=True) if original.is_file() else []
    new=path.read_text().splitlines(keepends=True)
    if old==new:continue
    metadata.append(dict(file=path.name,previous_C_sha256=sha(original) if original.is_file() else None,final_sha256=sha(path)))
    patch.extend(difflib.unified_diff(old,new,fromfile='a/'+path.name if old else '/dev/null',tofile='b/'+path.name))
target=HERE/'reports/COMBINED_PREPARATION.patch';target.write_text(''.join(patch),encoding='utf-8')
(HERE/'reports/SOURCE_PATCH_METADATA.json').write_text(json.dumps(dict(base=str(ROOT/'source'),changes=metadata,patch_sha256=sha(target)),ensure_ascii=False,indent=2),encoding='utf-8')
print('FINAL_PREPARATION_PATCH',len(metadata),len(patch),flush=True)
