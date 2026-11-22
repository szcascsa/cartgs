"""验证实际编译源码与本地最终清单；检查生产核心文件未被实验覆盖。"""
from pathlib import Path
import hashlib,json

root=Path('/home/prmi/zwj/anchor-si-20261024')
manifest=json.loads((root/'source_manifest.json').read_text())
result={}
for label,base,files in [('experiment',root/'source',manifest['files']),
    ('production',Path('/home/prmi/zwj/mycartgs'),json.loads((root/'production_baseline_manifest.json').read_text())['files'])]:
    missing=[];mismatch=[];raw_mismatch=[]
    for name,want in files.items():
        path=base/name
        if not path.exists():missing.append(name)
        else:
            data=path.read_bytes()
            if hashlib.sha256(data).hexdigest()!=want:raw_mismatch.append(name)
            compared=data.replace(b'\r\n',b'\n') if label=='production' else data
            if hashlib.sha256(compared).hexdigest()!=want:mismatch.append(name)
    result[label]=dict(files=len(files),missing=missing,mismatch=mismatch,raw_byte_mismatch=raw_mismatch)
(root/'source_verification.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
assert not result['experiment']['missing'] and not result['experiment']['mismatch']
assert not result['production']['missing']
# 服务器已有两个运行配置覆盖，保持原样；源码必须匹配基线。
assert not (set(result['production']['mismatch'])-{'cfg/selector/selector_config.yaml'})
