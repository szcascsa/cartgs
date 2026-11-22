"""更新用户指定的生产源码目录；备份差异，保留已有配置/依赖/产物。"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import tarfile

parser = argparse.ArgumentParser()
parser.add_argument('archive', type=Path)
parser.add_argument('--apply', action='store_true')
args = parser.parse_args()
root = Path('/home/prmi/zwj/mycartgs').resolve(strict=True)


def digest(data):
    return hashlib.sha256(data).hexdigest()


updates = {}
skipped_configs = []
with tarfile.open(args.archive) as archive:
    for member in archive:
        if not member.isfile():
            raise ValueError('Only regular source files are accepted')
        relative = Path(member.name)
        target = (root/relative).resolve()
        assert not relative.is_absolute() and target.is_relative_to(root)
        assert relative.parts[0] not in ['third_party', 'results', 'bin', 'lib', '.git']
        data = archive.extractfile(member).read()
        if relative.parts[0] == 'cfg' and target.exists():
            skipped_configs.append(str(relative))
            continue
        if not target.exists() or target.read_bytes().replace(b'\r\n', b'\n') != data:
            updates[str(relative)] = data

# Every existing user Selector YAML gets only this new switch. All other
# configuration bytes, including TF backend and local ablation values, remain.
selector_configs = sorted((root/'cfg/selector').glob('*.yaml'))
for config in selector_configs:
    name = str(config.relative_to(root))
    data = updates.get(name, config.read_bytes())
    text = data.decode('utf-8')
    if re.search(r'^Selector\.backend:', text, re.M):
        continue
    newline = '\r\n' if '\r\n' in text else '\n'
    insertion = '# legacy | cuda；cuda 需编译启用 CARTGS_EXPERIMENTAL_SELECTOR_CUDA。'+newline+'Selector.backend: "legacy"'+newline
    text, count = re.subn(r'(^Selector\.enabled:[^\r\n]*\r?\n)', lambda match: match[1]+insertion, text, count=1, flags=re.M)
    assert count == 1, name+' has no Selector.enabled'
    updates[name] = text.encode('utf-8')

plan = dict(target=str(root), changed_files=sorted(updates), changed_count=len(updates),
            preserved_existing_configs=skipped_configs, compiled=False)
(args.archive.parent/'plan.json').write_text(json.dumps(plan, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(plan, ensure_ascii=False, indent=2))
if not args.apply:
    raise SystemExit(0)

stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
backup = root.parent/('mycartgs-source-backup-'+stamp)
backup.mkdir()
record = dict(plan, backup=str(backup), files={})
with tarfile.open(backup/'previous_source.tar.gz', 'w:gz') as archive:
    for name in sorted(updates):
        target = root/name
        old = target.read_bytes() if target.exists() else None
        if old is not None:
            archive.add(target, arcname=name, recursive=False)
        record['files'][name] = dict(existed=old is not None, before=digest(old) if old is not None else None,
                                    after=digest(updates[name]))
shutil.copy2(args.archive.parent/'manifest.json', backup/'source_manifest.json')
(backup/'update_record.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
for name, data in updates.items():
    target = root/name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    if target.suffix == '.sh':
        target.chmod(target.stat().st_mode | 0o111)
    assert digest(target.read_bytes()) == record['files'][name]['after']
(root/'SOURCE_UPDATE_MANIFEST.json').write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
print('UPDATED', len(updates), 'BACKUP', backup)
