"""校验本地/服务器源码SHA，并安全提取、逐文件校验本轮证据包。"""
from pathlib import Path,PurePosixPath
import hashlib,json,subprocess,tarfile
D=Path(__file__).resolve().parent;ROOT=D.parents[2];manifest=json.loads((D/'final_source_manifest.json').read_text(encoding='utf-8'))
assert subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()==manifest['head']
assert not subprocess.check_output(['git','diff','--cached','--name-only'],cwd=ROOT,text=True).strip()
for name,expected in manifest['files'].items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
remote=json.loads((D/'evidence_final/source_verification.json').read_text());assert all(x['matched'] for x in remote)
expected={x['path']:x for x in json.loads((D/'evidence_final/evidence_archive_manifest.json').read_text())};archive=D/'phase2_final_evidence.tar.gz';dest=(D/'evidence_archive').resolve();dest.mkdir(exist_ok=True);seen=set()
with tarfile.open(archive,'r|gz') as tf:
    for member in tf:
        p=PurePosixPath(member.name);assert member.isfile() and not p.is_absolute() and '..' not in p.parts and member.name in expected
        target=(dest/member.name).resolve();assert target.is_relative_to(dest);assert member.name not in seen;target.parent.mkdir(parents=True,exist_ok=True)
        digest=hashlib.sha256();count=0
        with tf.extractfile(member) as src,target.open('wb') as out:
            while data:=src.read(1024*1024):out.write(data);digest.update(data);count+=len(data)
        assert count==expected[member.name]['bytes'] and digest.hexdigest()==expected[member.name]['sha256'],member.name;seen.add(member.name)
assert seen==set(expected)
checks=dict(source_files=len(manifest['files']),source_matches=True,remote_source_matches=True,head=manifest['head'],staged_files=0,archive_files=len(seen),archive_sha256=hashlib.file_digest(archive.open('rb'),'sha256').hexdigest(),evidence_files_verified=True)
(D/'delivery_checks.json').write_text(json.dumps(checks,indent=2),encoding='utf-8');print('DELIVERY_PASS',checks,flush=True)
