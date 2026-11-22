"""核验受保护文件及候选身份后，只打包本研究产物。"""
from pathlib import Path
import hashlib,json,tarfile

HERE=Path(__file__).resolve().parent


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8<<20),b''):value.update(chunk)
    return value.hexdigest()


def main():
    before=json.loads((HERE/'reports/formal/PROTECTED_BEFORE.json').read_text())
    after={p:digest(p) if Path(p).is_file() else None for p in before}
    changed=[p for p,v in before.items() if after[p]!=v]
    frozen=json.loads((HERE/'reports/formal/SOURCE_MANIFEST.json').read_text())
    source_changed=[p for p,v in frozen.items() if digest(HERE/p)!=v]
    binary=json.loads((HERE/'reports/formal/BINARY.json').read_text())
    result=dict(passed=not changed and not source_changed and digest(binary['path'])==binary['sha256'],
        protected_files=len(before),changed=changed,source_changed=source_changed,binary_sha256=digest(binary['path']))
    assert result['passed'],result
    assert json.loads((HERE/'reports/ALL_RENDER_INPUTS_VALIDATION.json').read_text())['cases']==896
    assert json.loads((HERE/'FINAL_VERIFICATION.json').read_text())['full_gate_passed']
    (HERE/'reports/FINAL_PROTECTED_AFTER.json').write_text(json.dumps(after,indent=2))
    (HERE/'reports/FINAL_INTEGRITY.json').write_text(json.dumps(result,indent=2))
    files=set(p for folder in ['source','reports','logs'] for p in (HERE/folder).rglob('*') if p.is_file())
    files.update(HERE.glob('*.py'))
    files.update(HERE.glob('*.md'));files.update(HERE.glob('*.csv'));files.update(HERE.glob('*.png'))
    files.update([HERE/'PROTOCOL.json',HERE/'FINAL_VERIFICATION.json',Path(binary['path'])])
    files={p for p in files if '__pycache__' not in p.parts}
    manifest={str(p.relative_to(HERE)):digest(p) for p in sorted(files)}
    archive=HERE/'evaluation/FULL_SAFE_LOW_RATIO_CUDA_DELIVERY.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for p in sorted(files):tar.add(p,arcname=str(p.relative_to(HERE)),recursive=False)
    receipt=dict(archive=str(archive),sha256=digest(archive),bytes=archive.stat().st_size,
        integrity=result,files=manifest)
    (HERE/'evaluation/DELIVERY_RECEIPT.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps({k:v for k,v in receipt.items() if k!='files'},indent=2))


if __name__=='__main__':main()
