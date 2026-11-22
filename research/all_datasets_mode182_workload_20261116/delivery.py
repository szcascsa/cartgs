"""打包工作量数据、实现和验收证据；不包含模型及渲染图片。"""
from pathlib import Path
import hashlib
import json
import zipfile

ROOT = Path(__file__).resolve().parent


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    assert json.loads((ROOT/'reports/VALIDATION.json').read_text())['passed']
    paths = list(ROOT.glob('*.py'))+list(ROOT.glob('*.md'))
    paths += [ROOT/name for name in ['BUILD_INPUTS.json','BUILD_VALIDATION.json','CONTROLLER.json','QUEUE.json']]
    paths += [p for p in (ROOT/'reports').iterdir() if p.is_file()]
    paths += [p for p in (ROOT/'source').iterdir() if p.is_file()]
    paths += [p for p in (ROOT/'runs').rglob('*.json') if p.is_file()]
    (ROOT/'inputs').mkdir(exist_ok=True)
    frozen = ROOT.parent/'all_datasets_joint182_ratio_speedup_20261115/SCENES_FROZEN.json'
    (ROOT/'inputs/SCENES_FROZEN.json').write_bytes(frozen.read_bytes())
    paths.append(ROOT/'inputs/SCENES_FROZEN.json')
    hashes = {str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))}
    (ROOT/'DELIVERY_SHA256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    archive = ROOT/'MODE182_WORKLOAD_DATA.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=5) as bundle:
        for path in [*sorted(set(paths)), ROOT/'DELIVERY_SHA256.json']:
            bundle.write(path, path.relative_to(ROOT),
                compress_type=zipfile.ZIP_STORED if path.suffix=='.gz' else zipfile.ZIP_DEFLATED)
    result = dict(passed=True,files=len(hashes),archive=archive.name,archive_bytes=archive.stat().st_size,sha256=sha(archive))
    (ROOT/'DELIVERY_VERIFIED.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
