"""本轮保护文件与复用二进制的前后完整性核验。"""
from pathlib import Path
import hashlib
import json


ROOT = Path(__file__).resolve().parents[3]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    result = {}
    for prefix in ['PRODUCTION','INPUTS','LEGACY_INPUTS','LARGE_INPUTS','SCENE_INPUTS','REUSED_INPUTS']:
        before = json.loads((ROOT/'reports'/f'{prefix}_BEFORE.json').read_text())
        after = {}
        changed = []
        for name,value in before.items():
            path = Path(name)
            digest = sha(path) if path.exists() else None
            after[name] = dict(bytes=path.stat().st_size,sha256=digest) if isinstance(value,dict) and digest else digest
            if after[name] != value:
                changed.append(name)
        (ROOT/'reports'/f'{prefix}_AFTER.json').write_text(json.dumps(after,indent=2))
        result[prefix] = dict(files=len(before),passed=not changed,changed=changed)
    binaries = {
        '/home/prmi/zwj/experiments/compensator_distillation_20261108/build/mycartgs_deployment_fp32_v31/mycartgs_deployment_fp32_v31.so':
            'fa7b855685ecd4f75ce7118ee07fadd27640af08099a6775ffee5fe87fa7d601',
        '/home/prmi/zwj/experiments/global_view_joint_lod_20261105/evaluation/python/diff-gaussian-rasterization/diff_gaussian_rasterization/_C.cpython-311-x86_64-linux-gnu.so':
            '27b5b468cf385dd1c96aeabdd65426da38a938a38d52d8181b1b1da485090c86'}
    result['BINARIES'] = dict(passed=all(sha(p)==value for p,value in binaries.items()),
        files={p:sha(p) for p in binaries})
    before = json.loads((ROOT/'reports/SOURCE_BEFORE.json').read_text())
    changed = [name for name,value in before.items() if name.startswith('python/') and sha(ROOT/'source'/name)!=value]
    result['ALGORITHM_SOURCE'] = dict(passed=not changed,changed=changed)
    figures = ['REPLICA_PSNR_MONO.png','REPLICA_PSNR_RGBD.png','REPLICA_GAP_MONO.png','REPLICA_GAP_RGBD.png']
    result['saved_render_images'] = len([p for folder in ['models','reports'] for p in (ROOT/folder).rglob('*')
        if p.suffix.lower() in ['.png','.jpg','.jpeg'] and p.name not in figures])
    result['passed'] = all(v['passed'] for v in result.values() if isinstance(v,dict)) and result['saved_render_images']==0
    (ROOT/'reports/FINAL_INTEGRITY.json').write_text(json.dumps(result,indent=2))
    source = {str(p.relative_to(ROOT/'source')):sha(p) for p in (ROOT/'source').rglob('*')
        if p.is_file() and '__pycache__' not in str(p)}
    (ROOT/'reports/SOURCE_MANIFEST.json').write_text(json.dumps(source,indent=2))
    assert result['passed'],result
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
