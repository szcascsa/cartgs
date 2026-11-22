from pathlib import Path
import hashlib
import json
import platform
import subprocess


ROOT=Path(__file__).resolve().parents[3]


def checksum(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(8*1024*1024),b''):value.update(data)
    return value.hexdigest()


def main():
    result={}
    for prefix in ['PRODUCTION','INPUTS','LEGACY_INPUTS','LARGE_INPUTS']:
        before=json.loads((ROOT/'reports'/f'{prefix}_BEFORE.json').read_text())
        after={}
        changed=[]
        for name,old in before.items():
            p=Path(name)
            if p.exists():
                digest=checksum(p)
                after[name]=dict(bytes=p.stat().st_size,sha256=digest) if isinstance(old,dict) else digest
            else:after[name]=None
            if after[name]!=old:changed.append(name)
        (ROOT/'reports'/f'{prefix}_AFTER.json').write_text(json.dumps(after,indent=2))
        result[prefix]=dict(files=len(before),changed=changed,passed=not changed)
    raster=Path('/home/prmi/zwj/experiments/global_view_joint_lod_20261105/evaluation/python/diff-gaussian-rasterization/diff_gaussian_rasterization/_C.cpython-311-x86_64-linux-gnu.so')
    expected='27b5b468cf385dd1c96aeabdd65426da38a938a38d52d8181b1b1da485090c86'
    result['RASTERIZER']=dict(path=str(raster),sha256=checksum(raster),passed=checksum(raster)==expected)
    source={str(p.relative_to(ROOT/'source')):checksum(p) for p in (ROOT/'source/python').glob('deployment*') if p.is_file()}
    frozen=ROOT/'source/python/frozen_inference_kernels.cu'
    source[str(frozen.relative_to(ROOT/'source'))]=checksum(frozen)
    binary=ROOT/'build/mycartgs_deployment_fp32_v30/mycartgs_deployment_fp32_v30.so'
    identity=dict(source=source,binary=dict(path=str(binary),sha256=checksum(binary)),kernel=platform.release())
    (ROOT/'reports/BINARY_IDENTITY.json').write_text(json.dumps(identity,indent=2))
    result['passed']=all(v['passed'] for v in result.values())
    (ROOT/'reports/FINAL_INTEGRITY.json').write_text(json.dumps(result,indent=2))
    assert result['passed'],result
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
