"""核验本轮保护范围与原Rasterizer身份，不修改被核验文件。"""
from pathlib import Path
import hashlib
import json
import subprocess
import time

ROOT=Path(__file__).resolve().parents[3]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert json.loads((ROOT/'reports/ACCEPTANCE_COMPLETE.json').read_text())['passed']
    groups={}
    for name in ['PRODUCTION','INPUTS','LEGACY_INPUTS','LARGE_INPUTS']:
        before=json.loads((ROOT/'reports'/f'{name}_BEFORE.json').read_text());after={};changed=[];missing=[]
        for filename,value in before.items():
            path=Path(filename)
            if not path.is_file():missing.append(filename);continue
            digest=sha(path);after[filename]=dict(bytes=path.stat().st_size,sha256=digest) if isinstance(value,dict) else digest
            if digest!=(value['sha256'] if isinstance(value,dict) else value):changed.append(filename)
        (ROOT/'reports'/f'{name}_AFTER.json').write_text(json.dumps(after,indent=2))
        groups[name]=dict(files=len(before),changed=changed,missing=missing,passed=not changed and not missing)
    raster=Path('/home/prmi/zwj/experiments/global_view_joint_lod_20261105/evaluation/python/diff-gaussian-rasterization/diff_gaussian_rasterization/_C.cpython-311-x86_64-linux-gnu.so')
    raster_sha=sha(raster);assert raster_sha=='27b5b468cf385dd1c96aeabdd65426da38a938a38d52d8181b1b1da485090c86'
    library=ROOT/'build/mycartgs_deployment_fp32_v13/mycartgs_deployment_fp32_v13.so'
    source={str(p.relative_to(ROOT/'source')):sha(p) for p in (ROOT/'source/python').glob('deployment*') if p.is_file()}
    source['python/frozen_inference_kernels.cu']=sha(ROOT/'source/python/frozen_inference_kernels.cu')
    oracle=Path('/home/prmi/zwj/experiments/deployment_radix_tiling_20261107/build/mycartgs_deployment_fp32_v11/mycartgs_deployment_fp32_v11.so')
    assert sha(oracle)=='cb41244904755c25c02a1dd601b84514c83b6982c6e862043a4eb9eca7e59899'
    (ROOT/'reports/BINARY_IDENTITY.json').write_text(json.dumps(dict(path=str(library),sha256=sha(library),source_sha256=source,
        oracle_path=str(oracle),oracle_sha256=sha(oracle),rasterizer_path=str(raster),rasterizer_sha256=raster_sha,
        base_commit='26c355b32bf373a75877fc4605a4e41b8de9b47f'),indent=2))
    summary=dict(passed=all(v['passed'] for v in groups.values()),groups=groups,rasterizer_unchanged=True,
        production_deployed=False,training_runs=0,commit=False,push=False,finished=time.time(),
        disk_free_bytes=__import__('shutil').disk_usage(ROOT).free)
    (ROOT/'reports/FINAL_INTEGRITY.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True);assert summary['passed']


if __name__=='__main__':main()
