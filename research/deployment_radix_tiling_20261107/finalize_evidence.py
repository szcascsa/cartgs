"""只读核验保护目录，记录独立产物身份。"""
from pathlib import Path
import hashlib
import json
import subprocess
import time

ROOT=Path(__file__).resolve().parents[3]


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    assert json.loads((ROOT/'reports/ACCEPTANCE_COMPLETE.json').read_text())['passed']
    groups={}
    for name in ['PRODUCTION','INPUTS','LEGACY_INPUTS','LARGE_INPUTS']:
        before=json.loads((ROOT/'reports'/f'{name}_BEFORE.json').read_text());after={};changed=[];missing=[]
        for filename,value in before.items():
            path=Path(filename)
            if not path.is_file():missing.append(filename);continue
            result=digest(path)
            if isinstance(value,dict):
                after[filename]=dict(bytes=path.stat().st_size,sha256=result)
                if value.get('sha256')!=result:changed.append(filename)
            else:
                after[filename]=result
                if value!=result:changed.append(filename)
        (ROOT/'reports'/f'{name}_AFTER.json').write_text(json.dumps(after,indent=2))
        groups[name]=dict(files=len(before),changed=changed,missing=missing,passed=not changed and not missing)
    library=ROOT/'build/mycartgs_deployment_fp32_v11/mycartgs_deployment_fp32_v11.so'
    source={str(p.relative_to(ROOT/'source')):digest(p) for p in (ROOT/'source/python').glob('deployment*') if p.is_file()}
    source['python/frozen_inference_kernels.cu']=digest(ROOT/'source/python/frozen_inference_kernels.cu')
    oracle=Path('/home/prmi/zwj/experiments/deployment_device_counts_20261107/build/mycartgs_deployment_fp32_v7/mycartgs_deployment_fp32_v7.so')
    (ROOT/'reports/BINARY_IDENTITY.json').write_text(json.dumps(dict(path=str(library),sha256=digest(library),
        oracle_path=str(oracle),oracle_sha256=digest(oracle),
        source_sha256=source,base_commit='26c355b32bf373a75877fc4605a4e41b8de9b47f'),indent=2))
    summary=dict(passed=all(v['passed'] for v in groups.values()),groups=groups,finished=time.time(),
        production_deployed=False,commit=False,push=False,training_runs=0,
        gpu_state=subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used,clocks.sm','--format=csv,noheader'],text=True),
        disk_free_bytes=__import__('shutil').disk_usage(ROOT).free)
    (ROOT/'reports/FINAL_INTEGRITY.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
    assert summary['passed']


if __name__=='__main__':main()
