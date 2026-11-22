"""构建后的生命周期回归；所有输出留在隔离目录。"""
from pathlib import Path
import json,subprocess

root=Path('/home/prmi/zwj/anchor-si-20261024');out=root/'regression';out.mkdir(exist_ok=True)
cases=[('private_compensation_mapping_test',['cuda','fused']),
       ('private_compensation_cuda_test',['cuda']),
       ('gaussian_split_sampling_test',['cuda']),
       ('local_anchor_migration_test',[])]
rows=[]
try:
    for executable,arguments in cases:
        with (out/(executable+'.log')).open('w') as log:
            result=subprocess.run([str(root/'source/bin'/executable)]+arguments,cwd=out,stdout=log,stderr=subprocess.STDOUT)
        rows.append(dict(case=executable,arguments=arguments,returncode=result.returncode));result.check_returncode()
    print('REGRESSION_PASS')
finally:
    (out/'results.json').write_text(json.dumps(rows,indent=2))
