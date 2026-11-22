"""Run both verified targets sequentially and retain independent monitoring."""
from pathlib import Path
import json
import subprocess

root = Path(__file__).resolve().parent
python = '/home/prmi/anaconda3/envs/zwj/bin/python'
for name, target in [('target05', '0.5'), ('target07', '0.7')]:
    folder = root/name
    assert json.loads((folder/'smoke/VALIDATION.json').read_text())['passed']
    job = folder/'logs/formal_job.json'
    with job.open('x') as stream:
        json.dump(dict(name='formal', measured=False, block_activity_acceptance=True,
            command=['/usr/bin/env', 'DYNAMIC_TARGET_MS='+target, python, '-B', '-u', str(root/'benchmark.py')]), stream)
    with (folder/'logs/formal_runner.log').open('x') as out:
        subprocess.run([python, '-B', '-u', str(folder/'job_runner.py'), str(job)], cwd=folder,
            stdin=subprocess.DEVNULL, stdout=out, stderr=subprocess.STDOUT, check=True)
    status = json.loads(job.read_text())
    print(name, status['state'], status['returncode'], flush=True)
    assert status['returncode'] == 0
