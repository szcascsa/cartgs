"""串行构建、验收、训练；后处理由已准备的独立脚本继续。"""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
PYTHON='/home/prmi/anaconda3/envs/zwj/bin/python'


def main():
    lock=(ROOT/'reports/pipeline.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state=dict(pid=os.getpid(),start_ticks=Path(f'/proc/{os.getpid()}/stat').read_text().split()[21],
        boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),state='running')
    for name in ['preflight.py','build_isolated.py','validate_native.py','validate_inference.py','run_full.py']:
        state.update(stage=name,updated_unix=time.time())
        (ROOT/'reports/PIPELINE.json').write_text(json.dumps(state,indent=2))
        env=dict(os.environ);env['OMP_NUM_THREADS']='2'
        with (ROOT/'logs'/f'pipeline_{name}.log').open('w') as f:
            child=subprocess.Popen([PYTHON,'-u',str(HERE/name)],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT)
            if name=='build_isolated.py':
                identity=dict(pid=child.pid,start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split()[21],boot_id=state['boot_id'])
                (ROOT/'reports/BUILD_PID.json').write_text(json.dumps(identity,indent=2))
            code=child.wait()
        if code:
            state.update(state='failed',returncode=code)
            (ROOT/'reports/PIPELINE.json').write_text(json.dumps(state,indent=2))
            raise RuntimeError(f'{name}失败，请检查对应日志，禁止覆盖现场或盲重试')
    state.update(state='training_complete',updated_unix=time.time())
    (ROOT/'reports/PIPELINE.json').write_text(json.dumps(state,indent=2))


if __name__=='__main__':main()
