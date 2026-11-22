"""阶段A完成后先形成固定分析，再执行预注册Rotation对照，不据评价改变配置。"""
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def execute(script,phase,log):
    with (ROOT/log).open('w') as stream:
        result=subprocess.run([sys.executable,'-u',str(ROOT/script),phase],cwd=ROOT,
            stdin=subprocess.DEVNULL,stdout=stream,stderr=subprocess.STDOUT)
    assert result.returncode==0,(script,phase,result.returncode)


try:
    while not (ROOT/'phase_A.exit').exists():
        time.sleep(10)
    assert (ROOT/'phase_A.exit').read_text().strip()=='0'
    execute('analyze.py','A','analyze_A.log')
    (ROOT/'analysis_A.exit').write_text('0')
    execute('run.py','B','phase_B.log')
    execute('analyze.py','B','analyze_B.log')
except Exception:
    (ROOT/'controller.exit').write_text('1')
    raise
(ROOT/'controller.exit').write_text('0')
