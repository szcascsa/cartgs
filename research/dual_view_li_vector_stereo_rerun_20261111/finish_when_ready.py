"""当前独立队列结束后执行只读对照汇总；遇到真实故障保留现场。"""
from pathlib import Path
import fcntl
import json
import subprocess
import sys
import time

R=Path(__file__).resolve().parent


def main():
    lock=(R/'finish.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        state=json.loads((R/'LOD_PIPELINE.json').read_text())
        if state['phase']=='finished':
            if not (R/'reports/RERUN_VERIFIED.json').exists():
                result=subprocess.run([sys.executable,str(R/'report_rerun.py')],cwd=R)
                assert result.returncode==0,'配对报告核验失败，保留日志'
            return
        if state['phase'] in ['held','blocked']:
            print('PAUSED_WITH_STATE',state['phase'],state.get('error'),flush=True)
            return
        time.sleep(30)


if __name__=='__main__':main()
