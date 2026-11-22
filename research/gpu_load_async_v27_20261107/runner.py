"""串行执行已锁定协议的速度、质量与升降负载验证。"""
from pathlib import Path
import json
import os
import subprocess
import time

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
PYTHON='/home/prmi/anaconda3/envs/zwj/bin/python'


def main():
    assert (ROOT/'reports/PROTOCOL_LOCK.json').is_file()
    tasks=[]
    for stage,label in [('benchmark','benchmark02'),('quality','quality02'),('curves','curves02')]:
        assert not (ROOT/'runs'/label).exists()
        row=dict(stage=stage,label=label,state='running',start_unix=time.time())
        tasks.append(row)
        with (ROOT/'logs'/f'{label}.log').open('w') as stream:
            process=subprocess.Popen([PYTHON,'-u',str(HERE/'experiment.py'),stage,'--label',label],
                stdout=stream,stderr=subprocess.STDOUT)
            row.update(pid=process.pid,start_ticks=Path(f'/proc/{process.pid}/stat').read_text().split()[21],
                       boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
            (ROOT/'reports/PIPELINE.json').write_text(json.dumps(tasks,indent=2))
            code=process.wait()
        row.update(state='complete' if code==0 else 'failed',exit_code=code,end_unix=time.time())
        (ROOT/'reports/PIPELINE.json').write_text(json.dumps(tasks,indent=2))
        print(stage,row['state'],flush=True)
        if code:
            raise SystemExit(code)


if __name__=='__main__':
    main()
