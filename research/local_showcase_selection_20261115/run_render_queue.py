"""顺序补渲染四组最佳视角及两组Replica明亮备选；失败立即保留现场。"""
from pathlib import Path
import json
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
TASKS=[('Replica/office1/rgbd',False),('Replica/office1/mono',False),
       ('TUM_RGBD/freiburg2_xyz/rgbd',False),('TUM_RGBD/freiburg2_xyz/mono',False),
       ('Replica/office4/rgbd',True),('Replica/office0/mono',True)]


def main():
    for index,(group,alternative) in enumerate(TASKS):
        label='alternatives' if alternative else 'best'
        receipt=ROOT/'images'/label/group/'RENDER_VERIFIED.json'
        if receipt.exists():continue
        command=[sys.executable,'-u',str(ROOT/'render_candidates.py'),'--group',group]
        if alternative:command.append('--alternative')
        with (ROOT/'logs'/f"{group.replace('/','_')}_final.log").open('w') as log:
            process=subprocess.Popen(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
            state=dict(index=index,group=group,pid=process.pid,state='running',started=time.time())
            (ROOT/'RENDER_QUEUE.json').write_text(json.dumps(state)+'\n')
            code=process.wait()
        assert code==0 and receipt.exists(),(group,code)
    (ROOT/'RENDER_QUEUE.json').write_text(json.dumps(dict(state='complete',tasks=len(TASKS)))+'\n')


if __name__=='__main__':
    main()
