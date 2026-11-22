"""仅对有明确分配失败或内核OOM证据的训练失败重试，最多额外两次。"""
from pathlib import Path
import json
import os
import re
import subprocess

MAX_OOM_RETRIES=2


def classify_oom(job, exit_info, journal_records=None):
    if exit_info['returncode']==0:
        return None
    log=(Path(job['output'])/'train.log').read_text(errors='replace')
    allocation=re.compile(r'CUDA out of memory|CUDA error: out of memory|torch\.OutOfMemoryError|'
                          r'DefaultCPUAllocator:.*(?:allocate memory|out of memory)|std::bad_alloc')
    hits=[line for line in log.splitlines() if allocation.search(line)]
    if hits:
        return dict(kind='allocation_oom',evidence=hits[-8:])
    if exit_info['returncode'] not in [-9,137] or 'process' not in job:
        return None
    identity=job['process'];pid=identity['pid']
    if journal_records is None:
        pattern=rf'Killed process {pid} \(|task=[^,]+,pid={pid},'
        result=subprocess.run(['journalctl','-k','-b','--no-pager','-o','json','--grep',pattern],
                              stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
        if result.returncode!=0:
            return None
        journal_records=[json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
    lower=int(identity['start_ticks'])/os.sysconf('SC_CLK_TCK')
    upper=lower+exit_info['seconds']+30
    evidence=[]
    for record in journal_records:
        if record.get('_BOOT_ID',record.get('__BOOT_ID','')).replace('-','')!=identity['boot_id'].replace('-',''):
            continue
        when=int(record.get('__MONOTONIC_TIMESTAMP','0'))/1e6
        message=record.get('MESSAGE','')
        matches=(f'Killed process {pid} (' in message and 'Out of memory' in message) or (
            'oom-kill:' in message and re.search(rf'\bpid={pid},',message))
        if matches and lower<=when<=upper:
            evidence.append(message)
    return dict(kind='kernel_cpu_oom',evidence=evidence) if evidence else None


def record_attempt(job, exit_info, oom):
    history=job.setdefault('attempt_history',[])
    if any(item['output']==job['output'] for item in history):
        return
    item=dict(output=job['output'],process=job.get('process'),returncode=exit_info['returncode'],
              seconds=exit_info['seconds'],state=job['state'],error=job.get('error',''),oom=oom)
    if 'result' in job:
        item['result']=job['result']
    history.append(item)


def schedule_failure(job, oom):
    job['failure_kind']=oom['kind'] if oom else 'non_oom'
    if not oom:
        job['state']='failed'
    elif job.get('oom_retry_count',0)<MAX_OOM_RETRIES:
        job['state']='retry_pending'
    else:
        job.update(state='skipped_oom',error='OOM after original attempt and 2 retries; skip; preserve all attempts')


def next_attempt(job):
    assert job['state']=='retry_pending' and job.get('oom_retry_count',0)<MAX_OOM_RETRIES
    old=job['output'];base=Path(old).parent
    number=max(int(p.name.split('_')[1]) for p in base.glob('attempt_*') if p.is_dir())+1
    output=base/('attempt_%02d'%number)
    assert not output.exists() and job['command'].count(old+'/model')==1
    job['command']=[str(output/'model') if part==old+'/model' else part for part in job['command']]
    job.update(output=str(output),oom_retry_count=job.get('oom_retry_count',0)+1)
    for key in ['result','process','error']:
        job.pop(key,None)
