"""在原服务器每30分钟核查云节点，流式拉回模型和数值，过滤渲染图片。"""
from pathlib import Path
import base64
import hashlib
import json
import os
import shlex
import stat
import sys
import time
import traceback
import paramiko
from cluster_transfer import R,write

TERMINAL={'complete','skipped_zero_selection','skipped_oom','failed','partial_evaluation'}


def remote(client,code):
    _,out,err=client.exec_command('python3 -c '+shlex.quote(code),timeout=300)
    result=out.read().decode();error=err.read().decode();assert out.channel.recv_exit_status()==0,error
    return json.loads(result)


def pull_job(client,sftp,job):
    folders=[R/'runs'/job['id'],R/'evaluation'/job['id']]
    code="""from pathlib import Path
import json,hashlib
result={}
for folder in FOLDERS:
 for p in Path(folder).rglob('*'):
  if p.is_file() and p.suffix.lower() not in ['.png','.jpg','.jpeg','.bmp','.webp']:
   h=hashlib.sha256()
   with p.open('rb') as f:
    for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
   result[str(p)]=dict(size=p.stat().st_size,sha256=h.hexdigest())
print(json.dumps(result))
""".replace('FOLDERS',repr([str(p) for p in folders]))
    files=remote(client,code);assert files,'无任务输出'
    assert __import__('shutil').disk_usage(R).free>5*2**30+sum(f['size'] for f in files.values())
    for name,expected in files.items():
        p=Path(name);assert any(p==f or f in p.parents for f in folders)
        if p.exists():
            h=hashlib.sha256(p.read_bytes()).hexdigest()
            if h==expected['sha256']:continue
            raise RuntimeError('汇总目标已有不同内容：'+name)
        p.parent.mkdir(parents=True,exist_ok=True);temp=p.with_name(p.name+'.cluster_tmp')
        sftp.get(name,str(temp));h=hashlib.sha256()
        with temp.open('rb') as f:
            for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
        assert temp.stat().st_size==expected['size'] and h.hexdigest()==expected['sha256'],name
        temp.replace(p)
    if job['state']=='complete':
        assert len(job['evaluations'])==23 and all(e['state']=='complete' for e in job['evaluations'])
        for e in job['evaluations']:
            out=Path(e['result']['output']);v=json.loads((out/'metric_summary.json').read_text())
            assert v['evaluated_frames']==job['full_evaluation']['result']['evaluated_frames']
    return dict(passed=True,files=files,images_copied=0,imported_unix=time.time())


def work(config):
    node=config['worker'];directory=R/'cluster'/node;directory.mkdir(exist_ok=True)
    password=config.pop('password');client=None
    while True:
        try:
            if client is None or not client.get_transport().is_active():
                client=paramiko.SSHClient();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
                client.connect(config['host'],port=config['port'],username='root',password=password,look_for_keys=False,allow_agent=False,timeout=30)
                fingerprint=base64.b64encode(hashlib.sha256(client.get_transport().get_remote_server_key().asbytes()).digest()).decode().rstrip('=')
                assert fingerprint=='yswdgtshuc+hy/m0jR576i7Nu0Es7lZ71jIrFKXnkE0'
                client.get_transport().set_keepalive(30)
            code="""from pathlib import Path
import json,os,time,subprocess
r=Path(ROOT);worker=WORKER
v=dict(checked_unix=time.time(),disk_free=__import__('shutil').disk_usage(r).free)
v['validation']=json.loads((r/'reports/CLOUD_VALIDATION.json').read_text()) if (r/'reports/CLOUD_VALIDATION.json').exists() else None
v['pipeline']=json.loads((r/'LOD_PIPELINE.json').read_text())
v['controller']=json.loads((r/'LOD_CONTROLLER.json').read_text()) if (r/'LOD_CONTROLLER.json').exists() else None
v['session']=json.loads((r/'reports/CLOUD_SESSION.json').read_text()) if (r/'reports/CLOUD_SESSION.json').exists() else None
if v['controller']:
 ident=v['controller'];p=Path('/proc')/str(ident['pid'])/'stat';fields=p.read_text().split(') ',1)[1].split() if p.exists() else []
 v['controller_alive']=bool(fields) and fields[0]!='Z' and fields[19]==ident['start_ticks'] and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==ident['boot_id']
else:v['controller_alive']=None
if v['session'] and v['session']['phase'].startswith('failed_'):
 p=Path(v['session']['log']);v['session_log_tail']=p.read_text(errors='replace')[-4000:] if p.exists() else ''
v['gpu']=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
v['active']=[]
for job in v['pipeline']['jobs']:
 if job.get('assigned_worker')!=worker:continue
 for item in [job,*job['evaluations']]:
  if item['state']!='running':continue
  proc=item['process'];d=Path('/proc')/str(proc['pid']);out=Path(item['attempt']);trace=out/'model/real_lod_steps.csv';log=out/('train.log' if item is job else 'evaluate.log');progress=trace if trace.exists() else log
  fields=(d/'stat').read_text().split(') ',1)[1].split() if d.exists() else []
  identity=bool(fields) and fields[19]==proc['start_ticks'] and Path('/proc/sys/kernel/random/boot_id').read_text().strip()==proc['boot_id']
  entry=dict(id=job['id'],process=proc,identity_matches=identity,progress_age=time.time()-progress.stat().st_mtime,
   progress_bytes=progress.stat().st_size,step=item.get('step'),process_state=fields[0] if fields else 'gone')
  if not identity or entry['progress_age']>600:
   entry['log_tail']=log.read_text(errors='replace')[-3000:]
   for name in ['wchan','io','status']:
    p=d/name
    if p.exists():entry[name]=p.read_text()[-3000:]
  v['active'].append(entry)
print(json.dumps(v))
""".replace('ROOT',repr(str(R))).replace('WORKER',repr(node))
            snapshot=remote(client,code);write(directory/'REMOTE_STATUS.json',snapshot)
            write(directory/'REMOTE_PIPELINE.json',snapshot['pipeline'])
            own=[j for j in snapshot['pipeline']['jobs'] if j.get('assigned_worker')==node]
            if snapshot['validation'] and snapshot['validation']['passed']:
                sftp=client.open_sftp()
                for job in own:
                    if job['state'] not in TERMINAL:continue
                    marker=directory/'imports'/('/'.join(job['id'].split('/'))+'.json');marker.parent.mkdir(parents=True,exist_ok=True)
                    if marker.exists():continue
                    receipt=pull_job(client,sftp,job);write(marker,dict(job=job,receipt=receipt));print('IMPORTED',job['id'],job['state'],flush=True)
                sftp.close()
            complete=all(j['state'] in TERMINAL and (directory/'imports'/(j['id']+'.json')).exists() for j in own)
            write(directory/'COLLECTOR_STATUS.json',dict(complete=complete,last_checked=time.time(),imported=sum((directory/'imports'/(j['id']+'.json')).exists() for j in own),jobs=len(own)))
            if complete:
                client.close();return
        except Exception:
            write(directory/'COLLECTOR_ERROR.json',dict(time=time.time(),traceback=traceback.format_exc()))
            traceback.print_exc()
            if client is not None:client.close();client=None
        time.sleep(1800)


if __name__=='__main__':
    config=json.loads(sys.stdin.readline());node=config['worker'];assert node in ['cloud_a','cloud_b']
    pid=os.fork()
    if pid:print('COLLECTOR_PID',pid,flush=True);sys.exit(0)
    os.setsid()
    with (R/f'logs/collector_{node}.log').open('a') as log:
        os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
        with open(os.devnull) as source:os.dup2(source.fileno(),0)
        work(config)
