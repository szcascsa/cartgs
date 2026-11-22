"""原服务器向分派节点流式传包；SSH凭据仅经stdin进入内存。"""
from pathlib import Path
import base64
import hashlib
import json
import os
import shlex
import sys
import time
import traceback
import paramiko

R=Path('/home/prmi/zwj/experiments/no_importance_all_scenes_20261115')
D='/root/autodl-tmp/no_importance_20261115'


def write(path,value):
    path=Path(path);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)


def execute(client,command):
    _,out,err=client.exec_command(command,timeout=300)
    value=out.read().decode(errors='replace');error=err.read().decode(errors='replace')
    assert out.channel.recv_exit_status()==0,(value,error)
    return value


def work(config):
    node=config['worker'];status=R/f'cluster/{node}_TRANSFER.json'
    try:
        client=paramiko.SSHClient();client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        client.connect(config['host'],port=config['port'],username='root',password=config.pop('password'),
            look_for_keys=False,allow_agent=False,timeout=30)
        key=client.get_transport().get_remote_server_key()
        fingerprint=base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip('=')
        assert fingerprint=='yswdgtshuc+hy/m0jR576i7Nu0Es7lZ71jIrFKXnkE0'
        setup="from pathlib import Path; p=Path('/home/prmi'); d=Path("+repr(D)+"); (d/'mirror/home/prmi').mkdir(parents=True,exist_ok=True); (d/'receipts').mkdir(exist_ok=True); assert not p.exists() or (p.is_symlink() and p.resolve()==(d/'mirror/home/prmi').resolve()); p.parent.mkdir(exist_ok=True); p.symlink_to(d/'mirror/home/prmi',target_is_directory=True) if not p.is_symlink() else None"
        execute(client,'python3 -c '+shlex.quote(setup))
        sftp=client.open_sftp()
        for label in ['environment','runtime',node]:
            receipt=R/f'cluster/{label}.receipt.json';start=time.time()
            while not receipt.exists():
                write(status,dict(state='waiting_package',label=label,updated_at=time.time()))
                assert time.time()-start<7200,'打包超过2小时'
                time.sleep(5)
            expected=json.loads(receipt.read_text());file=R/f'cluster/{label}.tar.gz'
            channel=client.get_transport().open_session();channel.settimeout(120)
            channel.exec_command('tar --no-same-owner -xzf - -C '+shlex.quote(D+'/mirror'))
            h=hashlib.sha256();sent=0;last=0;start=time.time()
            with file.open('rb') as stream:
                for block in iter(lambda:stream.read(4*1024**2),b''):
                    channel.sendall(block);h.update(block);sent+=len(block)
                    if time.time()-last>3:
                        write(status,dict(state='transferring',label=label,bytes=sent,total=expected['bytes'],
                            mib_per_second=round(sent/max(1,time.time()-start)/2**20,2),updated_at=time.time()));last=time.time()
            channel.shutdown_write()
            out=channel.makefile('r').read();err=channel.makefile_stderr('r').read()
            assert channel.recv_exit_status()==0,(out,err)
            assert h.hexdigest()==expected['sha256']
            sftp.put(str(R/f'cluster/{label}.files.json'),D+f'/receipts/{label}.files.json')
            sftp.put(str(receipt),D+f'/receipts/{label}.receipt.json')
            print('TRANSFERRED',node,label,sent,round(time.time()-start),flush=True)
        verification="""from pathlib import Path
import json,hashlib,time
d=Path(DIRECTORY);count=0;size=0
for label in ['environment','runtime',WORKER]:
 for name,expected in json.loads((d/'receipts'/(label+'.files.json')).read_text()).items():
  p=d/'mirror'/name;h=hashlib.sha256()
  with p.open('rb') as f:
   for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
  assert p.stat().st_size==expected['size'] and h.hexdigest()==expected['sha256'],str(p)
  count+=1;size+=expected['size']
v=dict(passed=True,files=count,bytes=size,verified_unix=time.time())
(d/'receipts/FILES_VERIFIED.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
""".replace('DIRECTORY',repr(D)).replace('WORKER',repr(node))
        write(status,dict(state='verifying_files',updated_at=time.time()))
        # 大量文件SHA验证允许长于单次SSH默认命令超时。
        channel=client.get_transport().open_session();channel.settimeout(7200)
        channel.exec_command('python3 -c '+shlex.quote(verification))
        out=channel.makefile('r').read().decode();err=channel.makefile_stderr('r').read().decode()
        assert channel.recv_exit_status()==0,(out,err)
        write(status,dict(state='complete',verification=json.loads(out),updated_at=time.time()))
        print('FILES_VERIFIED',node,out,flush=True)
        sftp.close();client.close()
    except Exception:
        write(status,dict(state='failed',error=traceback.format_exc(),updated_at=time.time()))
        traceback.print_exc();raise


if __name__=='__main__':
    config=json.loads(sys.stdin.readline());worker=config['worker']
    assert worker in ['cloud_a','cloud_b']
    pid=os.fork()
    if pid:
        print('TRANSFER_PID',pid,flush=True);sys.exit(0)
    os.setsid()
    with (R/f'logs/transfer_{worker}.log').open('a') as log:
        os.dup2(log.fileno(),1);os.dup2(log.fileno(),2)
        with open(os.devnull) as source:os.dup2(source.fileno(),0)
        work(config)
