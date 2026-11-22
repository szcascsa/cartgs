"""交互SSH通道；授权凭据只在内存，不写文件或日志。"""
import getpass,json,sys
from pathlib import Path
import paramiko

client=paramiko.SSHClient();client.load_host_keys(str(Path.home()/'.ssh/known_hosts'))
client.set_missing_host_key_policy(paramiko.RejectPolicy())
secret=getpass.getpass('SSH credential: ')
client.connect('172.19.163.58',username='prmi',password=secret,look_for_keys=False,allow_agent=False,timeout=15)
del secret
sftp=client.open_sftp();print('SSH_READY',flush=True)
for line in sys.stdin:
    try:
        request=json.loads(line);op=request['op']
        if op=='put':sftp.put(request['local'],request['remote']);print('UPLOADED',request['remote'],flush=True)
        elif op=='get':sftp.get(request['remote'],request['local']);print('DOWNLOADED',request['local'],flush=True)
        elif op=='exec':
            _,out,err=client.exec_command(request['cmd'],timeout=55)
            print(out.read().decode('utf-8',errors='replace'),end='',flush=True)
            print(err.read().decode('utf-8',errors='replace'),end='',flush=True)
            print('EXIT',out.channel.recv_exit_status(),flush=True)
        elif op=='close':break
        else:raise ValueError('Unsupported operation')
    except Exception as exc:print(type(exc).__name__,str(exc),flush=True)
sftp.close();client.close()
