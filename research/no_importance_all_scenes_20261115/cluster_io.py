"""通用SSH/SFTP通道；凭据仅从不回显输入读取，不写入文件。"""
import argparse
import base64
import getpass
import hashlib
import json
import sys
import paramiko

p=argparse.ArgumentParser();p.add_argument('host');p.add_argument('port',type=int);p.add_argument('user');args=p.parse_args()
c=paramiko.SSHClient();c.load_system_host_keys();c.set_missing_host_key_policy(paramiko.AutoAddPolicy())
c.connect(args.host,port=args.port,username=args.user,password=getpass.getpass('SSH password: '),look_for_keys=False,allow_agent=False,timeout=20)
key=c.get_transport().get_remote_server_key()
print('CONNECTED',args.host,args.port,'SHA256:'+base64.b64encode(hashlib.sha256(key.asbytes()).digest()).decode().rstrip('='),flush=True)
s=c.open_sftp()
try:
    for line in sys.stdin:
        try:
            task=json.loads(line)
            if task['op']=='exec':
                _,out,err=c.exec_command(task['cmd'],timeout=300)
                print(out.read().decode(errors='replace'),end='',flush=True)
                print(err.read().decode(errors='replace'),end='',flush=True)
                print('EXIT',out.channel.recv_exit_status(),flush=True)
            elif task['op']=='put':s.put(task['local'],task['remote']);print('UPLOADED',task['remote'],flush=True)
            elif task['op']=='get':s.get(task['remote'],task['local']);print('DOWNLOADED',task['local'],flush=True)
            elif task['op']=='relay':
                password=getpass.getpass('Destination SSH password: ')
                channel=c.get_transport().open_session();channel.exec_command(task['command'])
                stream=channel.makefile_stdin('wb');stream.write((json.dumps(dict(task['payload'],password=password))+'\n').encode());stream.flush();stream.close();password=None
                while True:
                    if channel.recv_ready():print(channel.recv(65536).decode(errors='replace'),end='',flush=True)
                    if channel.recv_stderr_ready():print(channel.recv_stderr(65536).decode(errors='replace'),end='',flush=True)
                    if channel.exit_status_ready() and not channel.recv_ready() and not channel.recv_stderr_ready():break
                    __import__('time').sleep(.2)
                print('EXIT',channel.recv_exit_status(),flush=True)
            elif task['op']=='quit':break
        except Exception as error:print(type(error).__name__,str(error),flush=True)
finally:s.close();c.close()
