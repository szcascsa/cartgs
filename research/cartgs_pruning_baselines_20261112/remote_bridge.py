"""带SSH保活的本次实验文件与命令通道，凭据仅留在内存。"""
import getpass
import json
import sys
import paramiko

password=getpass.getpass('SSH password: ')


def connect():
    connection=paramiko.SSHClient();connection.load_system_host_keys()
    connection.connect('172.19.163.58',username='prmi',password=password,
                       look_for_keys=False,allow_agent=False,timeout=15)
    connection.get_transport().set_keepalive(15)
    return connection,connection.open_sftp()


client,sftp=connect();print('CONNECTED',flush=True)
try:
    for line in sys.stdin:
        try:
            task=json.loads(line)
            if task['op']=='quit':break
            if not client.get_transport().is_active():
                client.close();client,sftp=connect();print('RECONNECTED',flush=True)
            if task['op']=='put':
                sftp.put(task['local'],task['remote']);print('UPLOADED',task['remote'],flush=True)
            elif task['op']=='get':
                sftp.get(task['remote'],task['local']);print('DOWNLOADED',task['local'],flush=True)
            else:
                _,out,err=client.exec_command(task['cmd'],timeout=300)
                print(out.read().decode(errors='replace'),end='',flush=True)
                print(err.read().decode(errors='replace'),end='',flush=True)
                print('EXIT',out.channel.recv_exit_status(),flush=True)
        except Exception as error:print(type(error).__name__,str(error),flush=True)
finally:
    sftp.close();client.close()
