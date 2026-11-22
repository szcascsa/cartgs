"""授权SSH通道；凭据仅驻留进程内存，断线按原认证重连。"""
import getpass
import json
import sys
import paramiko

secret=getpass.getpass('SSH authentication: ')
client=None


def connected():
    global client
    if client is None or not client.get_transport() or not client.get_transport().is_active():
        if client is not None:client.close()
        client=paramiko.SSHClient();client.load_system_host_keys()
        client.set_missing_host_key_policy(paramiko.RejectPolicy())
        client.connect('172.19.163.58',username='prmi',password=secret,
            look_for_keys=False,allow_agent=False,timeout=15)
        client.get_transport().set_keepalive(20)
    return client


connected();print('CONNECTED',flush=True)
try:
    for line in sys.stdin:
        try:
            task=json.loads(line);session=connected()
            if task['op']=='exec':
                _,out,err=session.exec_command(task['cmd'],timeout=task.get('timeout',120))
                print(out.read().decode(errors='replace'),end='',flush=True)
                print(err.read().decode(errors='replace'),end='',flush=True)
                print('EXIT',out.channel.recv_exit_status(),flush=True)
            elif task['op'] in ['get','put']:
                with session.open_sftp() as transfer:
                    if task['op']=='get':transfer.get(task['remote'],task['local'])
                    else:transfer.put(task['local'],task['remote'])
                print('TRANSFER_OK',task['op'],flush=True)
            elif task['op']=='quit':break
        except Exception as error:
            print(type(error).__name__,str(error),flush=True)
finally:
    if client is not None:client.close()
