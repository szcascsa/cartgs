"""实验文件传输与命令通道；密码只保留在进程内存。"""
import getpass
import json
import sys
import paramiko

client = paramiko.SSHClient()
client.load_system_host_keys()
client.connect('172.19.163.58', username='prmi', password=getpass.getpass('SSH password: '),
               look_for_keys=False, allow_agent=False, timeout=15)
sftp = client.open_sftp()
print('CONNECTED', flush=True)
try:
    for line in sys.stdin:
        try:
            task = json.loads(line)
            if task['op'] == 'put':
                sftp.put(task['local'], task['remote'])
                print('UPLOADED', task['remote'], flush=True)
            elif task['op'] == 'get':
                sftp.get(task['remote'], task['local'])
                print('DOWNLOADED', task['local'], flush=True)
            elif task['op'] == 'exec':
                _, stdout, stderr = client.exec_command(task['cmd'], timeout=300)
                print(stdout.read().decode(errors='replace'), end='', flush=True)
                print(stderr.read().decode(errors='replace'), end='', flush=True)
                print('EXIT', stdout.channel.recv_exit_status(), flush=True)
            elif task['op'] == 'quit':
                break
        except Exception as error:
            print(type(error).__name__, str(error), flush=True)
finally:
    sftp.close()
    client.close()
