"""管理员只读Profiler通道；凭据仅保留在进程内存，不输出或落盘。"""
import getpass
import json
import sys
import paramiko

secret=getpass.getpass('SSH password: ')
client=paramiko.SSHClient()
client.load_system_host_keys()
client.connect('172.19.163.58',username='prmi',password=secret,
    look_for_keys=False,allow_agent=False,timeout=15)
print('CONNECTED',flush=True)
try:
    for line in sys.stdin:
        request=json.loads(line)
        if request['op']=='quit':break
        assert request['op']=='profile'
        command=request['cmd']
        assert command.startswith('/usr/local/cuda-11.8/nsight-compute-2022.3.0/ncu ')
        incoming,stdout,stderr=client.exec_command('sudo -S -p "" -- '+command,timeout=300)
        incoming.write(secret+'\n');incoming.flush();incoming.channel.shutdown_write()
        print(stdout.read().decode(errors='replace'),end='',flush=True)
        print(stderr.read().decode(errors='replace'),end='',flush=True)
        print('EXIT',stdout.channel.recv_exit_status(),flush=True)
finally:
    secret=None
    client.close()
