"""交互凭据仅保留在内存，用于恢复已授权的本批内存隔离组。"""
import getpass
import paramiko

credential=getpass.getpass('SSH/sudo password: ')
client=paramiko.SSHClient();client.load_system_host_keys()
client.connect('172.19.163.58',username='prmi',password=credential,
               look_for_keys=False,allow_agent=False,timeout=15)
try:
    stdin,stdout,stderr=client.exec_command(
        'sudo -S -p "" /usr/bin/python3 /home/prmi/zwj/realLOD/ablation/restore_cgroup.py',timeout=30)
    stdin.write(credential+'\n');stdin.flush();stdin.channel.shutdown_write()
    credential=None
    print(stdout.read().decode(errors='replace'),end='')
    print(stderr.read().decode(errors='replace'),end='')
    status=stdout.channel.recv_exit_status();print('EXIT',status)
    raise SystemExit(status)
finally:
    credential=None;client.close()
