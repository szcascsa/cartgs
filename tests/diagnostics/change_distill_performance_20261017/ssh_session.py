"""本次审计的临时 SSH 会话；密码只从进程 stdin 读取，不保存。"""
import argparse
import getpass
import json
import sys
import paramiko

p = argparse.ArgumentParser()
p.add_argument('--host', required=True)
p.add_argument('--user', required=True)
a = p.parse_args()
client = paramiko.SSHClient()
client.load_system_host_keys()
client.set_missing_host_key_policy(paramiko.RejectPolicy())
print('AUTH_STDIN', flush=True)
password = getpass.getpass('SSH password: ')
client.connect(a.host, username=a.user, password=password,
               look_for_keys=False, allow_agent=False, timeout=15)
password = None
print('CONNECTED', flush=True)
for line in sys.stdin:
    try:
        request = json.loads(line)
        if request.get('close'):
            break
        if 'put' in request or 'get' in request:
            operation = 'put' if 'put' in request else 'get'
            with client.open_sftp() as sftp:
                for source, target in request[operation]:
                    getattr(sftp, operation)(source, target)
            print(json.dumps({'transfers': len(request[operation])}), flush=True)
        else:
            _, stdout, stderr = client.exec_command(request['cmd'], timeout=request.get('timeout', 60))
            for stream, label in ((stdout, 'stdout'), (stderr, 'stderr')):
                data = stream.read().decode('utf-8', errors='replace')
                if data:
                    print(label + ':\n' + data, flush=True)
            print('EXIT=' + str(stdout.channel.recv_exit_status()), flush=True)
    except Exception as error:
        print(json.dumps({'error': str(error)}), flush=True)
client.close()
