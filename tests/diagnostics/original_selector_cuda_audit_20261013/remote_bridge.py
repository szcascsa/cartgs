"""SSH diagnostic transport. Credentials arrive on stdin and are never saved."""
import json
import sys
from pathlib import Path
import paramiko

credentials = json.loads(sys.stdin.readline())
client = paramiko.SSHClient()
known = Path(__file__).resolve().parents[4] / 'ultra-light-validation-20261010/ssh_known_hosts'
client.load_host_keys(str(known))
client.set_missing_host_key_policy(paramiko.RejectPolicy())
client.connect(credentials['host'], username=credentials['username'],
               password=credentials.pop('password'), look_for_keys=False,
               allow_agent=False, timeout=20)
sftp = client.open_sftp()
print('READY', flush=True)
for line in sys.stdin:
    try:
        args = json.loads(line)
        if args['op'] == 'put':
            sftp.put(args['local'], args['remote'])
            result = {'ok': True}
        elif args['op'] == 'get':
            sftp.get(args['remote'], args['local'])
            result = {'ok': True}
        elif args['op'] == 'exec':
            _, out, err = client.exec_command(args['command'], timeout=55)
            result = {'stdout': out.read().decode(errors='replace'),
                      'stderr': err.read().decode(errors='replace'),
                      'exit': out.channel.recv_exit_status()}
        elif args['op'] == 'close':
            break
        else:
            raise ValueError('Unknown operation')
        print(json.dumps(result, ensure_ascii=True), flush=True)
    except Exception as error:
        print(json.dumps({'error': str(error)}), flush=True)
sftp.close()
client.close()
