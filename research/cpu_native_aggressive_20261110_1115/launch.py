"""仅启动本轮独占的一个实验任务，按PID/start_ticks防止重复。"""
from pathlib import Path
import argparse,json,os,subprocess,sys

HERE=Path(__file__).resolve().parent


def alive(identity):
    if identity.get('state') in ['complete','failed']:return False
    if identity.get('state')=='launching':return True
    p=Path('/proc')/str(identity.get('runner_pid',0))/'stat'
    return p.is_file() and p.read_text().rpartition(')')[2].split()[0]!='Z' and p.read_text().rpartition(')')[2].split()[19]==identity.get('runner_start_ticks')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('name');parser.add_argument('command',nargs=argparse.REMAINDER)
    parser.add_argument('--measure',action='store_true');args=parser.parse_args()
    assert args.name.replace('_','').replace('-','').isalnum() and args.command
    directory=HERE/'reports/jobs';directory.mkdir(exist_ok=True)
    for path in directory.glob('*.json'):
        identity=json.loads(path.read_text())
        assert not alive(identity),('Existing own job still running',identity)
    path=directory/(args.name+'.json');assert not path.exists(),path
    identity=dict(name=args.name,command=args.command,measured=args.measure,state='launching')
    path.write_text(json.dumps(identity,indent=2))
    log=(HERE/'logs'/('runner_'+args.name+'.log')).open('x')
    process=subprocess.Popen([sys.executable,'-u',str(HERE/'job_runner.py'),str(path)],cwd=HERE,
        stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print('OWN_JOB_LAUNCHED',args.name,process.pid,flush=True)


if __name__=='__main__':main()
