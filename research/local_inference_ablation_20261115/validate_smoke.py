"""六种入口的首中末帧验收，通过后启动已授权正式局部评价。"""
import os
import subprocess
import sys
import time
from common import ROOT, load, write

GROUPS = ['TUM_RGBD/freiburg1_xyz/mono','TUM_RGBD/freiburg1_xyz/rgbd',
          'VECtor/corner-slow/mono','VECtor/corner-slow/stereo',
          'Replica/office0/mono','Replica/office0/rgbd']


def main():
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    checks = []
    for group in GROUPS:
        folder = ROOT/'smoke'/group/'attempt_01'
        done = folder/'SMOKE.json'
        write(ROOT/'STATUS.json', dict(state='validating', group=group, validated=len(checks), updated_unix=time.time()))
        if not done.exists():
            folder.mkdir(parents=True, exist_ok=True)
            with (folder/'worker.log').open('a') as log:
                result = subprocess.run([sys.executable,'-B',str(ROOT/'worker.py'),'--group',group,'--smoke'],
                                        stdout=log,stderr=subprocess.STDOUT,env=env)
            assert result.returncode == 0, f'SMOKE_FAILED {group} {folder}'
        value = load(done)
        assert value['smoke'] and value['all_frame_ids_and_K_verified'] and value['source_unchanged']
        checks.append(dict(group=group,historical_max_error=value['historical_max_error'],
            scalar_metric_max_error=value['scalar_metric_max_error'],packet_checks=len(value['packet_checks'])))
        print('SMOKE_PASSED',group,flush=True)
    write(ROOT/'SMOKE_VERIFIED.json',dict(passed=True,groups=checks,updated_unix=time.time()))
    with (ROOT/'controller.log').open('a') as log:
        process = subprocess.Popen([sys.executable,'-B',str(ROOT/'controller.py')],cwd=ROOT,env=env,
            stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    print('FORMAL_CONTROLLER_STARTED',process.pid,flush=True)


if __name__ == '__main__':
    main()
