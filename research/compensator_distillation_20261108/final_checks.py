from pathlib import Path
import json
import os
import subprocess
import sys
import time


ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent


def main():
    identity=json.loads((ROOT/'reports/IMAGE_CONTROLLER.json').read_text())
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==identity['boot_id']
    while not (ROOT/'reports/IMAGE_CALIBRATION_COMPLETE.json').exists():
        proc=Path(f'/proc/{identity["pid"]}/stat')
        assert proc.exists() and proc.read_text().rpartition(')')[2].split()[19]==identity['start_ticks'],'Image calibration exited before completion'
        time.sleep(5)
    tasks=[('parity','trained_parity.py'),('functional_v2','functional_training.py'),('integrity','integrity.py')]
    result=dict(pid=os.getpid(),tasks=[])
    for name,script in tasks:
        with (ROOT/'logs'/('delivery_'+name+'.log')).open('w') as out:
            ec=subprocess.call([sys.executable,'-u',str(HERE/script)],stdout=out,stderr=subprocess.STDOUT)
        result['tasks'].append(dict(name=name,exit=ec));(ROOT/'reports/DELIVERY_TESTS.json').write_text(json.dumps(result,indent=2))
        print(name,ec,flush=True)
        if ec:raise SystemExit(ec)
    result['complete']=True;(ROOT/'reports/DELIVERY_TESTS.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
