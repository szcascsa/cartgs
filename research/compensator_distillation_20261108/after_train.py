from pathlib import Path
import json
import subprocess
import sys
import time


ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent


def main():
    identity=json.loads((ROOT/'reports/TRAIN_CONTROLLER.json').read_text())
    while not (ROOT/'reports/FR1_DISTILLATION_COMPLETE.json').exists():
        proc=Path(f'/proc/{identity["pid"]}/stat')
        assert proc.exists() and proc.read_text().rpartition(')')[2].split()[19]==identity['start_ticks'], 'Distillation process exited before completion'
        time.sleep(5)
    tasks=[('schemas',[sys.executable,'-u',str(HERE/'verify_schemas.py')]),
        ('memcheck',['/usr/local/cuda-11.8/bin/compute-sanitizer','--tool','memcheck','--error-exitcode','97',sys.executable,'-u',str(HERE/'verify_schemas.py')]),
        ('initcheck',['/usr/local/cuda-11.8/bin/compute-sanitizer','--tool','initcheck','--error-exitcode','98',sys.executable,'-u',str(HERE/'verify_schemas.py')]),
        ('integrity',[sys.executable,'-u',str(HERE/'integrity.py')])]
    rows=[]
    for name,args in tasks:
        with (ROOT/'logs'/('final_'+name+'.log')).open('w') as out:
            p=subprocess.Popen(args,stdout=out,stderr=subprocess.STDOUT);code=p.wait()
        rows.append(dict(name=name,exit=code));(ROOT/'reports/FINAL_TESTS_STATUS.json').write_text(json.dumps(dict(tasks=rows,complete=False),indent=2))
        print('AFTER_TRAIN',name,code,flush=True)
        if code:raise SystemExit(code)
    (ROOT/'reports/FINAL_TESTS_STATUS.json').write_text(json.dumps(dict(tasks=rows,complete=True),indent=2))


if __name__=='__main__':main()
