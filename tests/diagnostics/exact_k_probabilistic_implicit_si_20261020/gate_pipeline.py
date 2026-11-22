from pathlib import Path
import subprocess,sys,json
ROOT=Path(__file__).resolve().parent;py=sys.executable
def run(cmd,cwd,log):
    with open(ROOT/log,'w')as f:subprocess.run(cmd,cwd=cwd,stdout=f,stderr=subprocess.STDOUT,check=True)
run([str(ROOT/'build/pl_tests'),str(ROOT)],ROOT,'toy.log')
for scene in ['fr1_room','fr1_xyz','fr2_xyz']:
    run([py,'baseline.py'],ROOT/scene,scene+'/baseline_driver.log')
(ROOT/'baseline_all_passed.json').write_text(json.dumps(dict(all_passed=True)))
for scene in ['fr1_room','fr1_xyz','fr2_xyz']:run([py,'legacy_gate.py',scene],ROOT,scene+'/legacy_driver.log')
(ROOT/'legacy_gate.json').write_text(json.dumps(dict(all_passed=True)))
run([py,'run_training.py'],ROOT,'training_driver.log')
