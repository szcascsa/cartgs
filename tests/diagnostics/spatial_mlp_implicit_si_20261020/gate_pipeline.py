from pathlib import Path
import subprocess,sys,json
ROOT=Path(__file__).resolve().parent
for scene in ['fr1_room','fr1_xyz','fr2_xyz']:
    with open(ROOT/scene/'baseline_driver.log','w')as f:subprocess.run([sys.executable,'baseline.py'],cwd=ROOT/scene,stdout=f,stderr=subprocess.STDOUT,check=True)
(ROOT/'baseline_all_passed.json').write_text(json.dumps(dict(all_passed=True)))
for scene in ['fr1_room','fr1_xyz','fr2_xyz']:
    with open(ROOT/scene/'legacy_driver.log','w')as f:subprocess.run([sys.executable,'legacy_gate.py',scene],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
(ROOT/'legacy_gate.json').write_text(json.dumps(dict(all_passed=True)))
with open(ROOT/'training_driver.log','w')as f:subprocess.run([sys.executable,'run_training.py'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
