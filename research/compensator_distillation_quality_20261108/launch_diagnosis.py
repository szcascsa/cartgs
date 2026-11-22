from pathlib import Path
import json,hashlib,subprocess,os
ROOT=Path('/home/prmi/zwj/experiments/compensator_distillation_quality_20261108')
for prefix in ['PRODUCTION','INPUTS','LEGACY_INPUTS','LARGE_INPUTS']:
 path=ROOT/'reports'/(prefix+'_BEFORE.json'); expected=json.loads(path.read_text()); current={}
 for name,value in expected.items():
  p=Path(name); digest=hashlib.sha256(p.read_bytes()).hexdigest()
  current[name]=dict(bytes=p.stat().st_size,sha256=digest) if isinstance(value,dict) else digest
 assert current==expected,prefix
 path.write_text(json.dumps(current,indent=2))
base=Path('/home/prmi/zwj/experiments/compensator_distillation_20261108/models/fr1_student32_image_calibration/best.pt')
(ROOT/'reports/STUDENT_BEFORE.json').write_text(json.dumps({str(base):hashlib.sha256(base.read_bytes()).hexdigest()},indent=2))
log=(ROOT/'logs/diagnosis.log').open('x')
p=subprocess.Popen(['/home/prmi/anaconda3/envs/zwj/bin/python','-u',str(ROOT/'source/research/compensator_distillation_quality_20261108/diagnose.py')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
stat=Path(f'/proc/{p.pid}/stat').read_text().rpartition(')')[2].split()
identity=dict(pid=p.pid,start_ticks=int(stat[19]),boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip())
(ROOT/'reports/DIAGNOSIS_PROCESS.json').write_text(json.dumps(identity,indent=2));print(identity)
