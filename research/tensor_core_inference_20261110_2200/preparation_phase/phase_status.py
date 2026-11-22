from pathlib import Path
import json

HERE=Path(__file__).resolve().parent
for path in sorted((HERE/'reports/jobs').glob('*.json')):
    j=json.loads(path.read_text())
    if j.get('state') not in ['complete','failed']:
        proc=Path('/proc')/str(j.get('child_pid',0))/'stat'
        print(path.name,j['state'],'process',proc.read_text().rpartition(')')[2].split()[:2] if proc.exists() else 'missing')
path=HERE/'reports/FORMAL_CONTROLLER.json'
if path.is_file():
    j=json.loads(path.read_text());print('FORMAL_CONTROLLER',j['state'],'accepted',len(j['accepted']),'rejected',j['rejected'],j.get('error'))
