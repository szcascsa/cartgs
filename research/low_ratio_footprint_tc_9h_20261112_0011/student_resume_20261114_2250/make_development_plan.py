from pathlib import Path
import json

HERE = Path(__file__).resolve().parent
jobs = [dict(name=f'student_development_r{iteration}', measure=True,
             command=['{python}', '-u', '-B', 'development.py', '--output', '{job}', '--round', str(iteration)])
        for iteration in [1, 2]]
path = HERE / 'reports/DEVELOPMENT_PLAN.json'
assert not path.exists()
path.write_text(json.dumps(dict(name='DEVELOPMENT', jobs=jobs), indent=2))
