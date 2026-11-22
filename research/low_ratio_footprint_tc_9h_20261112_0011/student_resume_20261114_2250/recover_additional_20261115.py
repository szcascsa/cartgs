"""修复exp2编译后只接续剩余任务，保留失败日志、已完成样本和控制器身份。"""
from pathlib import Path
import json
import subprocess

HERE = Path(__file__).resolve().parent


def main():
    old = json.loads((HERE / 'reports/ADDITIONAL_20261115_CONTROLLER.json').read_text())
    assert old['state'] == 'failed' and old['current_task'] == 'new_exp2_encoded_build'
    source = HERE / 'source_exp2_encoded/exp2_encoded_pixels.cuh'
    text = source.read_text()
    anchor = '  const float value=fminf(.99f,__exp2f(power));'
    assert text.count(anchor) == 1
    source.write_text(text.replace(anchor,
        '  float exponent;asm("ex2.approx.ftz.f32 %0,%1;":"=f"(exponent):"f"(power));\n  const float value=fminf(.99f,exponent);'))
    plan = json.loads((HERE / 'reports/ADDITIONAL_20261115_PLAN.json').read_text())
    remaining = []
    for task in plan['jobs']:
        if any(name.startswith(task['name'] + '_a') for name in old['accepted']):
            continue
        remaining.append(dict(task, name=task['name'] + '_continued'))
    for label in ['grouped_view']:
        remaining.append(dict(name=f'new_{label}_source', measure=False, command=['{python}', '-u', '-B', f'implement_{label}.py']))
        remaining.append(dict(name=f'new_{label}_build', measure=False, command=['{python}', '-u', '-B', f'{label}_runtime.py']))
        for repeat in [1, 2]:
            remaining.append(dict(name=f'new_{label}_r{repeat}', measure=True,
                command=['{python}', '-u', '-B', f'{label}_development.py', '--output', '{job}', '--round', str(repeat)]))
    path = HERE / 'reports/ADDITIONAL_CONTINUED_PLAN.json'
    assert not path.exists()
    path.write_text(json.dumps(dict(name='ADDITIONAL_CONTINUED', jobs=remaining), indent=2))
    with (HERE / 'logs/additional_continued_controller.log').open('x') as log:
        child = subprocess.Popen(['/home/prmi/anaconda3/envs/zwj/bin/python', '-u', '-B', 'run_stage.py', str(path)],
            cwd=HERE, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    print('ADDITIONAL_CONTINUED', child.pid, len(remaining), flush=True)


if __name__ == '__main__':
    main()
