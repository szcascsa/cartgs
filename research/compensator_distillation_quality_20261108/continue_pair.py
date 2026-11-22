"""在新测试集评价前，把两组匹配预算同时扩展至 6000 步。"""
from paired_runs import *
import shutil


def main():
    status = json.loads((ROOT/'reports/PAIRED_RUNS_STATUS.json').read_text())
    assert status['state'] == 'all_complete'
    protocol_path = ROOT/'reports/EXPERIMENT_PROTOCOL.json'
    protocol = json.loads(protocol_path.read_text())
    protocol['stage1_steps'] = 2400
    protocol['steps_per_arm'] = 6000
    protocol['budget_amendment'] = dict(reason='弱残差候选在2100步继续改善验证图像MSE，匹配扩展两组预算，尚未评价新测试集',
        before_fresh_test=True, time=time.time())
    protocol_path.write_text(json.dumps(protocol, indent=2, ensure_ascii=False))
    for name, weight in [('control', 1.), ('residual001', .01)]:
        output = ROOT/'models'/name
        shutil.copy2(output/'best.pt', output/'stage1_best.pt')
        shutil.copy2(output/'last.pt', output/'stage1_last.pt')
        command = [PYTHON, '-u', str(HERE/'run.py'), 'train', TEACHER, '--dataset', DATASET,
            '--output', str(output), '--resume', str(output/'last.pt'), '--steps', '6000', '--lr', '.0002',
            '--image-every', '1', '--image-weight', '200', '--residual-weight', str(weight),
            '--validation-metric', 'image_mse', '--validate-every', '300', '--seed', '0']
        with (ROOT/'logs'/f'{name}_continue.log').open('x') as log:
            p = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
            state = dict(state='continuing', experiment=name, pid=p.pid,
                start_ticks=int(Path(f'/proc/{p.pid}/stat').read_text().rpartition(')')[2].split()[19]),
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                started=time.time(), command=command)
            save(state)
            code = p.wait()
            state.update(state='complete' if code == 0 else 'failed', returncode=code, ended=time.time())
            save(state)
            if code:
                raise SystemExit(code)
    save(dict(state='all_complete', steps_per_arm=6000, experiments=['control', 'residual001'], images_saved=0))
    print('PAIRED_CONTINUATION_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
