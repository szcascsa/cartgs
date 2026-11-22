"""延续现场，按用户最新指令记录九小时预算及只用Student的范围。"""
from pathlib import Path
from datetime import datetime, timedelta
import hashlib
import json
import os
import shutil

HERE = Path(__file__).resolve().parent
PARENT = HERE.parent


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    for name in ['reports', 'logs', 'build', 'figures']:
        (HERE / name).mkdir(exist_ok=True)
    assert not (HERE / 'reports/PROTOCOL.json').exists()
    assert shutil.disk_usage(HERE).free >= 5 * (1 << 30)
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    audit = []
    for path in (PARENT / 'reports').glob('*CONTROLLER*.json'):
        state = json.loads(path.read_text())
        proc = Path(f'/proc/{state.get("pid", 0)}/stat')
        alive = False
        if proc.exists():
            fields = proc.read_text().rpartition(')')[2].split()
            alive = (state.get('boot_id') == boot and fields[0] != 'Z'
                     and fields[19] == str(state.get('start_ticks')))
        assert not alive, (path, '已有研究控制器仍运行')
        audit.append(dict(path=str(path), state=state.get('state'), alive=alive))
    start = datetime.fromisoformat('2026-11-14T22:52:35+08:00')
    protocol = dict(start=start.isoformat(), deadline=(start + timedelta(hours=9)).isoformat(),
        final_validation_reserve=(start + timedelta(hours=8)).isoformat(),
        previous_pause=json.loads((PARENT / 'reports/USER_PAUSE_20261112_1109.json').read_text()),
        budget_authority='用户最新明确指令：实验从现在开始计时9个小时',
        phase='student_joint_inference', stage_transition=str(PARENT / 'reports/PHASE_TRANSITION.json'),
        models='仅冻结Student Compensator及对应checkpoint的原Selector',
        quality=dict(mean_psnr_loss=.20, scene_condition_psnr_loss=.50),
        tune=['Replica__office2__rgbd', 'Replica__room0__rgbd',
              'TUM_RGBD__freiburg1_xyz__rgbd', 'VECtor__corner-slow__stereo'],
        network_tune_available=['Replica__office2__rgbd', 'Replica__room0__rgbd'],
        no_images=True, no_training=True, no_teacher_forward=True,
        no_commit_push_deploy=True, controllers=audit)
    (HERE / 'reports/PROTOCOL.json').write_text(json.dumps(protocol, indent=2, ensure_ascii=False))
    previous = {}
    for name in ['INPUT_SHA256.json', 'PROTECTED_SHA256.json']:
        manifest = json.loads((PARENT / 'reports' / name).read_text())
        mismatches = [str(path) for path, value in manifest.items() if sha(path) != value]
        assert not mismatches, (name, mismatches)
        previous[name] = dict(files=len(manifest), passed=True)
    frozen = json.loads((PARENT / 'reports/FROZEN_RENDER.json').read_text())
    assert sha(frozen['binary']) == frozen['binary_sha256']
    for name, value in frozen['sources'].items():
        assert sha(PARENT / name) == value, name
    identities = {}
    controller = json.loads((PARENT / 'reports/JOINT_FORMAL_CONTROLLER.json').read_text())
    assert controller['state'] == 'complete'
    for job in controller['accepted']:
        path = PARENT / 'reports' / job / 'IDENTITIES.json'
        if path.exists():
            for row in json.loads(path.read_text()):
                if row['model'] == 'student':
                    assert sha(row['student']) == row['student_sha256']
                    identities[row['scene']] = row
    assert len(identities) == 18
    (HERE / 'reports/EXISTING_STUDENTS.json').write_text(json.dumps(identities, indent=2))
    protected = {}
    for folder in ['joint_source_pose', 'joint_source_v2']:
        for path in (PARENT / folder).rglob('*'):
            if path.is_file():
                protected[str(path)] = sha(path)
    for folder in ['joint_build_pose', 'joint_build_v2']:
        for path in (PARENT / folder).glob('*.so'):
            protected[str(path)] = sha(path)
    for folder in ['/home/prmi/zwj/realLOD/python', '/home/prmi/zwj/realLOD/src',
                   '/home/prmi/zwj/realLOD/improvements/selector']:
        root = Path(folder)
        for path in root.rglob('*'):
            if path.is_file() and path.suffix in ['.py', '.cpp', '.cu', '.cuh', '.h', '.yaml']:
                protected[str(path)] = sha(path)
    env = Path('/home/prmi/anaconda3/envs/zwj/conda-meta')
    for path in env.glob('*.json'):
        protected[str(path)] = sha(path)
    (HERE / 'reports/PROTECTED_SHA256.json').write_text(json.dumps(protected, indent=2))
    (HERE / 'reports/PREFLIGHT.json').write_text(json.dumps(dict(passed=True,
        old_manifests=previous, frozen_render_binary=frozen['binary'], protected_files=len(protected),
        identity_groups=len(identities), boot_id=boot), indent=2))
    shutil.copytree(PARENT / 'joint_source_pose', HERE / 'source')
    for name in ['launch.py', 'job_runner.py', 'run_stage.py']:
        shutil.copy2(PARENT / name, HERE / name)
    print('STUDENT_RESUME_READY', protocol['start'], protocol['deadline'], len(identities), len(protected))


if __name__ == '__main__':
    main()
