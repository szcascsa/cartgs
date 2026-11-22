"""本地交付核验：压缩包、逐文件SHA、生产状态；不训练或修改生产。"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    checks = json.loads((ROOT / 'archive_integrity.json').read_text())
    for name, expected in checks.items():
        path = ROOT / name
        assert path.stat().st_size == expected['bytes'], name
        assert sha(path) == expected['sha256'], name
        with tarfile.open(path) as stream:
            for member in stream.getmembers():
                assert (ROOT / member.name).resolve().is_relative_to(ROOT), member.name
                assert member.isfile() or member.isdir(), member.name
            stream.extractall(ROOT, filter='data')
    manifest = json.loads((ROOT / 'artifact_manifest.json').read_text())
    for item in manifest['evidence'] + manifest['final_models']:
        path = ROOT / item['path']
        assert path.stat().st_size == item['bytes'], item['path']
        assert sha(path) == item['sha256'], item['path']
    start = json.loads((ROOT / 'research_state.json').read_text())
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()
    assert head == start['head']
    assert not subprocess.check_output(['git', 'status', '--porcelain', '--untracked-files=no'], cwd=REPO, text=True).strip()
    assert not subprocess.check_output(['git', 'diff', '--cached', '--name-only'], cwd=REPO, text=True).strip()
    for name, expected in start['production_sha256'].items():
        assert sha(REPO / name) == expected, name
    required = ['NEURAL_V1_RESEARCH_REPORT.md', 'NEURAL_V1_EXPERIMENTS.csv',
                'NEURAL_V1_BEST_ARCHITECTURE.md', 'NEURAL_V2_RESEARCH_PROPOSAL.md',
                'endpoint_support_probe.csv', 'NEURAL_V1_TRAINABLE_PARAMETERS.csv']
    assert all((ROOT / name).stat().st_size > 0 for name in required)
    now = datetime.now(timezone.utc)
    result = dict(utc=now.isoformat(), elapsed_minutes=(now-datetime.fromisoformat(start['started_utc'])).total_seconds()/60, head=head,
                  evidence_files=len(manifest['evidence']), formal_models=len(manifest['final_models']),
                  all_sha_pass=True, production_unchanged=True, staged_clean=True,
                  commit_stage_push=False, archives=checks)
    (ROOT / 'delivery_verification.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
