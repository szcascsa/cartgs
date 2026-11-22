"""同盘迁移完整结果批次，旧路径保留符号链接，核验模型和评价内容。"""
from pathlib import Path
from collections import Counter
import hashlib
import json
import os
import time

ROOT = Path('/home/prmi/zwj/realLOD')
SAVED = ROOT/'saved_experiments'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    batches = [('dual_view_li_best_full_20261107', 'baseline'),
               ('dual_view_li_vector_best_full_20261107', 'baseline'),
               ('vanilla_all_datasets_20261031_2011', 'vanilla')]
    assert all(not (ROOT/name).exists() for name in ['baseline', 'vanilla'])
    protected = {}
    for name, category in batches:
        source = SAVED/name
        assert source.is_dir() and not source.is_symlink()
        files = []
        if category == 'baseline':
            state = json.loads((source/'LOD_PIPELINE.json').read_text())
            assert state['phase'] == 'finished'
            for job in state['jobs']:
                assert job['state'] == 'complete'
                model = Path(job['trained_model'])
                files.extend(p for p in model.rglob('*') if p.is_file())
                for evaluation in job['evaluations']:
                    assert evaluation['state'] == 'complete'
                    files.append(Path(evaluation['result']['output'])/'metric_summary.json')
            files.extend([source/'LOD_PIPELINE.json', source/'reports/LOD_METRICS.csv'])
        else:
            results = json.loads((source/'results.json').read_text())
            assert len(results) == 101 and all(item['state'] == 'complete' for item in results)
            for item in results:
                files.extend(p for p in Path(item['checkpoint']).rglob('*') if p.is_file())
            files.extend([source/'summary.json', source/'results.json', source/'summary.csv'])
            files.extend(p for p in (source/'reports').rglob('*') if p.is_file())
        protected[name] = {str(path): dict(sha256=sha(path), inode=path.stat().st_ino,
                                          size=path.stat().st_size) for path in set(files)}
    for category in ['baseline', 'vanilla']:
        (ROOT/category).mkdir()
    receipt = dict(started_unix=time.time(), operation='同一文件系统迁移完整批次，旧路径保留链接',
                   protected=protected, moved=[])
    audit = ROOT/'baseline/RESULTS_RELOCATION.json'
    audit.write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
    for name, category in batches:
        source = SAVED/name
        destination = ROOT/category/name
        assert source.stat().st_dev == destination.parent.stat().st_dev
        inode = source.stat().st_ino
        source.rename(destination)
        try:
            source.symlink_to(destination, target_is_directory=True)
        except Exception:
            destination.rename(source)
            raise
        assert destination.stat().st_ino == inode and source.resolve() == destination
        for path, expected in protected[name].items():
            file = Path(path)
            assert file.stat().st_ino == expected['inode'] and file.stat().st_size == expected['size']
            assert sha(file) == expected['sha256'], file
        receipt['moved'].append(dict(old=str(source), new=str(destination),
                                     old_path_compatible=True, root_inode=inode))
        audit.write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
    for dataset in ['Replica', 'TUM_RGBD', 'VECtor']:
        batch = batches[1][0] if dataset == 'VECtor' else batches[0][0]
        (ROOT/'baseline'/dataset).symlink_to(Path(batch)/'runs'/dataset, target_is_directory=True)
    vanilla = ROOT/'vanilla'/batches[2][0]
    for label, folder in [('Replica', 'replica'), ('TUM_RGBD', 'tum'), ('VECtor', 'vector'), ('3DLite', '3dlite')]:
        (ROOT/'vanilla'/label).symlink_to(Path(vanilla.name)/'jobs'/folder, target_is_directory=True)
    receipt.update(passed=True, finished_unix=time.time(), baseline_groups=32,
                   baseline_evaluations=736, vanilla_complete_runs=101, vanilla_skipped_runs=1,
                   protected_files=sum(len(values) for values in protected.values()))
    audit.write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
    (ROOT/'vanilla/RESULTS_RELOCATION.json').write_text(json.dumps(receipt['moved'][-1], indent=2))
    print(json.dumps({key: value for key, value in receipt.items() if key != 'protected'}, ensure_ascii=False))


if __name__ == '__main__':
    main()
