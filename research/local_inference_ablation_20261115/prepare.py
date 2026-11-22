"""冻结32组指标与对应评价Python；原模型和CUDA扩展仅只读复用。"""
from pathlib import Path
import json
import shutil
from common import ROOT, SOURCE, SOURCE_SHA, records, load, sha, write


def main():
    rows = records()
    protected = {str(SOURCE): SOURCE_SHA}
    sources = {}
    for group in dict.fromkeys(row['group'] for row in rows):
        items = [row for row in rows if row['group'] == group]
        launch = load(Path(items[0]['source'])/'launch.json')['command']
        original = Path(next(arg for arg in launch if arg.endswith('/eval_joint_entry.py'))).parent
        if original.name not in sources:
            source = original/'source/python'
            destination = ROOT/'snapshots'/original.name/'python'
            for path in source.rglob('*.py'):
                if '__pycache__' in path.parts:
                    continue
                output = destination/path.relative_to(source)
                output.parent.mkdir(parents=True, exist_ok=True)
                if output.exists():
                    assert sha(path) == sha(output)
                else:
                    shutil.copyfile(path, output)
                protected[str(path)] = sha(path)
            runtime = original/'evaluation_runtime'
            protected.update({str(path): sha(path) for path in runtime.rglob('*') if path.is_file()})
            sources[original.name] = dict(source=str(source), snapshot=str(destination), runtime=str(runtime))
        checkpoint = Path(launch[launch.index('--selector')+1]).parent
        protected.update({str(path): sha(path) for path in checkpoint.rglob('*') if path.is_file()})
        for item in items:
            output = Path(item['source'])
            protected.update({str(output/name): sha(output/name) for name in
                ['metric_summary.json','selection_hashes.json','launch.json','per_frame_metrics.csv',
                 'anchor_frustum_per_view.csv','evaluation_frames.json','image_preprocessing.json']})
    write(ROOT/'REFERENCE_RECORDS.json', rows)
    write(ROOT/'PROTECTED_BEFORE.json', protected)
    write(ROOT/'PREPARED.json', dict(groups=32, baseline_records=352, new_evaluations=704,
        source=str(SOURCE), source_sha256=SOURCE_SHA, snapshots=sources, protected_files=len(protected),
        reference_total_views=sum(int(row['frames']) for row in rows)//11,
        reference_selection_checks=sum(int(row['frames']) for row in rows), training_steps=0))
    print('PREPARED', 32, 704, 'PROTECTED', len(protected), flush=True)


if __name__ == '__main__':
    main()
