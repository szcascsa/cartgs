"""局部推理消融的冻结来源与结果工具。"""
from pathlib import Path
import csv
import hashlib
import json

ROOT = Path(__file__).resolve().parent
SOURCE = Path('/home/prmi/zwj/realLOD/saved_experiments/cartgs_pruning_baselines_20261112/reports/METHODS_ALL_METRICS.csv')
SOURCE_SHA = 'e56cc01dfe0dcfb1613f06d4211ce5d7e137b6c6ce341c2667db200cc027c891'
RATIOS = [.01, .03, .05, .08, .1, .2, .3, .4, .5, .6, .7]
TRAIN = [.01, .05, .1, .3, .5, .7]
VARIANTS = ['selector_no_comp', 'random_with_comp']


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def records():
    assert sha(SOURCE) == SOURCE_SHA, '原32组指标来源发生变化'
    rows = [row for row in csv.DictReader(SOURCE.open(encoding='utf-8-sig'))
            if row['method'] == 'ours_view_ratio']
    assert len(rows) == 352 and len({row['group'] for row in rows}) == 32
    return rows
