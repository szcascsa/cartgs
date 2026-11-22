"""CaRtGS随机／Opacity裁剪对照：固定预算与审计工具。"""
from decimal import Decimal
from pathlib import Path
import hashlib
import json

ROOT = Path(__file__).resolve().parent
SAVED = Path('/home/prmi/zwj/realLOD/saved_experiments')
RATIOS = ['0.01','0.03','0.05','0.08','0.1','0.2','0.3','0.4','0.5','0.6','0.7']
METHODS = ['random','opacity']


def load(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
    temp.replace(path)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def budget(ratio, count):
    return int(Decimal(str(ratio))*count)


def order_points(method, opacity):
    import numpy as np
    if method == 'random':
        return np.random.Generator(np.random.PCG64(0)).permutation(len(opacity))
    if method == 'opacity':
        assert np.isfinite(opacity).all()
        return np.argsort(-opacity,kind='stable')
    raise ValueError(method)
