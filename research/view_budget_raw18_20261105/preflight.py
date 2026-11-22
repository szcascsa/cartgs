"""创建独立实验的保护清单和固定输入配置，不触碰生产文件。"""
import hashlib
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[3]
OLD=Path('/home/prmi/zwj/experiments/view_budget_residual_full_20261105')
IMPLEMENTATION=Path('/home/prmi/zwj/experiments/view_budget_residual_20261105')


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
    return h.hexdigest()


def main():
    assert ROOT==Path('/home/prmi/zwj/experiments/view_budget_raw18_20261105')
    for d in ['reports','configs','logs','runs','evaluation','evaluation/reports']:(ROOT/d).mkdir(exist_ok=True)
    assert not (ROOT/'reports/PRODUCTION_BEFORE.json').exists()
    assert json.loads((OLD/'reports/FINAL_INTEGRITY.json').read_text())['passed']
    names=json.loads((OLD/'reports/PRODUCTION_BEFORE.json').read_text())
    before={n:dict(bytes=Path(n).stat().st_size,sha256=sha(Path(n))) for n in names}
    (ROOT/'reports/PRODUCTION_BEFORE.json').write_text(json.dumps(before,indent=2))
    source=json.loads((ROOT/'reports/SOURCE_HASHES.json').read_text())
    assert all(sha(ROOT/'source'/n)==h for n,h in source.items())
    for p in (IMPLEMENTATION/'configs').iterdir():
        if p.is_file() and p.suffix=='.yaml':shutil.copyfile(p,ROOT/'configs'/p.name)
    shutil.copyfile(ROOT/'source/research/view_budget_raw18_20261105/PROTOCOL.md',ROOT/'reports/PROTOCOL.md')
    print(json.dumps(dict(protected_files=len(before),source_files=len(source),free_GiB=shutil.disk_usage(ROOT).free/1024**3)),flush=True)


if __name__=='__main__':main()
