"""评价完成后生成中文图表和可下载结果包，不改写任何旧结果。"""
import fcntl
import subprocess
import sys
import time
import zipfile
from common import ROOT,load,write,sha


def main():
    lock=(ROOT/'finish.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        state=load(ROOT/'EXPERIMENTS.json')
        if state['phase']=='held':
            print('队列出现错误并保留现场。',state.get('error'),flush=True);return
        if state['phase']=='finished':break
        time.sleep(30)
    assert load(ROOT/'reports/VERIFIED.json')['complete']
    assert subprocess.run([sys.executable,str(ROOT/'plot.py')],cwd=ROOT).returncode==0
    figures=list((ROOT/'figures').glob('*.png'))
    assert len(figures)==len(list((ROOT/'figures').glob('*.svg')))==32
    receipt=dict(complete=True,evaluations=704,figures=32,
                 figure_sha256={p.name:sha(p) for p in (ROOT/'figures').iterdir() if p.is_file()})
    write(ROOT/'reports/DELIVERY_VERIFIED.json',receipt)
    with zipfile.ZipFile(ROOT/'PRUNING_BASELINE_RESULTS.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for directory in ['reports','figures']:
            for path in (ROOT/directory).rglob('*'):
                if path.is_file():archive.write(path,path.relative_to(ROOT))
        for path in ROOT.glob('*.py'):archive.write(path,path.name)
        for name in ['EXPERIMENTS.json','PROTOCOL.md','PROTECTED_BEFORE.json',
                     'PROTECTED_AFTER.json','SOURCE_MANIFEST.json','SMOKE_PASSED.json']:
            archive.write(ROOT/name,name)
    write(ROOT/'FINISHED.json',dict(complete=True,result_archive=str(ROOT/'PRUNING_BASELINE_RESULTS.zip'),
                                  sha256=sha(ROOT/'PRUNING_BASELINE_RESULTS.zip')))
    print('全部结果、32张曲线图和压缩包已生成。',flush=True)


if __name__=='__main__':main()
