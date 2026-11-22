"""核对全部质量／时间记录、只读输入与运行库，归档交付。"""
from common import *
import math
import tarfile
from zipfile import ZipFile,ZIP_DEFLATED


def read(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def main():
    assert json.loads((ROOT/'STATUS.json').read_text())['state']=='complete'
    r=ROOT/'reports';assert json.loads((r/'QUALITY_GATES.json').read_text())['passed']
    for name,expected in [('reports/TIMINGS.csv',11880),('reports/CUDA_EVENTS.csv',480),('evaluation/PER_FRAME.csv',7128),
                          ('evaluation/OUTPUT_DIFFERENCES.csv',3168),('evaluation/PAIRED_PER_FRAME.csv',3168)]:assert len(read(ROOT/name))==expected,name
    for v in read(r/'TIMINGS.csv'):
        assert abs(float(v['prepare_ms'])+float(v['render_ms'])-float(v['total_ms']))<1e-8
        assert int(v['K'])==min(int(v['requested_K']),int(v['eligible_N']))
        if v['case']!='full':assert int(v['requested_K'])==math.floor(float(v['ratio'])*int(v['candidate_N']))
    assert len(json.loads((r/'SPEED_SUMMARY.json').read_text()))==20
    assert len(json.loads((r/'QUALITY_SUMMARY.json').read_text()))==27
    for prefix in ['PRODUCTION','INPUTS']:
        before=json.loads((r/(prefix+'_BEFORE.json')).read_text());after={}
        for n,v in before.items():after[n]=dict(bytes=Path(n).stat().st_size,sha256=sha(n)) if isinstance(v,dict) else sha(n)
        assert before==after,prefix;save(r/(prefix+'_AFTER.json'),after)
    source=json.loads((r/'RUNTIME_SOURCE_HASHES.json').read_text());assert all(sha(ROOT/'source'/n)==h for n,h in source.items())
    env=json.loads((r/'ENVIRONMENT.json').read_text())
    for binary in env['binaries'].values():assert sha(binary['path'])==binary['sha256']
    assert (r/'DEPLOYMENT_LOCAL_RATIO_SWEEP_REPORT.md').exists()
    assert len(list((r/'figures').glob('*.png')))==len(list((r/'figures').glob('*.pdf')))==2
    assert not [p for p in (ROOT/'evaluation').rglob('*') if p.suffix.lower() in ['.png','.jpg','.jpeg','.webp']]
    source.update({str(p.relative_to(ROOT/'source')):sha(p) for p in HERE.iterdir() if p.suffix in ['.py','.md']})
    save(r/'SOURCE_FINAL_HASHES.json',source)
    with tarfile.open(ROOT/'source_final.tar.gz','w:gz') as z:
        for n in sorted(source):z.add(ROOT/'source'/n,arcname=n,recursive=False)
    integrity=json.loads((r/'FINAL_INTEGRITY.json').read_text());assert integrity['passed'] and integrity['protected_production_files']==604
    integrity.update(delivery_verified=True,quality_gates_passed=True,budget_verified=True,figures=2,source_final_sha256=sha(ROOT/'source_final.tar.gz'))
    save(r/'FINAL_INTEGRITY.json',integrity);save(r/'COMPLETION.json',dict(state='complete',passed=True,report='reports/DEPLOYMENT_LOCAL_RATIO_SWEEP_REPORT.md'))
    archive=ROOT/'DEPLOYMENT_LOCAL_RATIO_SWEEP_RESULTS.zip';assert not archive.exists()
    with ZipFile(archive,'w',ZIP_DEFLATED) as z:
        for folder in ['reports','evaluation','logs']:
            for p in (ROOT/folder).rglob('*'):
                if p.is_file() and p.suffix not in ['.gz','.lock']:z.write(p,p.relative_to(ROOT))
        z.write(ROOT/'STATUS.json','STATUS.json');z.write(ROOT/'source_final.tar.gz','source_final.tar.gz')
    save(r/'ARCHIVE_SHA256.json',dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive)))
    print('FINAL_INTEGRITY_PASS',archive,archive.stat().st_size,flush=True)


if __name__=='__main__':main()
