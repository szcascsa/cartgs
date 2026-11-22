"""核验只读输入／部署源码／实际二进制，归档全部原始数值。"""
from common import *
from zipfile import ZipFile,ZIP_DEFLATED
import tarfile


def count(path):
    with Path(path).open() as f:return sum(1 for _ in csv.DictReader(f))


def main():
    assert json.loads((ROOT/'STATUS.json').read_text())['state']=='complete'
    r=ROOT/'reports';assert json.loads((r/'QUALITY_GATES.json').read_text())['passed']
    assert len(json.loads((r/'SUMMARY.json').read_text()))==28
    assert len(json.loads((r/'PROFILE_SUMMARY.json').read_text()))==28
    for name,expected in [('reports/TIMINGS.csv',16632),('reports/CUDA_EVENTS.csv',672),('reports/COMPONENTS.csv',1600),
                          ('evaluation/PER_FRAME.csv',1584),('evaluation/PERCEPTUAL.csv',132)]:assert count(ROOT/name)==expected,name
    for v in json.loads((r/'PAIRED_SPEED.json').read_text()):assert v['same_K_fraction']==v['same_candidate_N_fraction']==1.
    for prefix in ['PRODUCTION','INPUTS']:
        before=json.loads((r/(prefix+'_BEFORE.json')).read_text());after={}
        for n,v in before.items():
            p=Path(n);after[n]=dict(bytes=p.stat().st_size,sha256=sha(p)) if isinstance(v,dict) else sha(p)
        assert before==after,prefix
        save(r/(prefix+'_AFTER.json'),after)
    source=json.loads((r/'SOURCE_HASHES.json').read_text());assert all(sha(ROOT/'source'/n)==h for n,h in source.items())
    environment=json.loads((r/'ENVIRONMENT.json').read_text())
    for binary in environment['binaries'].values():assert sha(binary['path'])==binary['sha256']
    assert (r/'DEPLOYMENT_LOCAL500_RATIO50_REPORT.md').is_file()
    assert len(list((r/'figures').glob('*.png')))==3 and len(list((r/'figures').glob('*.pdf')))==3
    assert not [p for p in (ROOT/'evaluation').rglob('*') if p.suffix.lower() in ['.png','.jpg','.jpeg','.webp']]
    source.update({str(p.relative_to(ROOT/'source')):sha(p) for p in HERE.iterdir() if p.suffix in ['.py','.md']})
    save(r/'SOURCE_FINAL_HASHES.json',source)
    with tarfile.open(ROOT/'source_final.tar.gz','w:gz') as z:
        for n in sorted(source):z.add(ROOT/'source'/n,arcname=n,recursive=False)
    value=json.loads((r/'FINAL_INTEGRITY.json').read_text());assert value['passed'] and value['protected_production_files']==604
    value.update(delivery_verified=True,quality_gates_passed=True,same_K_and_candidate_N=True,figures=3,source_final_sha256=sha(ROOT/'source_final.tar.gz'))
    save(r/'FINAL_INTEGRITY.json',value)
    save(r/'COMPLETION.json',dict(state='complete',passed=True,report='reports/DEPLOYMENT_LOCAL500_RATIO50_REPORT.md'))
    archive=ROOT/'DEPLOYMENT_LOCAL500_RATIO50_RESULTS.zip';assert not archive.exists()
    with ZipFile(archive,'w',ZIP_DEFLATED) as z:
        for folder in ['reports','evaluation','logs']:
            for p in (ROOT/folder).rglob('*'):
                if p.is_file() and p.suffix not in ['.lock','.gz']:z.write(p,p.relative_to(ROOT))
        z.write(ROOT/'source_final.tar.gz','source_final.tar.gz');z.write(ROOT/'STATUS.json','STATUS.json')
    save(r/'ARCHIVE_SHA256.json',dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive)))
    print('DELIVERY_VERIFIED',archive,archive.stat().st_size,flush=True)


if __name__=='__main__':main()
