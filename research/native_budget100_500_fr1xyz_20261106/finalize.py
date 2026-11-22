"""交付前核对三栏图的数据、版本、计数与历史输入。"""
from common import *
from zipfile import ZipFile,ZIP_DEFLATED
import tarfile


def read(path):
    with Path(path).open() as f:return list(csv.DictReader(f))


def main():
    assert json.loads((ROOT/'STATUS.json').read_text())['state']=='complete'
    r=ROOT/'reports'
    for name,expected in [('reports/TIMINGS.csv',11880),('evaluation/OURS_PER_VIEW.csv',3960),('evaluation/CARTGS_PER_VIEW.csv',792),
        ('reports/PER_VIEW_COMPARISON.csv',4752),('reports/PLOT_DATA_RAW_AND_SMOOTHED.csv',4752),('reference/CARTGS_TIMINGS_ONLY.csv',2376)]:assert len(read(ROOT/name))==expected,name
    for v in read(r/'TIMINGS.csv'):
        assert int(v['K'])==int(v['budget']) and int(v['shortfall'])==0
        assert abs(float(v['prepare_ms'])+float(v['render_ms'])-float(v['total_ms']))<1e-8
    for v in read(ROOT/'evaluation/OURS_PER_VIEW.csv'):assert 0<=int(v['contributing_K'])<=int(v['projected_K'])<=int(v['K'])==int(v['budget'])
    summary=json.loads((r/'SUMMARY.json').read_text());assert len(summary)==6
    p=json.loads((r/'COUNT_RENDERER_PARITY.json').read_text());assert p['passed'] and len(p['cases'])==15
    protocol=json.loads((r/'PLOT_PROTOCOL.json').read_text());assert protocol['columns']==3 and protocol['psnr']['sigma']==35 and protocol['psnr']['window']==211
    for prefix in ['PRODUCTION','INPUTS']:
        before=json.loads((r/(prefix+'_BEFORE.json')).read_text());after={}
        for n,v in before.items():after[n]=dict(bytes=Path(n).stat().st_size,sha256=sha(n)) if isinstance(v,dict) else sha(n)
        assert before==after,prefix;save(r/(prefix+'_AFTER.json'),after)
    history=json.loads((r/'HISTORICAL_INPUTS.json').read_text());assert all(sha(n)==h for n,h in history.items())
    source=json.loads((r/'RUNTIME_SOURCE_HASHES.json').read_text());assert all(sha(ROOT/'source'/n)==h for n,h in source.items())
    env=json.loads((r/'ENVIRONMENT.json').read_text());assert sha(env['binary']['path'])==env['binary']['sha256'] and 'fp32_v4' in env['binary']['path']
    assert (r/'NATIVE_FIXED100_500_REPORT.md').exists()
    for n in ['NATIVE_FIXED100_500_THREE_COLUMNS.png','NATIVE_FIXED100_500_THREE_COLUMNS.pdf','NATIVE_FIXED100_500_THREE_COLUMNS.svg','NATIVE_FIXED100_500_RAW.png']:
        assert (r/'figures'/n).is_file()
    assert not [p for p in (ROOT/'evaluation').rglob('*') if p.suffix.lower() in ['.png','.jpg','.jpeg','.webp']]
    source.update({str(p.relative_to(ROOT/'source')):sha(p) for p in HERE.iterdir() if p.suffix in ['.py','.md']});save(r/'SOURCE_FINAL_HASHES.json',source)
    with tarfile.open(ROOT/'source_final.tar.gz','w:gz') as z:
        for n in sorted(source):z.add(ROOT/'source'/n,arcname=n,recursive=False)
    integrity=json.loads((r/'FINAL_INTEGRITY.json').read_text());assert integrity['passed'] and integrity['protected_production_files']==604
    integrity.update(delivery_verified=True,plot_columns=3,psnr_sigma_frames=35,raw_data_preserved=True,source_final_sha256=sha(ROOT/'source_final.tar.gz'))
    save(r/'FINAL_INTEGRITY.json',integrity);save(r/'COMPLETION.json',dict(state='complete',passed=True,figure='reports/figures/NATIVE_FIXED100_500_THREE_COLUMNS.png',report='reports/NATIVE_FIXED100_500_REPORT.md'))
    archive=ROOT/'NATIVE_FIXED100_500_RESULTS.zip';assert not archive.exists()
    with ZipFile(archive,'w',ZIP_DEFLATED) as z:
        for folder in ['reports','evaluation','reference','logs']:
            for p in (ROOT/folder).rglob('*'):
                if p.is_file() and p.suffix not in ['.lock','.gz']:z.write(p,p.relative_to(ROOT))
        z.write(ROOT/'source_final.tar.gz','source_final.tar.gz');z.write(ROOT/'STATUS.json','STATUS.json')
    save(r/'ARCHIVE_SHA256.json',dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive)))
    print('FINAL_INTEGRITY_PASS',archive,archive.stat().st_size,flush=True)


if __name__=='__main__':main()
