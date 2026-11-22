"""引言图交付前核验数据、原版曲线和受保护文件。"""
from common import *
from collections import defaultdict
import math
import tarfile
from zipfile import ZipFile,ZIP_DEFLATED


def read(path):
    with Path(path).open(encoding='utf-8') as f:return list(csv.DictReader(f))


def main():
    r=ROOT/'reports';assert json.loads((ROOT/'STATUS.json').read_text())['state']=='complete'
    rows=read(r/'PAPER_INTRO_PER_VIEW.csv');quality=read(ROOT/'evaluation/OURS_PER_VIEW.csv');times=read(r/'TIMINGS.csv');all_times=read(r/'ALL_TIMINGS.csv')
    assert len(rows)==4752 and len(quality)==3960 and len(times)==14256
    assert json.loads((r/'ANALYSIS_VALIDATION.json').read_text())['passed']
    protocol=json.loads((r/'PLOT_PROTOCOL.json').read_text());assert protocol['grid']==[2,3] and protocol['legend_cell']==[2,2] and not protocol['timing']['smoothing'] and protocol['psnr']['sigma']==35
    attempts=json.loads((r/'TIMING_ATTEMPTS.json').read_text());passed=[v for v in attempts if v['accepted']]
    assert len(passed)==3 and {v['repeat'] for v in passed}=={0,1,2}
    for v in passed:assert v['coverage'] and not any(s['external_active'] for s in v['samples'])
    accepted_attempts={v['attempt'] for v in passed};group=defaultdict(list);keys=set()
    for v in times:
        key=(v['method'],int(v['frame']),int(v['repeat']));assert key not in keys;keys.add(key)
        assert v['accepted']=='True' and int(v['attempt']) in accepted_attempts
        assert all(math.isfinite(float(v[k])) and float(v[k])>=0 for k in ['prepare_ms','render_ms','total_ms'])
        assert abs(float(v['prepare_ms'])+float(v['render_ms'])-float(v['total_ms']))<1e-9;group[v['method'],int(v['frame'])].append(v)
    old={(v['method'],int(v['frame'])):v for v in read(ROOT/'reference/PREVIOUS_PER_VIEW.csv')}
    for v in rows:
        assert 0<=int(v['contributing_K'])<=int(v['projected_K'])<=int(v['K'])
        if v['method']=='cartgs_full':
            orig=old[v['method'],int(v['frame'])]
            for key in ['K','contributing_K','psnr','total_ms']:assert float(v[key])==float(orig[key])
        else:
            a=group[v['method'],int(v['frame'])];assert len(a)==3 and {int(x['repeat']) for x in a}=={0,1,2}
            assert abs(sum(float(x['total_ms']) for x in a)/3-float(v['total_ms']))<1e-12
        if v['method'].endswith('full'):assert not v['network_ratio'] and not v['actual_retention_ratio']
        else:
            assert int(v['K'])==min(int(v['requested_K']),int(v['eligible_N']))
            if v['method'].startswith('fixed_'):assert float(v['network_ratio'])==min(1.,int(v['requested_K'])/int(v['candidate_N']))
    parity=json.loads((r/'RENDERER_PARITY.json').read_text());assert parity['passed'] and len(parity['cases'])==15
    for prefix in ['PRODUCTION','INPUTS']:
        before=json.loads((r/(prefix+'_BEFORE.json')).read_text());after={n:dict(bytes=Path(n).stat().st_size,sha256=sha(n)) if isinstance(v,dict) else sha(n) for n,v in before.items()}
        assert before==after,prefix;save(r/(prefix+'_AFTER.json'),after)
    for lang in ['CN','EN']:
        for ext in ['png','pdf','svg']:assert (r/'figures'/f'PAPER_INTRO_{lang}.{ext}').is_file()
    assert not [p for p in (ROOT/'evaluation').rglob('*') if p.suffix.lower() in ['.png','.jpg','.jpeg','.webp']]
    assert (r/'PAPER_INTRO_REPORT.md').is_file()
    source={str(p.relative_to(ROOT/'source')):sha(p) for p in (ROOT/'source').rglob('*') if p.is_file()};save(r/'SOURCE_FINAL_HASHES.json',source)
    with tarfile.open(ROOT/'source_final.tar.gz','w:gz') as z:
        for n in sorted(source):z.add(ROOT/'source'/n,arcname=n,recursive=False)
    original=json.loads((r/'FINAL_INTEGRITY.json').read_text());assert original['passed'] and original['production_files']==604
    original.update(delivery_verified=True,methods=6,views=792,combined_rows=4752,fresh_quality_rows=3960,timing_rows=14256,
        all_timing_rows=len(all_times),cartgs_plot_unchanged=True,grid=[2,3],legend_cell=[2,2],timing_smoothing=False,psnr_sigma_frames=35,
        source_final_sha256=sha(ROOT/'source_final.tar.gz'))
    save(r/'FINAL_INTEGRITY.json',original);save(r/'COMPLETION.json',dict(state='complete',passed=True,figure_cn='reports/figures/PAPER_INTRO_CN.png',figure_en='reports/figures/PAPER_INTRO_EN.pdf',report='reports/PAPER_INTRO_REPORT.md'))
    archive=ROOT/'PAPER_INTRO_RESULTS.zip';assert not archive.exists()
    with ZipFile(archive,'w',ZIP_DEFLATED) as z:
        for folder in ['reports','evaluation','reference','logs']:
            for p in (ROOT/folder).rglob('*'):
                if p.is_file() and p.suffix!='.lock':z.write(p,p.relative_to(ROOT))
        for p in (ROOT/'build').rglob('*.so'):z.write(p,p.relative_to(ROOT))
        z.write(ROOT/'source_final.tar.gz','source_final.tar.gz');z.write(ROOT/'STATUS.json','STATUS.json')
    info=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive));save(r/'ARCHIVE_SHA256.json',info);print('PAPER_INTRO_DELIVERY_VERIFIED',json.dumps(info),flush=True)


if __name__=='__main__':main()
