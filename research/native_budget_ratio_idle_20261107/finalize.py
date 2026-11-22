"""核验本轮数据与完整性，再封装服务器交付。"""
from common import *
from collections import defaultdict
import math
import tarfile
from zipfile import ZipFile,ZIP_DEFLATED


def read(path):
    with Path(path).open(encoding='utf-8') as f:return list(csv.DictReader(f))


def main():
    r=ROOT/'reports';status=json.loads((ROOT/'STATUS.json').read_text());assert status['state']=='complete'
    for n,count in [('TIMINGS.csv',19008),('ALL_TIMINGS.csv',57024),('PER_VIEW_COMPARISON.csv',6336),('PLOT_DATA_RAW_AND_SMOOTHED.csv',6336)]:assert len(read(r/n))==count,n
    accepted=json.loads((r/'TIMING_ACCEPTANCE.json').read_text());assert accepted['complete'] and accepted['accepted_rounds']==3
    attempts=json.loads((r/'TIMING_ATTEMPTS.json').read_text());passing=[v for v in attempts if v['accepted']];assert [v['attempt'] for v in passing]==[5,6,8]
    for a in passing:assert a['coverage'] and not any(s['external_active'] for s in a['samples'])
    times=read(r/'TIMINGS.csv');groups=defaultdict(list);keys=set()
    for v in times:
        key=(v['method'],int(v['frame']),int(v['repeat']));assert key not in keys;keys.add(key);groups[v['method'],int(v['repeat'])].append(v)
        assert v['accepted']=='True' and int(v['attempt']) in [5,6,8]
        assert all(math.isfinite(float(v[k])) and float(v[k])>=0 for k in ['prepare_ms','render_ms','total_ms'])
        assert abs(float(v['prepare_ms'])+float(v['render_ms'])-float(v['total_ms']))<1e-9
        if v['method']=='cartgs_full':assert int(v['K'])==35301 and float(v['prepare_ms'])==0
        else:
            target=int(v['requested_K']);assert int(v['K'])==min(target,int(v['eligible_N'])) and target-int(v['K'])==int(v['shortfall'])
            if v['method'].startswith('ours_'):assert target==int(v['method'].split('_')[1])
            else:assert target==math.floor(float(v['network_ratio'])*int(v['candidate_N']))
    assert len(groups)==24
    for (method,repeat),values in groups.items():
        assert len(values)==792 and {int(v['frame']) for v in values}==set(range(792))
    parity=json.loads((r/'QUALITY_PARITY.json').read_text());assert parity['passed'] and len(parity['cases'])==24
    counts=json.loads((ROOT/'reference/COUNT_RENDERER_PARITY.json').read_text());assert counts['passed'] and len(counts['cases'])==21
    assert json.loads((r/'ANALYSIS_VALIDATION.json').read_text())['passed']
    summary=json.loads((r/'SUMMARY.json').read_text());assert len(summary)==8
    for v in summary:
        raw=[x for x in times if x['method']==v['method']]
        assert len(raw)==2376
        for k in ['prepare_ms','render_ms','total_ms']:assert abs(sum(float(x[k]) for x in raw)/2376-v[k+'_mean'])<1e-10
    for prefix in ['INPUTS','PRODUCTION']:
        before=json.loads((r/(prefix+'_BEFORE.json')).read_text());after={n:dict(bytes=Path(n).stat().st_size,sha256=sha(n)) if isinstance(v,dict) else sha(n) for n,v in before.items()}
        assert before==after,prefix;save(r/(prefix+'_AFTER.json'),after)
    previous=Path('/home/prmi/zwj/experiments/native_budget100_500_fr1xyz_20261106')
    identity=json.loads((previous/'reference/NATIVE_BINARY_IDENTITY.json').read_text())
    assert all(sha(ROOT/'source/python'/n)==h for n,h in identity['sources'].items())
    save(r/'NATIVE_REFERENCE_IDENTITY.json',identity)
    for n in ['NATIVE_FIXED_AND_RATIO_THREE_COLUMNS.png','NATIVE_FIXED_AND_RATIO_THREE_COLUMNS.pdf','NATIVE_FIXED_AND_RATIO_THREE_COLUMNS.svg','NATIVE_FIXED_AND_RATIO_RAW.png']:assert (r/'figures'/n).is_file()
    assert (r/'NATIVE_FIXED_AND_RATIO_IDLE_REPORT.md').is_file()
    assert not [p for p in (ROOT/'reference').rglob('*') if p.suffix.lower() in ['.png','.jpg','.jpeg','.webp']]
    source={str(p.relative_to(ROOT/'source')):sha(p) for p in (ROOT/'source').rglob('*') if p.is_file()}
    save(r/'SOURCE_FINAL_HASHES.json',source)
    with tarfile.open(ROOT/'source_final.tar.gz','w:gz') as z:
        for n in sorted(source):z.add(ROOT/'source'/n,arcname=n,recursive=False)
    original=json.loads((r/'FINAL_INTEGRITY.json').read_text());assert original['passed'] and original['production_files']==604
    original.update(delivery_verified=True,methods=8,frames=792,accepted_timing_rows=19008,all_timing_rows=57024,combined_rows=6336,
        actual_external_gpu_activity_samples_in_accepted_rounds=0,pmon_resolution_seconds=1,quality_parity_cases=24,count_renderer_bitwise_cases=21,
        figure_columns=3,psnr_sigma_frames=35,source_final_sha256=sha(ROOT/'source_final.tar.gz'))
    save(r/'FINAL_INTEGRITY.json',original);save(r/'COMPLETION.json',dict(state='complete',passed=True,figure='reports/figures/NATIVE_FIXED_AND_RATIO_THREE_COLUMNS.png',
        report='reports/NATIVE_FIXED_AND_RATIO_IDLE_REPORT.md',server_directory=str(ROOT)))
    archive=ROOT/'NATIVE_FIXED_AND_RATIO_IDLE_RESULTS.zip';assert not archive.exists()
    with ZipFile(archive,'w',ZIP_DEFLATED) as z:
        for folder in ['reports','reference','logs']:
            for p in (ROOT/folder).rglob('*'):
                if p.is_file() and p.suffix!='.lock':z.write(p,p.relative_to(ROOT))
        for p in (ROOT/'build').rglob('*.so'):z.write(p,p.relative_to(ROOT))
        z.write(ROOT/'source_final.tar.gz','source_final.tar.gz');z.write(ROOT/'STATUS.json','STATUS.json')
    info=dict(path=str(archive),bytes=archive.stat().st_size,sha256=sha(archive));save(r/'ARCHIVE_SHA256.json',info)
    print('DELIVERY_VERIFIED',json.dumps(info),flush=True)


if __name__=='__main__':main()
