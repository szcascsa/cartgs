"""最终源码/二进制/输入与生产完整性，生成可回退独立交付包。"""
from pathlib import Path
import json
import hashlib
import tarfile
import time
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(4194304),b''):h.update(part)
    return h.hexdigest()


def save(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False))


def main():
    reports=ROOT/'reports'
    assert json.loads((reports/'LAST_PASS_STATUS.json').read_text())['state']=='complete'
    assert json.loads((reports/'REVIEW_COMPLETE.json').read_text())['passed']
    before_source=json.loads((reports/'LOCAL_BEFORE.json').read_text())
    assert all(sha(ROOT/'original'/n)==h for n,h in before_source.items())
    checks={}
    for name in ['LAYER_VALIDATION','PIPELINE_VALIDATION','STATE_VALIDATION','TOPOLOGY_VALIDATION','TRAJECTORY_VALIDATION','UNIT_VALIDATION','DISPLAY_VALIDATION','PURITY_VALIDATION']:
        v=json.loads((reports/(name+'.json')).read_text());assert v['passed'];checks[name]=v.get('comparisons',v.get('tests',len(v.get('checks',[]))))
    assert checks['LAYER_VALIDATION']==7312 and checks['PIPELINE_VALIDATION']==50000 and checks['TRAJECTORY_VALIDATION']==269280
    assert len(json.loads((reports/'MOVING_BENCHMARK.json').read_text())['results'])==5
    for i in [1,2]:
        summary=json.loads((reports/('BENCHMARK_'+str(i)+'.json')).read_text())['summary'];lookup={(x['condition'],x['backend']):x for x in summary}
        for c in {x['condition'] for x in summary}:
            assert lookup[c,'cuda']['total_host_ms']<lookup[c,'pytorch']['total_host_ms']
            assert lookup[c,'cuda_graph_no_timing']['total_host_ms']<lookup[c,'pytorch']['total_host_ms']
    before=json.loads((reports/'PRODUCTION_BEFORE.json').read_text());after={n:dict(bytes=Path(n).stat().st_size,sha256=sha(n)) for n in before}
    assert len(before)==604 and after==before;save(reports/'PRODUCTION_AFTER.json',after)
    inputs=json.loads((reports/'INPUTS_BEFORE.json').read_text());assert all(sha(p)==h for p,h in inputs.items())
    assert json.loads((reports/'LEGACY_CHECKPOINT_INTEGRITY.json').read_text())['passed']
    # Cache tests and fixed fixtures must not have touched any input checkpoint.
    save(reports/'INPUTS_AFTER.json',{p:sha(p) for p in inputs})
    files=[p for p in (ROOT/'source').rglob('*') if p.is_file() and p.suffix in ['.py','.cu','.cpp','.html','.md'] and '__pycache__' not in p.parts]
    source={str(p.relative_to(ROOT/'source')):sha(p) for p in files};save(reports/'SOURCE_FINAL_HASHES.json',source)
    with tarfile.open(ROOT/'source_final.tar.gz','w:gz') as a:
        for p in sorted(files):a.add(p,arcname=str(p.relative_to(ROOT/'source')),recursive=False)
    binary=ROOT/'build/mycartgs_frozen_inference_v2/mycartgs_frozen_inference_v2.so';assert binary.exists()
    assert str(binary) in (reports/'RUNTIME_MAPS.txt').read_text()
    save(reports/'BINARY_IDENTITY.json',dict(path=str(binary),sha256=sha(binary),build_ninja=(binary.parent/'build.ninja').read_text()))
    assert (reports/'JOINT_VIEW_INFERENCE_OPTIMIZATION_REPORT.md').exists()
    assert len(list((reports/'figures').glob('*.png')))==2
    images=[str(p) for p in (ROOT/'evaluation').rglob('*') if p.suffix.lower() in ['.png','.jpg','.jpeg','.webp']]
    assert not images
    result=dict(passed=True,completed_unix=time.time(),protected_production_files=604,changed_production_files=[],
        protected_checkpoint_files=len(inputs),checkpoint_files_unchanged=True,checks=checks,trajectory_frames=792,trajectory_cases=26928,
        images_saved=0,training_runs=0,source_files=len(source),production_deployed=False,commit=False,push=False,
        free_GiB=shutil.disk_usage(ROOT).free/1024**3)
    save(reports/'FINAL_INTEGRITY.json',result)
    destination=reports/'JOINT_VIEW_INFERENCE_RESULTS.zip';assert not destination.exists()
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED) as z:
        for d in ['reports','logs','evaluation']:
            for p in sorted((ROOT/d).rglob('*')):
                if p.is_file() and p!=destination and p.suffix not in ['.sqlite','.nsys-rep','.lock','.tar.gz','.pt'] and not p.name.endswith('.tar.gz'):
                    z.write(p,p.relative_to(ROOT))
        z.write(ROOT/'source_final.tar.gz','source_final.tar.gz')
        for p in (ROOT/'original').rglob('*'):
            if p.is_file():z.write(p,p.relative_to(ROOT))
    save(reports/'ARCHIVE_SHA256.json',dict(path=str(destination),bytes=destination.stat().st_size,sha256=sha(destination),source_sha256=sha(ROOT/'source_final.tar.gz')))
    save(reports/'COMPLETION.json',dict(state='complete',integrity_passed=True,report='reports/JOINT_VIEW_INFERENCE_OPTIMIZATION_REPORT.md',source='source_final.tar.gz'))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
