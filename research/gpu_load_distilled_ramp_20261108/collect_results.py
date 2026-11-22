"""正式测量退出后，仅CPU核验并打包原始证据，不运行渲染。"""
import csv
import datetime
import hashlib
import json
import os
from pathlib import Path
import tarfile

from run_guard import implementation_identity, require_gpu_acceptance

ROOT=Path(__file__).resolve().parents[3]
RUN=ROOT/'runs/formal03'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name,value):
    (ROOT/'reports'/name).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')


def main():
    status=json.loads((ROOT/'reports/STATUS.json').read_text())
    assert status['state']=='complete' and status['label']=='formal03'
    assert not Path(f"/proc/{status['pid']}").exists(),'测量进程尚未退出'
    identity=implementation_identity(ROOT)
    proof=require_gpu_acceptance(ROOT,identity)
    assert identity==json.loads((RUN/'IMPLEMENTATION.json').read_text())
    assert proof==json.loads((RUN/'GPU_ACCEPTANCE_USED.json').read_text())
    protected=json.loads((ROOT/'reports/PROTECTED_BEFORE.json').read_text())
    changed=[name for name,value in protected.items() if not Path(name).is_file() or sha(Path(name))!=value]
    assert not changed,changed
    environment=json.loads((RUN/'ENVIRONMENT.json').read_text())
    assert sha(Path(environment['student']))==environment['student_sha256']
    with (RUN/'ALL_TIMINGS.csv').open() as stream: rows=list(csv.DictReader(stream))
    assert len(rows)==30000
    assert len({(r['method'],r['repeat'],r['frame']) for r in rows})==30000
    conditions=sorted(RUN.glob('*/ACCEPTANCE.json'));assert len(conditions)==15
    acceptances=[json.loads(path.read_text()) for path in conditions]
    assert all(x['passed'] and x['bin_match'] and x['hardware']['billing_sm_area'] for x in acceptances)
    assert all(x['artificial_load_prepare_overlap']==0 and x['summary']['requests']==2000 for x in acceptances)
    resident=json.loads((ROOT/'reports/RESIDENT_PROCESS.json').read_text())
    records=[json.loads(line) for line in (RUN/'GPU_MONITOR.jsonl').read_text().splitlines()]
    assert records
    for record in records:
        assert all(item['pid']==resident['pid'] for item in record['foreign'])
        assert all(item['pid'] in (status['pid'],resident['pid']) for item in record['processes'])
        for line in record['pmon'].splitlines():
            values=line.split()
            if len(values)<7 or values[0].startswith('#') or values[2]!='C': continue
            if int(values[1])!=status['pid']:
                assert int(values[1])==resident['pid']
                assert all(value=='-' or int(value)==0 for value in values[3:7])
    image_files=[str(path.relative_to(ROOT)) for name in ('formal03','smoke07')
                 for path in (ROOT/'runs'/name).rglob('*') if path.suffix.lower() in ('.png','.jpg','.jpeg','.exr')]
    assert not image_files,image_files
    validation=dict(passed=True,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        raw_requests=len(rows),conditions=len(conditions),views=2000,repeats=3,
        raw_sha256=sha(RUN/'ALL_TIMINGS.csv'),protected_files=len(protected),changed_protected_files=changed,
        implementation_unchanged=True,implementation_files=len(identity['files']),
        gpu_preflight_run=proof['source_run'],graph_native_bitwise_cases=proof['graph_cases'],
        gpu_preflight_requests=proof['rows'],artificial_load_prepare_overlap=0,
        maximum_sm_duty_error_pp=max(abs(item['sm_error'])*100 for x in acceptances for item in x['bins']),
        maximum_packet_ms=max(x['maximum_on_ns'] for x in acceptances)/1e6,
        maximum_integrated_actual_sm_duty=max(x['summary']['actual_sm_duty'] for x in acceptances),
        gpu_monitor_samples=len(records),gpu_monitor_duration_s=records[-1]['unix']-records[0]['unix'],
        unexpected_compute_process_samples=0,allowed_idle_resident_pid=resident['pid'],
        final_measurement_pid=status['pid'],final_measurement_pid_exited=True,
        rendered_image_files=len(image_files),source_model_unchanged=True,production_unchanged=True,
        committed=False,pushed=False,load_definition='normalized_SM_residency_time_duty',
        disk_free_bytes=os.statvfs(ROOT).f_bavail*os.statvfs(ROOT).f_frsize)
    save('FINAL_VALIDATION.json',validation)
    files=sorted(path for name in ('runs/formal03','runs/smoke07','reports')
                 for path in (ROOT/name).rglob('*') if path.is_file()
                 and path.suffix in ('.csv','.json','.md','.log')
                 and path.name!='RESULTS_EVIDENCE_SHA256.json')
    files+=sorted(path for path in (ROOT/'logs').glob('*.log'))
    source=ROOT/'source/research/gpu_load_distilled_ramp_20261108'
    files+=sorted(path for path in source.glob('*') if path.is_file()
                  and path.suffix in ('.py','.h','.cu','.cpp','.md'))
    files+=[ROOT/'build/libramp_load.so']
    files=list(dict.fromkeys(files))
    manifest={str(path.relative_to(ROOT)):sha(path) for path in files}
    save('RESULTS_EVIDENCE_SHA256.json',manifest)
    files.append(ROOT/'reports/RESULTS_EVIDENCE_SHA256.json')
    archive=ROOT/'reports/results_evidence.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for path in files:tar.add(path,arcname=str(path.relative_to(ROOT)),recursive=False)
    digest=sha(archive)
    (ROOT/'reports/results_evidence.tar.gz.sha256').write_text(digest+'\n')
    print(json.dumps(validation,indent=2,ensure_ascii=False))
    print('ARCHIVE',archive.stat().st_size,digest,'FILES',len(files))


if __name__=='__main__':
    main()
