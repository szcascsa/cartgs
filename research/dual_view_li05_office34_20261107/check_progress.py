"""只读核查两阶段进展及真实进程，记录当前巡检证据。"""
from pathlib import Path
import collections
import datetime
import json
import time

R = Path(__file__).resolve().parent
B = R.parent/'dual_view_li_best_full_20261107'


def process(info):
    if not info:
        return None
    try:
        path = Path('/proc')/str(info['pid'])
        fields = (path/'stat').read_text().rsplit(') ', 1)[1].split()
        match = fields[19] == str(info['start_ticks']) and Path('/proc/sys/kernel/random/boot_id').read_text().strip() == info['boot_id']
        return dict(pid=info['pid'], live=match and fields[0] != 'Z', state=fields[0])
    except FileNotFoundError:
        return dict(pid=info['pid'], live=False)


def main():
    original = json.loads((B/'LOD_PIPELINE.json').read_text())
    before = next(j for j in original['jobs'] if j['id'] == 'Replica/office4/rgbd')
    state = json.loads((R/'LOD_PIPELINE.json').read_text())
    result = dict(time_hk=datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
                  original_phase=original['phase'], office4_original_evaluations=sum(e['state'] == 'complete' for e in before['evaluations']),
                  phase=state['phase'], jobs=[], active=[], recent_metrics=[], error=state.get('error'))
    result['baseline_active'] = [dict(scope=e['scope'], ratio=e['ratio'], process=process(e.get('process')),
                                     log_bytes=(Path(e['attempt'])/'evaluate.log').stat().st_size)
                                 for e in before['evaluations'] if e['state'] == 'running']
    for job in [*state['jobs']] + ([state['active_validation']] if state.get('active_validation') else []):
        result['jobs'].append(dict(id=job['id'], state=job['state'], target_steps=job['target_steps'],
            health_failures=job.get('health_failures', 0), oom_failures=job.get('oom_failures', 0),
            evaluations=dict(collections.Counter(e['state'] for e in job.get('evaluations', [])))))
        baseline = next(j for j in original['jobs'] if j['id'] == job['id'])
        lookup = {(e['scope'], e['ratio']): e['result'] for e in baseline['evaluations'] if e['state'] == 'complete'}
        for item in [e for e in job.get('evaluations', []) if e['state'] == 'complete'][-2:]:
            psnr = item['result']['psnr']
            result['recent_metrics'].append(dict(job=job['id'], scope=item['scope'], ratio=item['ratio'], psnr=psnr,
                delta_li01=psnr-lookup[(item['scope'], item['ratio'])]['psnr']))
        items = ([job] if job['state'] == 'running' else []) + [e for e in job.get('evaluations', []) if e['state'] == 'running']
        for item in items:
            folder = Path(item['attempt'])
            trace = folder/'model/real_lod_steps.csv'
            log = folder/('evaluate.log' if 'scope' in item else 'train.log')
            detail = dict(job=job['id'], scope=item.get('scope', 'training'), ratio=item.get('ratio'),
                          process=process(item.get('process')), attempt=str(folder))
            if log.exists():
                detail.update(log_bytes=log.stat().st_size, log_age_seconds=round(time.time()-log.stat().st_mtime, 1))
            if trace.exists():
                with trace.open('rb') as stream:
                    stream.seek(max(0, trace.stat().st_size-4096))
                    lines = stream.read().decode().splitlines()
                detail.update(step=next((int(line.split(',')[0]) for line in reversed(lines) if line.split(',')[0].isdigit()), None),
                              trace_bytes=trace.stat().st_size, trace_age_seconds=round(time.time()-trace.stat().st_mtime, 1))
            result['active'].append(detail)
    receipt = R/'reports/LATEST_CHECK.json'
    receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
