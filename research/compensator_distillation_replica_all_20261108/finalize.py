"""等待本轮队列结束，核验实际预算、同选点和交付完整性。"""
from pathlib import Path
import csv
import hashlib
import json
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def main():
    controller = json.loads((ROOT/'reports/CONTROLLER.json').read_text())
    while True:
        state = json.loads((ROOT/'reports/BATCH_STATUS.json').read_text())
        if state['status']=='finished':
            break
        path = Path(f'/proc/{controller["pid"]}/stat')
        assert path.exists(),'Controller exited before finishing'
        parts = path.read_text().rpartition(')')[2].split()
        assert parts[0]!='Z' and int(parts[19])==controller['start_ticks'],'Controller identity changed'
        time.sleep(30)
    protocol = json.loads((ROOT/'reports/PROTOCOL.json').read_text())
    scenes = json.loads((ROOT/'reports/SCENES.json').read_text())
    rows_checked = updates = comparisons = 0
    completed = 0
    for scene in scenes:
        job = next(j for j in state['scenes'] if j['id']==scene['id'])
        if job['status']!='complete':
            continue
        completed += 1
        folder = ROOT/'reports'/scene['slug']
        preflight = json.loads((folder/'PREFLIGHT.json').read_text())
        assert preflight['passed'] and len(preflight['test_views'])==32
        assert not set(preflight['test_views'])&set(preflight['validation_views'])
        assert all(frame%5==0 for frame in preflight['test_views'])
        for stage in protocol['stages']:
            output = ROOT/'models'/scene['slug']/stage['name']
            config = json.loads((output/'configuration.json').read_text())
            rows = list(csv.DictReader((output/'training.csv').open()))
            assert len(rows)==stage['steps']
            assert [int(row['step']) for row in rows]==list(range(1,stage['steps']+1))
            updates += len(rows)
            for key,value in [('seed',0),('learning_rate',stage['lr']),('residual_weight',stage['residual_weight']),
                    ('image_every',stage['image_every']),('image_weight',stage['image_weight']),('ratio_jitter',stage['ratio_jitter'])]:
                assert config[key]==value,(scene['id'],stage['name'],key)
            assert config['teacher_fingerprint']==preflight['fingerprint']
        quality = json.loads((folder/'QUALITY_SUMMARY.json').read_text())
        for name,identity in quality['identities'].items():
            path = ROOT/'models'/scene['slug']/name/'best.pt'
            assert hashlib.sha256(path.read_bytes()).hexdigest()==identity['sha256']
        groups = {}
        rows = list(csv.DictReader((folder/'QUALITY_PER_VIEW.csv').open()))
        assert len(rows)==2816
        rows_checked += len(rows)
        for row in rows:
            groups.setdefault((row['frame'],row['ratio'],row['local']),[]).append(row)
        assert len(groups)==704
        assert all(len(group)==4 and len({(r['K'],r['selection_sha256']) for r in group})==1 for group in groups.values())
        assert json.loads((folder/'DEPLOYMENT_VALIDATION.json').read_text())['passed']
        checks = json.loads((folder/'GRAPH_PARITY.json').read_text())
        assert checks['passed']
        comparisons += checks['comparisons']
    result = dict(passed=state['failed']==0,completed=completed,failed=state['failed'],
        model_updates_including_reused=updates,per_view_metric_rows=rows_checked,
        graph_comparisons=comparisons,training_configs_verified=True,
        checkpoint_sha_verified=True,selection_identities_verified=True,holdout_disjoint=True)
    (ROOT/'reports/RESULTS_VALIDATION.json').write_text(json.dumps(result,indent=2))
    for script in ['integrity.py','summarize.py','write_report.py']:
        with (ROOT/'logs'/f'final_{Path(script).stem}.log').open('x') as log:
            code = subprocess.call([sys.executable,str(HERE/script)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        assert code==0,(script,code)
    (ROOT/'reports/FINALIZED.json').write_text(json.dumps(result,indent=2))
    print('REPLICA_FINALIZED',json.dumps(result),flush=True)


if __name__=='__main__':
    main()
