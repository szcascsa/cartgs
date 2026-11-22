"""合并已退出原控制器和接续验收；不覆盖原失败现场和attempt历史。"""
from pathlib import Path
import argparse
import json
import time

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--unseen', action='store_true')
    args = parser.parse_args()
    r = HERE / 'reports'
    kind = 'UNSEEN' if args.unseen else 'JOINT'
    sources = [f'RELEASE_{kind}_20261115', f'RELEASE_{kind}_CONTINUED_20261115']
    first, second = [json.loads((r / (name + '_CONTROLLER.json')).read_text()) for name in sources]
    assert first['state'] == 'failed' and second['state'] == 'complete'
    plan = json.loads((r / (sources[0] + '_PLAN.json')).read_text())
    continuing = json.loads((r / (sources[1] + '_PLAN.json')).read_text())
    mapping = {task['name']: task['original_name'] for task in continuing['jobs']}
    accepted = first['accepted'] + second['accepted']
    base_names = []
    for name in accepted:
        task = name.rsplit('_a', 1)[0]
        base_names.append(mapping.get(task, task))
        job = json.loads((r / 'jobs' / (name + '.json')).read_text())
        assert job['returncode'] == 0 and job['measurement_valid'], name
    assert len(set(base_names)) == len(base_names)
    assert set(base_names) == {task['name'] for task in plan['jobs']}
    destination = r / (sources[0] + '_MERGED.json')
    with destination.open('x') as stream:
        json.dump(dict(state='complete', accepted=accepted, rejected=first['rejected'] + second['rejected'],
            source_controllers=sources, tasks=len(accepted), merged_at=time.time(),
            started_at=first['started_at'], ended_at=second['ended_at'],
            original_failure_retained=True), stream, indent=2)
    print('RELEASE_CONTROLLERS_MERGED', len(accepted), flush=True)


if __name__ == '__main__':
    main()
