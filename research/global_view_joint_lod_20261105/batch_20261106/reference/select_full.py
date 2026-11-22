"""按各场景/模式的Full全轨迹PSNR选一次，双目使用修正后的评价口径。"""
from collections import defaultdict
import copy
import math


def select_winners(training, evaluation, corrected_stereo):
    lookup = {j['id']: copy.deepcopy(j) for j in evaluation}
    for row in corrected_stereo:
        if not row['id'].endswith('/full_before'):
            continue
        job_id = row['id'].removesuffix('/full_before')
        item = lookup[job_id]
        result = item['result']
        assert int(row['frames']) == result['evaluated_frames']
        result['uncorrected_trajectory_psnr'] = result['trajectory_psnr']
        for key in ['psnr', 'ssim', 'lpips']:
            result['trajectory_' + key] = float(row[key])
        result['trajectory_ate_rmse'] = float(row['ate_rmse'])
        result['output'] = row['output']
        item['metric_scope'] = 'stereo_left_rectified_training_domain'

    groups = defaultdict(list)
    for item in training:
        if item['prune'] == 1:
            assert item['seed'] == 0
            groups[(item.get('dataset', 'TUM_RGBD'), item['scene'], item['mode'])].append(item)
    selected = []
    for (dataset, scene, mode), candidates in sorted(groups.items()):
        assert sorted(j['repeat'] for j in candidates) == [1, 2, 3]
        valid = [j for j in candidates if j['state'] == 'complete' and lookup[j['id']]['state'] == 'complete']
        assert valid, (dataset, scene, mode, '缺少有效Full模型')
        assert all(math.isfinite(lookup[j['id']]['result']['trajectory_psnr']) for j in valid)
        if mode == 'stereo':
            assert all(lookup[j['id']].get('metric_scope') == 'stereo_left_rectified_training_domain' for j in valid)
        best = max(valid, key=lambda j: (lookup[j['id']]['result']['trajectory_psnr'], -j['repeat']))
        selected.append(dict(
            id=f'{dataset}/{scene}/{mode}', dataset=dataset, scene=scene, mode=mode,
            winner=best['id'], full_psnr=lookup[best['id']]['result']['trajectory_psnr'],
            full_job=copy.deepcopy(best), full_evaluation=lookup[best['id']],
            target_steps=best['result']['online_steps'], state='pending', evaluations=[],
            candidates=[dict(id=j['id'], repeat=j['repeat'], train_state=j['state'],
                             eval_state=lookup[j['id']]['state'], result=lookup[j['id']].get('result'))
                        for j in candidates]))
    return selected
