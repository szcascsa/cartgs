"""逐帧配对LI=0.1/0.5结果，核对训练曝光并输出中文报告。"""
from pathlib import Path
import csv
import json
import statistics
import numpy as np

R = Path(__file__).resolve().parent
B = R.parent/'dual_view_li_best_full_20261107'


def read(path):
    return json.loads(path.read_text())


def csv_rows(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if rows:
        with path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def paired(before, after):
    a = csv_rows(Path(before['output'])/'per_frame_metrics.csv')
    b = csv_rows(Path(after['output'])/'per_frame_metrics.csv')
    keys = ['synchronized_trajectory_index', 'rgb_index', 'trajectory_timestamp', 'rgb_timestamp', 'rgb_path']
    assert len(a) == len(b) == before['evaluated_frames'] == after['evaluated_frames']
    assert all(tuple(x[k] for k in keys) == tuple(y[k] for k in keys) for x, y in zip(a, b)), '评价帧不一致'
    for name in ['psnr', 'ssim', 'lpips']:
        assert abs(statistics.mean(float(x[name]) for x in a)-before[name]) < 1e-7
        assert abs(statistics.mean(float(x[name]) for x in b)-after[name]) < 1e-7
    delta = np.array([float(y['psnr'])-float(x['psnr']) for x, y in zip(a, b)])
    result = dict(paired_psnr_median=float(np.median(delta)), paired_psnr_p10=float(np.quantile(delta, .1)),
                  paired_psnr_p90=float(np.quantile(delta, .9)), improved_view_fraction=float(np.mean(delta > 0)))
    ka = Path(before['output'])/'anchor_frustum_per_view.csv'
    kb = Path(after['output'])/'anchor_frustum_per_view.csv'
    if ka.exists() and kb.exists():
        aa, bb = csv_rows(ka), csv_rows(kb)
        assert len(aa) == len(bb) == len(a)
        assert all((x['trajectory_index'], x['gt_index']) == (y['trajectory_index'], y['gt_index']) for x, y in zip(aa, bb))
        result['same_actual_K_views_fraction'] = statistics.mean(
            int(x['active_gaussians']) == int(y['active_gaussians']) for x, y in zip(aa, bb))
    else:
        result['same_actual_K_views_fraction'] = float(before['actual_K_mean'] == after['actual_K_mean'])
    return result


def exposure(before, after):
    a = Path(before['trained_model'])
    b = Path(after['trained_model'])
    ra = csv_rows(a/'real_lod_steps.csv')
    rb = csv_rows(b/'real_lod_steps.csv')
    keys = ['step', 'ratio', 'N', 'master_update', 'selector_update', 'xyz_lr']
    assert len(ra) == len(rb) == before['target_steps'] == after['target_steps']
    assert all(tuple(x[k] for k in keys) == tuple(y[k] for k in keys) for x, y in zip(ra, rb)), '训练调度不一致'
    def cameras(path):
        with (path/'view_budget_steps.jsonl').open() as stream:
            return [(x['step'], x['keyframe_id'], x['ratio']) for x in map(json.loads, stream)]
    assert cameras(a) == cameras(b), '训练相机或ratio序列不一致'
    sa = read(next(a.glob('*_shutdown/ply/real_lod_state.json')))
    sb = read(next(b.glob('*_shutdown/ply/real_lod_state.json')))
    keys = ['full_training_renders', 'full_training_backwards', 'lod_training_renders', 'lod_training_backwards',
            'view_budget_training_renders', 'view_budget_training_backwards', 'master_updates', 'selector_updates',
            'optimizer_commits', 'importance_full_renders', 'ratio_steps', 'view_budget_ratio_steps']
    assert all(sa[k] == sb[k] for k in keys), '实际训练曝光不一致'
    return dict(scene=after['scene'], steps=after['target_steps'], camera_ratio_sequence_equal=True,
                counters={k: sb[k] for k in keys},
                initialization_bitwise_verified=(b/'initialization_verified.json').is_file(),
                baseline_health_retries=before.get('health_failures', 0), li05_health_retries=after.get('health_failures', 0),
                baseline_oom_retries=before.get('oom_failures', 0), li05_oom_retries=after.get('oom_failures', 0))


def geometry(job, label):
    groups = {'global': [], 'local': []}
    with (Path(job['trained_model'])/'lod_health.jsonl').open() as stream:
        for item in map(json.loads, stream):
            if 'scale_factor_max' in item:
                groups['local' if item.get('branch') == 'view_budget' else 'global'].append(item)
    return [dict(scene=job['scene'], li_weight=label, scope=scope, checked_steps=len(items),
                 max_scale_factor=max(v['scale_factor_max'] for v in items),
                 max_scale_tail_fraction=max(v['scale_tail_fraction'] for v in items),
                 max_xyz_anchor_units=max(v['xyz_anchor_units_max'] for v in items))
            for scope, items in groups.items() if items]


def main():
    state = read(R/'LOD_PIPELINE.json')
    assert state['phase'] == 'ready_for_comparison'
    original = {j['id']: j for j in read(B/'LOD_PIPELINE.json')['jobs']}
    rows, exposures, tails = [], [], []
    lines = ['# office3／office4 RGB-D：LI=0.5额外对照结果', '',
             '仅将LI权重从0.1改为0.5，复用相同源码快照和逐位相同的二进制；从同一原Full起点重新训练。',
             'seed0、fresh Adam/RNG、固定拓扑、Master可更新；原始Full权重转换逐位验收。',
             '全局和局部同ratio使用不同点数预算；此处按相同策略比较两个模型，不强制跨模型same-K。', '',
             '|场景|LI=0.5状态|健康停止重试次数|OOM重试次数|', '|---|---|---:|---:|']
    for job in state['jobs']:
        before = original[job['id']]
        lines.append(f"|{job['scene']} RGB-D|{job['state']}|{job.get('health_failures', 0)}|{job.get('oom_failures', 0)}|")
        if 'trained_model' not in job:
            lines += ['', f"{job['scene']}未得到成功模型，原因：{job.get('error')}。未把失败训练作为有效质量结果。"]
            continue
        exposures.append(exposure(before, job))
        tails += geometry(before, .1)+geometry(job, .5)
        old = {(e['scope'], e['ratio']): e['result'] for e in before['evaluations'] if e['state'] == 'complete'}
        for item in job['evaluations']:
            if item['state'] != 'complete':
                continue
            a = old[(item['scope'], item['ratio'])]
            b = item['result']
            assert abs(b['li_weight']-.5) < 1e-6 and a['gaussian_count'] == b['gaussian_count']
            row = dict(scene=job['scene'], scope=item['scope'], ratio=item['ratio'], kind=item['kind'],
                       frames=b['evaluated_frames'], N=b['gaussian_count'], K_li01=a['actual_K_mean'], K_li05=b['actual_K_mean'])
            for name in ['psnr', 'ssim', 'lpips', 'ate_rmse']:
                row.update({name+'_li01': a[name], name+'_li05': b[name], name+'_delta': b[name]-a[name]})
            row.update(paired(a, b))
            rows.append(row)
    for scene in ['office3', 'office4']:
        lines += ['', f'## {scene} RGB-D', '', '|域|Ratio|PSNR 0.1|PSNR 0.5|ΔPSNR|ΔSSIM|ΔLPIPS|改善视图比例|',
                  '|---|---:|---:|---:|---:|---:|---:|---:|']
        for row in rows:
            if row['scene'] == scene:
                scope = {'global': '全局', 'view_ratio': '局部', 'full': 'Full'}[row['scope']]
                lines.append(f"|{scope}|{row['ratio']:.2f}|{row['psnr_li01']:.4f}|{row['psnr_li05']:.4f}|"
                             f"{row['psnr_delta']:+.4f}|{row['ssim_delta']:+.5f}|{row['lpips_delta']:+.5f}|"
                             f"{row['improved_view_fraction']:.1%}|")
    lines += ['', '## 验证与局限', '',
              '逐步核对训练相机、ratio、N、Master/Selector更新标记、XYZ学习率；逐项核对Full/全局LoD/局部LoD的渲染和反传次数。',
              '逐帧核对评价RGB关联和时间戳；逐帧PSNR均值与汇总值一致。数值、实际K和配对分布见CSV。',
              'Scale尾部统计见GEOMETRY.csv，倍数按相对未补偿Master的最大轴计算；表中最大值是整个成功训练期间的极值，不能当成最终点分布。',
              '每组仅单seed。LI=0.1历史两场景均在首次健康停止后完成第2次训练；LI=0.5次数见表。'
              '相同seed不保证CUDA原子归约逐位确定，成功重跑之间也可能有差异，不能把这次全部变化都确定归因于LI权重。'
              '结果未用于选择checkpoint、学习率或停止时间。',
              '无渲染帧图片；不改变生产默认。原队列在本次对照完成并汇报前保持HOLD。']
    write_csv(R/'reports/LI05_COMPARISON.csv', rows)
    write_csv(R/'reports/GEOMETRY.csv', tails)
    (R/'reports/TRAINING_EXPOSURE.json').write_text(json.dumps(exposures, ensure_ascii=False, indent=2)+'\n')
    (R/'reports/LI05_COMPARISON_REPORT.md').write_text('\n'.join(lines)+'\n')
    (R/'reports/COMPARISON_READY.json').write_text(json.dumps(dict(
        completed_groups=sum(j['state'] == 'complete' for j in state['jobs']), paired_cases=len(rows),
        common_frame_checks=True, exposure_checks=len(exposures), saved_render_images=0), indent=2)+'\n')
    for row in rows:
        if row['scope'] == 'full' or (row['scope'] == 'view_ratio' and row['ratio'] in [.01, .05, .1]):
            print(row['scene'], row['scope'], row['ratio'], row['psnr_li01'], row['psnr_li05'], row['psnr_delta'])


if __name__ == '__main__':
    main()
