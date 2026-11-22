"""将q80原运行和两次重跑全部纳入对照，并核对共同RGB帧质量。"""
from collections import defaultdict
from pathlib import Path
import csv
import math
import statistics
from prepare_experiment import R,Q,F,SCENES,read,write,sha

METRICS=['psnr','ssim','lpips']


def records(root):
    jobs={x['id']:x for x in read(root/'QUEUE.json')};result=defaultdict(list)
    for e in read(root/'TRAJECTORY_EVALUATION.json')['jobs']:
        if e['dataset']!='Replica' or e['scene'] not in SCENES:continue
        job=jobs[e['id']];assert e['state']==job['state']=='complete'
        s=job['result'];m=e['result'];model=Path(s['checkpoint']);meta=read(next(model.rglob('anchor_selector_state.json')))
        path=Path(m['output'])/'per_frame_metrics.csv'
        with path.open() as stream:frames=list(csv.DictReader(stream))
        assert len(frames)==m['evaluated_frames']
        for metric in METRICS:
            raw=[float(x) for x in (path.parent/(metric+'.txt')).read_text().splitlines() if x.strip()]
            assert len(raw)==len(frames) and all(math.isfinite(x) for x in raw)
            assert all(float(row[metric])==value for row,value in zip(frames,raw))
            assert abs(statistics.mean(raw)-m['trajectory_'+metric])<1e-9
        by_rgb=defaultdict(list)
        for row in frames:by_rgb[int(row['rgb_index'])].append(row)
        unique={k:{metric:statistics.mean(float(row[metric]) for row in values) for metric in METRICS}
                for k,values in by_rgb.items()}
        result[e['scene'],e['mode']].append(dict(id=e['id'],root=str(root),scene=e['scene'],mode=e['mode'],
            seed=e['seed'],repeat=e['repeat'],psnr=m['trajectory_psnr'],ssim=m['trajectory_ssim'],lpips=m['trajectory_lpips'],
            ate_rmse=m['trajectory_ate_rmse'],frames=len(frames),unique_rgb_frames=len(unique),duplicate_rgb_associations=len(frames)-len(unique),
            gaussian_count=s['gaussian_count'],anchor_count=s['anchor_count'],growth_tau0=meta['growth_tau0'],
            online_steps=s['online_steps'],tail_steps=s['tail_steps'],checkpoint=str(model),evaluation=m['output'],
            frame_metrics=unique,frame_sha256={str(path):sha(path)}))
    assert len(result)==4
    return result


def stats(rows,metric):
    values=[x[metric] for x in rows]
    return dict(mean=statistics.mean(values),std=statistics.stdev(values),minimum=min(values),maximum=max(values))


def main():
    old=records(Q);reruns=records(R);default=records(F)
    old_levels={(x['scene'],x['mode'],1):x for x in read(Q/'reports/ANCHOR_LEVEL_DISTRIBUTION.json')['rows']}
    new_levels={(x['scene'],x['mode'],x['repeat']):x for x in read(R/'reports/ANCHOR_LEVEL_DISTRIBUTION.json')['rows']}
    levels={**old_levels,**new_levels};groups=[];flat=[]
    for key in sorted(old):
        q80=sorted(old[key]+reruns[key],key=lambda x:x['repeat']);baseline=sorted(default[key],key=lambda x:x['repeat'])
        assert len(q80)==len(baseline)==3 and [x['repeat'] for x in q80]==[1,2,3]
        assert [x['repeat'] for x in baseline]==[1,2,3] and all(x['seed']==0 for x in q80+baseline)
        common=set.intersection(*(set(x['frame_metrics']) for x in q80+baseline));assert common
        for row in q80+baseline:
            row['common_frame_metrics']={metric:statistics.mean(row['frame_metrics'][i][metric] for i in sorted(common)) for metric in METRICS}
            row.pop('frame_metrics')
        for row in q80:
            level=levels[key[0],key[1],row['repeat']]
            assert row['gaussian_count']==level['gaussian_count'] and row['anchor_count']==level['anchor_count']
            row.update(l2_anchor_fraction=level['l2_anchor_fraction'],l2_gaussian_fraction=level['l2_gaussian_fraction'],
                       anchor_level_counts=level['anchor_level_counts'],gaussian_level_counts=level['gaussian_level_counts'])
            flat.append({k:v for k,v in row.items() if not isinstance(v,(dict,list))}|{
                'default_mean_psnr':statistics.mean(x['psnr'] for x in baseline),
                'delta_default_mean_psnr':row['psnr']-statistics.mean(x['psnr'] for x in baseline),
                **{'common_'+k:v for k,v in row['common_frame_metrics'].items()}})
        qstats={m:stats(q80,m) for m in METRICS+['gaussian_count','growth_tau0','l2_anchor_fraction','l2_gaussian_fraction','online_steps','tail_steps','ate_rmse']}
        bstats={m:stats(baseline,m) for m in METRICS+['gaussian_count','online_steps','tail_steps','ate_rmse']}
        delta={m:qstats[m]['mean']-bstats[m]['mean'] for m in METRICS}
        common_delta={m:statistics.mean(x['common_frame_metrics'][m] for x in q80)
                        -statistics.mean(x['common_frame_metrics'][m] for x in baseline) for m in METRICS}
        groups.append(dict(scene=key[0],mode=key[1],q80_runs=q80,default_runs=baseline,q80_statistics=qstats,
            default_statistics=bstats,mean_differences=delta,common_frame_mean_differences=common_delta,
            common_unique_rgb_frames=len(common),below_default_mean=sum(x['psnr']<bstats['psnr']['mean'] for x in q80),
            below_default_minimum=sum(x['psnr']<bstats['psnr']['minimum'] for x in q80)))
    sources=[root/name for root in [R,Q,F] for name in ['QUEUE.json','TRAJECTORY_EVALUATION.json']]
    write(R/'reports/ROOM01_REPEAT_COMPARISON.json',dict(groups=groups,
        source_hashes={str(p):sha(p) for p in sources},all_repetitions_retained=True,seed=0,new_training_runs=8))
    with (R/'reports/ROOM01_REPEAT_COMPARISON.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    lines=['# room0／room1的q80、alpha1.5重复验证','',
      '每场景／模式包括原运行和两次从头重跑，共3遍seed0；新增8组Full SLAM＋正常tail及完整轨迹评价，不训练LoD，不保存渲染图片。',
      '默认对照为q90、alpha2的seed0三次均值，不选择最佳。PSNR／SSIM越高越好，LPIPS越低越好。标准差为三次运行的样本标准差。','',
      '|场景|模式|q80三次PSNR（原／重跑1／重跑2）|q80均值±标准差|默认均值±标准差|均值差|低于默认均值次数|共同帧差|',
      '|---|---|---|---:|---:|---:|---:|---:|']
    for g in groups:
        qs=g['q80_statistics']['psnr'];bs=g['default_statistics']['psnr'];values='／'.join(f"{x['psnr']:.4f}" for x in g['q80_runs'])
        lines.append(f"|{g['scene']}|{g['mode']}|{values}|{qs['mean']:.4f}±{qs['std']:.4f}|"
            f"{bs['mean']:.4f}±{bs['std']:.4f}|{g['mean_differences']['psnr']:+.4f}|{g['below_default_mean']}/3|"
            f"{g['common_frame_mean_differences']['psnr']:+.4f}|")
    lines+=['','|场景|模式|重复|PSNR|SSIM|LPIPS|Gaussian数|L2 Anchor占比|L2成员占比|校准t|online／tail步数|ATE RMSE|',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in flat:
        lines.append(f"|{row['scene']}|{row['mode']}|{row['repeat']}|{row['psnr']:.4f}|{row['ssim']:.4f}|"
            f"{row['lpips']:.4f}|{row['gaussian_count']:,}|{row['l2_anchor_fraction']:.2%}|{row['l2_gaussian_fraction']:.2%}|"
            f"{row['growth_tau0']:.8f}|{row['online_steps']}／{row['tail_steps']}|{row['ate_rmse']:.6f}|")
    lines+=['','共同帧核对：每组q80三次和默认三次取共同唯一RGB身份交集，重复关联先在同一RGB内平均，再等权平均共同帧；只重聚合已有指标，不重新渲染。']
    for g in groups:lines.append(f"- {g['scene']} {g['mode']}：{g['common_unique_rgb_frames']}个共同唯一RGB帧；ΔSSIM={g['common_frame_mean_differences']['ssim']:+.5f}，ΔLPIPS={g['common_frame_mean_differences']['lpips']:+.5f}。")
    lines+=['','三次同seed重复用于衡量运行波动，不能彻底排除随机性。异步SLAM的训练曝光和实际步数、q80校准输入、相机轨迹和拓扑仍随执行变化。',
      'L2占比与PSNR在本实验中的同时变化仅是相关性，不能据此确认L2增多导致质量下降。必须结合三次质量范围、共同帧结果和实际训练预算判断是否复现。',
      '所有重复均保留，不因指标低重跑；OOM次数与失败记录独立保存。Anchor及成员统计包含inactive slot，不是逐视角可见点。',
      '逐次完整路径、原始SHA256和统计见JSON／CSV；完整性核验见FINISHED.json及PROTECTED_AFTER.json。']
    (R/'reports/ROOM01_REPEAT_COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('ROOM01_ALL_THREE_REPEATS_REPORT_READY',[(g['scene'],g['mode'],g['mean_differences']) for g in groups],flush=True)


if __name__=='__main__':main()
