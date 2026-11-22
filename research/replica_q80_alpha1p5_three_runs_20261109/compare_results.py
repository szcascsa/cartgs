"""汇总全部48遍q80结果，按每组完整轨迹PSNR选同次模型。"""
from collections import Counter,defaultdict
from pathlib import Path
import csv
import math
import statistics
from prepare_experiment import R,Q,A,F,read,write,sha

SCENES=['office0','office1','office2','office3','office4','room0','room1','room2']
MODES=['mono','rgbd']


def records(root):
    jobs={x['id']:x for x in read(root/'QUEUE.json')};rows=[]
    for e in read(root/'TRAJECTORY_EVALUATION.json')['jobs']:
        if e['dataset']!='Replica':continue
        job=jobs[e['id']];assert job['state']==e['state']=='complete',e['id']
        m=e['result'];s=job['result'];checkpoint=Path(s['checkpoint'])
        metadata=read(next(checkpoint.rglob('anchor_selector_state.json')))
        evaluation=Path(m['output'])
        with (evaluation/'per_frame_metrics.csv').open() as stream:frames=list(csv.DictReader(stream))
        assert len(frames)==m['evaluated_frames'] and len({int(x['rgb_index']) for x in frames})>0
        for name in ['psnr','ssim','lpips']:
            values=[float(x) for x in (evaluation/(name+'.txt')).read_text().splitlines() if x.strip()]
            assert len(values)==len(frames) and all(math.isfinite(x) for x in values)
            assert all(float(row[name])==value for row,value in zip(frames,values))
            assert abs(statistics.mean(values)-m['trajectory_'+name])<1e-9
        ply,=checkpoint.rglob('point_cloud.ply')
        with ply.open('rb') as stream:
            vertex=None
            for _ in range(150):
                line=stream.readline().decode('ascii').strip()
                if line.startswith('element vertex '):vertex=int(line.split()[-1]);break
                if line=='end_header':break
        assert vertex==s['gaussian_count']==metadata['gaussian_count']
        rows.append(dict(dataset='Replica',scene=e['scene'],mode=e['mode'],repeat=e['repeat'],seed=e['seed'],
            psnr=m['trajectory_psnr'],ssim=m['trajectory_ssim'],lpips=m['trajectory_lpips'],ate_rmse=m['trajectory_ate_rmse'],
            gaussian_count=s['gaussian_count'],anchor_count=s['anchor_count'],online_steps=s['online_steps'],tail_steps=s['tail_steps'],
            evaluated_frames=m['evaluated_frames'],unique_rgb_frames=len({int(x['rgb_index']) for x in frames}),
            growth_tau0=metadata['growth_tau0'],growth_quantile=metadata.get('growth_quantile',.9),
            growth_alpha=metadata.get('growth_alpha',2),growth_strict_upper=metadata.get('growth_strict_upper',False),
            checkpoint=str(checkpoint),evaluation=str(evaluation),source_batch=str(root),
            frame_metrics_sha256=sha(evaluation/'per_frame_metrics.csv')))
    return rows


def select_best(rows):
    expected=Counter((s,m,r) for s in SCENES for m in MODES for r in [1,2,3])
    assert Counter((x['scene'],x['mode'],x['repeat']) for x in rows)==expected,'必须恰有16组×3遍，禁止遗漏或重复'
    assert all(x['seed']==0 and math.isfinite(x['psnr']) for x in rows)
    grouped=defaultdict(list)
    for row in rows:grouped[row['scene'],row['mode']].append(row)
    selected=[];groups=[]
    for scene in SCENES:
        for mode in MODES:
            candidates=sorted(grouped[scene,mode],key=lambda x:x['repeat'])
            best=max(candidates,key=lambda x:(x['psnr'],-x['repeat']))
            selected.append(dict(best))
            groups.append(dict(scene=scene,mode=mode,runs=candidates,best_repeat=best['repeat'],
                best_checkpoint=best['checkpoint'],psnr_mean=statistics.mean(x['psnr'] for x in candidates),
                psnr_std=statistics.stdev(x['psnr'] for x in candidates),
                psnr_min=min(x['psnr'] for x in candidates),psnr_max=best['psnr']))
    return selected,groups


def csv_write(path,rows):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader();writer.writerows(rows)


def main():
    all_rows=records(Q)+records(A)+records(R);default=records(F)
    levels={}
    for root in [Q,A,R]:
        for row in read(root/'reports/ANCHOR_LEVEL_DISTRIBUTION.json')['rows']:
            key=row['scene'],row['mode'],row.get('repeat',1);assert key not in levels;levels[key]=row
    for row in all_rows:
        level=levels[row['scene'],row['mode'],row['repeat']]
        assert row['gaussian_count']==level['gaussian_count'] and row['anchor_count']==level['anchor_count']
        assert row['growth_quantile']==.8 and row['growth_alpha']==1.5 and row['growth_strict_upper']
        row.update(l2_anchor_count=level['anchor_level_counts'][2],l2_gaussian_count=level['gaussian_level_counts'][2],
            l2_anchor_fraction=level['l2_anchor_fraction'],l2_gaussian_fraction=level['l2_gaussian_fraction'])
    selected,groups=select_best(all_rows);default_best,default_groups=select_best(default)
    default_map={(x['scene'],x['mode']):x for x in default_groups}
    default_best_map={(x['scene'],x['mode']):x for x in default_best}
    for best,group in zip(selected,groups):
        key=best['scene'],best['mode'];bg=default_map[key];bb=default_best_map[key]
        best.update(default_best_psnr=bb['psnr'],delta_default_best_psnr=best['psnr']-bb['psnr'],
            default_best_gaussian_count=bb['gaussian_count'],default_best_checkpoint=bb['checkpoint'],
            default_three_run_mean_psnr=bg['psnr_mean'],delta_default_mean_psnr=best['psnr']-bg['psnr_mean'],
            psnr_three_run_mean=group['psnr_mean'],psnr_three_run_std=group['psnr_std'])
        group.update(default_runs=bg['runs'],default_psnr_mean=bg['psnr_mean'],default_psnr_std=bg['psnr_std'],
            mean_psnr_delta=group['psnr_mean']-bg['psnr_mean'])
    ordered=sorted(all_rows,key=lambda x:(SCENES.index(x['scene']),MODES.index(x['mode']),x['repeat']))
    sources=[root/name for root in [R,Q,A,F] for name in ['QUEUE.json','TRAJECTORY_EVALUATION.json']]
    hashes={str(p):sha(p) for p in sources}
    write(R/'reports/Q80_ALL_48_RUNS.json',dict(protocol='8 scenes x 2 modes x 3 seed0 repeats; all retained',
        rows=ordered,groups=groups,source_hashes=hashes))
    csv_write(R/'reports/Q80_ALL_48_RUNS.csv',ordered)
    write(R/'reports/Q80_FINAL_BEST.json',dict(protocol='per scene/mode maximum Full-trajectory mean PSNR; all metrics and N bound to same run; lower repeat wins ties',
        rows=selected,source_hashes=hashes))
    csv_write(R/'reports/Q80_FINAL_BEST.csv',selected)
    summaries={}
    for mode in MODES+['all']:
        items=[x for x in selected if mode=='all' or x['mode']==mode]
        gs=[x for x in groups if mode=='all' or x['mode']==mode]
        summaries[mode]=dict(groups=len(items),best_psnr_mean=statistics.mean(x['psnr'] for x in items),
            best_delta_default_best=statistics.mean(x['delta_default_best_psnr'] for x in items),
            best_delta_default_mean=statistics.mean(x['delta_default_mean_psnr'] for x in items),
            three_run_mean_psnr=statistics.mean(x['psnr_mean'] for x in gs),
            three_run_mean_delta_default_mean=statistics.mean(x['mean_psnr_delta'] for x in gs),
            best_gaussian_total=sum(x['gaussian_count'] for x in items),
            gaussian_total_change_vs_default_best=sum(x['gaussian_count'] for x in items)/sum(x['default_best_gaussian_count'] for x in items)-1)
    write(R/'reports/Q80_FINAL_SUMMARY.json',summaries)
    lines=['# q80、alpha1.5最终方案：Replica三遍结果与最佳模型','',
      '8场景×Mono／RGB-D共16组，每组seed0运行三遍，合计48遍。复用原q80的16遍和room0／room1额外8遍，本轮新增24遍。全部Full SLAM＋正常tail，未追加LoD。',
      '每组按完整轨迹平均PSNR选最高一遍，同分选repeat编号较小者。SSIM、LPIPS、ATE、Gaussian数和checkpoint均来自同一遍。所有三遍原始记录保留，不按质量删除或重跑。','',
      '|场景|模式|最佳PSNR|SSIM|LPIPS|Gaussian数|最佳repeat|三遍PSNR均值±标准差|Δ默认最佳|',
      '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for x in selected:
        lines.append(f"|{x['scene']}|{x['mode']}|{x['psnr']:.4f}|{x['ssim']:.4f}|{x['lpips']:.4f}|{x['gaussian_count']:,}|"
          f"{x['repeat']}|{x['psnr_three_run_mean']:.4f}±{x['psnr_three_run_std']:.4f}|{x['delta_default_best_psnr']:+.4f}|")
    lines+=['','|模式|最佳PSNR均值|最佳对默认最佳差值|三遍均值对默认三遍均值差值|最佳模型点数变化|',
            '|---|---:|---:|---:|---:|']
    for mode,s in summaries.items():
        lines.append(f"|{mode}|{s['best_psnr_mean']:.4f}|{s['best_delta_default_best']:+.4f}|"
          f"{s['three_run_mean_delta_default_mean']:+.4f}|{s['gaussian_total_change_vs_default_best']:+.2%}|")
    lines+=['','最佳checkpoint和评价路径见Q80_FINAL_BEST.csv／JSON；全部48遍及每组三次均值、标准差见Q80_ALL_48_RUNS.csv／JSON。最佳清单引用已有模型，不复制覆盖checkpoint。',
      'PSNR／SSIM越高越好，LPIPS越低越好。N为Full地图存储Gaussian总数，包含inactive slot，不是逐视角可见点数。Mono沿用尺度对齐，RGB-D不作尺度对齐。',
      '本报告使用各自完整轨迹均值，没有重新采样或共同帧重聚合。逐帧数值均与原汇总核对，PLY顶点数与对应N一致。',
      '同seed三遍衡量在线运行波动；异步Mapping导致曝光、位姿和实际训练步数变化。最佳对最佳用于用户指定的模型选择，不替代三次均值对照，也不构成算法稳定收益或统计显著证明。',
      'q80和alpha1.5保持独立实验配置；本次没有修改生产默认、没有commit／push、不保存渲染图片。完整性和失败情况见FINISHED.json及FINAL_VALIDATION.json。']
    (R/'reports/Q80_FINAL_REPLICA_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('FINAL_48_RUNS_AND_16_BEST_READY',summaries,flush=True)


if __name__=='__main__':main()
