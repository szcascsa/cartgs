"""主对照使用默认三次均值，读取已有逐帧指标核验，不重新渲染。"""
from collections import defaultdict
from pathlib import Path
import csv
import math
import statistics
from prepare_experiment import R,F,PREVIOUS,read,write,sha

ADAPTIVE=R.parent/'replica_l0_l1_grad1p5_l1_l2_grad2p5_20261108'
METRICS=['psnr','ssim','lpips','gaussian_count','online_steps','tail_steps']


def records(root):
    jobs={j['id']:j for j in read(root/'QUEUE.json')};result=defaultdict(list)
    for evaluation in read(root/'TRAJECTORY_EVALUATION.json')['jobs']:
        job=jobs[evaluation['id']]
        if job['dataset']!='Replica':continue
        assert job['state']==evaluation['state']=='complete',job['id']
        m=evaluation['result'];s=job['result'];meta=read(next(Path(s['checkpoint']).rglob('anchor_selector_state.json')))
        row=dict(scene=job['scene'],mode=job['mode'],seed=job['seed'],repeat=job['repeat'],
            psnr=m['trajectory_psnr'],ssim=m['trajectory_ssim'],lpips=m['trajectory_lpips'],
            ate_rmse=m['trajectory_ate_rmse'],frames=m['evaluated_frames'],
            gaussian_count=s['gaussian_count'],anchor_count=s['anchor_count'],online_steps=s['online_steps'],
            tail_steps=s['tail_steps'],growth_tau0=meta['growth_tau0'],checkpoint=s['checkpoint'],evaluation=m['output'])
        for metric in ['psnr','ssim','lpips']:
            values=[float(x) for x in (Path(row['evaluation'])/(metric+'.txt')).read_text().splitlines() if x.strip()]
            assert len(values)==row['frames'] and all(math.isfinite(x) for x in values)
            assert abs(sum(values)/len(values)-row[metric])<1e-9
        result[job['scene'],job['mode']].append(row)
    assert len(result)==16
    return result


def main():
    current=records(R);default=records(F);fixed=records(PREVIOUS);adaptive=records(ADAPTIVE)
    assert all(len(x)==3 for x in default.values())
    levels={(x['scene'],x['mode']):x for x in read(R/'reports/ANCHOR_LEVEL_DISTRIBUTION.json')['rows']}
    groups=[];flat=[]
    for key,items in current.items():
        new,=items;last,=fixed[key];old,=adaptive[key];first,=[x for x in default[key] if x['repeat']==1]
        baseline={k:statistics.mean(x[k] for x in default[key]) for k in METRICS}
        references=dict(default_three_run_mean=baseline,default_repeat1=first,fixed_t001_alpha1p2=last,q90_1p5_2p5=old)
        differences={name:{k:new[k]-ref[k] for k in METRICS} for name,ref in references.items()}
        meta=read(next(Path(new['checkpoint']).rglob('anchor_selector_state.json')))
        assert meta['growth_quantile']==.8 and meta['growth_alpha']==1.5 and meta['growth_strict_upper']
        assert meta['growth_threshold_mode']=='quantile' and meta['growth_tau0']>0
        groups.append(dict(scene=key[0],mode=key[1],current=new,references=references,differences=differences,
             default_repeats=default[key],anchor_levels=levels[key]))
        flat.append(dict(**new,quantile=.8,alpha=1.5,strict_upper=True,
            l0_low=new['growth_tau0'],l0_high=new['growth_tau0']*1.5,
            l1_high=new['growth_tau0']*2.25,l2_high=new['growth_tau0']*3.375,
            l2_anchor_fraction=levels[key]['l2_anchor_fraction'],l2_gaussian_fraction=levels[key]['l2_gaussian_fraction'],
            default_psnr_mean=baseline['psnr'],default_psnr_std=statistics.stdev(x['psnr'] for x in default[key]),
            default_psnr_min=min(x['psnr'] for x in default[key]),default_psnr_max=max(x['psnr'] for x in default[key]),
            **{name+'_delta_'+k:delta[k] for name,delta in differences.items() for k in delta}))
    summaries={}
    for mode in ['mono','rgbd','all']:
        subset=[g for g in groups if mode=='all' or g['mode']==mode]
        summary=dict(groups=len(subset),current_means={k:statistics.mean(g['current'][k] for g in subset) for k in METRICS})
        for ref in groups[0]['references']:
            summary[ref]=dict(reference_means={k:statistics.mean(g['references'][ref][k] for g in subset) for k in METRICS},
                mean_differences={k:statistics.mean(g['differences'][ref][k] for g in subset) for k in METRICS},
                psnr_improved=sum(g['differences'][ref]['psnr']>0 for g in subset),
                gaussian_total_change=sum(g['current']['gaussian_count'] for g in subset)
                    /sum(g['references'][ref]['gaussian_count'] for g in subset)-1)
        summaries[mode]=summary
    inputs=[root/name for root in [R,F,PREVIOUS,ADAPTIVE] for name in ['QUEUE.json','TRAJECTORY_EVALUATION.json']]
    write(R/'reports/THRESHOLD_COMPARISON.json',dict(groups=groups,summaries=summaries,
          source_hashes={str(p):sha(p) for p in inputs}))
    with (R/'reports/THRESHOLD_COMPARISON.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    lines=['# Replica q80基础阈值、alpha1.5实验结果','',
      '16组seed0各一次Full SLAM＋正常tail，无LoD。t取首个有效L0有限正Full梯度窗口的q80后固定，层级倍率1.5，严格超过上界细化。',
      '主对照为原默认q90／alpha2的每组三次均值。各重复均为完整轨迹均值，先等权平均三次，再等权平均场景；不选择历史最佳。','',
      '|模式|本轮PSNR|默认三次均值|ΔPSNR|ΔSSIM|ΔLPIPS|Gaussian数量变化|PSNR提升组数|',
      '|---|---:|---:|---:|---:|---:|---:|---:|']
    for mode,s in summaries.items():
        ref=s['default_three_run_mean'];delta=ref['mean_differences']
        lines.append(f"|{mode}|{s['current_means']['psnr']:.4f}|{ref['reference_means']['psnr']:.4f}|"
           f"{delta['psnr']:+.4f}|{delta['ssim']:+.5f}|{delta['lpips']:+.5f}|"
           f"{ref['gaussian_total_change']:+.2%}|{ref['psnr_improved']}/{s['groups']}|")
    lines+=['','|场景|模式|PSNR|Δ默认三次均值|Δ默认repeat1|Δ固定0.001／1.2|SSIM|LPIPS|N|校准t|L2 Anchor占比|',
            '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in flat:
        lines.append(f"|{row['scene']}|{row['mode']}|{row['psnr']:.4f}|{row['default_three_run_mean_delta_psnr']:+.4f}|"
          f"{row['default_repeat1_delta_psnr']:+.4f}|{row['fixed_t001_alpha1p2_delta_psnr']:+.4f}|"
          f"{row['ssim']:.4f}|{row['lpips']:.4f}|{row['gaussian_count']:,}|{row['growth_tau0']:.8f}|{row['l2_anchor_fraction']:.2%}|")
    lines+=['','PSNR／SSIM越高越好，LPIPS越低越好。N为完整地图存储Gaussian数，包含inactive slot，不是逐帧可见数量。',
      '新方案每组仅一次。默认三次均值保留低质量重复，可能受到离群运行影响；逐组三次数据、范围、标准差及默认repeat1对照全部保留。没有据此宣称统计显著。',
      '本轮同时更改分位数、alpha及相对原默认的上界等号语义，尚未单独隔离三者作用。异步Tracking／Mapping造成实际预算和曝光差异；全部实际步数、ATE、帧数及路径见CSV／JSON。',
      'Mono沿用尺度对齐，RGB-D不作尺度对齐；使用各自完整轨迹均值，未重聚合共同RGB帧。所有PSNR／SSIM／LPIPS均值与原始逐帧数据核对。',
      '未保存渲染图片。完整性核验见FINISHED.json及PROTECTED_AFTER.json。']
    (R/'reports/THRESHOLD_COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Q80_ALPHA1P5_REPORT_READY',summaries,flush=True)


if __name__=='__main__':main()
