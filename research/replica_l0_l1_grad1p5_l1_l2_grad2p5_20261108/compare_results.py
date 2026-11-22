"""只读本轮与历史Full全轨迹结果；历史最佳的其他指标绑定同次重复。"""
from collections import defaultdict
from pathlib import Path
import csv
import json
import math
from prepare_experiment import R,F,PREVIOUS,read,write,sha


def full_records(root):
    jobs={j['id']:j for j in read(root/'QUEUE.json')}
    result=defaultdict(list)
    for evaluation in read(root/'TRAJECTORY_EVALUATION.json')['jobs']:
        job=jobs[evaluation['id']]
        if job['dataset']!='Replica':continue
        assert job['state']==evaluation['state']=='complete',job['id']
        m=evaluation['result'];s=job['result']
        metadata=read(next(Path(s['checkpoint']).rglob('anchor_selector_state.json')))
        row=dict(scene=job['scene'],mode=job['mode'],repeat=job['repeat'],seed=job['seed'],
                 psnr=m['trajectory_psnr'],ssim=m['trajectory_ssim'],lpips=m['trajectory_lpips'],
                 ate_rmse=m['trajectory_ate_rmse'],frames=m['evaluated_frames'],
                 gaussian_count=s['gaussian_count'],anchor_count=s['anchor_count'],
                 online_steps=s['online_steps'],tail_steps=s['tail_steps'],
                 checkpoint=s['checkpoint'],evaluation=m['output'],growth_tau0=metadata['growth_tau0'])
        for metric in ['psnr','ssim','lpips']:
            values=[float(x) for x in (Path(row['evaluation'])/(metric+'.txt')).read_text().splitlines() if x.strip()]
            assert len(values)==row['frames'] and all(math.isfinite(x) for x in values)
            assert abs(sum(values)/len(values)-row[metric])<1e-9
        result[job['scene'],job['mode']].append(row)
    assert len(result)==16
    return result


def main():
    current=full_records(R);previous=full_records(PREVIOUS);original=full_records(F)
    historical=PREVIOUS/'figures/full_vs_vanilla/data/FULL_VS_VANILLA.json'
    vanilla_data=read(historical)
    assert all(sha(Path(p))==h for p,h in vanilla_data['source_hashes'].items())
    vanilla={(g['scene'],g['mode']):g['vanilla'] for g in vanilla_data['groups']}
    levels={(g['scene'],g['mode']):g for g in read(R/'reports/ANCHOR_LEVEL_DISTRIBUTION.json')['rows']}
    groups=[];flat=[]
    for key,rows in current.items():
        new,=rows;old3,=previous[key]
        old4=max(original[key],key=lambda r:(r['psnr'],-r['repeat']))
        repeat1,=[r for r in original[key] if r['repeat']==1]
        v=vanilla[key];level=levels[key]
        marker=read(Path(new['checkpoint'])/'checkpoint_experiment.json')
        assert marker['l0_to_l1_tau0_multiplier']==1.5 and marker['l1_to_l2_tau0_multiplier']==2.5
        references=dict(previous_2_3=old3,original_2_4_repeat1=repeat1,original_2_4_best=old4,vanilla_best=v)
        differences={name:{m:new[m]-row[m] for m in ['psnr','ssim','lpips','gaussian_count']}
                     for name,row in references.items()}
        groups.append(dict(scene=key[0],mode=key[1],current_1p5_2p5=new,references=references,
                           differences=differences,anchor_levels=level))
        flat.append(dict(scene=key[0],mode=key[1],**{k:new[k] for k in ['psnr','ssim','lpips','ate_rmse',
             'frames','gaussian_count','anchor_count','online_steps','tail_steps','growth_tau0','checkpoint','evaluation']},
             l0_to_l1_threshold=new['growth_tau0']*1.5,l1_to_l2_threshold=new['growth_tau0']*2.5,
             l2_anchor_fraction=level['l2_anchor_fraction'],l2_gaussian_fraction=level['l2_gaussian_fraction'],
             **{name+'_delta_'+m:delta[m] for name,delta in differences.items() for m in delta}))
    averages={}
    for mode in ['mono','rgbd','all']:
        subset=[g for g in groups if mode=='all' or g['mode']==mode]
        averages[mode]=dict(current_mean_psnr=sum(g['current_1p5_2p5']['psnr'] for g in subset)/len(subset))
        for reference in groups[0]['references']:
            averages[mode][reference]=dict(
                mean_differences={m:sum(g['differences'][reference][m] for g in subset)/len(subset)
                                  for m in ['psnr','ssim','lpips']},
                psnr_improved=sum(g['differences'][reference]['psnr']>0 for g in subset),
                total_gaussian_change=sum(g['current_1p5_2p5']['gaussian_count'] for g in subset)
                   /sum(g['references'][reference]['gaussian_count'] for g in subset)-1)
    write(R/'reports/THRESHOLD_COMPARISON.json',dict(groups=groups,averages=averages,
          source_hashes={str(p):sha(p) for p in [historical,R/'QUEUE.json',R/'TRAJECTORY_EVALUATION.json',
                         PREVIOUS/'QUEUE.json',PREVIOUS/'TRAJECTORY_EVALUATION.json',F/'QUEUE.json',F/'TRAJECTORY_EVALUATION.json']}))
    with (R/'reports/THRESHOLD_COMPARISON.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    lines=['# Replica 1.5／2.5倍阈值实验结果','',
       '16组seed0单次Full SLAM＋正常tail；未训练LoD。仅覆盖L0→L1和L1→L2的增生上界，分别为1.5τ₀和2.5τ₀。',
       'PSNR／SSIM越大越好，LPIPS越小越好。下表质量均为各自完整轨迹均值，N为Full地图存储Gaussian总数。','',
       '|场景|模式|PSNR|Δ上轮2／3|Δ原2／4最佳|ΔVanilla最佳|SSIM|LPIPS|N|L2 Anchor占比|',
       '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for row in flat:
        lines.append(f"|{row['scene']}|{row['mode']}|{row['psnr']:.4f}|{row['previous_2_3_delta_psnr']:+.4f}|"
          f"{row['original_2_4_best_delta_psnr']:+.4f}|{row['vanilla_best_delta_psnr']:+.4f}|"
          f"{row['ssim']:.4f}|{row['lpips']:.4f}|{row['gaussian_count']:,}|{row['l2_anchor_fraction']:.2%}|")
    lines+=['','|模式|本轮平均PSNR|Δ上轮2／3|Δ原2／4同repeat1|Δ原2／4最佳|ΔVanilla最佳|','|---|---:|---:|---:|---:|---:|']
    for mode,a in averages.items():
        lines.append('|'+mode+f"|{a['current_mean_psnr']:.4f}|"+'|'.join(
             f"{a[ref]['mean_differences']['psnr']:+.4f}" for ref in groups[0]['references'])+'|')
    lines+=['','本轮和上轮2／3阈值各运行一次，原2／4与Vanilla最佳按每场景／模式的三次Full全轨迹PSNR选取，其他指标绑定同次。另提供原2／4同repeat1对照，不能将单次对最佳的差值解释为稳定阈值收益。',
       '异步Tracking／Mapping会改变实际online步数、相机曝光、初始化与τ₀校准值。训练步数、逐组ATE／评价帧数／模型路径及所有差值见CSV／JSON；未进行共同RGB帧重聚合。',
       '所有完整轨迹均值与原始逐帧文件核对。未保存渲染图片；生产与旧checkpoint完整性核验见FINISHED.json和PROTECTED_AFTER.json。']
    (R/'reports/THRESHOLD_COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('THRESHOLD_COMPARISON_READY',averages,flush=True)


if __name__=='__main__':main()
