"""只读完整轨迹指标，比较固定阈值与历史对照，核验逐帧均值。"""
from collections import defaultdict
from pathlib import Path
import csv
import math
from prepare_experiment import R,F,PREVIOUS,OLDER,read,write,sha


def full_records(root):
    jobs={j['id']:j for j in read(root/'QUEUE.json')}
    result=defaultdict(list)
    for evaluation in read(root/'TRAJECTORY_EVALUATION.json')['jobs']:
        job=jobs[evaluation['id']]
        if job['dataset']!='Replica':continue
        assert job['state']==evaluation['state']=='complete',job['id']
        m=evaluation['result'];s=job['result'];metadata=read(next(Path(s['checkpoint']).rglob('anchor_selector_state.json')))
        row=dict(scene=job['scene'],mode=job['mode'],repeat=job['repeat'],seed=job['seed'],
            psnr=m['trajectory_psnr'],ssim=m['trajectory_ssim'],lpips=m['trajectory_lpips'],
            ate_rmse=m['trajectory_ate_rmse'],frames=m['evaluated_frames'],gaussian_count=s['gaussian_count'],
            anchor_count=s['anchor_count'],online_steps=s['online_steps'],tail_steps=s['tail_steps'],
            checkpoint=s['checkpoint'],evaluation=m['output'],growth_tau0=metadata['growth_tau0'])
        for metric in ['psnr','ssim','lpips']:
            values=[float(x) for x in (Path(row['evaluation'])/(metric+'.txt')).read_text().splitlines() if x.strip()]
            assert len(values)==row['frames'] and all(math.isfinite(x) for x in values)
            assert abs(sum(values)/len(values)-row[metric])<1e-9
        result[job['scene'],job['mode']].append(row)
    assert len(result)==16
    return result


def main():
    current=full_records(R);previous=full_records(PREVIOUS);older=full_records(OLDER);original=full_records(F)
    historical=OLDER/'figures/full_vs_vanilla/data/FULL_VS_VANILLA.json';vdata=read(historical)
    assert all(sha(Path(p))==h for p,h in vdata['source_hashes'].items())
    vanilla={(g['scene'],g['mode']):g['vanilla'] for g in vdata['groups']}
    levels={(g['scene'],g['mode']):g for g in read(R/'reports/ANCHOR_LEVEL_DISTRIBUTION.json')['rows']}
    groups=[];flat=[]
    for key,rows in current.items():
        new,=rows;last,=previous[key];old,=older[key]
        best=max(original[key],key=lambda x:(x['psnr'],-x['repeat']))
        repeat1,=[x for x in original[key] if x['repeat']==1]
        metadata=read(next(Path(new['checkpoint']).rglob('anchor_selector_state.json')))
        assert metadata['growth_tau0']==.001 and metadata['growth_alpha']==1.2
        assert metadata['growth_threshold_mode']=='manual' and metadata['growth_strict_upper']
        references=dict(previous_1p5_2p5=last,older_2_3=old,default_2_4_repeat1=repeat1,
                        default_2_4_best=best,vanilla_best=vanilla[key])
        differences={name:{metric:new[metric]-ref[metric] for metric in ['psnr','ssim','lpips','gaussian_count']}
                     for name,ref in references.items()}
        groups.append(dict(scene=key[0],mode=key[1],current_fixed=new,references=references,
                           differences=differences,anchor_levels=levels[key]))
        flat.append(dict(**new,alpha=1.2,strict_upper=True,l0_low=.001,l0_high=.0012,
             l1_low=.0012,l1_high=.00144,l2_low=.00144,l2_high=.001728,
             l2_anchor_fraction=levels[key]['l2_anchor_fraction'],l2_gaussian_fraction=levels[key]['l2_gaussian_fraction'],
             **{name+'_delta_'+metric:delta[metric] for name,delta in differences.items() for metric in delta}))
    averages={}
    for mode in ['mono','rgbd','all']:
        subset=[g for g in groups if mode=='all' or g['mode']==mode]
        summary=dict(current_mean_psnr=sum(g['current_fixed']['psnr'] for g in subset)/len(subset))
        for reference in groups[0]['references']:
            summary[reference]=dict(mean_differences={metric:sum(g['differences'][reference][metric] for g in subset)/len(subset)
                    for metric in ['psnr','ssim','lpips']},
                psnr_improved=sum(g['differences'][reference]['psnr']>0 for g in subset),
                total_gaussian_change=sum(g['current_fixed']['gaussian_count'] for g in subset)
                   /sum(g['references'][reference]['gaussian_count'] for g in subset)-1)
        averages[mode]=summary
    sources=[historical]+[root/name for root in [R,F,PREVIOUS,OLDER] for name in ['QUEUE.json','TRAJECTORY_EVALUATION.json']]
    write(R/'reports/THRESHOLD_COMPARISON.json',dict(groups=groups,averages=averages,
          source_hashes={str(p):sha(p) for p in sources}))
    with (R/'reports/THRESHOLD_COMPARISON.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
    lines=['# Replica固定t=0.001、alpha=1.2实验结果','',
      '16组seed0单次Full SLAM＋正常tail，未追加LoD。使用固定基础阈值，严格超过每层上界才细化，不做q90校准。',
      'PSNR／SSIM越大越好，LPIPS越小越好；指标为各自完整轨迹均值，N为Full地图存储Gaussian数量。','',
      '|场景|模式|PSNR|Δ默认2／4最佳|Δ上轮1.5／2.5|SSIM|LPIPS|N|L2 Anchor占比|',
      '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in flat:
        lines.append(f"|{row['scene']}|{row['mode']}|{row['psnr']:.4f}|{row['default_2_4_best_delta_psnr']:+.4f}|"
          f"{row['previous_1p5_2p5_delta_psnr']:+.4f}|{row['ssim']:.4f}|{row['lpips']:.4f}|"
          f"{row['gaussian_count']:,}|{row['l2_anchor_fraction']:.2%}|")
    lines+=['','|模式|本轮平均PSNR|Δ默认2／4同repeat1|Δ默认2／4最佳|Δ上轮1.5／2.5|Δ更早2／3|ΔVanilla最佳|',
            '|---|---:|---:|---:|---:|---:|---:|']
    for mode,a in averages.items():
        lines.append('|'+mode+f"|{a['current_mean_psnr']:.4f}|"+'|'.join(f"{a[ref]['mean_differences']['psnr']:+.4f}"
          for ref in ['default_2_4_repeat1','default_2_4_best','previous_1p5_2p5','older_2_3','vanilla_best'])+'|')
    lines+=['','本轮和两轮阈值实验各一遍，原默认及Vanilla最佳选自各组seed0三次中Full全轨迹PSNR最高的一遍，SSIM／LPIPS／N绑定同次。不得将新单次对历史最佳的差值直接解释为稳定因果收益。',
      '异步SLAM产生实际Mapping步数／相机曝光差异；本轮同时改变基础阈值来源、层级倍率及等号边界，尚未单独隔离每项因素。Mono沿用尺度对齐，RGB-D不作尺度对齐；未进行共同RGB帧重聚合。',
      '所有PSNR／SSIM／LPIPS均值已与原始逐帧文件核对。ATE、实际步数、阈值、层级数量、模型及指标路径和完整差值见CSV／JSON。',
      '不保存渲染图片。原项目及历史checkpoint的完整性核验见FINISHED.json和PROTECTED_AFTER.json。']
    (R/'reports/THRESHOLD_COMPARISON.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('FIXED_THRESHOLD_COMPARISON_READY',averages,flush=True)


if __name__=='__main__':main()
