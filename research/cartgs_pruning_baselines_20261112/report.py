"""导出裁剪对照的真实指标，保留各自N、K与历史来源。"""
import csv
import json
import math
from pathlib import Path
import numpy as np
from common import ROOT, RATIOS, METHODS, load, write, sha, budget


def main():
    state=load(ROOT/'EXPERIMENTS.json')
    sources=load(ROOT/'SOURCE_MANIFEST.json')
    for source,item in sources.items():
        assert sha(source)==sha(item['copy'])==item['sha256']
    rows=[]
    comparisons=[]
    for group in state['groups']:
        completed=ROOT/'runs'/group['id']/'COMPLETE.json'
        assert completed.exists(),group['id']
        result=load(completed)
        assert result['passed'] and not result['smoke'] and len(result['evaluations'])==22
        for item in result['evaluations']:
            assert item['frames']==group['vanilla']['frames']
            assert item['N']==group['vanilla']['gaussians']
            assert item['actual_K']==item['target_K']==budget(item['ratio'],item['N'])
            output=Path(item['output'])
            per_frame=load_csv(output/'per_frame_metrics.csv')
            assert len(per_frame)==item['frames']
            selected=np.load(output/'selected_ids.npy')
            ranking=np.load(completed.parent/(item['method']+'_ranking.npy'))
            assert len(selected)==len(np.unique(selected))==item['actual_K']
            assert np.array_equal(selected,np.sort(ranking[:item['actual_K']]))
            assert all(int(frame['N'])==item['N'] and int(frame['K'])==item['actual_K'] for frame in per_frame)
            for name in ['psnr','ssim','lpips']:
                assert math.isfinite(item[name])
                assert abs(sum(float(x[name]) for x in per_frame)/len(per_frame)-item[name])<1e-9
            ours=next(x for x in group['ours'] if x['scope']=='global' and float(x['ratio'])==item['ratio'])
            rows.append(dict(group=group['id'],method=item['method'],ratio=item['ratio'],
                ratio_kind=ours.get('ratio_kind'),
                seed=item['seed'],cartgs_N=item['N'],cartgs_K=item['actual_K'],
                actual_ratio=item['actual_ratio'],frames=item['frames'],
                **{name:item[name] for name in ['psnr','ssim','lpips','render_ms','seconds']},
                cartgs_full_psnr=group['vanilla']['psnr'],cartgs_full_ssim=group['vanilla']['ssim'],
                cartgs_full_lpips=group['vanilla']['lpips'],
                ours_N=ours['gaussian_count'],ours_K=ours.get('actual_K_mean'),
                ours_psnr=ours['psnr'],ours_ssim=ours['ssim'],ours_lpips=ours['lpips'],
                ours_source=group['ours_source'],checkpoint=item['checkpoint'],output=item['output']))
        comparisons.append(dict(group=group['id'],vanilla=group['vanilla'],ours=group['ours'],
                                ours_source=group['ours_source'],pruning=result['evaluations']))
    assert len(rows)==704
    reports=ROOT/'reports';reports.mkdir(exist_ok=True)
    with (reports/'PRUNING_BASELINE_METRICS.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,list(rows[0]));writer.writeheader();writer.writerows(rows)
    assert load(ROOT/'PROTECTED_BEFORE.json')==load(ROOT/'PROTECTED_AFTER.json')
    images=[str(p) for folder in ['runs','smoke'] for p in (ROOT/folder).rglob('*')
            if p.suffix.lower() in ['.png','.jpg','.jpeg','.bmp']]
    assert not images
    write(reports/'VERIFIED.json',dict(complete=True,groups=32,evaluations=704,images_saved=0,
          protected_files_unchanged=True,source_files_verified=len(sources),
          random_seeds=[0],same_ratio_not_same_K=True))
    write(reports/'COMPARISON_DATA.json',comparisons)
    with (reports/'METHODS_ALL_METRICS.csv').open('w',newline='',encoding='utf-8-sig') as f:
        fields=['group','method','ratio','N','K_mean','frames','psnr','ssim','lpips','source']
        writer=csv.DictWriter(f,fields);writer.writeheader()
        for g in comparisons:
            v=g['vanilla']
            writer.writerow(dict(group=g['group'],method='cartgs_full',ratio=1,N=v['gaussians'],
                K_mean=v['gaussians'],frames=v['frames'],psnr=v['psnr'],ssim=v['ssim'],lpips=v['lpips'],source=v['evaluation']))
            for x in g['pruning']:
                writer.writerow(dict(group=g['group'],method='cartgs_'+x['method'],ratio=x['ratio'],N=x['N'],
                    K_mean=x['actual_K'],frames=x['frames'],psnr=x['psnr'],ssim=x['ssim'],lpips=x['lpips'],source=x['output']))
            for x in g['ours']:
                writer.writerow(dict(group=g['group'],method='ours_'+x['scope'],ratio=x['ratio'],N=x['gaussian_count'],
                    K_mean=x['actual_K_mean'],frames=x.get('frames',x.get('evaluated_frames')),
                    psnr=x['psnr'],ssim=x['ssim'],lpips=x['lpips'],source=x['output']))
    lines=['# CaRtGS随机与Opacity裁剪对照结果','',
      '32组CaRtGS Vanilla在自身点数N上执行严格K=floor(r×N)裁剪，共704项完整轨迹评价。'
      '没有重新训练、补偿或保存渲染图片。所有源模型及历史指标哈希保持不变。','',
      '随机使用一次seed0 PCG64排列；Opacity按activated opacity稳定降序，选中点保持原行序。'
      '同一场景各Ratio形成嵌套子集。图像、相机和TorchMetrics评价路径沿用该CaRtGS历史评价。','',
      '本表按相同比例比较，各方法模型N不同，实际K也不同；不能称同K实验。'
      '当前方法的corner-slow／robot-normal双目引用11月11日复跑，其余沿用已完成的32组结果。','',
      '| Ratio | 随机PSNR均值 | Opacity PSNR均值 | 我们全局PSNR均值 |',
      '|---:|---:|---:|---:|']
    for ratio in RATIOS:
        data=[r for r in rows if r['ratio']==float(ratio)]
        a=[r['psnr'] for r in data if r['method']=='random']
        b=[r['psnr'] for r in data if r['method']=='opacity']
        c=[r['ours_psnr'] for r in data if r['method']=='random']
        lines.append(f'| {ratio} | {sum(a)/len(a):.6f} | {sum(b)/len(b):.6f} | {sum(c)/len(c):.6f} |')
    lines+=['','均值为32组等权宏平均，不替代逐场景结果。完整PSNR／SSIM／LPIPS、N、K与逐帧来源见CSV。',
            '随机仅一次，不报告随机方法跨seed稳定性。共享GPU下render_ms仅为本次观察值，不作公平加速结论。']
    (reports/'PRUNING_BASELINE_REPORT.md').write_text('\n'.join(lines)+'\n')
    print('REPORT_VERIFIED',len(rows),flush=True)


def load_csv(path):
    with path.open() as f:return list(csv.DictReader(f))


if __name__=='__main__':main()
