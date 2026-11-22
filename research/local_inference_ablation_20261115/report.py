"""导出每场景每局部Ratio的推理消融及其原始来源。"""
from pathlib import Path
import argparse
import csv
import json
import math
import subprocess
import sys
import zipfile
from common import ROOT, RATIOS, VARIANTS, SOURCE, SOURCE_SHA, records, load, write, sha


def main(final=False):
    references = records()
    completed = []
    for group in dict.fromkeys(row['group'] for row in references):
        marker = ROOT/'runs'/group/'GROUP_COMPLETE.json'
        if marker.exists():
            completed.append(load(Path(load(marker)['output'])/'COMPLETE.json'))
    results = [row for group in completed for row in group['results']]
    report = ROOT/'reports'
    report.mkdir(exist_ok=True)
    if results:
        with (report/'LOCAL_ABLATION_METRICS.csv').open('w',newline='',encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream,list(results[0]))
            writer.writeheader(); writer.writerows(results)
        methods = []
        for group in completed:
            for ratio in RATIOS:
                pair = [row for row in group['results'] if row['ratio'] == ratio]
                first = pair[0]
                for name, item in [('selector_with_comp', None), *[(row['variant'],row) for row in pair]]:
                    row = dict(group=group['group'],method=name,ratio=ratio,ratio_kind=first['ratio_kind'],
                               N=first['N'],K_mean=first['K_mean'],frames=first['frames'])
                    for metric in ['psnr','ssim','lpips']:
                        row[metric] = first['baseline_'+metric] if item is None else item[metric]
                    row['source'] = first['baseline_source'] if item is None else item['output']
                    methods.append(row)
        with (report/'LOCAL_ALL_METHODS.csv').open('w',newline='',encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream,list(methods[0]))
            writer.writeheader();writer.writerows(methods)
    comparisons = []
    for group in completed:
        for ratio in RATIOS:
            pair = [row for row in group['results'] if row['ratio'] == ratio]
            assert len(pair) == 2 and pair[0]['K_mean'] == pair[1]['K_mean']
            row = dict(group=group['group'],ratio=ratio,N=pair[0]['N'],K_mean=pair[0]['K_mean'],
                       frames=pair[0]['frames'],baseline_source=pair[0]['baseline_source'])
            for name in ['psnr','ssim','lpips']:
                row['baseline_'+name] = pair[0]['baseline_'+name]
                for item in pair:
                    assert math.isfinite(item[name])
                    row[item['variant']+'_'+name] = item[name]
                    row[item['variant']+'_delta_'+name] = item['delta_'+name]
            comparisons.append(row)
    if comparisons:
        with (report/'LOCAL_ABLATION_COMPARISON.csv').open('w',newline='',encoding='utf-8-sig') as stream:
            writer = csv.DictWriter(stream,list(comparisons[0]))
            writer.writeheader();writer.writerows(comparisons)
    lines = ['# 局部Ratio推理消融结果','',f'已完成{len(completed)}／32组，新增{len(results)}／704项评价。','',
        '完整方案复用冻结checkpoint的既有局部指标。关闭Compensator后，所选点使用Fine终点Master属性。'
        '随机选择保留原Compensator及真实视角输入。三种方法逐视角使用相同候选池、实际K、Ratio、相机与原图。','',
        '随机使用当前冻结Renderer的seed0固定Gaussian优先级，在合格候选中均匀无放回选择，Ratio之间嵌套。'
        '每个场景仅有一次随机实验，统计范围限于本次抽样。','',
        '本次测量已训练网络在推理时关闭模块或改变选择集合后的结果。Master保留与Selector、Compensator共同训练的历史。','',
        'PSNR／SSIM越高越好，LPIPS越低越好。数值是全轨迹逐帧算术平均，沿用CaRtGS未校正原图口径。'
        '六个训练Ratio：[.01,.05,.1,.3,.5,.7]；未见Ratio：[.03,.08,.2,.4,.6]。','']
    if completed:
        lines += ['## 各Ratio的场景平均','',
            f'下面先对每组完整轨迹求均值，再对已完成的{len(completed)}组等权平均。', '',
            '| Ratio | 完整PSNR | 无补偿PSNR | 随机＋补偿PSNR | 完整SSIM | 无补偿SSIM | 随机＋补偿SSIM | 完整LPIPS | 无补偿LPIPS | 随机＋补偿LPIPS |',
            '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for ratio in RATIOS:
            selected = [row for row in comparisons if row['ratio'] == ratio]
            values = [math.fsum(row[prefix+name] for row in selected)/len(selected)
                      for name in ['psnr','ssim','lpips']
                      for prefix in ['baseline_','selector_no_comp_','random_with_comp_']]
            lines.append(f'| {ratio:g} | '+' | '.join(f'{value:.6f}' for value in values)+' |')
        lines.append('')
    for group in completed:
        lines += [f"## {group['group']}",'',
            '| Ratio | 平均K | 完整PSNR | 无补偿PSNR | 随机＋补偿PSNR | 完整SSIM | 无补偿SSIM | 随机＋补偿SSIM | 完整LPIPS | 无补偿LPIPS | 随机＋补偿LPIPS |',
            '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
        for row in comparisons:
            if row['group'] != group['group']:
                continue
            values = [row[prefix+name] for name in ['psnr','ssim','lpips']
                      for prefix in ['baseline_','selector_no_comp_','random_with_comp_']]
            lines.append(f"| {row['ratio']:g} | {row['K_mean']:.2f} | "+' | '.join(f'{value:.6f}' for value in values)+' |')
        lines += ['',f"地图N={group['results'][0]['N']}，评价帧数={group['frames']}，逐帧原选点ID与K核验通过。",'']
    (report/'LOCAL_INFERENCE_ABLATION_REPORT.md').write_text('\n'.join(lines)+'\n')
    write(report/'PROGRESS.json',dict(groups=len(completed),evaluations=len(results),total_groups=32,total_evaluations=704))
    if final:
        assert len(completed) == 32 and len(results) == 704 and len(comparisons) == 352
        subprocess.run([sys.executable, '-B', str(ROOT/'verify_results.py'), '--final'], check=True)
        before = load(ROOT/'PROTECTED_BEFORE.json')
        after = {name:sha(name) for name in before}
        assert before == after
        write(ROOT/'PROTECTED_AFTER.json',after)
        assert not [path for path in (ROOT/'runs').rglob('*') if path.suffix.lower() in ['.png','.jpg','.jpeg']]
        verification = dict(complete=True,groups=32,new_evaluations=704,comparison_rows=352,
            protected_files=len(before),protected_hashes_unchanged=True,training_steps=0,images_saved=0,
            all_frame_ids_and_K_verified=all(group['all_frame_ids_and_K_verified'] for group in completed),
            source_sha256=SOURCE_SHA,total_reference_selection_checks=sum(group['frames']*11 for group in completed),
            reference_max_error={name:max(group['historical_max_error'][name] for group in completed)
                                 for name in ['psnr','ssim','lpips']})
        write(report/'VERIFIED.json',verification)
        with zipfile.ZipFile(ROOT/'LOCAL_INFERENCE_ABLATION_RESULTS.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for path in report.rglob('*'):
                if path.is_file():archive.write(path,path.relative_to(ROOT))
            for name in ['PROTOCOL.md','PREPARED.json','REFERENCE_RECORDS.json',
                         'PROTECTED_BEFORE.json','PROTECTED_AFTER.json']:
                archive.write(ROOT/name,name)
        print('ALL_ABLATIONS_VERIFIED',json.dumps(verification),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--final',action='store_true')
    args = parser.parse_args();main(args.final)
