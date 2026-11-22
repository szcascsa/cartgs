"""逐视角工作量和历史独占GPU耗时配对；场景等权汇总，不生成平滑或合成数据。"""
from pathlib import Path
import csv
import gzip
import hashlib
import json

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent
PRIOR = ROOT.parent/'all_datasets_joint182_ratio_speedup_20261115'
RATIOS = [.01, .03, .05, .08, .1, .2, .3, .4, .5, .6, .7]
METHODS = ['ours_full']+[f'ratio_{round(r*100):02d}' for r in RATIOS]
METRICS = ['K', 'projected_gaussians', 'gaussian_tile_instances', 'potential_valid_pixel_pairs',
    'scheduled_pixel_pair_slots', 'active_pixel_tests', 'effective_alpha_blends']
MORE = ['alpha_blends_per_pixel', 'active_tests_per_pixel', 'scheduled_slots_per_pixel',
    'covered_pixels', 'terminated_pixels', 'overflow_tiles', 'nonempty_tiles', 'max_tile_gaussians',
    'mean_tile_gaussians', 'std_tile_gaussians', 'prior_raster_median_ms', 'M', 'eligible']


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def correlation(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if x.std() == 0 or y.std() == 0:
        return None, None
    return float(np.corrcoef(x, y)[0, 1]), float(spearmanr(x, y).statistic)


def main():
    queue = json.loads((ROOT/'QUEUE.json').read_text())
    assert queue['state'] == 'complete' and all(r['formal']=='complete' for r in queue['scenes'])
    scenes = json.loads((PRIOR/'SCENES_FROZEN.json').read_text())
    assert len(scenes) == 32
    reports = ROOT/'reports'
    reports.mkdir(exist_ok=True)
    summary, correlations, checks, provenance = [], [], [], {}
    raw_path = reports/'PER_FRAME_WORKLOAD.csv.gz'
    raw = gzip.open(raw_path, 'wt', encoding='utf-8', newline='', compresslevel=5)
    raw_writer = None
    total_rows, total_frames, reference_checks = 0, 0, 0
    prior_protected = {}
    for scene in scenes:
        run = ROOT/'runs'/scene['slug']/'formal'
        validation = json.loads((run/'VALIDATION.json').read_text())
        assert validation['passed'] and validation['all_rows_diagnostic_RGB_bitwise']
        assert validation['all_rows_prior_ids_K_exact'] and validation['protected_unchanged']
        assert validation['saved_render_images'] == 0 and not validation['training']
        assert sha(run/'WORKLOAD.csv') == validation['workload_csv_sha256']
        prior_protected.update(json.loads((PRIOR/'runs'/scene['slug']/'formal/PROTECTED_BEFORE.json').read_text()))
        count = scene['coverage']['evaluated_views']
        assert validation['frames'] == count and validation['rows'] == count*len(METHODS)
        values = {key: np.empty((len(METHODS), count), np.float64) for key in METRICS+MORE}
        seen = np.zeros((len(METHODS), count), np.int8)
        current_full = None
        with (run/'WORKLOAD.csv').open(newline='', encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                frame, method = int(row['frame']), row['method']
                mi = METHODS.index(method)
                assert row['scene'] == scene['id'] and row['diagnostic_RGB_bitwise'] == 'True'
                assert int(row['N']) == scene['gaussians'] and int(row['timing_repeats']) == 3
                seen[mi, frame] += 1
                for key in values:
                    values[key][mi, frame] = float(row[key])
                if method == 'ours_full':
                    current_full = row.copy()
                assert current_full is not None and current_full['frame'] == row['frame']
                for key in METRICS:
                    denominator = int(current_full[key])
                    assert denominator > 0, (scene['id'], frame, key)
                    row['complete_'+key] = denominator
                    row['relative_'+key] = int(row[key])/denominator
                row['complete_raster_median_ms'] = float(current_full['prior_raster_median_ms'])
                row['relative_raster_latency'] = float(row['prior_raster_median_ms'])/row['complete_raster_median_ms']
                row['raster_speedup'] = 1./row['relative_raster_latency']
                if raw_writer is None:
                    raw_writer = csv.DictWriter(raw, list(row))
                    raw_writer.writeheader()
                raw_writer.writerow(row)
                total_rows += 1
        assert (seen == 1).all() and all(np.isfinite(v).all() for v in values.values())
        assert all((v[0] > 0).all() for key, v in values.items() if key in METRICS)
        normalized = {key: values[key]/values[key][0:1] for key in METRICS}
        relative_time = values['prior_raster_median_ms']/values['prior_raster_median_ms'][0:1]
        for mi, method in enumerate(METHODS):
            row = dict(scene=scene['id'], dataset=scene['dataset_kind'], scene_name=scene['scene'], mode=scene['mode'],
                method=method, ratio=1. if mi==0 else RATIOS[mi-1], N=scene['gaussians'], frames=count,
                width=scene['calibration']['width'], height=scene['calibration']['height'],
                mean_raster_speedup=float((1./relative_time[mi]).mean()),
                mean_relative_raster_latency=float(relative_time[mi].mean()))
            for key in METRICS:
                v, r = values[key][mi], normalized[key][mi]
                row.update({key+'_mean':float(v.mean()), key+'_sum':int(v.sum()),
                    'relative_'+key+'_mean':float(r.mean()), 'relative_'+key+'_p05':float(np.quantile(r,.05)),
                    'relative_'+key+'_p95':float(np.quantile(r,.95)),
                    'relative_'+key+'_ratio_of_sums':float(v.sum()/values[key][0].sum()),
                    key+'_higher_than_complete_views':int((r>1).sum())})
            for key in MORE:
                row[key+'_mean'] = float(values[key][mi].mean())
            summary.append(row)
            for key in METRICS:
                pearson, spearman = correlation(values[key][mi], values['prior_raster_median_ms'][mi])
                correlations.append(dict(scene=scene['id'], dataset=scene['dataset_kind'], mode=scene['mode'],
                    scope='within_ratio_across_views', method=method, metric=key, samples=count,
                    pearson=pearson, spearman=spearman))
        for key in METRICS:
            for scope, x, y in [
                ('all_methods_raw_same_scene', values[key].ravel(), values['prior_raster_median_ms'].ravel()),
                ('all_ratios_relative_to_same_view_complete', normalized[key][1:].ravel(), relative_time[1:].ravel())]:
                pearson, spearman = correlation(x, y)
                correlations.append(dict(scene=scene['id'], dataset=scene['dataset_kind'], mode=scene['mode'],
                    scope=scope, method='all', metric=key, samples=len(x), pearson=pearson, spearman=spearman))
        total_frames += count
        reference_checks += validation['production_RGB_radii_ids_checked']
        checks.append(dict(scene=scene['id'], frames=count, rows=validation['rows'],
            diagnostic_RGB_bitwise=True, prior_selected_ids_K_exact=True,
            production_reference_checks=validation['production_RGB_radii_ids_checked'],
            original_counter_checks=validation['existing_counter_checked'], protected_unchanged=True,
            peak_allocated_bytes=validation['peak_allocated_bytes'], peak_reserved_bytes=validation['peak_reserved_bytes']))
        for path in [run/'WORKLOAD.csv',run/'VALIDATION.json',run/'REFERENCE_CHECKS.json',run/'PROTOCOL.json']:
            provenance[str(path)] = sha(path)
        print(json.dumps(dict(scene=scene['id'], rows=validation['rows'], complete=len(checks))),flush=True)
    raw.close()
    groups = [('Replica','mono'),('Replica','rgbd'),('TUM_RGBD','mono'),('TUM_RGBD','rgbd'),('VECtor','mono'),('VECtor','stereo')]
    group_rows = []
    for dataset, mode in groups:
        for method in METHODS:
            subset = [r for r in summary if (r['dataset'],r['mode'],r['method'])==(dataset,mode,method)]
            row = dict(dataset=dataset,mode=mode,method=method,ratio=subset[0]['ratio'],scenes=len(subset),
                total_views=sum(r['frames'] for r in subset), scene_equal_weight=True)
            fields = ['mean_raster_speedup','mean_relative_raster_latency']
            fields += [key+'_mean' for key in METRICS+MORE]
            fields += ['relative_'+key+'_mean' for key in METRICS]
            for key in fields:
                row[key] = float(np.mean([r[key] for r in subset]))
            group_rows.append(row)
    corr_summary = []
    for scope in ['all_methods_raw_same_scene','all_ratios_relative_to_same_view_complete','within_ratio_across_views']:
        for key in METRICS:
            subset = [r for r in correlations if r['scope']==scope and r['metric']==key and r['pearson'] is not None]
            corr_summary.append(dict(scope=scope,metric=key,records=len(subset),
                mean_pearson=float(np.mean([r['pearson'] for r in subset])),
                median_pearson=float(np.median([r['pearson'] for r in subset])),
                mean_spearman=float(np.mean([r['spearman'] for r in subset]))))
    common = [r for r in correlations if r['scope']=='within_ratio_across_views' and r['method']!='ours_full']
    valid_keys = {(r['scene'],r['method']) for r in common if r['pearson'] is not None}
    for key in METRICS:
        valid_keys &= {(r['scene'],r['method']) for r in common if r['metric']==key and r['pearson'] is not None}
    for key in METRICS:
        subset = [r for r in common if r['metric']==key and (r['scene'],r['method']) in valid_keys]
        corr_summary.append(dict(scope='within_lod_ratio_across_views_common_cases',metric=key,records=len(subset),
            mean_pearson=float(np.mean([r['pearson'] for r in subset])),
            median_pearson=float(np.median([r['pearson'] for r in subset])),
            mean_spearman=float(np.mean([r['spearman'] for r in subset]))))
    write(reports/'SCENE_RATIO_WORKLOAD.csv',summary)
    write(reports/'SIX_CURVE_DATA.csv',group_rows)
    write(reports/'CORRELATIONS.csv',correlations)
    write(reports/'CORRELATION_SUMMARY.csv',corr_summary)
    write(reports/'COVERAGE_VALIDATION.csv',checks)
    assert all(sha(path)==value for path,value in prior_protected.items()), '先前测速依赖文件发生变化'
    image_files = [p for p in ROOT.rglob('*') if p.suffix.lower() in ['.png','.jpg','.jpeg','.exr']]
    assert not image_files, image_files
    validation = dict(passed=True,groups=len(scenes),dataset_modes=len(groups),views=total_frames,
        workload_rows=total_rows,ratios=RATIOS,complete_map=True,timing_repeats=3,
        diagnostic_RGB_bitwise_rows=total_rows,prior_ids_K_exact_rows=total_rows,
        production_RGB_radii_reference_checks=reference_checks,
        protected_unchanged=True,prior_timing_protected_files_verified=len(prior_protected),
        production_changed=False,training=False,saved_render_images=0,
        figures_generated=False,raw_data_sha256=sha(raw_path),sources=provenance,
        summary_sha256=sha(reports/'SCENE_RATIO_WORKLOAD.csv'),group_data_sha256=sha(reports/'SIX_CURVE_DATA.csv'))
    assert total_frames == 59507 and total_rows == total_frames*12 and len(group_rows)==72
    (reports/'VALIDATION.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2)+'\n')
    (reports/'DATA_MANIFEST.md').write_text(
        '# 工作量数据清单\n\n全部为真实计数，无模拟、插值或平滑。\n\n'
        '|文件|内容|\n|---|---|\n'
        '|PER_FRAME_WORKLOAD.csv.gz|32组、有效全轨迹、Complete Map及11档ratio；原始计数、同视角基准、归一化工作量、历史三轮中位耗时、选点ID哈希|\n'
        '|SCENE_RATIO_WORKLOAD.csv|每场景/相机模式/ratio的原始均值、总量、逐帧归一化均值与分位数|\n'
        '|SIX_CURVE_DATA.csv|后续六条曲线的数据，同数据集同模式内场景等权|\n'
        '|CORRELATIONS.csv|逐场景、逐ratio及混合ratio的工作量–耗时描述性相关性|\n'
        '|CORRELATION_SUMMARY.csv|各相关性口径的汇总，不作因果推断|\n'
        '|COVERAGE_VALIDATION.csv / VALIDATION.json|逐场景验收、覆盖、保护文件和来源哈希|\n\n'
        '计数实现：build_diagnostic.py生成独立CUDA扩展；采集：collect.py；汇总：analyze.py。'
        '各次渲染使用冻结Student、名义Anchor局部候选池和同一mode182配置；不训练、不保存渲染图片。'
        '基准名称：Our Complete Map (1×)。\n',encoding='utf-8')
    paragraphs = [
        '# mode182 渲染工作量实测报告\n',
        f'完成三数据集32组，共{total_frames:,}个有效全轨迹视角、{total_rows:,}条工作量记录。'
        '使用原始分辨率、各组已冻结的Student与匹配Master，ratio为1%、3%、5%、8%、10%、20%、30%、40%、50%、60%、70%，另测完整地图。',
        '工作量计数扩展独立构建。每条记录的诊断RGB与执行路径逐位一致，选点身份及K与此前测速一致；'
        f'额外对{reference_checks:,}个包与原发布二进制比较RGB/radii/选点，并与已有Alpha计数和Tile快照交叉验证。'
        '未改训练或生产版本，未保存渲染图片。',
        '计数采集允许三进程并发，因为计数不依赖GPU占用。耗时全部引用上一轮相同输入、相同选点的三轮独占GPU测量中位数，'
        '不把诊断计数开销当作渲染速度。VECtor Stereo表示双目训练模型的单张左图，不是左右两张总量。',
        '## 指标定义\n',
        '- `effective_alpha_blends`：有效图像像素上，执行了非零 `T×alpha` 权重合成的Gaussian–像素对数。计的是合成次数，不是Alpha或贡献权重的和。',
        '- `active_pixel_tests`：有效像素尚未结束时实际测试的Gaussian–像素对数，包括后来被Alpha阈值或形状条件拒绝的候选。',
        '- `scheduled_pixel_pair_slots`：合成循环实际遍历的像素槽位，包括Warp尚未共同退出时已结束的像素和边界无效像素；是执行槽位代理，不是精确指令数。',
        '- `gaussian_tile_instances`：Gaussian–Tile成员关系数量，反映投影footprint和成员构建负载。',
        '- `potential_valid_pixel_pairs`：每Tile的Gaussian成员数乘以Tile内有效像素数再求和，为不考虑逐像素提前结束的潜在交互量。',
        '另保留K、投影有效点数、覆盖/终止像素、重Tile/溢出Tile数量及每像素归一化计数。所有计数遵循mode182自己的阈值与溢出回退路径，不能跨渲染器直接视为同一工作。',
        '## 各ratio工作量变化\n',
        '下表是逐帧除以同视角 Our Complete Map (1×) 后，先场景内平均，再按场景等权平均。百分比越低，工作量越少；不是保留点数百分比。\n',
        '|数据集/模式|ratio|Alpha合成量|有效测试量|执行槽位|Tile展开量|纯渲染加速|\n|---|---:|---:|---:|---:|---:|---:|']
    for row in group_rows:
        if row['ratio'] in [.01,.1,.3,.5,.7]:
            paragraphs[-1] += (f"\n|{row['dataset']}/{row['mode']}|{row['ratio']:.0%}|{row['relative_effective_alpha_blends_mean']:.1%}|"
                f"{row['relative_active_pixel_tests_mean']:.1%}|{row['relative_scheduled_pixel_pair_slots_mean']:.1%}|"
                f"{row['relative_gaussian_tile_instances_mean']:.1%}|{row['mean_raster_speedup']:.3f}×|")
    paragraphs += ['\n## 与耗时的关系\n',
        '下表先把每帧工作量及耗时各自除以同视角完整地图的值，再在每个场景内合并11档ratio计算相关系数，最后对32场景等权平均。'
        '这是与ratio–工作量图对应的口径。原始值相关性、每ratio固定条件结果及各指标共同有效样本的结果也保存在CSV中。相关性不能证明因果或线性比例。\n',
        '|指标|平均Pearson r|平均Spearman rho|\n|---|---:|---:|']
    for row in corr_summary:
        if row['scope']=='all_ratios_relative_to_same_view_complete':
            paragraphs[-1] += f"\n|{row['metric']}|{row['mean_pearson']:.4f}|{row['mean_spearman']:.4f}|"
    norm = {r['metric']:r for r in corr_summary if r['scope']=='all_ratios_relative_to_same_view_complete'}
    raw_corr = {r['metric']:r for r in corr_summary if r['scope']=='all_methods_raw_same_scene'}
    fixed = {r['metric']:r for r in corr_summary if r['scope']=='within_lod_ratio_across_views_common_cases'}
    paragraphs += ['\n## 指标是否适合\n',
        'Alpha合成次数适合描述有效合成工作，但本轮没有证据支持它全面优于高斯点数预测耗时。'
        f"归一化口径的平均Pearson r：有效测试量{norm['active_pixel_tests']['mean_pearson']:.3f}、Alpha合成次数{norm['effective_alpha_blends']['mean_pearson']:.3f}、K为{norm['K']['mean_pearson']:.3f}。"
        f"在每场景原始值混合12种条件的口径下，Alpha为{raw_corr['effective_alpha_blends']['mean_pearson']:.3f}，K为{raw_corr['K']['mean_pearson']:.3f}，投影有效点数为{raw_corr['projected_gaussians']['mean_pearson']:.3f}。"
        '不同口径的指标排序会变化，不能只挑一个有利相关性宣称其普遍最好。',
        f"进一步固定LoD ratio，只在{fixed['K']['records']}个所有指标都有定义的场景/ratio组合中比较跨视角变化："
        f"执行槽位的平均Pearson r为{fixed['scheduled_pixel_pair_slots']['mean_pearson']:.3f}，有效测试量为{fixed['active_pixel_tests']['mean_pearson']:.3f}，"
        f"Alpha合成次数为{fixed['effective_alpha_blends']['mean_pearson']:.3f}。"
        '论文建议同时报告有效候选测试量、Tile展开量和实测毫秒，避免把有效贡献次数等同于完整GPU成本。',
        '限制：被拒绝的候选也耗费计算，Warp尾部会继续执行，Tile溢出可能切换实现；排序、预处理、图像输出和固定启动开销都未包含在有效Alpha次数中。'
        '低ratio减少遮挡、Compensator扩大footprint时，工作量可能不按K同比下降，也不保证逐ratio单调。不能据此宣称减少90%点数就减少90%工作或达到10倍加速。',
        '下一张ratio–工作量图建议主轴使用 `relative_active_pixel_tests_mean`，另用Alpha合成量作为有效工作量对照；'
        '六条线沿用数据集×相机模式，1×基准明确标注 `Our Complete Map (1×)`。原始逐帧数据允许后续选择其他统计口径，不需要再次渲染。']
    (ROOT/'MODE182_RENDERING_WORKLOAD_REPORT.md').write_text('\n\n'.join(paragraphs)+'\n',encoding='utf-8')
    print('WORKLOAD_ANALYSIS_PASS',json.dumps({k:v for k,v in validation.items() if k!='sources'}),flush=True)


if __name__=='__main__':
    main()
