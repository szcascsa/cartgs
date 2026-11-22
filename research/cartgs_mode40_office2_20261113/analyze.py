"""核验同轮测量，输出两侧均使用Mode40的未平滑FPS及三联图。"""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from scipy.ndimage import gaussian_filter1d

ROOT = Path(__file__).resolve().parent
METHODS = ['cartgs_original', 'cartgs_mode40', 'ours_full', 'dynamic_04']
LABELS = ['CaRtGS (original renderer)', 'CaRtGS (Mode 40)', 'Ours Full', 'Ours (dynamic ratio)']
COLORS = ['#888888', '#252525', '#00A6C8', '#D55E00']


def read(path):
    with path.open(encoding='utf-8', newline='') as stream:
        return list(csv.DictReader(stream))


def write(path, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, list(rows[0])); writer.writeheader(); writer.writerows(rows)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source, output, reports = ROOT/'formal', ROOT/'figures', ROOT/'reports'
    output.mkdir(exist_ok=True); reports.mkdir(exist_ok=True)
    validation = json.loads((source/'VALIDATION.json').read_text())
    job = json.loads((ROOT/'logs/formal_job.json').read_text())
    assert validation['passed'] and not validation['changed'] and validation['frames'] == 2000 and job['returncode'] == 0
    assert validation['frozen_budget_sha256'] == 'c18dfb71b1c98505ab159490b024a4cd8dbc9639bb211c1d38eddb19a3d04c8c'
    quality, timings = read(source/'QUALITY_COUNTS.csv'), read(source/'TIMINGS.csv')
    assert len(quality) == 8000 and len(timings) == 24000
    attempts = json.loads((source/'BLOCK_ATTEMPTS.json').read_text())
    accepted = [a for a in attempts if a['accepted']]
    assert sorted((a['repeat'], a['block']) for a in accepted) == [(r, b) for r in range(3) for b in range(10)]
    assert all(a['external_gpu_samples'] == a['external_cpu_samples'] == 0 for a in accepted)
    groups = {}
    for row in timings: groups.setdefault((row['method'], int(row['frame'])), []).append(row)
    data, summary, combined = {}, [], []
    for method, label in zip(METHODS, LABELS):
        records = sorted((q for q in quality if q['method'] == method), key=lambda r:int(r['frame']))
        assert [int(q['frame']) for q in records] == list(range(2000))
        values = {key:[] for key in ['K', 'M', 'ratio', 'projected_K', 'contributing_K', 'psnr', 'raster_ms', 'raster_wall_ms']}
        for i, record in enumerate(records):
            samples = groups[method, i]
            assert sorted(int(x['repeat']) for x in samples) == [0, 1, 2]
            assert all(int(x['K']) == int(record['K']) and int(x['M']) == int(record['M']) for x in samples)
            assert 0 <= int(record['contributing_K']) <= int(record['projected_K']) <= int(record['K'])
            for key in values:
                values[key].append(float(record[key]) if key in record else float(np.median([float(x[key]) for x in samples])))
        values = {key:np.asarray(x) for key,x in values.items()}
        assert all(np.isfinite(x).all() for x in values.values())
        assert (values['raster_ms'] > 0).all()
        values['equivalent_fps'] = 1000./values['raster_ms']
        values['psnr_smooth'] = gaussian_filter1d(values['psnr'], 35, mode='reflect')
        data[method] = values
        combined.extend(dict(method=method,frame=i,**{k:float(a[i]) for k,a in values.items()}) for i in range(2000))
        ms, fps = values['raster_ms'], values['equivalent_fps']
        raw = np.asarray([float(x['raster_ms']) for x in timings if x['method'] == method])
        summary.append(dict(method=method,label=label,frames=2000,K_mean=float(values['K'].mean()),
            contributing_K_mean=float(values['contributing_K'].mean()),psnr_mean=float(values['psnr'].mean()),
            raster_ms_mean=float(ms.mean()),raster_ms_p95=float(np.percentile(ms,95)),
            raster_ms_cv_percent=float(100*ms.std()/ms.mean()),
            equivalent_fps_from_mean_latency=float(1000/ms.mean()),mean_per_frame_equivalent_fps=float(fps.mean()),
            single_call_p95_ms=float(np.percentile(raw,95)),single_call_max_ms=float(raw.max())))
    assert np.array_equal(data['cartgs_original']['K'], data['cartgs_mode40']['K'])
    assert (data['cartgs_mode40']['K'] == 84203).all() and (data['ours_full']['K'] == 43197).all()
    write(reports/'PER_FRAME.csv',combined); write(reports/'SUMMARY.csv',summary)
    differences = read(source/'CARTGS_RENDERER_DIFFERENCES.csv')
    assert [int(x['frame']) for x in differences] == list(range(2000))
    assert all(x['radii_bitwise'] == x['same_packet'] == 'True' for x in differences)
    delta = np.asarray([float(x['delta_psnr']) for x in differences])
    paired = dict(mean_delta_psnr=float(delta.mean()),min_delta_psnr=float(delta.min()),max_delta_psnr=float(delta.max()),
        rgb_abs_max=max(float(x['rgb_abs_max']) for x in differences),
        rgb_abs_mean=float(np.mean([float(x['rgb_abs_mean']) for x in differences])),
        mean_speedup_original_over_Mode40=summary[0]['raster_ms_mean']/summary[1]['raster_ms_mean'],
        latency_reduction_percent=100*(1-summary[1]['raster_ms_mean']/summary[0]['raster_ms_mean']),
        mean_contribution_count_delta=float(np.mean(data['cartgs_mode40']['contributing_K']-data['cartgs_original']['contributing_K'])))
    (reports/'PAIRED_RENDERER_RESULTS.json').write_text(json.dumps(paired,indent=2)+'\n',encoding='utf-8')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,
        'axes.spines.right':False,'svg.fonttype':'path','pdf.fonttype':42})
    x = np.arange(2000)
    fig, ax = plt.subplots(figsize=(10,4.3))
    for method,label,color in [('cartgs_mode40','CaRtGS (Mode 40)',COLORS[1]),('dynamic_04','Ours (dynamic ratio)',COLORS[3])]:
        line,=ax.plot(x,data[method]['equivalent_fps'],color=color,label=label,lw=1.)
        assert np.array_equal(line.get_ydata(),data[method]['equivalent_fps'])
    max_fps = max(data[m]['equivalent_fps'].max() for m in ['cartgs_mode40','dynamic_04'])
    ax.set(xlabel='RGB frame index',ylabel='Equivalent rendering speed (FPS)',
        title='Replica office2 RGB-D | 2064 × 2208 | Both use Mode 40',xlim=(0,1999),ylim=(0,float(max(3200,500*np.ceil(max_fps/500)))))
    ax.set_xticks([0,500,1000,1500,1999]); ax.grid(alpha=.16); ax.legend(frameon=False,loc='upper center',ncol=2)
    fig.subplots_adjust(left=.095,right=.98,top=.85,bottom=.23)
    fig.text(.535,.06,'Pure rendering only; FPS = 1000 / latency (ms); no temporal smoothing.',ha='center',fontsize=9)
    for ext in ['png','svg','pdf']: fig.savefig(output/f'OFFICE2_BOTH_MODE40_FPS_UNSMOOTHED.{ext}',dpi=450,bbox_inches='tight')
    plt.close(fig)
    fig, axes = plt.subplots(1,3,figsize=(13.2,4.3))
    chosen = ['cartgs_mode40','ours_full','dynamic_04']
    labels = ['CaRtGS (Mode 40)','Ours Full','Ours (dynamic ratio)']
    colors = [COLORS[1],COLORS[2],COLORS[3]]
    for method,label,color in zip(chosen,labels,colors):
        v=data[method]
        axes[0].plot(x,v['contributing_K']/1000,color=color,lw=1.,label=label)
        axes[1].plot(x,v['psnr_smooth'],color=color,lw=1.4)
        axes[2].plot(x,v['raster_ms'],color=color,lw=.7)
    axes[0].set(title='(a) Contributing Gaussians',ylabel='Gaussians (k)',ylim=(0,None))
    axes[1].set(title='(b) Rendering quality',ylabel='PSNR (dB)')
    axes[2].set(title='(c) Rendering latency',ylabel='Pure rendering time (ms)',ylim=(0,None))
    for ax in axes: ax.set(xlim=(0,1999),xlabel='RGB frame index'); ax.set_xticks([0,500,1000,1500,1999]); ax.grid(alpha=.14)
    handles=[Line2D([],[],color=c,lw=1.5,label=l) for c,l in zip(colors,labels)]
    fig.legend(handles=handles,ncol=3,loc='upper center',bbox_to_anchor=(.52,.99),frameon=False)
    fig.subplots_adjust(left=.065,right=.99,wspace=.33,top=.78,bottom=.25)
    fig.text(.52,.11,'Replica office2 RGB-D | 2064 × 2208 | All three use Mode 40; one image per call.',ha='center',fontsize=8.5)
    fig.text(.52,.055,'PSNR smoothed (σ = 35 frames); latency unsmoothed; median of 3 repeats. Dynamic K remains offline.',ha='center',fontsize=8.5)
    for ext in ['png','svg','pdf']: fig.savefig(output/f'OFFICE2_BOTH_MODE40_THREE_METHODS.{ext}',dpi=450,bbox_inches='tight')
    plt.close(fig)
    lines=['# CaRtGS使用Mode40：office2配对渲染复测','',
        '## 核心结果','',
        'CaRtGS原版84,203点地图分别使用原版Rasterizer和Mode40渲染，相同每帧输入包、位姿和相机。'
        'Ours Full为43,197点的未补偿Master；动态ratio沿用此前冻结的逐视角预算，未重新搜索或训练。',
        'RTX3090，Replica office2 RGB-D完整2000帧，2064×2208，每次一张图，四方法同轮交替测量三遍。','',
        '| 方法 | 平均提交K | 平均Alpha贡献点数 | 平均PSNR/dB | 平均纯渲染/ms | 1000/平均耗时 FPS |',
        '|---|---:|---:|---:|---:|---:|']
    for s in summary: lines.append(f'|{s["label"]}|{s["K_mean"]:.2f}|{s["contributing_K_mean"]:.2f}|{s["psnr_mean"]:.6f}|{s["raster_ms_mean"]:.6f}|{s["equivalent_fps_from_mean_latency"]:.2f}|')
    lines+=['','## CaRtGS同输入换渲染器','',
        f'- 平均纯渲染提速{paired["mean_speedup_original_over_Mode40"]:.3f}倍，耗时降低{paired["latency_reduction_percent"]:.2f}%。',
        f'- PSNR平均变化{paired["mean_delta_psnr"]:+.8f} dB；最差帧{paired["min_delta_psnr"]:+.8f} dB。',
        f'- RGB最大绝对误差{paired["rgb_abs_max"]:.8f}，平均绝对误差{paired["rgb_abs_mean"]:.10f}。',
        '- 投影radii逐位一致。Mode40是已有快速exp/FMA近似路径，不因PSNR接近而宣称RGB逐位等价。','',
        '## 计时与图表口径','',
        '- CUDA Event围住一次实际Rasterizer调用，排除准备、Selector、Compensator、贡献计数、GT、图像保存和显示；Host同步墙钟另记录。',
        '- 三轮分别正序、倒序、固定seed打乱；每帧三次计时取中位数，再统计均值。原始单次p95/max另见SUMMARY.csv。',
        '- FPS图逐帧计算1000/该帧耗时，不做平滑；平均逐帧FPS与1000/平均耗时均保留，二者定义不同。',
        '- 三联图仅PSNR展示采用σ=35帧高斯平滑，Alpha点数和纯渲染时间不平滑。',
        '- CaRtGS和Ours分别使用自身SLAM地图及对应RGB时间戳的位姿；相同渲染器后仍存在地图、选点和补偿差异。若隔离点数影响，应在同一地图上做消融。',
        '- 两个Mode40实例各自持有有界Workspace，避免CaRtGS较大地图导致Ours容量被放大。',
        '- GT仍为原RGB经INTER_LINEAR放大，非原生2064×2208真值；RTX3090结果不代表Quest3设备。','',
        '## 验证与保护','',
        f'- 8000条质量/点数，24000条正式计时；30个验收块，{len(attempts)-len(accepted)}块因外部活动重测。验收窗口无外部CPU/GPU活动。',
        '- 原版CaRtGS的PSNR与历史逐帧相同；Ours的选点、PSNR和Alpha贡献点数与历史逐帧一致。所有Mode40计数诊断RGB与正常输出逐位一致。',
        f'- {validation["protected_files"]}个当前受保护文件前后哈希不变。较旧清单的运行前漂移/已清理文件单独记录在HISTORICAL_MANIFEST_DIFFERENCES.json，不影响本轮保护核验。',
        '- 本轮不编译、不训练、不更改生产配置或共享环境，无commit/push，不保存渲染图片。','']
    (ROOT/'CARTGS_MODE40_COMPARISON_REPORT.md').write_text('\n'.join(lines),encoding='utf-8')
    sources=[source/'QUALITY_COUNTS.csv',source/'TIMINGS.csv',source/'CARTGS_RENDERER_DIFFERENCES.csv',
        source/'VALIDATION.json',source/'BLOCK_ATTEMPTS.json',ROOT/'logs/formal_job.json',ROOT/'benchmark.py',Path(__file__)]
    manifest=dict(real_data=True,same_round=True,frames=2000,dataset='Replica/office2/rgbd',dpi=450,
        fps_smoothing=False,latency_smoothing=False,psnr_sigma=35,sources={p.relative_to(ROOT).as_posix():sha(p) for p in sources})
    (output/'DATA_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (output/'data-manifest.md').write_text('# 图表数据清单\n\n真实2000帧、四方法同轮测量；源文件SHA见DATA_MANIFEST.json。'
        '绘图输入及等效FPS见reports/PER_FRAME.csv，绘图脚本analyze.py。450 DPI，PNG/SVG/PDF。'
        'FPS、耗时和点数不平滑；三联图PSNR仅展示时σ=35帧平滑。\n',encoding='utf-8')
    result=dict(passed=True,frames=2000,quality_rows=8000,timing_rows=24000,same_renderer_comparison=True,
        fps_unsmoothed=True,source_data_verified=True,protected_files=validation['protected_files'])
    (reports/'ANALYSIS_VALIDATION.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(validation=result,summary=summary,paired=paired),ensure_ascii=False))


if __name__ == '__main__': main()
