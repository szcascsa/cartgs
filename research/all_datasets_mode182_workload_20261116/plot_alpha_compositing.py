"""Alpha合成量–ratio论文图：沿用既有配色、线型和六组场景等权口径。"""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FormatStrFormatter, PercentFormatter
import numpy as np

ROOT = Path(__file__).resolve().parent
RATIOS = [.01, .03, .05, .08, .1, .2, .3, .4, .5, .6, .7]
GROUPS = [('Replica', 'mono', 'Replica Mono', '#0072B2', '--', 'o', 8),
          ('Replica', 'rgbd', 'Replica RGB-D', '#0072B2', '-', 's', 8),
          ('TUM_RGBD', 'mono', 'TUM Mono', '#D55E00', '--', 'o', 5),
          ('TUM_RGBD', 'rgbd', 'TUM RGB-D', '#D55E00', '-', 's', 5),
          ('VECtor', 'mono', 'VECtor Mono', '#009E73', '--', 'o', 3),
          ('VECtor', 'stereo', 'VECtor Stereo', '#009E73', '-', 's', 3)]
COLUMN = 'relative_effective_alpha_blends_mean'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = ROOT/'reports/SIX_CURVE_DATA.csv'
    validation = ROOT/'reports/VALIDATION.json'
    evidence = json.loads(validation.read_text(encoding='utf-8'))
    assert evidence['passed'] and evidence['groups']==32 and evidence['views']==59507
    assert sha(source)==evidence['group_data_sha256']
    before = {str(path):sha(path) for path in [source,validation]}
    with source.open(encoding='utf-8',newline='') as stream:
        records = list(csv.DictReader(stream))
    assert len(records)==72
    output = ROOT/'figures/six_dataset_mode_curves'
    output.mkdir(parents=True,exist_ok=True)
    family = 'Times New Roman' if any(f.name=='Times New Roman' for f in font_manager.fontManager.ttflist) else 'DejaVu Serif'
    plt.rcParams.update({'font.family':family,'font.size':9,'axes.labelsize':10,
        'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    fig,ax = plt.subplots(figsize=(5.3,3.6))
    data,peak = [],1.
    for dataset,mode,label,color,linestyle,marker,expected in GROUPS:
        complete = [r for r in records if (r['dataset'],r['mode'],r['method'])==(dataset,mode,'ours_full')]
        assert len(complete)==1 and float(complete[0][COLUMN])==1.
        values = []
        for ratio in RATIOS:
            rows = [r for r in records if r['dataset']==dataset and r['mode']==mode and float(r['ratio'])==ratio]
            assert len(rows)==1 and int(rows[0]['scenes'])==expected and rows[0]['scene_equal_weight']=='True'
            value = float(rows[0][COLUMN])
            assert np.isfinite(value) and value>=0
            values.append(value)
            data.append(dict(dataset=dataset,mode=mode,label=label,ratio=ratio,
                relative_alpha_compositing_count=value,alpha_compositing_percent=value*100,
                scene_count=expected,total_views=int(rows[0]['total_views'])))
        line, = ax.plot(RATIOS,values,color=color,ls=linestyle,marker=marker,
            lw=1.6,ms=3.6,markerfacecolor='white',markeredgewidth=.85,label=label)
        assert np.array_equal(line.get_xdata(),np.asarray(RATIOS))
        assert np.array_equal(line.get_ydata(),np.asarray(values))
        peak = max(peak,max(values))
    assert len(ax.lines)==6
    ax.axhline(1.,color='#6B6B6B',ls=':',lw=.9,zorder=0)
    annotation = ax.annotate('Our Complete Map (100%)',xy=(.705,1.),xytext=(-3,6),
        textcoords='offset points',ha='right',va='bottom',fontsize=8,color='#606060')
    ax.set(xlabel='Ratio',ylabel='Alpha compositing workload (%)',
        xlim=(0,.72),ylim=(0,max(1.12,np.ceil(peak*10)/10+.08)))
    ax.set_title('Alpha Compositing Workload',fontsize=12,pad=10)
    ax.set_xticks(np.arange(0,.71,.1))
    ax.set_yticks(np.arange(0,peak+.01,.2))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.,decimals=0))
    ax.tick_params(direction='out',length=3,width=.7)
    ax.grid(axis='y',alpha=.23,lw=.5)
    ax.legend(loc='lower right',frameon=False,handlelength=2.4,
        handletextpad=.5,labelspacing=.5,borderaxespad=.6)
    fig.subplots_adjust(left=.13,right=.98,top=.86,bottom=.17)
    fig.canvas.draw()
    assert annotation.get_window_extent().y1 < ax.get_window_extent().y1
    outputs = []
    stem = 'ALL_DATASETS_MODE182_SIX_CURVES_ALPHA_COMPOSITING_PERCENT'
    for suffix in ['png','pdf','svg']:
        path = output/f'{stem}.{suffix}'
        fig.savefig(path,dpi=600,facecolor='white')
        outputs.append(path)
    preview = output/'ALPHA_COMPOSITING_PERCENT_PREVIEW.png'
    fig.savefig(preview,dpi=300,facecolor='white')
    plt.close(fig)
    csv_path = output/'ALPHA_COMPOSITING_PERCENT_DATA.csv'
    with csv_path.open('w',encoding='utf-8',newline='') as stream:
        writer = csv.DictWriter(stream,list(data[0]))
        writer.writeheader()
        writer.writerows(data)
    assert before=={str(path):sha(path) for path in [source,validation]}
    manifest = dict(passed=True,data_curves=6,points_per_curve=11,points=66,
        input_sha256=before,column=COLUMN,baseline_label='Our Complete Map (100%)',y_unit='percent',
        aggregation='每帧Alpha合成次数/同视角完整地图次数，先场景内均值，再同数据集/模式内场景等权均值',
        smoothing=False,additional_rendering=False,training=False,
        data_sha256=sha(csv_path),script_sha256=sha(Path(__file__)),
        outputs={path.name:sha(path) for path in [*outputs,preview]})
    (output/'ALPHA_COMPOSITING_PERCENT_FIGURE_VALIDATION.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (output/'alpha_percent_data-manifest.md').write_text('# Alpha合成量与ratio六条曲线（百分比）\n\n'
        '|图|数据|类型|来源|脚本|输出|\n|---|---|---|---|---|---|\n'
        '|Alpha Compositing Workload|ALPHA_COMPOSITING_PERCENT_DATA.csv|真实计数|reports/SIX_CURVE_DATA.csv|plot_alpha_compositing.py|600dpi PNG、PDF、SVG|\n\n'
        '32组、59,507个有效全轨迹视角；Replica每模式8场景、TUM每模式5场景、VECtor每模式3场景。'
        '每帧有效Alpha合成次数除以同视角未选择、未补偿完整地图次数，再场景内平均，最后对场景等权平均。'
        '纵轴50%表示合成次数为完整地图的一半，不表示加速2倍。CSV同时保留原始相对值和百分比列。六条线不做平滑；虚线圆点为Mono，实线方点为RGB-D/Stereo。'
        '100%参考线名称为Our Complete Map (100%)。沿用mode182原始分辨率与名义Anchor局部预算；VECtor Stereo只计单张左图。\n',encoding='utf-8')
    (output/'caption_alpha_percent_zh.md').write_text('不同局部预算ratio下，相对于Our Complete Map的有效Alpha合成次数百分比。'
        '统计有效像素上非零T×alpha合成的Gaussian–像素对数，逐视角归一化后场景内平均，再在同数据集同模式内对场景等权平均。'
        '六条曲线分别对应Replica Mono/RGB-D、TUM Mono/RGB-D和VECtor Mono/Stereo。'
        '该指标描述有效合成次数，不包含被拒绝候选和其他渲染开销。\n',encoding='utf-8')
    print(json.dumps(dict(passed=True,curves=6,points=66,png=str(outputs[0])),ensure_ascii=False))


if __name__=='__main__':
    main()
