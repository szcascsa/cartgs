"""双纵轴：既有六组纯渲染加速比与Alpha合成负载，保持原始统计口径。"""
from pathlib import Path
import csv
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter, PercentFormatter
import numpy as np

ROOT = Path(__file__).resolve().parent
SPEED = ROOT.parent/'all_datasets_joint182_ratio_speedup_20261115/figures/six_dataset_mode_curves'
ALPHA = ROOT/'figures/six_dataset_mode_curves'
RATIOS = [.01,.03,.05,.08,.1,.2,.3,.4,.5,.6,.7]
GROUPS = [('Replica','mono','Replica Mono','#0072B2','--','o',8),
          ('Replica','rgbd','Replica RGB-D','#0072B2','-','s',8),
          ('TUM_RGBD','mono','TUM Mono','#D55E00','--','o',5),
          ('TUM_RGBD','rgbd','TUM RGB-D','#D55E00','-','s',5),
          ('VECtor','mono','VECtor Mono','#009E73','--','o',3),
          ('VECtor','stereo','VECtor Stereo','#009E73','-','s',3)]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    with path.open(encoding='utf-8',newline='') as stream:
        return list(csv.DictReader(stream))


def main():
    speed_data = SPEED/'SIX_CURVES_DATA.csv'
    alpha_data = ALPHA/'ALPHA_COMPOSITING_PERCENT_DATA.csv'
    speed_check = SPEED/'FIGURE_VALIDATION.json'
    alpha_check = ALPHA/'ALPHA_COMPOSITING_PERCENT_FIGURE_VALIDATION.json'
    speed_v = json.loads(speed_check.read_text(encoding='utf-8'))
    alpha_v = json.loads(alpha_check.read_text(encoding='utf-8'))
    assert speed_v['passed'] and speed_v['curves']==6 and speed_v['repeats']==3
    assert alpha_v['passed'] and alpha_v['data_curves']==6 and alpha_v['points']==66
    assert sha(speed_data)==speed_v['data_sha256'] and sha(alpha_data)==alpha_v['data_sha256']
    sources = {str(path):sha(path) for path in [speed_data,alpha_data,speed_check,alpha_check]}
    speed = {(r['dataset'],r['mode'],float(r['ratio'])):r for r in read(speed_data) if r['scope']=='PURE_RENDER'}
    alpha = {(r['dataset'],r['mode'],float(r['ratio'])):r for r in read(alpha_data)}
    assert len(speed)==len(alpha)==66 and speed.keys()==alpha.keys()
    out = ROOT/'figures/speedup_alpha_dual_axis'
    out.mkdir(parents=True,exist_ok=True)
    family = 'Times New Roman' if any(f.name=='Times New Roman' for f in font_manager.fontManager.ttflist) else 'DejaVu Serif'
    plt.rcParams.update({'font.family':family,'font.size':9,'axes.labelsize':10,
        'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
        'axes.spines.top':False,'axes.linewidth':.7,
        'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none'})
    fig,left = plt.subplots(figsize=(7.,4.8))
    right = left.twinx()
    data,handles = [],[]
    for dataset,mode,label,color,linestyle,marker,expected in GROUPS:
        speed_values,alpha_values = [],[]
        for ratio in RATIOS:
            s,a = speed[dataset,mode,ratio],alpha[dataset,mode,ratio]
            assert int(s['scene_count'])==int(a['scene_count'])==expected
            assert s['total_views']==a['total_views'] and int(s['repeats'])==3
            sv,av = float(s['scene_equal_weight_mean_speedup']),float(a['relative_alpha_compositing_count'])
            assert np.isfinite([sv,av]).all() and sv>0 and av>=0
            assert float(a['alpha_compositing_percent'])==100*av
            speed_values.append(sv)
            alpha_values.append(av)
            data.append(dict(dataset=dataset,mode=mode,label=label,ratio=ratio,
                rendering_speedup=sv,alpha_compositing_percent=100*av,
                relative_alpha_compositing_count=av,scene_count=expected,total_views=int(a['total_views'])))
        sl, = left.plot(RATIOS,speed_values,color=color,ls=linestyle,marker=marker,
            lw=1.7,ms=3.8,markerfacecolor='white',markeredgewidth=.9,zorder=3)
        al, = right.plot(RATIOS,alpha_values,color=color,ls=':',marker=marker,
            lw=1.3,ms=3.4,markerfacecolor=color,markeredgewidth=.55,alpha=.85,zorder=2)
        assert np.array_equal(sl.get_ydata(),np.asarray(speed_values))
        assert np.array_equal(al.get_ydata(),np.asarray(alpha_values))
        handles.append(Line2D([],[],color=color,ls=linestyle,marker=marker,lw=1.6,ms=3.8,
            markerfacecolor='white',markeredgewidth=.9,label=label))
    assert len(left.lines)==len(right.lines)==6
    left.axhline(1.,color='#6B6B6B',ls=':',lw=.8,zorder=0)
    right.axhline(1.,color='#6B6B6B',ls=':',lw=.8,zorder=0)
    left.annotate('Our Complete Map (1×)',xy=(.705,1.),xytext=(-3,-7),textcoords='offset points',
        ha='right',va='top',fontsize=8,color='#606060')
    right.annotate('Our Complete Map (100%)',xy=(.705,1.),xytext=(-3,6),textcoords='offset points',
        ha='right',va='bottom',fontsize=8,color='#606060')
    left.set(xlabel='Ratio',ylabel='Rendering speedup (×)',xlim=(0,.72),ylim=(0,4.6))
    right.set(ylabel='Alpha compositing workload (%)',ylim=(0,1.12))
    left.set_xticks(np.arange(0,.71,.1))
    left.set_yticks(np.arange(0,4.51,1.))
    right.set_yticks(np.arange(0,1.01,.2))
    left.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    right.yaxis.set_major_formatter(PercentFormatter(xmax=1.,decimals=0))
    for axis in [left,right]:
        axis.tick_params(direction='out',length=3,width=.7)
    left.grid(axis='y',alpha=.20,lw=.5)
    left.spines['right'].set_visible(False)
    fig.suptitle('Rendering Speedup and Alpha Compositing Workload',fontsize=13,y=.97)
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.91),ncol=3,frameon=False,
        columnspacing=1.7,handlelength=2.4,handletextpad=.5,labelspacing=.55)
    metric_handles = [
        Line2D([],[],color='#303030',ls='-',marker='o',lw=1.7,ms=3.8,markerfacecolor='white',label='Speedup: left axis (open markers)'),
        Line2D([],[],color='#303030',ls=':',marker='o',lw=1.3,ms=3.4,markerfacecolor='#303030',label='Alpha workload: right axis (filled markers)')]
    fig.legend(handles=metric_handles,loc='lower center',bbox_to_anchor=(.5,.016),ncol=2,
        frameon=False,handlelength=2.5,columnspacing=1.7)
    fig.subplots_adjust(left=.105,right=.89,top=.77,bottom=.18)
    fig.canvas.draw()
    outputs = []
    stem = 'SIX_CURVES_SPEEDUP_ALPHA_DUAL_AXIS'
    for extension in ['png','pdf','svg']:
        path = out/f'{stem}.{extension}'
        fig.savefig(path,dpi=600,facecolor='white')
        outputs.append(path)
    preview = out/'PREVIEW.png'
    fig.savefig(preview,dpi=300,facecolor='white')
    plt.close(fig)
    csv_path = out/'SPEEDUP_ALPHA_DATA.csv'
    with csv_path.open('w',encoding='utf-8',newline='') as stream:
        writer = csv.DictWriter(stream,list(data[0]))
        writer.writeheader()
        writer.writerows(data)
    assert sources=={path:sha(Path(path)) for path in sources}
    result = dict(passed=True,groups=6,curves_per_axis=6,ratio_points=11,matched_rows=66,
        left_axis='pure rendering speedup',right_axis='alpha compositing percentage of complete map',
        alpha_aggregation='逐视角相对完整地图，再场景内平均，最后组内场景等权',
        speedup_aggregation='逐视角三轮中位耗时的完整地图/ratio，场景内平均，再组内场景等权',
        source_sha256=sources,script_sha256=sha(Path(__file__)),data_sha256=sha(csv_path),
        outputs={p.name:sha(p) for p in [*outputs,preview]},
        smoothing=False,extra_rendering=False,extra_training=False)
    (out/'FIGURE_VALIDATION.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (out/'data-manifest.md').write_text('# 渲染加速比与Alpha合成负载双轴图\n\n'
        '|图|数据|性质|来源|脚本|输出|\n|---|---|---|---|---|---|\n'
        '|六组曲线、左右纵轴|SPEEDUP_ALPHA_DATA.csv|真实测量|既有加速比SIX_CURVES_DATA.csv及Alpha百分比ALPHA_COMPOSITING_PERCENT_DATA.csv|plot_speedup_alpha.py|600dpi PNG、PDF、SVG|\n\n'
        '左轴保留纯渲染加速比，右轴为有效Alpha合成次数占完整地图的百分比。六组各有两条曲线：空心标记为左轴，填充标记与点线为右轴。'
        '三数据集颜色沿用原图；圆点为单目，方点为RGB-D/双目。32场景模式、59,507个视角，原有11档ratio，不平滑、不重测。'
        '全部66个数据对的场景数、相机模式、ratio及视角数匹配，输入输出哈希见FIGURE_VALIDATION.json。'
        '两纵轴尺度独立，曲线交点不表示两个物理量相等。\n',encoding='utf-8')
    (out/'caption_zh.md').write_text('不同ratio下的纯渲染加速比（左轴，空心标记）与Alpha合成负载百分比（右轴，填充标记与点线）。'
        '同一颜色对应同一数据集，圆点为Mono，方点为RGB-D/Stereo。加速比与负载分别相对于同视角完整地图归一化，再场景内平均、组内场景等权平均。'
        'Our Complete Map的参考值分别为1×和100%。双轴尺度独立，交点不具有相等含义。\n',encoding='utf-8')
    print(json.dumps(dict(passed=True,groups=6,curves_per_axis=6,matched_points=66,png=str(outputs[0])),ensure_ascii=False))


if __name__=='__main__':
    main()
