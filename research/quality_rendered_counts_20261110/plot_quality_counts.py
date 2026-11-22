"""质量、平均实际渲染点数、提交点数及完整地图规模：32组逐场景图和总览。"""
from pathlib import Path
import argparse
import csv
import json
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
import numpy as np

ROOT = Path(__file__).resolve().parent
FIGURES = ROOT/'figures'
RATIOS = [.01, .03, .05, .08, .1, .2, .3, .4, .5, .6, .7]
TRAIN = {.01, .05, .1, .3, .5, .7}
COLORS = dict(global_lod='#0077BB', local_lod='#EE7733', ours_full='#009988', vanilla='#555555')
LABELS = dict(global_lod='我们：全局 LoD', local_lod='我们：局部 LoD', ours_full='我们 Full', vanilla='CaRtGS 原版')


def compact(value, digits=1):
    if value >= 10000:
        return f'{value/1000:.1f}k'
    return f'{value:,.{digits}f}' if value != round(value) else f'{value:,.0f}'


def style(ax, title, ratio_axis=True):
    ax.set_title(title, loc='left', fontweight='bold', fontsize=11.2, pad=9)
    ax.set_facecolor('#FAFBFC')
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color('#BCC5CC')
    ax.tick_params(labelsize=8.2)
    ax.grid(axis='y', alpha=.24, linewidth=.6)
    if ratio_axis:
        ax.set_xlim(0., .72)
        ax.set_xticks(RATIOS, [f'{r:g}' for r in RATIOS], rotation=55)
        ax.set_xlabel('Ratio（实际数值线性轴）', fontsize=8.6, labelpad=4)


def plot_curves(ax, series, key):
    for rows, color in series:
        x, y = np.array([r['ratio'] for r in rows]), np.array([r[key] for r in rows])
        ax.plot(x, y, lw=1.8, color=color)
        for trained in [True, False]:
            mask = np.array([r in TRAIN for r in x]) == trained
            ax.scatter(x[mask], y[mask], s=22, edgecolors=color, facecolors=color if trained else 'white',
                       linewidths=1.2, zorder=4)


def make_table(ax, rows, columns, widths):
    ax.axis('off')
    table = ax.table(cellText=rows, colLabels=columns, cellLoc='center', colLoc='center',
                     colWidths=widths, bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(8.2)
    for (row, column), cell in table.get_celld().items():
        cell.set_edgecolor('#D9E0E5')
        cell.set_linewidth(.5)
        if row == 0:
            cell.set_facecolor('#E9EFF4')
            cell.set_text_props(weight='bold', color='#293E4C')
        elif row in [1,2]:
            cell.set_facecolor('#F0F4F3' if row==2 else '#F0F0F0')
        else:
            cell.set_facecolor('#FAFBFC' if row%2==0 else 'white')


def group_records(group, lookup):
    vanilla = lookup[(group['id'],'vanilla',1.)]
    full = lookup[(group['id'],'full',1.)]
    global_rows = [lookup[(group['id'],'global',r)] for r in RATIOS]
    local_rows = [lookup[(group['id'],'view_ratio',r)] for r in RATIOS]
    return vanilla, full, global_rows, local_rows


def plot_group(group, lookup):
    vanilla, full, global_rows, local_rows = group_records(group, lookup)
    series = [(global_rows, COLORS['global_lod']), (local_rows, COLORS['local_lod'])]
    fig = plt.figure(figsize=(14.8,16.9), facecolor='white')
    grid = fig.add_gridspec(4,2,left=.068,right=.975,top=.86,bottom=.065,
                           height_ratios=[1,1,1,1.16],hspace=.46,wspace=.21)
    mode = {'mono':'单目','rgbd':'RGB-D','stereo':'双目'}[group['mode']]
    fig.suptitle(f"{group['dataset']} · {group['scene']} · {mode}",y=.985,fontsize=19,
                 fontweight='bold',color='#203040')
    fig.text(.5,.959,'渲染质量 × 平均实际渲染 Gaussian 数量 × 完整地图规模',ha='center',fontsize=13,color='#334B5B')
    fig.text(.5,.938,f"CaRtGS 未校正原图口径 · 原版 {vanilla['frames']:,}帧 / 我们 {full['frames']:,}帧 · 各自完整轨迹，不重采样",
             ha='center',fontsize=9,color='#596670')
    handles = [Line2D([],[],color=COLORS['global_lod'],marker='o',lw=1.9,label=LABELS['global_lod']),
               Line2D([],[],color=COLORS['local_lod'],marker='o',lw=1.9,label=LABELS['local_lod']),
               Line2D([],[],color=COLORS['ours_full'],ls='-.',lw=1.7,
                      label=f"我们 Full：地图 N={full['model_total_N']:,}；平均贡献 {full['contributing_K_mean']:,.1f}"),
               Line2D([],[],color=COLORS['vanilla'],ls='--',lw=1.7,
                      label=f"CaRtGS 原版：地图 N={vanilla['model_total_N']:,}；平均贡献 {vanilla['contributing_K_mean']:,.1f}")]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.923),ncol=2,
               frameon=False,fontsize=9.2,handlelength=3,columnspacing=1.8)
    panels = [('psnr','PSNR ↑'),('ssim','SSIM ↑'),('lpips','LPIPS ↓'),
              ('contributing_K_mean','每视角实际参与 Alpha 合成的点数均值'),
              ('submitted_K_mean','完整地图 N 与导出／提交点数 K')]
    for index,(key,title) in enumerate(panels):
        ax = fig.add_subplot(grid[index//2,index%2])
        style(ax,title)
        plot_curves(ax,series,key)
        if key in ['psnr','ssim','lpips']:
            ax.axhline(vanilla[key],color=COLORS['vanilla'],ls='--',lw=1.5)
            ax.axhline(full[key],color=COLORS['ours_full'],ls='-.',lw=1.5)
            digits=2 if key=='psnr' else 3
            ax.text(.985,.04,f"原版 {vanilla[key]:.{digits}f} | 我们 Full {full[key]:.{digits}f}",
                    ha='right',transform=ax.transAxes,fontsize=8.1,color='#44515B',
                    bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=2))
            if key=='psnr':ax.set_ylabel('dB',fontsize=8.5)
        else:
            for row,color,linestyle in [(vanilla,COLORS['vanilla'],'--'),(full,COLORS['ours_full'],'-.')]:
                value=row['model_total_N'] if key=='submitted_K_mean' else row[key]
                ax.axhline(value,color=color,ls=linestyle,lw=1.5)
            ax.set_yscale('log')
            ax.set_ylabel('点（对数轴）',fontsize=8.5)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda value,pos:f'{value/1000:g}k' if value>=1000 else f'{value:g}'))
            note='每个点至少对一个像素产生有效 Alpha 贡献；按所有评价帧取均值' if key=='contributing_K_mean' else '全局 K＝完整导出包点数；局部 K 为逐视角提交均值，仍需保留完整地图 N'
            ax.text(.025,.05,note,transform=ax.transAxes,fontsize=7.7,color='#596670',
                    bbox=dict(facecolor='white',edgecolor='none',alpha=.9,pad=2))
    ax=fig.add_subplot(grid[2,1])
    style(ax,'PSNR 与平均实际渲染点数',ratio_axis=False)
    for rows,color in series:
        ax.plot([r['contributing_K_mean'] for r in rows],[r['psnr'] for r in rows],color=color,lw=1.7,
                marker='o',markersize=4)
        for row in [rows[0],rows[4],rows[-1]]:
            ax.annotate(f"r={row['ratio']:g}",(row['contributing_K_mean'],row['psnr']),
                        xytext=(3,5 if color==COLORS['global_lod'] else -11),textcoords='offset points',
                        fontsize=7,color=color)
    ax.scatter([vanilla['contributing_K_mean']],[vanilla['psnr']],marker='*',s=115,color=COLORS['vanilla'],zorder=5)
    ax.scatter([full['contributing_K_mean']],[full['psnr']],marker='*',s=115,color=COLORS['ours_full'],zorder=5)
    ax.set_xscale('log')
    ax.xaxis.set_major_formatter(FuncFormatter(lambda value,pos:f'{value/1000:g}k' if value>=1000 else f'{value:g}'))
    ax.set_xlabel('平均实际贡献点数（对数轴）',fontsize=8.6)
    ax.set_ylabel('PSNR（dB）',fontsize=8.6)
    ax.margins(x=.16,y=.16)
    ax.text(.025,.04,'星号：原版 / 我们 Full；该图不等同于同点数配对实验',transform=ax.transAxes,fontsize=7.8,color='#596670')

    quality, counts=[],[]
    for label,row in [('原版 Full',vanilla),('我们 Full',full)]:
        quality.append([label,f"{row['psnr']:.3f}",'—',f"{row['ssim']:.4f}",'—',f"{row['lpips']:.4f}",'—'])
        counts.append([label,f"{row['submitted_K_mean']:,.0f}",'—',f"{row['contributing_K_mean']:,.1f}",'—'])
    for global_row,local_row in zip(global_rows,local_rows):
        r=global_row['ratio'];label=f"{r:g}{' *' if r not in TRAIN else ''}"
        quality.append([label,f"{global_row['psnr']:.3f}",f"{local_row['psnr']:.3f}",
                        f"{global_row['ssim']:.4f}",f"{local_row['ssim']:.4f}",
                        f"{global_row['lpips']:.4f}",f"{local_row['lpips']:.4f}"])
        counts.append([label,f"{global_row['submitted_K_mean']:,.0f}",f"{local_row['submitted_K_mean']:,.1f}",
                       f"{global_row['contributing_K_mean']:,.1f}",f"{local_row['contributing_K_mean']:,.1f}"])
    make_table(fig.add_subplot(grid[3,0]),quality,['Ratio / Full','全局\nPSNR','局部\nPSNR','全局\nSSIM','局部\nSSIM','全局\nLPIPS','局部\nLPIPS'],[.19,.135,.135,.135,.135,.135,.135])
    make_table(fig.add_subplot(grid[3,1]),counts,['Ratio / Full','全局 K\n/ Full N','局部平均 K','全局 / Full\n平均贡献','局部平均\n实际贡献'],[.18,.205,.205,.205,.205])
    fig.text(.068,.037,'* / 空心点：未见 Ratio。N＝完整源地图；K＝导出／提交点数；实际贡献点数≤投影有效点数≤K。',fontsize=8.6,color='#56636D')
    fig.text(.068,.023,'计数使用原 checkpoint、原轨迹和原选点；质量直接复用对应逐帧结果。各自全轨迹，未作共同帧配对，不保存渲染图片。',fontsize=8.3,color='#56636D')
    target=FIGURES/group['dataset'];target.mkdir(parents=True,exist_ok=True)
    stem=group['id'].replace('/','__')
    fig.savefig(target/(stem+'.png'),dpi=450)
    fig.savefig(target/(stem+'.svg'))
    plt.close(fig)
    rows=[dict(dataset=group['dataset'],scene=group['scene'],mode=group['mode'],**row)
          for row in [vanilla,full,*global_rows,*local_rows]]
    return dict(id=group['id'],png=(target/(stem+'.png')).relative_to(ROOT).as_posix(),
                svg=(target/(stem+'.svg')).relative_to(ROOT).as_posix()), rows


def overview(groups,lookup):
    fig,axes=plt.subplots(8,4,figsize=(18,23.4))
    fig.subplots_adjust(left=.055,right=.985,top=.923,bottom=.04,hspace=.57,wspace=.34)
    for group,ax in zip(groups,axes.flat):
        vanilla,full,global_rows,local_rows=group_records(group,lookup)
        for rows,color in [(global_rows,COLORS['global_lod']),(local_rows,COLORS['local_lod'])]:
            ax.plot([r['contributing_K_mean'] for r in rows],[r['psnr'] for r in rows],lw=1.4,color=color,marker='o',markersize=2.7)
        for row,color in [(vanilla,COLORS['vanilla']),(full,COLORS['ours_full'])]:
            ax.scatter(row['contributing_K_mean'],row['psnr'],marker='*',s=65,color=color,zorder=4)
        ax.set_xscale('log')
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value,pos:f'{value/1000:g}k' if value>=1000 else f'{value:g}'))
        ax.tick_params(labelsize=7.5)
        ax.set_title(group['id'].replace('TUM_RGBD/','TUM/'),fontsize=9.5,fontweight='bold',pad=5)
        ax.text(.02,.98,f"完整 N：原版 {compact(vanilla['model_total_N'])} / 我们 {compact(full['model_total_N'])}",
                transform=ax.transAxes,va='top',fontsize=7.4,color='#56636D')
        ax.margins(y=.25,x=.15)
        ax.grid(alpha=.2)
        ax.spines[['top','right']].set_visible(False)
        ax.set_xlabel('平均实际 Alpha 贡献点数（对数轴）',fontsize=7.5)
        ax.set_ylabel('PSNR（dB）',fontsize=7.5)
    fig.suptitle('32组结果总览：渲染质量与平均实际渲染 Gaussian 数量',y=.985,fontsize=20,fontweight='bold',color='#203040')
    fig.text(.5,.965,'每个场景／相机模式独立一格；每条曲线包含11档 Ratio；完整地图规模 N 标于各格上方。',ha='center',fontsize=11,color='#56636D')
    handles=[Line2D([],[],lw=2,color=COLORS['global_lod'],marker='o',label='全局 LoD'),
             Line2D([],[],lw=2,color=COLORS['local_lod'],marker='o',label='局部 LoD'),
             Line2D([],[],lw=0,color=COLORS['ours_full'],marker='*',markersize=12,label='我们 Full'),
             Line2D([],[],lw=0,color=COLORS['vanilla'],marker='*',markersize=12,label='CaRtGS 原版')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,.952),ncol=4,frameon=False,fontsize=11)
    fig.text(.5,.012,'实际渲染点数定义为每视角至少对一个像素产生有效 Alpha 合成贡献的独立 Gaussian 数，跨完整评价轨迹取均值；不是提交 K。',ha='center',fontsize=10,color='#56636D')
    for suffix in ['png','svg']:fig.savefig(FIGURES/('ALL_SCENES_OVERVIEW.'+suffix),dpi=450)
    plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--group')
    parser.add_argument('--input',default=str(ROOT/'data/QUALITY_RENDERED_COUNTS.json'));args=parser.parse_args()
    source=Path(args.input);data=json.loads(source.read_text(encoding='utf-8'))
    font_manager.fontManager.addfont('C:/Windows/Fonts/msyh.ttc')
    plt.rcParams.update({'font.family':'Microsoft YaHei','axes.unicode_minus':False,'svg.fonttype':'none'})
    FIGURES.mkdir(exist_ok=True)
    for row in data['counts']:
        row['count_backend']=row.get('count_backend','legacy_GI')
        row['source_map_N']=row['model_total_N']
        row['render_model_N']=row['exported_gaussians'] if row['scope']=='global' else row['source_map_N']
    lookup={(r['group'],r['scope'],r['ratio']):r for r in data['counts']}
    groups=[g for g in data['groups'] if not args.group or g['id']==args.group]
    index,records=[],[]
    for group in groups:
        item,rows=plot_group(group,lookup);index.append(item);records+=rows
        print('PLOTTED',group['id'],flush=True)
    if args.group:return
    assert len(index)==32 and len(records)==768
    overview(groups,lookup)
    fields=['dataset','scene','mode','group','scope','ratio','ratio_kind','frames',
            'psnr','ssim','lpips','source_map_N','render_model_N','exported_gaussians',
            'submitted_K_mean','submitted_K_min','submitted_K_max','projected_K_mean','projected_K_min','projected_K_max',
            'contributing_K_mean','contributing_K_min','contributing_K_max','count_backend','checkpoint','quality_source','count_source']
    with (ROOT/'QUALITY_RENDERED_COUNTS.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fields,extrasaction='ignore');writer.writeheader();writer.writerows(records)
    for group in groups:
        target=FIGURES/group['dataset']/(group['id'].replace('/','__')+'.csv')
        with target.open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fields,extrasaction='ignore');writer.writeheader();writer.writerows([r for r in records if r['group']==group['id']])
    lines=['# 渲染质量、平均实际渲染点数与完整地图规模','',
           '已完成32组逐场景图和一张总览，覆盖Replica 16组、TUM 10组、VECtor 6组。', '',
           '## 点数口径','',
           '- 完整源地图N：读取checkpoint的完整Gaussian数；原版和我们可能不同。',
           '- 全局导出K：全图选择并补偿后固定导出包的完整点数；同一包在所有视角保持不变。',
           '- 局部提交K：当前名义Anchor视锥内按ratio预算选择并补偿后，实际送入Renderer的点数；随视角变化，表中给出平均K，完整源地图N仍需保留。',
           '- CSV中source_map_N表示完整源地图，render_model_N在全局模式为固定导出包K、在局部模式为需要保留的完整地图N；exported_gaussians仅对固定包填写。',
           '- 平均实际渲染点数：每帧至少对一个像素产生有效Alpha合成贡献的独立Gaussian数量，再按该次评价全部帧求平均。复用生产Rasterizer已有GI整数贡献计数，不用Opacity阈值近似，不等同于radii>0或K。',
           '- 全局／局部LoD均在选点、补偿和原有属性变换之后的最终渲染输入上计数；Full使用未补偿Master，未用名义Anchor可见性代替真实像素贡献。',
           '- 另记录radii>0的投影有效点数，所有帧满足贡献数≤投影有效数≤提交K≤源地图N。', '',
           '## 质量及验证','',
           'PSNR／SSIM／LPIPS直接复用此前CaRtGS未校正原图口径的完整逐帧评价，768条汇总质量数值不变。仅为缺失的实际点数补做冻结前向计数，不重训、不保存渲染图片。', '',
           '全部局部选点身份哈希与原评价逐帧一致，全局导出身份一致；每个工作项在首／中／末帧验证计数版与普通Rasterizer图像和radii逐bit一致，并用原图复核PSNR。模型与历史评价文件哈希前后保持一致。', '',
           '统计器先使用原有GI整数贡献次数，后续使用独立编译的单调原子0/1标记，只记录是否贡献，免去不需要的浮点GI累加。三个数据集的72组验收输入逐Gaussian贡献集合、图像和radii完全一致；后续每个计数项仍在首／中／末帧与原接口对照。独立统计器不替代生产GI或训练模块。', '',
           '原版继续按每场景、每相机模式Full轨迹PSNR最高的一次重复取值，SSIM、LPIPS、N和实际贡献点数绑定同一次checkpoint。未重新挑选最优模型。双方使用各自完整轨迹，不作为共同帧的同点数配对实验。', '',
           '所有ratio横轴使用实际线性距离，质量不做平滑或插值；空心点及*为未见ratio。计数纵轴及质量—点数图横轴使用对数，已明确标注。', '',
           '总览：[PNG](figures/ALL_SCENES_OVERVIEW.png) / [SVG](figures/ALL_SCENES_OVERVIEW.svg)。总表：[QUALITY_RENDERED_COUNTS.csv](QUALITY_RENDERED_COUNTS.csv)。','',
           '| 场景／模式 | 图 | 矢量图 | 数据 |','|---|---|---|---|']
    for item in index:
        lines.append(f"| {item['id']} | [PNG]({item['png']}) | [SVG]({item['svg']}) | [CSV]({Path(item['png']).with_suffix('.csv').as_posix()}) |")
    (ROOT/'README_结果说明.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (FIGURES/'data-manifest.md').write_text('# 数据清单\n\n真实实验数据。\n\n'+
        '\n'.join(f"- {item['id']}：QUALITY_RENDERED_COUNTS.csv 同名 group；源 data/QUALITY_RENDERED_COUNTS.json；绘图 plot_quality_counts.py；输出 {item['png']} 和 {item['svg']}。" for item in index)+'\n',encoding='utf-8')
    (ROOT/'FIGURES_INDEX.json').write_text(json.dumps(index,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with zipfile.ZipFile(ROOT/'QUALITY_RENDERED_COUNTS_RESULTS.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for p in sorted(FIGURES.rglob('*')):
            if p.is_file():archive.write(p,p.relative_to(ROOT))
        for name in ['QUALITY_RENDERED_COUNTS.csv','README_结果说明.md','FIGURES_INDEX.json',
                     'data/QUALITY_RENDERED_COUNTS.json','COMPLETE.json','count_worker.py','run_counts.py','plot_quality_counts.py',
                     'BOOLEAN_COUNTER_BUILD.json','BOOLEAN_COUNTER_VALIDATED.json','build_boolean_counter.py','validate_boolean_counter.py',
                     'verify_drawings.py']:
            archive.write(ROOT/name,name)
    print('DELIVERY_COMPLETE',len(index),'scene figures + overview',flush=True)


if __name__=='__main__':main()
