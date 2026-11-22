"""原等距质量雷达图叠加平均保留Gaussian数K；同色虚线使用独立点数刻度。"""
from pathlib import Path
import csv
import hashlib
import json
import math
import shutil
import zipfile

import plot_replica_radar as base
import numpy as np
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
ROOT = HERE/'with_counts'
SOURCE = HERE/'data/REPLICA_RADAR_MEANS.csv'
SOURCE_SHA = '54d2ee964f4886ad7c17bb167f92908933768eeb2353a39254ee73716742e932'
METRICS = ['psnr','ssim','lpips']


def load_rows():
    assert base.digest(SOURCE)==SOURCE_SHA
    rows=list(csv.DictReader(SOURCE.open(encoding='utf-8-sig')))
    expected={(mode,method,ratio) for mode in ['all','mono','rgbd']
              for method in base.METHODS for ratio in base.RATIOS}
    for row in rows:
        for key in ['ratio','N_mean','K_mean',*METRICS]:row[key]=float(row[key])
        assert all(math.isfinite(row[key]) for key in ['K_mean',*METRICS])
        assert 0<=row['K_mean']<=row['N_mean']
    assert len(rows)==99
    assert {(r['mode'],r['method'],r['ratio']) for r in rows}==expected
    for mode in ['all','mono','rgbd']:
        for ratio in base.RATIOS:
            points=[r for r in rows if r['mode']==mode and r['ratio']==ratio]
            assert points[0]['K_mean']==points[1]['K_mean']
    return rows


def draw_radar(ax,rows,metric,mode,k_limit,small=False):
    base.radar(ax,rows,metric,mode,small=small)
    angles=np.pi/2-np.arange(len(base.RATIOS))*2*np.pi/len(base.RATIOS)
    unit=np.column_stack((np.cos(angles),np.sin(angles)))
    styles={'cartgs_random':(0,(4,12)), 'cartgs_opacity':(8,(4,12)),
            'ours_global':(0,(4,4))}
    for method in base.METHODS:
        points=[next(r for r in rows if r['mode']==mode and r['method']==method and r['ratio']==ratio)
                for ratio in base.RATIOS]
        rr=np.array([p['K_mean'] for p in points])/k_limit
        assert np.all((rr>=0)&(rr<=1))
        xy=unit*rr[:,None]
        line,=ax.plot(np.r_[xy[:,0],xy[0,0]],np.r_[xy[:,1],xy[0,1]],
                     color=base.COLORS[method],lw=1.5,ls=styles[method],zorder=7,
                     dash_capstyle='butt')
        line.set_gid('retained_K_'+method)
    # 点数刻度与质量刻度共用网格半径；右侧单位为千点，左侧仍为原始质量指标。
    for tick in base.METRICS[metric][3]:
        r=float(base.radius(metric,[tick])[0])
        if r==0:continue
        ax.text(r,.035,f'{r*k_limit/1000:g}k',fontsize=7.5 if small else 8.8,
                color='#52636F',va='bottom',ha='center',zorder=8,
                bbox=dict(fc='white',ec='none',pad=.5,alpha=.88))
    return ax


def count_table(fig,rows,mode,bounds,fontsize=7.8):
    ax=fig.add_axes(bounds);ax.axis('off')
    cell_text=[]
    labels=['随机删除','Opacity删除','我们']
    for method,label in zip(base.METHODS,labels):
        points=[next(r for r in rows if r['mode']==mode and r['method']==method and r['ratio']==ratio)
                for ratio in base.RATIOS]
        cell_text.append([label]+[f"{p['K_mean']/1000:.2f}" for p in points])
    table=ax.table(cellText=cell_text,
                   colLabels=['Ratio']+[f'{100*r:g}%' for r in base.RATIOS],
                   colWidths=[.15]+[.85/11]*11,cellLoc='center',bbox=[0,0,1,1])
    table.auto_set_font_size(False);table.set_fontsize(fontsize)
    for (row,column),cell in table.get_celld().items():
        cell.set_edgecolor('#E1E6EA');cell.set_linewidth(.5)
        cell.set_facecolor('#F1F4F6' if row==0 else ('white' if row%2 else '#FAFBFC'))
        if row==0:cell.get_text().set_color('#33434F')
        elif column==0:
            cell.get_text().set_color(base.COLORS[base.METHODS[row-1]])
            cell.get_text().set_fontweight('bold')
    return table


def legends(fig,method_y,type_y,fontsize=10):
    fig.legend(handles=base.legend_handles(),loc='lower center',bbox_to_anchor=(.5,method_y),
               ncol=3,frameon=False,fontsize=fontsize,columnspacing=1.25,handlelength=1.7)
    handles=[Line2D([],[],color='#44515C',lw=1.7,label='实线：渲染质量（左刻度）'),
             Line2D([],[],color='#44515C',lw=1.5,ls=(0,(4,4)),label='虚线：保留数K（右刻度）')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,type_y),
               ncol=2,frameon=False,fontsize=fontsize-.5,columnspacing=2,handlelength=2.4)


def main():
    assert not base.EXPANDED_SCALE, '本次沿用原等距质量刻度'
    base.configure_font();rows=load_rows()
    k_limit=math.ceil(max(r['K_mean'] for r in rows)/10000)*10000
    figures=ROOT/'figures';data=ROOT/'data'
    figures.mkdir(parents=True,exist_ok=True);data.mkdir(exist_ok=True)
    for name in ['REPLICA_RADAR_MEANS.csv','REPLICA_PER_SCENE.csv']:
        shutil.copyfile(HERE/'data'/name,data/name)
    manifest=dict(source=str(SOURCE),source_sha256=SOURCE_SHA,real_data=True,
        methods=base.METHODS,modes=['all','mono','rgbd'],ratios=base.RATIOS,
        average='每场景先对单目与RGB-D等权，再对8场景等权；分模式图分别对8场景等权',
        solid_lines='原等距质量刻度：PSNR 0–40；SSIM 0–1；LPIPS中心1、外圈0',
        dashed_lines='平均保留Gaussian数K，径向半径=K/K上限，独立右刻度，单位千点',
        K_axis_max=k_limit,count_definition='全局导出模型保留数／Renderer输入K，不是逐视角可见或参与Alpha合成数',
        shared_cartgs_counts='两个CaRtGS删除方法K完全相同；共用几何路径，用蓝金交错虚线显示，不偏移数据',
        table='平均K以千点为单位保留两位小数，CSV保留完整精度',
        algorithm_data_changed=False)
    (data/'SOURCE_MANIFEST.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (figures/'data-manifest.md').write_text('# 数据清单\n\n'
        '|图|数据|性质|来源|输出|\n|---|---|---|---|---|\n'
        '|三指标主图、分模式图和总览|data/REPLICA_RADAR_MEANS.csv|真实|已有Replica全轨迹指标和实际保留K|PNG/SVG/PDF|\n\n'
        '实线读取质量指标，虚线读取同表K_mean；不重新渲染、不改质量刻度、不重新计算K。\n',encoding='utf-8')
    note=f'右刻度为K：中心0、外圈{k_limit/1000:g}千点；CaRtGS两虚线重合，以交错线段显示'
    for metric in METRICS:
        fig=base.plt.figure(figsize=(7.2,9.4))
        ax=fig.add_axes([.035,.325,.93,.565]);draw_radar(ax,rows,metric,'all',k_limit)
        fig.suptitle('Replica · '+base.METRICS[metric][0],fontsize=20,fontweight='bold',y=.983,color='#22313B')
        fig.text(.5,.935,'8场景等权平均 · 每场景单目与RGB-D等权',ha='center',fontsize=11,color='#52636F')
        legends(fig,.273,.244,fontsize=9.8)
        fig.text(.5,.220,'平均保留Gaussian数K（千点）',ha='center',fontsize=10,color='#33434F')
        count_table(fig,rows,'all',[.045,.080,.91,.123])
        fig.text(.5,.048,note,ha='center',fontsize=8.8,color='#52636F')
        fig.text(.5,.023,'实线质量越外越好；各方法按自身N保留，点数越多不代表质量越高',ha='center',fontsize=8.8,color='#52636F')
        base.save(fig,figures/f'REPLICA_{metric.upper()}_RADAR')
        fig=base.plt.figure(figsize=(13.6,9.4))
        for index,mode in enumerate(['mono','rgbd']):
            ax=fig.add_axes([.025+index*.5,.325,.45,.555]);draw_radar(ax,rows,metric,mode,k_limit,small=True)
            count_table(fig,rows,mode,[.025+index*.5,.080,.45,.123],fontsize=7.6)
            fig.text(.25+index*.5,.220,'平均保留Gaussian数K（千点）',ha='center',fontsize=10,color='#33434F')
        fig.suptitle('Replica · '+base.METRICS[metric][0]+' · 分模式核对',fontsize=20,fontweight='bold',y=.985,color='#22313B')
        legends(fig,.273,.244,fontsize=10)
        fig.text(.5,.035,'office0–4、room0–2等权；'+note,ha='center',fontsize=10,color='#52636F')
        base.save(fig,figures/f'REPLICA_{metric.upper()}_BY_MODE')
    fig=base.plt.figure(figsize=(17.4,8.4))
    for index,metric in enumerate(METRICS):
        ax=fig.add_axes([.005+index/3,.345,.323,.46]);draw_radar(ax,rows,metric,'all',k_limit,small=True)
        ax.set_title(base.METRICS[metric][0],fontsize=18,fontweight='bold',pad=12)
        count_table(fig,rows,'all',[.012+index/3,.095,.309,.11],fontsize=6.9)
        fig.text(1/6+index/3,.222,'平均保留Gaussian数K（千点）',ha='center',fontsize=9,color='#33434F')
    fig.suptitle('Replica · 8场景均值 · 质量与保留点数',fontsize=20,fontweight='bold',y=.985,color='#22313B')
    fig.text(.5,.930,'每场景先平均单目与RGB-D，再对8个场景等权平均；蓝：随机删除，金：Opacity删除，红：我们全局LoD',ha='center',fontsize=11,color='#52636F')
    legends(fig,.280,.249,fontsize=11)
    fig.text(.5,.043,'质量沿用原等距刻度；'+note+'；虚线向外仅表示保留点更多。',ha='center',fontsize=10,color='#52636F')
    base.save(fig,figures/'REPLICA_RADAR_OVERVIEW')
    (ROOT/'README_结果说明.md').write_text('# Replica质量与保留Gaussian数\n\n'
        '实线沿用原版质量雷达图与原始等距刻度；同色虚线表示该方法每个Ratio的平均保留数K。'
        f'质量看左刻度，点数看右刻度；三种指标及三种模式共用K范围0–{k_limit:g}，右刻度以千点显示。\n\n'
        '平均K和质量使用同样的8场景／16组等权方式，均读取已有原始汇总，不重新训练或渲染。'
        'K为全局保留模型的Gaussian数／Renderer输入数，不能理解为当前视角实际可见点数。\n\n'
        'CaRtGS随机删除与Opacity删除保留K完全相同，虚线重合；蓝金使用交错线段显示，未人为移动曲线。'
        '图下表列出每种方法逐Ratio的平均K，单位千点，显示两位小数，完整精度见CSV。\n\n'
        '实线质量越外越好；虚线越外仅表示点数更多。相同Ratio时各方法的N和K不同，本图不是相同K试验。\n',encoding='utf-8')
    assert all(len(list(figures.glob('*.'+ext)))==7 for ext in ['png','svg','pdf'])
    receipt=dict(complete=True,source_unchanged=base.digest(SOURCE)==SOURCE_SHA,
        mean_rows=99,original_metric_scales=True,K_axis_max=k_limit,
        figures={str(p.relative_to(ROOT)).replace('\\','/'):base.digest(p) for p in figures.iterdir() if p.is_file()})
    (ROOT/'VERIFIED.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with zipfile.ZipFile(ROOT/'REPLICA_RATIO_RADAR_WITH_COUNTS.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for directory in ['data','figures']:
            for path in (ROOT/directory).iterdir():
                if path.is_file():archive.write(path,path.relative_to(ROOT))
        for name in ['README_结果说明.md','VERIFIED.json']:archive.write(ROOT/name,name)
        for name in ['plot_replica_radar.py','plot_replica_radar_with_counts.py']:archive.write(HERE/name,name)
    print('REPLICA_COUNTS_COMPLETE',len(rows),'行；7张图；K刻度上限',k_limit)


if __name__=='__main__':main()
