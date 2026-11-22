"""Replica八场景Ratio雷达图：真实宏平均、固定刻度、独立指标。"""
from collections import defaultdict
from pathlib import Path
import csv
import hashlib
import json
import math
import sys
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors, font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Polygon, Wedge
import numpy as np

ROOT = Path(__file__).resolve().parent
EXPANDED_SCALE = '--expanded-scale' in sys.argv
OUTPUT = ROOT/'expanded_scale' if EXPANDED_SCALE else ROOT
SOURCE = ROOT.parent/'cartgs_pruning_baselines_20261112/reports/METHODS_ALL_METRICS.csv'
SOURCE_SHA = 'e56cc01dfe0dcfb1613f06d4211ce5d7e137b6c6ce341c2667db200cc027c891'
RATIOS = [.01,.03,.05,.08,.1,.2,.3,.4,.5,.6,.7]
SCENES = ['office0','office1','office2','office3','office4','room0','room1','room2']
METHODS = ['cartgs_random','cartgs_opacity','ours_global']
LABELS = {'cartgs_random':'CaRtGS 随机删除', 'cartgs_opacity':'CaRtGS Opacity删除',
          'ours_global':'我们（全局LoD）'}
COLORS = {'cartgs_random':'#6A99C1','cartgs_opacity':'#C99B2E','ours_global':'#BC4851'}
MARKERS = {'cartgs_random':'s','cartgs_opacity':'D','ours_global':'o'}
METRICS = {'psnr':('PSNR ↑','dB',40,[10,20,30,40]),
           'ssim':('SSIM ↑','',1,[.2,.4,.6,.8,1]),
           'lpips':('LPIPS ↓','',1,[.8,.6,.4,.2,0])}
RADIAL_KNOTS = {'ssim':[0,.5,.7,.8,.85,.9,.95],
                'lpips':[.75,.5,.35,.25,.2,.15,.1]}
if EXPANDED_SCALE:
    METRICS['ssim']=('SSIM ↑','',1,RADIAL_KNOTS['ssim'][1:])
    METRICS['lpips']=('LPIPS ↓','',1,RADIAL_KNOTS['lpips'][1:])


def scale_description(metric):
    if EXPANDED_SCALE and metric=='ssim':
        return '非等距径向刻度：重点放大0.80–0.95，越外越好'
    if EXPANDED_SCALE and metric=='lpips':
        return '反向非等距径向刻度：重点放大0.10–0.25，越外越好'
    return '径向原始刻度，越外越好' if metric!='lpips' else 'LPIPS反向径向刻度：中心1、外圈0，越外越好'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mean(values):
    return math.fsum(values)/len(values)


def write_csv(path, rows):
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,list(rows[0]));writer.writeheader();writer.writerows(rows)


def aggregate():
    assert digest(SOURCE)==SOURCE_SHA
    raw=list(csv.DictReader(SOURCE.open(encoding='utf-8-sig')))
    selected=[row for row in raw if row['group'].startswith('Replica/') and row['method'] in METHODS]
    assert len(selected)==16*3*11
    lookup={}
    per_scene=[]
    for row in selected:
        dataset,scene,mode=row['group'].split('/')
        ratio=float(row['ratio']);method=row['method']
        assert scene in SCENES and mode in ['mono','rgbd'] and ratio in RATIOS
        key=(scene,mode,method,ratio);assert key not in lookup
        value=dict(group=row['group'],scene=scene,mode=mode,method=method,ratio=ratio,
                   N=int(row['N']),K=float(row['K_mean']),frames=int(row['frames']),source=row['source'])
        value.update({metric:float(row[metric]) for metric in METRICS})
        assert all(math.isfinite(value[metric]) for metric in METRICS)
        lookup[key]=value;per_scene.append(value)
    rows=[]
    for mode in ['all','mono','rgbd']:
        modes=['mono','rgbd'] if mode=='all' else [mode]
        for method in METHODS:
            for ratio in RATIOS:
                points=[lookup[scene,camera,method,ratio] for scene in SCENES for camera in modes]
                entry=dict(mode=mode,method=method,ratio=ratio,scenes=8,scene_mode_groups=len(points),
                           weighting='equal_scene_then_equal_camera_mode',
                           N_mean=mean([point['N'] for point in points]),
                           K_mean=mean([point['K'] for point in points]),
                           evaluated_frames=sum(point['frames'] for point in points))
                for metric in METRICS:
                    # 场景先等权；综合图在每场景内部对两种模式等权。
                    scene_means=[mean([lookup[scene,camera,method,ratio][metric] for camera in modes]) for scene in SCENES]
                    entry[metric]=mean(scene_means)
                    assert abs(entry[metric]-mean([point[metric] for point in points]))<1e-13
                rows.append(entry)
    (OUTPUT/'data').mkdir(parents=True,exist_ok=True)
    write_csv(OUTPUT/'data/REPLICA_PER_SCENE.csv',per_scene)
    write_csv(OUTPUT/'data/REPLICA_RADAR_MEANS.csv',rows)
    provenance=dict(source=str(SOURCE),source_sha256=SOURCE_SHA,
        server_source='/home/prmi/zwj/realLOD/saved_experiments/cartgs_pruning_baselines_20261112/reports/METHODS_ALL_METRICS.csv',
        scenes=SCENES,modes=['mono','rgbd'],ratios=RATIOS,methods=METHODS,
        averaging='先计算每场景全轨迹均值；综合图先平均同场景单目与RGB-D，再对8场景等权平均。分模式图对8场景等权平均。',
        selection_scope='global',random_seed=0,per_scene_rows=len(per_scene),mean_rows=len(rows),
        LPIPS_axis='reverse radius, r=1-LPIPS; labels are original LPIPS values',
        ratio_axis='11个离散Ratio等角度排列，不表示相邻Ratio数值距离相同',
        budget='裁剪K=floor(r*N_CaRtGS)，我们沿用已有实际K；不同方法并非相同K',
        smoothing=False)
    if EXPANDED_SCALE:
        provenance.update(radial_scale='SSIM与LPIPS使用分段线性非等距刻度；标注原始值；没有裁剪数据点',
                          radial_knots=RADIAL_KNOTS,radial_positions=np.linspace(0,1,7).tolist())
        provenance['LPIPS_axis']='反向分段线性刻度：中心0.75、外圈0.10，标注原始LPIPS'
    (OUTPUT/'data/SOURCE_MANIFEST.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    return rows


def radius(metric, values):
    values=np.asarray(values)
    if EXPANDED_SCALE and metric in RADIAL_KNOTS:
        knots=np.asarray(RADIAL_KNOTS[metric]);positions=np.linspace(0,1,len(knots))
        assert np.all((values>=min(knots))&(values<=max(knots))), '放大刻度不得裁剪数据'
        if metric=='lpips':knots,positions=knots[::-1],positions[::-1]
        return np.interp(values,knots,positions)
    return 1-values if metric=='lpips' else values/METRICS[metric][2]


def configure_font():
    path=Path('C:/Windows/Fonts/msyh.ttc')
    if path.exists():
        font_manager.fontManager.addfont(str(path));family='Microsoft YaHei'
    else:family='Noto Sans CJK SC'
    plt.rcParams.update({'font.family':family,'axes.unicode_minus':False,
        'svg.fonttype':'none','pdf.fonttype':42,'savefig.facecolor':'white'})


def radar(ax,rows,metric,mode,small=False):
    angles=np.pi/2-np.arange(len(RATIOS))*2*np.pi/len(RATIOS)
    unit=np.column_stack((np.cos(angles),np.sin(angles)))
    ax.set_aspect('equal');ax.axis('off');ax.set_xlim(-1.30,1.30);ax.set_ylim(-1.30,1.30)
    ax.add_patch(Circle((0,0),1,facecolor='#FBFCFD',edgecolor='none',zorder=0))
    for point in unit:ax.plot([0,point[0]],[0,point[1]],color='#DCE1E5',lw=.7,zorder=1)
    for tick in METRICS[metric][3]:
        r=float(radius(metric,[tick])[0])
        if r==0:continue
        ax.add_patch(Polygon(unit*r,closed=True,fill=False,edgecolor='#CCD3D9',lw=.8,
                             ls=(0,(3,3)),zorder=1))
        ax.text(-r,.035,f'{tick:g}',fontsize=8.3 if small else 9.5,color='#62717C',
                va='bottom',ha='center',zorder=8,bbox=dict(fc='white',ec='none',pad=.5,alpha=.88))
    start=np.array(colors.to_rgb('#A5B6C1'));end=np.array(colors.to_rgb('#C4CED4'))
    half=180/len(RATIOS)
    for index,(angle,ratio) in enumerate(zip(angles,RATIOS)):
        degree=math.degrees(angle);blend=index/(len(RATIOS)-1)
        ax.add_patch(Wedge((0,0),1.245,degree-half+.9,degree+half-.9,width=.122,
                           facecolor=start*(1-blend)+end*blend,edgecolor='none',zorder=2))
        x,y=1.184*math.cos(angle),1.184*math.sin(angle)
        rotation=degree-90
        while rotation<-90:rotation+=180
        while rotation>90:rotation-=180
        ax.text(x,y,f'{ratio*100:g}%',ha='center',va='center',rotation=rotation,
                fontsize=10 if small else 12,fontweight='bold',color='#26343F',zorder=3)
    for method in METHODS:
        points=[next(row for row in rows if row['mode']==mode and row['method']==method and row['ratio']==ratio) for ratio in RATIOS]
        values=[point[metric] for point in points]
        rr=radius(metric,values);assert np.all((rr>=0)&(rr<=1))
        xy=unit*rr[:,None]
        ax.fill(xy[:,0],xy[:,1],color=COLORS[method],alpha=.065,zorder=3)
        xx=np.r_[xy[:,0],xy[0,0]];yy=np.r_[xy[:,1],xy[0,1]]
        ax.plot(xx,yy,color=COLORS[method],lw=2 if method=='ours_global' else 1.7,
                marker=MARKERS[method],ms=4.2 if small else 5,markerfacecolor='white',
                markeredgewidth=1.2,zorder=6 if method=='ours_global' else 4)
        if method=='ours_global':
            for index,(angle,r,value) in enumerate(zip(angles,rr,values)):
                tx,ty=(r+.043)*math.cos(angle),(r+.043)*math.sin(angle)
                label=f'{value:.2f}' if metric=='psnr' else f'{value:.3f}'
                align='center'
                if EXPANDED_SCALE and metric=='lpips' and index<2:
                    align='right' if index==0 else 'left'
                ax.text(tx,ty,label,ha=align,va='center',color=COLORS[method],
                        fontsize=8 if small else 9.5,fontweight='bold',zorder=9,
                        bbox=dict(fc='white',ec='none',pad=.5,alpha=.83))
    if mode!='all':
        ax.set_title('单目 · 8场景均值' if mode=='mono' else 'RGB-D · 8场景均值',
                     fontsize=14,fontweight='bold',pad=14,color='#26343F')
    return ax


def legend_handles():
    return [Line2D([],[],color=COLORS[method],marker=MARKERS[method],mfc='white',
                   lw=1.8,ms=5,label=LABELS[method]) for method in METHODS]


def save(fig,path):
    for extension in ['png','svg','pdf']:fig.savefig(path.with_suffix('.'+extension),dpi=450)
    plt.close(fig)


def main():
    configure_font();rows=aggregate();figures=OUTPUT/'figures';figures.mkdir(exist_ok=True)
    # 数据清单先于出图写入；图中不存在模拟数据和逐Ratio归一化。
    (figures/'data-manifest.md').write_text('# Replica Ratio 雷达图数据清单\n\n'
      '|图|数据|真实/模拟|来源|脚本|输出|\n|---|---|---|---|---|---|\n'
      '|PSNR、SSIM、LPIPS各一张|data/REPLICA_RADAR_MEANS.csv|真实|704项评价中的Replica裁剪结果及已有全局LoD|plot_replica_radar.py|REPLICA_*_RADAR.png/.svg/.pdf|\n'
      '|单目/RGB-D分模式附图|同上|真实|各模式8场景均值|同上|REPLICA_*_BY_MODE.png/.svg/.pdf|\n',encoding='utf-8')
    for metric in METRICS:
        fig=plt.figure(figsize=(7.2,7.6))
        ax=fig.add_axes([.035,.175,.93,.705])
        radar(ax,rows,metric,'all')
        fig.suptitle('Replica · '+METRICS[metric][0],fontsize=20,fontweight='bold',y=.983,color='#22313B')
        fig.text(.5,.921,'8场景等权平均 · 每场景单目与RGB-D等权',ha='center',fontsize=11,color='#52636F')
        fig.legend(handles=legend_handles(),loc='lower center',bbox_to_anchor=(.5,.115),
                   ncol=3,frameon=False,fontsize=9.8,columnspacing=1.2,handlelength=1.7)
        direction=scale_description(metric)
        fig.text(.5,.079,direction+'；红色数字为我们的方法均值',ha='center',fontsize=9,color='#52636F')
        fig.text(.5,.044,'裁剪按CaRtGS自身N计算K；相同Ratio，不同方法实际K不同',ha='center',fontsize=9,color='#52636F')
        save(fig,figures/f'REPLICA_{metric.upper()}_RADAR')
        fig=plt.figure(figsize=(13.6,7.6))
        for index,mode in enumerate(['mono','rgbd']):
            ax=fig.add_axes([.025+index*.5,.16,.45,.72]);radar(ax,rows,metric,mode,small=True)
        fig.suptitle('Replica · '+METRICS[metric][0]+' · 分模式核对',fontsize=20,fontweight='bold',y=.99,color='#22313B')
        fig.legend(handles=legend_handles(),loc='lower center',bbox_to_anchor=(.5,.085),ncol=3,frameon=False,fontsize=11)
        fig.text(.5,.045,'office0–4、room0–2等权；径向刻度与综合图相同；'+direction,ha='center',fontsize=10,color='#52636F')
        save(fig,figures/f'REPLICA_{metric.upper()}_BY_MODE')
    fig=plt.figure(figsize=(17.4,6.5))
    for index,metric in enumerate(METRICS):
        ax=fig.add_axes([.005+index/3,.21,.323,.60]);radar(ax,rows,metric,'all',small=True)
        ax.set_title(METRICS[metric][0],fontsize=18,fontweight='bold',pad=12)
    fig.suptitle('Replica · 8场景均值 · Ratio对照',fontsize=20,fontweight='bold',y=.985,color='#22313B')
    fig.text(.5,.916,'每场景先平均单目与RGB-D，再对8个场景等权平均；蓝：随机删除，金：Opacity删除，红：我们全局LoD',ha='center',fontsize=11,color='#52636F')
    fig.legend(handles=legend_handles(),loc='lower center',bbox_to_anchor=(.5,.10),ncol=3,frameon=False,fontsize=11)
    overview_note=('三图均越外越好；SSIM／LPIPS采用非等距径向刻度放大差异，标注原始值；所有数据点完整保留。'
                   if EXPANDED_SCALE else '三图均为越外越好；LPIPS反向径向刻度。外圈是11个离散Ratio；裁剪预算使用CaRtGS自己的N，非同K比较。')
    fig.text(.5,.052,overview_note,ha='center',fontsize=10,color='#52636F')
    save(fig,figures/'REPLICA_RADAR_OVERVIEW')
    lines=['# Replica按Ratio的雷达图','',
      '三张主图分别为PSNR、SSIM与LPIPS。蓝色为CaRtGS随机删除，金色为CaRtGS Opacity排序删除，红色为我们全局LoD。','',
      '综合图使用office0–4、room0–2共8个场景：先平均同场景单目与RGB-D的全轨迹成绩，再对8场景等权平均。'
      '这等价于16组场景／模式等权平均，不按帧数加权。分模式附图分别对8场景等权平均。','',
      ('PSNR仍使用0–40 dB原始刻度。SSIM、LPIPS使用分段线性非等距径向刻度，全部Ratio与模式共用一套映射。'
       'SSIM中心0、外圈0.95，依次在0.50、0.70、0.80、0.85、0.90、0.95画等距圆环；'
       'LPIPS中心0.75、外圈0.10，依次在0.50、0.35、0.25、0.20、0.15、0.10画等距圆环。'
       'SSIM的0.80–0.95和LPIPS的0.10–0.25区间，每0.05差异的径向距离为原图的约3.33倍。'
       '没有裁剪任何数据点，所有文字标注仍为原始指标值。'
       if EXPANDED_SCALE else 'PSNR统一使用0–40 dB径向刻度；SSIM使用0–1；LPIPS反向显示，中心1、外圈0，标注值仍为原LPIPS。')+
      '所有图越靠外越好，没有逐Ratio归一化、平滑或变更原数值。圆周各Ratio是离散类别，等角度不表示实际数值距离。','',
      '我们使用已有全局导出的结果，不混入局部Ratio结果。两个删除基线均无补偿或再训练，随机仅seed0一次。'
      '删除预算K=floor(r×N_CaRtGS)，各方法实际K不同，不能称同K试验。','',
      '|Ratio|随机PSNR|Opacity PSNR|我们PSNR|随机SSIM|Opacity SSIM|我们SSIM|随机LPIPS|Opacity LPIPS|我们LPIPS|',
      '|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for ratio in RATIOS:
        values=[next(row for row in rows if row['mode']=='all' and row['method']==method and row['ratio']==ratio) for method in METHODS]
        parts=[f'{ratio:g}']+[f"{value[metric]:.6f}" for metric in METRICS for value in values]
        lines.append('| '+' | '.join(parts)+' |')
    lines+=['','准确逐场景值、平均值、平均N／K、评价帧数与来源在data目录；全部图同时提供PNG、SVG、PDF。']
    (OUTPUT/'README_结果说明.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    verification=dict(complete=True,source_sha256=SOURCE_SHA,source_unchanged=digest(SOURCE)==SOURCE_SHA,
        scenes=8,scene_mode_groups=16,methods=3,ratios=11,per_scene_rows=528,mean_rows=99,
        figures={str(path.relative_to(OUTPUT)).replace('\\','/'):digest(path) for path in figures.iterdir() if path.is_file()},
        normalization='共同的分段线性非等距刻度，全部Ratio和模式一致' if EXPANDED_SCALE else 'single fixed raw-metric scale across ratios and modes',LPIPS_reversed=True,
        pixel_images_not_rerendered=True)
    assert len(list(figures.glob('*.png')))==len(list(figures.glob('*.svg')))==len(list(figures.glob('*.pdf')))==7
    (OUTPUT/'VERIFIED.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with zipfile.ZipFile(OUTPUT/'REPLICA_RATIO_RADAR.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for folder in ['data','figures']:
            for path in (OUTPUT/folder).iterdir():
                if path.is_file():archive.write(path,path.relative_to(OUTPUT))
        archive.write(ROOT/'plot_replica_radar.py','plot_replica_radar.py')
        for name in ['README_结果说明.md','VERIFIED.json']:archive.write(OUTPUT/name,name)
    print('REPLICA_RADAR_COMPLETE',len(rows),'均值行，7张图（3主图、3分模式、1总览）')


if __name__=='__main__':main()
