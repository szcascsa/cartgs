"""三数据集四方法雷达图：局部可选逐视角均值K或跨视角去重总数U，沿用原等距质量刻度。"""
from pathlib import Path
import argparse
import csv
import json
import math
import shutil
import sys
import zipfile

ROOT = Path(__file__).resolve().parent
for directory in ['tum_vector_ratio_radar_20261113','replica_ratio_radar_20261113']:
    dependency=ROOT.parent/directory
    if dependency.is_dir():sys.path.insert(0,str(dependency))
import plot_tum_vector_radar as prior
import numpy as np
from matplotlib.lines import Line2D

base = prior.base
METHODS = ['cartgs_random','cartgs_opacity','ours_global','ours_view_ratio']
DATASETS = {'REPLICA':dict(source='Replica',title='Replica',modes=['mono','rgbd'],
                          scenes=['office0','office1','office2','office3','office4','room0','room1','room2']),
            **prior.DATASETS}
TABLE_LABELS = ['随机删除','Opacity删除','我们全局','我们局部']
COUNT_FIELD='K_mean'
COUNT_TITLE='平均保留Gaussian数K（千点）'
UNION_PATH=None
UNION_LOOKUP={}


def configure():
    assert not base.EXPANDED_SCALE
    base.METHODS=METHODS
    base.LABELS.update(ours_global='我们：全局Ratio',ours_view_ratio='我们：局部视角Ratio')
    base.COLORS['ours_view_ratio']='#319B80'
    base.MARKERS['ours_view_ratio']='^'
    base.configure_font()


def draw(ax,rows,metric,mode,k_limit,config,small=False):
    base.radar(ax,rows,metric,mode,small=small)
    if mode!='all':
        ax.set_title(f"{prior.MODE_LABELS[mode]} · {len(config['scenes'])}场景均值",
                     fontsize=14,fontweight='bold',pad=14,color='#26343F')
    angles=np.pi/2-np.arange(len(base.RATIOS))*2*np.pi/len(base.RATIOS)
    unit=np.column_stack((np.cos(angles),np.sin(angles)))
    for method in METHODS:
        line=next(line for line in ax.lines if line.get_color()==base.COLORS[method]
                  and len(line.get_xdata())==12)
        line.set_gid('quality_'+method)
        if method=='ours_view_ratio':line.set_zorder(6.1)
    styles={'cartgs_random':(0,(4,12)), 'cartgs_opacity':(8,(4,12)),
            'ours_global':(0,(4,4)), 'ours_view_ratio':(0,(4,4))}
    for method in METHODS:
        points=[next(r for r in rows if r['mode']==mode and r['method']==method and r['ratio']==ratio)
                for ratio in base.RATIOS]
        rr=np.array([p[COUNT_FIELD] for p in points])/k_limit
        assert np.all((rr>=0)&(rr<=1))
        xy=unit*rr[:,None]
        line,=ax.plot(np.r_[xy[:,0],xy[0,0]],np.r_[xy[:,1],xy[0,1]],
                     color=base.COLORS[method],lw=1.5,ls=styles[method],zorder=7,
                     dash_capstyle='butt')
        line.set_gid('retained_K_'+method)
    for tick in base.METRICS[metric][3]:
        r=float(base.radius(metric,[tick])[0])
        if r==0:continue
        ax.text(r,.035,f'{r*k_limit/1000:g}k',fontsize=7.5 if small else 8.8,
                color='#52636F',va='bottom',ha='center',zorder=8,
                bbox=dict(fc='white',ec='none',pad=.5,alpha=.88))


def count_table(fig,rows,mode,bounds,fontsize=7.8):
    ax=fig.add_axes(bounds);ax.axis('off')
    cell_text=[]
    for method,label in zip(METHODS,TABLE_LABELS):
        points=[next(r for r in rows if r['mode']==mode and r['method']==method and r['ratio']==ratio)
                for ratio in base.RATIOS]
        cell_text.append([label]+[f"{p[COUNT_FIELD]/1000:.2f}" for p in points])
    table=ax.table(cellText=cell_text,
                   colLabels=['Ratio']+[f'{100*r:g}%' for r in base.RATIOS],
                   colWidths=[.15]+[.85/11]*11,cellLoc='center',bbox=[0,0,1,1])
    table.auto_set_font_size(False);table.set_fontsize(fontsize)
    for (row,column),cell in table.get_celld().items():
        cell.set_edgecolor('#E1E6EA');cell.set_linewidth(.5)
        cell.set_facecolor('#F1F4F6' if row==0 else ('white' if row%2 else '#FAFBFC'))
        if row==0:cell.get_text().set_color('#33434F')
        elif column==0:
            cell.get_text().set_color(base.COLORS[METHODS[row-1]])
            cell.get_text().set_fontweight('bold')
    return table


def legends(fig,method_y,type_y,compact=False):
    fig.legend(handles=base.legend_handles(),loc='lower center',bbox_to_anchor=(.5,method_y),
               ncol=2 if compact else 4,frameon=False,fontsize=9.6 if compact else 10.5,
               columnspacing=1.6,handlelength=1.8)
    handles=[Line2D([],[],color='#44515C',lw=1.7,label='实线：渲染质量（左刻度）'),
             Line2D([],[],color='#44515C',lw=1.5,ls=(0,(4,4)),
                    label='虚线：去重总数U（右刻度）' if UNION_PATH else '虚线：平均保留K（右刻度）')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,type_y),
               ncol=2,frameon=False,fontsize=9,columnspacing=2,handlelength=2.4)


def plot_dataset(tag,config):
    destination=ROOT/tag;rows,per_scene=prior.aggregate(config,destination)
    if UNION_PATH:
        for point in per_scene:
            if point['method']=='ours_view_ratio':
                union=UNION_LOOKUP[point['group'],point['ratio']]
                assert int(union['N'])==point['N'] and int(union['views'])==point['frames']
                assert abs(float(union['local_mean_K'])-point['K'])<1e-9
                assert union['source']==point['source'] and union['all_view_hashes_match']=='True'
                point['unique_total']=int(union['local_union_count'])
            else:point['unique_total']=point['K']
        for row in rows:
            modes=config['modes'] if row['mode']=='all' else [row['mode']]
            points=[p for p in per_scene if p['mode'] in modes and p['method']==row['method'] and p['ratio']==row['ratio']]
            row['unique_total']=base.mean([p['unique_total'] for p in points])
        base.write_csv(destination/'data/PER_SCENE.csv',per_scene)
        base.write_csv(destination/'data/RADAR_MEANS.csv',rows)
    figures=destination/'figures';figures.mkdir(exist_ok=True)
    n=len(config['scenes']);camera_text='与'.join(prior.MODE_LABELS[m] for m in config['modes'])
    subtitle=f'每场景先平均{camera_text}，再对{n}个场景等权平均'
    k_limit=math.ceil(max(r[COUNT_FIELD] for r in rows)/10000)*10000
    note=(f'右刻度U：0–{k_limit/1000:g}千点；局部对所有视角选点ID取并集；CaRtGS两虚线重合'
          if UNION_PATH else f'右刻度K：0–{k_limit/1000:g}千点；局部K先对所有评价视角平均；CaRtGS两虚线重合')
    (figures/'data-manifest.md').write_text('# 数据清单\n\n'
        '|图|数据|性质|来源|输出|\n|---|---|---|---|---|\n'
        '|PSNR、SSIM、LPIPS及分模式图／总览|data/RADAR_MEANS.csv|真实|METHODS_ALL_METRICS.csv|PNG/SVG/PDF|\n\n'+
        ('实线读质量指标；虚线读unique_total，局部为所有评价视角选点ID并集基数，再按场景等权平均。\n'
         if UNION_PATH else '实线读质量指标，同色虚线读K_mean；绿色为局部视角Ratio，K来自所有评价视角的实际保留数算术平均。\n'),encoding='utf-8')
    for metric in prior.style.METRICS:
        fig=base.plt.figure(figsize=(7.2,10.0))
        ax=fig.add_axes([.035,.365,.93,.525]);draw(ax,rows,metric,'all',k_limit,config)
        fig.suptitle(config['title']+' · '+base.METRICS[metric][0],fontsize=20,fontweight='bold',y=.983,color='#22313B')
        fig.text(.5,.935,f'{n}场景等权平均 · 每场景{camera_text}等权',ha='center',fontsize=11,color='#52636F')
        legends(fig,.287,.260,compact=True)
        fig.text(.5,.233,COUNT_TITLE,ha='center',fontsize=10,color='#33434F')
        count_table(fig,rows,'all',[.045,.092,.91,.125])
        fig.text(.5,.059,note,ha='center',fontsize=8.4,color='#52636F')
        fig.text(.5,.034,'质量沿用原等距刻度；红色数字标注全局版指标；实线越外越好',ha='center',fontsize=8.6,color='#52636F')
        base.save(fig,figures/f'{tag}_{metric.upper()}_RADAR')
        fig=base.plt.figure(figsize=(13.6,10.0))
        for index,mode in enumerate(config['modes']):
            ax=fig.add_axes([.025+index*.5,.365,.45,.525]);draw(ax,rows,metric,mode,k_limit,config,small=True)
            count_table(fig,rows,mode,[.025+index*.5,.092,.45,.125],fontsize=7.6)
            fig.text(.25+index*.5,.233,COUNT_TITLE,ha='center',fontsize=10,color='#33434F')
        fig.suptitle(config['title']+' · '+base.METRICS[metric][0]+' · 分模式核对',fontsize=20,fontweight='bold',y=.985,color='#22313B')
        legends(fig,.300,.267)
        fig.text(.5,.043,f'{n}个场景等权；'+note,ha='center',fontsize=9.5,color='#52636F')
        base.save(fig,figures/f'{tag}_{metric.upper()}_BY_MODE')
    fig=base.plt.figure(figsize=(17.4,8.9))
    for index,metric in enumerate(prior.style.METRICS):
        ax=fig.add_axes([.005+index/3,.375,.323,.44]);draw(ax,rows,metric,'all',k_limit,config,small=True)
        ax.set_title(base.METRICS[metric][0],fontsize=18,fontweight='bold',pad=12)
        count_table(fig,rows,'all',[.012+index/3,.095,.309,.125],fontsize=6.9)
        fig.text(1/6+index/3,.240,COUNT_TITLE,ha='center',fontsize=9,color='#33434F')
    fig.suptitle(f"{config['title']} · {n}场景均值 · 全局／局部Ratio",fontsize=20,fontweight='bold',y=.985,color='#22313B')
    fig.text(.5,.930,subtitle+'；红：全局Ratio，绿：局部视角Ratio',ha='center',fontsize=11,color='#52636F')
    legends(fig,.305,.274)
    fig.text(.5,.043,'质量沿用原等距刻度；'+note+'；虚线向外仅表示保留点更多。',ha='center',fontsize=9.5,color='#52636F')
    base.save(fig,figures/f'{tag}_RADAR_OVERVIEW')
    provenance=dict(dataset=config['source'],scenes=config['scenes'],modes=config['modes'],
        source=str(prior.SOURCE),source_sha256=prior.SOURCE_SHA,source_rows=len(per_scene),mean_rows=len(rows),
        real_data=True,average=subtitle,methods=METHODS,ratios=base.RATIOS,
        local_K_formula='每个scene/mode/ratio的K_mean = sum_v(actual_K_v)/评价视角数；之后对相机模式、场景等权平均',
        quality_scale='PSNR 0–40；SSIM 0–1；LPIPS反向1–0；等距径向刻度',K_axis_max=k_limit,
        K_definition='全局：导出模型保留K；局部：所有评价视角实际送入Renderer的K均值；不是Alpha合成可见点数',
        comparison='同Ratio，不同候选池且实际K不同；各自原图轨迹均值，不是固定相机位姿同K对照',
        table='K以千点显示两位小数，CSV保存完整精度',new_training=False,new_rendering=False)
    if UNION_PATH:
        provenance.update(count_field='unique_total',union_source=str(UNION_PATH),union_source_sha256=base.digest(UNION_PATH),
            local_K_formula='U = |union_v selected_master_ids(v)|；同一场景／模式／Ratio内去重，再对模式和场景等权平均',
            K_definition='全局和删除方法为固定保留集大小；局部为所有评价视角选中母Gaussian ID的并集大小',
            table='去重总数U以千点显示两位小数；K_mean字段仍保留原逐视角均值',
            count_identity='固定checkpoint的Master原始行ID；不同视角补偿坐标不同不产生新ID')
    (destination/'data/SOURCE_MANIFEST.json').write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (destination/'README_结果说明.md').write_text(f"# {config['title']}全局／局部Ratio对比\n\n"+
        subtitle+'，不按轨迹帧数跨场景加权。场景：'+'、'.join(config['scenes'])+'。\n\n'
        '蓝色为CaRtGS随机删除，金色为CaRtGS Opacity删除，红色为我们全局Ratio，绿色为我们局部视角Ratio。'
        '实线为质量，同色虚线为平均保留K。红色数字仅标注全局版指标，局部完整指标见CSV。'
        '质量保持原版等距刻度：PSNR 0–40，SSIM 0–1，LPIPS反向1–0。\n\n'
        '局部的每个场景／模式／Ratio先对所有已评价视角的实际保留K求算术平均，读取评价汇总actual_K_mean，'
        '再按与质量一致的相机模式和场景等权平均。不是ratio乘全局N，也不是逐视角Alpha合成可见点数。'
        '全局K为导出模型保留数。图下表单位为千点，显示两位小数，CSV保留完整精度。\n\n'
        'CaRtGS两种删除方法K相同，虚线路径重合，用蓝金交错线段显示。各方法和全局／局部的候选池及实际K不同，'
        '属于同Ratio比较，不是同K或固定相机位姿对照。各自评价帧数及源路径保存在逐场景CSV。'
        '随机删除只有seed0。本次不重新训练或渲染，保留之前三方法图。\n',encoding='utf-8')
    if UNION_PATH:
        (destination/'README_结果说明.md').write_text(f"# {config['title']}跨视角去重保留总数\n\n"+
            subtitle+'。\n\n'
            '局部点数定义为U=|∪ᵥ Sᵥ|，Sᵥ为当前视角实际选中并提交的母Gaussian原始ID集合。'
            '一个点只要在任何评价视角被选中过就计入一次，不随出现帧数或补偿位置重复计数。'
            '逐帧选点哈希与原评价完全一致，原均值K_mean仍保存在CSV，图中虚线和表格读取unique_total。\n\n'
            'CaRtGS两删除版本及我们全局版使用固定保留集大小；不同场景／相机模式先分别去重，再等权汇总，'
            '不会将不同模型的ID混在一起去重。实线质量和原始等距刻度不变。'
            '蓝色随机删除，金色Opacity删除，红色我们全局Ratio，绿色我们局部Ratio。\n\n'
            '该点数表示整个评价轨迹累计需要的唯一母Gaussian数量，不是每帧渲染工作量。'
            '只统计既有评价视角，轨迹覆盖以逐场景CSV帧数和源路径为准；不改旧指标，不重新训练或渲染图片。\n',encoding='utf-8')
    assert all(len(list(figures.glob('*.'+ext)))==7 for ext in ['png','svg','pdf'])
    print(config['title'],'完成：',len(per_scene),'条逐场景记录，',len(rows),'条均值，7张图',flush=True)
    return provenance


def main():
    configure();provenance={tag:plot_dataset(tag,config) for tag,config in DATASETS.items()}
    data=ROOT/'data';data.mkdir(exist_ok=True)
    shutil.copyfile(prior.SOURCE,data/'METHODS_ALL_METRICS.csv')
    if UNION_PATH:shutil.copyfile(UNION_PATH,data/'LOCAL_UNION_COUNTS.csv')
    receipt=dict(complete=True,source_unchanged=base.digest(prior.SOURCE)==prior.SOURCE_SHA,datasets=provenance,
                 png=21,svg=21,pdf=21,
                 figures={str(p.relative_to(ROOT)).replace('\\','/'):base.digest(p) for p in ROOT.glob('*/figures/*') if p.is_file()})
    (ROOT/'VERIFIED.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    (ROOT/'README_结果说明.md').write_text('# 三数据集全局／局部Ratio雷达图\n\n'
        'Replica为8场景，TUM为5场景，VECtor为3场景。每个场景包含两种相机模式，均覆盖11个Ratio、4种方法。'
        '每个数据集提供3张独立指标图、3张分模式图、1张总览及原始CSV。'
        '局部视角Ratio以绿色显示，虚线点数先在每个模型所有已评价视角内求实际K均值，再按场景等权汇总。\n',encoding='utf-8')
    if UNION_PATH:
        (ROOT/'README_结果说明.md').write_text('# 三数据集跨视角去重总数雷达图\n\n'
            'Replica 8场景、TUM 5场景、VECtor 3场景；各有两种相机模式、11个Ratio、4种方法。'
            '局部U为所有评价视角选中母Gaussian ID的并集大小，重复出现只计一次；之后对相机模式、场景等权平均。'
            '全局及CaRtGS删除方法使用固定保留集大小。质量结果和等距刻度不变，原逐视角K均值仍保存在CSV。'
            'U衡量整段轨迹的唯一母点总需求，不表示单帧渲染工作量。\n',encoding='utf-8')
    archive_name='ALL_DATASETS_GLOBAL_LOCAL_UNION_RADAR.zip' if UNION_PATH else 'ALL_DATASETS_GLOBAL_LOCAL_RADAR.zip'
    with zipfile.ZipFile(ROOT/archive_name,'w',zipfile.ZIP_DEFLATED) as archive:
        for directory in [*DATASETS,'data']:
            for path in (ROOT/directory).rglob('*'):
                if path.is_file():archive.write(path,path.relative_to(ROOT))
        archive.write(Path(__file__),'plot_global_local_radar.py')
        for name in ['VERIFIED.json','README_结果说明.md']:archive.write(ROOT/name,name)
        archive.write(Path(prior.__file__),'plot_tum_vector_radar.py')
        archive.write(Path(prior.style.__file__),'plot_replica_radar_with_counts.py')
        archive.write(Path(base.__file__),'plot_replica_radar.py')
    print('GLOBAL_LOCAL_RADAR_COMPLETE：21张图，PNG／SVG／PDF及数据已打包',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--union-counts',type=Path)
    args=parser.parse_args()
    if args.union_counts:
        UNION_PATH=args.union_counts.resolve()
        values=list(csv.DictReader(UNION_PATH.open(encoding='utf-8-sig')))
        UNION_LOOKUP={(r['group'],float(r['ratio'])):r for r in values}
        assert len(UNION_LOOKUP)==len(values)==352
        ROOT=ROOT/'unique_union';COUNT_FIELD='unique_total'
        COUNT_TITLE='跨视角去重保留总数U（千点）'
    main()
